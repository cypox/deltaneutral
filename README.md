# Delta-Neutral Crypto Trading Bot

A cross-exchange delta-neutral funding rate arbitrage bot for cryptocurrency perpetual futures.

## Strategy Overview

**Delta-neutral funding rate arbitrage** captures funding payments from perpetual futures while hedging price risk:

1. **Detect opportunity**: Monitor funding rates across exchanges. When perpetual funding is positive (longs pay shorts), there's an arbitrage opportunity.
2. **Enter position**: Buy spot on Exchange A + Short perpetual on Exchange B.
3. **Collect funding**: While holding the position, receive funding payments every 8 hours.
4. **Exit**: Close both legs when funding rate drops below threshold.

The position is **delta-neutral** — price movements on the spot leg offset the perpetual leg, so PnL comes primarily from funding payments minus fees.

### Key Checks

| Check | Description |
|-------|-------------|
| **Funding rate APR** | Only enters when annualized funding exceeds threshold (default 10%) |
| **Spread validation** | Spot-perp price spread must be within tolerance |
| **Fee structure** | Full round-trip fees (4 legs × taker + slippage) deducted from APR |
| **Perp amount validation** | Continuously validates perp position matches spot (delta-neutral) |
| **Position drift detection** | Exits if perp amount drifts >1% from expected |
| **Margin monitoring** | Alerts when approaching liquidation price |
| **Max drawdown** | Emergency halt at configurable drawdown threshold |
| **Circuit breaker** | Pauses after consecutive execution errors |

## Architecture

```
src/tradingbot/
├── config/          # YAML-based configuration with env var support
├── core/
│   ├── engine.py    # Main trading loop orchestrator
│   ├── portfolio.py # Cross-exchange portfolio tracking
│   └── risk.py      # Risk management & circuit breakers
├── data/
│   ├── feed.py      # Real-time market data aggregation
│   ├── loader.py    # Historical data download pipeline
│   └── storage.py   # SQLAlchemy async database layer
├── exchanges/
│   ├── base.py      # Abstract exchange interface
│   ├── connector.py # CCXT-based implementation
│   └── factory.py   # Exchange instance factory
├── execution/
│   ├── executor.py  # Order execution with TWAP & retry
│   └── reconciliation.py  # Position state validation
├── strategy/
│   ├── base.py          # Strategy interface
│   └── delta_neutral.py # Funding rate arbitrage logic
├── backtesting/
│   ├── engine.py    # Backtest orchestrator
│   └── simulator.py # Exchange simulator with fills & funding
├── utils/
│   ├── logger.py    # Structured logging (structlog)
│   ├── metrics.py   # Performance calculations (Sharpe, drawdown, etc.)
│   └── helpers.py   # Precision rounding, retry, conversions
└── cli.py           # Click CLI entry point
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- (Optional) Docker & Docker Compose for containerized deployment
- Exchange API keys (Binance, Bybit, and/or OKX) for live trading

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd tradingbot

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install with all dependencies (dev tools + backtest plotting)
pip install -e ".[dev,backtest]"
```

### 2. Configuration

```bash
# Copy the example config
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` and add your exchange API keys. You can also use environment variables — the config supports `${VAR_NAME}` substitution:

```bash
# Windows PowerShell
$env:BINANCE_API_KEY = "your_api_key"
$env:BINANCE_SECRET = "your_secret"

# macOS / Linux
export BINANCE_API_KEY=your_api_key
export BINANCE_SECRET=your_secret
```

Key configuration options to review:

```yaml
strategy:
  symbols: ["BTC/USDT", "ETH/USDT"]   # Symbols to trade
  min_funding_rate_apr: 0.10            # Min 10% APR to enter
  max_position_usd: 10000.0            # Max position size per symbol
  max_total_exposure_usd: 50000.0      # Portfolio-wide exposure limit
  exit_funding_rate_apr: 0.02           # Exit below 2% APR

risk:
  max_drawdown_pct: 0.05               # Emergency halt at 5% drawdown
  position_amount_tolerance_pct: 0.01   # 1% tolerance on perp amount check
  min_free_margin_usd: 5000.0          # Keep $5k free on each exchange
```

### 3. Download Historical Market Data

Before backtesting, download OHLCV candles and funding rate history:

```bash
# Via CLI — download BTC/USDT from Binance, all of 2024
tradingbot download --exchange binance --symbol BTC/USDT --start 2024-01-01 --end 2025-01-01

# Download ETH as well
tradingbot download --exchange binance --symbol ETH/USDT --start 2024-01-01 --end 2025-01-01

# Or use the batch script to download all configured symbols
python scripts/download_data.py
```

Data is stored in a local SQLite database at `data/tradingbot.db`.

### 4. Run a Backtest

```bash
# Via CLI
tradingbot backtest --symbol BTC/USDT --output data/backtest_results

# Or run the backtest script (generates charts if matplotlib is installed)
python scripts/run_backtest.py
```

Example output:

```
============================================================
  BACKTEST RESULTS — Delta-Neutral Funding Arbitrage
============================================================
  Total Return:             4.23%
  Annualized Return:        8.51%
  Sharpe Ratio:            2.1345
  Sortino Ratio:           3.4521
  Max Drawdown:             1.12%
  Win Rate:                72.00%
  Total Trades:               25
  Funding Collected:      $5,230.00
  Fees Paid:                $987.00
  Net PnL:                $4,243.00
============================================================
```

### 5. Using the Python API

You can use the library programmatically for custom workflows:

```python
import asyncio
from tradingbot.config.settings import Settings, ExchangeConfig
from tradingbot.exchanges.factory import create_and_connect
from tradingbot.data.storage import DataStore
from tradingbot.data.loader import MarketDataLoader
from tradingbot.utils.metrics import funding_rate_to_apr

async def check_funding_rates():
    # Connect to an exchange (no API key needed for public data)
    exchange = await create_and_connect("binance", ExchangeConfig())

    # Fetch current funding rate
    fr = await exchange.fetch_funding_rate("BTC/USDT:USDT")
    apr = funding_rate_to_apr(fr.rate)
    print(f"BTC/USDT funding rate: {fr.rate:.6f} ({apr:.2%} APR)")

    # Fetch ticker for spread analysis
    ticker = await exchange.fetch_ticker("BTC/USDT")
    print(f"BTC/USDT  bid={ticker.bid}  ask={ticker.ask}")

    # Fetch fee schedule
    fees = await exchange.fetch_fee_schedule("BTC/USDT:USDT")
    print(f"Maker fee: {fees.maker:.4%}  Taker fee: {fees.taker:.4%}")

    await exchange.close()

asyncio.run(check_funding_rates())
```

**Running a backtest from code:**

```python
import asyncio
import pandas as pd
from tradingbot.config.settings import Settings
from tradingbot.backtesting.engine import BacktestEngine
from tradingbot.data.storage import DataStore
from tradingbot.data.loader import MarketDataLoader

async def run_custom_backtest():
    settings = Settings.default()

    # Override strategy params for experimentation
    settings.strategy.min_funding_rate_apr = 0.05   # Lower entry threshold
    settings.strategy.max_position_usd = 20000.0    # Larger positions
    settings.backtest.initial_capital = 200_000.0

    # Load data from local DB
    store = DataStore(settings.app.db_url)
    await store.initialize()
    loader = MarketDataLoader(store)

    spot = await loader.load_ohlcv_dataframe("binance", "BTC/USDT", "1h", "2024-01-01", "2025-01-01")
    perp = await loader.load_ohlcv_dataframe("binance", "BTC/USDT:USDT", "1h", "2024-01-01", "2025-01-01")
    funding = await loader.load_funding_dataframe("binance", "BTC/USDT:USDT", "2024-01-01", "2025-01-01")

    # Run backtest engine
    engine = BacktestEngine(settings.strategy, settings.backtest, settings.fees)
    result = engine.run(spot, perp if not perp.empty else spot, funding)

    print(result.print_summary())

    # Access raw data for custom analysis
    print(f"\nEquity curve points: {len(result.equity_curve)}")
    print(f"Total trades: {len(result.trade_log)}")
    print(f"Funding events: {len(result.funding_log)}")

    await store.close()

asyncio.run(run_custom_backtest())
```

**Live trading engine:**

```python
import asyncio
from tradingbot.config.settings import Settings
from tradingbot.core.engine import TradingEngine

async def run_live():
    settings = Settings.from_yaml("config/config.yaml")
    engine = TradingEngine(settings)

    try:
        await engine.initialize()   # Connect exchanges, restore positions
        await engine.run()           # Main loop: poll → signal → execute → reconcile
    except KeyboardInterrupt:
        pass
    finally:
        await engine.shutdown()      # Graceful disconnect

asyncio.run(run_live())
```

### 6. Run Live (CLI)

```bash
tradingbot run --config config/config.yaml
```

### 7. Docker Deployment

```bash
# Build and start the live trading bot
docker compose up -d tradingbot

# Run a backtest in a container
docker compose --profile backtest up backtest

# Download market data in a container
docker compose --profile download up download

# View logs
docker compose logs -f tradingbot
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=tradingbot --cov-report=html

# Skip integration tests
pytest tests/ -m "not integration"
```

## CI/CD

GitHub Actions pipelines in `.github/workflows/`:

- **CI** (`ci.yml`): Runs on every push/PR
  - Lint (ruff) + type check (mypy)
  - Tests on Python 3.11 & 3.12 with coverage
  - Security scan (bandit + pip-audit)
  - Docker build verification

- **CD** (`cd.yml`): Runs on version tags (`v*`)
  - Creates GitHub Release with changelog
  - Builds and optionally pushes Docker image
  - Deploy placeholder (configure for your infra)

## Configuration Reference

See [config/config.example.yaml](config/config.example.yaml) for all options:

| Section | Key Settings |
|---------|-------------|
| `strategy` | `min_funding_rate_apr`, `max_position_usd`, `exit_funding_rate_apr` |
| `fees` | `slippage_bps`, `include_withdrawal_fees` |
| `risk` | `max_drawdown_pct`, `margin_ratio_alert`, `position_amount_tolerance_pct` |
| `execution` | `default_order_type`, `twap_threshold_usd`, `limit_order_timeout` |
| `backtest` | `start_date`, `end_date`, `initial_capital` |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Exchange Connectivity | [ccxt](https://github.com/ccxt/ccxt) (async) |
| Data Storage | SQLAlchemy + aiosqlite |
| Configuration | Pydantic + YAML |
| Logging | structlog (JSON) |
| CLI | Click |
| Testing | pytest + pytest-asyncio |
| CI/CD | GitHub Actions |
| Containerization | Docker + docker-compose |

## License

MIT
