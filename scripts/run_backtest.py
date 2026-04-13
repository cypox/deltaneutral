#!/usr/bin/env python3
"""Script to run a backtest with downloaded data."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradingbot.backtesting.engine import BacktestEngine
from tradingbot.config.settings import Settings
from tradingbot.data.loader import MarketDataLoader
from tradingbot.data.storage import DataStore
from tradingbot.utils.logger import setup_logging

SYMBOL = "BTC/USDT"
EXCHANGE = "binance"


async def main() -> None:
    setup_logging("INFO")

    config_path = Path("config/config.yaml")
    if config_path.exists():
        settings = Settings.from_yaml(config_path)
    else:
        settings = Settings.default()

    store = DataStore(settings.app.db_url)
    await store.initialize()
    loader = MarketDataLoader(store)

    print(f"Loading spot prices for {SYMBOL} from {EXCHANGE}...")
    spot_prices = await loader.load_ohlcv_dataframe(
        EXCHANGE, SYMBOL, "1h",
        settings.backtest.start_date,
        settings.backtest.end_date,
    )
    print(f"  -> {len(spot_prices)} rows")

    perp_symbol = f"{SYMBOL}:USDT"
    print(f"Loading perp prices for {perp_symbol}...")
    perp_prices = await loader.load_ohlcv_dataframe(
        EXCHANGE, perp_symbol, "1h",
        settings.backtest.start_date,
        settings.backtest.end_date,
    )
    print(f"  -> {len(perp_prices)} rows")

    print(f"Loading funding rates for {perp_symbol}...")
    funding_rates = await loader.load_funding_dataframe(
        EXCHANGE, perp_symbol,
        settings.backtest.start_date,
        settings.backtest.end_date,
    )
    print(f"  -> {len(funding_rates)} rows")

    if spot_prices.empty:
        print("\nNo data found! Run scripts/download_data.py first.")
        await store.close()
        return

    # Use spot prices as perp if perp data missing
    if perp_prices.empty:
        print("Using spot prices as perp prices (no perp data)")
        perp_prices = spot_prices.copy()

    # Run backtest
    engine = BacktestEngine(settings.strategy, settings.backtest, settings.fees)
    result = engine.run(spot_prices, perp_prices, funding_rates)

    print(result.print_summary())

    # Save results
    output_dir = Path(settings.app.data_dir) / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    result.trade_log.to_csv(output_dir / "trades.csv", index=False)
    result.funding_log.to_csv(output_dir / "funding.csv", index=False)
    print(f"\nResults saved to {output_dir}/")

    # Optional: generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(result.equity_curve.index, result.equity_curve.values)
        axes[0].set_title("Equity Curve")
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)

        if not result.funding_log.empty:
            fl = result.funding_log.copy()
            fl["timestamp"] = fl["timestamp"].astype("datetime64[ns, UTC]") if "timestamp" in fl.columns else fl.index
            axes[1].bar(range(len(fl)), fl["payment"], alpha=0.7)
            axes[1].set_title("Funding Payments")
            axes[1].set_ylabel("Payment ($)")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "backtest_chart.png", dpi=150)
        print(f"Chart saved to {output_dir}/backtest_chart.png")
    except ImportError:
        print("Install matplotlib for charts: pip install matplotlib")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
