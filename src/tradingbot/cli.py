"""CLI entry point for the trading bot."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import click


@click.group()
def main() -> None:
    """Delta-neutral cryptocurrency cross-exchange trading bot."""


@main.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
def run(config: str) -> None:
    """Start the live trading engine."""
    from tradingbot.config.settings import Settings
    from tradingbot.core.engine import TradingEngine

    settings = Settings.from_yaml(config)
    engine = TradingEngine(settings)

    async def _run() -> None:
        try:
            await engine.initialize()
            await engine.run()
        except KeyboardInterrupt:
            pass
        finally:
            await engine.shutdown()

    asyncio.run(_run())


@main.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
@click.option("--symbol", "-s", default="BTC/USDT", help="Symbol to backtest")
@click.option("--output", "-o", default=None, help="Output directory for results")
def backtest(config: str, symbol: str, output: str | None) -> None:
    """Run a backtest of the delta-neutral strategy."""
    from tradingbot.backtesting.engine import BacktestEngine
    from tradingbot.config.settings import Settings
    from tradingbot.data.storage import DataStore

    settings = Settings.from_yaml(config) if Path(config).exists() else Settings.default()

    async def _backtest() -> None:
        store = DataStore(settings.app.db_url)
        await store.initialize()

        from tradingbot.data.loader import MarketDataLoader

        loader = MarketDataLoader(store)

        spot_prices = await loader.load_ohlcv_dataframe(
            "binance", symbol, "1h",
            settings.backtest.start_date,
            settings.backtest.end_date,
        )
        perp_prices = await loader.load_ohlcv_dataframe(
            "binance", f"{symbol}:USDT", "1h",
            settings.backtest.start_date,
            settings.backtest.end_date,
        )
        funding_rates = await loader.load_funding_dataframe(
            "binance", f"{symbol}:USDT",
            settings.backtest.start_date,
            settings.backtest.end_date,
        )

        if spot_prices.empty:
            click.echo("No price data found. Run 'tradingbot download' first.")
            await store.close()
            return

        if perp_prices.empty:
            perp_prices = spot_prices.copy()

        engine = BacktestEngine(settings.strategy, settings.backtest, settings.fees)
        result = engine.run(spot_prices, perp_prices, funding_rates)

        click.echo(result.print_summary())

        if output:
            out_dir = Path(output)
            out_dir.mkdir(parents=True, exist_ok=True)
            result.equity_curve.to_csv(out_dir / "equity_curve.csv")
            result.trade_log.to_csv(out_dir / "trades.csv", index=False)
            result.funding_log.to_csv(out_dir / "funding.csv", index=False)
            click.echo(f"Results saved to {out_dir}")

        await store.close()

    asyncio.run(_backtest())


@main.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
@click.option("--exchange", "-e", default="binance", help="Exchange to download from")
@click.option("--symbol", "-s", default="BTC/USDT", help="Symbol to download")
@click.option("--timeframe", "-t", default="1h", help="Candle timeframe")
@click.option("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
def download(
    config: str, exchange: str, symbol: str,
    timeframe: str, start: str, end: str | None,
) -> None:
    """Download historical market data."""
    from tradingbot.config.settings import Settings
    from tradingbot.data.loader import MarketDataLoader
    from tradingbot.data.storage import DataStore
    from tradingbot.exchanges.factory import create_and_connect

    settings = Settings.from_yaml(config) if Path(config).exists() else Settings.default()

    async def _download() -> None:
        store = DataStore(settings.app.db_url)
        await store.initialize()

        ex_config = settings.exchanges.get(exchange)
        if not ex_config:
            from tradingbot.config.settings import ExchangeConfig
            ex_config = ExchangeConfig()

        connector = await create_and_connect(exchange, ex_config)
        loader = MarketDataLoader(store)

        click.echo(f"Downloading OHLCV for {symbol} from {exchange}...")
        count = await loader.download_ohlcv(connector, symbol, timeframe, start, end)
        click.echo(f"Downloaded {count} candles")

        # Also download funding rates for perp symbol
        perp_symbol = f"{symbol}:USDT" if ":" not in symbol else symbol
        try:
            click.echo(f"Downloading funding rates for {perp_symbol}...")
            fr_count = await loader.download_funding_rates(connector, perp_symbol, start, end)
            click.echo(f"Downloaded {fr_count} funding rate records")
        except Exception as e:
            click.echo(f"Funding rate download skipped: {e}")

        await connector.close()
        await store.close()

    asyncio.run(_download())


@main.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
@click.option("--capital", default=500.0, help="Your total capital in USD across all exchanges")
@click.option(
    "--exchanges", "-e", multiple=True,
    default=["binance", "bybit"], help="Exchanges to scan",
)
@click.option("--top", default=20, help="Number of top opportunities to show")
def scan(config: str, capital: float, exchanges: tuple[str, ...], top: int) -> None:
    """Scan exchanges for the best funding rate arbitrage pairs.

    Shows which pairs have the highest funding rates, estimated daily/monthly
    returns for your capital, and how many days to break even on entry fees.

    Examples:
        tradingbot scan --capital 500
        tradingbot scan --capital 1000 -e binance -e bybit -e okx
    """
    from tradingbot.config.settings import ExchangeConfig, Settings
    from tradingbot.exchanges.factory import create_and_connect
    from tradingbot.strategy.scanner import PairScanner, format_scan_report

    settings = Settings.from_yaml(config) if Path(config).exists() else Settings.default()

    async def _scan() -> None:
        connectors: dict[str, Any] = {}
        for ex_id in exchanges:
            ex_config = settings.exchanges.get(ex_id, ExchangeConfig())
            try:
                click.echo(f"Connecting to {ex_id}...")
                connector = await create_and_connect(ex_id, ex_config)
                connectors[ex_id] = connector
            except Exception as e:
                click.echo(f"Failed to connect to {ex_id}: {e}")

        if not connectors:
            click.echo("No exchanges connected. Check your config or internet connection.")
            return

        scanner = PairScanner(
            exchanges=connectors,
            slippage_bps=settings.fees.slippage_bps,
            leverage=settings.strategy.max_leverage,
        )

        click.echo(f"Scanning {len(connectors)} exchange(s) for opportunities...")
        result = await scanner.scan(capital_usd=capital)

        click.echo(format_scan_report(result))

        for connector in connectors.values():
            await connector.close()

    asyncio.run(_scan())


@main.command()
@click.option("--capital", default=500.0, help="Your total capital in USD")
@click.option(
    "--funding-rate", default=0.0001,
    help="Assumed per-8h funding rate (e.g., 0.0001 = 0.01%)",
)
@click.option("--taker-fee", default=0.0005, help="Taker fee rate")
@click.option("--leverage", default=3, help="Perpetual leverage")
@click.option("--hold-days", default=30, help="Expected hold duration in days")
def estimate(
    capital: float, funding_rate: float, taker_fee: float,
    leverage: int, hold_days: int,
) -> None:
    """Estimate returns for a given capital and funding rate — no API keys needed.

    This is an offline calculator. Use `tradingbot scan` for live data.

    Examples:
        tradingbot estimate --capital 500
        tradingbot estimate --capital 500 --funding-rate 0.0003
        tradingbot estimate --capital 1000 --taker-fee 0.001
    """
    from tradingbot.utils.metrics import funding_rate_to_apr

    capital_per_exchange = capital / 2
    position_usd = capital_per_exchange  # limited by spot side

    funding_apr = funding_rate_to_apr(funding_rate)

    slippage = 0.0005  # 5 bps
    entry_fee_pct = taker_fee * 2 + slippage * 2  # spot + perp entry
    exit_fee_pct = taker_fee * 2 + slippage * 2   # spot + perp exit
    round_trip_pct = entry_fee_pct + exit_fee_pct

    entry_cost = position_usd * entry_fee_pct * 2  # both legs
    exit_cost = position_usd * exit_fee_pct * 2

    daily_funding = position_usd * funding_rate * 3  # 3 payments per day
    breakeven_days = entry_cost / daily_funding if daily_funding > 0 else float("inf")
    daily_amortized_exit = exit_cost / hold_days
    daily_net = daily_funding - daily_amortized_exit
    monthly_net = daily_net * 30
    yearly_net = daily_net * 365

    net_apr = funding_apr - (round_trip_pct * 365 / hold_days)

    click.echo("")
    click.echo("=" * 60)
    click.echo("  PROFITABILITY ESTIMATOR — Delta-Neutral Funding Arb")
    click.echo("=" * 60)
    click.echo(f"  Capital:           ${capital:,.2f}")
    click.echo(f"  Per exchange:      ${capital_per_exchange:,.2f}")
    click.echo(f"  Position size:     ${position_usd:,.2f} per leg")
    click.echo(f"  Leverage (perp):   {leverage}x")
    click.echo("-" * 60)
    click.echo(f"  Funding rate:      {funding_rate:.6f} per 8h")
    click.echo(f"  Funding APR:       {funding_apr:.2%}")
    click.echo(f"  Taker fee:         {taker_fee:.4%}")
    click.echo(f"  Round-trip fees:   {round_trip_pct:.4%}")
    click.echo(f"  Net APR:           {net_apr:.2%}")
    click.echo("-" * 60)
    click.echo(f"  Entry cost (fees): ${entry_cost:.4f}")
    click.echo(f"  Daily funding:     ${daily_funding:.4f}")
    click.echo(f"  Break-even:        {breakeven_days:.1f} days")
    click.echo("-" * 60)
    click.echo(f"  Daily net profit:  ${daily_net:.4f}")
    click.echo(f"  Monthly estimate:  ${monthly_net:.2f}")
    click.echo(f"  Yearly estimate:   ${yearly_net:.2f}")
    click.echo("=" * 60)
    click.echo("")

    if daily_net <= 0:
        click.echo("  ⚠ At this funding rate, fees exceed income.")
        click.echo("  Look for pairs with higher funding rates using 'tradingbot scan'.")
    elif breakeven_days > 7:
        click.echo(f"  ⚠ Takes {breakeven_days:.0f} days just to cover entry fees.")
        click.echo("  Only enter if you expect funding to stay elevated that long.")
    else:
        click.echo(f"  ✓ Profitable after ~{breakeven_days:.1f} days. Looks viable.")

    click.echo("")
    click.echo("  Note: Funding rates change every 8h and can turn negative.")
    click.echo("  These estimates assume the rate stays constant.")
    click.echo("")


@main.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
@click.option(
    "--exchanges", "-e", multiple=True,
    default=["binance", "bybit"], help="Exchanges to rank",
)
def rank(config: str, exchanges: tuple[str, ...]) -> None:
    """Rank exchanges by cost-effectiveness for delta-neutral trading.

    Scores each exchange on taker fees, funding rates, and symbol availability.

    Examples:
        tradingbot rank
        tradingbot rank -e binance -e bybit -e okx
    """
    from tradingbot.config.settings import ExchangeConfig, Settings
    from tradingbot.exchanges.factory import create_and_connect
    from tradingbot.strategy.scanner import PairScanner, format_exchange_ranking

    settings = Settings.from_yaml(config) if Path(config).exists() else Settings.default()

    async def _rank() -> None:
        connectors: dict[str, Any] = {}
        for ex_id in exchanges:
            ex_config = settings.exchanges.get(ex_id, ExchangeConfig())
            try:
                click.echo(f"Connecting to {ex_id}...")
                connector = await create_and_connect(ex_id, ex_config)
                connectors[ex_id] = connector
            except Exception as e:
                click.echo(f"Failed to connect to {ex_id}: {e}")

        if not connectors:
            click.echo("No exchanges connected.")
            return

        scanner = PairScanner(
            exchanges=connectors,
            slippage_bps=settings.fees.slippage_bps,
            leverage=settings.strategy.max_leverage,
        )

        click.echo(f"Ranking {len(connectors)} exchange(s)...")
        rankings = await scanner.rank_exchanges()
        click.echo(format_exchange_ranking(rankings))

        for connector in connectors.values():
            await connector.close()

    asyncio.run(_rank())


@main.command()
@click.option("--config", "-c", default="config/config.yaml", help="Config file path")
@click.option(
    "--exchanges", "-e", multiple=True,
    default=["binance", "bybit"], help="Exchanges to check",
)
@click.option("--top", default=20, help="Number of top routes to show")
def routes(config: str, exchanges: tuple[str, ...], top: int) -> None:
    """Find the optimal exchange route per symbol.

    For each trading pair, shows which exchange to buy spot on and which
    to short perp on for minimum fees and maximum funding income.

    Examples:
        tradingbot routes
        tradingbot routes -e binance -e bybit -e okx --top 10
    """
    from tradingbot.config.settings import ExchangeConfig, Settings
    from tradingbot.exchanges.factory import create_and_connect
    from tradingbot.strategy.scanner import PairScanner, format_route_report

    settings = Settings.from_yaml(config) if Path(config).exists() else Settings.default()

    async def _routes() -> None:
        connectors: dict[str, Any] = {}
        for ex_id in exchanges:
            ex_config = settings.exchanges.get(ex_id, ExchangeConfig())
            try:
                click.echo(f"Connecting to {ex_id}...")
                connector = await create_and_connect(ex_id, ex_config)
                connectors[ex_id] = connector
            except Exception as e:
                click.echo(f"Failed to connect to {ex_id}: {e}")

        if not connectors:
            click.echo("No exchanges connected.")
            return

        scanner = PairScanner(
            exchanges=connectors,
            slippage_bps=settings.fees.slippage_bps,
            leverage=settings.strategy.max_leverage,
        )

        click.echo(f"Finding optimal routes across {len(connectors)} exchange(s)...")
        best_routes = await scanner.find_best_routes()
        click.echo(format_route_report(best_routes[:top]))

        for connector in connectors.values():
            await connector.close()

    asyncio.run(_routes())


@main.command()
def status() -> None:
    """Show current bot status (positions, PnL, risk)."""
    click.echo("Status check requires a running instance. Use monitoring endpoint.")


if __name__ == "__main__":
    main()
