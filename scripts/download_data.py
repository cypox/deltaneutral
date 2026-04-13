#!/usr/bin/env python3
"""Script to download historical market data for backtesting."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradingbot.config.settings import ExchangeConfig, Settings
from tradingbot.data.loader import MarketDataLoader
from tradingbot.data.storage import DataStore
from tradingbot.exchanges.factory import create_and_connect
from tradingbot.utils.logger import setup_logging

EXCHANGES = ["binance"]
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
START_DATE = "2024-01-01"
END_DATE = "2025-01-01"


async def main() -> None:
    setup_logging("INFO")

    config_path = Path("config/config.yaml")
    if config_path.exists():
        settings = Settings.from_yaml(config_path)
    else:
        settings = Settings.default()

    Path(settings.app.data_dir).mkdir(parents=True, exist_ok=True)
    store = DataStore(settings.app.db_url)
    await store.initialize()
    loader = MarketDataLoader(store)

    for exchange_id in EXCHANGES:
        ex_config = settings.exchanges.get(exchange_id, ExchangeConfig())
        print(f"\nConnecting to {exchange_id}...")
        exchange = await create_and_connect(exchange_id, ex_config)

        for symbol in SYMBOLS:
            print(f"\n--- {exchange_id} / {symbol} ---")

            # Download spot OHLCV
            print(f"Downloading OHLCV ({TIMEFRAME})...")
            ohlcv_count = await loader.download_ohlcv(
                exchange, symbol, TIMEFRAME, START_DATE, END_DATE
            )
            print(f"  -> {ohlcv_count} candles stored")

            # Download perp funding rates
            perp_symbol = f"{symbol}:USDT"
            try:
                print(f"Downloading funding rates for {perp_symbol}...")
                fr_count = await loader.download_funding_rates(
                    exchange, perp_symbol, START_DATE, END_DATE
                )
                print(f"  -> {fr_count} funding rate records stored")
            except Exception as e:
                print(f"  -> Funding rates skipped: {e}")

            # Download perp OHLCV
            try:
                print(f"Downloading perp OHLCV for {perp_symbol}...")
                perp_count = await loader.download_ohlcv(
                    exchange, perp_symbol, TIMEFRAME, START_DATE, END_DATE
                )
                print(f"  -> {perp_count} perp candles stored")
            except Exception as e:
                print(f"  -> Perp OHLCV skipped: {e}")

        await exchange.close()

    await store.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
