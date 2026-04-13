"""Market data loader — downloads historical OHLCV and funding rate data."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from tradingbot.data.storage import DataStore
from tradingbot.exchanges.base import ExchangeBase
from tradingbot.utils.logger import get_logger

log = get_logger(__name__)

MS_PER_HOUR = 3_600_000
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": MS_PER_HOUR,
    "4h": 4 * MS_PER_HOUR,
    "1d": 24 * MS_PER_HOUR,
}


class MarketDataLoader:
    """Downloads and stores exchange market data."""

    def __init__(self, store: DataStore) -> None:
        self._store = store

    async def download_ohlcv(
        self,
        exchange: ExchangeBase,
        symbol: str,
        timeframe: str = "1h",
        start_date: str = "2024-01-01",
        end_date: str | None = None,
    ) -> int:
        """Download OHLCV data in paginated batches."""
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        if end_date:
            end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        tf_ms = TIMEFRAME_MS.get(timeframe, MS_PER_HOUR)
        total_inserted = 0
        since = start_ms
        batch_size = 500

        log.info(
            "downloading_ohlcv",
            exchange=exchange.name,
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )

        while since < end_ms:
            candles = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
            if not candles:
                break

            inserted = await self._store.insert_ohlcv_batch(exchange.name, symbol, timeframe, candles)
            total_inserted += inserted

            last_ts = int(candles[-1][0])
            if last_ts <= since:
                break
            since = last_ts + tf_ms

            # Respect rate limits
            await asyncio.sleep(0.5)

        log.info(
            "ohlcv_download_complete",
            exchange=exchange.name,
            symbol=symbol,
            records=total_inserted,
        )
        return total_inserted

    async def download_funding_rates(
        self,
        exchange: ExchangeBase,
        symbol: str,
        start_date: str = "2024-01-01",
        end_date: str | None = None,
    ) -> int:
        """Download historical funding rate data."""
        start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        if end_date:
            end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        total_inserted = 0
        since = start_ms

        log.info(
            "downloading_funding_rates",
            exchange=exchange.name,
            symbol=symbol,
            start=start_date,
        )

        while since < end_ms:
            rates = await exchange.fetch_funding_rate_history(symbol, since=since, limit=100)
            if not rates:
                break

            rate_dicts = [{"timestamp": r.next_funding_time, "rate": r.rate} for r in rates]
            inserted = await self._store.insert_funding_rates(exchange.name, symbol, rate_dicts)
            total_inserted += inserted

            last_ts = rates[-1].next_funding_time
            if last_ts <= since:
                break
            since = last_ts + 1

            await asyncio.sleep(0.5)

        log.info(
            "funding_download_complete",
            exchange=exchange.name,
            symbol=symbol,
            records=total_inserted,
        )
        return total_inserted

    async def load_ohlcv_dataframe(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str = "1h",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from storage into a DataFrame."""
        from sqlalchemy import select

        from tradingbot.data.storage import OHLCVRecord

        async with self._store.session() as session:
            stmt = select(OHLCVRecord).where(
                OHLCVRecord.exchange == exchange_name,
                OHLCVRecord.symbol == symbol,
                OHLCVRecord.timeframe == timeframe,
            )

            if start_date:
                start_ms = int(
                    datetime.strptime(start_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                )
                stmt = stmt.where(OHLCVRecord.timestamp >= start_ms)
            if end_date:
                end_ms = int(
                    datetime.strptime(end_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                )
                stmt = stmt.where(OHLCVRecord.timestamp <= end_ms)

            stmt = stmt.order_by(OHLCVRecord.timestamp)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        return df

    async def load_funding_dataframe(
        self,
        exchange_name: str,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load funding rate data from storage into a DataFrame."""
        from sqlalchemy import select

        from tradingbot.data.storage import FundingRateRecord

        async with self._store.session() as session:
            stmt = select(FundingRateRecord).where(
                FundingRateRecord.exchange == exchange_name,
                FundingRateRecord.symbol == symbol,
            )

            if start_date:
                start_ms = int(
                    datetime.strptime(start_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                )
                stmt = stmt.where(FundingRateRecord.timestamp >= start_ms)
            if end_date:
                end_ms = int(
                    datetime.strptime(end_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                )
                stmt = stmt.where(FundingRateRecord.timestamp <= end_ms)

            stmt = stmt.order_by(FundingRateRecord.timestamp)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame(columns=["timestamp", "rate"])

        df = pd.DataFrame([{"timestamp": r.timestamp, "rate": r.rate} for r in rows])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        return df
