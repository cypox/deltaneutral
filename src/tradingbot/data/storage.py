"""SQLAlchemy async storage for market data, trades, and positions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class OHLCVRecord(Base):
    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "timeframe", "timestamp", name="uq_ohlcv"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(30), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(BigInteger, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class FundingRateRecord(Base):
    __tablename__ = "funding_rates"
    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "timestamp", name="uq_funding"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(30), nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    rate = Column(Float, nullable=False)


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), nullable=False, unique=True)
    strategy = Column(String(50), nullable=False, index=True)
    symbol = Column(String(30), nullable=False)
    exchange = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    order_type = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    pnl = Column(Float, default=0.0)
    funding_pnl = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    info = Column(Text, default="{}")


class PositionRecord(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False, index=True)
    symbol = Column(String(30), nullable=False)
    spot_exchange = Column(String(50), nullable=False)
    perp_exchange = Column(String(50), nullable=False)
    spot_amount = Column(Float, nullable=False)
    perp_amount = Column(Float, nullable=False)
    spot_entry_price = Column(Float, nullable=False)
    perp_entry_price = Column(Float, nullable=False)
    entry_funding_rate = Column(Float, nullable=False)
    total_funding_collected = Column(Float, default=0.0)
    total_fees = Column(Float, default=0.0)
    status = Column(String(20), default="open", index=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)


class DataStore:
    """Async database interface."""

    def __init__(self, db_url: str) -> None:
        self._engine = create_async_engine(db_url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        """Create all tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    def session(self) -> AsyncSession:
        return self._session_factory()

    async def insert_ohlcv_batch(
        self, exchange: str, symbol: str, timeframe: str, candles: list[list[float]]
    ) -> int:
        """Insert OHLCV candles, skipping duplicates."""
        records = []
        for c in candles:
            records.append(
                OHLCVRecord(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=int(c[0]),
                    open=c[1],
                    high=c[2],
                    low=c[3],
                    close=c[4],
                    volume=c[5],
                )
            )

        inserted = 0
        async with self.session() as session:
            for record in records:
                try:
                    session.add(record)
                    await session.flush()
                    inserted += 1
                except Exception:
                    await session.rollback()
                    session = self._session_factory()
            await session.commit()
        return inserted

    async def insert_funding_rates(
        self, exchange: str, symbol: str, rates: list[dict[str, Any]]
    ) -> int:
        """Insert funding rate records."""
        inserted = 0
        async with self.session() as session:
            for r in rates:
                try:
                    record = FundingRateRecord(
                        exchange=exchange,
                        symbol=symbol,
                        timestamp=r["timestamp"],
                        rate=r["rate"],
                    )
                    session.add(record)
                    await session.flush()
                    inserted += 1
                except Exception:
                    await session.rollback()
                    session = self._session_factory()
            await session.commit()
        return inserted

    async def insert_trade(self, trade: dict[str, Any]) -> None:
        async with self.session() as session:
            session.add(TradeRecord(**trade))
            await session.commit()

    async def insert_position(self, position: dict[str, Any]) -> int:
        async with self.session() as session:
            record = PositionRecord(**position)
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id  # type: ignore[return-value]

    async def update_position(self, position_id: int, updates: dict[str, Any]) -> None:
        async with self.session() as session:
            from sqlalchemy import update

            stmt = update(PositionRecord).where(PositionRecord.id == position_id).values(**updates)
            await session.execute(stmt)
            await session.commit()

    async def get_open_positions(self, strategy: str) -> list[dict[str, Any]]:
        async with self.session() as session:
            from sqlalchemy import select

            stmt = select(PositionRecord).where(
                PositionRecord.strategy == strategy, PositionRecord.status == "open"
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "spot_exchange": r.spot_exchange,
                    "perp_exchange": r.perp_exchange,
                    "spot_amount": r.spot_amount,
                    "perp_amount": r.perp_amount,
                    "spot_entry_price": r.spot_entry_price,
                    "perp_entry_price": r.perp_entry_price,
                    "entry_funding_rate": r.entry_funding_rate,
                    "total_funding_collected": r.total_funding_collected,
                    "total_fees": r.total_fees,
                    "opened_at": r.opened_at,
                }
                for r in rows
            ]
