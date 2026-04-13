"""Abstract base for exchange connectors."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: int


@dataclass
class FundingRate:
    symbol: str
    rate: float  # per-period rate (e.g., 0.0001 = 0.01%)
    next_funding_time: int  # unix ms
    interval_hours: int = 8


@dataclass
class FeeSchedule:
    maker: float  # e.g., 0.0002 = 0.02%
    taker: float  # e.g., 0.0005 = 0.05%
    funding_rate: float = 0.0  # current funding rate


@dataclass
class OrderResult:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: float
    filled: float
    cost: float
    fee: float
    fee_currency: str = "USDT"
    status: str = "closed"
    timestamp: int = 0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: PositionSide
    amount: float  # signed: positive=long, negative=short
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    margin: float
    liquidation_price: float
    notional: float


@dataclass
class Balance:
    total: float
    free: float
    used: float


class ExchangeBase(abc.ABC):
    """Abstract exchange connector interface."""

    name: str

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    @abc.abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker: ...

    @abc.abstractmethod
    async def fetch_tickers(self, symbols: list[str]) -> dict[str, Ticker]: ...

    @abc.abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> FundingRate: ...

    @abc.abstractmethod
    async def fetch_fee_schedule(self, symbol: str) -> FeeSchedule: ...

    @abc.abstractmethod
    async def fetch_balance(self, currency: str = "USDT") -> Balance: ...

    @abc.abstractmethod
    async def fetch_positions(self, symbol: str | None = None) -> list[Position]: ...

    @abc.abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> OrderResult: ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool: ...

    @abc.abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> OrderResult: ...

    @abc.abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> None: ...

    @abc.abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", since: int | None = None, limit: int = 500
    ) -> list[list[float]]: ...

    @abc.abstractmethod
    async def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int = 100
    ) -> list[FundingRate]: ...

    @abc.abstractmethod
    async def get_market_info(self, symbol: str) -> dict[str, Any]: ...
