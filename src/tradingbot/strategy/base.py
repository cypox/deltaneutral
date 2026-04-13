"""Abstract base strategy."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Signal:
    """Trading signal emitted by a strategy."""

    symbol: str
    action: str  # "open_long_spot_short_perp", "close", "rebalance"
    spot_exchange: str = ""
    perp_exchange: str = ""
    amount_usd: float = 0.0
    funding_rate: float = 0.0
    funding_rate_apr: float = 0.0
    spread_pct: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.action.startswith("open")

    @property
    def is_exit(self) -> bool:
        return self.action in ("close", "close_position")


class BaseStrategy(abc.ABC):
    """Interface for all trading strategies."""

    name: str

    @abc.abstractmethod
    async def on_tick(self) -> list[Signal]:
        """Called each tick to generate signals."""
        ...

    @abc.abstractmethod
    async def on_funding(self, symbol: str, exchange: str, rate: float) -> None:
        """Called when a funding payment occurs."""
        ...

    @abc.abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """Return current strategy state for monitoring."""
        ...
