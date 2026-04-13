"""
Portfolio tracker — tracks aggregate state across all positions and exchanges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradingbot.exchanges.base import ExchangeBase
from tradingbot.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""

    timestamp: int
    total_equity: float
    free_capital: float
    total_positions_value: float
    total_unrealized_pnl: float
    total_funding_collected: float
    total_fees_paid: float
    exchange_balances: dict[str, float] = field(default_factory=dict)


class Portfolio:
    """Aggregates balances and positions across exchanges."""

    def __init__(self, exchanges: dict[str, ExchangeBase]) -> None:
        self._exchanges = exchanges
        self._snapshots: list[PortfolioSnapshot] = []
        self._total_funding: float = 0.0
        self._total_fees: float = 0.0

    def record_funding(self, amount: float) -> None:
        self._total_funding += amount

    def record_fee(self, amount: float) -> None:
        self._total_fees += amount

    async def snapshot(self) -> PortfolioSnapshot:
        """Take a portfolio snapshot across all exchanges."""
        total_equity = 0.0
        total_free = 0.0
        total_unrealized = 0.0
        exchange_balances: dict[str, float] = {}

        for ex_name, exchange in self._exchanges.items():
            try:
                balance = await exchange.fetch_balance()
                total_equity += balance.total
                total_free += balance.free
                exchange_balances[ex_name] = balance.total

                positions = await exchange.fetch_positions()
                for pos in positions:
                    total_unrealized += pos.unrealized_pnl

            except Exception as e:
                log.warning("snapshot_error", exchange=ex_name, error=str(e))

        import time

        snap = PortfolioSnapshot(
            timestamp=int(time.time() * 1000),
            total_equity=total_equity,
            free_capital=total_free,
            total_positions_value=total_equity - total_free,
            total_unrealized_pnl=total_unrealized,
            total_funding_collected=self._total_funding,
            total_fees_paid=self._total_fees,
            exchange_balances=exchange_balances,
        )
        self._snapshots.append(snap)
        return snap

    def get_history(self) -> list[PortfolioSnapshot]:
        return self._snapshots
