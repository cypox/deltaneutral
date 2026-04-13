"""
Risk management module.

Monitors:
- Portfolio-level drawdown
- Per-position loss limits
- Margin ratios across exchanges
- Funding rate anomalies
- Circuit breaker for consecutive errors
"""

from __future__ import annotations

from typing import Any

from tradingbot.config.settings import RiskConfig
from tradingbot.exchanges.base import ExchangeBase
from tradingbot.utils.logger import get_logger
from tradingbot.utils.metrics import funding_rate_to_apr

log = get_logger(__name__)


class RiskAlert:
    """A risk event that may require action."""

    def __init__(
        self, level: str, category: str, message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.level = level  # "warning", "critical", "emergency"
        self.category = category
        self.message = message
        self.data = data or {}

    def __repr__(self) -> str:
        return f"RiskAlert({self.level}, {self.category}: {self.message})"


class RiskManager:
    """Portfolio-wide risk monitoring and enforcement."""

    def __init__(self, config: RiskConfig, exchanges: dict[str, ExchangeBase]) -> None:
        self._config = config
        self._exchanges = exchanges
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._consecutive_errors: int = 0
        self._is_halted: bool = False

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    def update_equity(self, equity: float) -> list[RiskAlert]:
        """Update current equity and check drawdown limits."""
        alerts: list[RiskAlert] = []
        self._current_equity = equity

        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self._config.max_drawdown_pct:
                self._is_halted = True
                alerts.append(
                    RiskAlert(
                        level="emergency",
                        category="drawdown",
                        message=(
                            f"Max drawdown breached: {drawdown:.4%}"
                            f" >= {self._config.max_drawdown_pct:.4%}"
                        ),
                        data={"drawdown": drawdown, "peak": self._peak_equity, "current": equity},
                    )
                )
            elif drawdown >= self._config.max_drawdown_pct * 0.8:
                alerts.append(
                    RiskAlert(
                        level="warning",
                        category="drawdown",
                        message=f"Approaching max drawdown: {drawdown:.4%}",
                    )
                )

        return alerts

    def record_error(self) -> list[RiskAlert]:
        """Record a consecutive error. Returns alerts if circuit breaker triggers."""
        self._consecutive_errors += 1
        alerts: list[RiskAlert] = []

        if self._consecutive_errors >= self._config.max_consecutive_errors:
            self._is_halted = True
            alerts.append(
                RiskAlert(
                    level="emergency",
                    category="circuit_breaker",
                    message=f"Circuit breaker: {self._consecutive_errors} consecutive errors",
                )
            )

        return alerts

    def reset_errors(self) -> None:
        self._consecutive_errors = 0

    async def check_margin_health(self) -> list[RiskAlert]:
        """Check margin health across all exchanges."""
        alerts: list[RiskAlert] = []

        for ex_name, exchange in self._exchanges.items():
            try:
                balance = await exchange.fetch_balance()
                if balance.free < self._config.min_free_margin_usd:
                    alerts.append(
                        RiskAlert(
                            level="warning",
                            category="margin",
                            message=(
                                f"{ex_name}: free margin ${balance.free:.2f}"
                                f" below ${self._config.min_free_margin_usd:.2f}"
                            ),
                            data={"exchange": ex_name, "free": balance.free},
                        )
                    )

                positions = await exchange.fetch_positions()
                for pos in positions:
                    if pos.liquidation_price > 0 and pos.mark_price > 0:
                        distance = abs(pos.mark_price - pos.liquidation_price) / pos.mark_price
                        if distance < 0.05:
                            level = "critical" if distance < 0.03 else "warning"
                            alerts.append(
                                RiskAlert(
                                    level=level,
                                    category="liquidation",
                                    message=(
                                        f"{ex_name} {pos.symbol}:"
                                        f" {distance:.2%} from liquidation"
                                    ),
                                    data={
                                        "exchange": ex_name,
                                        "symbol": pos.symbol,
                                        "mark_price": pos.mark_price,
                                        "liquidation_price": pos.liquidation_price,
                                    },
                                )
                            )

            except Exception as e:
                alerts.append(
                    RiskAlert(
                        level="warning",
                        category="connectivity",
                        message=f"{ex_name}: health check failed: {e}",
                    )
                )

        return alerts

    async def validate_funding_rate(self, exchange: ExchangeBase, symbol: str) -> list[RiskAlert]:
        """Check if funding rate is within acceptable bounds."""
        alerts: list[RiskAlert] = []

        try:
            fr = await exchange.fetch_funding_rate(symbol)
            apr = funding_rate_to_apr(fr.rate)

            if apr > self._config.max_funding_rate_apr:
                alerts.append(
                    RiskAlert(
                        level="warning",
                        category="funding_rate",
                        message=f"{exchange.name} {symbol}: funding rate {apr:.2%} APR too high",
                        data={
                            "exchange": exchange.name, "symbol": symbol,
                            "rate": fr.rate, "apr": apr,
                        },
                    )
                )
        except Exception as e:
            log.warning(
                "funding_rate_check_failed",
                exchange=exchange.name, symbol=symbol, error=str(e),
            )

        return alerts

    def get_status(self) -> dict[str, Any]:
        return {
            "is_halted": self._is_halted,
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "drawdown": (
                (self._peak_equity - self._current_equity) / self._peak_equity
                if self._peak_equity > 0 else 0
            ),
            "consecutive_errors": self._consecutive_errors,
        }

    def resume(self) -> None:
        """Manually resume after halt (requires operator intervention)."""
        self._is_halted = False
        self._consecutive_errors = 0
        log.info("risk_manager_resumed")
