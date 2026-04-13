"""
Position reconciliation — validates that exchange positions match expected state.

Runs periodically to:
- Verify spot and perp amounts are delta-neutral (within tolerance)
- Detect position drift from partial fills or external modifications
- Alert on margin / liquidation risk
- Sync internal state with exchange state
"""

from __future__ import annotations

from typing import Any

from tradingbot.exchanges.base import ExchangeBase
from tradingbot.strategy.delta_neutral import ActivePosition
from tradingbot.utils.helpers import calculate_notional
from tradingbot.utils.logger import get_logger

log = get_logger(__name__)


class ReconciliationResult:
    def __init__(self) -> None:
        self.checks_passed: int = 0
        self.checks_failed: int = 0
        self.warnings: list[str] = []
        self.errors: list[str] = []

    @property
    def ok(self) -> bool:
        return self.checks_failed == 0 and len(self.errors) == 0

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.checks_passed,
            "failed": self.checks_failed,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class PositionReconciler:
    """Validates exchange positions against internal tracking."""

    def __init__(
        self,
        exchanges: dict[str, ExchangeBase],
        position_tolerance_pct: float = 0.01,
    ) -> None:
        self._exchanges = exchanges
        self._tolerance = position_tolerance_pct

    async def reconcile(self, positions: dict[str, ActivePosition]) -> ReconciliationResult:
        """Run full reconciliation on all active positions."""
        result = ReconciliationResult()

        for key, pos in positions.items():
            await self._check_position(pos, result)

        if result.ok:
            log.info("reconciliation_passed", positions=len(positions))
        else:
            log.warning(
                "reconciliation_issues",
                failed=result.checks_failed,
                warnings=len(result.warnings),
            )

        return result

    async def _check_position(self, pos: ActivePosition, result: ReconciliationResult) -> None:
        """Reconcile a single position."""
        perp_ex = self._exchanges.get(pos.perp_exchange)
        if not perp_ex:
            result.errors.append(f"{pos.symbol}: perp exchange {pos.perp_exchange} not connected")
            result.checks_failed += 1
            return

        # Check 1: Perp position exists and amount matches
        try:
            exchange_positions = await perp_ex.fetch_positions(pos.symbol)
            if not exchange_positions:
                result.errors.append(f"{pos.symbol}: no perp position found on {pos.perp_exchange}")
                result.checks_failed += 1
                return

            exchange_pos = exchange_positions[0]
            expected = abs(pos.perp_amount)
            actual = abs(exchange_pos.amount)

            if expected > 0:
                drift = abs(actual - expected) / expected
                if drift > self._tolerance:
                    result.errors.append(
                        f"{pos.symbol}: perp amount drift {drift:.4%} "
                        f"(expected={expected}, actual={actual})"
                    )
                    result.checks_failed += 1
                else:
                    result.checks_passed += 1

            # Check 2: Delta-neutrality (spot ≈ perp)
            spot_notional = calculate_notional(pos.spot_entry_price, pos.spot_amount)
            perp_notional = exchange_pos.notional or calculate_notional(
                exchange_pos.mark_price, abs(exchange_pos.amount)
            )

            if spot_notional > 0:
                delta = abs(spot_notional - perp_notional) / spot_notional
                if delta > 0.05:  # 5% delta imbalance
                    result.warnings.append(
                        f"{pos.symbol}: delta imbalance {delta:.4%} "
                        f"(spot_notional={spot_notional:.2f}, perp_notional={perp_notional:.2f})"
                    )
                else:
                    result.checks_passed += 1

            # Check 3: Margin health
            if exchange_pos.liquidation_price > 0 and exchange_pos.mark_price > 0:
                distance_to_liq = abs(exchange_pos.mark_price - exchange_pos.liquidation_price) / exchange_pos.mark_price
                if distance_to_liq < 0.10:  # Within 10% of liquidation
                    result.warnings.append(
                        f"{pos.symbol}: close to liquidation "
                        f"(mark={exchange_pos.mark_price}, liq={exchange_pos.liquidation_price})"
                    )
                else:
                    result.checks_passed += 1

        except Exception as e:
            result.errors.append(f"{pos.symbol}: reconciliation error: {e}")
            result.checks_failed += 1
