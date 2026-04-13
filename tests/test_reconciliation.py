"""Tests for reconciliation."""

from __future__ import annotations

import pytest

from tradingbot.exchanges.base import Position, PositionSide
from tradingbot.execution.reconciliation import PositionReconciler
from tradingbot.strategy.delta_neutral import ActivePosition


class TestReconciliation:
    @pytest.fixture
    def reconciler(self, mock_exchange):
        return PositionReconciler(
            exchanges={"test_exchange": mock_exchange},
            position_tolerance_pct=0.01,
        )

    @pytest.mark.asyncio
    async def test_reconciliation_passes(self, reconciler, mock_exchange):
        """Should pass when exchange state matches internal state."""
        positions = {
            "BTC/USDT:test_exchange:test_exchange": ActivePosition(
                db_id=1,
                symbol="BTC/USDT",
                spot_exchange="test_exchange",
                perp_exchange="test_exchange",
                spot_amount=0.2,
                perp_amount=0.2,
                spot_entry_price=50000.0,
                perp_entry_price=50000.0,
                entry_funding_rate=0.0001,
            )
        }

        result = await reconciler.reconcile(positions)
        assert result.ok

    @pytest.mark.asyncio
    async def test_reconciliation_detects_drift(self, reconciler, mock_exchange):
        """Should detect when perp amount doesn't match expected."""
        mock_exchange.fetch_positions.return_value = [
            Position(
                symbol="BTC/USDT",
                side=PositionSide.SHORT,
                amount=-0.1,  # Expected 0.2, actual 0.1 = 50% drift
                entry_price=50000.0,
                mark_price=50100.0,
                unrealized_pnl=-20.0,
                leverage=3,
                margin=3333.0,
                liquidation_price=65000.0,
                notional=5010.0,
            )
        ]

        positions = {
            "BTC/USDT:test_exchange:test_exchange": ActivePosition(
                db_id=1,
                symbol="BTC/USDT",
                spot_exchange="test_exchange",
                perp_exchange="test_exchange",
                spot_amount=0.2,
                perp_amount=0.2,
                spot_entry_price=50000.0,
                perp_entry_price=50000.0,
                entry_funding_rate=0.0001,
            )
        }

        result = await reconciler.reconcile(positions)
        assert not result.ok
        assert result.checks_failed > 0

    @pytest.mark.asyncio
    async def test_reconciliation_detects_missing_position(self, reconciler, mock_exchange):
        """Should detect when perp position is missing."""
        mock_exchange.fetch_positions.return_value = []

        positions = {
            "BTC/USDT:test_exchange:test_exchange": ActivePosition(
                db_id=1,
                symbol="BTC/USDT",
                spot_exchange="test_exchange",
                perp_exchange="test_exchange",
                spot_amount=0.2,
                perp_amount=0.2,
                spot_entry_price=50000.0,
                perp_entry_price=50000.0,
                entry_funding_rate=0.0001,
            )
        }

        result = await reconciler.reconcile(positions)
        assert not result.ok
