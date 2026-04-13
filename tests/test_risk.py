"""Tests for risk management."""

from __future__ import annotations

import pytest

from tradingbot.core.risk import RiskManager
from tradingbot.exchanges.base import Balance


class TestRiskManager:
    @pytest.fixture
    def risk_manager(self, risk_config, mock_exchange):
        return RiskManager(
            config=risk_config,
            exchanges={"test_exchange": mock_exchange},
        )

    def test_drawdown_detection(self, risk_manager):
        """Should detect max drawdown breach."""
        risk_manager.update_equity(100_000)
        alerts = risk_manager.update_equity(94_000)  # 6% drawdown > 5% limit

        assert len(alerts) > 0
        assert any(a.level == "emergency" for a in alerts)
        assert risk_manager.is_halted

    def test_drawdown_warning(self, risk_manager):
        """Should warn when approaching max drawdown."""
        risk_manager.update_equity(100_000)
        alerts = risk_manager.update_equity(96_000)  # 4% = 80% of 5% limit

        assert len(alerts) > 0
        assert any(a.level == "warning" for a in alerts)
        assert not risk_manager.is_halted

    def test_no_alert_small_drawdown(self, risk_manager):
        risk_manager.update_equity(100_000)
        alerts = risk_manager.update_equity(99_000)  # 1% drawdown

        assert len(alerts) == 0
        assert not risk_manager.is_halted

    def test_circuit_breaker(self, risk_manager):
        """Should halt after max consecutive errors."""
        for _ in range(4):
            alerts = risk_manager.record_error()
            assert not risk_manager.is_halted

        alerts = risk_manager.record_error()  # 5th error
        assert risk_manager.is_halted
        assert any(a.category == "circuit_breaker" for a in alerts)

    def test_error_reset(self, risk_manager):
        for _ in range(3):
            risk_manager.record_error()
        risk_manager.reset_errors()

        # Should be able to handle errors again
        for _ in range(4):
            risk_manager.record_error()
        assert not risk_manager.is_halted

    @pytest.mark.asyncio
    async def test_margin_health_check(self, risk_manager, mock_exchange):
        """Should check margin on all connected exchanges."""
        alerts = await risk_manager.check_margin_health()
        # With default mock (80k free > 5k min), should pass
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_low_margin_alert(self, risk_manager, mock_exchange):
        """Should alert on low margin."""
        mock_exchange.fetch_balance.return_value = Balance(total=6000, free=3000, used=3000)
        alerts = await risk_manager.check_margin_health()
        assert any(a.category == "margin" for a in alerts)

    def test_resume_after_halt(self, risk_manager):
        risk_manager.update_equity(100_000)
        risk_manager.update_equity(94_000)  # Trigger halt
        assert risk_manager.is_halted

        risk_manager.resume()
        assert not risk_manager.is_halted

    def test_status_reporting(self, risk_manager):
        risk_manager.update_equity(100_000)
        risk_manager.update_equity(97_000)

        status = risk_manager.get_status()
        assert status["peak_equity"] == 100_000
        assert status["current_equity"] == 97_000
        assert status["drawdown"] == pytest.approx(0.03)
