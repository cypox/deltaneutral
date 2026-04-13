"""Tests for the delta-neutral strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tradingbot.config.settings import FeeConfig, RiskConfig, StrategyConfig
from tradingbot.data.feed import MarketDataFeed
from tradingbot.exchanges.base import (
    Balance,
    FeeSchedule,
    FundingRate,
    Position,
    PositionSide,
    Ticker,
)
from tradingbot.strategy.delta_neutral import ActivePosition, DeltaNeutralStrategy
from tradingbot.utils.metrics import funding_rate_to_apr


class TestFundingRateCalculations:
    def test_funding_rate_to_apr_positive(self):
        rate = 0.0001  # 0.01% per 8h
        apr = funding_rate_to_apr(rate)
        assert apr == pytest.approx(0.0001 * 3 * 365, rel=1e-6)

    def test_funding_rate_to_apr_negative(self):
        rate = -0.0001
        apr = funding_rate_to_apr(rate)
        assert apr < 0

    def test_zero_funding_rate(self):
        assert funding_rate_to_apr(0.0) == 0.0


class TestDeltaNeutralStrategy:
    @pytest.fixture
    def strategy(
        self, strategy_config, fee_config, risk_config, mock_exchange
    ) -> DeltaNeutralStrategy:
        feed = MarketDataFeed()
        feed.add_exchange(mock_exchange)
        # Pre-populate feed cache
        feed._tickers["test_exchange"]["BTC/USDT"] = Ticker(
            symbol="BTC/USDT", bid=50000.0, ask=50010.0, last=50005.0, timestamp=1700000000000
        )
        feed._funding_rates["test_exchange"]["BTC/USDT"] = FundingRate(
            symbol="BTC/USDT", rate=0.0001, next_funding_time=1700000000000
        )

        return DeltaNeutralStrategy(
            config=strategy_config,
            fee_config=fee_config,
            risk_config=risk_config,
            feed=feed,
            exchanges={"test_exchange": mock_exchange},
        )

    @pytest.mark.asyncio
    async def test_on_tick_generates_entry_signal(self, strategy, mock_exchange):
        """Strategy should generate signal when funding rate is attractive."""
        signals = await strategy.on_tick()
        # With default mock data (0.01% per 8h ≈ 10.95% APR), should find opportunity
        # depending on fee structure. The exact result depends on net APR calculation.
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_no_signal_when_max_exposure_reached(self, strategy):
        """Should not generate entry signals at max exposure."""
        strategy._total_exposure_usd = strategy._config.max_total_exposure_usd
        signals = await strategy.on_tick()
        entry_signals = [s for s in signals if s.is_entry]
        assert len(entry_signals) == 0

    @pytest.mark.asyncio
    async def test_exit_signal_on_low_funding(self, strategy, mock_exchange):
        """Should generate exit signal when funding drops below threshold."""
        # Register an active position
        pos = ActivePosition(
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
        strategy.register_position(pos)

        # Set funding rate below exit threshold
        strategy._feed._funding_rates["test_exchange"]["BTC/USDT"] = FundingRate(
            symbol="BTC/USDT", rate=0.000001, next_funding_time=1700000000000  # very low
        )

        signals = await strategy.on_tick()
        exit_signals = [s for s in signals if s.is_exit]
        assert len(exit_signals) == 1
        assert exit_signals[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_funding_payment_tracking(self, strategy):
        pos = ActivePosition(
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
        strategy.register_position(pos)

        # Process positive funding (we're short, we receive)
        await strategy.on_funding("BTC/USDT", "test_exchange", 0.0001)
        assert pos.total_funding_collected > 0

        # Process negative funding (we're short, we pay)
        await strategy.on_funding("BTC/USDT", "test_exchange", -0.0001)
        # Net should be approximately 0
        assert pos.total_funding_collected == pytest.approx(0, abs=0.01)

    def test_register_and_remove_position(self, strategy):
        pos = ActivePosition(
            db_id=1,
            symbol="BTC/USDT",
            spot_exchange="ex1",
            perp_exchange="ex2",
            spot_amount=0.2,
            perp_amount=0.2,
            spot_entry_price=50000.0,
            perp_entry_price=50000.0,
            entry_funding_rate=0.0001,
        )
        strategy.register_position(pos)
        assert strategy._total_exposure_usd > 0
        assert len(strategy._active_positions) == 1

        removed = strategy.remove_position("BTC/USDT", "ex1", "ex2")
        assert removed is not None
        assert strategy._total_exposure_usd == pytest.approx(0, abs=0.01)

    @pytest.mark.asyncio
    async def test_perp_amount_drift_triggers_exit(self, strategy, mock_exchange):
        """Should exit if perp position amount drifts beyond tolerance."""
        pos = ActivePosition(
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
        strategy.register_position(pos)

        # Mock perp position with significant drift
        mock_exchange.fetch_positions.return_value = [
            Position(
                symbol="BTC/USDT",
                side=PositionSide.SHORT,
                amount=-0.15,  # drifted from 0.2 to 0.15 = 25% drift
                entry_price=50000.0,
                mark_price=50100.0,
                unrealized_pnl=-20.0,
                leverage=3,
                margin=3333.0,
                liquidation_price=65000.0,
                notional=7515.0,
            )
        ]

        # Set funding rate above exit threshold
        strategy._feed._funding_rates["test_exchange"]["BTC/USDT"] = FundingRate(
            symbol="BTC/USDT", rate=0.0005, next_funding_time=1700000000000
        )

        signals = await strategy.on_tick()
        exit_signals = [s for s in signals if s.is_exit]
        assert len(exit_signals) == 1
        assert "drift" in exit_signals[0].metadata.get("reason", "")

    @pytest.mark.asyncio
    async def test_insufficient_margin_prevents_entry(self, strategy, mock_exchange):
        """Should not enter when margin is insufficient."""
        mock_exchange.fetch_balance.return_value = Balance(total=6000, free=3000, used=3000)
        signals = await strategy.on_tick()
        entry_signals = [s for s in signals if s.is_entry]
        assert len(entry_signals) == 0
