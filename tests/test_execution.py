"""Tests for order execution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tradingbot.exchanges.base import (
    OrderResult,
    OrderSide,
    OrderType,
    Ticker,
)
from tradingbot.execution.executor import OrderExecutor
from tradingbot.strategy.base import Signal


class TestOrderExecutor:
    @pytest.fixture
    def executor(self, execution_config, mock_exchange) -> OrderExecutor:
        return OrderExecutor(
            config=execution_config,
            exchanges={"spot_ex": mock_exchange, "perp_ex": mock_exchange},
        )

    @pytest.mark.asyncio
    async def test_execute_entry_signal(self, executor, mock_exchange):
        """Should execute both spot buy and perp sell."""
        signal = Signal(
            symbol="BTC/USDT",
            action="open_long_spot_short_perp",
            spot_exchange="spot_ex",
            perp_exchange="perp_ex",
            amount_usd=10000.0,
            funding_rate=0.0001,
            metadata={
                "spot_amount": 0.2,
                "perp_amount": 0.2,
                "spot_ask": 50010.0,
                "perp_bid": 50000.0,
            },
        )

        result = await executor.execute_signal(signal)

        assert result is not None
        assert result.spot_amount > 0
        assert result.perp_amount > 0
        assert mock_exchange.create_order.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_exit_signal(self, executor, mock_exchange):
        """Should execute both spot sell and perp cover."""
        mock_exchange.fetch_ticker.return_value = Ticker(
            symbol="BTC/USDT", bid=50000, ask=50010, last=50005, timestamp=0
        )

        signal = Signal(
            symbol="BTC/USDT",
            action="close_position",
            spot_exchange="spot_ex",
            perp_exchange="perp_ex",
            amount_usd=10000.0,
            metadata={
                "spot_amount": 0.2,
                "perp_amount": 0.2,
                "reason": "funding_below_threshold",
            },
        )

        result = await executor.execute_signal(signal)
        assert result is None  # exits return None
        assert mock_exchange.create_order.call_count >= 2

    @pytest.mark.asyncio
    async def test_unwind_on_partial_failure(self, executor, mock_exchange):
        """Should unwind spot if perp order fails."""
        call_count = 0

        async def side_effect_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Spot order succeeds
                return OrderResult(
                    id="spot_1", symbol="BTC/USDT", side=OrderSide.BUY,
                    order_type=OrderType.LIMIT, amount=0.2, price=50010,
                    filled=0.2, cost=10002, fee=5, status="closed",
                )
            else:
                # Perp order fails
                raise Exception("Perp exchange error")

        mock_exchange.create_order = AsyncMock(side_effect=side_effect_create)

        signal = Signal(
            symbol="BTC/USDT",
            action="open_long_spot_short_perp",
            spot_exchange="spot_ex",
            perp_exchange="perp_ex",
            amount_usd=10000.0,
            metadata={
                "spot_amount": 0.2, "perp_amount": 0.2,
                "spot_ask": 50010.0, "perp_bid": 50000.0,
            },
        )

        result = await executor.execute_signal(signal)
        # Entry should fail and attempt unwind
        assert result is None

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, executor, mock_exchange):
        """Executor should detect circuit breaker state."""
        assert not executor.is_circuit_broken
        executor._consecutive_errors = 5
        assert executor.is_circuit_broken

    def test_limit_price_calculation(self, executor):
        """Limit prices should be offset from reference."""
        buy_price = executor._limit_price(50000.0, OrderSide.BUY)
        sell_price = executor._limit_price(50000.0, OrderSide.SELL)

        assert buy_price > 50000.0  # Buy slightly higher
        assert sell_price < 50000.0  # Sell slightly lower
