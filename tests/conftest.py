"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from tradingbot.config.settings import (
    ExecutionConfig,
    FeeConfig,
    RiskConfig,
    StrategyConfig,
)
from tradingbot.exchanges.base import (
    Balance,
    ExchangeBase,
    FeeSchedule,
    FundingRate,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    Ticker,
)


@pytest.fixture
def strategy_config() -> StrategyConfig:
    return StrategyConfig(
        symbols=["BTC/USDT"],
        min_funding_rate_apr=0.10,
        max_entry_spread_pct=0.003,
        max_position_usd=10000.0,
        min_position_usd=100.0,
        max_total_exposure_usd=50000.0,
        exit_funding_rate_apr=0.02,
        max_leverage=3,
    )


@pytest.fixture
def fee_config() -> FeeConfig:
    return FeeConfig(slippage_bps=5.0, include_withdrawal_fees=True)


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_drawdown_pct=0.05,
        max_position_loss_pct=0.03,
        margin_ratio_alert=0.7,
        max_funding_rate_apr=2.0,
        min_free_margin_usd=5000.0,
        max_consecutive_errors=5,
        position_amount_tolerance_pct=0.01,
    )


@pytest.fixture
def execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        default_order_type="limit",
        limit_offset_bps=2.0,
        max_retries=2,
        limit_order_timeout=5,
        twap_threshold_usd=50000.0,
    )


@pytest.fixture
def mock_exchange() -> AsyncMock:
    """Create a mock exchange with standard responses."""
    exchange = AsyncMock(spec=ExchangeBase)
    exchange.name = "test_exchange"

    exchange.fetch_ticker.return_value = Ticker(
        symbol="BTC/USDT", bid=50000.0, ask=50010.0, last=50005.0, timestamp=1700000000000
    )

    exchange.fetch_tickers.return_value = {
        "BTC/USDT": Ticker(
            symbol="BTC/USDT", bid=50000.0, ask=50010.0, last=50005.0, timestamp=1700000000000
        )
    }

    exchange.fetch_funding_rate.return_value = FundingRate(
        symbol="BTC/USDT", rate=0.0001, next_funding_time=1700000000000, interval_hours=8
    )

    exchange.fetch_fee_schedule.return_value = FeeSchedule(maker=0.0002, taker=0.0005)

    exchange.fetch_balance.return_value = Balance(total=100000.0, free=80000.0, used=20000.0)

    exchange.fetch_positions.return_value = [
        Position(
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            amount=-0.2,
            entry_price=50000.0,
            mark_price=50100.0,
            unrealized_pnl=-20.0,
            leverage=3,
            margin=3333.0,
            liquidation_price=65000.0,
            notional=10020.0,
        )
    ]

    exchange.create_order.return_value = OrderResult(
        id="order_123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.2,
        price=50010.0,
        filled=0.2,
        cost=10002.0,
        fee=5.0,
        status="closed",
    )

    exchange.fetch_order.return_value = OrderResult(
        id="order_123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.2,
        price=50010.0,
        filled=0.2,
        cost=10002.0,
        fee=5.0,
        status="closed",
    )

    exchange.get_market_info.return_value = {
        "symbol": "BTC/USDT",
        "base": "BTC",
        "quote": "USDT",
        "type": "swap",
        "price_precision": 0.1,
        "amount_precision": 0.001,
        "min_amount": 0.001,
        "min_cost": 10.0,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
    }

    exchange.set_leverage.return_value = None
    exchange.cancel_order.return_value = True

    return exchange


@pytest.fixture
def sample_spot_prices() -> pd.DataFrame:
    """Generate sample spot price data for backtesting."""
    dates = pd.date_range("2024-01-01", "2024-03-01", freq="1h", tz="UTC")
    np.random.seed(42)
    base_price = 45000.0
    returns = np.random.normal(0.0001, 0.005, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 - 0.001),
            "high": prices * (1 + 0.003),
            "low": prices * (1 - 0.003),
            "close": prices,
            "volume": np.random.uniform(100, 1000, len(dates)),
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_perp_prices(sample_spot_prices: pd.DataFrame) -> pd.DataFrame:
    """Perp prices slightly offset from spot."""
    df = sample_spot_prices.copy()
    np.random.seed(43)
    basis = np.random.normal(0.0002, 0.0005, len(df))
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * (1 + basis)
    return df


@pytest.fixture
def sample_funding_rates(sample_spot_prices: pd.DataFrame) -> pd.DataFrame:
    """Generate sample funding rate data — every 8 hours."""
    dates = pd.date_range("2024-01-01", "2024-03-01", freq="8h", tz="UTC")
    np.random.seed(44)
    # Mostly positive funding (longs pay shorts)
    rates = np.random.normal(0.0001, 0.00015, len(dates))
    rates = np.clip(rates, -0.001, 0.003)

    df = pd.DataFrame({"rate": rates}, index=dates)
    return df
