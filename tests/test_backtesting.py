"""Tests for the backtesting engine."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from tradingbot.backtesting.engine import BacktestEngine, BacktestResult
from tradingbot.backtesting.simulator import BacktestSimulator
from tradingbot.config.settings import BacktestConfig, FeeConfig, StrategyConfig


class TestBacktestSimulator:
    @pytest.fixture
    def simulator(self, sample_spot_prices, sample_perp_prices, sample_funding_rates):
        return BacktestSimulator(
            spot_prices=sample_spot_prices,
            perp_prices=sample_perp_prices,
            funding_rates=sample_funding_rates,
            initial_capital=100_000.0,
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage_bps=5.0,
        )

    def test_initial_state(self, simulator):
        assert simulator.capital == 100_000.0
        assert simulator.trade_log.empty
        assert simulator.equity_curve.empty

    def test_open_position(self, simulator, sample_spot_prices):
        ts = sample_spot_prices.index[10]
        fill = simulator.open_position("BTC/USDT", 0.5, ts)

        assert fill is not None
        assert fill.symbol == "BTC/USDT"
        assert fill.amount == 0.5
        assert fill.fee > 0
        assert simulator.capital < 100_000.0  # Capital reduced

    def test_open_and_close_position(self, simulator, sample_spot_prices):
        open_ts = sample_spot_prices.index[10]
        close_ts = sample_spot_prices.index[100]

        fill_open = simulator.open_position("BTC/USDT", 0.5, open_ts)
        assert fill_open is not None

        fill_close = simulator.close_position("BTC/USDT", close_ts)
        assert fill_close is not None
        assert len(simulator.trade_log) == 2

    def test_funding_payment_positive(self, simulator, sample_spot_prices, sample_funding_rates):
        """Positive funding = shorts receive."""
        ts = sample_spot_prices.index[10]
        simulator.open_position("BTC/USDT", 0.5, ts)

        # Process funding at a time when rate is positive
        funding_ts = sample_funding_rates.index[5]
        capital_before = simulator.capital
        simulator.process_funding(funding_ts)

        rate = sample_funding_rates.loc[funding_ts, "rate"]
        if rate > 0:
            assert simulator.capital >= capital_before

    def test_insufficient_capital(self, simulator, sample_spot_prices):
        """Should not open position if capital is insufficient."""
        ts = sample_spot_prices.index[10]
        fill = simulator.open_position("BTC/USDT", 1000.0, ts)  # Way too much
        assert fill is None

    def test_equity_recording(self, simulator, sample_spot_prices):
        ts = sample_spot_prices.index[10]
        simulator.record_equity(ts)
        assert not simulator.equity_curve.empty


class TestBacktestEngine:
    @pytest.fixture
    def engine(self, strategy_config, fee_config):
        backtest_config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-03-01",
            initial_capital=100_000.0,
        )
        return BacktestEngine(strategy_config, backtest_config, fee_config)

    def test_run_backtest(self, engine, sample_spot_prices, sample_perp_prices, sample_funding_rates):
        result = engine.run(sample_spot_prices, sample_perp_prices, sample_funding_rates)

        assert isinstance(result, BacktestResult)
        assert not result.equity_curve.empty
        assert result.metrics.total_return != 0 or result.metrics.total_trades == 0

    def test_backtest_summary(self, engine, sample_spot_prices, sample_perp_prices, sample_funding_rates):
        result = engine.run(sample_spot_prices, sample_perp_prices, sample_funding_rates)
        summary = result.print_summary()
        assert "BACKTEST RESULTS" in summary
        assert "Sharpe" in summary

    def test_empty_data(self, engine):
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty_df.index.name = "datetime"
        funding_df = pd.DataFrame(columns=["rate"])
        funding_df.index.name = "datetime"

        result = engine.run(empty_df, empty_df, funding_df)
        assert result.metrics.total_trades == 0
