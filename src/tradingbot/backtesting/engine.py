"""
Backtesting engine — orchestrates historical simulation of the
delta-neutral funding rate arbitrage strategy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingbot.backtesting.simulator import BacktestSimulator
from tradingbot.config.settings import BacktestConfig, FeeConfig, StrategyConfig
from tradingbot.utils.logger import get_logger
from tradingbot.utils.metrics import PerformanceMetrics, compute_metrics, funding_rate_to_apr

log = get_logger(__name__)


class BacktestEngine:
    """
    Runs a delta-neutral funding rate arbitrage backtest.

    Takes historical spot prices, perp prices, and funding rates,
    then simulates the strategy's entry/exit logic with realistic
    fee and slippage modeling.
    """

    def __init__(
        self,
        strategy_config: StrategyConfig,
        backtest_config: BacktestConfig,
        fee_config: FeeConfig,
    ) -> None:
        self._strategy_config = strategy_config
        self._backtest_config = backtest_config
        self._fee_config = fee_config

    def run(
        self,
        spot_prices: pd.DataFrame,
        perp_prices: pd.DataFrame,
        funding_rates: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            spot_prices: DataFrame with 'close' column, DatetimeIndex
            perp_prices: DataFrame with 'close' column, DatetimeIndex
            funding_rates: DataFrame with 'rate' column, DatetimeIndex
        """
        sim = BacktestSimulator(
            spot_prices=spot_prices,
            perp_prices=perp_prices,
            funding_rates=funding_rates,
            initial_capital=self._backtest_config.initial_capital,
            slippage_bps=self._fee_config.slippage_bps,
            funding_interval_hours=self._backtest_config.funding_interval_hours,
        )

        # Iterate over price timestamps
        funding_interval = pd.Timedelta(hours=self._backtest_config.funding_interval_hours)
        last_funding_time: pd.Timestamp | None = None
        position_open = False

        for timestamp in spot_prices.index:
            # Process funding at intervals
            if last_funding_time is None or (timestamp - last_funding_time) >= funding_interval:
                sim.process_funding(timestamp)
                last_funding_time = timestamp

                # Check if we should exit based on current funding rate
                if position_open:
                    rate = sim.get_funding_rate(timestamp)
                    if rate is not None:
                        apr = funding_rate_to_apr(rate)
                        if apr < self._strategy_config.exit_funding_rate_apr:
                            symbol = self._strategy_config.symbols[0] if self._strategy_config.symbols else "BTC/USDT"
                            sim.close_position(symbol, timestamp)
                            position_open = False

            # Check for entry opportunity
            if not position_open:
                rate = sim.get_funding_rate(timestamp)
                if rate is not None:
                    apr = funding_rate_to_apr(rate)
                    if apr >= self._strategy_config.min_funding_rate_apr:
                        spot_price = sim.get_spot_price(timestamp)
                        if spot_price and spot_price > 0:
                            amount_usd = min(
                                self._strategy_config.max_position_usd,
                                sim.capital * 0.5,  # Use at most 50% of capital
                            )
                            amount = amount_usd / spot_price
                            if amount_usd >= self._strategy_config.min_position_usd:
                                fill = sim.open_position("BTC/USDT", amount, timestamp)
                                if fill:
                                    position_open = True

            # Check exit conditions for open positions
            if position_open:
                rate = sim.get_funding_rate(timestamp)
                if rate is not None:
                    apr = funding_rate_to_apr(rate)
                    if apr < self._strategy_config.exit_funding_rate_apr:
                        sim.close_position("BTC/USDT", timestamp)
                        position_open = False

            sim.record_equity(timestamp)

        # Close any remaining positions at the end
        if position_open:
            last_ts = spot_prices.index[-1]
            sim.close_position("BTC/USDT", last_ts)

        # Compute metrics
        equity_curve = sim.equity_curve
        trade_log = sim.trade_log
        metrics = compute_metrics(equity_curve, trade_log)

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trade_log=trade_log,
            funding_log=sim.funding_log,
            config={
                "strategy": self._strategy_config.model_dump(),
                "backtest": self._backtest_config.model_dump(),
                "fees": self._fee_config.model_dump(),
            },
        )


class BacktestResult:
    """Container for backtest output."""

    def __init__(
        self,
        metrics: PerformanceMetrics,
        equity_curve: pd.Series,
        trade_log: pd.DataFrame,
        funding_log: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        self.metrics = metrics
        self.equity_curve = equity_curve
        self.trade_log = trade_log
        self.funding_log = funding_log
        self.config = config

    def print_summary(self) -> str:
        """Generate a text summary of backt results."""
        m = self.metrics.summary()
        lines = [
            "=" * 60,
            "  BACKTEST RESULTS — Delta-Neutral Funding Arbitrage",
            "=" * 60,
            f"  Total Return:        {m['total_return_pct']:>10.2f}%",
            f"  Annualized Return:   {m['annualized_return_pct']:>10.2f}%",
            f"  Sharpe Ratio:        {m['sharpe_ratio']:>10.4f}",
            f"  Sortino Ratio:       {m['sortino_ratio']:>10.4f}",
            f"  Max Drawdown:        {m['max_drawdown_pct']:>10.2f}%",
            f"  Win Rate:            {m['win_rate_pct']:>10.2f}%",
            f"  Total Trades:        {m['total_trades']:>10d}",
            f"  Funding Collected:   ${m['total_funding_collected']:>10.2f}",
            f"  Fees Paid:           ${m['total_fees_paid']:>10.2f}",
            f"  Net PnL:             ${m['net_pnl']:>10.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)
