"""Performance metrics and calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for backtest/live performance metrics."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    total_funding_collected: float = 0.0
    total_fees_paid: float = 0.0
    net_pnl: float = 0.0
    avg_position_duration_hours: float = 0.0
    daily_returns: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | int]:
        return {
            "total_return_pct": round(self.total_return * 100, 4),
            "annualized_return_pct": round(self.annualized_return * 100, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown * 100, 4),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "total_trades": self.total_trades,
            "total_funding_collected": round(self.total_funding_collected, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "net_pnl": round(self.net_pnl, 2),
        }


def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> PerformanceMetrics:
    """Compute performance metrics from an equity curve and trade log."""
    metrics = PerformanceMetrics()

    if equity_curve.empty:
        return metrics

    returns = equity_curve.pct_change().dropna()
    metrics.daily_returns = returns.tolist()

    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    metrics.total_return = (final - initial) / initial

    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if n_days > 0:
        metrics.annualized_return = (1 + metrics.total_return) ** (365.0 / n_days) - 1

    if len(returns) > 1 and returns.std() > 0:
        metrics.sharpe_ratio = float(np.sqrt(365) * returns.mean() / returns.std())

    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        metrics.sortino_ratio = float(np.sqrt(365) * returns.mean() / downside.std())

    cummax = equity_curve.cummax()
    drawdowns = (equity_curve - cummax) / cummax
    metrics.max_drawdown = float(abs(drawdowns.min()))

    if not trades.empty and "pnl" in trades.columns:
        metrics.total_trades = len(trades)
        metrics.win_rate = float((trades["pnl"] > 0).mean())
        metrics.net_pnl = float(trades["pnl"].sum())

        if "funding_pnl" in trades.columns:
            metrics.total_funding_collected = float(trades["funding_pnl"].sum())
        if "fees" in trades.columns:
            metrics.total_fees_paid = float(trades["fees"].sum())

    return metrics


def funding_rate_to_apr(rate: float, payments_per_day: int = 3) -> float:
    """Convert a single funding rate to annualized percentage rate."""
    return rate * payments_per_day * 365


def apr_to_funding_rate(apr: float, payments_per_day: int = 3) -> float:
    """Convert APR back to a single funding rate."""
    return apr / (payments_per_day * 365)
