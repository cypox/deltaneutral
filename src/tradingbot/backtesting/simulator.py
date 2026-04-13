"""
Simulated exchange for backtesting — replays historical data
and simulates order fills, funding payments, and fee deductions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SimulatedFill:
    symbol: str
    side: str
    amount: float
    price: float
    fee: float
    timestamp: pd.Timestamp


@dataclass
class SimulatedPosition:
    symbol: str
    side: str  # "long" or "short"
    amount: float
    entry_price: float
    unrealized_pnl: float = 0.0
    funding_pnl: float = 0.0
    total_fees: float = 0.0


class BacktestSimulator:
    """
    Simulates exchange behavior for backtesting delta-neutral strategies.

    Takes historical price and funding rate DataFrames and replays them,
    calculating fills, fees, funding payments, and PnL.
    """

    def __init__(
        self,
        spot_prices: pd.DataFrame,
        perp_prices: pd.DataFrame,
        funding_rates: pd.DataFrame,
        initial_capital: float = 100_000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage_bps: float = 5.0,
        funding_interval_hours: int = 8,
    ) -> None:
        self._spot_prices = spot_prices
        self._perp_prices = perp_prices
        self._funding_rates = funding_rates
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._maker_fee = maker_fee
        self._taker_fee = taker_fee
        self._slippage_bps = slippage_bps
        self._funding_interval_hours = funding_interval_hours

        self._positions: dict[str, SimulatedPosition] = {}
        self._spot_holdings: dict[str, float] = {}
        self._trade_log: list[dict[str, Any]] = []
        self._equity_curve: list[dict[str, Any]] = []
        self._funding_log: list[dict[str, Any]] = []

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log)

    @property
    def equity_curve(self) -> pd.Series:
        df = pd.DataFrame(self._equity_curve)
        if df.empty:
            return pd.Series(dtype=float)
        df.set_index("timestamp", inplace=True)
        return df["equity"]

    @property
    def funding_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._funding_log)

    def get_spot_price(self, timestamp: pd.Timestamp) -> float | None:
        """Get spot close price at timestamp."""
        if timestamp in self._spot_prices.index:
            return float(self._spot_prices.loc[timestamp, "close"])
        # Find nearest
        idx = self._spot_prices.index.get_indexer([timestamp], method="ffill")
        if idx[0] >= 0:
            return float(self._spot_prices.iloc[idx[0]]["close"])
        return None

    def get_perp_price(self, timestamp: pd.Timestamp) -> float | None:
        """Get perp close price at timestamp."""
        if timestamp in self._perp_prices.index:
            return float(self._perp_prices.loc[timestamp, "close"])
        idx = self._perp_prices.index.get_indexer([timestamp], method="ffill")
        if idx[0] >= 0:
            return float(self._perp_prices.iloc[idx[0]]["close"])
        return None

    def get_funding_rate(self, timestamp: pd.Timestamp) -> float | None:
        """Get funding rate at timestamp."""
        if timestamp in self._funding_rates.index:
            return float(self._funding_rates.loc[timestamp, "rate"])
        idx = self._funding_rates.index.get_indexer([timestamp], method="ffill")
        if idx[0] >= 0:
            return float(self._funding_rates.iloc[idx[0]]["rate"])
        return None

    def open_position(
        self,
        symbol: str,
        spot_amount: float,
        timestamp: pd.Timestamp,
    ) -> SimulatedFill | None:
        """Open a delta-neutral position: buy spot + short perp."""
        spot_price = self.get_spot_price(timestamp)
        perp_price = self.get_perp_price(timestamp)
        if spot_price is None or perp_price is None:
            return None

        # Apply slippage
        slippage = self._slippage_bps / 10000
        spot_fill_price = spot_price * (1 + slippage)  # Buy higher
        perp_fill_price = perp_price * (1 - slippage)  # Sell lower

        # Calculate fees
        spot_cost = spot_amount * spot_fill_price
        perp_notional = spot_amount * perp_fill_price
        spot_fee = spot_cost * self._taker_fee
        perp_fee = perp_notional * self._taker_fee
        total_fee = spot_fee + perp_fee

        # Check capital
        required = spot_cost + total_fee
        if required > self._capital:
            return None

        self._capital -= spot_cost + spot_fee
        self._spot_holdings[symbol] = self._spot_holdings.get(symbol, 0) + spot_amount

        self._positions[symbol] = SimulatedPosition(
            symbol=symbol,
            side="short",
            amount=spot_amount,
            entry_price=perp_fill_price,
            total_fees=total_fee,
        )

        fill = SimulatedFill(
            symbol=symbol,
            side="open",
            amount=spot_amount,
            price=spot_fill_price,
            fee=total_fee,
            timestamp=timestamp,
        )

        self._trade_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "action": "open",
            "spot_price": spot_fill_price,
            "perp_price": perp_fill_price,
            "amount": spot_amount,
            "fees": total_fee,
            "pnl": 0.0,
            "funding_pnl": 0.0,
        })

        return fill

    def close_position(self, symbol: str, timestamp: pd.Timestamp) -> SimulatedFill | None:
        """Close a delta-neutral position."""
        if symbol not in self._positions or symbol not in self._spot_holdings:
            return None

        pos = self._positions[symbol]
        spot_amount = self._spot_holdings[symbol]

        spot_price = self.get_spot_price(timestamp)
        perp_price = self.get_perp_price(timestamp)
        if spot_price is None or perp_price is None:
            return None

        slippage = self._slippage_bps / 10000
        spot_fill_price = spot_price * (1 - slippage)  # Sell lower
        perp_fill_price = perp_price * (1 + slippage)  # Buy higher (cover)

        # PnL from spot
        spot_proceeds = spot_amount * spot_fill_price
        spot_entry_cost = spot_amount * (self._trade_log[-1]["spot_price"] if self._trade_log else spot_price)

        # PnL from perp (short position)
        perp_pnl = pos.amount * (pos.entry_price - perp_fill_price)

        # Fees
        spot_fee = spot_proceeds * self._taker_fee
        perp_fee = pos.amount * perp_fill_price * self._taker_fee
        close_fees = spot_fee + perp_fee

        total_pnl = (spot_proceeds - spot_entry_cost) + perp_pnl + pos.funding_pnl - close_fees
        self._capital += spot_proceeds - spot_fee

        self._trade_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "action": "close",
            "spot_price": spot_fill_price,
            "perp_price": perp_fill_price,
            "amount": spot_amount,
            "fees": close_fees + pos.total_fees,
            "pnl": total_pnl,
            "funding_pnl": pos.funding_pnl,
        })

        del self._positions[symbol]
        del self._spot_holdings[symbol]

        return SimulatedFill(
            symbol=symbol,
            side="close",
            amount=spot_amount,
            price=spot_fill_price,
            fee=close_fees,
            timestamp=timestamp,
        )

    def process_funding(self, timestamp: pd.Timestamp) -> None:
        """Apply funding payments to all open perp positions."""
        rate = self.get_funding_rate(timestamp)
        if rate is None:
            return

        for symbol, pos in self._positions.items():
            # We're short: if rate > 0, longs pay us; if rate < 0, we pay longs
            payment = pos.amount * pos.entry_price * rate
            if rate > 0:
                pos.funding_pnl += payment
                self._capital += payment
            else:
                pos.funding_pnl -= abs(payment)
                self._capital -= abs(payment)

            self._funding_log.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "rate": rate,
                "payment": payment if rate > 0 else -abs(payment),
                "cumulative": pos.funding_pnl,
            })

    def record_equity(self, timestamp: pd.Timestamp) -> None:
        """Snapshot current equity."""
        equity = self._capital

        # Add unrealized PnL from open positions
        for symbol, pos in self._positions.items():
            perp_price = self.get_perp_price(timestamp)
            spot_price = self.get_spot_price(timestamp)
            if perp_price and spot_price:
                spot_amount = self._spot_holdings.get(symbol, 0)
                spot_value = spot_amount * spot_price
                perp_pnl = pos.amount * (pos.entry_price - perp_price)
                equity += spot_value + perp_pnl + pos.funding_pnl

        self._equity_curve.append({"timestamp": timestamp, "equity": equity})
