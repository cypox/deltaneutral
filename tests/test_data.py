"""Tests for data pipeline and storage."""

from __future__ import annotations

import pytest
import pandas as pd

from tradingbot.data.feed import MarketDataFeed
from tradingbot.data.storage import DataStore
from tradingbot.exchanges.base import FundingRate, Ticker
from tradingbot.utils.helpers import bps_to_decimal, round_to_precision
from tradingbot.utils.metrics import compute_metrics, funding_rate_to_apr


class TestDataStore:
    @pytest.fixture
    async def store(self, tmp_path):
        db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        store = DataStore(db_url)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_insert_ohlcv(self, store):
        candles = [
            [1700000000000, 50000, 50500, 49500, 50100, 1000],
            [1700003600000, 50100, 50600, 49600, 50200, 1100],
        ]
        inserted = await store.insert_ohlcv_batch("binance", "BTC/USDT", "1h", candles)
        assert inserted == 2

    @pytest.mark.asyncio
    async def test_insert_duplicate_ohlcv(self, store):
        candles = [[1700000000000, 50000, 50500, 49500, 50100, 1000]]
        await store.insert_ohlcv_batch("binance", "BTC/USDT", "1h", candles)
        inserted = await store.insert_ohlcv_batch("binance", "BTC/USDT", "1h", candles)
        assert inserted == 0  # Duplicate skipped

    @pytest.mark.asyncio
    async def test_insert_funding_rates(self, store):
        rates = [
            {"timestamp": 1700000000000, "rate": 0.0001},
            {"timestamp": 1700028800000, "rate": 0.00015},
        ]
        inserted = await store.insert_funding_rates("binance", "BTC/USDT", rates)
        assert inserted == 2

    @pytest.mark.asyncio
    async def test_position_lifecycle(self, store):
        pos_id = await store.insert_position({
            "strategy": "delta_neutral",
            "symbol": "BTC/USDT",
            "spot_exchange": "binance",
            "perp_exchange": "bybit",
            "spot_amount": 0.2,
            "perp_amount": 0.2,
            "spot_entry_price": 50000.0,
            "perp_entry_price": 50010.0,
            "entry_funding_rate": 0.0001,
        })
        assert pos_id > 0

        positions = await store.get_open_positions("delta_neutral")
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC/USDT"

        await store.update_position(pos_id, {"status": "closed"})
        positions = await store.get_open_positions("delta_neutral")
        assert len(positions) == 0


class TestMarketDataFeed:
    def test_add_exchange(self, mock_exchange):
        feed = MarketDataFeed()
        feed.add_exchange(mock_exchange)
        assert mock_exchange.name in feed._exchanges

    def test_get_ticker(self, mock_exchange):
        feed = MarketDataFeed()
        feed.add_exchange(mock_exchange)
        feed._tickers["test_exchange"]["BTC/USDT"] = Ticker(
            symbol="BTC/USDT", bid=50000, ask=50010, last=50005, timestamp=0
        )
        ticker = feed.get_ticker("test_exchange", "BTC/USDT")
        assert ticker is not None
        assert ticker.bid == 50000

    def test_get_best_bid_ask(self, mock_exchange):
        feed = MarketDataFeed()
        feed._tickers["exchange_a"] = {
            "BTC/USDT": Ticker(symbol="BTC/USDT", bid=50000, ask=50010, last=50005, timestamp=0)
        }
        feed._tickers["exchange_b"] = {
            "BTC/USDT": Ticker(symbol="BTC/USDT", bid=50005, ask=50015, last=50010, timestamp=0)
        }
        result = feed.get_best_bid_ask("BTC/USDT")
        assert "exchange_a" in result
        assert "exchange_b" in result


class TestHelpers:
    def test_round_to_precision(self):
        assert round_to_precision(0.12345, 0.001) == 0.123
        assert round_to_precision(100.999, 0.01) == 100.99
        assert round_to_precision(0.5, 0.1) == 0.5

    def test_bps_to_decimal(self):
        assert bps_to_decimal(10) == pytest.approx(0.001)
        assert bps_to_decimal(100) == pytest.approx(0.01)

    def test_funding_rate_to_apr(self):
        # 0.01% per 8h, 3 times per day, 365 days
        assert funding_rate_to_apr(0.0001) == pytest.approx(0.1095, rel=0.01)


class TestMetrics:
    def test_compute_metrics_basic(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="1D")
        equity = pd.Series(range(100_000, 100_100), index=dates, dtype=float)

        trades = pd.DataFrame({
            "pnl": [10, -5, 15, -3, 20],
            "fees": [1, 1, 1, 1, 1],
            "funding_pnl": [2, 2, 2, 2, 2],
        })

        metrics = compute_metrics(equity, trades)
        assert metrics.total_trades == 5
        assert metrics.win_rate == pytest.approx(0.6)
        assert metrics.total_fees_paid == 5
        assert metrics.total_funding_collected == 10

    def test_compute_metrics_empty(self):
        metrics = compute_metrics(pd.Series(dtype=float), pd.DataFrame())
        assert metrics.total_return == 0
        assert metrics.total_trades == 0
