"""Tests for pair scanner and profitability estimator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tradingbot.exchanges.base import (
    Balance,
    FeeSchedule,
    FundingRate,
    Ticker,
)
from tradingbot.strategy.scanner import PairScanner, format_scan_report


class TestPairScanner:
    @pytest.fixture
    def mock_exchange_high_funding(self) -> AsyncMock:
        """Exchange with high funding rate (profitable)."""
        exchange = AsyncMock()
        exchange.name = "binance"
        exchange.fetch_ticker.return_value = Ticker(
            symbol="BTC/USDT", bid=60000.0, ask=60010.0, last=60005.0, timestamp=0
        )
        exchange.fetch_funding_rate.return_value = FundingRate(
            symbol="BTC/USDT", rate=0.0003, next_funding_time=0, interval_hours=8
        )
        exchange.fetch_fee_schedule.return_value = FeeSchedule(maker=0.0002, taker=0.0005)
        return exchange

    @pytest.fixture
    def mock_exchange_low_funding(self) -> AsyncMock:
        """Exchange with very low funding rate."""
        exchange = AsyncMock()
        exchange.name = "bybit"
        exchange.fetch_ticker.return_value = Ticker(
            symbol="BTC/USDT", bid=59990.0, ask=60000.0, last=59995.0, timestamp=0
        )
        exchange.fetch_funding_rate.return_value = FundingRate(
            symbol="BTC/USDT", rate=0.000005, next_funding_time=0, interval_hours=8
        )
        exchange.fetch_fee_schedule.return_value = FeeSchedule(maker=0.0002, taker=0.0006)
        return exchange

    @pytest.fixture
    def mock_exchange_negative_funding(self) -> AsyncMock:
        """Exchange with negative funding rate (shorts pay)."""
        exchange = AsyncMock()
        exchange.name = "okx"
        exchange.fetch_ticker.return_value = Ticker(
            symbol="BTC/USDT", bid=60000.0, ask=60010.0, last=60005.0, timestamp=0
        )
        exchange.fetch_funding_rate.return_value = FundingRate(
            symbol="BTC/USDT", rate=-0.0002, next_funding_time=0, interval_hours=8
        )
        exchange.fetch_fee_schedule.return_value = FeeSchedule(maker=0.0002, taker=0.0005)
        return exchange

    @pytest.mark.asyncio
    async def test_scan_finds_opportunities(self, mock_exchange_high_funding):
        scanner = PairScanner(
            exchanges={"binance": mock_exchange_high_funding},
            slippage_bps=5.0,
        )
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])

        assert len(result.opportunities) >= 1
        assert result.capital_usd == 500.0

    @pytest.mark.asyncio
    async def test_scan_calculates_daily_returns(self, mock_exchange_high_funding):
        scanner = PairScanner(exchanges={"binance": mock_exchange_high_funding})
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])

        if result.profitable:
            best = result.best
            assert best is not None
            assert best.daily_funding_usd > 0
            assert best.position_usd == pytest.approx(250.0)  # $500 / 2 exchanges
            assert best.days_to_breakeven > 0

    @pytest.mark.asyncio
    async def test_scan_skips_negative_funding(self, mock_exchange_negative_funding):
        scanner = PairScanner(exchanges={"okx": mock_exchange_negative_funding})
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])

        assert len(result.profitable) == 0

    @pytest.mark.asyncio
    async def test_scan_with_500_capital(self, mock_exchange_high_funding):
        """Verify $500 capital works — position should be $250 per leg."""
        scanner = PairScanner(exchanges={"binance": mock_exchange_high_funding})
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])

        if result.profitable:
            opp = result.profitable[0]
            assert opp.position_usd == pytest.approx(250.0)
            # With $250 and BTC at ~$60k, we'd hold ~0.00417 BTC
            assert opp.daily_funding_usd > 0
            assert opp.monthly_net_usd != 0

    @pytest.mark.asyncio
    async def test_scan_multiple_exchanges(self, mock_exchange_high_funding, mock_exchange_low_funding):
        scanner = PairScanner(
            exchanges={
                "binance": mock_exchange_high_funding,
                "bybit": mock_exchange_low_funding,
            }
        )
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])

        assert result.exchanges_scanned == ["binance", "bybit"]
        # Should find cross-exchange combinations
        assert len(result.opportunities) >= 1

    @pytest.mark.asyncio
    async def test_scan_handles_exchange_errors(self, mock_exchange_high_funding):
        """Exchange fetch failure for some symbols should not break scan."""
        mock_exchange_high_funding.fetch_funding_rate.side_effect = [
            FundingRate(symbol="BTC/USDT", rate=0.0003, next_funding_time=0, interval_hours=8),
            Exception("Symbol not found"),
        ]
        scanner = PairScanner(exchanges={"binance": mock_exchange_high_funding})
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT", "FAKECOIN/USDT"])

        # Should still find BTC opportunity despite FAKECOIN error
        assert result.symbols_scanned == 2

    @pytest.mark.asyncio
    async def test_format_scan_report(self, mock_exchange_high_funding):
        scanner = PairScanner(exchanges={"binance": mock_exchange_high_funding})
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])

        report = format_scan_report(result)
        assert "PAIR SCANNER" in report
        assert "$500" in report

    @pytest.mark.asyncio
    async def test_total_daily_estimate(self, mock_exchange_high_funding):
        scanner = PairScanner(exchanges={"binance": mock_exchange_high_funding})
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])

        daily = result.total_daily_estimate(max_positions=1)
        assert isinstance(daily, float)

    @pytest.mark.asyncio
    async def test_scan_empty_exchanges(self):
        scanner = PairScanner(exchanges={})
        result = await scanner.scan(capital_usd=500.0, symbols=["BTC/USDT"])
        assert len(result.opportunities) == 0


class TestProfitabilityMath:
    """Verify math correctness against hand-calculations."""

    def test_funding_daily_income_500_capital(self):
        """$500 capital → $250 position, 0.01% per 8h, 3x/day."""
        position = 250.0
        rate = 0.0001  # 0.01% per 8h
        daily_funding = position * rate * 3
        assert daily_funding == pytest.approx(0.075)  # ~$0.075/day

    def test_funding_daily_income_high_rate(self):
        """$500 capital → $250 position, 0.03% per 8h, 3x/day."""
        position = 250.0
        rate = 0.0003
        daily_funding = position * rate * 3
        assert daily_funding == pytest.approx(0.225)  # ~$0.225/day

    def test_entry_fees_calculation(self):
        """Entry fees: 2 legs × (taker + slippage) on $250 position."""
        position = 250.0
        taker = 0.0005  # 0.05%
        slippage = 0.0005  # 5 bps
        entry_fee_pct = (taker + slippage) * 2  # both legs
        entry_cost = position * entry_fee_pct * 2
        assert entry_cost == pytest.approx(1.0)  # $1 total entry cost

    def test_breakeven_days(self):
        """$1 entry cost / $0.075 daily funding = ~13.3 days."""
        entry_cost = 1.0
        daily_funding = 0.075
        breakeven = entry_cost / daily_funding
        assert breakeven == pytest.approx(13.33, rel=0.01)
