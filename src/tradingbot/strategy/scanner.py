"""
Pair Scanner & Profitability Estimator

Scans all available perpetual futures across configured exchanges to find
the best delta-neutral funding arbitrage opportunities, ranked by net APR.

Also estimates realistic daily/monthly returns for a given capital amount,
accounting for fees, slippage, and position constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradingbot.exchanges.base import ExchangeBase, FeeSchedule, FundingRate, Ticker
from tradingbot.utils.helpers import bps_to_decimal
from tradingbot.utils.logger import get_logger
from tradingbot.utils.metrics import funding_rate_to_apr

log = get_logger(__name__)


# ─── Top perpetual pairs to scan (high-volume, available on most exchanges) ──

DEFAULT_SCAN_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "UNI/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT",
    "APT/USDT", "SUI/USDT", "FIL/USDT", "LTC/USDT", "ATOM/USDT",
    "TIA/USDT", "SEI/USDT", "INJ/USDT", "FET/USDT", "PEPE/USDT",
    "WIF/USDT", "RENDER/USDT", "STX/USDT", "IMX/USDT", "AAVE/USDT",
]


@dataclass
class PairOpportunity:
    """A scanned funding rate opportunity with profitability estimates."""

    symbol: str
    spot_exchange: str
    perp_exchange: str
    spot_price: float
    perp_price: float
    funding_rate: float         # per-period (e.g., 0.0001 = 0.01%)
    funding_rate_apr: float     # annualized
    spread_pct: float           # spot-perp spread
    spot_taker_fee: float       # e.g., 0.001 = 0.1%
    perp_taker_fee: float
    slippage_pct: float
    round_trip_fees_pct: float  # all 4 legs + slippage
    net_apr: float              # funding APR minus annualized fees
    # Profitability for a given capital
    capital_usd: float = 0.0
    position_usd: float = 0.0
    daily_funding_usd: float = 0.0
    daily_net_usd: float = 0.0
    monthly_net_usd: float = 0.0
    entry_cost_usd: float = 0.0       # one-time cost to enter
    days_to_breakeven: float = 0.0    # days of funding to cover entry fees
    min_hold_days: float = 0.0        # minimum days to be profitable

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spot_exchange": self.spot_exchange,
            "perp_exchange": self.perp_exchange,
            "spot_price": round(self.spot_price, 2),
            "funding_rate_8h": f"{self.funding_rate:.6f}",
            "funding_apr": f"{self.funding_rate_apr:.2%}",
            "spread": f"{self.spread_pct:.4%}",
            "round_trip_fees": f"{self.round_trip_fees_pct:.4%}",
            "net_apr": f"{self.net_apr:.2%}",
            "position_usd": f"${self.position_usd:.2f}",
            "daily_funding": f"${self.daily_funding_usd:.4f}",
            "daily_net": f"${self.daily_net_usd:.4f}",
            "monthly_net": f"${self.monthly_net_usd:.2f}",
            "entry_cost": f"${self.entry_cost_usd:.4f}",
            "days_to_breakeven": f"{self.days_to_breakeven:.1f}",
        }


@dataclass
class ScanResult:
    """Result of a full pair scan."""

    opportunities: list[PairOpportunity] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    exchanges_scanned: list[str] = field(default_factory=list)
    symbols_scanned: int = 0
    capital_usd: float = 0.0

    @property
    def profitable(self) -> list[PairOpportunity]:
        return [o for o in self.opportunities if o.net_apr > 0]

    @property
    def best(self) -> PairOpportunity | None:
        profitable = self.profitable
        return profitable[0] if profitable else None

    def total_daily_estimate(self, max_positions: int = 3) -> float:
        """Estimate total daily return if you spread capital across top N pairs."""
        top_n = self.profitable[:max_positions]
        if not top_n:
            return 0.0
        per_position_capital = self.capital_usd / len(top_n)
        total = 0.0
        for opp in top_n:
            ratio = per_position_capital / opp.position_usd if opp.position_usd > 0 else 0
            total += opp.daily_net_usd * ratio
        return total


class PairScanner:
    """
    Scans exchanges for the best delta-neutral funding arbitrage opportunities.

    Usage:
        scanner = PairScanner(exchanges={"binance": binance_connector, "bybit": bybit_connector})
        result = await scanner.scan(capital_usd=500.0)
        for opp in result.profitable:
            print(f"{opp.symbol}: {opp.net_apr:.2%} APR, ${opp.daily_net_usd:.4f}/day")
    """

    def __init__(
        self,
        exchanges: dict[str, ExchangeBase],
        slippage_bps: float = 5.0,
        leverage: int = 3,
    ) -> None:
        self._exchanges = exchanges
        self._slippage_bps = slippage_bps
        self._leverage = leverage

    async def scan(
        self,
        capital_usd: float = 500.0,
        symbols: list[str] | None = None,
        min_net_apr: float = 0.0,
    ) -> ScanResult:
        """
        Scan all symbol × exchange combinations for funding opportunities.

        Args:
            capital_usd: Your total available capital across all exchanges.
                         With $500, that means ~$250 per exchange.
            symbols: Symbols to scan. Defaults to top 30 perp pairs.
            min_net_apr: Only include pairs with net APR above this threshold.
        """
        scan_symbols = symbols or DEFAULT_SCAN_SYMBOLS
        result = ScanResult(
            exchanges_scanned=list(self._exchanges.keys()),
            symbols_scanned=len(scan_symbols),
            capital_usd=capital_usd,
        )

        # Gather data from all exchanges
        exchange_data: dict[str, dict[str, dict[str, Any]]] = {}
        for ex_name, exchange in self._exchanges.items():
            exchange_data[ex_name] = {}
            for symbol in scan_symbols:
                data = await self._fetch_symbol_data(exchange, ex_name, symbol)
                if data:
                    exchange_data[ex_name][symbol] = data

        # Find cross-exchange opportunities
        exchange_names = list(self._exchanges.keys())
        for symbol in scan_symbols:
            # Try every spot_exchange × perp_exchange combination
            for spot_ex in exchange_names:
                for perp_ex in exchange_names:
                    spot_data = exchange_data.get(spot_ex, {}).get(symbol)
                    perp_data = exchange_data.get(perp_ex, {}).get(symbol)
                    if not spot_data or not perp_data:
                        continue

                    opp = self._evaluate_opportunity(
                        symbol=symbol,
                        spot_exchange=spot_ex,
                        perp_exchange=perp_ex,
                        spot_data=spot_data,
                        perp_data=perp_data,
                        capital_usd=capital_usd,
                    )
                    if opp and opp.net_apr >= min_net_apr:
                        result.opportunities.append(opp)

        # Sort by net APR descending
        result.opportunities.sort(key=lambda o: o.net_apr, reverse=True)

        log.info(
            "scan_complete",
            exchanges=len(exchange_names),
            symbols_scanned=len(scan_symbols),
            opportunities_found=len(result.opportunities),
            profitable=len(result.profitable),
        )

        return result

    async def _fetch_symbol_data(
        self, exchange: ExchangeBase, ex_name: str, symbol: str
    ) -> dict[str, Any] | None:
        """Fetch ticker, funding rate, and fees for a symbol on one exchange."""
        try:
            # Try to fetch perp funding rate — if this works, the exchange has a perp market
            perp_symbol = symbol  # ccxt usually resolves this
            funding = await exchange.fetch_funding_rate(perp_symbol)

            ticker = await exchange.fetch_ticker(symbol)
            fees = await exchange.fetch_fee_schedule(symbol)

            if ticker.bid <= 0 or ticker.ask <= 0:
                return None

            return {
                "ticker": ticker,
                "funding": funding,
                "fees": fees,
            }
        except Exception as e:
            # Symbol not available on this exchange — expected for many pairs
            log.debug("symbol_not_available", exchange=ex_name, symbol=symbol, error=str(e))
            return None

    def _evaluate_opportunity(
        self,
        symbol: str,
        spot_exchange: str,
        perp_exchange: str,
        spot_data: dict[str, Any],
        perp_data: dict[str, Any],
        capital_usd: float,
    ) -> PairOpportunity | None:
        """Evaluate a single spot_exchange × perp_exchange opportunity."""
        funding: FundingRate = perp_data["funding"]
        spot_ticker: Ticker = spot_data["ticker"]
        perp_ticker: Ticker = perp_data["ticker"]
        spot_fees: FeeSchedule = spot_data["fees"]
        perp_fees: FeeSchedule = perp_data["fees"]

        # Only interested in positive funding (longs pay shorts → we collect as short)
        if funding.rate <= 0:
            return None

        funding_apr = funding_rate_to_apr(funding.rate)

        # Spread: cost of entering (buy spot at ask, sell perp at bid)
        if perp_ticker.bid <= 0 or spot_ticker.ask <= 0:
            return None
        spread_pct = (spot_ticker.ask - perp_ticker.bid) / perp_ticker.bid

        # Fee calculation (full round trip: open spot + open perp + close spot + close perp)
        slippage_pct = bps_to_decimal(self._slippage_bps)
        round_trip_fees_pct = (
            spot_fees.taker      # buy spot
            + perp_fees.taker    # sell perp (open short)
            + spot_fees.taker    # sell spot (close)
            + perp_fees.taker    # buy perp (close short)
            + slippage_pct * 4   # slippage on all 4 legs
        )

        # Entry-only fees (what you pay upfront)
        entry_fees_pct = (
            spot_fees.taker + perp_fees.taker + slippage_pct * 2
        )

        # Net APR = funding income - cost of fees amortized over average hold time
        # We assume ~30 day average hold, so annualize the round trip cost
        hold_days = 30
        annualized_fee_cost = round_trip_fees_pct * (365 / hold_days)
        net_apr = funding_apr - annualized_fee_cost

        # ─── Profitability calculation for given capital ─────────────
        # With cross-exchange: you need capital on BOTH exchanges
        # Spot leg: you spend capital to buy the asset
        # Perp leg: you need margin (capital / leverage) to open the short
        # So effective position = capital_per_exchange × leverage / (1 + leverage)
        # Simplified: with $500 total, ~$250 per exchange
        # Spot side: buy $250 worth of BTC
        # Perp side: $250 margin × 3x leverage = $750 capacity, but only need $250 notional
        # Net usable position ≈ min(spot_capital, perp_margin × leverage)

        capital_per_exchange = capital_usd / 2  # split across 2 exchanges
        spot_capital = capital_per_exchange
        perp_margin_capacity = capital_per_exchange * self._leverage
        position_usd = min(spot_capital, perp_margin_capacity)
        # In practice limited by spot side (you buy $250 of BTC, short $250 of perp)
        position_usd = spot_capital

        spot_price = spot_ticker.ask
        position_amount = position_usd / spot_price  # e.g., $250 / $60000 = 0.00417 BTC

        # Daily funding income: position_notional × rate × 3 payments/day
        daily_funding_usd = position_usd * funding.rate * 3

        # Entry cost (one-time)
        entry_cost_usd = position_usd * entry_fees_pct * 2  # both legs

        # Daily net after amortizing exit cost over hold period
        exit_cost_usd = position_usd * entry_fees_pct * 2  # closing both legs
        daily_amortized_exit = exit_cost_usd / hold_days
        daily_net_usd = daily_funding_usd - daily_amortized_exit

        # Break-even: how many days of funding to cover entry fees
        if daily_funding_usd > 0:
            days_to_breakeven = entry_cost_usd / daily_funding_usd
        else:
            days_to_breakeven = float("inf")

        monthly_net_usd = daily_net_usd * 30

        return PairOpportunity(
            symbol=symbol,
            spot_exchange=spot_exchange,
            perp_exchange=perp_exchange,
            spot_price=spot_price,
            perp_price=perp_ticker.bid,
            funding_rate=funding.rate,
            funding_rate_apr=funding_apr,
            spread_pct=spread_pct,
            spot_taker_fee=spot_fees.taker,
            perp_taker_fee=perp_fees.taker,
            slippage_pct=slippage_pct,
            round_trip_fees_pct=round_trip_fees_pct,
            net_apr=net_apr,
            capital_usd=capital_usd,
            position_usd=position_usd,
            daily_funding_usd=daily_funding_usd,
            daily_net_usd=daily_net_usd,
            monthly_net_usd=monthly_net_usd,
            entry_cost_usd=entry_cost_usd,
            days_to_breakeven=days_to_breakeven,
            min_hold_days=days_to_breakeven,
        )


def format_scan_report(result: ScanResult) -> str:
    """Format a scan result into a readable CLI report."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("  PAIR SCANNER — Delta-Neutral Funding Arbitrage Opportunities")
    lines.append("=" * 90)
    lines.append(f"  Capital: ${result.capital_usd:,.2f}  |  "
                 f"Exchanges: {', '.join(result.exchanges_scanned)}  |  "
                 f"Pairs scanned: {result.symbols_scanned}")
    lines.append("-" * 90)

    if not result.profitable:
        lines.append("  No profitable opportunities found at current funding rates.")
        lines.append("  Tip: Funding rates change every 8 hours. Try scanning again later.")
        lines.append("=" * 90)
        return "\n".join(lines)

    # Header
    lines.append(f"  {'#':<3} {'Symbol':<12} {'Spot':<9} {'Perp':<9} "
                 f"{'Fund 8h':>9} {'Net APR':>9} {'$/day':>9} {'$/month':>9} "
                 f"{'Break-even':>10}")
    lines.append("-" * 90)

    for i, opp in enumerate(result.profitable[:20], 1):
        lines.append(
            f"  {i:<3} {opp.symbol:<12} {opp.spot_exchange:<9} {opp.perp_exchange:<9} "
            f"{opp.funding_rate:>9.5f} {opp.net_apr:>8.2%} "
            f"{opp.daily_net_usd:>9.4f} {opp.monthly_net_usd:>9.2f} "
            f"{opp.days_to_breakeven:>8.1f}d"
        )

    lines.append("-" * 90)

    # Summary
    best = result.best
    if best:
        lines.append(f"  Best pair: {best.symbol} ({best.spot_exchange}→{best.perp_exchange})")
        lines.append(f"  Funding rate: {best.funding_rate:.6f} per 8h = {best.funding_rate_apr:.2%} APR")
        lines.append(f"  Net APR (after fees): {best.net_apr:.2%}")
        lines.append("")
        lines.append(f"  With ${result.capital_usd:,.0f} capital:")
        lines.append(f"    Position size:     ${best.position_usd:,.2f} per leg")
        lines.append(f"    Entry cost (fees): ${best.entry_cost_usd:.4f}")
        lines.append(f"    Daily funding:     ${best.daily_funding_usd:.4f}")
        lines.append(f"    Daily net profit:  ${best.daily_net_usd:.4f}")
        lines.append(f"    Monthly estimate:  ${best.monthly_net_usd:.2f}")
        lines.append(f"    Break-even:        {best.days_to_breakeven:.1f} days")

    # Multi-position estimate
    daily_multi = result.total_daily_estimate(max_positions=3)
    if daily_multi > 0:
        lines.append("")
        lines.append(f"  If spread across top 3 pairs: ~${daily_multi:.4f}/day = ~${daily_multi*30:.2f}/month")

    lines.append("")
    lines.append("  ⚠ These are estimates based on CURRENT funding rates.")
    lines.append("  Funding rates change every 8h and can turn negative.")
    lines.append("  Past rates do not guarantee future returns.")
    lines.append("=" * 90)

    return "\n".join(lines)
