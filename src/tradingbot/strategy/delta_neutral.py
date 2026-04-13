"""
Delta-Neutral Funding Rate Arbitrage Strategy

This strategy:
1. Scans funding rates across multiple exchanges for perpetual futures.
2. When funding is positive (longs pay shorts), goes long spot + short perp.
3. Collects funding payments while maintaining delta-neutral exposure.
4. Validates perp position amounts, fee structures, and margin requirements
   before entering any position.
5. Exits when funding drops below a threshold or risk limits are breached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradingbot.config.settings import FeeConfig, RiskConfig, StrategyConfig
from tradingbot.data.feed import MarketDataFeed
from tradingbot.exchanges.base import ExchangeBase, FundingRate, Ticker
from tradingbot.strategy.base import BaseStrategy, Signal
from tradingbot.utils.helpers import bps_to_decimal, calculate_notional, round_to_precision
from tradingbot.utils.logger import get_logger
from tradingbot.utils.metrics import funding_rate_to_apr

log = get_logger(__name__)


@dataclass
class FundingOpportunity:
    """A detected cross-exchange funding rate opportunity."""

    symbol: str
    spot_exchange: str
    perp_exchange: str
    spot_ask: float
    perp_bid: float
    funding_rate: float
    funding_rate_apr: float
    spread_pct: float
    estimated_fees_pct: float
    net_apr: float


@dataclass
class ActivePosition:
    """Tracks an active delta-neutral position."""

    db_id: int
    symbol: str
    spot_exchange: str
    perp_exchange: str
    spot_amount: float
    perp_amount: float
    spot_entry_price: float
    perp_entry_price: float
    entry_funding_rate: float
    total_funding_collected: float = 0.0
    total_fees: float = 0.0


class DeltaNeutralStrategy(BaseStrategy):
    """
    Cross-exchange delta-neutral funding rate arbitrage.

    Entry logic:
        - Funding rate APR on perp exchange exceeds min_funding_rate_apr
        - Spread between spot ask and perp bid is within max_entry_spread_pct
        - Net APR (after fees) is positive and attractive
        - Position size respects max_position_usd and total exposure limits
        - Sufficient free margin on both exchanges

    Position management:
        - Perp amount is validated to match spot amount (delta-neutral check)
        - Funding payments are tracked per position
        - Position rebalancing if amounts drift beyond tolerance

    Exit logic:
        - Funding rate drops below exit threshold
        - Max position loss exceeded
        - Risk limits breached
    """

    name = "delta_neutral_funding"

    def __init__(
        self,
        config: StrategyConfig,
        fee_config: FeeConfig,
        risk_config: RiskConfig,
        feed: MarketDataFeed,
        exchanges: dict[str, ExchangeBase],
    ) -> None:
        self._config = config
        self._fee_config = fee_config
        self._risk_config = risk_config
        self._feed = feed
        self._exchanges = exchanges
        self._active_positions: dict[str, ActivePosition] = {}
        self._total_exposure_usd: float = 0.0

    # ─── Core tick logic ─────────────────────────────────────────────

    async def on_tick(self) -> list[Signal]:
        signals: list[Signal] = []

        # 1. Check if existing positions should be closed
        exit_signals = await self._check_exits()
        signals.extend(exit_signals)

        # 2. Scan for new opportunities
        if self._total_exposure_usd < self._config.max_total_exposure_usd:
            entry_signals = await self._scan_opportunities()
            signals.extend(entry_signals)

        return signals

    async def on_funding(self, symbol: str, exchange: str, rate: float) -> None:
        """Record a funding payment for active positions."""
        for key, pos in self._active_positions.items():
            if pos.symbol == symbol and pos.perp_exchange == exchange:
                # If we're short perp and funding rate is positive, we receive funding
                funding_payment = abs(pos.perp_amount) * pos.perp_entry_price * rate
                if rate > 0:
                    # We're short = we receive
                    pos.total_funding_collected += funding_payment
                else:
                    # We're short but rate is negative = we pay
                    pos.total_funding_collected -= abs(funding_payment)

                log.info(
                    "funding_received",
                    symbol=symbol,
                    exchange=exchange,
                    rate=rate,
                    payment=funding_payment,
                    total_collected=pos.total_funding_collected,
                )

    async def get_status(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "active_positions": len(self._active_positions),
            "total_exposure_usd": self._total_exposure_usd,
            "positions": {
                k: {
                    "symbol": p.symbol,
                    "spot_exchange": p.spot_exchange,
                    "perp_exchange": p.perp_exchange,
                    "spot_amount": p.spot_amount,
                    "perp_amount": p.perp_amount,
                    "funding_collected": p.total_funding_collected,
                    "fees_paid": p.total_fees,
                }
                for k, p in self._active_positions.items()
            },
        }

    # ─── Opportunity scanning ────────────────────────────────────────

    async def _scan_opportunities(self) -> list[Signal]:
        """Scan all symbol × exchange pairs for funding arbitrage."""
        signals: list[Signal] = []

        for symbol in self._config.symbols:
            if symbol in self._active_positions:
                continue

            opportunities = await self._find_opportunities(symbol)
            for opp in opportunities:
                signal = await self._validate_and_size(opp)
                if signal:
                    signals.append(signal)

        return signals

    async def _find_opportunities(self, symbol: str) -> list[FundingOpportunity]:
        """Find cross-exchange funding opportunities for a symbol."""
        opportunities: list[FundingOpportunity] = []

        # Collect funding rates and prices from all exchanges
        exchange_data: dict[str, dict[str, Any]] = {}
        for ex_name, exchange in self._exchanges.items():
            ticker = self._feed.get_ticker(ex_name, symbol)
            funding = self._feed.get_funding_rate(ex_name, symbol)
            if ticker and funding:
                exchange_data[ex_name] = {
                    "ticker": ticker,
                    "funding": funding,
                    "fees": await exchange.fetch_fee_schedule(symbol),
                }

        # Find pairs: spot exchange (buy) × perp exchange (short)
        exchange_names = list(exchange_data.keys())
        for spot_ex in exchange_names:
            for perp_ex in exchange_names:
                funding: FundingRate = exchange_data[perp_ex]["funding"]
                funding_apr = funding_rate_to_apr(funding.rate)

                # Only interested in positive funding (longs pay shorts)
                if funding_apr < self._config.min_funding_rate_apr:
                    continue

                # Skip if funding rate is suspiciously high
                if funding_apr > self._risk_config.max_funding_rate_apr:
                    log.warning(
                        "funding_rate_too_high",
                        symbol=symbol,
                        exchange=perp_ex,
                        apr=funding_apr,
                    )
                    continue

                spot_ticker: Ticker = exchange_data[spot_ex]["ticker"]
                perp_ticker: Ticker = exchange_data[perp_ex]["ticker"]

                # Spread: how much more we pay via spot ask vs perp bid
                if perp_ticker.bid <= 0 or spot_ticker.ask <= 0:
                    continue
                spread_pct = (spot_ticker.ask - perp_ticker.bid) / perp_ticker.bid

                if abs(spread_pct) > self._config.max_entry_spread_pct:
                    continue

                # Estimate total round-trip fees
                spot_fees = exchange_data[spot_ex]["fees"]
                perp_fees = exchange_data[perp_ex]["fees"]
                slippage = bps_to_decimal(self._fee_config.slippage_bps)
                estimated_fees_pct = (
                    spot_fees.taker  # buy spot
                    + perp_fees.taker  # sell perp
                    + spot_fees.taker  # close spot
                    + perp_fees.taker  # close perp
                    + slippage * 4  # slippage on all 4 legs
                )

                net_apr = funding_apr - (estimated_fees_pct * 365 / 8)  # annualize fees

                if net_apr <= 0:
                    continue

                opportunities.append(
                    FundingOpportunity(
                        symbol=symbol,
                        spot_exchange=spot_ex,
                        perp_exchange=perp_ex,
                        spot_ask=spot_ticker.ask,
                        perp_bid=perp_ticker.bid,
                        funding_rate=funding.rate,
                        funding_rate_apr=funding_apr,
                        spread_pct=spread_pct,
                        estimated_fees_pct=estimated_fees_pct,
                        net_apr=net_apr,
                    )
                )

        # Sort by net APR descending
        opportunities.sort(key=lambda o: o.net_apr, reverse=True)
        return opportunities

    async def _validate_and_size(self, opp: FundingOpportunity) -> Signal | None:
        """Validate an opportunity and determine position size."""
        spot_ex = self._exchanges[opp.spot_exchange]
        perp_ex = self._exchanges[opp.perp_exchange]

        # 1. Check free margin on both exchanges
        spot_balance = await spot_ex.fetch_balance()
        perp_balance = await perp_ex.fetch_balance()

        if spot_balance.free < self._risk_config.min_free_margin_usd:
            log.info("insufficient_spot_margin", exchange=opp.spot_exchange, free=spot_balance.free)
            return None

        if perp_balance.free < self._risk_config.min_free_margin_usd:
            log.info("insufficient_perp_margin", exchange=opp.perp_exchange, free=perp_balance.free)
            return None

        # 2. Calculate position size
        remaining_exposure = self._config.max_total_exposure_usd - self._total_exposure_usd
        max_from_spot = spot_balance.free - self._risk_config.min_free_margin_usd
        max_from_perp = (perp_balance.free - self._risk_config.min_free_margin_usd) * self._config.max_leverage

        position_usd = min(
            self._config.max_position_usd,
            remaining_exposure,
            max_from_spot,
            max_from_perp,
        )

        if position_usd < self._config.min_position_usd:
            log.info("position_too_small", position_usd=position_usd)
            return None

        # 3. Validate perp market info (amount precision, min amounts)
        market_info = await perp_ex.get_market_info(opp.symbol)
        amount_precision = market_info.get("amount_precision") or 0.001
        min_amount = market_info.get("min_amount") or 0
        min_cost = market_info.get("min_cost") or 0

        amount = position_usd / opp.spot_ask
        amount = round_to_precision(amount, amount_precision)

        if amount < min_amount:
            log.info("below_min_amount", amount=amount, min_amount=min_amount)
            return None

        notional = calculate_notional(opp.spot_ask, amount)
        if notional < min_cost:
            log.info("below_min_cost", notional=notional, min_cost=min_cost)
            return None

        # 4. Verify the perp amount matches spot amount (delta-neutral check)
        perp_amount = round_to_precision(amount, amount_precision)
        amount_diff_pct = abs(amount - perp_amount) / amount if amount > 0 else 0
        if amount_diff_pct > self._risk_config.position_amount_tolerance_pct:
            log.warning(
                "position_amount_mismatch",
                spot_amount=amount,
                perp_amount=perp_amount,
                diff_pct=amount_diff_pct,
            )
            return None

        log.info(
            "opportunity_found",
            symbol=opp.symbol,
            spot_exchange=opp.spot_exchange,
            perp_exchange=opp.perp_exchange,
            funding_apr=round(opp.funding_rate_apr * 100, 2),
            net_apr=round(opp.net_apr * 100, 2),
            spread_pct=round(opp.spread_pct * 100, 4),
            position_usd=round(position_usd, 2),
        )

        return Signal(
            symbol=opp.symbol,
            action="open_long_spot_short_perp",
            spot_exchange=opp.spot_exchange,
            perp_exchange=opp.perp_exchange,
            amount_usd=position_usd,
            funding_rate=opp.funding_rate,
            funding_rate_apr=opp.funding_rate_apr,
            spread_pct=opp.spread_pct,
            metadata={
                "spot_amount": amount,
                "perp_amount": perp_amount,
                "spot_ask": opp.spot_ask,
                "perp_bid": opp.perp_bid,
                "estimated_fees_pct": opp.estimated_fees_pct,
                "net_apr": opp.net_apr,
            },
        )

    # ─── Exit logic ─────────────────────────────────────────────────

    async def _check_exits(self) -> list[Signal]:
        """Check if any active positions should be closed."""
        signals: list[Signal] = []

        for key, pos in list(self._active_positions.items()):
            should_exit, reason = await self._should_exit(pos)
            if should_exit:
                log.info("exit_signal", symbol=pos.symbol, reason=reason)
                signals.append(
                    Signal(
                        symbol=pos.symbol,
                        action="close_position",
                        spot_exchange=pos.spot_exchange,
                        perp_exchange=pos.perp_exchange,
                        amount_usd=calculate_notional(pos.spot_entry_price, pos.spot_amount),
                        metadata={
                            "reason": reason,
                            "position_key": key,
                            "spot_amount": pos.spot_amount,
                            "perp_amount": pos.perp_amount,
                            "funding_collected": pos.total_funding_collected,
                        },
                    )
                )

        return signals

    async def _should_exit(self, pos: ActivePosition) -> tuple[bool, str]:
        """Determine if a position should be closed."""
        perp_ex = self._exchanges.get(pos.perp_exchange)
        if not perp_ex:
            return True, "perp_exchange_unavailable"

        # Check current funding rate
        funding = self._feed.get_funding_rate(pos.perp_exchange, pos.symbol)
        if funding:
            current_apr = funding_rate_to_apr(funding.rate)
            if current_apr < self._config.exit_funding_rate_apr:
                return True, f"funding_below_threshold: {current_apr:.4f}"

        # Check perp position amount (hasn't been liquidated or partially closed)
        try:
            positions = await perp_ex.fetch_positions(pos.symbol)
            if not positions:
                return True, "perp_position_closed_externally"

            perp_pos = positions[0]
            expected_amount = abs(pos.perp_amount)
            actual_amount = abs(perp_pos.amount)
            if expected_amount > 0:
                drift = abs(actual_amount - expected_amount) / expected_amount
                if drift > self._risk_config.position_amount_tolerance_pct:
                    return True, f"perp_amount_drift: {drift:.4f}"

            # Check unrealized PnL
            notional = calculate_notional(pos.spot_entry_price, pos.spot_amount)
            if notional > 0:
                loss_pct = abs(min(perp_pos.unrealized_pnl, 0)) / notional
                if loss_pct > self._risk_config.max_position_loss_pct:
                    return True, f"max_loss_exceeded: {loss_pct:.4f}"

            # Check margin ratio
            if perp_pos.margin > 0:
                margin_usage = perp_pos.notional / (perp_pos.margin * perp_pos.leverage)
                if margin_usage > self._risk_config.margin_ratio_alert:
                    return True, f"margin_alert: {margin_usage:.4f}"

        except Exception as e:
            log.error("exit_check_error", symbol=pos.symbol, error=str(e))
            return True, f"exit_check_error: {e}"

        return False, ""

    # ─── Position tracking ───────────────────────────────────────────

    def register_position(self, position: ActivePosition) -> None:
        """Register a newly opened position."""
        key = f"{position.symbol}:{position.spot_exchange}:{position.perp_exchange}"
        self._active_positions[key] = position
        self._total_exposure_usd += calculate_notional(
            position.spot_entry_price, position.spot_amount
        )

    def remove_position(self, symbol: str, spot_exchange: str, perp_exchange: str) -> ActivePosition | None:
        """Remove a closed position."""
        key = f"{symbol}:{spot_exchange}:{perp_exchange}"
        pos = self._active_positions.pop(key, None)
        if pos:
            self._total_exposure_usd -= calculate_notional(
                pos.spot_entry_price, pos.spot_amount
            )
        return pos
