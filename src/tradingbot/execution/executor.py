"""
Order execution engine for delta-neutral strategies.

Handles:
- Atomic entry/exit of spot + perp legs
- TWAP for large orders
- Retry logic with circuit breaker
- Fee tracking and reconciliation
"""

from __future__ import annotations

import asyncio

from tradingbot.config.settings import ExecutionConfig
from tradingbot.exchanges.base import ExchangeBase, OrderResult, OrderSide, OrderType
from tradingbot.strategy.base import Signal
from tradingbot.strategy.delta_neutral import ActivePosition
from tradingbot.utils.helpers import bps_to_decimal, retry_async
from tradingbot.utils.logger import get_logger

log = get_logger(__name__)


class OrderExecutor:
    """Executes trading signals as exchange orders."""

    def __init__(
        self,
        config: ExecutionConfig,
        exchanges: dict[str, ExchangeBase],
    ) -> None:
        self._config = config
        self._exchanges = exchanges
        self._consecutive_errors = 0

    async def execute_signal(self, signal: Signal) -> ActivePosition | None:
        """Execute a trading signal. Returns ActivePosition on successful entry."""
        if signal.is_entry:
            return await self._execute_entry(signal)
        elif signal.is_exit:
            await self._execute_exit(signal)
            return None
        else:
            log.warning("unknown_signal_action", action=signal.action)
            return None

    # ─── Entry execution ─────────────────────────────────────────────

    async def _execute_entry(self, signal: Signal) -> ActivePosition | None:
        """Execute a delta-neutral entry: buy spot + short perp atomically."""
        spot_ex = self._exchanges.get(signal.spot_exchange)
        perp_ex = self._exchanges.get(signal.perp_exchange)

        if not spot_ex or not perp_ex:
            log.error("exchange_not_found", signal=signal)
            return None

        spot_amount = signal.metadata.get("spot_amount", 0)
        perp_amount = signal.metadata.get("perp_amount", 0)
        spot_ask = signal.metadata.get("spot_ask", 0)
        perp_bid = signal.metadata.get("perp_bid", 0)

        # Set leverage on perp exchange
        try:
            await perp_ex.set_leverage(signal.symbol, 3)  # from config
        except Exception as e:
            log.warning("set_leverage_failed", error=str(e))

        # Determine if we should use TWAP
        if signal.amount_usd > self._config.twap_threshold_usd:
            return await self._twap_entry(signal, spot_ex, perp_ex)

        # Execute both legs
        spot_order = None
        perp_order = None
        total_fees = 0.0

        try:
            # Leg 1: Buy spot
            spot_price = self._limit_price(spot_ask, OrderSide.BUY)
            spot_order = await self._place_order_with_retry(
                exchange=spot_ex,
                symbol=signal.symbol,
                side=OrderSide.BUY,
                amount=spot_amount,
                price=spot_price if self._config.default_order_type == "limit" else None,
            )
            total_fees += spot_order.fee

            # Leg 2: Short perpetual
            perp_price = self._limit_price(perp_bid, OrderSide.SELL)
            perp_order = await self._place_order_with_retry(
                exchange=perp_ex,
                symbol=signal.symbol,
                side=OrderSide.SELL,
                amount=perp_amount,
                price=perp_price if self._config.default_order_type == "limit" else None,
            )
            total_fees += perp_order.fee

            self._consecutive_errors = 0

            log.info(
                "entry_executed",
                symbol=signal.symbol,
                spot_exchange=signal.spot_exchange,
                perp_exchange=signal.perp_exchange,
                spot_filled=spot_order.filled,
                perp_filled=perp_order.filled,
                total_fees=total_fees,
            )

            return ActivePosition(
                db_id=0,  # Set by caller after DB insert
                symbol=signal.symbol,
                spot_exchange=signal.spot_exchange,
                perp_exchange=signal.perp_exchange,
                spot_amount=spot_order.filled,
                perp_amount=perp_order.filled,
                spot_entry_price=spot_order.price or spot_ask,
                perp_entry_price=perp_order.price or perp_bid,
                entry_funding_rate=signal.funding_rate,
                total_fees=total_fees,
            )

        except Exception as e:
            self._consecutive_errors += 1
            log.error("entry_failed", symbol=signal.symbol, error=str(e))
            # Attempt to unwind if one leg succeeded
            await self._unwind_partial(
                signal.symbol, spot_ex, perp_ex, spot_order, perp_order
            )
            return None

    async def _twap_entry(
        self, signal: Signal, spot_ex: ExchangeBase, perp_ex: ExchangeBase
    ) -> ActivePosition | None:
        """Execute entry using TWAP to minimize market impact."""
        spot_amount = signal.metadata.get("spot_amount", 0)
        perp_amount = signal.metadata.get("perp_amount", 0)

        n_slices = self._config.twap_slices
        spot_slice = spot_amount / n_slices
        perp_slice = perp_amount / n_slices

        total_spot_filled = 0.0
        total_perp_filled = 0.0
        total_fees = 0.0
        avg_spot_price = 0.0
        avg_perp_price = 0.0

        for i in range(n_slices):
            try:
                # Refresh prices
                spot_ticker = await spot_ex.fetch_ticker(signal.symbol)
                perp_ticker = await perp_ex.fetch_ticker(signal.symbol)

                spot_order = await self._place_order_with_retry(
                    exchange=spot_ex,
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    amount=spot_slice,
                    price=self._limit_price(spot_ticker.ask, OrderSide.BUY)
                    if self._config.default_order_type == "limit"
                    else None,
                )
                perp_order = await self._place_order_with_retry(
                    exchange=perp_ex,
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    amount=perp_slice,
                    price=self._limit_price(perp_ticker.bid, OrderSide.SELL)
                    if self._config.default_order_type == "limit"
                    else None,
                )

                total_spot_filled += spot_order.filled
                total_perp_filled += perp_order.filled
                total_fees += spot_order.fee + perp_order.fee
                avg_spot_price += (spot_order.price or spot_ticker.ask) * spot_order.filled
                avg_perp_price += (perp_order.price or perp_ticker.bid) * perp_order.filled

                log.info(
                    "twap_slice", slice=i + 1, of=n_slices,
                    spot=spot_order.filled, perp=perp_order.filled,
                )

                if i < n_slices - 1:
                    await asyncio.sleep(self._config.twap_interval)

            except Exception as e:
                log.error("twap_slice_failed", slice=i + 1, error=str(e))
                break

        if total_spot_filled == 0 or total_perp_filled == 0:
            return None

        return ActivePosition(
            db_id=0,
            symbol=signal.symbol,
            spot_exchange=signal.spot_exchange,
            perp_exchange=signal.perp_exchange,
            spot_amount=total_spot_filled,
            perp_amount=total_perp_filled,
            spot_entry_price=avg_spot_price / total_spot_filled if total_spot_filled else 0,
            perp_entry_price=avg_perp_price / total_perp_filled if total_perp_filled else 0,
            entry_funding_rate=signal.funding_rate,
            total_fees=total_fees,
        )

    # ─── Exit execution ──────────────────────────────────────────────

    async def _execute_exit(self, signal: Signal) -> None:
        """Close a delta-neutral position: sell spot + close perp short."""
        spot_ex = self._exchanges.get(signal.spot_exchange)
        perp_ex = self._exchanges.get(signal.perp_exchange)

        if not spot_ex or not perp_ex:
            log.error("exchange_not_found_for_exit", signal=signal)
            return

        spot_amount = signal.metadata.get("spot_amount", 0)
        perp_amount = signal.metadata.get("perp_amount", 0)

        try:
            # Sell spot
            spot_ticker = await spot_ex.fetch_ticker(signal.symbol)
            await self._place_order_with_retry(
                exchange=spot_ex,
                symbol=signal.symbol,
                side=OrderSide.SELL,
                amount=spot_amount,
                price=self._limit_price(spot_ticker.bid, OrderSide.SELL)
                if self._config.default_order_type == "limit"
                else None,
            )

            # Close perp short (buy to cover)
            perp_ticker = await perp_ex.fetch_ticker(signal.symbol)
            await self._place_order_with_retry(
                exchange=perp_ex,
                symbol=signal.symbol,
                side=OrderSide.BUY,
                amount=abs(perp_amount),
                price=self._limit_price(perp_ticker.ask, OrderSide.BUY)
                if self._config.default_order_type == "limit"
                else None,
            )

            log.info(
                "exit_executed",
                symbol=signal.symbol,
                reason=signal.metadata.get("reason", "unknown"),
            )

        except Exception as e:
            log.error("exit_failed", symbol=signal.symbol, error=str(e))

    # ─── Helpers ─────────────────────────────────────────────────────

    async def _place_order_with_retry(
        self,
        exchange: ExchangeBase,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float | None = None,
    ) -> OrderResult:
        """Place an order with retry logic and timeout for limit orders."""
        order_type = OrderType.LIMIT if price else OrderType.MARKET

        order = await retry_async(
            exchange.create_order,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            max_retries=self._config.max_retries,
            delay=1.0,
            exceptions=(Exception,),
        )

        # Wait for limit order fill
        if order_type == OrderType.LIMIT and order.status != "closed":
            order = await self._wait_for_fill(exchange, order, symbol)

        return order

    async def _wait_for_fill(
        self, exchange: ExchangeBase, order: OrderResult, symbol: str
    ) -> OrderResult:
        """Wait for a limit order to fill, cancel and use market if timeout."""
        timeout = self._config.limit_order_timeout
        elapsed = 0
        poll_interval = 2

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            updated = await exchange.fetch_order(order.id, symbol)
            if updated.status == "closed":
                return updated

        # Timeout: cancel and use market order
        log.warning("limit_order_timeout", order_id=order.id, symbol=symbol)
        await exchange.cancel_order(order.id, symbol)

        remaining = order.amount - order.filled
        if remaining > 0:
            market_order = await exchange.create_order(
                symbol=symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                amount=remaining,
            )
            # Combine results
            total_filled = order.filled + market_order.filled
            total_cost = order.cost + market_order.cost
            avg_price = total_cost / total_filled if total_filled else 0
            return OrderResult(
                id=market_order.id,
                symbol=symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                amount=order.amount,
                price=avg_price,
                filled=total_filled,
                cost=total_cost,
                fee=order.fee + market_order.fee,
                status="closed",
            )

        return order

    async def _unwind_partial(
        self,
        symbol: str,
        spot_ex: ExchangeBase,
        perp_ex: ExchangeBase,
        spot_order: OrderResult | None,
        perp_order: OrderResult | None,
    ) -> None:
        """Attempt to unwind a partially filled entry."""
        if spot_order and spot_order.filled > 0 and (not perp_order or perp_order.filled == 0):
            log.warning("unwinding_spot", symbol=symbol, amount=spot_order.filled)
            try:
                await spot_ex.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=spot_order.filled,
                )
            except Exception as e:
                log.error("unwind_spot_failed", error=str(e))

        if perp_order and perp_order.filled > 0 and (not spot_order or spot_order.filled == 0):
            log.warning("unwinding_perp", symbol=symbol, amount=perp_order.filled)
            try:
                await perp_ex.create_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=perp_order.filled,
                )
            except Exception as e:
                log.error("unwind_perp_failed", error=str(e))

    def _limit_price(self, reference_price: float, side: OrderSide) -> float:
        """Calculate limit order price with offset from reference."""
        offset = bps_to_decimal(self._config.limit_offset_bps)
        if side == OrderSide.BUY:
            return reference_price * (1 + offset)
        return reference_price * (1 - offset)

    @property
    def is_circuit_broken(self) -> bool:
        return self._consecutive_errors >= 5
