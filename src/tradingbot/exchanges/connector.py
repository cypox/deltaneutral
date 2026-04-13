"""CCXT-based exchange connector implementation."""

from __future__ import annotations

from typing import Any

import ccxt.async_support as ccxt

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
from tradingbot.utils.logger import get_logger

log = get_logger(__name__)


class CCXTConnector(ExchangeBase):
    """Unified exchange connector via ccxt."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        secret: str = "",
        password: str = "",
        sandbox: bool = False,
        rate_limit: bool = True,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.name = exchange_id
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._secret = secret
        self._password = password
        self._sandbox = sandbox
        self._rate_limit = rate_limit
        self._options = options or {}
        self._exchange: ccxt.Exchange | None = None
        self._markets_loaded = False

    async def connect(self) -> None:
        exchange_class = getattr(ccxt, self._exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{self._exchange_id}' not supported by ccxt")

        config: dict[str, Any] = {
            "enableRateLimit": self._rate_limit,
            "options": self._options,
        }
        if self._api_key:
            config["apiKey"] = self._api_key
        if self._secret:
            config["secret"] = self._secret
        if self._password:
            config["password"] = self._password

        self._exchange = exchange_class(config)

        if self._sandbox:
            self._exchange.set_sandbox_mode(True)

        await self._exchange.load_markets()
        self._markets_loaded = True
        log.info("exchange_connected", exchange=self.name, markets=len(self._exchange.markets))

    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            raise RuntimeError(f"Exchange {self.name} not connected. Call connect() first.")
        return self._exchange

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
            log.info("exchange_disconnected", exchange=self.name)

    async def fetch_ticker(self, symbol: str) -> Ticker:
        data = await self.exchange.fetch_ticker(symbol)
        return Ticker(
            symbol=symbol,
            bid=float(data.get("bid", 0) or 0),
            ask=float(data.get("ask", 0) or 0),
            last=float(data.get("last", 0) or 0),
            timestamp=int(data.get("timestamp", 0) or 0),
        )

    async def fetch_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        data = await self.exchange.fetch_tickers(symbols)
        result = {}
        for sym, d in data.items():
            result[sym] = Ticker(
                symbol=sym,
                bid=float(d.get("bid", 0) or 0),
                ask=float(d.get("ask", 0) or 0),
                last=float(d.get("last", 0) or 0),
                timestamp=int(d.get("timestamp", 0) or 0),
            )
        return result

    async def fetch_funding_rate(self, symbol: str) -> FundingRate:
        # ccxt unified: fetchFundingRate
        data = await self.exchange.fetch_funding_rate(symbol)
        return FundingRate(
            symbol=symbol,
            rate=float(data.get("fundingRate", 0) or 0),
            next_funding_time=int(data.get("fundingTimestamp", 0) or 0),
            interval_hours=8,
        )

    async def fetch_fee_schedule(self, symbol: str) -> FeeSchedule:
        market = self.exchange.market(symbol)
        maker = float(market.get("maker", 0) or 0)
        taker = float(market.get("taker", 0) or 0)
        return FeeSchedule(maker=maker, taker=taker)

    async def fetch_balance(self, currency: str = "USDT") -> Balance:
        data = await self.exchange.fetch_balance()
        bal = data.get(currency, {})
        return Balance(
            total=float(bal.get("total", 0) or 0),
            free=float(bal.get("free", 0) or 0),
            used=float(bal.get("used", 0) or 0),
        )

    async def fetch_positions(self, symbol: str | None = None) -> list[Position]:
        symbols_arg = [symbol] if symbol else None
        data = await self.exchange.fetch_positions(symbols_arg)
        positions = []
        for p in data:
            amt = float(p.get("contracts", 0) or 0)
            side_str = p.get("side", "long")
            if side_str == "short":
                amt = -abs(amt)
            positions.append(
                Position(
                    symbol=p.get("symbol", ""),
                    side=PositionSide.SHORT if side_str == "short" else PositionSide.LONG,
                    amount=amt,
                    entry_price=float(p.get("entryPrice", 0) or 0),
                    mark_price=float(p.get("markPrice", 0) or 0),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0) or 0),
                    leverage=int(p.get("leverage", 1) or 1),
                    margin=float(p.get("initialMargin", 0) or 0),
                    liquidation_price=float(p.get("liquidationPrice", 0) or 0),
                    notional=float(p.get("notional", 0) or 0),
                )
            )
        return [p for p in positions if p.amount != 0]

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        log.info(
            "creating_order",
            exchange=self.name,
            symbol=symbol,
            side=side.value,
            type=order_type.value,
            amount=amount,
            price=price,
        )
        data = await self.exchange.create_order(
            symbol=symbol,
            type=order_type.value,
            side=side.value,
            amount=amount,
            price=price,
            params=params or {},
        )
        fee_info = data.get("fee") or {}
        return OrderResult(
            id=str(data["id"]),
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=float(data.get("amount", amount)),
            price=float(data.get("price", 0) or price or 0),
            filled=float(data.get("filled", 0) or 0),
            cost=float(data.get("cost", 0) or 0),
            fee=float(fee_info.get("cost", 0) or 0),
            fee_currency=str(fee_info.get("currency", "USDT") or "USDT"),
            status=str(data.get("status", "open")),
            timestamp=int(data.get("timestamp", 0) or 0),
            info=data.get("info", {}),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        await self.exchange.cancel_order(order_id, symbol)
        return True

    async def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        data = await self.exchange.fetch_order(order_id, symbol)
        fee_info = data.get("fee") or {}
        return OrderResult(
            id=str(data["id"]),
            symbol=data.get("symbol", symbol),
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            amount=float(data.get("amount", 0)),
            price=float(data.get("price", 0) or 0),
            filled=float(data.get("filled", 0) or 0),
            cost=float(data.get("cost", 0) or 0),
            fee=float(fee_info.get("cost", 0) or 0),
            fee_currency=str(fee_info.get("currency", "USDT") or "USDT"),
            status=str(data.get("status", "open")),
            timestamp=int(data.get("timestamp", 0) or 0),
        )

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self.exchange.set_leverage(leverage, symbol)
        log.info("leverage_set", exchange=self.name, symbol=symbol, leverage=leverage)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", since: int | None = None, limit: int = 500
    ) -> list[list[float]]:
        return await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)  # type: ignore[no-any-return]

    async def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int = 100
    ) -> list[FundingRate]:
        data = await self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
        return [
            FundingRate(
                symbol=symbol,
                rate=float(entry.get("fundingRate", 0) or 0),
                next_funding_time=int(entry.get("timestamp", 0) or 0),
                interval_hours=8,
            )
            for entry in data
        ]

    async def get_market_info(self, symbol: str) -> dict[str, Any]:
        market = self.exchange.market(symbol)
        return {
            "symbol": symbol,
            "base": market.get("base"),
            "quote": market.get("quote"),
            "type": market.get("type"),
            "contract_size": market.get("contractSize", 1),
            "price_precision": market.get("precision", {}).get("price"),
            "amount_precision": market.get("precision", {}).get("amount"),
            "min_amount": market.get("limits", {}).get("amount", {}).get("min"),
            "min_cost": market.get("limits", {}).get("cost", {}).get("min"),
            "maker_fee": market.get("maker"),
            "taker_fee": market.get("taker"),
        }
