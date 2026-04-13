"""Real-time market data feed using exchange WebSocket connections."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from tradingbot.exchanges.base import ExchangeBase, FundingRate, Ticker
from tradingbot.utils.logger import get_logger

log = get_logger(__name__)

Callback = Callable[..., Coroutine[Any, Any, None]]


class MarketDataFeed:
    """Aggregates real-time market data from multiple exchanges."""

    def __init__(self) -> None:
        self._exchanges: dict[str, ExchangeBase] = {}
        self._tickers: dict[str, dict[str, Ticker]] = defaultdict(dict)
        self._funding_rates: dict[str, dict[str, FundingRate]] = defaultdict(dict)
        self._running = False
        self._callbacks: list[Callback] = []
        self._poll_interval: float = 5.0

    def add_exchange(self, exchange: ExchangeBase) -> None:
        self._exchanges[exchange.name] = exchange

    def on_update(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def get_ticker(self, exchange: str, symbol: str) -> Ticker | None:
        return self._tickers.get(exchange, {}).get(symbol)

    def get_funding_rate(self, exchange: str, symbol: str) -> FundingRate | None:
        return self._funding_rates.get(exchange, {}).get(symbol)

    def get_best_bid_ask(self, symbol: str) -> dict[str, dict[str, float]]:
        """Get best bid/ask across all exchanges for a symbol."""
        result: dict[str, dict[str, float]] = {}
        for ex_name, tickers in self._tickers.items():
            if symbol in tickers:
                t = tickers[symbol]
                result[ex_name] = {"bid": t.bid, "ask": t.ask, "last": t.last}
        return result

    async def start(self, symbols: list[str], poll_interval: float = 5.0) -> None:
        """Start polling market data."""
        self._running = True
        self._poll_interval = poll_interval

        log.info("feed_starting", exchanges=list(self._exchanges.keys()), symbols=symbols)

        while self._running:
            tasks = []
            for _ex_name, exchange in self._exchanges.items():
                tasks.append(self._poll_exchange(exchange, symbols))
            await asyncio.gather(*tasks, return_exceptions=True)

            for cb in self._callbacks:
                try:
                    await cb()
                except Exception as e:
                    log.error("feed_callback_error", error=str(e))

            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False

    async def _poll_exchange(self, exchange: ExchangeBase, symbols: list[str]) -> None:
        """Poll tickers and funding rates from one exchange."""
        try:
            tickers = await exchange.fetch_tickers(symbols)
            self._tickers[exchange.name].update(tickers)
        except Exception as e:
            log.warning("ticker_poll_failed", exchange=exchange.name, error=str(e))

        for symbol in symbols:
            try:
                fr = await exchange.fetch_funding_rate(symbol)
                self._funding_rates[exchange.name][symbol] = fr
            except Exception as e:
                log.warning(
                    "funding_poll_failed",
                    exchange=exchange.name,
                    symbol=symbol,
                    error=str(e),
                )

    async def poll_once(self, symbols: list[str]) -> None:
        """Single poll cycle — useful for strategy ticks."""
        for exchange in self._exchanges.values():
            await self._poll_exchange(exchange, symbols)
