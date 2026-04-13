"""Factory for creating exchange connector instances."""

from __future__ import annotations

from tradingbot.config.settings import ExchangeConfig
from tradingbot.exchanges.base import ExchangeBase
from tradingbot.exchanges.connector import CCXTConnector


def create_exchange(exchange_id: str, config: ExchangeConfig) -> ExchangeBase:
    """Create an exchange connector from config."""
    return CCXTConnector(
        exchange_id=exchange_id,
        api_key=config.api_key,
        secret=config.secret,
        password=config.password,
        sandbox=config.sandbox,
        rate_limit=config.rate_limit,
        options=config.options,
    )


async def create_and_connect(exchange_id: str, config: ExchangeConfig) -> ExchangeBase:
    """Create and connect an exchange connector."""
    connector = create_exchange(exchange_id, config)
    await connector.connect()
    return connector
