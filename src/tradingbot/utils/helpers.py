"""Helper utilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from decimal import ROUND_DOWN, Decimal
from typing import Any, TypeVar

T = TypeVar("T")


def round_to_precision(value: float, precision: float) -> float:
    """Round a value down to the exchange's step size / precision."""
    if precision <= 0:
        return value
    d_value = Decimal(str(value))
    d_precision = Decimal(str(precision))
    return float(d_value.quantize(d_precision, rounding=ROUND_DOWN))


def calculate_notional(price: float, amount: float) -> float:
    """Calculate notional value of a position."""
    return abs(price * amount)


def bps_to_decimal(bps: float) -> float:
    """Convert basis points to decimal (e.g., 10 bps -> 0.001)."""
    return bps / 10000.0


def decimal_to_bps(dec: float) -> float:
    """Convert decimal to basis points."""
    return dec * 10000.0


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Retry an async function with exponential backoff."""
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    raise last_exception  # type: ignore[misc]
