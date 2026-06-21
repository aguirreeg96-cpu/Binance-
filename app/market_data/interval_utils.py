"""Binance kline interval helpers."""

from app.market_data.exceptions import InvalidIntervalError

# Canonical mapping: interval string → milliseconds
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,  # approximation (30 days)
}

VALID_INTERVALS: frozenset[str] = frozenset(INTERVAL_MS)


def validate_interval(interval: str) -> str:
    """Return interval unchanged if valid, else raise InvalidIntervalError."""
    if interval not in VALID_INTERVALS:
        raise InvalidIntervalError(interval)
    return interval


def interval_to_ms(interval: str) -> int:
    """Convert a validated interval string to milliseconds."""
    try:
        return INTERVAL_MS[interval]
    except KeyError as exc:
        raise InvalidIntervalError(interval) from exc
