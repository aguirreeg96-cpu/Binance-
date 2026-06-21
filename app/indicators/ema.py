"""Exponential Moving Average (EMA).

Seed: SMA of the first `period` values.
First EMA value available at index period-1 (0-based).

Multiplier: 2 / (period + 1)   — standard EMA, not Wilder smoothing.

Wilder smoothing (used by RSI and ATR) uses 1/period instead;
those modules implement their own smoothing directly.
"""

from decimal import Decimal

from app.indicators.sma import compute_sma


def compute_ema(values: list[Decimal], period: int) -> list[Decimal | None]:
    """Return EMA values aligned with the input.  None during warm-up.

    Seed is the SMA of the first `period` values (index 0 … period-1).
    Each subsequent EMA is:
        EMA_i = (close_i - EMA_{i-1}) * multiplier + EMA_{i-1}
    which equals:
        EMA_i = close_i * multiplier + EMA_{i-1} * (1 - multiplier)

    A constant series produces a constant EMA after the seed.
    Period 1: multiplier = 1.0, so EMA_i = close_i for every candle.
    """
    if period <= 0:
        raise ValueError(f"EMA period must be > 0, got {period}")

    n = len(values)
    results: list[Decimal | None] = [None] * n

    if n < period:
        return results

    multiplier = Decimal("2") / Decimal(str(period + 1))

    # Seed from the SMA of the first `period` values
    sma_list = compute_sma(values[:period], period)
    seed = sma_list[-1]  # always a Decimal here (len == period)
    assert seed is not None  # guaranteed: window == period
    ema: Decimal = seed
    results[period - 1] = ema

    for i in range(period, n):
        ema = (values[i] - ema) * multiplier + ema
        results[i] = ema

    return results
