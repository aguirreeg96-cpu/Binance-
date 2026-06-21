"""Average True Range (ATR) with Wilder smoothing.

True Range for candle i:
    TR_i = max(high_i - low_i,
               |high_i - close_{i-1}|,
               |low_i  - close_{i-1}|)

For the first candle (no previous close):
    TR_0 = high_0 - low_0

Initial ATR: simple average of the first `period` True Ranges.
    First ATR available at index period-1 (0-based).

Subsequent ATR (Wilder smoothing):
    ATR_i = (ATR_{i-1} * (N-1) + TR_i) / N

ATR is always >= 0.
"""

from decimal import Decimal

_ZERO = Decimal("0")


def compute_atr(
    highs: list[Decimal],
    lows: list[Decimal],
    closes: list[Decimal],
    period: int,
) -> list[Decimal | None]:
    """Return ATR values aligned with the input.  None during warm-up."""
    if period <= 0:
        raise ValueError(f"ATR period must be > 0, got {period}")

    n = len(highs)
    results: list[Decimal | None] = [None] * n

    if n < period:
        return results

    period_d = Decimal(str(period))

    # Compute all True Ranges
    trs: list[Decimal] = []
    for i in range(n):
        prev_close = closes[i - 1] if i > 0 else None
        trs.append(_true_range(highs[i], lows[i], prev_close))

    # Seed: simple average of first `period` TRs
    seed = _ZERO
    for tr in trs[:period]:
        seed += tr
    atr: Decimal = seed / period_d
    results[period - 1] = atr

    # Wilder smoothing
    for i in range(period, n):
        atr = (atr * (period_d - 1) + trs[i]) / period_d
        results[i] = atr

    return results


def _true_range(high: Decimal, low: Decimal, prev_close: Decimal | None) -> Decimal:
    hl = high - low
    if prev_close is None:
        return hl
    hc = abs(high - prev_close)
    lc = abs(low - prev_close)
    return max(hl, hc, lc)
