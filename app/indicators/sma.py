"""Simple Moving Average (SMA) — O(n) sliding window."""

from collections import deque
from decimal import Decimal


def compute_sma(values: list[Decimal], period: int) -> list[Decimal | None]:
    """Return a list of SMA values aligned with the input, None during warm-up.

    O(n): maintains a running sum and a deque window; never re-sums the
    entire window for each new value.

    Period 1: every value equals itself (no warm-up).
    Missing/NaN values are not substituted with zero.
    """
    if period <= 0:
        raise ValueError(f"SMA period must be > 0, got {period}")

    results: list[Decimal | None] = []
    window: deque[Decimal] = deque()
    running_sum = Decimal("0")
    period_d = Decimal(str(period))

    for v in values:
        window.append(v)
        running_sum += v

        if len(window) > period:
            running_sum -= window.popleft()

        if len(window) == period:
            results.append(running_sum / period_d)
        else:
            results.append(None)

    return results
