"""EMA crossover detection between a short and a long EMA series.

Bullish crossover (golden cross):
    previous_short <= previous_long  AND  current_short > current_long

Bearish crossover (death cross):
    previous_short >= previous_long  AND  current_short < current_long

Equality on the previous bar can trigger a crossover if the current bar
breaks clearly in one direction — the definitions above cover this naturally.

Returns NONE when any of the four values (prev/curr for either series) is None.
No repeated signals: each candle is evaluated independently; the caller is
responsible for filtering consecutive identical signals if desired.
"""

from decimal import Decimal

from app.indicators.schemas import CrossSignal


def compute_crossovers(
    short_ema: list[Decimal | None],
    long_ema: list[Decimal | None],
) -> list[CrossSignal]:
    """Return a CrossSignal per candle aligned with the input series."""
    n = len(short_ema)
    results: list[CrossSignal] = [CrossSignal.NONE] * n

    for i in range(1, n):
        prev_s = short_ema[i - 1]
        prev_l = long_ema[i - 1]
        curr_s = short_ema[i]
        curr_l = long_ema[i]

        if prev_s is None or prev_l is None or curr_s is None or curr_l is None:
            continue

        if prev_s <= prev_l and curr_s > curr_l:
            results[i] = CrossSignal.BULLISH
        elif prev_s >= prev_l and curr_s < curr_l:
            results[i] = CrossSignal.BEARISH

    return results
