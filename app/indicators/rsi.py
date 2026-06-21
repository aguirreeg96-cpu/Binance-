"""Relative Strength Index (RSI) with Wilder smoothing.

Warm-up: RSI(period) requires period+1 closes (period changes).
First RSI value is available at index `period` (0-based).

Special cases (documented):
  avg_loss == 0 and avg_gain > 0  → RSI = 100
  avg_gain == 0 and avg_loss > 0  → RSI = 0
  avg_gain == 0 and avg_loss == 0 → RSI = 50  (flat / no-movement market)

RSI is always clamped to [0, 100].
"""

from decimal import Decimal

_ZERO = Decimal("0")
_FIFTY = Decimal("50")
_HUNDRED = Decimal("100")


def compute_rsi(closes: list[Decimal], period: int) -> list[Decimal | None]:
    """Return RSI values aligned with the input.  None during warm-up.

    Wilder smoothing after the initial simple-average seed:
        avg_gain_i = (avg_gain_{i-1} * (N-1) + gain_i) / N
        avg_loss_i = (avg_loss_{i-1} * (N-1) + loss_i) / N
    """
    if period <= 0:
        raise ValueError(f"RSI period must be > 0, got {period}")

    n = len(closes)
    results: list[Decimal | None] = [None] * n

    if n < period + 1:
        return results

    period_d = Decimal(str(period))

    # Seed: simple average of the first `period` gains and losses
    seed_gain = _ZERO
    seed_loss = _ZERO
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > _ZERO:
            seed_gain += change
        elif change < _ZERO:
            seed_loss += -change

    avg_gain = seed_gain / period_d
    avg_loss = seed_loss / period_d
    results[period] = _rsi_value(avg_gain, avg_loss)

    # Wilder smoothing for remaining candles
    for i in range(period + 1, n):
        change = closes[i] - closes[i - 1]
        current_gain = change if change > _ZERO else _ZERO
        current_loss = -change if change < _ZERO else _ZERO

        avg_gain = (avg_gain * (period_d - 1) + current_gain) / period_d
        avg_loss = (avg_loss * (period_d - 1) + current_loss) / period_d
        results[i] = _rsi_value(avg_gain, avg_loss)

    return results


def _rsi_value(avg_gain: Decimal, avg_loss: Decimal) -> Decimal:
    if avg_gain == _ZERO and avg_loss == _ZERO:
        return _FIFTY
    if avg_loss == _ZERO:
        return _HUNDRED
    if avg_gain == _ZERO:
        return _ZERO
    rs = avg_gain / avg_loss
    rsi = _HUNDRED - (_HUNDRED / (_ZERO + 1 + rs))
    # Clamp to guard against any Decimal rounding at extreme RS values
    return max(_ZERO, min(_HUNDRED, rsi))
