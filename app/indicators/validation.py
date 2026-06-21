"""Validation of candle series before indicator calculation."""

from decimal import Decimal

from app.indicators.exceptions import (
    DuplicateCandleError,
    EmptyCandleSeriesError,
    InvalidOHLCError,
    MixedIntervalError,
    MixedSymbolError,
    OpenCandleError,
    UnsortedCandlesError,
)
from app.models.candle import Candle

_ZERO = Decimal("0")


def validate_candle_series(candles: list[Candle]) -> None:
    """Validate ordering, uniqueness, OHLC invariants, and series homogeneity.

    Raises a specific IndicatorError subclass on the first violation found.
    """
    if not candles:
        raise EmptyCandleSeriesError("Candle series is empty.")

    symbol = candles[0].symbol
    interval = candles[0].interval
    seen: set[int] = set()
    prev_time: int | None = None

    for i, c in enumerate(candles):
        if c.symbol != symbol:
            raise MixedSymbolError(f"Candle {i} has symbol {c.symbol!r}, expected {symbol!r}.")
        if c.interval != interval:
            raise MixedIntervalError(
                f"Candle {i} has interval {c.interval!r}, expected {interval!r}."
            )
        if not c.is_closed:
            raise OpenCandleError(f"Candle {i} (open_time={c.open_time}) is not closed.")
        if c.open_time in seen:
            raise DuplicateCandleError(f"Duplicate open_time {c.open_time} at candle index {i}.")
        seen.add(c.open_time)

        if prev_time is not None and c.open_time <= prev_time:
            raise UnsortedCandlesError(
                f"Candle {i}: open_time {c.open_time} <= previous {prev_time}."
            )
        prev_time = c.open_time

        _validate_ohlc(c, i)


def _validate_ohlc(c: Candle, idx: int) -> None:
    for name, val in (
        ("open", c.open),
        ("high", c.high),
        ("low", c.low),
        ("close", c.close),
        ("volume", c.volume),
    ):
        if val < _ZERO:
            raise InvalidOHLCError(f"Candle {idx}: {name}={val} is negative.")

    if c.high < c.open:
        raise InvalidOHLCError(f"Candle {idx}: high={c.high} < open={c.open}.")
    if c.high < c.close:
        raise InvalidOHLCError(f"Candle {idx}: high={c.high} < close={c.close}.")
    if c.high < c.low:
        raise InvalidOHLCError(f"Candle {idx}: high={c.high} < low={c.low}.")
    if c.low > c.open:
        raise InvalidOHLCError(f"Candle {idx}: low={c.low} > open={c.open}.")
    if c.low > c.close:
        raise InvalidOHLCError(f"Candle {idx}: low={c.low} > close={c.close}.")
