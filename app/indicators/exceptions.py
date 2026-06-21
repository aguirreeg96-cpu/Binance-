"""Typed exceptions for the indicator calculation pipeline."""


class IndicatorError(Exception):
    """Base class for all indicator errors."""


class EmptyCandleSeriesError(IndicatorError):
    """Raised when the candle series is empty."""


class InsufficientDataError(IndicatorError):
    """Raised when there are not enough candles to compute any indicator."""

    def __init__(self, needed: int, got: int) -> None:
        super().__init__(f"Need at least {needed} candles, got {got}.")
        self.needed = needed
        self.got = got


class UnsortedCandlesError(IndicatorError):
    """Raised when candles are not in strictly ascending open_time order."""


class DuplicateCandleError(IndicatorError):
    """Raised when two candles share the same open_time."""


class MixedSymbolError(IndicatorError):
    """Raised when candles belong to different symbols."""


class MixedIntervalError(IndicatorError):
    """Raised when candles belong to different intervals."""


class OpenCandleError(IndicatorError):
    """Raised when a non-closed candle is included in the series."""


class InvalidOHLCError(IndicatorError):
    """Raised when OHLC price invariants are violated."""


class InvalidIndicatorPeriodError(IndicatorError):
    """Raised when a period is zero, negative, or logically inconsistent."""
