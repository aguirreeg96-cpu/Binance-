"""Tests for validate_candle_series."""

import pytest

from app.indicators.exceptions import (
    DuplicateCandleError,
    EmptyCandleSeriesError,
    InvalidOHLCError,
    MixedIntervalError,
    MixedSymbolError,
    OpenCandleError,
    UnsortedCandlesError,
)
from app.indicators.validation import validate_candle_series
from tests.indicators.conftest import make_candle


class TestEmptySeries:
    def test_empty_list_raises(self):
        with pytest.raises(EmptyCandleSeriesError):
            validate_candle_series([])


class TestUnsorted:
    def test_decreasing_open_time_raises(self):
        c1 = make_candle(idx=1)
        c2 = make_candle(idx=0)  # earlier time — wrong order
        with pytest.raises(UnsortedCandlesError):
            validate_candle_series([c1, c2])

    def test_equal_open_time_raises(self):
        c1 = make_candle(idx=0)
        c2 = make_candle(idx=0)  # same time — duplicate
        with pytest.raises((UnsortedCandlesError, DuplicateCandleError)):
            validate_candle_series([c1, c2])


class TestDuplicates:
    def test_same_open_time_raises(self):
        c1 = make_candle(idx=5)
        c2 = make_candle(idx=5)
        with pytest.raises(DuplicateCandleError):
            validate_candle_series([c1, c2])


class TestMixedSeries:
    def test_mixed_symbols_raises(self):
        c1 = make_candle(idx=0, symbol="BTCUSDT")
        c2 = make_candle(idx=1, symbol="ETHUSDT")
        with pytest.raises(MixedSymbolError):
            validate_candle_series([c1, c2])

    def test_mixed_intervals_raises(self):
        c1 = make_candle(idx=0, interval="1h")
        c2 = make_candle(idx=1, interval="15m")
        with pytest.raises(MixedIntervalError):
            validate_candle_series([c1, c2])


class TestOpenCandle:
    def test_unclosed_candle_raises(self):
        c = make_candle(idx=0, is_closed=False)
        with pytest.raises(OpenCandleError):
            validate_candle_series([c])


class TestOHLC:
    def test_negative_open_raises(self):
        c = make_candle(idx=0, open_="-1", high="5", low="-2", close="2")
        with pytest.raises(InvalidOHLCError, match="negative"):
            validate_candle_series([c])

    def test_high_less_than_open_raises(self):
        c = make_candle(idx=0, open_="100", high="90", low="85", close="95")
        with pytest.raises(InvalidOHLCError, match="high"):
            validate_candle_series([c])

    def test_high_less_than_close_raises(self):
        c = make_candle(idx=0, open_="95", high="98", low="90", close="99")
        with pytest.raises(InvalidOHLCError, match="high"):
            validate_candle_series([c])

    def test_high_less_than_low_raises(self):
        c = make_candle(idx=0, open_="100", high="95", low="99", close="100")
        with pytest.raises(InvalidOHLCError, match="high"):
            validate_candle_series([c])

    def test_low_greater_than_open_raises(self):
        c = make_candle(idx=0, open_="95", high="110", low="100", close="105")
        with pytest.raises(InvalidOHLCError, match="low"):
            validate_candle_series([c])

    def test_low_greater_than_close_raises(self):
        c = make_candle(idx=0, open_="105", high="110", low="104", close="103")
        with pytest.raises(InvalidOHLCError, match="low"):
            validate_candle_series([c])

    def test_negative_volume_raises(self):
        c = make_candle(idx=0, volume="-100")
        c.volume = __import__("decimal").Decimal("-100")
        with pytest.raises(InvalidOHLCError, match="negative"):
            validate_candle_series([c])


class TestValidSeries:
    def test_single_valid_candle_passes(self):
        c = make_candle(idx=0)
        validate_candle_series([c])  # no exception

    def test_many_valid_candles_pass(self):
        candles = [make_candle(idx=i) for i in range(50)]
        validate_candle_series(candles)  # no exception

    def test_high_equals_low_passes(self):
        """Doji candle: high == low == open == close — still valid."""
        c = make_candle(idx=0, open_="100", high="100", low="100", close="100")
        validate_candle_series([c])

    def test_zero_volume_passes(self):
        c = make_candle(idx=0, volume="0")
        validate_candle_series([c])
