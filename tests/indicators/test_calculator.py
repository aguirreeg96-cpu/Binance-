"""Tests for IndicatorCalculator.calculate()."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.indicators.calculator import IndicatorCalculator
from app.indicators.exceptions import EmptyCandleSeriesError, OpenCandleError
from app.indicators.schemas import CrossSignal, IndicatorConfig, IndicatorResult
from tests.indicators.conftest import make_candle, make_candles


def _small_config() -> IndicatorConfig:
    """Minimal periods so warmup completes in < 10 candles."""
    return IndicatorConfig(
        sma_short_period=2,
        sma_long_period=3,
        ema_short_period=2,
        ema_medium_period=3,
        ema_long_period=4,
        rsi_period=3,
        atr_period=3,
        volume_period=3,
    )


class TestCalculatorBasic:
    def test_output_length_equals_input(self):
        candles = make_candles(["100"] * 20)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        assert len(results) == 20

    def test_single_candle_returns_one_result(self):
        candles = [make_candle(idx=0)]
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        assert len(results) == 1

    def test_returns_list_of_indicator_result(self):
        candles = make_candles(["100"] * 5)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        assert all(isinstance(r, IndicatorResult) for r in results)

    def test_empty_series_raises(self):
        calc = IndicatorCalculator(_small_config())
        with pytest.raises(EmptyCandleSeriesError):
            calc.calculate([])

    def test_unclosed_candle_raises(self):
        candles = [make_candle(idx=0, is_closed=False)]
        calc = IndicatorCalculator(_small_config())
        with pytest.raises(OpenCandleError):
            calc.calculate(candles)

    def test_default_config_is_used_when_none_given(self):
        calc = IndicatorCalculator()
        assert calc.config == IndicatorConfig()

    def test_custom_config_is_stored(self):
        cfg = _small_config()
        calc = IndicatorCalculator(cfg)
        assert calc.config is cfg


class TestCalculatorFields:
    def test_symbol_interval_propagated(self):
        candles = [make_candle(idx=0, symbol="ETHUSDT", interval="15m")]
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        assert results[0].symbol == "ETHUSDT"
        assert results[0].interval == "15m"

    def test_open_close_time_propagated(self):
        c = make_candle(idx=3)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate([c])
        assert results[0].open_time == c.open_time
        assert results[0].close_time == c.close_time

    def test_close_price_propagated(self):
        c = make_candle(idx=0, open_="12340", high="12350", low="12330", close="12345")
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate([c])
        assert results[0].close == Decimal("12345")

    def test_source_candle_id_propagated(self):
        c = make_candle(idx=0)
        c.id = 42  # type: ignore[assignment]
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate([c])
        assert results[0].source_candle_id == 42

    def test_source_candle_id_none_when_not_persisted(self):
        candles = [make_candle(idx=0)]  # id set to None by make_candle
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        assert results[0].source_candle_id is None

    def test_calculated_at_is_set(self):
        before = datetime.now(UTC)
        candles = make_candles(["100"] * 3)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        after = datetime.now(UTC)
        for r in results:
            assert r.calculated_at is not None
            assert before <= r.calculated_at <= after

    def test_calculated_at_is_utc(self):
        candles = make_candles(["100"] * 3)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        for r in results:
            assert r.calculated_at is not None
            assert r.calculated_at.tzinfo == UTC


class TestCalculatorCrossSignal:
    def test_cross_signal_is_cross_signal_enum(self):
        candles = make_candles(["100"] * 5)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        for r in results:
            assert isinstance(r.ema_short_medium_cross, CrossSignal)

    def test_first_result_cross_signal_is_none(self):
        candles = make_candles(["100"] * 5)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        assert results[0].ema_short_medium_cross == CrossSignal.NONE

    def test_cross_signal_is_none_during_warmup(self):
        candles = make_candles(["100"] * 3)
        calc = IndicatorCalculator(_small_config())
        results = calc.calculate(candles)
        # EMA values are None during warmup → crossover must be NONE
        for r in results:
            assert r.ema_short_medium_cross == CrossSignal.NONE


class TestCalculatorWarmup:
    def test_warmup_complete_false_for_all_when_insufficient(self):
        cfg = _small_config()  # warmup=max(3,4,4,3,3)=4
        candles = make_candles(["100"] * 3)
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        assert all(r.warmup_complete is False for r in results)

    def test_warmup_complete_true_after_enough_candles(self):
        cfg = _small_config()
        candles = make_candles(["100"] * 10)
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        assert results[-1].warmup_complete is True

    def test_warmup_indicators_all_none_before_threshold(self):
        cfg = _small_config()
        candles = make_candles(["100", "101", "102"])
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        first = results[0]
        assert first.sma_short is None
        assert first.sma_long is None
        assert first.ema_long is None
        assert first.rsi is None
        assert first.atr is None

    def test_no_look_ahead(self):
        """Later candles must not change earlier results."""
        cfg = _small_config()
        short_candles = make_candles(["100", "101", "102", "103"])
        long_candles = make_candles(["100", "101", "102", "103", "200"])

        calc = IndicatorCalculator(cfg)
        r_short = calc.calculate(short_candles)
        r_long = calc.calculate(long_candles)

        # First 4 results must be identical regardless of the 5th candle
        for i in range(4):
            assert r_short[i].sma_short == r_long[i].sma_short
            assert r_short[i].rsi == r_long[i].rsi
            assert r_short[i].atr == r_long[i].atr


class TestCalculatorIdempotent:
    def test_same_input_same_output(self):
        cfg = _small_config()
        candles = make_candles([str(100 + i) for i in range(10)])
        calc = IndicatorCalculator(cfg)
        r1 = calc.calculate(candles)
        r2 = calc.calculate(candles)
        for a, b in zip(r1, r2, strict=True):
            assert a.sma_short == b.sma_short
            assert a.rsi == b.rsi
            assert a.atr == b.atr

    def test_reuse_same_calculator(self):
        """Calculator instance is stateless — can be called multiple times."""
        cfg = _small_config()
        calc = IndicatorCalculator(cfg)
        candles = make_candles(["100"] * 10)
        for _ in range(3):
            results = calc.calculate(candles)
            assert len(results) == 10
