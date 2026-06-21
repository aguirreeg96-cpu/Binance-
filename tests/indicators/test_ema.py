"""Tests for compute_ema."""

from decimal import Decimal

import pytest

from app.indicators.ema import compute_ema

D = Decimal


class TestEMABasic:
    def test_seed_is_sma(self):
        """First EMA value at index period-1 equals SMA of first period values."""
        values = [D("10"), D("11"), D("12"), D("13"), D("14")]
        result = compute_ema(values, period=3)
        # Seed = (10+11+12)/3 = 11
        assert result[2] == D("11")

    def test_subsequent_ema_manual(self):
        """Manual step-by-step verification of EMA formula."""
        values = [D("10"), D("11"), D("12"), D("13"), D("14")]
        result = compute_ema(values, period=3)
        # multiplier = 2/(3+1) = 0.5
        # EMA[2] = 11 (seed)
        # EMA[3] = (13 - 11) * 0.5 + 11 = 12
        # EMA[4] = (14 - 12) * 0.5 + 12 = 13
        assert result[2] == D("11")
        assert result[3] == D("12")
        assert result[4] == D("13")

    def test_warmup_returns_none(self):
        values = [D("1"), D("2"), D("3"), D("4"), D("5")]
        result = compute_ema(values, period=3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None

    def test_output_length_equals_input(self):
        values = [D(str(i)) for i in range(10)]
        result = compute_ema(values, period=3)
        assert len(result) == 10

    def test_period_larger_than_series(self):
        result = compute_ema([D("1"), D("2")], period=5)
        assert result == [None, None]

    def test_period_1_returns_every_value(self):
        """Period 1: multiplier = 1.0, EMA_i = close_i."""
        values = [D("5"), D("10"), D("15")]
        result = compute_ema(values, period=1)
        assert result == [D("5"), D("10"), D("15")]

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_ema([D("1")], period=0)

    def test_period_negative_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_ema([D("1")], period=-3)


class TestEMAProperties:
    def test_constant_series_produces_constant_ema(self):
        """A constant price series must produce a constant EMA after warm-up."""
        values = [D("50")] * 10
        result = compute_ema(values, period=3)
        for v in result[2:]:  # after warm-up
            assert v == D("50")

    def test_ema_lags_behind_rising_prices(self):
        """EMA should always be below a steadily rising close after warm-up."""
        values = [D(str(i * 10)) for i in range(1, 11)]  # 10,20,...,100
        result = compute_ema(values, period=3)
        for i in range(3, 10):
            assert result[i] is not None
            assert result[i] < values[i], f"EMA at {i} should lag below close"

    def test_no_float_contamination(self):
        """Decimal inputs must produce Decimal outputs with no float conversion."""
        values = [D("0.00000001")] * 5
        result = compute_ema(values, period=3)
        for v in result[2:]:
            assert isinstance(v, Decimal)
            assert v == D("0.00000001")

    def test_ema_20_first_value_index(self):
        """EMA(20) first value must be at index 19."""
        values = [D(str(i)) for i in range(1, 25)]
        result = compute_ema(values, period=20)
        assert result[18] is None
        assert result[19] is not None

    def test_ema_200_first_value_index(self):
        """EMA(200) first value must be at index 199."""
        values = [D(str(i)) for i in range(1, 205)]
        result = compute_ema(values, period=200)
        assert result[198] is None
        assert result[199] is not None
