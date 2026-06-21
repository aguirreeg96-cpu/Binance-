"""Tests for compute_sma."""

from decimal import Decimal

import pytest

from app.indicators.sma import compute_sma

D = Decimal


class TestSMABasic:
    def test_spec_example(self):
        """Series [1,2,3,4,5] period=3 → [None, None, 2, 3, 4]."""
        result = compute_sma([D("1"), D("2"), D("3"), D("4"), D("5")], period=3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == D("2")
        assert result[3] == D("3")
        assert result[4] == D("4")

    def test_period_1_returns_every_value(self):
        values = [D("10"), D("20"), D("30")]
        result = compute_sma(values, period=1)
        assert result == [D("10"), D("20"), D("30")]

    def test_period_equals_length(self):
        values = [D("2"), D("4"), D("6")]
        result = compute_sma(values, period=3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == D("4")

    def test_period_larger_than_series(self):
        values = [D("1"), D("2")]
        result = compute_sma(values, period=5)
        assert result == [None, None]

    def test_empty_series(self):
        assert compute_sma([], period=3) == []

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_sma([D("1")], period=0)

    def test_period_negative_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_sma([D("1")], period=-1)


class TestSMASliding:
    def test_sliding_window_evicts_oldest(self):
        """Window [10,20,30] → SMA 20; then [20,30,40] → SMA 30."""
        result = compute_sma([D("10"), D("20"), D("30"), D("40")], period=3)
        assert result[2] == D("20")
        assert result[3] == D("30")

    def test_running_sum_is_exact(self):
        """Verify O(n) running sum stays exact across many values."""
        values = [D(str(i)) for i in range(1, 101)]  # 1..100
        result = compute_sma(values, period=10)
        # Window [91..100] → sum = 955, SMA = 95.5
        assert result[-1] == D("955") / D("10")

    def test_output_length_equals_input(self):
        values = [D(str(i)) for i in range(20)]
        result = compute_sma(values, period=5)
        assert len(result) == len(values)


class TestSMAVolume:
    def test_volume_sma(self):
        """SMA can be applied to volume series, not just closes."""
        vols = [D("100"), D("200"), D("300")]
        result = compute_sma(vols, period=2)
        assert result[0] is None
        assert result[1] == D("150")
        assert result[2] == D("250")
