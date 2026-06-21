"""Tests for compute_crossovers."""

from decimal import Decimal

from app.indicators.crossovers import compute_crossovers
from app.indicators.schemas import CrossSignal

D = Decimal
B = CrossSignal.BULLISH
R = CrossSignal.BEARISH
N = CrossSignal.NONE


class TestCrossoverDetection:
    def test_bullish_crossover(self):
        # short crosses above long
        short = [D("9"), D("11")]
        long_ = [D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert result[0] == N
        assert result[1] == B

    def test_bearish_crossover(self):
        # short crosses below long
        short = [D("11"), D("9")]
        long_ = [D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert result[0] == N
        assert result[1] == R

    def test_no_crossover_when_short_stays_above(self):
        short = [D("12"), D("13"), D("14")]
        long_ = [D("10"), D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert all(v == N for v in result)

    def test_no_crossover_when_short_stays_below(self):
        short = [D("8"), D("7"), D("6")]
        long_ = [D("10"), D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert all(v == N for v in result)


class TestCrossoverEquality:
    def test_equality_then_bullish_break(self):
        """previous_short == previous_long → valid bullish trigger if curr_short > curr_long."""
        short = [D("10"), D("11")]
        long_ = [D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert result[1] == B

    def test_equality_then_bearish_break(self):
        """previous_short == previous_long → valid bearish trigger if curr_short < curr_long."""
        short = [D("10"), D("9")]
        long_ = [D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert result[1] == R

    def test_equality_then_equality_no_signal(self):
        """previous_short == previous_long, curr_short == curr_long → NONE."""
        short = [D("10"), D("10")]
        long_ = [D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert result[1] == N


class TestCrossoverNoneHandling:
    def test_none_short_current_gives_none_signal(self):
        short = [D("9"), None]
        long_ = [D("10"), D("10")]
        result = compute_crossovers(short, long_)  # type: ignore[arg-type]
        assert result[1] == N

    def test_none_long_previous_gives_none_signal(self):
        short = [D("9"), D("11")]
        long_ = [None, D("10")]
        result = compute_crossovers(short, long_)  # type: ignore[arg-type]
        assert result[1] == N

    def test_all_none_gives_all_none(self):
        short = [None] * 5
        long_ = [None] * 5
        result = compute_crossovers(short, long_)  # type: ignore[arg-type]
        assert all(v == N for v in result)

    def test_first_element_always_none(self):
        """Index 0 has no previous bar; always NONE."""
        short = [D("15")]
        long_ = [D("10")]
        result = compute_crossovers(short, long_)
        assert result[0] == N


class TestCrossoverSequence:
    def test_bullish_then_bearish(self):
        """Series that crosses up then crosses down."""
        short = [D("9"), D("11"), D("11"), D("9")]
        long_ = [D("10"), D("10"), D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert result[0] == N
        assert result[1] == B
        assert result[2] == N  # short stays above; no new cross
        assert result[3] == R

    def test_no_repeated_bullish_while_above(self):
        """Once short > long, staying there does NOT repeat the BULLISH signal."""
        short = [D("9"), D("11"), D("12"), D("13")]
        long_ = [D("10"), D("10"), D("10"), D("10")]
        result = compute_crossovers(short, long_)
        assert result[1] == B
        assert result[2] == N
        assert result[3] == N

    def test_output_length_equals_input(self):
        n = 10
        short = [D("10")] * n
        long_ = [D("10")] * n
        result = compute_crossovers(short, long_)
        assert len(result) == n
