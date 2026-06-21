"""Tests for compute_rsi."""

from decimal import Decimal

import pytest

from app.indicators.rsi import compute_rsi

D = Decimal


class TestRSIWarmup:
    def test_rsi_14_needs_15_closes(self):
        """RSI(14) first value at index 14 (0-based); 15 closes required."""
        closes = [D(str(i)) for i in range(1, 16)]  # 15 values
        result = compute_rsi(closes, period=14)
        assert result[13] is None
        assert result[14] is not None

    def test_insufficient_data_returns_all_none(self):
        closes = [D(str(i)) for i in range(1, 15)]  # only 14 — not enough
        result = compute_rsi(closes, period=14)
        assert all(v is None for v in result)

    def test_output_length_equals_input(self):
        closes = [D(str(i)) for i in range(20)]
        result = compute_rsi(closes, period=14)
        assert len(result) == 20

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_rsi([D("1")] * 20, period=0)


class TestRSISpecialCases:
    def _only_gains(self, n: int = 15) -> list[D]:
        return [D(str(i)) for i in range(1, n + 1)]

    def _only_losses(self, n: int = 15) -> list[D]:
        return [D(str(i)) for i in range(n, 0, -1)]

    def _flat(self, n: int = 15) -> list[D]:
        return [D("50")] * n

    def test_only_gains_gives_rsi_100(self):
        result = compute_rsi(self._only_gains(), period=14)
        assert result[14] == D("100")

    def test_only_losses_gives_rsi_0(self):
        result = compute_rsi(self._only_losses(), period=14)
        assert result[14] == D("0")

    def test_flat_market_gives_rsi_50(self):
        result = compute_rsi(self._flat(), period=14)
        assert result[14] == D("50")

    def test_rsi_always_in_0_100_range(self):
        import random

        random.seed(42)
        closes = [D(str(100 + random.gauss(0, 5))) for _ in range(50)]
        result = compute_rsi(closes, period=14)
        for v in result:
            if v is not None:
                assert D("0") <= v <= D("100"), f"RSI {v} out of range"

    def test_more_gains_than_losses_gives_rsi_above_50(self):
        """Mostly rising series → RSI > 50."""
        closes = [
            D("100"),
            D("101"),
            D("102"),
            D("103"),
            D("102"),
            D("103"),
            D("104"),
            D("105"),
            D("106"),
            D("105"),
            D("106"),
            D("107"),
            D("108"),
            D("109"),
            D("110"),
        ]
        result = compute_rsi(closes, period=14)
        assert result[14] is not None
        assert result[14] > D("50")


class TestRSIWilderSmoothing:
    def test_wilder_smoothing_is_applied_after_seed(self):
        """After the seed, the same all-gain series should keep RSI at 100."""
        closes = [D(str(i)) for i in range(1, 30)]  # long series, all gains
        result = compute_rsi(closes, period=14)
        for v in result[14:]:
            assert v == D("100")

    def test_alternating_gains_losses_produces_intermediate_rsi(self):
        """Alternating +1/-1 should produce RSI near 50."""
        closes = [D("100")]
        for i in range(28):
            closes.append(closes[-1] + (D("1") if i % 2 == 0 else D("-1")))
        result = compute_rsi(closes, period=14)
        for v in result[14:]:
            assert v is not None
            assert D("40") <= v <= D("60"), f"Expected RSI near 50, got {v}"
