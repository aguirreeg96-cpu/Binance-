"""Tests for compute_atr."""

from decimal import Decimal

import pytest

from app.indicators.atr import _true_range, compute_atr

D = Decimal


class TestTrueRange:
    def test_first_candle_no_prev_close(self):
        tr = _true_range(D("110"), D("100"), None)
        assert tr == D("10")  # high - low only

    def test_gap_up_expands_tr(self):
        # high=115, low=108, prev_close=105
        # hl=7, hc=|115-105|=10, lc=|108-105|=3 → TR=10
        tr = _true_range(D("115"), D("108"), D("105"))
        assert tr == D("10")

    def test_gap_down_expands_tr(self):
        # high=90, low=80, prev_close=105
        # hl=10, hc=|90-105|=15, lc=|80-105|=25 → TR=25
        tr = _true_range(D("90"), D("80"), D("105"))
        assert tr == D("25")

    def test_inside_bar(self):
        # high=104, low=98, prev_close=100
        # hl=6, hc=4, lc=2 → TR=6
        tr = _true_range(D("104"), D("98"), D("100"))
        assert tr == D("6")


class TestATRBasic:
    def test_manual_period_2(self):
        """Manual computation with period=2.

        candle[0]: H=110, L=100 → TR=10 (no prev close)
        candle[1]: H=115, L=108, prev_C=105 → TR=max(7,10,3)=10
        candle[2]: H=120, L=113, prev_C=112 → TR=max(7,8,1)=8

        ATR[1] = (TR[0]+TR[1])/2 = (10+10)/2 = 10
        ATR[2] = (10*(2-1)+8)/2 = 18/2 = 9
        """
        highs = [D("110"), D("115"), D("120")]
        lows = [D("100"), D("108"), D("113")]
        closes = [D("105"), D("112"), D("117")]

        result = compute_atr(highs, lows, closes, period=2)
        assert result[0] is None
        assert result[1] == D("10")
        assert result[2] == D("9")

    def test_warmup_returns_none(self):
        highs = [D("105")] * 3
        lows = [D("95")] * 3
        closes = [D("100")] * 3
        result = compute_atr(highs, lows, closes, period=3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None

    def test_first_atr_at_index_period_minus_1(self):
        n = 20
        highs = [D("105")] * n
        lows = [D("95")] * n
        closes = [D("100")] * n
        result = compute_atr(highs, lows, closes, period=14)
        assert result[12] is None
        assert result[13] is not None

    def test_output_length_equals_input(self):
        n = 10
        h = [D("105")] * n
        lows = [D("95")] * n
        c = [D("100")] * n
        result = compute_atr(h, lows, c, period=3)
        assert len(result) == n

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_atr([D("1")], [D("1")], [D("1")], period=0)

    def test_insufficient_series_all_none(self):
        h = [D("105"), D("106")]
        lows = [D("95"), D("96")]
        c = [D("100"), D("101")]
        result = compute_atr(h, lows, c, period=5)
        assert all(v is None for v in result)


class TestATRProperties:
    def test_atr_never_negative(self):
        """ATR must always be >= 0."""
        import random

        random.seed(0)
        n = 50
        closes = [D("100")]
        for _ in range(n - 1):
            closes.append(closes[-1] + D(str(random.gauss(0, 2))))
        highs = [c + D("3") for c in closes]
        lows = [c - D("3") for c in closes]
        result = compute_atr(highs, lows, closes, period=14)
        for v in result:
            if v is not None:
                assert v >= D("0"), f"ATR is negative: {v}"

    def test_constant_series_atr_is_constant_after_warmup(self):
        """Flat series → TR=0 always → ATR=0."""
        n = 20
        h = [D("100")] * n
        lows = [D("100")] * n
        c = [D("100")] * n
        result = compute_atr(h, lows, c, period=5)
        for v in result[4:]:
            assert v == D("0")

    def test_gap_increases_atr(self):
        """A large gap between candles should increase ATR."""
        h_normal = [D("105")] * 14
        l_normal = [D("95")] * 14
        c_normal = [D("100")] * 14
        # Compute ATR with no gap (baseline — not compared numerically, just verify it runs)
        compute_atr(h_normal, l_normal, c_normal, period=14)

        # Now add a big gap candle
        h_gap = h_normal + [D("150")]
        l_gap = l_normal + [D("140")]
        c_gap = c_normal + [D("145")]
        r2 = compute_atr(h_gap, l_gap, c_gap, period=14)
        assert r2[14] is not None
        assert r2[13] is not None
        assert r2[14] > r2[13]  # ATR rises after the gap candle
