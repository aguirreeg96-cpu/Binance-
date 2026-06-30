"""Tests that all indicator output is Decimal — no float contamination."""

from decimal import Decimal

from app.indicators.atr import compute_atr
from app.indicators.calculator import IndicatorCalculator
from app.indicators.ema import compute_ema
from app.indicators.rsi import compute_rsi
from app.indicators.schemas import IndicatorConfig
from app.indicators.sma import compute_sma
from app.indicators.volume import compute_volume_indicators
from tests.indicators.conftest import make_candles

D = Decimal


def _assert_no_float(values: list) -> None:
    for v in values:
        if v is not None:
            assert isinstance(v, Decimal), f"Expected Decimal, got {type(v).__name__}: {v!r}"


class TestSMAPrecision:
    def test_output_type_is_decimal(self):
        values = [D("1"), D("2"), D("3"), D("4"), D("5")]
        result = compute_sma(values, period=3)
        _assert_no_float(result)

    def test_exact_decimal_arithmetic(self):
        """1/3 must remain exact Decimal, not lose precision to float."""
        values = [D("1"), D("1"), D("1")]
        result = compute_sma(values, period=3)
        # 1+1+1 / 3 = Decimal("1")
        assert result[2] == D("1")

    def test_non_terminating_division_stays_decimal(self):
        """(1+2+3)/3 = 2 exactly; verify type."""
        values = [D("1"), D("2"), D("3")]
        result = compute_sma(values, period=3)
        assert isinstance(result[2], Decimal)


class TestEMAPrecision:
    def test_output_type_is_decimal(self):
        values = [D(str(i)) for i in range(1, 11)]
        result = compute_ema(values, period=5)
        _assert_no_float(result)

    def test_multiplier_is_decimal(self):
        """EMA multiplier = 2/(N+1). For N=3: 2/4 = 0.5 — stays Decimal."""
        values = [D("10"), D("10"), D("10"), D("10")]
        result = compute_ema(values, period=3)
        # Constant input → EMA = constant
        assert result[2] == D("10")
        assert isinstance(result[3], Decimal)

    def test_no_float_in_long_ema_chain(self):
        values = [D("100") + D(str(i)) for i in range(50)]
        result = compute_ema(values, period=20)
        _assert_no_float(result)


class TestRSIPrecision:
    def test_output_type_is_decimal(self):
        closes = [D(str(i)) for i in range(1, 20)]
        result = compute_rsi(closes, period=14)
        _assert_no_float(result)

    def test_rsi_100_is_exact_decimal(self):
        closes = [D(str(i)) for i in range(1, 16)]
        result = compute_rsi(closes, period=14)
        assert result[14] == D("100")
        assert type(result[14]) is Decimal

    def test_rsi_0_is_exact_decimal(self):
        closes = [D(str(i)) for i in range(15, 0, -1)]
        result = compute_rsi(closes, period=14)
        assert result[14] == D("0")
        assert type(result[14]) is Decimal

    def test_rsi_50_is_exact_decimal(self):
        closes = [D("50")] * 15
        result = compute_rsi(closes, period=14)
        assert result[14] == D("50")
        assert type(result[14]) is Decimal


class TestATRPrecision:
    def test_output_type_is_decimal(self):
        n = 20
        h = [D("105")] * n
        lows = [D("95")] * n
        c = [D("100")] * n
        result = compute_atr(h, lows, c, period=5)
        _assert_no_float(result)

    def test_constant_atr_is_exact_zero(self):
        n = 10
        h = [D("100")] * n
        lows = [D("100")] * n
        c = [D("100")] * n
        result = compute_atr(h, lows, c, period=5)
        for v in result[4:]:
            assert v == D("0")
            assert type(v) is Decimal


class TestVolumePrecision:
    def test_sma_output_type_is_decimal(self):
        vols = [D("100"), D("200"), D("300"), D("400"), D("500")]
        sma, ratios = compute_volume_indicators(vols, period=3)
        _assert_no_float(sma)

    def test_ratio_output_type_is_decimal(self):
        vols = [D("100"), D("200"), D("300"), D("400"), D("500")]
        sma, ratios = compute_volume_indicators(vols, period=3)
        _assert_no_float(ratios)


class TestCalculatorPrecision:
    def _small_config(self) -> IndicatorConfig:
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

    def test_close_field_is_decimal(self):
        candles = make_candles(["100", "101", "102", "103", "104", "105"])
        calc = IndicatorCalculator(self._small_config())
        results = calc.calculate(candles)
        for r in results:
            assert isinstance(r.close, Decimal)

    def test_all_indicator_fields_are_decimal_or_none(self):
        candles = make_candles(["100", "101", "102", "103", "104", "105", "106"])
        calc = IndicatorCalculator(self._small_config())
        results = calc.calculate(candles)
        for r in results:
            for field in (
                "sma_short",
                "sma_long",
                "ema_short",
                "ema_medium",
                "ema_long",
                "rsi",
                "atr",
                "volume_sma",
                "volume_ratio",
            ):
                v = getattr(r, field)
                if v is not None:
                    assert isinstance(v, Decimal), (
                        f"Field {field} has type {type(v).__name__} (expected Decimal)"
                    )

    def test_no_float_in_any_indicator(self):
        """Regression: float must never leak from division operations."""
        candles = make_candles([str(100 + i) for i in range(20)])
        calc = IndicatorCalculator(self._small_config())
        results = calc.calculate(candles)
        for r in results:
            for field in (
                "sma_short",
                "sma_long",
                "ema_short",
                "ema_medium",
                "ema_long",
                "rsi",
                "atr",
                "volume_sma",
                "volume_ratio",
            ):
                v = getattr(r, field)
                assert not isinstance(v, float), f"Float leaked in {field}: {v}"
