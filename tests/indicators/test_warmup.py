"""Tests for warmup behaviour: IndicatorConfig.warmup_candles and warmup_complete."""

import pytest

from app.indicators.calculator import IndicatorCalculator
from app.indicators.schemas import IndicatorConfig
from tests.indicators.conftest import make_candles


class TestWarmupCandlesProperty:
    def test_default_config_warmup_is_ema_long(self):
        """Default config: warmup = max(50, 200, 15, 14, 20) = 200."""
        cfg = IndicatorConfig()
        assert cfg.warmup_candles == 200

    def test_custom_config_warmup_dominated_by_ema_long(self):
        cfg = IndicatorConfig(
            sma_short_period=5,
            sma_long_period=10,
            ema_short_period=5,
            ema_medium_period=10,
            ema_long_period=100,
            rsi_period=14,
            atr_period=14,
            volume_period=20,
        )
        assert cfg.warmup_candles == 100

    def test_custom_config_warmup_dominated_by_rsi(self):
        """If rsi_period+1 is largest, warmup = rsi_period+1."""
        cfg = IndicatorConfig(
            sma_short_period=1,
            sma_long_period=2,
            ema_short_period=1,
            ema_medium_period=2,
            ema_long_period=3,
            rsi_period=50,
            atr_period=5,
            volume_period=5,
        )
        # max(2, 3, 51, 5, 5) = 51
        assert cfg.warmup_candles == 51

    def test_custom_config_warmup_dominated_by_sma_long(self):
        cfg = IndicatorConfig(
            sma_short_period=10,
            sma_long_period=80,
            ema_short_period=5,
            ema_medium_period=10,
            ema_long_period=20,
            rsi_period=14,
            atr_period=14,
            volume_period=10,
        )
        # max(80, 20, 15, 14, 10) = 80
        assert cfg.warmup_candles == 80


class TestIndicatorConfigValidation:
    def test_sma_short_must_be_less_than_sma_long(self):
        with pytest.raises(Exception, match="sma"):
            IndicatorConfig(sma_short_period=50, sma_long_period=20)

    def test_sma_equal_raises(self):
        with pytest.raises(Exception, match="sma"):
            IndicatorConfig(sma_short_period=20, sma_long_period=20)

    def test_ema_must_be_strictly_ascending(self):
        with pytest.raises(Exception, match="EMA"):
            IndicatorConfig(ema_short_period=50, ema_medium_period=20, ema_long_period=200)

    def test_ema_equal_medium_long_raises(self):
        with pytest.raises(Exception, match="EMA"):
            IndicatorConfig(ema_short_period=20, ema_medium_period=50, ema_long_period=50)

    def test_period_zero_raises(self):
        with pytest.raises(ValueError):
            IndicatorConfig(rsi_period=0)

    def test_valid_custom_config_accepted(self):
        cfg = IndicatorConfig(
            sma_short_period=5,
            sma_long_period=10,
            ema_short_period=3,
            ema_medium_period=5,
            ema_long_period=10,
            rsi_period=7,
            atr_period=7,
            volume_period=7,
        )
        assert cfg.sma_short_period == 5

    def test_config_is_frozen(self):
        from pydantic import ValidationError

        cfg = IndicatorConfig()
        with pytest.raises((ValidationError, TypeError)):
            cfg.rsi_period = 99  # type: ignore[misc]


class TestWarmupCompleteFlag:
    def _small_config(self) -> IndicatorConfig:
        """Tiny periods so warmup completes quickly."""
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

    def test_warmup_complete_false_before_threshold(self):
        cfg = self._small_config()
        # warmup = max(3, 4, 4, 3, 3) = 4
        candles = make_candles(["100"] * 3)
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        for r in results:
            assert r.warmup_complete is False

    def test_warmup_complete_true_at_threshold(self):
        cfg = self._small_config()
        # warmup = 4; need ≥4 candles for any to have warmup_complete=True
        candles = make_candles(["100"] * 5)
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        # At least the last result should have warmup_complete=True
        assert results[-1].warmup_complete is True

    def test_all_warmup_incomplete_insufficient_data(self):
        cfg = IndicatorConfig()  # warmup=200
        candles = make_candles(["100"] * 50)
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        assert all(r.warmup_complete is False for r in results)

    def test_warmup_indicators_are_none(self):
        cfg = self._small_config()
        candles = make_candles(["100"] * 3)
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        first = results[0]
        assert first.sma_short is None
        assert first.sma_long is None
        assert first.ema_short is None
        assert first.rsi is None
        assert first.atr is None
        assert first.volume_sma is None

    def test_completed_result_has_all_values(self):
        cfg = self._small_config()
        candles = make_candles(["100", "102", "101", "103", "104", "105"])
        calc = IndicatorCalculator(cfg)
        results = calc.calculate(candles)
        last = results[-1]
        assert last.warmup_complete is True
        assert last.sma_short is not None
        assert last.sma_long is not None
        assert last.ema_short is not None
        assert last.ema_medium is not None
        assert last.ema_long is not None
        assert last.rsi is not None
        assert last.atr is not None
        assert last.volume_sma is not None
