"""Tests for BUY rule evaluation."""

from app.indicators.schemas import CrossSignal
from app.strategy.config import StrategyEngineConfig
from app.strategy.reasons import ReasonCode
from app.strategy.rules import evaluate_buy_conditions
from tests.strategy.conftest import buy_ready_result, make_indicator_result


class TestBuyAllConditionsMet:
    def test_all_conditions_returns_true(self):
        result = buy_ready_result()
        cfg = StrategyEngineConfig()
        should_buy, met, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is True
        assert not failed

    def test_met_contains_all_reason_codes(self):
        result = buy_ready_result()
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_buy_conditions(result, cfg)
        assert ReasonCode.BULLISH_CROSSOVER in met
        assert ReasonCode.PRICE_ABOVE_LONG_EMA in met
        assert ReasonCode.RSI_IN_BUY_RANGE in met
        assert ReasonCode.VOLUME_CONFIRMED in met


class TestBuyNoBullishCrossover:
    def test_no_crossover_fails(self):
        result = buy_ready_result()
        result = make_indicator_result(
            close=result.close,
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.NONE,  # no crossover
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.NO_NEW_CROSSOVER in failed

    def test_bearish_crossover_fails(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BEARISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.NO_NEW_CROSSOVER in failed

    def test_crossover_not_required_passes(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.NONE,
        )
        cfg = StrategyEngineConfig(require_bullish_crossover=False)
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert ReasonCode.NO_NEW_CROSSOVER not in failed


class TestBuyPriceBelowLongEMA:
    def test_price_below_ema_long_fails(self):
        result = make_indicator_result(
            close="28000",  # below ema_long=30000
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.PRICE_BELOW_LONG_EMA in failed

    def test_price_equal_to_ema_long_fails(self):
        result = make_indicator_result(
            close="30000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.PRICE_BELOW_LONG_EMA in failed  # not strictly above

    def test_price_check_skipped_when_not_required(self):
        result = make_indicator_result(
            close="28000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig(require_price_above_long_ema=False)
        _, _, failed = evaluate_buy_conditions(result, cfg)
        assert ReasonCode.PRICE_BELOW_LONG_EMA not in failed


class TestBuyRSIOutOfRange:
    def test_rsi_below_min_fails(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="49",  # below buy_rsi_min=50
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.RSI_OUTSIDE_BUY_RANGE in failed

    def test_rsi_above_max_fails(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="66",  # above buy_rsi_max=65
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.RSI_OUTSIDE_BUY_RANGE in failed

    def test_rsi_at_min_boundary_passes(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="50",  # exactly at min
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_buy_conditions(result, cfg)
        assert ReasonCode.RSI_IN_BUY_RANGE in met

    def test_rsi_at_max_boundary_passes(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="65",  # exactly at max
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_buy_conditions(result, cfg)
        assert ReasonCode.RSI_IN_BUY_RANGE in met


class TestBuyVolumeInsufficient:
    def test_volume_below_minimum_fails(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="0.9",  # below minimum=1.0
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.VOLUME_INSUFFICIENT in failed

    def test_volume_at_minimum_passes(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.0",  # exactly at minimum
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_buy_conditions(result, cfg)
        assert ReasonCode.VOLUME_CONFIRMED in met

    def test_volume_ratio_none_fails(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio=None,
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.VOLUME_INSUFFICIENT in failed


class TestBuyMissingIndicator:
    def test_rsi_none_fails(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi=None,
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        cfg = StrategyEngineConfig()
        should_buy, _, failed = evaluate_buy_conditions(result, cfg)
        assert should_buy is False
        assert ReasonCode.MISSING_INDICATOR in failed
