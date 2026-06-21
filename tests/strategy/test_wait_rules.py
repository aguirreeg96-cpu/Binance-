"""Tests for WAIT outcomes from the StrategyEngine."""

from app.indicators.schemas import CrossSignal
from app.strategy.engine import StrategyEngine
from app.strategy.reasons import ReasonCode
from app.strategy.schemas import PositionContext, StrategyAction
from tests.strategy.conftest import buy_ready_result, make_indicator_result, small_strategy_config


class TestWaitWarmupIncomplete:
    def test_warmup_incomplete_gives_wait(self):
        result = buy_ready_result()
        result = make_indicator_result(
            close=str(result.close),
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
            warmup_complete=False,
        )
        engine = StrategyEngine(small_strategy_config())
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.WARMUP_INCOMPLETE in decision.reasons

    def test_warmup_incomplete_overrides_all_conditions(self):
        """Even if all other buy conditions are perfect, warmup blocks."""
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
            warmup_complete=False,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT


class TestWaitNoCrossover:
    def test_no_new_crossover_gives_wait(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.NONE,  # no crossover
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.NO_NEW_CROSSOVER in decision.failed_conditions


class TestWaitRSIOutOfRange:
    def test_rsi_too_low_gives_wait(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="30",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.RSI_OUTSIDE_BUY_RANGE in decision.failed_conditions

    def test_rsi_too_high_gives_wait(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="70",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.RSI_OUTSIDE_BUY_RANGE in decision.failed_conditions


class TestWaitVolumeInsufficient:
    def test_low_volume_gives_wait(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="0.5",
            crossover=CrossSignal.BULLISH,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.VOLUME_INSUFFICIENT in decision.failed_conditions


class TestWaitPriceBelowLongEMA:
    def test_price_below_ema_long_gives_wait_when_no_position(self):
        result = make_indicator_result(
            close="28000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.PRICE_BELOW_LONG_EMA in decision.failed_conditions


class TestWaitPositionAlreadyOpen:
    def test_position_open_no_sell_trigger_gives_wait(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.NONE,  # no bearish cross
        )
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.POSITION_ALREADY_OPEN in decision.reasons

    def test_position_open_buy_conditions_met_still_gives_wait(self):
        """BUY is impossible when position is already open."""
        result = buy_ready_result()
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action != StrategyAction.BUY


class TestWaitBearishSignalWithoutPosition:
    def test_bearish_crossover_without_position_gives_wait(self):
        """SELL conditions met but no position → WAIT + NO_POSITION_TO_CLOSE."""
        result = make_indicator_result(
            close="29000",
            ema_long="30000",
            rsi="45",
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,
        )
        pos = PositionContext(has_open_long_position=False)
        cfg = small_strategy_config()
        engine = StrategyEngine(cfg)
        decision = engine.evaluate(result, pos)
        # No position → can't sell → WAIT
        assert decision.action == StrategyAction.WAIT

    def test_allow_sell_without_position_flag(self):
        """With allow_sell_without_open_position=True, SELL may appear without position."""
        from app.strategy.config import StrategyEngineConfig

        result = make_indicator_result(
            close="29000",
            ema_long="30000",
            rsi="45",
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,
        )
        # Default (False): no position → WAIT
        engine = StrategyEngine(StrategyEngineConfig(allow_sell_without_open_position=False))
        decision = engine.evaluate(result, PositionContext(has_open_long_position=False))
        assert decision.action == StrategyAction.WAIT


class TestWaitMissingIndicators:
    def test_missing_ema_long_gives_wait(self):
        result = make_indicator_result(
            close="35000",
            ema_long=None,
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.MISSING_INDICATOR in decision.reasons

    def test_missing_rsi_gives_wait(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi=None,
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.WAIT
