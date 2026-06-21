"""Tests for StrategyEngine.evaluate() and evaluate_series()."""

from datetime import UTC, datetime
from decimal import Decimal

from app.indicators.schemas import CrossSignal
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine
from app.strategy.reasons import ReasonCode
from app.strategy.schemas import (
    IndicatorsSnapshot,
    PositionContext,
    StrategyAction,
)
from tests.strategy.conftest import (
    buy_ready_result,
    make_indicator_result,
    sell_ready_result,
)


class TestEngineBUY:
    def test_buy_when_all_conditions_met(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result, PositionContext(has_open_long_position=False))
        assert decision.action == StrategyAction.BUY

    def test_buy_has_buy_conditions_met_reason(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert ReasonCode.BUY_CONDITIONS_MET in decision.reasons

    def test_buy_never_when_position_open(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        pos = PositionContext(has_open_long_position=True)
        decision = engine.evaluate(result, pos)
        assert decision.action != StrategyAction.BUY


class TestEngineSELL:
    def test_sell_when_bearish_cross_and_position(self):
        result = sell_ready_result()
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action == StrategyAction.SELL

    def test_sell_has_sell_conditions_met_reason(self):
        result = sell_ready_result()
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert ReasonCode.SELL_CONDITIONS_MET in decision.reasons

    def test_sell_never_without_position(self):
        result = sell_ready_result()
        pos = PositionContext(has_open_long_position=False)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action != StrategyAction.SELL

    def test_sell_and_buy_never_simultaneous(self):
        result = sell_ready_result()
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action in (StrategyAction.SELL, StrategyAction.WAIT)
        # never BUY when position open
        assert decision.action != StrategyAction.BUY


class TestEnginePriority:
    def test_sell_takes_priority_over_buy_when_position_open(self):
        """Even if BUY conditions are met, SELL fires first when position open."""
        # A result that technically satisfies buy rules
        result = buy_ready_result()
        # but position is open → sell rules checked, and bearish cross present
        result = make_indicator_result(
            close=str(result.close),
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BEARISH,
        )
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action == StrategyAction.SELL

    def test_warmup_incomplete_highest_priority(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BULLISH,
            warmup_complete=False,
        )
        engine = StrategyEngine()
        decision = engine.evaluate(result, PositionContext(has_open_long_position=True))
        # Even with position open, warmup gate fires first
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.WARMUP_INCOMPLETE in decision.reasons

    def test_missing_indicator_before_position_check(self):
        result = make_indicator_result(
            close="35000",
            ema_long=None,
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.BEARISH,
        )
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action == StrategyAction.WAIT
        assert ReasonCode.MISSING_INDICATOR in decision.reasons


class TestEngineDecisionFields:
    def test_decision_symbol_and_interval(self):
        result = buy_ready_result()
        result = make_indicator_result(symbol="ETHUSDT", interval="15m")
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.symbol == "ETHUSDT"
        assert decision.interval == "15m"

    def test_decision_has_close_price(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert isinstance(decision.close_price, Decimal)

    def test_decision_warmup_complete_flag(self):
        result = make_indicator_result(warmup_complete=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.warmup_complete is True

    def test_decision_has_open_position_flag(self):
        result = buy_ready_result()
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.has_open_position is True

    def test_indicators_snapshot_is_set(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert isinstance(decision.indicators_snapshot, IndicatorsSnapshot)
        assert decision.indicators_snapshot.close == result.close

    def test_indicators_snapshot_not_modified(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        snap = decision.indicators_snapshot
        assert snap.close == result.close
        assert snap.rsi == result.rsi
        assert snap.ema_long == result.ema_long

    def test_strategy_name_and_version_from_config(self):
        cfg = StrategyEngineConfig(strategy_name="my-strat", strategy_version="2.0.0")
        result = buy_ready_result()
        engine = StrategyEngine(cfg)
        decision = engine.evaluate(result)
        assert decision.strategy_name == "my-strat"
        assert decision.strategy_version == "2.0.0"


class TestEngineDeterminism:
    def test_same_input_same_output(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        d1 = engine.evaluate(result)
        d2 = engine.evaluate(result)
        assert d1.action == d2.action
        assert d1.reasons == d2.reasons
        assert d1.failed_conditions == d2.failed_conditions

    def test_clock_injection_gives_fixed_timestamp(self):
        fixed = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        engine = StrategyEngine(clock=lambda: fixed)
        result = make_indicator_result()
        decision = engine.evaluate(result)
        assert decision.generated_at == fixed

    def test_reuse_same_engine_multiple_times(self):
        engine = StrategyEngine()
        result = buy_ready_result()
        for _ in range(5):
            decision = engine.evaluate(result)
            assert decision.action == StrategyAction.BUY


class TestEngineEvaluateSeries:
    def test_series_returns_one_decision_per_result(self):
        results = [buy_ready_result(idx=i) for i in range(5)]
        engine = StrategyEngine()
        decisions = engine.evaluate_series(results)
        assert len(decisions) == 5

    def test_buy_opens_position_for_next_result(self):
        """After a BUY, subsequent results are evaluated with position open."""
        r_buy = buy_ready_result(idx=0)
        r_hold = make_indicator_result(
            idx=1,
            close="35500",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.NONE,  # no sell trigger
        )
        engine = StrategyEngine()
        decisions = engine.evaluate_series([r_buy, r_hold])
        assert decisions[0].action == StrategyAction.BUY
        # After BUY, position is open → hold result gets has_open_position=True
        assert decisions[1].has_open_position is True

    def test_sell_closes_position(self):
        r_buy = buy_ready_result(idx=0)
        r_sell = sell_ready_result(idx=1)
        engine = StrategyEngine()
        decisions = engine.evaluate_series([r_buy, r_sell])
        assert decisions[0].action == StrategyAction.BUY
        assert decisions[1].action == StrategyAction.SELL
        assert decisions[1].has_open_position is True  # was open when evaluated

    def test_wait_does_not_change_position_state(self):
        r_wait = make_indicator_result(idx=0, crossover=CrossSignal.NONE)
        r_buy = buy_ready_result(idx=1)
        engine = StrategyEngine()
        decisions = engine.evaluate_series([r_wait, r_buy])
        assert decisions[0].action == StrategyAction.WAIT
        # After WAIT, still no position → BUY can fire
        assert decisions[1].action == StrategyAction.BUY

    def test_no_double_buy_while_position_open(self):
        r_buy1 = buy_ready_result(idx=0)
        r_buy2 = buy_ready_result(idx=1)
        engine = StrategyEngine()
        decisions = engine.evaluate_series([r_buy1, r_buy2])
        assert decisions[0].action == StrategyAction.BUY
        # Second BUY attempt blocked by open position
        assert decisions[1].action != StrategyAction.BUY

    def test_no_double_sell_without_intervening_buy(self):
        r_sell1 = sell_ready_result(idx=0)
        r_sell2 = sell_ready_result(idx=1)
        engine = StrategyEngine()
        initial = PositionContext(has_open_long_position=True)
        decisions = engine.evaluate_series([r_sell1, r_sell2], initial_position=initial)
        assert decisions[0].action == StrategyAction.SELL
        # Second sell: position already closed → no SELL
        assert decisions[1].action != StrategyAction.SELL

    def test_series_with_initial_position_open(self):
        r_sell = sell_ready_result(idx=0)
        engine = StrategyEngine()
        initial = PositionContext(has_open_long_position=True)
        decisions = engine.evaluate_series([r_sell], initial_position=initial)
        assert decisions[0].action == StrategyAction.SELL
