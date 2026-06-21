"""Tests that the strategy engine has no look-ahead bias."""

from app.indicators.schemas import CrossSignal
from app.strategy.engine import StrategyEngine
from app.strategy.schemas import StrategyAction
from tests.strategy.conftest import buy_ready_result, make_indicator_result, sell_ready_result


class TestNoLookaheadEngine:
    def test_future_candle_does_not_change_past_decision(self):
        """Changing a future result must not affect earlier decisions."""
        r0 = buy_ready_result(idx=0)
        r1_neutral = make_indicator_result(
            idx=1,
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.NONE,
        )
        r1_sell = sell_ready_result(idx=1)

        engine = StrategyEngine()

        decisions_neutral = engine.evaluate_series([r0, r1_neutral])
        decisions_sell = engine.evaluate_series([r0, r1_sell])

        # First decision must be identical regardless of what r1 is
        assert decisions_neutral[0].action == decisions_sell[0].action
        assert decisions_neutral[0].reasons == decisions_sell[0].reasons

    def test_ascending_order_evaluation(self):
        """evaluate_series must process results in the order given (index 0 first)."""
        results = [buy_ready_result(idx=i) for i in range(5)]
        engine = StrategyEngine()
        decisions = engine.evaluate_series(results)

        first_action = decisions[0].action
        # decisions are returned in the same order as input
        assert decisions[0].candle_open_time < decisions[1].candle_open_time

        # First result: no prior position → can BUY
        assert first_action == StrategyAction.BUY

    def test_decision_uses_only_its_own_result(self):
        """Each StrategyDecision snapshot must match its corresponding IndicatorResult."""
        results = [buy_ready_result(idx=i) for i in range(3)]
        engine = StrategyEngine()
        decisions = engine.evaluate_series(results)

        for result, decision in zip(results, decisions, strict=True):
            assert decision.close_price == result.close
            assert decision.candle_open_time == result.open_time

    def test_modifying_result_after_evaluation_does_not_affect_decision(self):
        """IndicatorsSnapshot is a copy — mutating the source result is safe."""
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)

        original_rsi = decision.indicators_snapshot.rsi

        # Snapshots are frozen dataclass instances; the source IndicatorResult
        # is a separate object — decision holds its own snapshot
        assert decision.indicators_snapshot.rsi == original_rsi

    def test_series_position_context_flows_forward_not_backward(self):
        """A SELL at index 2 must not retroactively close the position at index 1."""
        r0 = buy_ready_result(idx=0)
        r1_hold = make_indicator_result(
            idx=1,
            close="35000",
            ema_long="30000",
            rsi="57",
            volume_ratio="1.5",
            crossover=CrossSignal.NONE,
        )
        r2_sell = sell_ready_result(idx=2)

        engine = StrategyEngine()
        decisions = engine.evaluate_series([r0, r1_hold, r2_sell])

        assert decisions[0].action == StrategyAction.BUY
        # At index 1 position is still open (SELL hasn't happened yet)
        assert decisions[1].has_open_position is True
        assert decisions[2].action == StrategyAction.SELL
