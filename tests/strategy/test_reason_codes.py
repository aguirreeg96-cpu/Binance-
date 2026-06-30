"""Tests for ReasonCode completeness and stability."""

from app.indicators.schemas import CrossSignal
from app.strategy.engine import StrategyEngine
from app.strategy.reasons import REASON_DESCRIPTIONS, ReasonCode
from app.strategy.schemas import PositionContext, StrategyAction
from tests.strategy.conftest import (
    buy_ready_result,
    make_indicator_result,
    sell_ready_result,
)


class TestReasonCodeCoverage:
    def test_all_reason_codes_have_descriptions(self):
        for code in ReasonCode:
            assert code in REASON_DESCRIPTIONS, (
                f"ReasonCode.{code.name} has no description in REASON_DESCRIPTIONS"
            )

    def test_descriptions_are_non_empty(self):
        for code, desc in REASON_DESCRIPTIONS.items():
            assert desc.strip(), f"Description for {code.name} is empty"

    def test_all_reason_codes_are_strings(self):
        for code in ReasonCode:
            assert isinstance(code.value, str)
            assert code.value == code.value.upper()  # stable UPPER_SNAKE_CASE


class TestReasonCodeProducedByEngine:
    def test_buy_decision_has_buy_conditions_met(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert decision.action == StrategyAction.BUY
        assert ReasonCode.BUY_CONDITIONS_MET in decision.reasons

    def test_sell_decision_has_sell_conditions_met(self):
        result = sell_ready_result()
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert decision.action == StrategyAction.SELL
        assert ReasonCode.SELL_CONDITIONS_MET in decision.reasons

    def test_warmup_incomplete_reason_code(self):
        result = make_indicator_result(warmup_complete=False)
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert ReasonCode.WARMUP_INCOMPLETE in decision.reasons

    def test_missing_indicator_reason_code(self):
        result = make_indicator_result(ema_long=None)
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert ReasonCode.MISSING_INDICATOR in decision.reasons

    def test_no_new_crossover_reason_code(self):
        result = make_indicator_result(crossover=CrossSignal.NONE)
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert ReasonCode.NO_NEW_CROSSOVER in decision.failed_conditions

    def test_bearish_crossover_reason_code_in_sell(self):
        result = sell_ready_result()
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert ReasonCode.BEARISH_CROSSOVER in decision.reasons

    def test_position_already_open_reason_code(self):
        result = make_indicator_result(crossover=CrossSignal.NONE)
        pos = PositionContext(has_open_long_position=True)
        engine = StrategyEngine()
        decision = engine.evaluate(result, pos)
        assert ReasonCode.POSITION_ALREADY_OPEN in decision.reasons

    def test_rsi_in_buy_range_reason_code(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert ReasonCode.RSI_IN_BUY_RANGE in decision.reasons

    def test_volume_confirmed_reason_code(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert ReasonCode.VOLUME_CONFIRMED in decision.reasons

    def test_price_above_long_ema_reason_code(self):
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        assert ReasonCode.PRICE_ABOVE_LONG_EMA in decision.reasons

    def test_reasons_and_failed_are_disjoint(self):
        """No reason code should appear in both reasons and failed_conditions."""
        result = buy_ready_result()
        engine = StrategyEngine()
        decision = engine.evaluate(result)
        overlap = set(decision.reasons) & set(decision.failed_conditions)
        assert not overlap, f"Overlap between reasons and failed_conditions: {overlap}"
