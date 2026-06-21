"""Tests for SELL rule evaluation."""

from app.indicators.schemas import CrossSignal
from app.strategy.config import StrategyEngineConfig
from app.strategy.reasons import ReasonCode
from app.strategy.rules import evaluate_sell_conditions
from tests.strategy.conftest import make_indicator_result, sell_ready_result


class TestSellBearishCrossover:
    def test_bearish_crossover_triggers_sell(self):
        result = sell_ready_result()
        cfg = StrategyEngineConfig()
        should_sell, met, _ = evaluate_sell_conditions(result, cfg)
        assert should_sell is True
        assert ReasonCode.BEARISH_CROSSOVER in met

    def test_no_bearish_crossover_no_sell_when_required(self):
        result = make_indicator_result(
            close="29000",
            ema_long="30000",
            rsi="45",
            volume_ratio="1.2",
            crossover=CrossSignal.NONE,
        )
        cfg = StrategyEngineConfig(require_bearish_crossover_for_sell=True)
        should_sell, _, failed = evaluate_sell_conditions(result, cfg)
        assert should_sell is False
        assert ReasonCode.NO_NEW_CROSSOVER in failed

    def test_bearish_crossover_not_required_sell_on_other_conditions(self):
        result = make_indicator_result(
            close="29000",
            ema_long="30000",  # close < ema_long → price exit
            rsi="45",
            volume_ratio="1.2",
            crossover=CrossSignal.NONE,
        )
        cfg = StrategyEngineConfig(require_bearish_crossover_for_sell=False)
        should_sell, met, _ = evaluate_sell_conditions(result, cfg)
        assert should_sell is True
        assert ReasonCode.PRICE_BELOW_LONG_EMA in met


class TestSellPriceBelowLongEMA:
    def test_price_below_ema_long_is_sell_reason(self):
        result = make_indicator_result(
            close="29000",
            ema_long="30000",  # close < ema_long
            rsi="45",
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_sell_conditions(result, cfg)
        assert ReasonCode.PRICE_BELOW_LONG_EMA in met

    def test_price_above_ema_long_not_a_sell_reason(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",  # close > ema_long
            rsi="45",
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_sell_conditions(result, cfg)
        assert ReasonCode.PRICE_BELOW_LONG_EMA not in met


class TestSellRSIOverbought:
    def test_rsi_at_threshold_triggers_sell_reason(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="75",  # exactly at sell_rsi_overbought
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_sell_conditions(result, cfg)
        assert ReasonCode.RSI_OVERBOUGHT in met

    def test_rsi_above_threshold_triggers_sell_reason(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="80",
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_sell_conditions(result, cfg)
        assert ReasonCode.RSI_OVERBOUGHT in met

    def test_rsi_below_threshold_not_overbought(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="74",
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,
        )
        cfg = StrategyEngineConfig()
        _, met, _ = evaluate_sell_conditions(result, cfg)
        assert ReasonCode.RSI_OVERBOUGHT not in met

    def test_rsi_overbought_alone_not_sufficient_when_crossover_required(self):
        result = make_indicator_result(
            close="35000",
            ema_long="30000",
            rsi="80",  # overbought
            volume_ratio="1.2",
            crossover=CrossSignal.NONE,  # no bearish cross
        )
        cfg = StrategyEngineConfig(require_bearish_crossover_for_sell=True)
        should_sell, _, _ = evaluate_sell_conditions(result, cfg)
        assert should_sell is False


class TestSellMultipleReasons:
    def test_multiple_exit_reasons_collected(self):
        result = make_indicator_result(
            close="29000",
            ema_long="30000",  # price exit
            rsi="78",  # RSI overbought
            volume_ratio="1.2",
            crossover=CrossSignal.BEARISH,  # cross exit
        )
        cfg = StrategyEngineConfig()
        should_sell, met, _ = evaluate_sell_conditions(result, cfg)
        assert should_sell is True
        assert ReasonCode.BEARISH_CROSSOVER in met
        assert ReasonCode.PRICE_BELOW_LONG_EMA in met
        assert ReasonCode.RSI_OVERBOUGHT in met

    def test_never_generates_short(self):
        """SELL rule output carries no concept of opening a short position."""
        result = sell_ready_result()
        cfg = StrategyEngineConfig()
        should_sell, met, failed = evaluate_sell_conditions(result, cfg)
        # The reason codes returned must never imply a short
        all_codes = {str(r) for r in met + failed}
        assert "SHORT" not in " ".join(all_codes).upper()
        assert "MARGIN" not in " ".join(all_codes).upper()
        assert "FUTURES" not in " ".join(all_codes).upper()
