"""Tests for StrategyEngineConfig validation."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.strategy.config import StrategyEngineConfig


class TestStrategyEngineConfigDefaults:
    def test_default_config_is_valid(self):
        cfg = StrategyEngineConfig()
        assert cfg.strategy_name == "ema-rsi-volume-v1"
        assert cfg.strategy_version == "1.0.0"
        assert cfg.require_warmup_complete is True
        assert cfg.require_bullish_crossover is True
        assert cfg.require_bearish_crossover_for_sell is True

    def test_default_rsi_bounds(self):
        cfg = StrategyEngineConfig()
        assert cfg.buy_rsi_min == Decimal("50")
        assert cfg.buy_rsi_max == Decimal("65")
        assert cfg.sell_rsi_overbought == Decimal("75")

    def test_default_volume_ratio(self):
        cfg = StrategyEngineConfig()
        assert cfg.minimum_volume_ratio == Decimal("1")

    def test_allow_sell_without_open_position_is_false(self):
        cfg = StrategyEngineConfig()
        assert cfg.allow_sell_without_open_position is False

    def test_config_is_frozen(self):
        cfg = StrategyEngineConfig()
        with pytest.raises((ValidationError, TypeError)):
            cfg.buy_rsi_min = Decimal("40")  # type: ignore[misc]


class TestStrategyEngineConfigRSIValidation:
    def test_rsi_min_must_be_less_than_max(self):
        with pytest.raises(ValidationError, match="buy_rsi_min"):
            StrategyEngineConfig(buy_rsi_min=Decimal("65"), buy_rsi_max=Decimal("50"))

    def test_rsi_min_equal_to_max_raises(self):
        with pytest.raises(ValidationError, match="buy_rsi_min"):
            StrategyEngineConfig(buy_rsi_min=Decimal("60"), buy_rsi_max=Decimal("60"))

    def test_rsi_min_below_zero_raises(self):
        with pytest.raises(ValidationError):
            StrategyEngineConfig(buy_rsi_min=Decimal("-1"), buy_rsi_max=Decimal("65"))

    def test_rsi_max_above_100_raises(self):
        with pytest.raises(ValidationError):
            StrategyEngineConfig(buy_rsi_min=Decimal("50"), buy_rsi_max=Decimal("101"))

    def test_sell_rsi_above_100_raises(self):
        with pytest.raises(ValidationError):
            StrategyEngineConfig(sell_rsi_overbought=Decimal("101"))

    def test_sell_rsi_below_zero_raises(self):
        with pytest.raises(ValidationError):
            StrategyEngineConfig(sell_rsi_overbought=Decimal("-1"))


class TestStrategyEngineConfigVolumeValidation:
    def test_negative_volume_ratio_raises(self):
        with pytest.raises(ValidationError, match="minimum_volume_ratio"):
            StrategyEngineConfig(minimum_volume_ratio=Decimal("-0.1"))

    def test_zero_volume_ratio_is_valid(self):
        cfg = StrategyEngineConfig(minimum_volume_ratio=Decimal("0"))
        assert cfg.minimum_volume_ratio == Decimal("0")

    def test_large_volume_ratio_is_valid(self):
        cfg = StrategyEngineConfig(minimum_volume_ratio=Decimal("5"))
        assert cfg.minimum_volume_ratio == Decimal("5")


class TestStrategyEngineConfigIdentity:
    def test_empty_strategy_name_raises(self):
        with pytest.raises(ValidationError, match="strategy_name"):
            StrategyEngineConfig(strategy_name="")

    def test_whitespace_strategy_name_raises(self):
        with pytest.raises(ValidationError, match="strategy_name"):
            StrategyEngineConfig(strategy_name="   ")

    def test_empty_strategy_version_raises(self):
        with pytest.raises(ValidationError, match="strategy_version"):
            StrategyEngineConfig(strategy_version="")

    def test_custom_valid_config_accepted(self):
        cfg = StrategyEngineConfig(
            buy_rsi_min=Decimal("40"),
            buy_rsi_max=Decimal("60"),
            sell_rsi_overbought=Decimal("80"),
            minimum_volume_ratio=Decimal("1.5"),
            strategy_name="custom-v2",
            strategy_version="2.0.0",
        )
        assert cfg.buy_rsi_min == Decimal("40")
        assert cfg.strategy_name == "custom-v2"
