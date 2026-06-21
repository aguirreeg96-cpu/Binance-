"""Tests for BacktestConfig validation."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.backtesting.config import BacktestConfig


class TestBacktestConfig:
    def _valid(self, **kwargs) -> BacktestConfig:
        defaults = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "start_ms": 1_000_000,
            "end_ms": 2_000_000,
            "initial_capital": Decimal("10000"),
        }
        defaults.update(kwargs)
        return BacktestConfig(**defaults)

    def test_valid_config_creates_successfully(self):
        cfg = self._valid()
        assert cfg.symbol == "BTCUSDT"
        assert cfg.interval == "1h"

    def test_symbol_uppercased(self):
        cfg = self._valid(symbol="btcusdt")
        assert cfg.symbol == "BTCUSDT"

    def test_end_ms_must_be_greater_than_start_ms(self):
        with pytest.raises(Exception, match="end_ms"):
            self._valid(start_ms=2_000_000, end_ms=1_000_000)

    def test_end_ms_equal_to_start_ms_rejected(self):
        with pytest.raises(ValueError):
            self._valid(start_ms=1_000_000, end_ms=1_000_000)

    def test_invalid_interval_rejected(self):
        with pytest.raises(Exception, match="interval"):
            self._valid(interval="99x")

    def test_initial_capital_must_be_positive(self):
        with pytest.raises(ValueError):
            self._valid(initial_capital=Decimal("0"))

    def test_fee_percentage_cannot_exceed_5(self):
        with pytest.raises(ValueError):
            self._valid(fee_percentage=Decimal("6"))

    def test_slippage_percentage_cannot_exceed_5(self):
        with pytest.raises(ValueError):
            self._valid(slippage_percentage=Decimal("6"))

    def test_fee_rate_property(self):
        cfg = self._valid(fee_percentage=Decimal("0.1"))
        assert cfg.fee_rate == Decimal("0.001")

    def test_slippage_rate_property(self):
        cfg = self._valid(slippage_percentage=Decimal("0.05"))
        assert cfg.slippage_rate == Decimal("0.0005")

    def test_zero_fee_allowed(self):
        cfg = self._valid(fee_percentage=Decimal("0"))
        assert cfg.fee_rate == Decimal("0")

    def test_zero_slippage_allowed(self):
        cfg = self._valid(slippage_percentage=Decimal("0"))
        assert cfg.slippage_rate == Decimal("0")

    def test_force_close_default_true(self):
        cfg = self._valid()
        assert cfg.force_close_at_end is True

    def test_force_close_can_be_disabled(self):
        cfg = self._valid(force_close_at_end=False)
        assert cfg.force_close_at_end is False

    def test_config_is_frozen(self):
        cfg = self._valid()
        with pytest.raises((TypeError, ValidationError)):
            cfg.symbol = "ETHUSDT"  # type: ignore[misc]

    def test_all_valid_intervals_accepted(self):
        from app.market_data.interval_utils import VALID_INTERVALS

        for interval in VALID_INTERVALS:
            cfg = self._valid(interval=interval)
            assert cfg.interval == interval

    def test_initial_capital_as_string_accepted(self):
        cfg = self._valid(initial_capital=Decimal("50000.50"))
        assert cfg.initial_capital == Decimal("50000.50")
