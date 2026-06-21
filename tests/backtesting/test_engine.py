"""Tests for BacktestEngine — core simulation loop."""

from decimal import Decimal

import pytest

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.exceptions import BacktestError, BacktestInsufficientDataError
from app.indicators.schemas import IndicatorConfig
from tests.backtesting.conftest import (
    AlwaysBuyEngine,
    AlwaysWaitEngine,
    BuyThenSellEngine,
    make_candles,
)

_D = Decimal
_BASE_TIME = 1_000_000
_INTERVAL_MS = 3_600_000


def _minimal_config(**kwargs) -> IndicatorConfig:
    defaults = {
        "sma_short_period": 2,
        "sma_long_period": 3,
        "ema_short_period": 2,
        "ema_medium_period": 3,
        "ema_long_period": 4,
        "rsi_period": 2,
        "atr_period": 2,
        "volume_period": 2,
    }
    defaults.update(kwargs)
    return IndicatorConfig(**defaults)


def _config(start_offset: int = 0, n_candles: int = 20, **kwargs) -> BacktestConfig:
    defaults = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start_ms": _BASE_TIME,
        "end_ms": _BASE_TIME + n_candles * _INTERVAL_MS,
        "initial_capital": _D("10000"),
        "fee_percentage": _D("0"),
        "slippage_percentage": _D("0"),
        "force_close_at_end": True,
    }
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


class TestBacktestEngineBasic:
    def test_raises_on_empty_eval_candles(self):
        candles = make_candles(3)
        engine = BacktestEngine(_config(), AlwaysWaitEngine(), _minimal_config())
        with pytest.raises(BacktestInsufficientDataError):
            engine.run(candles, warmup_len=3)

    def test_raises_on_negative_warmup(self):
        candles = make_candles(5)
        engine = BacktestEngine(_config(), AlwaysWaitEngine(), _minimal_config())
        with pytest.raises(BacktestError, match="warmup_len"):
            engine.run(candles, warmup_len=-1)

    def test_always_wait_no_trades(self):
        candles = make_candles(20)
        engine = BacktestEngine(_config(), AlwaysWaitEngine(), _minimal_config())
        result = engine.run(candles, warmup_len=0)
        assert result.total_trades == 0
        assert result.final_equity == _D("10000")

    def test_equity_curve_length_equals_eval_candles(self):
        candles = make_candles(20)
        engine = BacktestEngine(_config(), AlwaysWaitEngine(), _minimal_config())
        result = engine.run(candles, warmup_len=5)
        assert len(result.equity_curve) == 15

    def test_equity_curve_all_eval_candles_with_zero_warmup(self):
        n = 10
        candles = make_candles(n)
        engine = BacktestEngine(_config(n_candles=n), AlwaysWaitEngine(), _minimal_config())
        result = engine.run(candles, warmup_len=0)
        assert len(result.equity_curve) == n

    def test_result_has_correct_candle_counts(self):
        candles = make_candles(20)
        engine = BacktestEngine(_config(), AlwaysWaitEngine(), _minimal_config())
        result = engine.run(candles, warmup_len=5)
        assert result.total_candles == 15


class TestBacktestEngineWithTrades:
    def test_buy_then_forced_close(self):
        """AlwaysBuy buys once, then force-closes at last candle."""
        n = 10
        candles = make_candles(n, price="100")
        ind = _minimal_config()
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
            force_close_at_end=True,
        )
        engine = BacktestEngine(cfg, AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        assert result.total_trades == 1
        assert result.trades[0].is_forced_close is True

    def test_flat_market_zero_costs_returns_zero(self):
        """At constant price with no costs, return should be 0."""
        n = 15
        candles = make_candles(n, price="100")
        ind = _minimal_config()
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
            force_close_at_end=True,
        )
        engine = BacktestEngine(cfg, AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)
        assert result.total_return_pct == _D("0")

    def test_buy_then_sell_creates_trade(self):
        """BuyThenSell: one trade, not forced."""
        n = 15
        candles = make_candles(n, price="100")
        ind = _minimal_config()
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
            force_close_at_end=False,
        )
        engine = BacktestEngine(cfg, BuyThenSellEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        assert result.total_trades >= 1
        # At least one trade should NOT be forced
        non_forced = [t for t in result.trades if not t.is_forced_close]
        assert len(non_forced) >= 1

    def test_final_equity_after_no_trades_equals_initial(self):
        n = 20
        candles = make_candles(n)
        ind = _minimal_config()
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("5000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
            force_close_at_end=True,
        )
        engine = BacktestEngine(cfg, AlwaysWaitEngine(), ind)
        result = engine.run(candles, warmup_len=0)
        assert result.final_equity == _D("5000")
        assert not result.has_open_position_at_end

    def test_has_open_position_at_end_when_force_close_disabled(self):
        n = 10
        candles = make_candles(n, price="100")
        ind = _minimal_config()
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
            force_close_at_end=False,
        )
        engine = BacktestEngine(cfg, AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)
        assert result.has_open_position_at_end is True

    def test_warmup_candles_excluded_from_equity_curve(self):
        n_warmup = 5
        n_eval = 10
        candles = make_candles(n_warmup + n_eval)
        ind = _minimal_config()
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME + n_warmup * _INTERVAL_MS,
            end_ms=_BASE_TIME + (n_warmup + n_eval) * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
        )
        engine = BacktestEngine(cfg, AlwaysWaitEngine(), ind)
        result = engine.run(candles, warmup_len=n_warmup)
        assert len(result.equity_curve) == n_eval
        assert result.equity_curve[0].open_time == candles[n_warmup].open_time

    def test_fees_reduce_return(self):
        n = 15
        candles = make_candles(n, price="100")
        ind = _minimal_config()
        base = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "start_ms": _BASE_TIME,
            "end_ms": _BASE_TIME + n * _INTERVAL_MS,
            "initial_capital": _D("10000"),
            "slippage_percentage": _D("0"),
            "force_close_at_end": True,
        }
        no_fee_cfg = BacktestConfig(**{**base, "fee_percentage": _D("0")})
        fee_cfg = BacktestConfig(**{**base, "fee_percentage": _D("0.1")})

        r_no_fee = BacktestEngine(no_fee_cfg, AlwaysBuyEngine(), ind).run(candles, 0)
        r_fee = BacktestEngine(fee_cfg, AlwaysBuyEngine(), ind).run(candles, 0)

        assert r_fee.total_fees > _D("0")
        assert r_fee.final_equity < r_no_fee.final_equity
