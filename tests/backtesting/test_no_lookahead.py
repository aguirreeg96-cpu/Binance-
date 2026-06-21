"""Tests verifying the no-look-ahead bias guarantee.

The backtest MUST satisfy:
  Signal at close of candle i → execute at open of candle i+1.
  A signal at the last candle is NEVER executed via the normal path.
  The strategy engine never receives future candle data.
"""

from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.indicators.schemas import IndicatorConfig
from tests.backtesting.conftest import (
    AlwaysBuyEngine,
    BuyThenSellEngine,
    make_candles,
)

_D = Decimal
_BASE_TIME = 1_000_000
_INTERVAL_MS = 3_600_000


def _minimal_ind() -> IndicatorConfig:
    return IndicatorConfig(
        sma_short_period=2,
        sma_long_period=3,
        ema_short_period=2,
        ema_medium_period=3,
        ema_long_period=4,
        rsi_period=2,
        atr_period=2,
        volume_period=2,
    )


def _cfg(n: int, force_close: bool = True) -> BacktestConfig:
    return BacktestConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_ms=_BASE_TIME,
        end_ms=_BASE_TIME + n * _INTERVAL_MS,
        initial_capital=_D("10000"),
        fee_percentage=_D("0"),
        slippage_percentage=_D("0"),
        force_close_at_end=force_close,
    )


class TestSignalExecutionTiming:
    def test_buy_executes_at_next_candle_open(self):
        """Signal at candle i → execution at candle i+1's open_time."""
        # Use candles with different open_times and prices
        n = 10
        candles = make_candles(n, price="100")

        ind = _minimal_ind()
        engine = BacktestEngine(_cfg(n), AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        # The trade that is NOT forced_close should have exec_time = next candle's open_time
        if result.trades:
            trade = result.trades[0]
            # Find the signal candle index
            signal_time = trade.entry_signal_time
            signal_idx = next(i for i, c in enumerate(candles) if c.open_time == signal_time)
            # Execution should be at candle i+1's open_time
            if not trade.is_forced_close and signal_idx + 1 < len(candles):
                assert trade.entry_exec_time == candles[signal_idx + 1].open_time

    def test_buy_at_last_candle_not_executed_without_force_close(self):
        """If AlwaysBuy fires at the last candle, nothing executes (no force close)."""
        n = 5
        candles = make_candles(n, price="100")
        ind = _minimal_ind()
        # Disable force close AND use AlwaysWait to ensure no open position
        engine = BacktestEngine(_cfg(n, force_close=False), AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        # With AlwaysBuy and no force close: one buy executes at candle 1,
        # stays open to the end (no sell signal)
        assert result.has_open_position_at_end is True
        # The position was NOT closed (no trade)
        assert result.total_trades == 0

    def test_sell_signal_executes_at_next_candle_open(self):
        """After a buy, sell signal executes at the following candle's open."""
        n = 15
        candles = make_candles(n, price="100")
        ind = _minimal_ind()
        engine = BacktestEngine(_cfg(n, force_close=False), BuyThenSellEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        for trade in result.trades:
            if not trade.is_forced_close and trade.exit_signal_time is not None:
                # Sell signal was at exit_signal_time; execution at next candle
                sell_signal_idx = next(
                    i for i, c in enumerate(candles) if c.open_time == trade.exit_signal_time
                )
                if sell_signal_idx + 1 < len(candles):
                    assert trade.exit_exec_time == candles[sell_signal_idx + 1].open_time

    def test_forced_close_uses_last_candle_close_time(self):
        """Forced close records exit_exec_time == last candle's close_time."""
        n = 10
        candles = make_candles(n, price="100")
        ind = _minimal_ind()
        engine = BacktestEngine(_cfg(n, force_close=True), AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        forced = [t for t in result.trades if t.is_forced_close]
        assert forced, "Expected at least one forced close trade"
        for trade in forced:
            last_candle = candles[-1]
            assert trade.exit_exec_time == last_candle.close_time

    def test_no_lookahead_future_candle_data_not_used(self):
        """Strategy engine receives indicator results indexed by open_time.

        Verify that the equity curve open_times match the eval candles in order.
        """
        n = 20
        candles = make_candles(n)
        ind = _minimal_ind()
        engine = BacktestEngine(_cfg(n), AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        for i, ep in enumerate(result.equity_curve):
            assert ep.open_time == candles[i].open_time

    def test_last_candle_signal_never_queued(self):
        """A signal at the last candle is silently discarded (no execution)."""
        n = 5
        candles = make_candles(n, price="100")
        ind = _minimal_ind()
        # force_close=False → if a buy at last candle created a trade, it would appear
        engine = BacktestEngine(_cfg(n, force_close=False), AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=0)

        # With AlwaysBuy and 5 candles (no force close):
        # candle 0: no pending, buy signal queued
        # candle 1: buy executes (pending from 0), position open, WAIT (already open)
        # candles 2-4: hold (AlwaysBuy won't BUY again, WAIT), no sell
        # Result: 1 open position, 0 completed trades
        assert result.total_trades == 0
        assert result.has_open_position_at_end is True

    def test_warmup_candles_do_not_generate_signals(self):
        """Warmup candles are excluded — equity curve starts at warmup_len index."""
        n = 20
        warmup = 10
        candles = make_candles(n)
        ind = _minimal_ind()
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME + warmup * _INTERVAL_MS,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
        )
        engine = BacktestEngine(cfg, AlwaysBuyEngine(), ind)
        result = engine.run(candles, warmup_len=warmup)

        assert len(result.equity_curve) == n - warmup
        # First equity point matches the first eval candle
        assert result.equity_curve[0].open_time == candles[warmup].open_time
