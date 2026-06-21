"""Tests for compute_metrics and BacktestResult field correctness."""

from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import _compute_streaks
from app.backtesting.schemas import BacktestTrade
from tests.backtesting.conftest import (
    AlwaysBuyEngine,
    AlwaysWaitEngine,
    BuyThenSellEngine,
    make_candles,
)

_D = Decimal
_BASE_TIME = 1_000_000
_INTERVAL_MS = 3_600_000


def _minimal_ind():
    from app.indicators.schemas import IndicatorConfig

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


def _cfg(n: int, fee: str = "0", slip: str = "0") -> BacktestConfig:
    return BacktestConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_ms=_BASE_TIME,
        end_ms=_BASE_TIME + n * _INTERVAL_MS,
        initial_capital=_D("10000"),
        fee_percentage=_D(fee),
        slippage_percentage=_D(slip),
        force_close_at_end=True,
    )


class TestComputeStreaks:
    def _trade(self, net_pnl: str) -> BacktestTrade:
        d = _D(net_pnl)
        return BacktestTrade(
            trade_id=1,
            entry_signal_time=0,
            entry_exec_time=0,
            entry_exec_price=_D("100"),
            entry_fee=_D("0"),
            quantity=_D("1"),
            exit_signal_time=None,
            exit_exec_time=1,
            exit_exec_price=_D("100"),
            exit_fee=_D("0"),
            gross_pnl=d,
            net_pnl=d,
            return_pct=d,
            is_forced_close=False,
            capital_at_entry=_D("100"),
        )

    def test_no_trades(self):
        assert _compute_streaks([]) == (0, 0)

    def test_all_wins(self):
        trades = [self._trade("10")] * 5
        assert _compute_streaks(trades) == (5, 0)

    def test_all_losses(self):
        trades = [self._trade("-10")] * 4
        assert _compute_streaks(trades) == (0, 4)

    def test_alternating(self):
        trades = [
            self._trade("10"),
            self._trade("-5"),
            self._trade("10"),
            self._trade("-5"),
        ]
        assert _compute_streaks(trades) == (1, 1)

    def test_long_win_streak(self):
        trades = [self._trade("-5"), self._trade("10")] * 3 + [self._trade("10")] * 4
        max_wins, max_losses = _compute_streaks(trades)
        assert max_wins >= 4
        assert max_losses == 1


class TestBacktestResultMetrics:
    def test_no_trades_win_rate_is_none(self):
        n = 10
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(make_candles(n), 0)
        assert result.win_rate_pct is None
        assert result.profit_factor is None
        assert result.expectancy_pct is None
        assert result.avg_win_pct is None
        assert result.avg_loss_pct is None

    def test_total_return_pct_calculation(self):
        n = 10
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(make_candles(n), 0)
        expected = (result.final_equity - result.initial_capital) / result.initial_capital * 100
        assert result.total_return_pct == expected

    def test_max_drawdown_non_negative(self):
        n = 10
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(make_candles(n), 0)
        assert result.max_drawdown_pct >= _D("0")

    def test_exposure_zero_when_no_positions(self):
        n = 10
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(make_candles(n), 0)
        assert result.exposure_pct == _D("0")

    def test_exposure_positive_when_positions_held(self):
        n = 20
        result = BacktestEngine(_cfg(n), AlwaysBuyEngine(), _minimal_ind()).run(make_candles(n), 0)
        assert result.exposure_pct > _D("0")

    def test_total_fees_sum_matches_trades(self):
        n = 15
        cfg = _cfg(n, fee="0.1")
        result = BacktestEngine(cfg, BuyThenSellEngine(), _minimal_ind()).run(make_candles(n), 0)
        trade_fees = sum(t.entry_fee + t.exit_fee for t in result.trades)
        assert result.total_fees == trade_fees

    def test_first_and_last_candle_times(self):
        n = 15
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(candles, 0)
        assert result.first_candle_open_time == candles[0].open_time
        assert result.last_candle_open_time == candles[-1].open_time

    def test_winning_plus_losing_equals_total(self):
        n = 20
        result = BacktestEngine(_cfg(n), BuyThenSellEngine(), _minimal_ind()).run(
            make_candles(n), 0
        )
        assert result.winning_trades + result.losing_trades == result.total_trades

    def test_bah_return_is_decimal(self):
        n = 10
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(make_candles(n), 0)
        assert isinstance(result.buy_and_hold_return_pct, Decimal)
