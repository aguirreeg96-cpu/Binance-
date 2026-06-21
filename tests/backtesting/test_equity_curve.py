"""Tests for EquityCurveBuilder and EquityPoint correctness."""

from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.equity_curve import EquityCurveBuilder
from app.backtesting.portfolio import PortfolioState
from tests.backtesting.conftest import (
    AlwaysBuyEngine,
    AlwaysWaitEngine,
    make_candle,
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


def _cfg(n: int) -> BacktestConfig:
    return BacktestConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_ms=_BASE_TIME,
        end_ms=_BASE_TIME + n * _INTERVAL_MS,
        initial_capital=_D("10000"),
        fee_percentage=_D("0"),
        slippage_percentage=_D("0"),
    )


class TestEquityCurveBuilder:
    def test_empty_builder_returns_empty_list(self):
        builder = EquityCurveBuilder()
        assert builder.build() == []

    def test_appended_point_has_correct_fields(self):
        builder = EquityCurveBuilder()
        candle = make_candle(open_time=1000, close_p="200")
        portfolio = PortfolioState(
            initial_capital=_D("10000"),
            quote_balance=_D("10000"),
        )
        portfolio.peak_equity = _D("10000")
        builder.append(candle, portfolio, _D("10000"))

        points = builder.build()
        assert len(points) == 1
        ep = points[0]
        assert ep.open_time == 1000
        assert ep.equity == _D("10000")
        assert ep.quote_balance == _D("10000")
        assert ep.base_balance == _D("0")
        assert ep.drawdown_pct == _D("0")
        assert not ep.has_open_position

    def test_drawdown_calculated_correctly(self):
        builder = EquityCurveBuilder()
        candle = make_candle(open_time=1000, close_p="100")
        portfolio = PortfolioState(
            initial_capital=_D("10000"),
            quote_balance=_D("8000"),
        )
        portfolio.peak_equity = _D("10000")
        equity = _D("8000")
        builder.append(candle, portfolio, equity)

        ep = builder.build()[0]
        # drawdown = (10000 - 8000) / 10000 * 100 = 20%
        assert ep.drawdown_pct == _D("20")

    def test_no_drawdown_at_peak(self):
        builder = EquityCurveBuilder()
        candle = make_candle(open_time=1000)
        portfolio = PortfolioState(
            initial_capital=_D("12000"),
            quote_balance=_D("12000"),
        )
        portfolio.peak_equity = _D("12000")
        builder.append(candle, portfolio, _D("12000"))

        ep = builder.build()[0]
        assert ep.drawdown_pct == _D("0")


class TestEquityPointsFromEngine:
    def test_one_point_per_eval_candle(self):
        n = 15
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(candles, 0)
        assert len(result.equity_curve) == n

    def test_equity_points_ascending_open_time(self):
        n = 15
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(candles, 0)
        times = [ep.open_time for ep in result.equity_curve]
        assert times == sorted(times)

    def test_equity_all_decimal_type(self):
        n = 10
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(candles, 0)
        for ep in result.equity_curve:
            assert isinstance(ep.equity, Decimal)
            assert isinstance(ep.drawdown_pct, Decimal)
            assert isinstance(ep.close_price, Decimal)

    def test_equity_non_negative(self):
        n = 15
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysBuyEngine(), _minimal_ind()).run(candles, 0)
        for ep in result.equity_curve:
            assert ep.equity >= _D("0")

    def test_drawdown_bounded_zero_to_hundred(self):
        n = 15
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysBuyEngine(), _minimal_ind()).run(candles, 0)
        for ep in result.equity_curve:
            assert _D("0") <= ep.drawdown_pct <= _D("100")

    def test_peak_equity_non_decreasing(self):
        n = 10
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysWaitEngine(), _minimal_ind()).run(candles, 0)
        peaks = [ep.peak_equity for ep in result.equity_curve]
        for i in range(1, len(peaks)):
            assert peaks[i] >= peaks[i - 1]

    def test_base_value_equals_base_balance_times_close(self):
        n = 10
        candles = make_candles(n)
        result = BacktestEngine(_cfg(n), AlwaysBuyEngine(), _minimal_ind()).run(candles, 0)
        for ep in result.equity_curve:
            assert ep.base_value == ep.base_balance * ep.close_price
