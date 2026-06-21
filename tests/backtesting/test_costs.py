"""Tests for fee and slippage simulation correctness."""

from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.indicators.schemas import IndicatorConfig
from tests.backtesting.conftest import AlwaysBuyEngine, BuyThenSellEngine, make_candles

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


def _cfg(n: int, fee: str = "0", slip: str = "0", force_close: bool = True) -> BacktestConfig:
    return BacktestConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_ms=_BASE_TIME,
        end_ms=_BASE_TIME + n * _INTERVAL_MS,
        initial_capital=_D("10000"),
        fee_percentage=_D(fee),
        slippage_percentage=_D(slip),
        force_close_at_end=force_close,
    )


class TestFeeSimulation:
    def test_zero_fee_leaves_full_capital_available(self):
        n = 5
        candles = make_candles(n, price="100")
        result = BacktestEngine(_cfg(n, fee="0"), AlwaysBuyEngine(), _minimal_ind()).run(candles, 0)
        assert result.total_fees == _D("0")
        # At flat price, return should be exactly 0
        assert result.total_return_pct == _D("0")

    def test_fee_is_charged_on_entry(self):
        n = 15
        candles = make_candles(n, price="100")
        result = BacktestEngine(
            _cfg(n, fee="0.1", force_close=True), AlwaysBuyEngine(), _minimal_ind()
        ).run(candles, 0)
        # At least one trade should have non-zero entry fee
        assert any(t.entry_fee > _D("0") for t in result.trades)

    def test_fee_is_charged_on_exit(self):
        n = 15
        candles = make_candles(n, price="100")
        result = BacktestEngine(
            _cfg(n, fee="0.1", force_close=True), AlwaysBuyEngine(), _minimal_ind()
        ).run(candles, 0)
        assert any(t.exit_fee > _D("0") for t in result.trades)

    def test_total_fees_sum_of_all_trade_fees(self):
        n = 15
        candles = make_candles(n, price="100")
        result = BacktestEngine(_cfg(n, fee="0.1"), BuyThenSellEngine(), _minimal_ind()).run(
            candles, 0
        )
        expected = sum(t.entry_fee + t.exit_fee for t in result.trades)
        assert result.total_fees == expected

    def test_fee_reduces_quantity_on_entry(self):
        """With fee, quantity = (capital - fee) / price < capital / price."""
        n = 10
        candles = make_candles(n, price="100")
        r_no_fee = BacktestEngine(_cfg(n, fee="0"), AlwaysBuyEngine(), _minimal_ind()).run(
            candles, 0
        )
        r_fee = BacktestEngine(_cfg(n, fee="0.1"), AlwaysBuyEngine(), _minimal_ind()).run(
            candles, 0
        )
        if r_no_fee.trades and r_fee.trades:
            assert r_fee.trades[0].quantity < r_no_fee.trades[0].quantity

    def test_fee_convention_buy_deducted_before_conversion(self):
        """Verify: fee = capital * fee_rate; qty = (capital - fee) / price."""
        n = 10
        candles = make_candles(n, price="100")
        fee_pct = _D("0.1")
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=fee_pct,
            slippage_percentage=_D("0"),
            force_close_at_end=True,
        )
        result = BacktestEngine(cfg, AlwaysBuyEngine(), _minimal_ind()).run(candles, 0)
        if result.trades:
            t = result.trades[0]
            expected_fee = t.capital_at_entry * (fee_pct / _D("100"))
            assert t.entry_fee == expected_fee
            expected_qty = (t.capital_at_entry - expected_fee) / t.entry_exec_price
            assert t.quantity == expected_qty

    def test_fee_convention_sell_deducted_from_gross(self):
        """Verify: gross = qty * price; fee = gross * fee_rate; net = gross - fee."""
        n = 10
        candles = make_candles(n, price="100")
        fee_pct = _D("0.1")
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=fee_pct,
            slippage_percentage=_D("0"),
            force_close_at_end=True,
        )
        result = BacktestEngine(cfg, AlwaysBuyEngine(), _minimal_ind()).run(candles, 0)
        if result.trades:
            t = result.trades[0]
            gross = t.quantity * t.exit_exec_price
            expected_exit_fee = gross * (fee_pct / _D("100"))
            assert t.exit_fee == expected_exit_fee


class TestSlippageSimulation:
    def test_zero_slippage_executes_at_open_price(self):
        n = 10
        candles = make_candles(n, price="100")
        result = BacktestEngine(_cfg(n, slip="0"), AlwaysBuyEngine(), _minimal_ind()).run(
            candles, 0
        )
        for t in result.trades:
            if not t.is_forced_close:
                # Entry should be at open price (candle.open = 100)
                assert t.entry_exec_price == _D("100")

    def test_buy_slippage_increases_entry_price(self):
        n = 10
        candles = make_candles(n, price="100")
        r_no_slip = BacktestEngine(_cfg(n, slip="0"), AlwaysBuyEngine(), _minimal_ind()).run(
            candles, 0
        )
        r_slip = BacktestEngine(_cfg(n, slip="0.1"), AlwaysBuyEngine(), _minimal_ind()).run(
            candles, 0
        )
        if r_no_slip.trades and r_slip.trades:
            assert r_slip.trades[0].entry_exec_price > r_no_slip.trades[0].entry_exec_price

    def test_sell_slippage_decreases_exit_price(self):
        n = 15
        candles = make_candles(n, price="100")
        r_no_slip = BacktestEngine(_cfg(n, slip="0"), BuyThenSellEngine(), _minimal_ind()).run(
            candles, 0
        )
        r_slip = BacktestEngine(_cfg(n, slip="0.1"), BuyThenSellEngine(), _minimal_ind()).run(
            candles, 0
        )
        # Compare non-forced trades
        no_slip_trades = [t for t in r_no_slip.trades if not t.is_forced_close]
        slip_trades = [t for t in r_slip.trades if not t.is_forced_close]
        if no_slip_trades and slip_trades:
            assert slip_trades[0].exit_exec_price < no_slip_trades[0].exit_exec_price

    def test_slippage_adverse_reduces_return(self):
        n = 15
        candles = make_candles(n, price="100")
        r_no_slip = BacktestEngine(_cfg(n, slip="0"), AlwaysBuyEngine(), _minimal_ind()).run(
            candles, 0
        )
        r_slip = BacktestEngine(_cfg(n, slip="0.5"), AlwaysBuyEngine(), _minimal_ind()).run(
            candles, 0
        )
        assert r_slip.final_equity < r_no_slip.final_equity

    def test_forced_close_uses_close_price_with_slippage(self):
        n = 10
        candles = make_candles(n, price="100")
        slip = _D("0.05")
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + n * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=slip,
            force_close_at_end=True,
        )
        result = BacktestEngine(cfg, AlwaysBuyEngine(), _minimal_ind()).run(candles, 0)
        forced = [t for t in result.trades if t.is_forced_close]
        if forced:
            last_candle = candles[-1]
            expected_price = last_candle.close * (_D("1") - slip / _D("100"))
            assert forced[0].exit_exec_price == expected_price
