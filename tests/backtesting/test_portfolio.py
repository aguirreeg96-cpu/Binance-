"""Tests for PortfolioState."""

from decimal import Decimal

from app.backtesting.portfolio import PortfolioState, _OpenPosition

_D = Decimal


class TestPortfolioState:
    def _make(self, capital: str = "10000") -> PortfolioState:
        return PortfolioState(
            initial_capital=_D(capital),
            quote_balance=_D(capital),
        )

    def test_initial_state(self):
        p = self._make("10000")
        assert p.quote_balance == _D("10000")
        assert p.base_balance == _D("0")
        assert p.open_position is None
        assert p.total_fees == _D("0")
        assert p.next_trade_id == 1
        assert p.total_candles_evaluated == 0
        assert p.candles_with_position == 0

    def test_peak_equity_initialized_to_quote_balance(self):
        p = self._make("5000")
        assert p.peak_equity == _D("5000")

    def test_current_equity_no_position(self):
        p = self._make("10000")
        assert p.current_equity(_D("50000")) == _D("10000")

    def test_current_equity_with_position(self):
        p = self._make("0")
        p.base_balance = _D("2")
        assert p.current_equity(_D("5000")) == _D("10000")

    def test_current_equity_mixed(self):
        p = self._make("5000")
        p.base_balance = _D("1")
        assert p.current_equity(_D("3000")) == _D("8000")

    def test_get_position_context_no_position(self):
        p = self._make()
        ctx = p.get_position_context()
        assert not ctx.has_open_long_position
        assert ctx.entry_price is None

    def test_get_position_context_with_position(self):
        p = self._make()
        p.open_position = _OpenPosition(
            signal_time=1000,
            exec_time=2000,
            exec_price=_D("45000"),
            fee=_D("10"),
            quantity=_D("0.22"),
            capital_committed=_D("10000"),
        )
        ctx = p.get_position_context()
        assert ctx.has_open_long_position
        assert ctx.entry_price == _D("45000")
        assert ctx.entry_time == 2000

    def test_base_balance_defaults_to_zero(self):
        p = PortfolioState(initial_capital=_D("1000"), quote_balance=_D("1000"))
        assert p.base_balance == _D("0")

    def test_total_fees_defaults_to_zero(self):
        p = PortfolioState(initial_capital=_D("1000"), quote_balance=_D("1000"))
        assert p.total_fees == _D("0")
