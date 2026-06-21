"""Tests for pure execution price and fee calculation functions."""

from decimal import Decimal

from app.backtesting.execution import (
    buy_and_hold_return_pct,
    buy_exec_price,
    compute_buy,
    compute_sell,
    forced_close_price,
    sell_exec_price,
)

_D = Decimal


class TestBuyExecPrice:
    def test_zero_slippage(self):
        assert buy_exec_price(_D("100"), _D("0")) == _D("100")

    def test_positive_slippage_increases_price(self):
        result = buy_exec_price(_D("100"), _D("0.001"))
        assert result == _D("100.1")

    def test_result_higher_than_open(self):
        result = buy_exec_price(_D("50000"), _D("0.0005"))
        assert result > _D("50000")


class TestSellExecPrice:
    def test_zero_slippage(self):
        assert sell_exec_price(_D("100"), _D("0")) == _D("100")

    def test_positive_slippage_decreases_price(self):
        result = sell_exec_price(_D("100"), _D("0.001"))
        assert result == _D("99.9")

    def test_result_lower_than_open(self):
        result = sell_exec_price(_D("50000"), _D("0.0005"))
        assert result < _D("50000")


class TestForcedClosePrice:
    def test_zero_slippage(self):
        assert forced_close_price(_D("100"), _D("0")) == _D("100")

    def test_positive_slippage_decreases_price(self):
        result = forced_close_price(_D("100"), _D("0.001"))
        assert result == _D("99.9")


class TestComputeBuy:
    def test_zero_fee(self):
        qty, fee = compute_buy(_D("1000"), _D("100"), _D("0"))
        assert qty == _D("10")
        assert fee == _D("0")

    def test_with_fee(self):
        # fee = 1000 * 0.001 = 1; qty = (1000 - 1) / 100 = 9.99
        qty, fee = compute_buy(_D("1000"), _D("100"), _D("0.001"))
        assert fee == _D("1")
        assert qty == _D("9.99")

    def test_capital_fully_consumed(self):
        qty, fee = compute_buy(_D("1000"), _D("200"), _D("0.001"))
        # qty * exec_price + fee = total capital
        assert qty * _D("200") + fee == _D("1000")

    def test_result_is_decimal(self):
        qty, fee = compute_buy(_D("5000"), _D("250"), _D("0.002"))
        assert isinstance(qty, Decimal)
        assert isinstance(fee, Decimal)


class TestComputeSell:
    def test_zero_fee(self):
        net, fee = compute_sell(_D("10"), _D("100"), _D("0"))
        assert net == _D("1000")
        assert fee == _D("0")

    def test_with_fee(self):
        # gross = 10 * 100 = 1000; fee = 1000 * 0.001 = 1; net = 999
        net, fee = compute_sell(_D("10"), _D("100"), _D("0.001"))
        assert fee == _D("1")
        assert net == _D("999")

    def test_net_plus_fee_equals_gross(self):
        qty, exec_price, fee_rate = _D("5"), _D("200"), _D("0.002")
        net, fee = compute_sell(qty, exec_price, fee_rate)
        gross = qty * exec_price
        assert net + fee == gross

    def test_result_is_decimal(self):
        net, fee = compute_sell(_D("3"), _D("150"), _D("0.001"))
        assert isinstance(net, Decimal)
        assert isinstance(fee, Decimal)


class TestBuyAndHoldReturnPct:
    def test_flat_market_with_zero_costs(self):
        result = buy_and_hold_return_pct(
            initial_capital=_D("1000"),
            first_open=_D("100"),
            last_close=_D("100"),
            fee_rate=_D("0"),
            slippage_rate=_D("0"),
        )
        assert result == _D("0")

    def test_doubling_market_with_zero_costs(self):
        result = buy_and_hold_return_pct(
            initial_capital=_D("1000"),
            first_open=_D("100"),
            last_close=_D("200"),
            fee_rate=_D("0"),
            slippage_rate=_D("0"),
        )
        assert result == _D("100")

    def test_costs_reduce_return(self):
        no_cost = buy_and_hold_return_pct(
            initial_capital=_D("1000"),
            first_open=_D("100"),
            last_close=_D("200"),
            fee_rate=_D("0"),
            slippage_rate=_D("0"),
        )
        with_cost = buy_and_hold_return_pct(
            initial_capital=_D("1000"),
            first_open=_D("100"),
            last_close=_D("200"),
            fee_rate=_D("0.001"),
            slippage_rate=_D("0.001"),
        )
        assert with_cost < no_cost

    def test_returns_decimal(self):
        result = buy_and_hold_return_pct(
            initial_capital=_D("1000"),
            first_open=_D("100"),
            last_close=_D("150"),
            fee_rate=_D("0.001"),
            slippage_rate=_D("0.001"),
        )
        assert isinstance(result, Decimal)
