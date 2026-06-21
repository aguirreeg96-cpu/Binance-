"""Pure price and fee calculation functions for the backtesting engine.

All inputs and outputs are Decimal — no floats allowed in financial calculations.
These functions are stateless and have no side effects.

Fee convention:
  Buy:  fee = available_capital * fee_rate
        quantity = (available_capital - fee) / exec_price

  Sell: gross = quantity * exec_price
        fee = gross * fee_rate
        net = gross - fee

Slippage convention (adverse):
  Buy:  exec_price = open * (1 + slippage_rate)   — pays more
  Sell: exec_price = open * (1 - slippage_rate)   — receives less
  Forced close (at close price):
        exec_price = close * (1 - slippage_rate)
"""

from decimal import Decimal

_ONE = Decimal("1")
_HUNDRED = Decimal("100")


def buy_exec_price(open_price: Decimal, slippage_rate: Decimal) -> Decimal:
    """Buy execution price: open with adverse (higher) slippage."""
    return open_price * (_ONE + slippage_rate)


def sell_exec_price(open_price: Decimal, slippage_rate: Decimal) -> Decimal:
    """Sell execution price: open with adverse (lower) slippage."""
    return open_price * (_ONE - slippage_rate)


def forced_close_price(close_price: Decimal, slippage_rate: Decimal) -> Decimal:
    """Forced close execution price: last candle's close with adverse (lower) slippage."""
    return close_price * (_ONE - slippage_rate)


def compute_buy(
    available_capital: Decimal,
    exec_price: Decimal,
    fee_rate: Decimal,
) -> tuple[Decimal, Decimal]:
    """Compute buy order result.  Returns (quantity, fee).

    Fee is deducted from capital before converting to base asset so that
    the total spend (quantity * exec_price + fee) equals available_capital.
    """
    fee = available_capital * fee_rate
    quantity = (available_capital - fee) / exec_price
    return quantity, fee


def compute_sell(
    quantity: Decimal,
    exec_price: Decimal,
    fee_rate: Decimal,
) -> tuple[Decimal, Decimal]:
    """Compute sell order result.  Returns (net_proceeds, fee).

    Fee is deducted from gross proceeds so that the total received
    (net_proceeds) equals quantity * exec_price * (1 - fee_rate).
    """
    gross = quantity * exec_price
    fee = gross * fee_rate
    return gross - fee, fee


def buy_and_hold_return_pct(
    initial_capital: Decimal,
    first_open: Decimal,
    last_close: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> Decimal:
    """Compute buy-and-hold return percentage, including fees and slippage.

    Buy at first eval candle's open with adverse slippage and entry fee.
    Sell at last eval candle's close with adverse slippage and exit fee.
    """
    entry_price = buy_exec_price(first_open, slippage_rate)
    entry_fee = initial_capital * fee_rate
    quantity = (initial_capital - entry_fee) / entry_price

    exit_price = forced_close_price(last_close, slippage_rate)
    net_proceeds, _ = compute_sell(quantity, exit_price, fee_rate)

    return (net_proceeds - initial_capital) / initial_capital * _HUNDRED
