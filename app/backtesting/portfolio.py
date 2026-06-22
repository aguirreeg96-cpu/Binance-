"""Mutable portfolio state for the backtesting engine.

Tracks quote/base balances, open position, and running counters.
Does not create BacktestTrade objects — that is the engine's responsibility.

Spot long-only invariants:
  - quote_balance >= 0 at all times
  - base_balance >= 0 at all times
  - At most one open position at a time
  - Single long positions only, no derivatives, no borrowing
"""

from dataclasses import dataclass, field
from decimal import Decimal

from app.strategy.schemas import PositionContext


@dataclass
class _OpenPosition:
    """Internal state for a currently open long position."""

    signal_time: int  # open_time of the BUY signal candle
    exec_time: int  # open_time of the execution candle
    exec_price: Decimal  # entry price (with adverse slippage)
    fee: Decimal  # entry fee paid (quote)
    quantity: Decimal  # base asset held
    capital_committed: Decimal  # quote capital used to open this position
    entry_reasons: tuple[str, ...] = ()  # reason codes from the BUY decision


@dataclass
class PortfolioState:
    """Mutable portfolio state for one backtest simulation.

    Initialised with all capital in quote_balance and no open position.
    """

    initial_capital: Decimal
    quote_balance: Decimal
    base_balance: Decimal = field(default_factory=lambda: Decimal("0"))
    open_position: _OpenPosition | None = None
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    next_trade_id: int = 1
    total_candles_evaluated: int = 0
    candles_with_position: int = 0
    peak_equity: Decimal = field(init=False)

    def __post_init__(self) -> None:
        self.peak_equity = self.quote_balance

    def current_equity(self, price: Decimal) -> Decimal:
        """Total portfolio value at the given base asset price."""
        return self.quote_balance + self.base_balance * price

    def get_position_context(self) -> PositionContext:
        """Return current position state for the strategy engine."""
        if self.open_position is None:
            return PositionContext(has_open_long_position=False)
        return PositionContext(
            has_open_long_position=True,
            entry_price=self.open_position.exec_price,
            entry_time=self.open_position.exec_time,
        )
