"""Backtesting result schemas.

All Decimal fields stay Decimal — serialisation happens in the API layer.
No ORM models, no HTTP, no DB sessions.

PAPER/TEST only — no real orders, no real capital at risk.
Past performance does not predict future results.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.backtesting.config import BacktestConfig


@dataclass(frozen=True)
class BacktestTrade:
    """A completed round-trip trade (buy then sell).

    Spot long-only: entry is always a buy, exit is always a sell.
    No short trades, no leverage, no margin.
    """

    trade_id: int
    entry_signal_time: int  # open_time of the BUY signal candle
    entry_exec_time: int  # open_time of the execution candle
    entry_exec_price: Decimal  # execution price (open + adverse slippage)
    entry_fee: Decimal  # quote fee paid on entry
    quantity: Decimal  # base asset quantity purchased

    exit_signal_time: int | None  # open_time of the SELL signal candle; None for forced close
    exit_exec_time: int  # open_time (normal) or close_time (forced) of exit candle
    exit_exec_price: Decimal  # execution price (open/close - adverse slippage)
    exit_fee: Decimal  # quote fee paid on exit

    gross_pnl: Decimal  # (exit_exec_price - entry_exec_price) * quantity
    net_pnl: Decimal  # gross_pnl - entry_fee - exit_fee
    return_pct: Decimal  # net_pnl / capital_at_entry * 100
    is_forced_close: bool  # True when closed by force_close_at_end
    capital_at_entry: Decimal  # quote capital committed to this trade

    # Reason codes captured from strategy decision (empty tuple if unavailable)
    entry_reasons: tuple[str, ...] = ()
    exit_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EquityPoint:
    """Portfolio snapshot at the close of one evaluated candle."""

    open_time: int
    close_time: int
    close_price: Decimal
    equity: Decimal  # quote_balance + base_balance * close_price
    quote_balance: Decimal
    base_balance: Decimal
    base_value: Decimal  # base_balance * close_price
    drawdown_pct: Decimal  # (peak_equity - equity) / peak_equity * 100
    peak_equity: Decimal
    has_open_position: bool


@dataclass(frozen=True)
class BacktestResult:
    """Complete, immutable result of a single backtest run.

    PAPER/TEST only — no real orders placed, no real capital at risk.
    Past performance does not predict future results.
    """

    config: BacktestConfig
    trades: list[BacktestTrade]
    equity_curve: list[EquityPoint]

    # Candle counts
    first_candle_open_time: int
    last_candle_open_time: int
    total_candles: int  # eval candles (excluding warmup prefix)
    evaluated_candles: int  # candles where warmup_complete=True

    # P&L
    initial_capital: Decimal
    final_equity: Decimal
    total_return_pct: Decimal
    buy_and_hold_return_pct: Decimal

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: Decimal | None  # None if no completed trades
    avg_win_pct: Decimal | None  # None if no winning trades
    avg_loss_pct: Decimal | None  # None if no losing trades
    profit_factor: Decimal | None  # None if no losing trades
    expectancy_pct: Decimal | None  # None if missing win/loss data

    # Risk
    max_drawdown_pct: Decimal
    exposure_pct: Decimal  # pct of eval candles with an open position

    # Streaks
    max_win_streak: int
    max_loss_streak: int

    # Costs
    total_fees: Decimal

    has_open_position_at_end: bool
