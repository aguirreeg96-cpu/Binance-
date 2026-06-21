"""compute_metrics — aggregate engine outputs into BacktestResult.

All financial arithmetic uses Decimal.  No floats in financial fields.
No side effects, no DB, no network.
"""

from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.schemas import BacktestResult, BacktestTrade, EquityPoint
from app.models.candle import Candle

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


def compute_metrics(
    config: BacktestConfig,
    completed_trades: list[BacktestTrade],
    equity_curve: list[EquityPoint],
    eval_candles: list[Candle],
    total_candles: int,
    evaluated_candles: int,
    has_open_position_at_end: bool,
    bah_return: Decimal,
    candles_with_position: int,
) -> BacktestResult:
    """Derive all BacktestResult fields from raw engine outputs."""
    assert equity_curve, "equity_curve must not be empty"
    total_fees = sum((t.entry_fee + t.exit_fee for t in completed_trades), _ZERO)

    initial_capital = config.initial_capital
    final_equity = equity_curve[-1].equity
    total_return_pct = (final_equity - initial_capital) / initial_capital * _HUNDRED

    wins = [t for t in completed_trades if t.net_pnl > _ZERO]
    losses = [t for t in completed_trades if t.net_pnl <= _ZERO]

    win_rate_pct: Decimal | None = None
    avg_win_pct: Decimal | None = None
    avg_loss_pct: Decimal | None = None
    profit_factor: Decimal | None = None
    expectancy_pct: Decimal | None = None

    if completed_trades:
        win_rate_pct = Decimal(str(len(wins))) / Decimal(str(len(completed_trades))) * _HUNDRED

    if wins:
        avg_win_pct = sum((t.return_pct for t in wins), _ZERO) / Decimal(str(len(wins)))

    if losses:
        avg_loss_pct = sum((t.return_pct for t in losses), _ZERO) / Decimal(str(len(losses)))

    total_gross_wins: Decimal = sum((t.net_pnl for t in wins), _ZERO) if wins else _ZERO
    total_gross_losses: Decimal = abs(sum((t.net_pnl for t in losses), _ZERO)) if losses else _ZERO

    if total_gross_losses > _ZERO:
        profit_factor = total_gross_wins / total_gross_losses

    if win_rate_pct is not None and avg_win_pct is not None and avg_loss_pct is not None:
        wr = win_rate_pct / _HUNDRED
        expectancy_pct = wr * avg_win_pct + (_ONE - wr) * avg_loss_pct

    max_drawdown_pct = max((ep.drawdown_pct for ep in equity_curve), default=_ZERO)

    exposure_pct = (
        Decimal(str(candles_with_position)) / Decimal(str(total_candles)) * _HUNDRED
        if total_candles > 0
        else _ZERO
    )

    max_win_streak, max_loss_streak = _compute_streaks(completed_trades)

    return BacktestResult(
        config=config,
        trades=completed_trades,
        equity_curve=equity_curve,
        first_candle_open_time=eval_candles[0].open_time,
        last_candle_open_time=eval_candles[-1].open_time,
        total_candles=total_candles,
        evaluated_candles=evaluated_candles,
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        buy_and_hold_return_pct=bah_return,
        total_trades=len(completed_trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate_pct=win_rate_pct,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        profit_factor=profit_factor,
        expectancy_pct=expectancy_pct,
        max_drawdown_pct=max_drawdown_pct,
        exposure_pct=exposure_pct,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        total_fees=total_fees,
        has_open_position_at_end=has_open_position_at_end,
    )


def _compute_streaks(trades: list[BacktestTrade]) -> tuple[int, int]:
    """Return (max_win_streak, max_loss_streak) over completed trades."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for t in trades:
        if t.net_pnl > _ZERO:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses
