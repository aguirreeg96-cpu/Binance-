"""BacktestEngine — deterministic, look-ahead-free backtesting engine.

Execution model (no look-ahead bias):
  Signal at close of candle i → execute at open of candle i+1.
  A signal at the LAST candle is never executed (no next candle),
  unless force_close_at_end=True, in which case the position is closed
  at the LAST candle's CLOSE price (with adverse slippage).

Spot long-only:
  BUY  → open a long position at next candle's open (with adverse slippage).
  SELL → close the long position at next candle's open (with adverse slippage).
  Spot BUY/SELL only — no derivatives, no borrowing.

Fee convention:
  Buy:  fee = capital * fee_rate
        quantity = (capital - fee) / exec_price
  Sell: gross = quantity * exec_price
        fee = gross * fee_rate
        net = gross - fee

Forced close (last candle):
  exec_price = last_candle.close * (1 - slippage_rate)
"""

from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.equity_curve import EquityCurveBuilder
from app.backtesting.exceptions import BacktestError, BacktestInsufficientDataError
from app.backtesting.execution import (
    buy_and_hold_return_pct,
    buy_exec_price,
    compute_buy,
    compute_sell,
    forced_close_price,
    sell_exec_price,
)
from app.backtesting.metrics import compute_metrics
from app.backtesting.portfolio import PortfolioState, _OpenPosition
from app.backtesting.schemas import BacktestResult, BacktestTrade
from app.indicators.calculator import IndicatorCalculator
from app.indicators.schemas import IndicatorConfig
from app.models.candle import Candle
from app.strategy.engine import StrategyEngine
from app.strategy.schemas import StrategyAction

_ZERO = Decimal("0")


class BacktestEngine:
    """Run a deterministic backtest over historical candles.

    Reads candles from memory — no DB sessions, no Binance API calls.
    No real orders are placed. PAPER/TEST only.
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy_engine: StrategyEngine,
        indicator_config: IndicatorConfig | None = None,
    ) -> None:
        self.config = config
        self.strategy_engine = strategy_engine
        self.indicator_config = indicator_config or IndicatorConfig()

    def run(
        self,
        all_candles: list[Candle],
        warmup_len: int,
    ) -> BacktestResult:
        """Run the backtest and return the complete result.

        all_candles: warmup prefix + eval candles, sorted ascending by open_time.
        warmup_len:  number of leading warmup candles (not evaluated).
        """
        if warmup_len < 0:
            raise BacktestError(f"warmup_len must be >= 0, got {warmup_len}")

        eval_candles = all_candles[warmup_len:]
        if not eval_candles:
            raise BacktestInsufficientDataError(
                f"No evaluation candles after warmup "
                f"(total={len(all_candles)}, warmup={warmup_len})"
            )

        # Calculate indicators for all candles in one O(n) pass
        calculator = IndicatorCalculator(self.indicator_config)
        all_results = calculator.calculate(all_candles)
        eval_results = all_results[warmup_len:]

        portfolio = PortfolioState(
            initial_capital=self.config.initial_capital,
            quote_balance=self.config.initial_capital,
        )
        completed_trades: list[BacktestTrade] = []
        equity_builder = EquityCurveBuilder()

        pending_action: StrategyAction | None = None
        pending_signal_time: int | None = None

        for i, (candle, result) in enumerate(zip(eval_candles, eval_results, strict=False)):
            is_last = i == len(eval_candles) - 1

            # ---- Execute pending signal at THIS candle's OPEN ----
            if pending_action is not None and pending_signal_time is not None:
                if pending_action == StrategyAction.BUY and portfolio.open_position is None:
                    _execute_buy(portfolio, candle, self.config, pending_signal_time)
                elif pending_action == StrategyAction.SELL and portfolio.open_position is not None:
                    trade = _execute_sell(portfolio, candle, self.config, pending_signal_time)
                    completed_trades.append(trade)
                pending_action = None
                pending_signal_time = None

            # ---- Update candle counters ----
            portfolio.total_candles_evaluated += 1
            if portfolio.open_position is not None:
                portfolio.candles_with_position += 1

            # ---- Generate signal at CLOSE of this candle ----
            decision = self.strategy_engine.evaluate(result, portfolio.get_position_context())

            # ---- Force close at LAST candle's CLOSE ----
            if is_last and self.config.force_close_at_end and portfolio.open_position is not None:
                trade = _execute_forced_close(portfolio, candle, self.config)
                completed_trades.append(trade)

            # ---- Record equity point (after all state changes) ----
            equity = portfolio.current_equity(candle.close)
            portfolio.peak_equity = max(portfolio.peak_equity, equity)
            equity_builder.append(candle, portfolio, equity)

            # ---- Queue signal for next candle (not if last) ----
            if not is_last and decision.action in (StrategyAction.BUY, StrategyAction.SELL):
                pending_action = decision.action
                pending_signal_time = candle.open_time

        bah_return = buy_and_hold_return_pct(
            initial_capital=self.config.initial_capital,
            first_open=eval_candles[0].open,
            last_close=eval_candles[-1].close,
            fee_rate=self.config.fee_rate,
            slippage_rate=self.config.slippage_rate,
        )
        evaluated_count = sum(1 for r in eval_results if r.warmup_complete)

        return compute_metrics(
            config=self.config,
            completed_trades=completed_trades,
            equity_curve=equity_builder.build(),
            eval_candles=eval_candles,
            total_candles=len(eval_candles),
            evaluated_candles=evaluated_count,
            has_open_position_at_end=portfolio.open_position is not None,
            bah_return=bah_return,
            candles_with_position=portfolio.candles_with_position,
        )


# ---------------------------------------------------------------------------
# Private execution helpers
# ---------------------------------------------------------------------------


def _execute_buy(
    portfolio: PortfolioState,
    candle: Candle,
    config: BacktestConfig,
    signal_time: int,
) -> None:
    """Open a long position at this candle's open with adverse slippage.

    All quote capital is committed; quote_balance becomes 0.
    """
    exec_price = buy_exec_price(candle.open, config.slippage_rate)
    capital = portfolio.quote_balance
    quantity, fee = compute_buy(capital, exec_price, config.fee_rate)

    portfolio.quote_balance = _ZERO
    portfolio.base_balance = quantity
    portfolio.total_fees += fee
    portfolio.open_position = _OpenPosition(
        signal_time=signal_time,
        exec_time=candle.open_time,
        exec_price=exec_price,
        fee=fee,
        quantity=quantity,
        capital_committed=capital,
    )


def _execute_sell(
    portfolio: PortfolioState,
    candle: Candle,
    config: BacktestConfig,
    signal_time: int,
) -> BacktestTrade:
    """Close the long position at this candle's open with adverse slippage.

    Returns the completed BacktestTrade.
    """
    pos = portfolio.open_position
    assert pos is not None, "_execute_sell called with no open position"

    exec_price = sell_exec_price(candle.open, config.slippage_rate)
    net_proceeds, fee = compute_sell(pos.quantity, exec_price, config.fee_rate)
    gross_pnl = (exec_price - pos.exec_price) * pos.quantity
    net_pnl = gross_pnl - pos.fee - fee
    return_pct = net_pnl / pos.capital_committed * Decimal("100")

    trade = BacktestTrade(
        trade_id=portfolio.next_trade_id,
        entry_signal_time=pos.signal_time,
        entry_exec_time=pos.exec_time,
        entry_exec_price=pos.exec_price,
        entry_fee=pos.fee,
        quantity=pos.quantity,
        exit_signal_time=signal_time,
        exit_exec_time=candle.open_time,
        exit_exec_price=exec_price,
        exit_fee=fee,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        return_pct=return_pct,
        is_forced_close=False,
        capital_at_entry=pos.capital_committed,
    )

    portfolio.quote_balance = net_proceeds
    portfolio.base_balance = _ZERO
    portfolio.total_fees += fee
    portfolio.open_position = None
    portfolio.next_trade_id += 1

    return trade


def _execute_forced_close(
    portfolio: PortfolioState,
    candle: Candle,
    config: BacktestConfig,
) -> BacktestTrade:
    """Force close the open position at this candle's CLOSE with adverse slippage.

    Used at the end of the backtest when force_close_at_end=True.
    Returns the completed BacktestTrade.
    """
    pos = portfolio.open_position
    assert pos is not None, "_execute_forced_close called with no open position"

    exec_price = forced_close_price(candle.close, config.slippage_rate)
    net_proceeds, fee = compute_sell(pos.quantity, exec_price, config.fee_rate)
    gross_pnl = (exec_price - pos.exec_price) * pos.quantity
    net_pnl = gross_pnl - pos.fee - fee
    return_pct = net_pnl / pos.capital_committed * Decimal("100")

    trade = BacktestTrade(
        trade_id=portfolio.next_trade_id,
        entry_signal_time=pos.signal_time,
        entry_exec_time=pos.exec_time,
        entry_exec_price=pos.exec_price,
        entry_fee=pos.fee,
        quantity=pos.quantity,
        exit_signal_time=None,
        exit_exec_time=candle.close_time,
        exit_exec_price=exec_price,
        exit_fee=fee,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        return_pct=return_pct,
        is_forced_close=True,
        capital_at_entry=pos.capital_committed,
    )

    portfolio.quote_balance = net_proceeds
    portfolio.base_balance = _ZERO
    portfolio.total_fees += fee
    portfolio.open_position = None
    portfolio.next_trade_id += 1

    return trade
