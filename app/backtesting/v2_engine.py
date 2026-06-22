"""V2BacktestEngine — deterministic backtest engine with risk-based exits.

Exit evaluation order per candle (when position is open):
  1. Stop-loss   : candle.low  <= current_stop_price → intrabar exit
  2. Take-profit : candle.high >= take_profit_price  → intrabar exit
  3. Ambiguity   : both SL and TP triggered in same candle
                   → conservative SL-first policy + AMBIGUOUS_INTRABAR_STOP_FIRST
  4. Trailing stop update: raise current_stop_price based on candle.high (no look-ahead)
  5. Max holding : candles_held >= maximum_holding_candles → exit at candle CLOSE
  6. Crossover   : bearish-crossover signal queued for NEXT candle OPEN (same as V1)

Entry rules are identical to V1:
  bullish crossover + price above EMA-long + RSI in range + volume.
  Signal at CLOSE of candle i → execute at OPEN of candle i+1 (no look-ahead).

Position allocation:
  Only position_allocation_percentage% of available quote_balance is deployed per trade.
  Undeployed capital remains in quote_balance and contributes to total equity.

Execution prices (all sells use adverse slippage):
  Stop-loss   : stop_price  * (1 − slippage_rate)
  Take-profit : tp_price    * (1 − slippage_rate)
  Time exit   : candle.close * (1 − slippage_rate)
  Crossover   : candle.open  * (1 − slippage_rate)
  Buy         : candle.open  * (1 + slippage_rate)

PAPER/TEST only — no real orders, no real capital at risk.
Past results do NOT predict future performance.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from app.backtesting.config import BacktestConfig
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
from app.backtesting.risk_exit_config import RiskExitConfig
from app.backtesting.schemas import BacktestResult, BacktestTrade, EquityPoint
from app.indicators.calculator import IndicatorCalculator
from app.indicators.schemas import IndicatorConfig
from app.models.candle import Candle
from app.strategy.engine import StrategyEngine
from app.strategy.reasons import ReasonCode
from app.strategy.schemas import PositionContext, StrategyAction

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# Internal state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class V2OpenPosition:
    """Mutable state for a V2 open long position with risk management levels."""

    # Entry tracking
    signal_time: int  # open_time of the BUY signal candle
    exec_time: int  # open_time of the execution candle
    exec_price: Decimal  # entry price (with adverse slippage)
    fee: Decimal  # entry fee paid (quote)
    quantity: Decimal  # base asset held
    capital_committed: Decimal  # quote capital deployed for this trade
    entry_reasons: tuple[str, ...]

    # Risk levels (computed at entry, from signal-candle ATR — no look-ahead)
    initial_stop_price: Decimal  # entry_price − ATR × atr_stop_multiplier
    current_stop_price: Decimal  # raised by trailing stop; never lowered
    take_profit_price: Decimal | None  # None when use_take_profit=False
    risk_per_unit: Decimal  # entry_price − initial_stop_price
    entry_atr: Decimal  # ATR from the signal candle
    stop_distance_pct: Decimal  # (entry − stop) / entry × 100

    # Dynamic counters
    candles_held: int = 0
    trailing_activated: bool = False
    highest_price_seen: Decimal = field(default_factory=lambda: Decimal("0"))


@dataclass
class _V2PortfolioState:
    """Mutable portfolio state for the V2 engine."""

    initial_capital: Decimal
    quote_balance: Decimal
    base_balance: Decimal = field(default_factory=lambda: Decimal("0"))
    open_position: V2OpenPosition | None = None
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    next_trade_id: int = 1
    total_candles_evaluated: int = 0
    candles_with_position: int = 0
    peak_equity: Decimal = field(init=False)

    def __post_init__(self) -> None:
        self.peak_equity = self.quote_balance

    def current_equity(self, price: Decimal) -> Decimal:
        """Total portfolio value: undeployed quote + base asset at current price."""
        return self.quote_balance + self.base_balance * price


# ---------------------------------------------------------------------------
# V2BacktestEngine
# ---------------------------------------------------------------------------


class V2BacktestEngine:
    """V2 backtesting engine with risk-based exits.

    Reads candles from memory — no DB, no Binance API, no real orders.
    PAPER/TEST only.
    """

    def __init__(
        self,
        config: BacktestConfig,
        risk_exit_config: RiskExitConfig,
        strategy_engine: StrategyEngine,
        indicator_config: IndicatorConfig | None = None,
    ) -> None:
        self.config = config
        self.risk_exit_config = risk_exit_config
        self.strategy_engine = strategy_engine
        self.indicator_config = indicator_config or IndicatorConfig()

    def run(
        self,
        all_candles: list[Candle],
        warmup_len: int,
    ) -> BacktestResult:
        """Run the V2 backtest and return the complete result.

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

        calculator = IndicatorCalculator(self.indicator_config)
        all_results = calculator.calculate(all_candles)
        eval_results = all_results[warmup_len:]

        portfolio = _V2PortfolioState(
            initial_capital=self.config.initial_capital,
            quote_balance=self.config.initial_capital,
        )
        completed_trades: list[BacktestTrade] = []
        equity_points: list[EquityPoint] = []

        pending_action: StrategyAction | None = None
        pending_signal_time: int | None = None
        pending_reasons: tuple[str, ...] = ()
        pending_atr: Decimal | None = None

        for i, (candle, result) in enumerate(zip(eval_candles, eval_results, strict=False)):
            is_last = i == len(eval_candles) - 1

            # ---- Execute pending signal at THIS candle's OPEN ----
            if pending_action is not None and pending_signal_time is not None:
                if pending_action == StrategyAction.BUY and portfolio.open_position is None:
                    _v2_execute_buy(
                        portfolio,
                        candle,
                        self.config,
                        self.risk_exit_config,
                        pending_signal_time,
                        pending_reasons,
                        pending_atr,
                    )
                elif pending_action == StrategyAction.SELL and portfolio.open_position is not None:
                    trade = _v2_execute_crossover_sell(
                        portfolio,
                        candle,
                        self.config,
                        pending_signal_time,
                        pending_reasons,
                    )
                    completed_trades.append(trade)
                pending_action = None
                pending_signal_time = None
                pending_reasons = ()
                pending_atr = None

            # ---- Candle counters ----
            portfolio.total_candles_evaluated += 1
            if portfolio.open_position is not None:
                portfolio.candles_with_position += 1
                portfolio.open_position.candles_held += 1

            # ---- Check V2 exits (SL / TP / trailing / time) ----
            if portfolio.open_position is not None:
                intrabar_trade = _check_v2_exits(
                    portfolio, candle, self.config, self.risk_exit_config
                )
                if intrabar_trade is not None:
                    completed_trades.append(intrabar_trade)

            # ---- Force close at last candle ----
            if is_last and self.config.force_close_at_end and portfolio.open_position is not None:
                trade = _v2_force_close(portfolio, candle, self.config)
                completed_trades.append(trade)

            # ---- Record equity point ----
            equity = portfolio.current_equity(candle.close)
            portfolio.peak_equity = max(portfolio.peak_equity, equity)
            equity_points.append(_make_equity_point(candle, portfolio, equity))

            # ---- Generate signal at CLOSE of this candle (not if last) ----
            if not is_last:
                if portfolio.open_position is not None:
                    # Position open: check for secondary bearish-crossover exit
                    if self.risk_exit_config.use_bearish_crossover_exit:
                        decision = self.strategy_engine.evaluate(
                            result, PositionContext(has_open_long_position=True)
                        )
                        if decision.action == StrategyAction.SELL:
                            pending_action = StrategyAction.SELL
                            pending_signal_time = candle.open_time
                            pending_reasons = tuple(str(r) for r in decision.reasons)
                else:
                    # No position: evaluate buy conditions
                    decision = self.strategy_engine.evaluate(
                        result, PositionContext(has_open_long_position=False)
                    )
                    if decision.action == StrategyAction.BUY:
                        pending_action = StrategyAction.BUY
                        pending_signal_time = candle.open_time
                        pending_reasons = tuple(str(r) for r in decision.reasons)
                        pending_atr = result.atr  # ATR at signal candle — no look-ahead

        evaluated_count = sum(1 for r in eval_results if r.warmup_complete)

        bah_return = buy_and_hold_return_pct(
            initial_capital=self.config.initial_capital,
            first_open=eval_candles[0].open,
            last_close=eval_candles[-1].close,
            fee_rate=self.config.fee_rate,
            slippage_rate=self.config.slippage_rate,
        )

        return compute_metrics(
            config=self.config,
            completed_trades=completed_trades,
            equity_curve=equity_points,
            eval_candles=eval_candles,
            total_candles=len(eval_candles),
            evaluated_candles=evaluated_count,
            has_open_position_at_end=portfolio.open_position is not None,
            bah_return=bah_return,
            candles_with_position=portfolio.candles_with_position,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_equity_point(
    candle: Candle,
    portfolio: _V2PortfolioState,
    equity: Decimal,
) -> EquityPoint:
    """Build an EquityPoint from current portfolio state."""
    drawdown_pct = (
        (portfolio.peak_equity - equity) / portfolio.peak_equity * _HUNDRED
        if portfolio.peak_equity > _ZERO
        else _ZERO
    )
    return EquityPoint(
        open_time=candle.open_time,
        close_time=candle.close_time,
        close_price=candle.close,
        equity=equity,
        quote_balance=portfolio.quote_balance,
        base_balance=portfolio.base_balance,
        base_value=portfolio.base_balance * candle.close,
        drawdown_pct=drawdown_pct,
        peak_equity=portfolio.peak_equity,
        has_open_position=portfolio.open_position is not None,
    )


def _v2_execute_buy(
    portfolio: _V2PortfolioState,
    candle: Candle,
    config: BacktestConfig,
    risk_config: RiskExitConfig,
    signal_time: int,
    entry_reasons: tuple[str, ...],
    atr: Decimal | None,
) -> None:
    """Open a V2 long position at this candle's open.

    Deploys only position_allocation_percentage% of available quote_balance.
    Stop and take-profit levels are computed from the signal-candle ATR (no look-ahead).
    """
    exec_price = buy_exec_price(candle.open, config.slippage_rate)

    # Allocation: deploy only allocation_pct% of available capital
    alloc_fraction = risk_config.position_allocation_percentage / _HUNDRED
    capital_to_deploy = portfolio.quote_balance * alloc_fraction
    quantity, fee = compute_buy(capital_to_deploy, exec_price, config.fee_rate)

    # Risk levels — ATR from the signal candle (no look-ahead)
    entry_atr = atr if (atr is not None and atr > _ZERO) else _ZERO
    risk_exits_viable = risk_config.enabled and entry_atr > _ZERO

    if risk_exits_viable:
        risk_per_unit = entry_atr * risk_config.atr_stop_multiplier
        initial_stop = exec_price - risk_per_unit
    else:
        risk_per_unit = _ZERO
        initial_stop = _ZERO  # disabled

    stop_distance_pct = (
        (exec_price - initial_stop) / exec_price * _HUNDRED
        if exec_price > _ZERO and initial_stop > _ZERO
        else _ZERO
    )

    take_profit_price: Decimal | None = None
    if risk_exits_viable and risk_config.use_take_profit:
        take_profit_price = exec_price + risk_per_unit * risk_config.reward_to_risk_ratio

    portfolio.quote_balance -= capital_to_deploy
    portfolio.base_balance = quantity
    portfolio.total_fees += fee
    portfolio.open_position = V2OpenPosition(
        signal_time=signal_time,
        exec_time=candle.open_time,
        exec_price=exec_price,
        fee=fee,
        quantity=quantity,
        capital_committed=capital_to_deploy,
        entry_reasons=entry_reasons,
        initial_stop_price=initial_stop,
        current_stop_price=initial_stop,
        take_profit_price=take_profit_price,
        risk_per_unit=risk_per_unit,
        entry_atr=entry_atr,
        stop_distance_pct=stop_distance_pct,
        highest_price_seen=exec_price,
    )


def _check_v2_exits(
    portfolio: _V2PortfolioState,
    candle: Candle,
    config: BacktestConfig,
    risk_config: RiskExitConfig,
) -> BacktestTrade | None:
    """Check all V2 intrabar and time-based exits in deterministic order.

    Order: SL → TP → trailing update → max holding time.
    Crossover exit is handled separately in the main loop (signal-based).
    Returns a completed BacktestTrade if an exit triggered, else None.
    """
    pos = portfolio.open_position
    assert pos is not None

    # Step 1 & 2: Stop-loss and take-profit checks
    sl_active = risk_config.enabled and pos.current_stop_price > _ZERO
    tp_active = (
        risk_config.enabled and risk_config.use_take_profit and pos.take_profit_price is not None
    )
    sl_hit = sl_active and candle.low <= pos.current_stop_price
    tp_hit = (
        tp_active and pos.take_profit_price is not None and candle.high >= pos.take_profit_price
    )

    if sl_hit and tp_hit:
        # Ambiguous candle: both SL and TP hit — conservative SL-first policy
        return _build_and_close(
            portfolio,
            candle,
            config,
            exec_price=sell_exec_price(pos.current_stop_price, config.slippage_rate),
            exit_signal_time=None,
            exit_exec_time=candle.open_time,
            is_forced_close=False,
            exit_reasons=(
                str(ReasonCode.ATR_STOP_LOSS),
                str(ReasonCode.AMBIGUOUS_INTRABAR_STOP_FIRST),
            ),
        )

    if sl_hit:
        return _build_and_close(
            portfolio,
            candle,
            config,
            exec_price=sell_exec_price(pos.current_stop_price, config.slippage_rate),
            exit_signal_time=None,
            exit_exec_time=candle.open_time,
            is_forced_close=False,
            exit_reasons=(str(ReasonCode.ATR_STOP_LOSS),),
        )

    if tp_hit:
        assert pos.take_profit_price is not None
        return _build_and_close(
            portfolio,
            candle,
            config,
            exec_price=sell_exec_price(pos.take_profit_price, config.slippage_rate),
            exit_signal_time=None,
            exit_exec_time=candle.open_time,
            is_forced_close=False,
            exit_reasons=(str(ReasonCode.RISK_REWARD_TAKE_PROFIT),),
        )

    # Step 3: Update trailing stop (position still open)
    if risk_config.trailing_stop_enabled and pos.entry_atr > _ZERO:
        _update_trailing_stop(pos, candle, risk_config)

    # Step 4: Max holding time — exit at candle CLOSE
    if (
        risk_config.maximum_holding_candles > 0
        and pos.candles_held >= risk_config.maximum_holding_candles
    ):
        return _build_and_close(
            portfolio,
            candle,
            config,
            exec_price=forced_close_price(candle.close, config.slippage_rate),
            exit_signal_time=None,
            exit_exec_time=candle.close_time,
            is_forced_close=False,
            exit_reasons=(str(ReasonCode.MAX_HOLDING_TIME),),
        )

    return None


def _update_trailing_stop(
    pos: V2OpenPosition,
    candle: Candle,
    risk_config: RiskExitConfig,
) -> None:
    """Raise the trailing stop based on candle high. Stop never moves down."""
    if candle.high > pos.highest_price_seen:
        pos.highest_price_seen = candle.high

    activation_price = pos.exec_price + risk_config.trailing_activation_r * pos.risk_per_unit
    if pos.highest_price_seen >= activation_price:
        pos.trailing_activated = True

    if pos.trailing_activated:
        new_stop = pos.highest_price_seen - risk_config.trailing_distance_atr * pos.entry_atr
        if new_stop > pos.current_stop_price:
            pos.current_stop_price = new_stop


def _v2_execute_crossover_sell(
    portfolio: _V2PortfolioState,
    candle: Candle,
    config: BacktestConfig,
    signal_time: int,
    exit_reasons: tuple[str, ...],
) -> BacktestTrade:
    """Close the position at this candle's OPEN (crossover signal from prior close)."""
    exec_price = sell_exec_price(candle.open, config.slippage_rate)
    reasons = exit_reasons if exit_reasons else (str(ReasonCode.BEARISH_CROSSOVER),)
    return _build_and_close(
        portfolio,
        candle,
        config,
        exec_price=exec_price,
        exit_signal_time=signal_time,
        exit_exec_time=candle.open_time,
        is_forced_close=False,
        exit_reasons=reasons,
    )


def _v2_force_close(
    portfolio: _V2PortfolioState,
    candle: Candle,
    config: BacktestConfig,
) -> BacktestTrade:
    """Force close at candle CLOSE with adverse slippage (end of backtest period)."""
    exec_price = forced_close_price(candle.close, config.slippage_rate)
    return _build_and_close(
        portfolio,
        candle,
        config,
        exec_price=exec_price,
        exit_signal_time=None,
        exit_exec_time=candle.close_time,
        is_forced_close=True,
        exit_reasons=(str(ReasonCode.FORCED_END_OF_BACKTEST),),
    )


def _build_and_close(
    portfolio: _V2PortfolioState,
    candle: Candle,
    config: BacktestConfig,
    exec_price: Decimal,
    exit_signal_time: int | None,
    exit_exec_time: int,
    is_forced_close: bool,
    exit_reasons: tuple[str, ...],
) -> BacktestTrade:
    """Build a BacktestTrade and update portfolio state. Closes the open position."""
    pos = portfolio.open_position
    assert pos is not None, "_build_and_close called with no open position"

    net_proceeds, fee = compute_sell(pos.quantity, exec_price, config.fee_rate)
    gross_pnl = (exec_price - pos.exec_price) * pos.quantity
    net_pnl = gross_pnl - pos.fee - fee
    return_pct = net_pnl / pos.capital_committed * _HUNDRED

    trade = BacktestTrade(
        trade_id=portfolio.next_trade_id,
        entry_signal_time=pos.signal_time,
        entry_exec_time=pos.exec_time,
        entry_exec_price=pos.exec_price,
        entry_fee=pos.fee,
        quantity=pos.quantity,
        exit_signal_time=exit_signal_time,
        exit_exec_time=exit_exec_time,
        exit_exec_price=exec_price,
        exit_fee=fee,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        return_pct=return_pct,
        is_forced_close=is_forced_close,
        capital_at_entry=pos.capital_committed,
        entry_reasons=pos.entry_reasons,
        exit_reasons=exit_reasons,
    )

    portfolio.quote_balance += net_proceeds
    portfolio.base_balance = _ZERO
    portfolio.total_fees += fee
    portfolio.open_position = None
    portfolio.next_trade_id += 1

    return trade
