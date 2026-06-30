"""Donchian breakout engine — Stage 6.0.

Entry signal: close > N-bar Donchian high (previous bars only, current excluded),
with EMA-trend filter (close > EMA-period) and EMA-slope filter (EMA[i] > EMA[i-k]).

Stop management: ATR trailing stop using Wilder smoothing.
Conservative intrabar order per candle:
  1. Check stop against PREVIOUS stop level (before updating).
  2. If not hit, update trailing stop from current candle high (stop never decreases).
  3. Donchian channel exit checked at candle close; queued for next open.

Gap handling: if candle.open < stop_level → exit at candle.open (gap-down fills at open).

Signal timing: signal at CLOSE of candle i → execution at OPEN of candle i+1.
No pyramiding; allocation_pct% of available quote per trade.

PAPER/TEST only. No real orders, no real capital.
Past results do NOT predict future performance.
"""

from __future__ import annotations

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
from app.backtesting.schemas import BacktestResult, BacktestTrade, EquityPoint
from app.models.candle import Candle

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# DonchianConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DonchianConfig:
    """Frozen parameters for one Donchian breakout configuration.

    Pre-registered at Stage 6.0; never modified after results are seen.
    """

    entry_lookback: int
    exit_lookback: int
    atr_period: int
    atr_multiplier: Decimal
    ema_period: int = 200
    ema_slope_lookback: int = 10
    allocation_pct: Decimal = Decimal("25")


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _DonchianPosition:
    signal_time: int
    exec_time: int
    exec_price: Decimal
    fee: Decimal
    quantity: Decimal
    capital_committed: Decimal
    entry_reasons: tuple[str, ...]
    initial_stop_price: Decimal
    current_stop_price: Decimal
    highest_price_seen: Decimal
    candles_held: int = 0


@dataclass
class _DonchianPortfolio:
    initial_capital: Decimal
    quote_balance: Decimal
    base_balance: Decimal = field(default_factory=lambda: Decimal("0"))
    open_position: _DonchianPosition | None = None
    next_trade_id: int = 1
    total_candles_evaluated: int = 0
    candles_with_position: int = 0
    peak_equity: Decimal = field(init=False)

    def __post_init__(self) -> None:
        self.peak_equity = self.quote_balance

    def current_equity(self, price: Decimal) -> Decimal:
        return self.quote_balance + self.base_balance * price


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def _compute_atr(candles: list[Candle], period: int) -> list[Decimal | None]:
    """Wilder's ATR. Seeded with SMA of TR[1..period]; Wilder smoothing thereafter."""
    n = len(candles)
    result: list[Decimal | None] = [None] * n
    if n <= period:
        return result

    # TR[i] uses candle[i] vs candle[i-1].close; index 0 has no predecessor.
    trs: list[Decimal] = [_ZERO]
    for i in range(1, n):
        h = candles[i].high
        lo = candles[i].low
        pc = candles[i - 1].close
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))

    seed = sum(trs[1 : period + 1], _ZERO) / Decimal(str(period))
    result[period] = seed
    p = Decimal(str(period))
    for i in range(period + 1, n):
        prev = result[i - 1]
        if prev is not None:
            result[i] = (prev * (p - _ONE) + trs[i]) / p

    return result


def _compute_ema(prices: list[Decimal], period: int) -> list[Decimal | None]:
    """EMA seeded with SMA of first `period` prices; exponential thereafter."""
    n = len(prices)
    result: list[Decimal | None] = [None] * n
    if n < period:
        return result

    k = Decimal("2") / Decimal(str(period + 1))
    sma = sum(prices[:period], _ZERO) / Decimal(str(period))
    result[period - 1] = sma
    for i in range(period, n):
        prev = result[i - 1]
        if prev is not None:
            result[i] = prices[i] * k + prev * (_ONE - k)

    return result


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DonchianBreakoutEngine:
    """Donchian breakout backtesting engine.

    Reads candles from memory. No DB, no Binance API, no real orders.
    PAPER/TEST only.
    """

    def __init__(
        self,
        config: BacktestConfig,
        donchian_config: DonchianConfig,
    ) -> None:
        self.config = config
        self.donchian_config = donchian_config

    def run(
        self,
        all_candles: list[Candle],
        warmup_len: int,
    ) -> BacktestResult:
        """Run backtest and return result.

        all_candles: warmup prefix + eval candles, ascending by open_time.
        warmup_len:  number of leading warmup candles (not evaluated for trades).
        """
        if warmup_len < 0:
            raise BacktestError(f"warmup_len must be >= 0, got {warmup_len}")

        eval_candles = all_candles[warmup_len:]
        if not eval_candles:
            raise BacktestInsufficientDataError(
                f"No evaluation candles " f"(total={len(all_candles)}, warmup={warmup_len})"
            )

        dcfg = self.donchian_config
        closes = [c.close for c in all_candles]
        highs = [c.high for c in all_candles]
        lows = [c.low for c in all_candles]

        atrs = _compute_atr(all_candles, dcfg.atr_period)
        emas = _compute_ema(closes, dcfg.ema_period)

        portfolio = _DonchianPortfolio(
            initial_capital=self.config.initial_capital,
            quote_balance=self.config.initial_capital,
        )
        completed_trades: list[BacktestTrade] = []
        equity_points: list[EquityPoint] = []

        pending_buy = False
        pending_sell = False
        pending_signal_time: int | None = None
        pending_atr: Decimal | None = None

        for eval_i, candle in enumerate(eval_candles):
            all_i = warmup_len + eval_i
            is_last = eval_i == len(eval_candles) - 1
            atr_val = atrs[all_i]
            ema_val = emas[all_i]

            # ---- Execute pending signal at THIS candle's OPEN ----
            if pending_buy and portfolio.open_position is None:
                _execute_buy(
                    portfolio,
                    candle,
                    self.config,
                    dcfg,
                    signal_time=pending_signal_time or candle.open_time,
                    entry_atr=pending_atr,
                )
            elif pending_sell and portfolio.open_position is not None:
                trade = _build_and_close(
                    portfolio,
                    candle,
                    self.config,
                    exec_price=sell_exec_price(candle.open, self.config.slippage_rate),
                    exit_signal_time=pending_signal_time,
                    exit_exec_time=candle.open_time,
                    is_forced_close=False,
                    exit_reasons=("DONCHIAN_CHANNEL_EXIT",),
                )
                completed_trades.append(trade)
            pending_buy = False
            pending_sell = False
            pending_signal_time = None
            pending_atr = None

            # ---- Candle counters ----
            portfolio.total_candles_evaluated += 1
            if portfolio.open_position is not None:
                portfolio.open_position.candles_held += 1
                portfolio.candles_with_position += 1

            # ---- Intrabar stop exit (conservative: check prev level, then update) ----
            if portfolio.open_position is not None:
                exit_trade = _check_stop_exit(portfolio, candle, self.config, dcfg, atr_val)
                if exit_trade is not None:
                    completed_trades.append(exit_trade)

            # ---- Force close at last candle ----
            if is_last and self.config.force_close_at_end and portfolio.open_position is not None:
                trade = _build_and_close(
                    portfolio,
                    candle,
                    self.config,
                    exec_price=forced_close_price(candle.close, self.config.slippage_rate),
                    exit_signal_time=None,
                    exit_exec_time=candle.close_time,
                    is_forced_close=True,
                    exit_reasons=("FORCED_END_OF_BACKTEST",),
                )
                completed_trades.append(trade)

            # ---- Record equity ----
            equity = portfolio.current_equity(candle.close)
            portfolio.peak_equity = max(portfolio.peak_equity, equity)
            equity_points.append(_make_equity_point(candle, portfolio, equity))

            # ---- Generate signal at CLOSE (if not last candle) ----
            if not is_last:
                if portfolio.open_position is None:
                    # Entry: close > Donchian high AND close > EMA AND EMA slope up
                    if ema_val is not None and atr_val is not None and all_i >= dcfg.entry_lookback:
                        don_high = max(highs[all_i - dcfg.entry_lookback : all_i])
                        ema_prev = (
                            emas[all_i - dcfg.ema_slope_lookback]
                            if all_i >= dcfg.ema_slope_lookback
                            else None
                        )
                        slope_ok = ema_prev is not None and ema_val > ema_prev
                        if candle.close > don_high and candle.close > ema_val and slope_ok:
                            pending_buy = True
                            pending_signal_time = candle.open_time
                            pending_atr = atr_val
                else:
                    # Donchian exit: close < Donchian low → queue sell for next open
                    if all_i >= dcfg.exit_lookback:
                        don_low = min(lows[all_i - dcfg.exit_lookback : all_i])
                        if candle.close < don_low:
                            pending_sell = True
                            pending_signal_time = candle.open_time

        evaluated_count = sum(
            1
            for ei in range(len(eval_candles))
            if atrs[warmup_len + ei] is not None and emas[warmup_len + ei] is not None
        )

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


def _check_stop_exit(
    portfolio: _DonchianPortfolio,
    candle: Candle,
    config: BacktestConfig,
    dcfg: DonchianConfig,
    atr_val: Decimal | None,
) -> BacktestTrade | None:
    """Stop-loss with gap handling. Conservative: check prev level, then update trailing."""
    pos = portfolio.open_position
    assert pos is not None
    stop = pos.current_stop_price

    if stop <= _ZERO:
        return None

    # Gap: open gapped below stop → exit at open (can't fill at stop)
    if candle.open < stop:
        return _build_and_close(
            portfolio,
            candle,
            config,
            exec_price=sell_exec_price(candle.open, config.slippage_rate),
            exit_signal_time=None,
            exit_exec_time=candle.open_time,
            is_forced_close=False,
            exit_reasons=("DONCHIAN_GAP_STOP",),
        )

    # Intrabar low hit stop
    if candle.low <= stop:
        return _build_and_close(
            portfolio,
            candle,
            config,
            exec_price=sell_exec_price(stop, config.slippage_rate),
            exit_signal_time=None,
            exit_exec_time=candle.open_time,
            is_forced_close=False,
            exit_reasons=("DONCHIAN_STOP_LOSS",),
        )

    # Stop not hit — update trailing (stop never decreases)
    if candle.high > pos.highest_price_seen:
        pos.highest_price_seen = candle.high
    if atr_val is not None and atr_val > _ZERO:
        new_stop = pos.highest_price_seen - dcfg.atr_multiplier * atr_val
        if new_stop > pos.current_stop_price:
            pos.current_stop_price = new_stop

    return None


def _execute_buy(
    portfolio: _DonchianPortfolio,
    candle: Candle,
    config: BacktestConfig,
    dcfg: DonchianConfig,
    signal_time: int,
    entry_atr: Decimal | None,
) -> None:
    """Open a long position at candle open with ATR-based initial stop."""
    exec_price = buy_exec_price(candle.open, config.slippage_rate)
    alloc_fraction = dcfg.allocation_pct / _HUNDRED
    capital_to_deploy = portfolio.quote_balance * alloc_fraction
    quantity, fee = compute_buy(capital_to_deploy, exec_price, config.fee_rate)

    atr = entry_atr if (entry_atr is not None and entry_atr > _ZERO) else _ZERO
    initial_stop = (exec_price - dcfg.atr_multiplier * atr) if atr > _ZERO else _ZERO

    portfolio.quote_balance -= capital_to_deploy
    portfolio.base_balance = quantity
    portfolio.open_position = _DonchianPosition(
        signal_time=signal_time,
        exec_time=candle.open_time,
        exec_price=exec_price,
        fee=fee,
        quantity=quantity,
        capital_committed=capital_to_deploy,
        entry_reasons=("DONCHIAN_BREAKOUT",),
        initial_stop_price=initial_stop,
        current_stop_price=initial_stop,
        highest_price_seen=exec_price,
    )


def _build_and_close(
    portfolio: _DonchianPortfolio,
    candle: Candle,
    config: BacktestConfig,
    exec_price: Decimal,
    exit_signal_time: int | None,
    exit_exec_time: int,
    is_forced_close: bool,
    exit_reasons: tuple[str, ...],
) -> BacktestTrade:
    """Build trade record and update portfolio. Closes the open position."""
    pos = portfolio.open_position
    assert pos is not None

    net_proceeds, fee = compute_sell(pos.quantity, exec_price, config.fee_rate)
    gross_pnl = (exec_price - pos.exec_price) * pos.quantity
    net_pnl = gross_pnl - pos.fee - fee
    return_pct = (
        net_pnl / pos.capital_committed * _HUNDRED if pos.capital_committed > _ZERO else _ZERO
    )

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
    portfolio.open_position = None
    portfolio.next_trade_id += 1

    return trade


def _make_equity_point(
    candle: Candle,
    portfolio: _DonchianPortfolio,
    equity: Decimal,
) -> EquityPoint:
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
