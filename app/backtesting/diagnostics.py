"""Backtest diagnostics — comprehensive post-run analysis.

Computes additional analytical layers on top of a completed BacktestResult:
cost sensitivity, MFE/MAE excursions, entry blocker counts, periodic
breakdowns, trade distribution statistics, and benchmark comparisons.

No DB, no network, no real orders.  PAPER/TEST only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from app.backtesting.engine import BacktestEngine
from app.backtesting.execution import buy_and_hold_return_pct
from app.backtesting.schemas import BacktestResult, BacktestTrade
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
# Frozen result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostScenarioResult:
    name: str
    fee_percentage: Decimal
    slippage_percentage: Decimal
    final_equity: Decimal
    return_pct: Decimal
    total_fees: Decimal
    estimated_slippage_cost: Decimal
    max_drawdown_pct: Decimal
    total_trades: int
    win_rate_pct: Decimal | None
    profit_factor: Decimal | None


@dataclass(frozen=True)
class ExitReasonSummary:
    exit_reason: str
    trade_count: int
    win_count: int
    win_rate_pct: Decimal | None
    gross_pnl: Decimal
    net_pnl: Decimal
    avg_net_pnl: Decimal | None
    avg_duration_candles: Decimal | None
    max_loss: Decimal
    max_gain: Decimal


@dataclass(frozen=True)
class MonthlyResult:
    year: int
    month: int
    trade_count: int
    net_pnl: Decimal
    total_fees: Decimal
    win_count: int
    win_rate_pct: Decimal | None


@dataclass(frozen=True)
class QuarterlyResult:
    year: int
    quarter: int
    trade_count: int
    net_pnl: Decimal
    total_fees: Decimal
    win_count: int
    win_rate_pct: Decimal | None


@dataclass(frozen=True)
class TradeExcursion:
    trade_id: int
    entry_exec_price: Decimal
    max_favorable_excursion_pct: Decimal
    max_adverse_excursion_pct: Decimal
    highest_price: Decimal
    lowest_price: Decimal
    candles_until_mfe: int
    candles_until_mae: int
    candles_in_trade: int


@dataclass(frozen=True)
class EntryBlockerCounts:
    warmup_incomplete: int
    no_bullish_crossover: int
    price_below_long_ema: int
    rsi_outside_buy_range: int
    insufficient_volume: int
    position_already_open: int
    bearish_crossover: int
    rsi_overbought: int
    price_below_ema_for_sell: int
    total_evaluated: int


@dataclass(frozen=True)
class CostBreakdown:
    gross_profit_before_costs: Decimal
    gross_loss_before_costs: Decimal
    net_profit_after_costs: Decimal
    average_gross_trade: Decimal | None
    average_net_trade: Decimal | None
    average_entry_fee: Decimal | None
    average_exit_fee: Decimal | None
    average_total_cost_per_trade: Decimal | None
    costs_as_pct_of_initial_capital: Decimal
    costs_as_pct_of_gross_profit: Decimal | None
    profitable_before_costs_but_losing_after: int
    trades_losing_before_costs: int
    trades_losing_after_costs: int


@dataclass(frozen=True)
class BenchmarkInfo:
    full_period_start_time: int
    full_period_start_price: Decimal
    effective_start_time: int
    effective_start_price: Decimal
    last_candle_time: int
    last_candle_price: Decimal
    buy_and_hold_full_period_pct: Decimal
    buy_and_hold_effective_period_pct: Decimal


@dataclass(frozen=True)
class TradeDistribution:
    median_net_pnl: Decimal | None
    p25_net_pnl: Decimal | None
    p75_net_pnl: Decimal | None
    median_duration_candles: Decimal | None
    p25_duration_candles: Decimal | None
    p75_duration_candles: Decimal | None
    best_5_trade_ids: tuple[int, ...]
    worst_5_trade_ids: tuple[int, ...]
    pct_result_from_top_5: Decimal | None
    pct_loss_from_bottom_5: Decimal | None


@dataclass(frozen=True)
class BacktestDiagnostics:
    benchmark: BenchmarkInfo
    cost_breakdown: CostBreakdown
    cost_scenarios: tuple[CostScenarioResult, ...]
    exit_reason_breakdown: tuple[ExitReasonSummary, ...]
    monthly_breakdown: tuple[MonthlyResult, ...]
    quarterly_breakdown: tuple[QuarterlyResult, ...]
    trade_distribution: TradeDistribution
    entry_blockers: EntryBlockerCounts
    trade_excursions: tuple[TradeExcursion, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[Decimal], p: float) -> Decimal | None:
    if not data:
        return None
    s = sorted(data)
    n = len(s)
    k = (n - 1) * p
    lo, hi = int(k), min(int(k) + 1, n - 1)
    frac = Decimal(str(k - int(k)))
    return s[lo] * (Decimal("1") - frac) + s[hi] * frac


def _trade_candle_indices(trade: BacktestTrade, eval_candles: list[Candle]) -> list[int]:
    result = []
    for i, c in enumerate(eval_candles):
        if trade.is_forced_close:
            in_range = c.open_time >= trade.entry_exec_time and c.close_time <= trade.exit_exec_time
        else:
            in_range = c.open_time >= trade.entry_exec_time and c.open_time <= trade.exit_exec_time
        if in_range:
            result.append(i)
    return result


def _safe_div(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == _ZERO:
        return _ZERO
    return numerator / denominator


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _estimate_slippage_cost(
    trades: list[BacktestTrade],
    slippage_rate: Decimal,
) -> Decimal:
    total = _ZERO
    denom = _ONE - slippage_rate
    for t in trades:
        entry_slip = t.quantity * t.entry_exec_price * slippage_rate / (_ONE + slippage_rate)
        if denom == _ZERO:
            exit_slip = _ZERO
        else:
            exit_slip = t.quantity * t.exit_exec_price * slippage_rate / denom
        total += entry_slip + exit_slip
    return total


def _primary_exit_reason(trade: BacktestTrade) -> str:
    if trade.is_forced_close:
        return "FORCED_CLOSE"
    if hasattr(trade, "exit_reasons") and trade.exit_reasons:
        # Return first non-composite reason
        for r in trade.exit_reasons:
            if r not in (ReasonCode.SELL_CONDITIONS_MET,):
                return str(r)
        return str(trade.exit_reasons[0])
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _compute_benchmark(
    result: BacktestResult,
    all_candles: list[Candle],
    warmup_len: int,
) -> BenchmarkInfo:
    eval_candles = all_candles[warmup_len:]
    full_start = all_candles[0]
    last_candle = eval_candles[-1]

    # Find first candle where warmup_complete in the equity curve
    # Use equity_curve to detect effective start; equity_curve covers eval_candles
    effective_candle = eval_candles[0]
    for ep, candle in zip(result.equity_curve, eval_candles, strict=False):
        # equity_curve is one entry per eval candle
        _ = ep  # we rely on warmup detection via evaluated_candles count
        effective_candle = candle
        break

    # Re-derive effective start from all_candles indicator warmup
    # The simplest approach: the evaluated_candles vs total_candles difference
    # gives us how many eval candles were warmup-incomplete from the front.
    skipped = result.total_candles - result.evaluated_candles
    eff_idx = min(skipped, len(eval_candles) - 1)
    effective_candle = eval_candles[eff_idx]

    cfg = result.config
    bah_full = buy_and_hold_return_pct(
        initial_capital=cfg.initial_capital,
        first_open=full_start.open,
        last_close=last_candle.close,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    bah_effective = buy_and_hold_return_pct(
        initial_capital=cfg.initial_capital,
        first_open=effective_candle.open,
        last_close=last_candle.close,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )

    return BenchmarkInfo(
        full_period_start_time=full_start.open_time,
        full_period_start_price=full_start.open,
        effective_start_time=effective_candle.open_time,
        effective_start_price=effective_candle.open,
        last_candle_time=last_candle.open_time,
        last_candle_price=last_candle.close,
        buy_and_hold_full_period_pct=bah_full,
        buy_and_hold_effective_period_pct=bah_effective,
    )


def _compute_cost_breakdown(result: BacktestResult) -> CostBreakdown:
    trades = result.trades
    n = len(trades)

    gross_profit = sum((t.gross_pnl for t in trades if t.gross_pnl > _ZERO), _ZERO)
    gross_loss = abs(sum((t.gross_pnl for t in trades if t.gross_pnl <= _ZERO), _ZERO))
    net_profit = sum((t.net_pnl for t in trades), _ZERO)

    avg_gross = (sum((t.gross_pnl for t in trades), _ZERO) / Decimal(n)) if n else None
    avg_net = (sum((t.net_pnl for t in trades), _ZERO) / Decimal(n)) if n else None
    avg_entry_fee = (sum((t.entry_fee for t in trades), _ZERO) / Decimal(n)) if n else None
    avg_exit_fee = (sum((t.exit_fee for t in trades), _ZERO) / Decimal(n)) if n else None
    avg_cost = (sum((t.entry_fee + t.exit_fee for t in trades), _ZERO) / Decimal(n)) if n else None

    total_costs = result.total_fees
    costs_pct_capital = _safe_div(total_costs, result.initial_capital) * _HUNDRED
    costs_pct_gross = (
        _safe_div(total_costs, gross_profit) * _HUNDRED if gross_profit > _ZERO else None
    )

    profitable_before_losing_after = sum(
        1 for t in trades if t.gross_pnl > _ZERO and t.net_pnl <= _ZERO
    )
    losing_before = sum(1 for t in trades if t.gross_pnl <= _ZERO)
    losing_after = sum(1 for t in trades if t.net_pnl <= _ZERO)

    return CostBreakdown(
        gross_profit_before_costs=gross_profit,
        gross_loss_before_costs=gross_loss,
        net_profit_after_costs=net_profit,
        average_gross_trade=avg_gross,
        average_net_trade=avg_net,
        average_entry_fee=avg_entry_fee,
        average_exit_fee=avg_exit_fee,
        average_total_cost_per_trade=avg_cost,
        costs_as_pct_of_initial_capital=costs_pct_capital,
        costs_as_pct_of_gross_profit=costs_pct_gross,
        profitable_before_costs_but_losing_after=profitable_before_losing_after,
        trades_losing_before_costs=losing_before,
        trades_losing_after_costs=losing_after,
    )


def _compute_cost_scenarios(
    result: BacktestResult,
    all_candles: list[Candle],
    warmup_len: int,
    strategy_engine: StrategyEngine,
    indicator_config: IndicatorConfig,
) -> tuple[CostScenarioResult, ...]:
    orig_cfg = result.config
    orig_fee = orig_cfg.fee_percentage
    orig_slip = orig_cfg.slippage_percentage

    scenarios = [
        ("no_costs", Decimal("0"), Decimal("0")),
        ("fee_only", orig_fee, Decimal("0")),
        ("slippage_only", Decimal("0"), orig_slip),
        ("fee_and_slippage", orig_fee, orig_slip),
    ]

    out: list[CostScenarioResult] = []
    for name, fee_pct, slip_pct in scenarios:
        mod_cfg = orig_cfg.model_copy(
            update={"fee_percentage": fee_pct, "slippage_percentage": slip_pct}
        )
        r = BacktestEngine(mod_cfg, strategy_engine, indicator_config).run(all_candles, warmup_len)
        slip_cost = _estimate_slippage_cost(r.trades, mod_cfg.slippage_rate)
        out.append(
            CostScenarioResult(
                name=name,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                final_equity=r.final_equity,
                return_pct=r.total_return_pct,
                total_fees=r.total_fees,
                estimated_slippage_cost=slip_cost,
                max_drawdown_pct=r.max_drawdown_pct,
                total_trades=r.total_trades,
                win_rate_pct=r.win_rate_pct,
                profit_factor=r.profit_factor,
            )
        )
    return tuple(out)


def _compute_exit_reason_breakdown(
    result: BacktestResult,
    eval_candles: list[Candle],
) -> tuple[ExitReasonSummary, ...]:
    groups: dict[str, list[BacktestTrade]] = {}
    for trade in result.trades:
        key = _primary_exit_reason(trade)
        groups.setdefault(key, []).append(trade)

    summaries: list[ExitReasonSummary] = []
    for reason, trades in sorted(groups.items()):
        wins = [t for t in trades if t.net_pnl > _ZERO]
        win_count = len(wins)
        n = len(trades)
        win_rate = (Decimal(win_count) / Decimal(n) * _HUNDRED) if n else None
        gross_sum = sum((t.gross_pnl for t in trades), _ZERO)
        net_sum = sum((t.net_pnl for t in trades), _ZERO)
        avg_net = net_sum / Decimal(n) if n else None

        durations: list[Decimal] = []
        for t in trades:
            indices = _trade_candle_indices(t, eval_candles)
            if indices:
                durations.append(Decimal(len(indices)))
        avg_dur = sum(durations, _ZERO) / Decimal(len(durations)) if durations else None

        net_pnls = [t.net_pnl for t in trades]
        max_gain = max(net_pnls) if net_pnls else _ZERO
        max_loss = min(net_pnls) if net_pnls else _ZERO

        summaries.append(
            ExitReasonSummary(
                exit_reason=reason,
                trade_count=n,
                win_count=win_count,
                win_rate_pct=win_rate,
                gross_pnl=gross_sum,
                net_pnl=net_sum,
                avg_net_pnl=avg_net,
                avg_duration_candles=avg_dur,
                max_loss=max_loss,
                max_gain=max_gain,
            )
        )
    return tuple(summaries)


def _compute_monthly_breakdown(result: BacktestResult) -> tuple[MonthlyResult, ...]:
    groups: dict[tuple[int, int], list[BacktestTrade]] = {}
    for trade in result.trades:
        dt = _ms_to_utc(trade.entry_exec_time)
        key = (dt.year, dt.month)
        groups.setdefault(key, []).append(trade)

    out: list[MonthlyResult] = []
    for (year, month), trades in sorted(groups.items()):
        n = len(trades)
        wins = sum(1 for t in trades if t.net_pnl > _ZERO)
        net = sum((t.net_pnl for t in trades), _ZERO)
        fees = sum((t.entry_fee + t.exit_fee for t in trades), _ZERO)
        wr = (Decimal(wins) / Decimal(n) * _HUNDRED) if n else None
        out.append(
            MonthlyResult(
                year=year,
                month=month,
                trade_count=n,
                net_pnl=net,
                total_fees=fees,
                win_count=wins,
                win_rate_pct=wr,
            )
        )
    return tuple(out)


def _compute_quarterly_breakdown(result: BacktestResult) -> tuple[QuarterlyResult, ...]:
    groups: dict[tuple[int, int], list[BacktestTrade]] = {}
    for trade in result.trades:
        dt = _ms_to_utc(trade.entry_exec_time)
        quarter = (dt.month - 1) // 3 + 1
        key = (dt.year, quarter)
        groups.setdefault(key, []).append(trade)

    out: list[QuarterlyResult] = []
    for (year, quarter), trades in sorted(groups.items()):
        n = len(trades)
        wins = sum(1 for t in trades if t.net_pnl > _ZERO)
        net = sum((t.net_pnl for t in trades), _ZERO)
        fees = sum((t.entry_fee + t.exit_fee for t in trades), _ZERO)
        wr = (Decimal(wins) / Decimal(n) * _HUNDRED) if n else None
        out.append(
            QuarterlyResult(
                year=year,
                quarter=quarter,
                trade_count=n,
                net_pnl=net,
                total_fees=fees,
                win_count=wins,
                win_rate_pct=wr,
            )
        )
    return tuple(out)


def _compute_trade_distribution(
    result: BacktestResult,
    eval_candles: list[Candle],
) -> TradeDistribution:
    trades = result.trades
    if not trades:
        return TradeDistribution(
            median_net_pnl=None,
            p25_net_pnl=None,
            p75_net_pnl=None,
            median_duration_candles=None,
            p25_duration_candles=None,
            p75_duration_candles=None,
            best_5_trade_ids=(),
            worst_5_trade_ids=(),
            pct_result_from_top_5=None,
            pct_loss_from_bottom_5=None,
        )

    net_pnls = [t.net_pnl for t in trades]
    durations = [Decimal(len(_trade_candle_indices(t, eval_candles))) for t in trades]

    sorted_by_net = sorted(trades, key=lambda t: t.net_pnl, reverse=True)
    best5 = tuple(t.trade_id for t in sorted_by_net[:5])
    worst5 = tuple(t.trade_id for t in sorted_by_net[-5:][::-1])

    total_net = sum(net_pnls, _ZERO)
    top5_net = sum((t.net_pnl for t in sorted_by_net[:5]), _ZERO)
    pct_top5 = (top5_net / total_net * _HUNDRED) if total_net != _ZERO else None

    total_loss = abs(sum((p for p in net_pnls if p < _ZERO), _ZERO))
    bottom5_loss = abs(sum((t.net_pnl for t in sorted_by_net[-5:]), _ZERO))
    pct_bottom5 = (bottom5_loss / total_loss * _HUNDRED) if total_loss > _ZERO else None

    return TradeDistribution(
        median_net_pnl=_percentile(net_pnls, 0.5),
        p25_net_pnl=_percentile(net_pnls, 0.25),
        p75_net_pnl=_percentile(net_pnls, 0.75),
        median_duration_candles=_percentile(durations, 0.5),
        p25_duration_candles=_percentile(durations, 0.25),
        p75_duration_candles=_percentile(durations, 0.75),
        best_5_trade_ids=best5,
        worst_5_trade_ids=worst5,
        pct_result_from_top_5=pct_top5,
        pct_loss_from_bottom_5=pct_bottom5,
    )


def _compute_excursions(
    result: BacktestResult,
    eval_candles: list[Candle],
) -> tuple[TradeExcursion, ...]:
    excursions: list[TradeExcursion] = []
    for trade in result.trades:
        indices = _trade_candle_indices(trade, eval_candles)
        if not indices:
            excursions.append(
                TradeExcursion(
                    trade_id=trade.trade_id,
                    entry_exec_price=trade.entry_exec_price,
                    max_favorable_excursion_pct=_ZERO,
                    max_adverse_excursion_pct=_ZERO,
                    highest_price=trade.entry_exec_price,
                    lowest_price=trade.entry_exec_price,
                    candles_until_mfe=0,
                    candles_until_mae=0,
                    candles_in_trade=0,
                )
            )
            continue

        highs = [eval_candles[i].high for i in indices]
        lows = [eval_candles[i].low for i in indices]

        highest = max(highs)
        lowest = min(lows)
        entry = trade.entry_exec_price

        mfe_pct = _safe_div(highest - entry, entry) * _HUNDRED
        mae_pct = _safe_div(entry - lowest, entry) * _HUNDRED

        mfe_local = highs.index(highest)
        mae_local = lows.index(lowest)

        excursions.append(
            TradeExcursion(
                trade_id=trade.trade_id,
                entry_exec_price=entry,
                max_favorable_excursion_pct=mfe_pct,
                max_adverse_excursion_pct=mae_pct,
                highest_price=highest,
                lowest_price=lowest,
                candles_until_mfe=mfe_local,
                candles_until_mae=mae_local,
                candles_in_trade=len(indices),
            )
        )
    return tuple(excursions)


def _compute_entry_blockers(
    result: BacktestResult,
    all_candles: list[Candle],
    warmup_len: int,
    strategy_engine: StrategyEngine,
    indicator_config: IndicatorConfig,
) -> EntryBlockerCounts:
    calculator = IndicatorCalculator(indicator_config)
    all_results = calculator.calculate(all_candles)
    eval_results = all_results[warmup_len:]

    counts: dict[ReasonCode, int] = {
        ReasonCode.WARMUP_INCOMPLETE: 0,
        ReasonCode.NO_NEW_CROSSOVER: 0,
        ReasonCode.PRICE_BELOW_LONG_EMA: 0,
        ReasonCode.RSI_OUTSIDE_BUY_RANGE: 0,
        ReasonCode.VOLUME_INSUFFICIENT: 0,
    }
    total_evaluated = 0

    no_pos = PositionContext(has_open_long_position=False)
    for ind_result in eval_results:
        decision = strategy_engine.evaluate(ind_result, no_pos)
        total_evaluated += 1
        if decision.action == StrategyAction.WAIT:
            for rc in decision.failed_conditions:
                if rc in counts:
                    counts[rc] += 1
            # Also count warmup as a reason (it appears in reasons, not failed_conditions)
            for rc in decision.reasons:
                if rc == ReasonCode.WARMUP_INCOMPLETE:
                    counts[ReasonCode.WARMUP_INCOMPLETE] += 1

    position_already_open = sum(1 for ep in result.equity_curve if ep.has_open_position)

    # Count sell-side reasons from trade exit_reasons
    bearish_crossover = 0
    rsi_overbought = 0
    price_below_ema_sell = 0
    for trade in result.trades:
        if hasattr(trade, "exit_reasons") and trade.exit_reasons:
            for r in trade.exit_reasons:
                if r == ReasonCode.BEARISH_CROSSOVER:
                    bearish_crossover += 1
                elif r == ReasonCode.RSI_OVERBOUGHT:
                    rsi_overbought += 1
                elif r == ReasonCode.PRICE_BELOW_LONG_EMA:
                    price_below_ema_sell += 1

    return EntryBlockerCounts(
        warmup_incomplete=counts[ReasonCode.WARMUP_INCOMPLETE],
        no_bullish_crossover=counts[ReasonCode.NO_NEW_CROSSOVER],
        price_below_long_ema=counts[ReasonCode.PRICE_BELOW_LONG_EMA],
        rsi_outside_buy_range=counts[ReasonCode.RSI_OUTSIDE_BUY_RANGE],
        insufficient_volume=counts[ReasonCode.VOLUME_INSUFFICIENT],
        position_already_open=position_already_open,
        bearish_crossover=bearish_crossover,
        rsi_overbought=rsi_overbought,
        price_below_ema_for_sell=price_below_ema_sell,
        total_evaluated=total_evaluated,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_diagnostics(
    result: BacktestResult,
    all_candles: list[Candle],
    warmup_len: int,
    strategy_engine: StrategyEngine,
    indicator_config: IndicatorConfig,
) -> BacktestDiagnostics:
    """Compute comprehensive diagnostics for a completed backtest.

    all_candles: full list including warmup prefix, same as passed to engine.run()
    warmup_len: number of warmup candles prefix.
    strategy_engine: same engine used for the backtest.
    indicator_config: same indicator config used for the backtest.

    Does NOT modify result or re-run the main backtest with the same config.
    Runs 4 additional lightweight backtests for cost scenario analysis.
    """
    eval_candles = all_candles[warmup_len:]

    benchmark = _compute_benchmark(result, all_candles, warmup_len)
    cost_breakdown = _compute_cost_breakdown(result)
    cost_scenarios = _compute_cost_scenarios(
        result, all_candles, warmup_len, strategy_engine, indicator_config
    )
    exit_reason_breakdown = _compute_exit_reason_breakdown(result, eval_candles)
    monthly_breakdown = _compute_monthly_breakdown(result)
    quarterly_breakdown = _compute_quarterly_breakdown(result)
    trade_distribution = _compute_trade_distribution(result, eval_candles)
    entry_blockers = _compute_entry_blockers(
        result, all_candles, warmup_len, strategy_engine, indicator_config
    )
    trade_excursions = _compute_excursions(result, eval_candles)

    return BacktestDiagnostics(
        benchmark=benchmark,
        cost_breakdown=cost_breakdown,
        cost_scenarios=cost_scenarios,
        exit_reason_breakdown=exit_reason_breakdown,
        monthly_breakdown=monthly_breakdown,
        quarterly_breakdown=quarterly_breakdown,
        trade_distribution=trade_distribution,
        entry_blockers=entry_blockers,
        trade_excursions=trade_excursions,
    )
