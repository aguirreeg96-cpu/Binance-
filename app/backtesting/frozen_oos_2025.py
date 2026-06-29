"""Stage 5.3 — Frozen out-of-sample evaluation on 2025 data.

Frozen candidate: ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY + 25% allocation.
Source interval: 15m (BTCUSDT).  Trading timeframe: 30m (aggregated from 15m).
Out-of-sample period: 2025-01-01 to 2026-01-01 exclusively.

3 cost scenarios: NO_COSTS, BASE_COSTS, CONSERVATIVE.
Historical reference (2023–2024, BASE_COSTS, read-only):
  2023=+3.8785%, 2024=+4.4320%, combined=+8.4824%, 44 trades, worst DD=3.0626%.

Constraints (Stage 5.3 spec — immutable):
  - No strategy modification after seeing the result.
  - No new parameter tuning.
  - No other symbols.
  - If 2025 data is missing, report the missing range — do not fabricate candles.
  - No automatic data download without explicit command.
  - PAPER/TEST only. Not a profitability guarantee.
  Past results do NOT predict future performance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from app.backtesting.risk_exit_config import RiskExitConfig
from app.strategy.entry_filter import EntryFilterConfig, EntryFilterType

if TYPE_CHECKING:
    from app.backtesting.schemas import BacktestResult, BacktestTrade
    from app.indicators.schemas import IndicatorConfig
    from app.models.candle import Candle
    from app.strategy.config import StrategyEngineConfig

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_ONE = Decimal("1")
_TWO = Decimal("2")

# ---------------------------------------------------------------------------
# Public constants — frozen candidate identity
# ---------------------------------------------------------------------------

OOS_TRADING_TIMEFRAME = "30m"
OOS_YEAR = "2025"
OOS_SOURCE_INTERVAL_MS = 900_000  # 15m in milliseconds
OOS_INTERVAL_30M_MS = 1_800_000  # 30m in milliseconds

OOS_SCENARIO_NAMES: tuple[str, ...] = ("NO_COSTS", "BASE_COSTS", "CONSERVATIVE")
OOS_SCENARIOS: dict[str, tuple[Decimal, Decimal]] = {
    "NO_COSTS": (_ZERO, _ZERO),
    "BASE_COSTS": (Decimal("0.1"), Decimal("0.05")),
    "CONSERVATIVE": (Decimal("0.1"), Decimal("0.10")),
}

# Cost ordering for monotonicity checks (lower int = cheaper).
_OOS_COST_LEVEL: dict[str, int] = {
    "NO_COSTS": 0,
    "BASE_COSTS": 1,
    "CONSERVATIVE": 2,
}

# Historical in-sample reference — READ-ONLY.  Do NOT use to modify parameters.
OOS_HISTORICAL_REFERENCE: dict[str, str] = {
    "symbol": "BTCUSDT",
    "trading_timeframe": "30m",
    "entry": "ENTRY_V3_ALIGNED_TREND",
    "exit_config": "V2_STOP_ONLY",
    "allocation_pct": "25",
    "cost_scenario": "BASE_COSTS",
    "period_2023_return_pct": "3.8785",
    "period_2024_return_pct": "4.4320",
    "combined_return_pct": "8.4824",
    "total_trades_2023_2024": "44",
    "worst_drawdown_pct": "3.0626",
    "note": (
        "READ-ONLY reference. 2023–2024 in-sample results under BASE_COSTS. "
        "Do NOT use these values to modify strategy parameters."
    ),
}

# Frozen filter and risk configs — duplicated here (not imported from
# timeframe_cost_comparison) to keep OOS module self-contained.
_OOS_FILTER_CFG = EntryFilterConfig(
    filter_type=EntryFilterType.V3_ALIGNED_TREND,
    crossover_lookback_candles=1,
)
_OOS_RISK_CFG = RiskExitConfig(
    use_take_profit=False,
    maximum_holding_candles=0,
    use_bearish_crossover_exit=True,
    position_allocation_percentage=Decimal("25"),
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OosScenarioResult:
    """Per-scenario OOS metrics for 2025.

    Non-frozen: contains exit_reason_counts dict.
    PAPER/TEST only.  Not a profitability claim.
    """

    scenario: str
    initial_capital: Decimal
    final_equity: Decimal
    return_pct: Decimal
    total_trades: int
    win_rate_pct: Decimal | None
    profit_factor: Decimal | None
    max_drawdown_pct: Decimal
    total_fees: Decimal
    slippage_cost: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    avg_net_pnl: Decimal | None
    median_net_pnl: Decimal | None
    exposure_pct: Decimal
    avg_trade_duration_candles: Decimal | None
    buy_and_hold_return_pct: Decimal
    first_trade_date: str | None
    last_trade_date: str | None
    exit_reason_counts: dict[str, int]


@dataclass(frozen=True)
class OosAudit:
    """Data-integrity and boundary audit for the 2025 OOS run.

    PAPER/TEST only.
    """

    candles_15m_total: int
    warmup_15m: int
    eval_15m: int
    candles_30m_total: int
    warmup_30m: int
    eval_30m: int
    entry_signal_hash: str  # SHA-256[:16] of entry_signal_times from BASE_COSTS
    warmup_boundary_ok: bool
    end_boundary_ok: bool
    costs_monotonic: bool
    missing_data_ranges: tuple[str, ...]
    violations: tuple[str, ...]


@dataclass(frozen=True)
class OosInterpretation:
    """Boolean interpretation flags for the 2025 OOS result.

    None of these flags constitute a trading recommendation.
    PAPER/TEST only.
    """

    profitable_no_costs: bool
    profitable_base_costs: bool
    profitable_conservative: bool
    profit_factor_above_1_base: bool
    costs_monotonic: bool
    drawdown_below_historical: bool
    return_above_historical_min: bool
    sufficient_trades: bool


@dataclass
class FrozenOos2025Report:
    """Complete Stage 5.3 out-of-sample report for 2025.

    Non-frozen: contains raw_results dict.
    PAPER/TEST only.  Past results do NOT predict future performance.
    """

    symbol: str
    trading_timeframe: str
    oos_year: str
    start_ms: int
    end_ms: int
    initial_capital: Decimal
    scenarios: list[OosScenarioResult]
    historical_reference: dict[str, str]
    interpretation: OosInterpretation
    audit: OosAudit
    raw_results: dict[str, BacktestResult]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sha16(items: list[str]) -> str:
    data = "\n".join(items).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _median_net_pnl(trades: list[BacktestTrade]) -> Decimal | None:
    if not trades:
        return None
    sorted_pnls = sorted(t.net_pnl for t in trades)
    n = len(sorted_pnls)
    mid = n // 2
    if n % 2 == 1:
        return sorted_pnls[mid]
    return (sorted_pnls[mid - 1] + sorted_pnls[mid]) / _TWO


def _gross_profit(trades: list[BacktestTrade]) -> Decimal:
    return sum((t.gross_pnl for t in trades if t.gross_pnl > _ZERO), _ZERO)


def _gross_loss(trades: list[BacktestTrade]) -> Decimal:
    return sum((t.gross_pnl for t in trades if t.gross_pnl < _ZERO), _ZERO)


def _exit_reason_counts(trades: list[BacktestTrade]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trades:
        for reason in t.exit_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d")


def _detect_gaps(candles: list[Candle], start_ms: int, interval_ms: int) -> tuple[str, ...]:
    """Return descriptions of missing-data gaps in the eval portion of candles."""
    eval_candles = [c for c in candles if c.open_time >= start_ms]
    gaps: list[str] = []
    for i in range(len(eval_candles) - 1):
        gap = eval_candles[i + 1].open_time - eval_candles[i].open_time
        if gap > interval_ms:
            missing_count = (gap - interval_ms) // interval_ms
            gap_start_ms = eval_candles[i].open_time + interval_ms
            gap_end_ms = eval_candles[i + 1].open_time
            gaps.append(
                f"{_ms_to_date(gap_start_ms)}–{_ms_to_date(gap_end_ms)}"
                f" ({missing_count} missing 15m candles)"
            )
    return tuple(gaps)


def _build_audit(
    candles_15m: list[Candle],
    candles_30m: list[Candle],
    warmup_30m: int,
    start_ms: int,
    end_ms: int,
    raw_results: dict[str, BacktestResult],
) -> OosAudit:
    warmup_15m = sum(1 for c in candles_15m if c.open_time < start_ms)
    candles_15m_total = len(candles_15m)
    eval_15m = candles_15m_total - warmup_15m
    candles_30m_total = len(candles_30m)
    eval_30m = candles_30m_total - warmup_30m

    # Entry signal hash from BASE_COSTS (or NO_COSTS fallback).
    ref = raw_results.get("BASE_COSTS") or raw_results.get("NO_COSTS")
    if ref and ref.trades:
        entry_signal_hash = _sha16([str(t.entry_signal_time) for t in ref.trades])
    else:
        entry_signal_hash = _sha16([])

    violations: list[str] = []

    # Warmup boundary: no entry before start_ms.
    warmup_boundary_ok = True
    for scenario, result in raw_results.items():
        for trade in result.trades:
            if trade.entry_exec_time < start_ms:
                warmup_boundary_ok = False
                violations.append(
                    f"Warmup violation in {scenario}: trade entered at "
                    f"{trade.entry_exec_time} < start_ms {start_ms}"
                )

    # End boundary: no non-forced exit after end_ms.
    end_boundary_ok = True
    for scenario, result in raw_results.items():
        for trade in result.trades:
            if not trade.is_forced_close and trade.exit_exec_time > end_ms:
                end_boundary_ok = False
                violations.append(
                    f"End violation in {scenario}: non-forced exit at "
                    f"{trade.exit_exec_time} > end_ms {end_ms}"
                )

    # Monotonicity: lower cost scenario must have return_pct >= higher cost scenario.
    costs_monotonic = True
    ordered = ["NO_COSTS", "BASE_COSTS", "CONSERVATIVE"]
    for i in range(len(ordered) - 1):
        lo, hi = ordered[i], ordered[i + 1]
        if lo in raw_results and hi in raw_results:
            ret_lo = raw_results[lo].total_return_pct
            ret_hi = raw_results[hi].total_return_pct
            if ret_lo < ret_hi:
                costs_monotonic = False
                violations.append(
                    f"Monotonicity violated: {hi} return ({ret_hi:.4f}%) "
                    f"> {lo} return ({ret_lo:.4f}%)"
                )

    missing_data_ranges = _detect_gaps(candles_15m, start_ms, OOS_SOURCE_INTERVAL_MS)

    return OosAudit(
        candles_15m_total=candles_15m_total,
        warmup_15m=warmup_15m,
        eval_15m=eval_15m,
        candles_30m_total=candles_30m_total,
        warmup_30m=warmup_30m,
        eval_30m=eval_30m,
        entry_signal_hash=entry_signal_hash,
        warmup_boundary_ok=warmup_boundary_ok,
        end_boundary_ok=end_boundary_ok,
        costs_monotonic=costs_monotonic,
        missing_data_ranges=missing_data_ranges,
        violations=tuple(violations),
    )


def _build_interpretation(
    scenarios: list[OosScenarioResult],
) -> OosInterpretation:
    nc = next((s for s in scenarios if s.scenario == "NO_COSTS"), None)
    bc = next((s for s in scenarios if s.scenario == "BASE_COSTS"), None)
    co = next((s for s in scenarios if s.scenario == "CONSERVATIVE"), None)

    hist_dd = Decimal(OOS_HISTORICAL_REFERENCE["worst_drawdown_pct"])
    hist_min_return = min(
        Decimal(OOS_HISTORICAL_REFERENCE["period_2023_return_pct"]),
        Decimal(OOS_HISTORICAL_REFERENCE["period_2024_return_pct"]),
    )

    return OosInterpretation(
        profitable_no_costs=nc is not None and nc.return_pct > _ZERO,
        profitable_base_costs=bc is not None and bc.return_pct > _ZERO,
        profitable_conservative=co is not None and co.return_pct > _ZERO,
        profit_factor_above_1_base=(
            bc is not None and bc.profit_factor is not None and bc.profit_factor > _ONE
        ),
        costs_monotonic=(
            nc is not None
            and bc is not None
            and co is not None
            and nc.return_pct >= bc.return_pct >= co.return_pct
        ),
        drawdown_below_historical=bc is not None and bc.max_drawdown_pct <= hist_dd,
        return_above_historical_min=bc is not None and bc.return_pct >= hist_min_return,
        sufficient_trades=bc is not None and bc.total_trades >= 5,
    )


def _run_oos_single(
    all_candles: list[Candle],
    warmup_len: int,
    fee: Decimal,
    slip: Decimal,
    symbol: str,
    initial_capital: Decimal,
    start_ms: int,
    end_ms: int,
    force_close_at_end: bool,
    all_results: list,
    ind_config: IndicatorConfig,
    strat_config: StrategyEngineConfig,
) -> BacktestResult:
    from app.backtesting.config import BacktestConfig
    from app.backtesting.entry_comparison import _FilteredStrategyEngine
    from app.backtesting.v2_engine import V2BacktestEngine
    from app.strategy.engine import StrategyEngine

    cfg = BacktestConfig(
        symbol=symbol,
        interval=OOS_TRADING_TIMEFRAME,
        start_ms=start_ms,
        end_ms=end_ms,
        initial_capital=initial_capital,
        fee_percentage=fee,
        slippage_percentage=slip,
        force_close_at_end=force_close_at_end,
    )
    base_engine = StrategyEngine(strat_config)
    filter_engine = _FilteredStrategyEngine(
        base_engine, all_results, _OOS_FILTER_CFG, htf_bars=None
    )
    v2 = V2BacktestEngine(
        config=cfg,
        risk_exit_config=_OOS_RISK_CFG,
        strategy_engine=filter_engine,
        indicator_config=ind_config,
    )
    return v2.run(all_candles, warmup_len)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_frozen_oos_2025(
    symbol: str,
    initial_capital: Decimal,
    candles_15m_2025: list[Candle],
    start_ms_2025: int,
    end_ms_2025: int,
    force_close_at_end: bool = True,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
) -> FrozenOos2025Report:
    """Run the frozen candidate on 2025 out-of-sample data only.

    Aggregates 15m candles to 30m, computes indicators once, then runs
    3 cost scenarios.  Strategy config is ignored for filter/risk settings —
    those use the frozen module-level constants exclusively.

    Raises BacktestInsufficientDataError if there are no eval candles after warmup.
    Does not fabricate candles for missing data ranges — gaps are reported in audit.

    PAPER/TEST only.  Past results do NOT predict future performance.
    """
    from app.backtesting.exceptions import BacktestInsufficientDataError
    from app.backtesting.timeframe_aggregator import aggregate_candles, warmup_len_for
    from app.indicators.calculator import IndicatorCalculator
    from app.indicators.schemas import IndicatorConfig as _IndicatorConfig
    from app.strategy.config import StrategyEngineConfig as _StrategyEngineConfig

    ind_config = indicator_config or _IndicatorConfig()
    strat_config = strategy_config or _StrategyEngineConfig()

    candles_30m = aggregate_candles(candles_15m_2025, OOS_TRADING_TIMEFRAME, OOS_SOURCE_INTERVAL_MS)
    warmup_30m = warmup_len_for(candles_30m, start_ms_2025)

    if len(candles_30m) <= warmup_30m:
        raise BacktestInsufficientDataError(
            f"No evaluation candles for {symbol} {OOS_TRADING_TIMEFRAME} in 2025 after "
            f"aggregation (total={len(candles_30m)}, warmup={warmup_30m}). "
            "Ensure 15m candles for 2025 are downloaded."
        )

    ind_results = IndicatorCalculator(ind_config).calculate(candles_30m)

    raw_results: dict[str, BacktestResult] = {}
    for scenario in OOS_SCENARIO_NAMES:
        fee, slip = OOS_SCENARIOS[scenario]
        raw_results[scenario] = _run_oos_single(
            all_candles=candles_30m,
            warmup_len=warmup_30m,
            fee=fee,
            slip=slip,
            symbol=symbol,
            initial_capital=initial_capital,
            start_ms=start_ms_2025,
            end_ms=end_ms_2025,
            force_close_at_end=force_close_at_end,
            all_results=ind_results,
            ind_config=ind_config,
            strat_config=strat_config,
        )

    # Build per-scenario summaries.
    no_costs_return = raw_results["NO_COSTS"].total_return_pct
    scenarios: list[OosScenarioResult] = []
    for scenario in OOS_SCENARIO_NAMES:
        result = raw_results[scenario]
        trades = result.trades

        cost_drag = no_costs_return - result.total_return_pct
        slip_cost = max(
            _ZERO,
            cost_drag * result.initial_capital / _HUNDRED - result.total_fees,
        )

        n = len(trades)
        avg_net = sum((t.net_pnl for t in trades), _ZERO) / Decimal(str(n)) if n else None
        avg_dur: Decimal | None = None
        if trades:
            durations = [
                Decimal(str(t.exit_exec_time - t.entry_exec_time))
                / Decimal(str(OOS_INTERVAL_30M_MS))
                for t in trades
            ]
            avg_dur = sum(durations, _ZERO) / Decimal(str(len(durations)))

        first_date = _ms_to_date(trades[0].entry_exec_time) if trades else None
        last_date = _ms_to_date(trades[-1].exit_exec_time) if trades else None

        scenarios.append(
            OosScenarioResult(
                scenario=scenario,
                initial_capital=result.initial_capital,
                final_equity=result.final_equity,
                return_pct=result.total_return_pct,
                total_trades=result.total_trades,
                win_rate_pct=result.win_rate_pct,
                profit_factor=result.profit_factor,
                max_drawdown_pct=result.max_drawdown_pct,
                total_fees=result.total_fees,
                slippage_cost=slip_cost,
                gross_profit=_gross_profit(trades),
                gross_loss=_gross_loss(trades),
                avg_net_pnl=avg_net,
                median_net_pnl=_median_net_pnl(trades),
                exposure_pct=result.exposure_pct,
                avg_trade_duration_candles=avg_dur,
                buy_and_hold_return_pct=result.buy_and_hold_return_pct,
                first_trade_date=first_date,
                last_trade_date=last_date,
                exit_reason_counts=_exit_reason_counts(trades),
            )
        )

    audit = _build_audit(
        candles_15m=candles_15m_2025,
        candles_30m=candles_30m,
        warmup_30m=warmup_30m,
        start_ms=start_ms_2025,
        end_ms=end_ms_2025,
        raw_results=raw_results,
    )
    interpretation = _build_interpretation(scenarios)

    return FrozenOos2025Report(
        symbol=symbol,
        trading_timeframe=OOS_TRADING_TIMEFRAME,
        oos_year=OOS_YEAR,
        start_ms=start_ms_2025,
        end_ms=end_ms_2025,
        initial_capital=initial_capital,
        scenarios=scenarios,
        historical_reference=OOS_HISTORICAL_REFERENCE,
        interpretation=interpretation,
        audit=audit,
        raw_results=raw_results,
    )
