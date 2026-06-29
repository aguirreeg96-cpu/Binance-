"""Stage 5.2C — Timeframe and cost robustness comparison.

Frozen candidate: ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY + 25 % allocation.
Tests robustness across:
  - 3 timeframes: 15m, 30m, 1h  (30m and 1h built from 15m candles only)
  - 4 cost scenarios: NO_COSTS, BASE_COSTS, LOW_SLIPPAGE, CONSERVATIVE
  - 2 periods: 2023 and 2024 (2024 compounded from 2023 final equity per scenario)

Indicator periods are NOT scaled across timeframes (Stage 5.2C design decision):
  EMA-200 on 30m uses 200 × 30m candles; EMA-200 on 1h uses 200 × 1h candles.
  This is a literal parameter comparison, not a duration-equivalent comparison.

Stop-price note (Stage 5.2C.1 audit):
  initial_stop = entry_exec_price − atr × multiplier; entry_exec_price varies with
  slippage, so the stop trigger differs across cost scenarios.  Entry signals are
  identical (indicator-based); exit timing and type may differ between scenarios.

Cost-drag note for 2024 (Stage 5.2C.1 audit):
  Each scenario's 2024 run starts from its own 2023 final equity, so cost_drag
  in 2024 conflates 2024 execution costs with 2023 compounding differences.

No timeframe is declared superior. PAPER/TEST only.
Past results do NOT predict future performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from app.backtesting.config import BacktestConfig
from app.backtesting.exceptions import BacktestInsufficientDataError
from app.backtesting.risk_exit_config import RiskExitConfig
from app.backtesting.v2_engine import V2BacktestEngine
from app.indicators.schemas import IndicatorConfig, IndicatorResult
from app.models.candle import Candle
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine
from app.strategy.entry_filter import EntryFilterConfig, EntryFilterType

if TYPE_CHECKING:
    from app.backtesting.schemas import BacktestResult, BacktestTrade
    from app.backtesting.timeframe_cost_audit import TimeframeCostAuditReport

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_ONE = Decimal("1")
_TWO = Decimal("2")

# Frozen candidate configuration — immutable per Stage 5.2C spec.
FROZEN_ENTRY_VARIANT = "ENTRY_V3_ALIGNED_TREND"
FROZEN_EXIT_CONFIG = "V2_STOP_ONLY"
FROZEN_ALLOCATION_PCT = Decimal("25")

TIMEFRAMES: tuple[str, ...] = ("15m", "30m", "1h")
_SOURCE_INTERVAL_MS = 900_000  # 15m in milliseconds

COST_SCENARIO_NAMES: tuple[str, ...] = (
    "NO_COSTS",
    "BASE_COSTS",
    "LOW_SLIPPAGE",
    "CONSERVATIVE",
)
COST_SCENARIOS: dict[str, tuple[Decimal, Decimal]] = {
    "NO_COSTS": (_ZERO, _ZERO),
    "BASE_COSTS": (Decimal("0.1"), Decimal("0.05")),
    "LOW_SLIPPAGE": (Decimal("0.1"), Decimal("0.02")),
    "CONSERVATIVE": (Decimal("0.1"), Decimal("0.10")),
}

# Frozen candidate filter and exit configs — not modified between timeframes.
_FROZEN_FILTER_CFG = EntryFilterConfig(
    filter_type=EntryFilterType.V3_ALIGNED_TREND,
    crossover_lookback_candles=1,
)
_FROZEN_RISK_CFG = RiskExitConfig(
    use_take_profit=False,
    maximum_holding_candles=0,
    use_bearish_crossover_exit=True,
    position_allocation_percentage=FROZEN_ALLOCATION_PCT,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeframeCostResult:
    """Performance metrics for one (timeframe, cost_scenario, year) combination.

    slippage_cost is approximated as max(0, cost_drag * initial_capital / 100 - total_fees).
    cost_drag is the return_pct difference vs the NO_COSTS reference for the same
    (timeframe, year).  Both are zero for the NO_COSTS scenario itself.
    PAPER/TEST only — not a profitability claim.
    """

    timeframe: str
    cost_scenario: str
    year: str
    initial_capital: Decimal
    final_equity: Decimal
    return_pct: Decimal
    buy_and_hold_return_pct: Decimal
    total_trades: int
    win_rate_pct: Decimal | None
    profit_factor: Decimal | None
    max_drawdown_pct: Decimal
    total_fees: Decimal
    slippage_cost: Decimal
    exposure_pct: Decimal
    cost_drag: Decimal
    avg_trade_duration_candles: Decimal | None
    median_net_pnl: Decimal | None
    avg_gross_pnl: Decimal | None
    avg_net_pnl: Decimal | None
    avg_cost_per_trade: Decimal | None


@dataclass(frozen=True)
class TimeframeCostYearlySummary:
    """Multi-year summary for one (timeframe, cost_scenario) pair.

    Compounded equity: 2023 final → 2024 initial.
    combined_return_pct = ((1 + r23) × (1 + r24) − 1) × 100.
    PAPER/TEST only.
    """

    timeframe: str
    cost_scenario: str
    return_pct_2023: Decimal
    return_pct_2024: Decimal
    combined_return_pct: Decimal
    compounded_initial_2023: Decimal
    compounded_final_2023: Decimal
    compounded_initial_2024: Decimal
    compounded_final_2024: Decimal
    positive_years: int
    worst_drawdown_pct: Decimal
    worst_profit_factor: Decimal | None
    total_trades_2023: int
    total_trades_2024: int
    total_fees_2023: Decimal
    total_fees_2024: Decimal
    cost_drag_2023: Decimal
    cost_drag_2024: Decimal
    buy_and_hold_2023: Decimal
    buy_and_hold_2024: Decimal


@dataclass(frozen=True)
class TimeframeCostRobustness:
    """Robustness flags for one (timeframe, cost_scenario) pair.

    depends_on_low_slippage: LOW_SLIPPAGE scenario is profitable but BASE_COSTS is not
    (for the same timeframe).  This flag has the same value for every scenario in the
    same timeframe since it compares two fixed cost scenarios.
    PAPER/TEST only — not a trading recommendation.
    """

    timeframe: str
    cost_scenario: str
    positive_both_years: bool
    profit_factor_above_1_both_years: bool
    depends_on_low_slippage: bool
    fails_with_conservative: bool
    low_trade_count: bool


@dataclass(frozen=True)
class TimeframeCostReport:
    """Complete Stage 5.2C robustness report.

    results: 3 timeframes × 4 scenarios × 2 years = 24 entries.
    yearly_summary: 3 × 4 = 12 entries.
    robustness: 3 × 4 = 12 entries.
    PAPER/TEST only.
    """

    results: list[TimeframeCostResult]
    yearly_summary: list[TimeframeCostYearlySummary]
    robustness: list[TimeframeCostRobustness]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _median_net_pnl(trades: list[BacktestTrade]) -> Decimal | None:
    if not trades:
        return None
    sorted_pnls = sorted(t.net_pnl for t in trades)
    n = len(sorted_pnls)
    mid = n // 2
    if n % 2 == 1:
        return sorted_pnls[mid]
    return (sorted_pnls[mid - 1] + sorted_pnls[mid]) / _TWO


def _run_frozen_single(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    all_results: list[IndicatorResult],
    ind_config: IndicatorConfig,
    strat_config: StrategyEngineConfig,
) -> BacktestResult:
    """Run the frozen candidate for one (config, year) and return the result."""
    from app.backtesting.entry_comparison import _FilteredStrategyEngine

    base_engine = StrategyEngine(strat_config)
    filter_engine = _FilteredStrategyEngine(
        base_engine, all_results, _FROZEN_FILTER_CFG, htf_bars=None
    )
    v2 = V2BacktestEngine(
        config=config,
        risk_exit_config=_FROZEN_RISK_CFG,
        strategy_engine=filter_engine,
        indicator_config=ind_config,
    )
    return v2.run(all_candles, warmup_len)


@dataclass
class _ComparisonCore:
    """Internal container shared by the report builder and the audit builder."""

    raw: dict[tuple[str, str, str], BacktestResult]
    agg_by_tf_year: dict[tuple[str, str], list[Candle]]
    warmup_by_tf_year: dict[tuple[str, str], int]
    candles_15m_by_year: dict[str, list[Candle]]
    start_ms_by_year: dict[str, int]
    end_ms_by_year: dict[str, int]


def _run_comparison_core(
    symbol: str,
    initial_capital: Decimal,
    candles_15m_2023: list[Candle],
    candles_15m_2024: list[Candle],
    start_ms_2023: int,
    start_ms_2024: int,
    end_ms_2023: int,
    end_ms_2024: int,
    force_close_at_end: bool,
    ind_config: IndicatorConfig,
    strat_config: StrategyEngineConfig,
) -> _ComparisonCore:
    """Aggregate candles, compute indicators, run all backtests, return shared core."""
    from app.backtesting.timeframe_aggregator import aggregate_candles, warmup_len_for
    from app.indicators.calculator import IndicatorCalculator

    raw: dict[tuple[str, str, str], BacktestResult] = {}
    agg_by_tf_year: dict[tuple[str, str], list[Candle]] = {}
    warmup_by_tf_year: dict[tuple[str, str], int] = {}

    candles_15m_by_year: dict[str, list[Candle]] = {
        "2023": candles_15m_2023,
        "2024": candles_15m_2024,
    }
    start_ms_by_year: dict[str, int] = {"2023": start_ms_2023, "2024": start_ms_2024}
    end_ms_by_year: dict[str, int] = {"2023": end_ms_2023, "2024": end_ms_2024}

    for tf in TIMEFRAMES:
        if tf == "15m":
            agg_23: list[Candle] = candles_15m_2023
            agg_24: list[Candle] = candles_15m_2024
        else:
            agg_23 = aggregate_candles(candles_15m_2023, tf, _SOURCE_INTERVAL_MS)
            agg_24 = aggregate_candles(candles_15m_2024, tf, _SOURCE_INTERVAL_MS)

        warmup_23 = warmup_len_for(agg_23, start_ms_2023)
        warmup_24 = warmup_len_for(agg_24, start_ms_2024)

        agg_by_tf_year[(tf, "2023")] = agg_23
        agg_by_tf_year[(tf, "2024")] = agg_24
        warmup_by_tf_year[(tf, "2023")] = warmup_23
        warmup_by_tf_year[(tf, "2024")] = warmup_24

        if len(agg_23) <= warmup_23:
            raise BacktestInsufficientDataError(
                f"No evaluation candles for {symbol} {tf} in 2023 after aggregation "
                f"(total={len(agg_23)}, warmup={warmup_23}). "
                "Ensure 15m candles span the full 2023 period."
            )
        if len(agg_24) <= warmup_24:
            raise BacktestInsufficientDataError(
                f"No evaluation candles for {symbol} {tf} in 2024 after aggregation "
                f"(total={len(agg_24)}, warmup={warmup_24}). "
                "Ensure 15m candles span the full 2024 period."
            )

        # Pre-compute indicator results once per (timeframe, year); reused across scenarios.
        ind_results_23 = IndicatorCalculator(ind_config).calculate(agg_23)
        ind_results_24 = IndicatorCalculator(ind_config).calculate(agg_24)

        for scenario in COST_SCENARIO_NAMES:
            fee, slip = COST_SCENARIOS[scenario]

            cfg_23 = BacktestConfig(
                symbol=symbol,
                interval=tf,
                start_ms=start_ms_2023,
                end_ms=end_ms_2023,
                initial_capital=initial_capital,
                fee_percentage=fee,
                slippage_percentage=slip,
                force_close_at_end=force_close_at_end,
            )
            result_23 = _run_frozen_single(
                agg_23, warmup_23, cfg_23, ind_results_23, ind_config, strat_config
            )
            raw[(tf, scenario, "2023")] = result_23

            # 2024 uses compounded initial_capital (= 2023 final equity for this scenario).
            cfg_24 = BacktestConfig(
                symbol=symbol,
                interval=tf,
                start_ms=start_ms_2024,
                end_ms=end_ms_2024,
                initial_capital=result_23.final_equity,
                fee_percentage=fee,
                slippage_percentage=slip,
                force_close_at_end=force_close_at_end,
            )
            result_24 = _run_frozen_single(
                agg_24, warmup_24, cfg_24, ind_results_24, ind_config, strat_config
            )
            raw[(tf, scenario, "2024")] = result_24

    return _ComparisonCore(
        raw=raw,
        agg_by_tf_year=agg_by_tf_year,
        warmup_by_tf_year=warmup_by_tf_year,
        candles_15m_by_year=candles_15m_by_year,
        start_ms_by_year=start_ms_by_year,
        end_ms_by_year=end_ms_by_year,
    )


def _build_report(core: _ComparisonCore) -> TimeframeCostReport:
    """Build the TimeframeCostReport from the shared _ComparisonCore."""
    from app.market_data.interval_utils import INTERVAL_MS

    raw = core.raw

    # ---- Build TimeframeCostResult objects ----
    results: list[TimeframeCostResult] = []
    for tf in TIMEFRAMES:
        tf_ms = INTERVAL_MS[tf]
        for scenario in COST_SCENARIO_NAMES:
            for year in ("2023", "2024"):
                result = raw[(tf, scenario, year)]
                no_cost = raw[(tf, "NO_COSTS", year)]

                cost_drag = no_cost.total_return_pct - result.total_return_pct
                slippage_cost = max(
                    _ZERO,
                    cost_drag * result.initial_capital / _HUNDRED - result.total_fees,
                )

                trades = result.trades
                avg_dur: Decimal | None = None
                if trades:
                    durations = [
                        Decimal(str(t.exit_exec_time - t.entry_exec_time)) / Decimal(str(tf_ms))
                        for t in trades
                    ]
                    avg_dur = sum(durations, _ZERO) / Decimal(str(len(durations)))

                n_trades = len(trades)
                avg_gross = (
                    sum((t.gross_pnl for t in trades), _ZERO) / Decimal(str(n_trades))
                    if n_trades
                    else None
                )
                avg_net = (
                    sum((t.net_pnl for t in trades), _ZERO) / Decimal(str(n_trades))
                    if n_trades
                    else None
                )
                avg_cost = (
                    (result.total_fees + slippage_cost) / Decimal(str(n_trades))
                    if n_trades
                    else None
                )

                results.append(
                    TimeframeCostResult(
                        timeframe=tf,
                        cost_scenario=scenario,
                        year=year,
                        initial_capital=result.initial_capital,
                        final_equity=result.final_equity,
                        return_pct=result.total_return_pct,
                        buy_and_hold_return_pct=result.buy_and_hold_return_pct,
                        total_trades=result.total_trades,
                        win_rate_pct=result.win_rate_pct,
                        profit_factor=result.profit_factor,
                        max_drawdown_pct=result.max_drawdown_pct,
                        total_fees=result.total_fees,
                        slippage_cost=slippage_cost,
                        exposure_pct=result.exposure_pct,
                        cost_drag=cost_drag,
                        avg_trade_duration_candles=avg_dur,
                        median_net_pnl=_median_net_pnl(trades),
                        avg_gross_pnl=avg_gross,
                        avg_net_pnl=avg_net,
                        avg_cost_per_trade=avg_cost,
                    )
                )

    # ---- Build yearly summaries ----
    yearly: list[TimeframeCostYearlySummary] = []
    for tf in TIMEFRAMES:
        for scenario in COST_SCENARIO_NAMES:
            r23 = raw[(tf, scenario, "2023")]
            r24 = raw[(tf, scenario, "2024")]
            nc23 = raw[(tf, "NO_COSTS", "2023")]
            nc24 = raw[(tf, "NO_COSTS", "2024")]

            ret_23 = r23.total_return_pct / _HUNDRED
            ret_24 = r24.total_return_pct / _HUNDRED
            combined = ((_ONE + ret_23) * (_ONE + ret_24) - _ONE) * _HUNDRED
            positive = sum(1 for r in (r23.total_return_pct, r24.total_return_pct) if r > _ZERO)
            worst_dd = max(r23.max_drawdown_pct, r24.max_drawdown_pct)

            pf23 = r23.profit_factor
            pf24 = r24.profit_factor
            if pf23 is not None and pf24 is not None:
                worst_pf: Decimal | None = min(pf23, pf24)
            elif pf23 is not None:
                worst_pf = pf23
            elif pf24 is not None:
                worst_pf = pf24
            else:
                worst_pf = None

            yearly.append(
                TimeframeCostYearlySummary(
                    timeframe=tf,
                    cost_scenario=scenario,
                    return_pct_2023=r23.total_return_pct,
                    return_pct_2024=r24.total_return_pct,
                    combined_return_pct=combined,
                    compounded_initial_2023=r23.initial_capital,
                    compounded_final_2023=r23.final_equity,
                    compounded_initial_2024=r24.initial_capital,
                    compounded_final_2024=r24.final_equity,
                    positive_years=positive,
                    worst_drawdown_pct=worst_dd,
                    worst_profit_factor=worst_pf,
                    total_trades_2023=r23.total_trades,
                    total_trades_2024=r24.total_trades,
                    total_fees_2023=r23.total_fees,
                    total_fees_2024=r24.total_fees,
                    cost_drag_2023=nc23.total_return_pct - r23.total_return_pct,
                    cost_drag_2024=nc24.total_return_pct - r24.total_return_pct,
                    buy_and_hold_2023=r23.buy_and_hold_return_pct,
                    buy_and_hold_2024=r24.buy_and_hold_return_pct,
                )
            )

    # ---- Build robustness flags ----
    robustness: list[TimeframeCostRobustness] = []
    for tf in TIMEFRAMES:
        base_23 = raw[(tf, "BASE_COSTS", "2023")]
        base_24 = raw[(tf, "BASE_COSTS", "2024")]
        ls_23 = raw[(tf, "LOW_SLIPPAGE", "2023")]
        ls_24 = raw[(tf, "LOW_SLIPPAGE", "2024")]
        cons_23 = raw[(tf, "CONSERVATIVE", "2023")]
        cons_24 = raw[(tf, "CONSERVATIVE", "2024")]

        ls_positive = ls_23.total_return_pct > _ZERO or ls_24.total_return_pct > _ZERO
        base_not_positive = base_23.total_return_pct <= _ZERO or base_24.total_return_pct <= _ZERO
        depends_ls = ls_positive and base_not_positive

        fails_cons = cons_23.total_return_pct < _ZERO or cons_24.total_return_pct < _ZERO

        for scenario in COST_SCENARIO_NAMES:
            r23 = raw[(tf, scenario, "2023")]
            r24 = raw[(tf, scenario, "2024")]

            positive_both = r23.total_return_pct > _ZERO and r24.total_return_pct > _ZERO

            pf23 = r23.profit_factor
            pf24 = r24.profit_factor
            pf_both = pf23 is not None and pf24 is not None and pf23 > _ONE and pf24 > _ONE

            total_trades = r23.total_trades + r24.total_trades

            robustness.append(
                TimeframeCostRobustness(
                    timeframe=tf,
                    cost_scenario=scenario,
                    positive_both_years=positive_both,
                    profit_factor_above_1_both_years=pf_both,
                    depends_on_low_slippage=depends_ls,
                    fails_with_conservative=fails_cons,
                    low_trade_count=total_trades < 20,
                )
            )

    return TimeframeCostReport(
        results=results,
        yearly_summary=yearly,
        robustness=robustness,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_timeframe_cost_comparison(
    symbol: str,
    initial_capital: Decimal,
    candles_15m_2023: list[Candle],
    candles_15m_2024: list[Candle],
    start_ms_2023: int,
    start_ms_2024: int,
    end_ms_2023: int,
    end_ms_2024: int,
    force_close_at_end: bool = True,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
) -> TimeframeCostReport:
    """Run frozen ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY across timeframes and cost scenarios.

    Parameters
    ----------
    symbol:
        Trading pair symbol (e.g. ``"BTCUSDT"``).
    initial_capital:
        Starting quote balance for 2023.  2024 uses 2023 final equity (compounded).
    candles_15m_2023:
        15m source candles for 2023, including extended warmup prefix.  Must be
        ordered ascending by open_time.
    candles_15m_2024:
        15m source candles for 2024, including extended warmup prefix.
    start_ms_2023 / start_ms_2024:
        Inclusive lower bound of the evaluation period for each year (ms UTC).
        Candles with open_time < start_ms are treated as warmup.
    end_ms_2023 / end_ms_2024:
        Exclusive upper bound of the evaluation period (ms UTC).
    force_close_at_end:
        Whether to force-close any open position at period end.
    indicator_config / strategy_config:
        Overrides for indicator and strategy parameters.  Defaults used when None.

    Returns
    -------
    TimeframeCostReport
        24 per-(timeframe, scenario, year) results, 12 yearly summaries, 12 robustness flags.

    Notes
    -----
    Indicator periods are NOT scaled across timeframes.  EMA-200 means 200 candles of
    whichever timeframe is used, not the same calendar duration.
    PAPER/TEST only.  No result implies profitability.
    Past results do NOT predict future performance.
    """
    ind_config = indicator_config or IndicatorConfig()
    strat_config = strategy_config or StrategyEngineConfig()
    core = _run_comparison_core(
        symbol=symbol,
        initial_capital=initial_capital,
        candles_15m_2023=candles_15m_2023,
        candles_15m_2024=candles_15m_2024,
        start_ms_2023=start_ms_2023,
        start_ms_2024=start_ms_2024,
        end_ms_2023=end_ms_2023,
        end_ms_2024=end_ms_2024,
        force_close_at_end=force_close_at_end,
        ind_config=ind_config,
        strat_config=strat_config,
    )
    return _build_report(core)


def run_timeframe_cost_audit(
    symbol: str,
    initial_capital: Decimal,
    candles_15m_2023: list[Candle],
    candles_15m_2024: list[Candle],
    start_ms_2023: int,
    start_ms_2024: int,
    end_ms_2023: int,
    end_ms_2024: int,
    force_close_at_end: bool = True,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
) -> tuple[TimeframeCostReport, TimeframeCostAuditReport]:
    """Run Stage 5.2C.1 audit: comparison report + data integrity audit.

    Runs all backtests once; builds both the cost-comparison report and the
    full audit report from the same raw results.
    PAPER/TEST only.  No result implies profitability.
    Past results do NOT predict future performance.
    """
    from app.backtesting.timeframe_cost_audit import build_audit_from_raw

    ind_config = indicator_config or IndicatorConfig()
    strat_config = strategy_config or StrategyEngineConfig()
    core = _run_comparison_core(
        symbol=symbol,
        initial_capital=initial_capital,
        candles_15m_2023=candles_15m_2023,
        candles_15m_2024=candles_15m_2024,
        start_ms_2023=start_ms_2023,
        start_ms_2024=start_ms_2024,
        end_ms_2023=end_ms_2023,
        end_ms_2024=end_ms_2024,
        force_close_at_end=force_close_at_end,
        ind_config=ind_config,
        strat_config=strat_config,
    )
    report = _build_report(core)
    audit = build_audit_from_raw(
        raw=core.raw,
        agg_by_tf_year=core.agg_by_tf_year,
        warmup_by_tf_year=core.warmup_by_tf_year,
        candles_15m_by_year=core.candles_15m_by_year,
        start_ms_by_year=core.start_ms_by_year,
        end_ms_by_year=core.end_ms_by_year,
    )
    return report, audit
