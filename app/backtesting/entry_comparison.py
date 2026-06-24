"""Stage 5.2B — Controlled entry variant comparison.

Runs 5 entry variants × 2 exit configurations = 10 combinations, all at 25 %
capital allocation.

Entry variants (additive filters on top of V1 baseline entry logic):
  ENTRY_V1_BASELINE         — no additional filter
  ENTRY_V2_TREND_SLOPE      — EMA50 ascending 8 candles + EMA200 ascending 16 candles
  ENTRY_V3_ALIGNED_TREND    — EMA20>EMA50>EMA200 (aligned + ascending) + crossover lookback
  ENTRY_V4_CROSSOVER_STRENGTH — (EMA20−EMA50)/EMA50 × 100 ≥ threshold
  ENTRY_V5_HIGHER_TIMEFRAME — 1h confirmation built from 4 × 15m candles

Exit configurations:
  V2_STOP_ONLY — ATR stop-loss, no take-profit, bearish crossover secondary
  V2_STOP_TP   — ATR stop-loss, 2R take-profit, bearish crossover secondary

Development data: 2023.  Validation data: 2024.  2025 untouched.
Capital compounding: 2023 final equity → 2024 initial capital (per combo).

PAPER/TEST only. No variant is declared optimal.
Past results do NOT predict future performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from app.backtesting.config import BacktestConfig
from app.backtesting.risk_exit_config import RiskExitConfig
from app.backtesting.v2_engine import V2BacktestEngine
from app.indicators.schemas import IndicatorConfig, IndicatorResult
from app.models.candle import Candle
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine
from app.strategy.entry_filter import (
    EntryFilterConfig,
    EntryFilterType,
    HtfBarData,
    build_htf_bars,
    passes_entry_filter,
)
from app.strategy.reasons import ReasonCode
from app.strategy.schemas import PositionContext, StrategyAction, StrategyDecision

if TYPE_CHECKING:
    from app.backtesting.schemas import BacktestResult, BacktestTrade

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_TWO = Decimal("2")

ENTRY_VARIANT_NAMES: tuple[str, ...] = (
    "ENTRY_V1_BASELINE",
    "ENTRY_V2_TREND_SLOPE",
    "ENTRY_V3_ALIGNED_TREND",
    "ENTRY_V4_CROSSOVER_STRENGTH",
    "ENTRY_V5_HIGHER_TIMEFRAME",
)

_EXIT_STOP_ONLY = "V2_STOP_ONLY"
_EXIT_STOP_TP = "V2_STOP_TP"
EXIT_CONFIG_NAMES: tuple[str, ...] = (_EXIT_STOP_ONLY, _EXIT_STOP_TP)

_ALLOCATION_25 = Decimal("25")

_RISK_STOP_ONLY = RiskExitConfig(
    use_take_profit=False,
    maximum_holding_candles=0,
    use_bearish_crossover_exit=True,
    position_allocation_percentage=_ALLOCATION_25,
)
_RISK_STOP_TP = RiskExitConfig(
    use_take_profit=True,
    maximum_holding_candles=0,
    use_bearish_crossover_exit=True,
    position_allocation_percentage=_ALLOCATION_25,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalCounts:
    """Buy-signal statistics for one (entry_variant, exit_config) combination.

    PAPER/TEST only.
    """

    signals_detected: int  # BUY decisions from base engine when no position open
    buys_executed: int  # actual trades entered (= total_trades in result)
    blocked_by_filter: int  # signals that passed base engine but were blocked by filter


@dataclass(frozen=True)
class FilterAnalysis:
    """Comparison of filtered variant vs ENTRY_V1_BASELINE with the same exit config.

    Trade matching is by entry_signal_time.  Metrics are approximate because
    capital paths diverge after the first filtered trade.
    PAPER/TEST only — not a profitability claim.
    """

    entries_eliminated: int  # V1 entry signal times NOT in this variant
    entries_added: int  # This variant signal times NOT in V1 (V3 with looser crossover)
    eliminated_pnl: Decimal  # Sum of net_pnl of eliminated V1 trades (approx)
    eliminated_winners: int  # Winners among those V1 trades
    eliminated_losers: int  # Losers among those V1 trades
    return_pct_vs_baseline: Decimal  # this_return − V1_return
    max_dd_vs_baseline: Decimal  # this_max_dd − V1_max_dd
    trade_count_vs_baseline: int  # this_trades − V1_trades
    exposure_vs_baseline: Decimal  # this_exposure − V1_exposure


@dataclass(frozen=True)
class EntryVariantResult:
    """Metrics for one (entry_variant, exit_config) combination.

    PAPER/TEST only. No result implies profitability.
    """

    entry_variant: str
    exit_config_name: str
    result: BacktestResult
    result_nc: BacktestResult
    signal_counts: SignalCounts
    filter_analysis: FilterAnalysis
    sl_exits: int
    tp_exits: int
    crossover_exits: int
    median_net_pnl: Decimal | None
    avg_trade_pnl: Decimal | None


@dataclass(frozen=True)
class EntryComparisonReport:
    """10-combo entry × exit comparison matrix for one period.

    combinations has len(ENTRY_VARIANT_NAMES) × len(EXIT_CONFIG_NAMES) rows = 10 total.
    PAPER/TEST only.
    """

    combinations: list[EntryVariantResult]
    bah_return_pct: Decimal
    period_label: str


@dataclass(frozen=True)
class EntryYearlySummary:
    """Cross-year statistics for one (entry_variant, exit_config) combo."""

    entry_variant: str
    exit_config_name: str
    return_pct_2023: Decimal
    return_pct_2024: Decimal
    combined_return_pct: Decimal
    capital_2023_end: Decimal  # = 2024 initial capital (compounding)
    capital_2024_end: Decimal
    positive_years: int
    worst_drawdown_pct: Decimal
    total_trades_2023: int
    total_trades_2024: int


@dataclass(frozen=True)
class EntryMultiPeriodReport:
    """Multi-year entry comparison with per-combo capital compounding.

    2023 final equity becomes 2024 initial capital for each combo independently.
    PAPER/TEST only. Past results do NOT predict future performance.
    """

    period_2023: EntryComparisonReport
    period_2024: EntryComparisonReport
    yearly_summary: list[EntryYearlySummary]


# ---------------------------------------------------------------------------
# Private: FilteredStrategyEngine
# ---------------------------------------------------------------------------


class _FilteredStrategyEngine(StrategyEngine):
    """Subclass of StrategyEngine that applies an entry filter before BUY decisions.

    SELL decisions are passed through unchanged.
    Tracks detected_signals (BUY before filter) and blocked_count (blocked by filter).
    """

    def __init__(
        self,
        base_engine: StrategyEngine,
        all_results: list[IndicatorResult],
        entry_filter: EntryFilterConfig,
        htf_bars: list[HtfBarData] | None,
    ) -> None:
        super().__init__(base_engine.config)
        self._filter_base = base_engine
        self._all_results = all_results
        self._entry_filter = entry_filter
        self._htf_bars = htf_bars
        self._idx_by_time: dict[int, int] = {r.open_time: i for i, r in enumerate(all_results)}
        self.detected_signals: int = 0
        self.blocked_count: int = 0

    def evaluate(
        self,
        result: IndicatorResult,
        position: PositionContext | None = None,
    ) -> StrategyDecision:
        decision = self._filter_base.evaluate(result, position)
        if decision.action == StrategyAction.BUY:
            self.detected_signals += 1
            idx = self._idx_by_time.get(result.open_time, -1)
            if idx >= 0 and not passes_entry_filter(
                idx, self._all_results, self._entry_filter, self._htf_bars
            ):
                self.blocked_count += 1
                return StrategyDecision(
                    action=StrategyAction.WAIT,
                    symbol=decision.symbol,
                    interval=decision.interval,
                    candle_open_time=decision.candle_open_time,
                    candle_close_time=decision.candle_close_time,
                    close_price=decision.close_price,
                    strategy_name=decision.strategy_name,
                    strategy_version=decision.strategy_version,
                    reasons=decision.reasons,
                    failed_conditions=decision.failed_conditions,
                    indicators_snapshot=decision.indicators_snapshot,
                    warmup_complete=decision.warmup_complete,
                    has_open_position=decision.has_open_position,
                    generated_at=decision.generated_at,
                )
        return decision


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zero_cost(config: BacktestConfig) -> BacktestConfig:
    return BacktestConfig(
        symbol=config.symbol,
        interval=config.interval,
        start_ms=config.start_ms,
        end_ms=config.end_ms,
        initial_capital=config.initial_capital,
        fee_percentage=_ZERO,
        slippage_percentage=_ZERO,
        force_close_at_end=config.force_close_at_end,
    )


def _filter_cfg_for(variant_name: str) -> EntryFilterConfig:
    if variant_name == "ENTRY_V1_BASELINE":
        return EntryFilterConfig(filter_type=EntryFilterType.V1_BASELINE)
    if variant_name == "ENTRY_V2_TREND_SLOPE":
        return EntryFilterConfig(filter_type=EntryFilterType.V2_TREND_SLOPE)
    if variant_name == "ENTRY_V3_ALIGNED_TREND":
        return EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            crossover_lookback_candles=1,
        )
    if variant_name == "ENTRY_V4_CROSSOVER_STRENGTH":
        return EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=Decimal("0.05"),
        )
    if variant_name == "ENTRY_V5_HIGHER_TIMEFRAME":
        return EntryFilterConfig(filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME)
    raise ValueError(f"Unknown entry variant: {variant_name!r}")


def _base_engine_for(variant_name: str, strat_config: StrategyEngineConfig) -> StrategyEngine:
    if variant_name == "ENTRY_V3_ALIGNED_TREND":
        # V3 relaxes the crossover requirement in the base engine so the filter
        # controls the crossover-lookback window explicitly.
        cfg = strat_config.model_copy(update={"require_bullish_crossover": False})
        return StrategyEngine(cfg)
    return StrategyEngine(strat_config)


def _count_exit(trades: list[BacktestTrade], code: str) -> int:
    return sum(1 for t in trades if code in t.exit_reasons)


def _median_net_pnl(trades: list[BacktestTrade]) -> Decimal | None:
    if not trades:
        return None
    sorted_pnls = sorted(t.net_pnl for t in trades)
    n = len(sorted_pnls)
    mid = n // 2
    if n % 2 == 1:
        return sorted_pnls[mid]
    return (sorted_pnls[mid - 1] + sorted_pnls[mid]) / _TWO


def _compute_filter_analysis(
    baseline_trades: list[BacktestTrade],
    filtered_trades: list[BacktestTrade],
    baseline_result: BacktestResult,
    filtered_result: BacktestResult,
) -> FilterAnalysis:
    baseline_times = {t.entry_signal_time for t in baseline_trades}
    filtered_times = {t.entry_signal_time for t in filtered_trades}

    eliminated_times = baseline_times - filtered_times
    added_times = filtered_times - baseline_times

    eliminated = [t for t in baseline_trades if t.entry_signal_time in eliminated_times]
    elim_pnl = sum((t.net_pnl for t in eliminated), _ZERO)
    elim_winners = sum(1 for t in eliminated if t.net_pnl > _ZERO)
    elim_losers = sum(1 for t in eliminated if t.net_pnl <= _ZERO)

    return FilterAnalysis(
        entries_eliminated=len(eliminated_times),
        entries_added=len(added_times),
        eliminated_pnl=elim_pnl,
        eliminated_winners=elim_winners,
        eliminated_losers=elim_losers,
        return_pct_vs_baseline=(
            filtered_result.total_return_pct - baseline_result.total_return_pct
        ),
        max_dd_vs_baseline=(filtered_result.max_drawdown_pct - baseline_result.max_drawdown_pct),
        trade_count_vs_baseline=(filtered_result.total_trades - baseline_result.total_trades),
        exposure_vs_baseline=(filtered_result.exposure_pct - baseline_result.exposure_pct),
    )


def _run_combo(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    risk_cfg: RiskExitConfig,
    ind_config: IndicatorConfig,
    base_engine: StrategyEngine,
    filter_cfg: EntryFilterConfig,
    all_results: list[IndicatorResult],
    htf_bars: list[HtfBarData] | None,
) -> tuple[BacktestResult, BacktestResult, _FilteredStrategyEngine]:
    """Run one combo (with costs and without costs).  Returns (result, result_nc, engine)."""
    filter_engine = _FilteredStrategyEngine(base_engine, all_results, filter_cfg, htf_bars)
    v2 = V2BacktestEngine(
        config=config,
        risk_exit_config=risk_cfg,
        strategy_engine=filter_engine,
        indicator_config=ind_config,
    )
    result = v2.run(all_candles, warmup_len)

    filter_engine_nc = _FilteredStrategyEngine(base_engine, all_results, filter_cfg, htf_bars)
    v2_nc = V2BacktestEngine(
        config=_zero_cost(config),
        risk_exit_config=risk_cfg,
        strategy_engine=filter_engine_nc,
        indicator_config=ind_config,
    )
    result_nc = v2_nc.run(all_candles, warmup_len)

    return result, result_nc, filter_engine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_entry_comparison(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
    period_label: str = "",
) -> EntryComparisonReport:
    """Run all 10 (entry_variant × exit_config) combinations and return a report.

    Each combination runs twice: with configured costs and with zero costs.
    All variants share identical candles and warmup period (no look-ahead).
    PAPER/TEST only. No variant is declared optimal.
    """
    from app.indicators.calculator import IndicatorCalculator

    ind_config = indicator_config or IndicatorConfig()
    strat_config = strategy_config or StrategyEngineConfig()

    # Pre-compute indicator series once (shared by all FilteredStrategyEngine instances)
    all_results = IndicatorCalculator(ind_config).calculate(all_candles)

    # Pre-compute HTF bars for V5 filter (1h from 15m: close prices only)
    closes = [c.close for c in all_candles]
    htf_bars = build_htf_bars(closes, candles_per_htf_bar=4)

    # First pass: run all 10 combos and collect raw results
    raw: dict[tuple[str, str], tuple[BacktestResult, BacktestResult, _FilteredStrategyEngine]] = {}
    bah = _ZERO

    for exit_name, risk_cfg in [
        (_EXIT_STOP_ONLY, _RISK_STOP_ONLY),
        (_EXIT_STOP_TP, _RISK_STOP_TP),
    ]:
        for variant_name in ENTRY_VARIANT_NAMES:
            filter_cfg = _filter_cfg_for(variant_name)
            base_engine = _base_engine_for(variant_name, strat_config)
            htf = htf_bars if variant_name == "ENTRY_V5_HIGHER_TIMEFRAME" else None

            result, result_nc, fengine = _run_combo(
                all_candles,
                warmup_len,
                config,
                risk_cfg,
                ind_config,
                base_engine,
                filter_cfg,
                all_results,
                htf,
            )
            raw[(variant_name, exit_name)] = (result, result_nc, fengine)
            if bah == _ZERO:
                bah = result.buy_and_hold_return_pct

    # Second pass: build EntryVariantResult with filter analysis
    combinations: list[EntryVariantResult] = []
    for exit_name in EXIT_CONFIG_NAMES:
        baseline_result = raw[("ENTRY_V1_BASELINE", exit_name)][0]
        for variant_name in ENTRY_VARIANT_NAMES:
            result, result_nc, fengine = raw[(variant_name, exit_name)]
            trades = result.trades

            avg_pnl: Decimal | None = None
            if trades:
                avg_pnl = sum((t.net_pnl for t in trades), _ZERO) / Decimal(str(len(trades)))

            combinations.append(
                EntryVariantResult(
                    entry_variant=variant_name,
                    exit_config_name=exit_name,
                    result=result,
                    result_nc=result_nc,
                    signal_counts=SignalCounts(
                        signals_detected=fengine.detected_signals,
                        buys_executed=result.total_trades,
                        blocked_by_filter=fengine.blocked_count,
                    ),
                    filter_analysis=_compute_filter_analysis(
                        baseline_result.trades, trades, baseline_result, result
                    ),
                    sl_exits=_count_exit(trades, str(ReasonCode.ATR_STOP_LOSS)),
                    tp_exits=_count_exit(trades, str(ReasonCode.RISK_REWARD_TAKE_PROFIT)),
                    crossover_exits=_count_exit(trades, str(ReasonCode.BEARISH_CROSSOVER)),
                    median_net_pnl=_median_net_pnl(trades),
                    avg_trade_pnl=avg_pnl,
                )
            )

    return EntryComparisonReport(
        combinations=combinations,
        bah_return_pct=bah,
        period_label=period_label,
    )


def run_entry_multi_period(
    candles_2023: list[Candle],
    warmup_2023: int,
    config_2023: BacktestConfig,
    candles_2024: list[Candle],
    warmup_2024: int,
    config_2024: BacktestConfig,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
) -> EntryMultiPeriodReport:
    """Run entry comparison for 2023 and 2024 with per-combo capital compounding.

    Each (entry_variant, exit_config) combination carries its 2023 final equity
    forward as the 2024 initial capital, independently of other combinations.
    PAPER/TEST only. Past results do NOT predict future performance.
    """
    from app.indicators.calculator import IndicatorCalculator

    ind_config = indicator_config or IndicatorConfig()
    strat_config = strategy_config or StrategyEngineConfig()

    all_results_23 = IndicatorCalculator(ind_config).calculate(candles_2023)
    all_results_24 = IndicatorCalculator(ind_config).calculate(candles_2024)

    closes_23 = [c.close for c in candles_2023]
    closes_24 = [c.close for c in candles_2024]
    htf_23 = build_htf_bars(closes_23, candles_per_htf_bar=4)
    htf_24 = build_htf_bars(closes_24, candles_per_htf_bar=4)

    # Run 2023 combos
    raw_23: dict[tuple[str, str], tuple] = {}
    bah_23 = _ZERO

    for exit_name, risk_cfg in [
        (_EXIT_STOP_ONLY, _RISK_STOP_ONLY),
        (_EXIT_STOP_TP, _RISK_STOP_TP),
    ]:
        for variant_name in ENTRY_VARIANT_NAMES:
            filter_cfg = _filter_cfg_for(variant_name)
            base_engine = _base_engine_for(variant_name, strat_config)
            htf = htf_23 if variant_name == "ENTRY_V5_HIGHER_TIMEFRAME" else None

            result, result_nc, fengine = _run_combo(
                candles_2023,
                warmup_2023,
                config_2023,
                risk_cfg,
                ind_config,
                base_engine,
                filter_cfg,
                all_results_23,
                htf,
            )
            raw_23[(variant_name, exit_name)] = (result, result_nc, fengine)
            if bah_23 == _ZERO:
                bah_23 = result.buy_and_hold_return_pct

    # Run 2024 combos with capital compounding
    raw_24: dict[tuple[str, str], tuple] = {}
    bah_24 = _ZERO

    for exit_name, risk_cfg_base in [
        (_EXIT_STOP_ONLY, _RISK_STOP_ONLY),
        (_EXIT_STOP_TP, _RISK_STOP_TP),
    ]:
        for variant_name in ENTRY_VARIANT_NAMES:
            final_equity_23 = raw_23[(variant_name, exit_name)][0].final_equity
            cfg_24_compounded = BacktestConfig(
                symbol=config_2024.symbol,
                interval=config_2024.interval,
                start_ms=config_2024.start_ms,
                end_ms=config_2024.end_ms,
                initial_capital=final_equity_23,
                fee_percentage=config_2024.fee_percentage,
                slippage_percentage=config_2024.slippage_percentage,
                force_close_at_end=config_2024.force_close_at_end,
            )
            # Rebuild risk config with the same allocation and parameters
            risk_cfg = RiskExitConfig(
                use_take_profit=risk_cfg_base.use_take_profit,
                maximum_holding_candles=risk_cfg_base.maximum_holding_candles,
                use_bearish_crossover_exit=risk_cfg_base.use_bearish_crossover_exit,
                position_allocation_percentage=risk_cfg_base.position_allocation_percentage,
                atr_stop_multiplier=risk_cfg_base.atr_stop_multiplier,
                reward_to_risk_ratio=risk_cfg_base.reward_to_risk_ratio,
            )

            filter_cfg = _filter_cfg_for(variant_name)
            base_engine = _base_engine_for(variant_name, strat_config)
            htf = htf_24 if variant_name == "ENTRY_V5_HIGHER_TIMEFRAME" else None

            result, result_nc, fengine = _run_combo(
                candles_2024,
                warmup_2024,
                cfg_24_compounded,
                risk_cfg,
                ind_config,
                base_engine,
                filter_cfg,
                all_results_24,
                htf,
            )
            raw_24[(variant_name, exit_name)] = (result, result_nc, fengine)
            if bah_24 == _ZERO:
                bah_24 = result.buy_and_hold_return_pct

    # Build EntryComparisonReport for each year
    def _build_report(
        raw: dict[tuple[str, str], tuple[BacktestResult, BacktestResult, _FilteredStrategyEngine]],
        bah: Decimal,
        period_label: str,
    ) -> EntryComparisonReport:
        combinations: list[EntryVariantResult] = []
        for exit_name in EXIT_CONFIG_NAMES:
            baseline_result = raw[("ENTRY_V1_BASELINE", exit_name)][0]
            for variant_name in ENTRY_VARIANT_NAMES:
                result, result_nc, fengine = raw[(variant_name, exit_name)]
                trades = result.trades
                avg_pnl: Decimal | None = None
                if trades:
                    avg_pnl = sum((t.net_pnl for t in trades), _ZERO) / Decimal(str(len(trades)))
                combinations.append(
                    EntryVariantResult(
                        entry_variant=variant_name,
                        exit_config_name=exit_name,
                        result=result,
                        result_nc=result_nc,
                        signal_counts=SignalCounts(
                            signals_detected=fengine.detected_signals,
                            buys_executed=result.total_trades,
                            blocked_by_filter=fengine.blocked_count,
                        ),
                        filter_analysis=_compute_filter_analysis(
                            baseline_result.trades, trades, baseline_result, result
                        ),
                        sl_exits=_count_exit(trades, str(ReasonCode.ATR_STOP_LOSS)),
                        tp_exits=_count_exit(trades, str(ReasonCode.RISK_REWARD_TAKE_PROFIT)),
                        crossover_exits=_count_exit(trades, str(ReasonCode.BEARISH_CROSSOVER)),
                        median_net_pnl=_median_net_pnl(trades),
                        avg_trade_pnl=avg_pnl,
                    )
                )
        return EntryComparisonReport(
            combinations=combinations,
            bah_return_pct=bah,
            period_label=period_label,
        )

    report_23 = _build_report(raw_23, bah_23, "2023")
    report_24 = _build_report(raw_24, bah_24, "2024")

    # Build yearly summary with capital compounding
    yearly: list[EntryYearlySummary] = []
    for exit_name in EXIT_CONFIG_NAMES:
        for variant_name in ENTRY_VARIANT_NAMES:
            r23 = raw_23[(variant_name, exit_name)][0]
            r24 = raw_24[(variant_name, exit_name)][0]

            ret_23 = r23.total_return_pct / _HUNDRED
            ret_24 = r24.total_return_pct / _HUNDRED
            combined = ((1 + ret_23) * (1 + ret_24) - 1) * _HUNDRED
            positive = sum(1 for r in (r23.total_return_pct, r24.total_return_pct) if r > _ZERO)
            worst_dd = max(r23.max_drawdown_pct, r24.max_drawdown_pct)

            yearly.append(
                EntryYearlySummary(
                    entry_variant=variant_name,
                    exit_config_name=exit_name,
                    return_pct_2023=r23.total_return_pct,
                    return_pct_2024=r24.total_return_pct,
                    combined_return_pct=combined,
                    capital_2023_end=r23.final_equity,
                    capital_2024_end=r24.final_equity,
                    positive_years=positive,
                    worst_drawdown_pct=worst_dd,
                    total_trades_2023=r23.total_trades,
                    total_trades_2024=r24.total_trades,
                )
            )

    return EntryMultiPeriodReport(
        period_2023=report_23,
        period_2024=report_24,
        yearly_summary=yearly,
    )
