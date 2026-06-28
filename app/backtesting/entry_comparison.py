"""Stage 5.2B.1 — Audited entry variant comparison.

Runs 7 entry variants × 2 exit configurations = 14 combinations, all at 25 %
capital allocation.

Entry variants (additive filters applied after the V1 baseline signal):
  ENTRY_V1_BASELINE         — no additional filter; all V1 BUY signals pass
  ENTRY_V2_TREND_SLOPE      — EMA50 ascending 8 candles + EMA200 ascending 16 candles
  ENTRY_V3_ALIGNED_TREND    — EMA20>EMA50>EMA200 (aligned + ascending)
  ENTRY_V4_SEPARATION_0     — (EMA20−EMA50)/EMA50 × 100 ≥ 0 %
  ENTRY_V4_SEPARATION_005   — (EMA20−EMA50)/EMA50 × 100 ≥ 0.05 %
  ENTRY_V4_SEPARATION_010   — (EMA20−EMA50)/EMA50 × 100 ≥ 0.10 %
  ENTRY_V5_HIGHER_TIMEFRAME — 1h confirmation built from 4 × 15m candles

All variants share the SAME V1 base strategy engine so baseline_buy_candidates
is identical across all variants.  Filters can only REDUCE (or leave unchanged)
the set of passing candidates — they never CREATE new signals.

Signal-count invariants (enforced per combination):
  filter_passed_candidates + filter_rejected_candidates = baseline_buy_candidates
  executed_buys + blocked_by_open_position           = filter_passed_candidates

Equity conventions:
  standalone  — each year uses the original initial_capital (e.g. 10 000 USDT)
  compounded  — 2023 final equity becomes 2024 initial capital, per combo

Development data: 2023.  Validation data: 2024.  2025 untouched.

PAPER/TEST only. No variant is declared optimal.
Past results do NOT predict future performance.
"""

from __future__ import annotations

import hashlib
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
    "ENTRY_V4_SEPARATION_0",
    "ENTRY_V4_SEPARATION_005",
    "ENTRY_V4_SEPARATION_010",
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
class TradeIdentifier:
    """Deterministic identifier for a completed trade.

    Trade matching across variants uses signal_timestamp as the primary key
    because identical candle data produces identical BUY signal timestamps.
    PAPER/TEST only.
    """

    symbol: str
    interval: str
    signal_timestamp: int  # entry_signal_time (open_time of BUY signal candle)
    execution_timestamp: int  # entry_exec_time
    entry_price: Decimal
    exit_timestamp: int | None  # exit_exec_time; None for forced close with no exit
    exit_reason: str  # "|".join(trade.exit_reasons)

    def digest(self) -> str:
        """First 16 hex chars of SHA-256 over all fields."""
        key = (
            f"{self.symbol}|{self.interval}|{self.signal_timestamp}"
            f"|{self.execution_timestamp}|{self.entry_price}"
            f"|{self.exit_timestamp}|{self.exit_reason}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SignalCounts:
    """Buy-signal pipeline statistics for one (entry_variant, exit_config) combination.

    Invariants (enforced by _FilteredStrategyEngine):
      filter_passed_candidates + filter_rejected_candidates = baseline_buy_candidates
      executed_buys + blocked_by_open_position           = filter_passed_candidates

    PAPER/TEST only.
    """

    baseline_buy_candidates: int  # V1 BUY signals (no position filter, same for all variants)
    filter_passed_candidates: int  # baseline candidates that passed the entry filter
    filter_rejected_candidates: int  # baseline candidates blocked by the entry filter
    executed_buys: int  # signals that resulted in a trade = result.total_trades
    blocked_by_open_position: int  # filter-passed signals skipped (position already open)


@dataclass(frozen=True)
class FilterAnalysis:
    """Trade-level comparison of a filtered variant vs ENTRY_V1_BASELINE.

    Trade matching uses entry_signal_time.  PnL metrics are approximate because
    capital paths diverge after the first filtered trade.
    PAPER/TEST only — not a profitability claim.
    """

    baseline_candidates: int  # V1 BUY candidates (= SignalCounts.baseline_buy_candidates)
    passed_candidates: int  # filter-passed candidates (= SignalCounts.filter_passed_candidates)
    rejected_candidates: int  # baseline signals rejected by filter
    executed_buys: int  # actual trades entered (= result.total_trades)
    blocked_by_open_position: int  # filter-passed but position was open

    trades_conserved: int  # trades with same signal_time in both V1 and this variant
    trades_eliminated: int  # V1 trades NOT in this variant (by signal_time)
    trades_added: int  # this variant trades NOT in V1 (should be 0 for all standard filters)

    pnl_conserved: Decimal  # net_pnl of conserved trades (this variant's version)
    pnl_eliminated: Decimal  # net_pnl of eliminated V1 trades (approx; capital paths differ)

    return_pct_vs_baseline: Decimal  # this_return − V1_return (same exit config)
    max_dd_vs_baseline: Decimal  # this_max_dd − V1_max_dd
    trade_count_vs_baseline: int  # this_trades − V1_trades
    exposure_vs_baseline: Decimal  # this_exposure − V1_exposure


@dataclass(frozen=True)
class EntryVariantResult:
    """Metrics for one (entry_variant, exit_config) combination.

    standalone_initial_capital and standalone_final_equity always reflect
    the original per-year starting capital (e.g. 10 000 USDT), regardless
    of compounding.  result.final_equity may reflect a compounded starting
    capital when this comes from run_entry_multi_period.
    PAPER/TEST only. No result implies profitability.
    """

    entry_variant: str
    exit_config_name: str
    result: BacktestResult  # actual run result
    result_nc: BacktestResult  # zero-cost run result
    standalone_initial_capital: Decimal  # original per-year capital (e.g. 10 000)
    standalone_final_equity: Decimal  # standalone_initial × (1 + return_pct / 100)
    signal_counts: SignalCounts
    filter_analysis: FilterAnalysis
    trade_identifiers: list[TradeIdentifier]
    sl_exits: int
    tp_exits: int
    crossover_exits: int
    median_net_pnl: Decimal | None
    avg_trade_pnl: Decimal | None


@dataclass(frozen=True)
class EntryComparisonReport:
    """14-combo entry × exit comparison matrix for one period.

    combinations has len(ENTRY_VARIANT_NAMES) × len(EXIT_CONFIG_NAMES) rows = 14 total.
    PAPER/TEST only.
    """

    combinations: list[EntryVariantResult]
    bah_return_pct: Decimal
    period_label: str


@dataclass(frozen=True)
class EntryYearlySummary:
    """Cross-year statistics for one (entry_variant, exit_config) combo.

    Standalone: each year starts with the original initial_capital.
    Compounded: 2023 final equity becomes 2024 initial capital.
    PAPER/TEST only.
    """

    entry_variant: str
    exit_config_name: str

    # Per-year returns (scale-invariant; same whether standalone or compounded)
    return_pct_2023: Decimal
    return_pct_2024: Decimal

    # Standalone equity (each year starts with original initial_capital)
    standalone_initial: Decimal  # = original initial_capital (e.g. 10 000)
    standalone_final_2023: Decimal  # standalone_initial × (1 + ret_23 / 100)
    standalone_final_2024: Decimal  # standalone_initial × (1 + ret_24 / 100)

    # Compounded equity (2023 final → 2024 initial)
    compounded_initial_2023: Decimal  # = standalone_initial (always 10 000)
    compounded_final_2023: Decimal  # = standalone_final_2023 (same starting capital)
    compounded_initial_2024: Decimal  # = compounded_final_2023
    compounded_final_2024: Decimal  # compounded_initial_2024 × (1 + ret_24 / 100)

    combined_return_pct: Decimal  # ((1+ret_23)×(1+ret_24) − 1) × 100

    positive_years: int
    worst_drawdown_pct: Decimal
    total_trades_2023: int
    total_trades_2024: int
    win_rate_pct_2023: Decimal | None
    win_rate_pct_2024: Decimal | None
    profit_factor_2023: Decimal | None
    profit_factor_2024: Decimal | None
    total_fees_2023: Decimal
    total_fees_2024: Decimal


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
    """StrategyEngine subclass that applies an entry filter before BUY decisions.

    All variants share the SAME underlying base engine (V1 config).  Filters
    are applied after the base engine signals BUY, so they can only reduce —
    never increase — the number of signals.

    Signal counting (all counts incremented during engine.run()):
      baseline_buy_candidates: times base engine (no position context) returns BUY
      filter_passed_candidates: subset that also pass the entry filter
      filter_rejected_candidates: subset blocked by the entry filter
      blocked_by_open_position: filter-passed signals skipped (position was open)

    After run(), set executed_buys = result.total_trades from outside.
    Invariants hold:
      filter_passed + filter_rejected = baseline_buy_candidates
      executed_buys + blocked_by_open_position = filter_passed_candidates
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

        # Signal pipeline counters (incremented during evaluate())
        self.baseline_buy_candidates: int = 0
        self.filter_passed_candidates: int = 0
        self.filter_rejected_candidates: int = 0
        self.blocked_by_open_position: int = 0

    def evaluate(
        self,
        result: IndicatorResult,
        position: PositionContext | None = None,
    ) -> StrategyDecision:
        # Always check the base engine WITHOUT position context to count candidates.
        # StrategyEngine is stateless; calling twice with different position args is safe.
        candidate = self._filter_base.evaluate(result, position=None)

        if candidate.action == StrategyAction.BUY:
            self.baseline_buy_candidates += 1
            idx = self._idx_by_time.get(result.open_time, -1)
            filter_ok = idx >= 0 and passes_entry_filter(
                idx, self._all_results, self._entry_filter, self._htf_bars
            )

            if filter_ok:
                self.filter_passed_candidates += 1
                if position is not None:
                    # Position is open: signal is blocked by open position.
                    # Delegate to the real position-aware evaluation for SELL/WAIT handling.
                    self.blocked_by_open_position += 1
                    return self._filter_base.evaluate(result, position)
                # No open position: execute the BUY.
                return candidate

            # Filter rejected this signal.
            self.filter_rejected_candidates += 1
            if position is not None:
                # Delegate for SELL/WAIT handling.
                return self._filter_base.evaluate(result, position)
            # No position and filter blocked: return WAIT.
            return StrategyDecision(
                action=StrategyAction.WAIT,
                symbol=candidate.symbol,
                interval=candidate.interval,
                candle_open_time=candidate.candle_open_time,
                candle_close_time=candidate.candle_close_time,
                close_price=candidate.close_price,
                strategy_name=candidate.strategy_name,
                strategy_version=candidate.strategy_version,
                reasons=candidate.reasons,
                failed_conditions=candidate.failed_conditions,
                indicators_snapshot=candidate.indicators_snapshot,
                warmup_complete=candidate.warmup_complete,
                has_open_position=candidate.has_open_position,
                generated_at=candidate.generated_at,
            )

        # Not a BUY candidate — delegate with real position context.
        return self._filter_base.evaluate(result, position)


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
    if variant_name == "ENTRY_V4_SEPARATION_0":
        return EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=Decimal("0"),
        )
    if variant_name == "ENTRY_V4_SEPARATION_005":
        return EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=Decimal("0.05"),
        )
    if variant_name == "ENTRY_V4_SEPARATION_010":
        return EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=Decimal("0.10"),
        )
    if variant_name == "ENTRY_V5_HIGHER_TIMEFRAME":
        return EntryFilterConfig(filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME)
    raise ValueError(f"Unknown entry variant: {variant_name!r}")


def _trade_identifiers(
    trades: list[BacktestTrade], config: BacktestConfig
) -> list[TradeIdentifier]:
    return [
        TradeIdentifier(
            symbol=config.symbol,
            interval=config.interval,
            signal_timestamp=t.entry_signal_time,
            execution_timestamp=t.entry_exec_time,
            entry_price=t.entry_exec_price,
            exit_timestamp=t.exit_exec_time,
            exit_reason="|".join(t.exit_reasons),
        )
        for t in trades
    ]


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
    fengine: _FilteredStrategyEngine,
    baseline_trades: list[BacktestTrade],
    filtered_trades: list[BacktestTrade],
    baseline_result: BacktestResult,
    filtered_result: BacktestResult,
) -> FilterAnalysis:
    baseline_times = {t.entry_signal_time for t in baseline_trades}
    filtered_times = {t.entry_signal_time for t in filtered_trades}

    conserved_times = baseline_times & filtered_times
    eliminated_times = baseline_times - filtered_times
    added_times = filtered_times - baseline_times

    conserved = [t for t in filtered_trades if t.entry_signal_time in conserved_times]
    eliminated = [t for t in baseline_trades if t.entry_signal_time in eliminated_times]

    pnl_conserved = sum((t.net_pnl for t in conserved), _ZERO)
    pnl_eliminated = sum((t.net_pnl for t in eliminated), _ZERO)

    executed = filtered_result.total_trades
    blocked = fengine.blocked_by_open_position

    return FilterAnalysis(
        baseline_candidates=fengine.baseline_buy_candidates,
        passed_candidates=fengine.filter_passed_candidates,
        rejected_candidates=fengine.filter_rejected_candidates,
        executed_buys=executed,
        blocked_by_open_position=blocked,
        trades_conserved=len(conserved_times),
        trades_eliminated=len(eliminated_times),
        trades_added=len(added_times),
        pnl_conserved=pnl_conserved,
        pnl_eliminated=pnl_eliminated,
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
    """Run one combo with costs and without costs; return (result, result_nc, engine)."""
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


def _build_variant_result(
    variant_name: str,
    exit_name: str,
    result: BacktestResult,
    result_nc: BacktestResult,
    fengine: _FilteredStrategyEngine,
    standalone_initial: Decimal,
    baseline_result: BacktestResult,
) -> EntryVariantResult:
    trades = result.trades
    avg_pnl: Decimal | None = None
    if trades:
        avg_pnl = sum((t.net_pnl for t in trades), _ZERO) / Decimal(str(len(trades)))

    standalone_final = standalone_initial * (Decimal("1") + result.total_return_pct / _HUNDRED)

    return EntryVariantResult(
        entry_variant=variant_name,
        exit_config_name=exit_name,
        result=result,
        result_nc=result_nc,
        standalone_initial_capital=standalone_initial,
        standalone_final_equity=standalone_final,
        signal_counts=SignalCounts(
            baseline_buy_candidates=fengine.baseline_buy_candidates,
            filter_passed_candidates=fengine.filter_passed_candidates,
            filter_rejected_candidates=fengine.filter_rejected_candidates,
            executed_buys=result.total_trades,
            blocked_by_open_position=fengine.blocked_by_open_position,
        ),
        filter_analysis=_compute_filter_analysis(
            fengine, baseline_result.trades, trades, baseline_result, result
        ),
        trade_identifiers=_trade_identifiers(trades, result.config),
        sl_exits=_count_exit(trades, str(ReasonCode.ATR_STOP_LOSS)),
        tp_exits=_count_exit(trades, str(ReasonCode.RISK_REWARD_TAKE_PROFIT)),
        crossover_exits=_count_exit(trades, str(ReasonCode.BEARISH_CROSSOVER)),
        median_net_pnl=_median_net_pnl(trades),
        avg_trade_pnl=avg_pnl,
    )


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
    """Run all 14 (entry_variant × exit_config) combinations and return a report.

    Each combination runs twice: with configured costs and with zero costs.
    All 7 variants share the same V1 base engine and same candle data (no look-ahead).
    Filters reduce — never increase — the set of executable signals.
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

    # All variants use the same V1 base engine for consistent baseline counting.
    base_engine = StrategyEngine(strat_config)
    standalone_initial = config.initial_capital

    # First pass: run all 14 combos and collect raw results
    raw: dict[tuple[str, str], tuple] = {}
    bah = _ZERO

    for exit_name, risk_cfg in [
        (_EXIT_STOP_ONLY, _RISK_STOP_ONLY),
        (_EXIT_STOP_TP, _RISK_STOP_TP),
    ]:
        for variant_name in ENTRY_VARIANT_NAMES:
            filter_cfg = _filter_cfg_for(variant_name)
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

    # Second pass: build EntryVariantResult with filter analysis vs V1 baseline
    combinations: list[EntryVariantResult] = []
    for exit_name in EXIT_CONFIG_NAMES:
        baseline_result = raw[("ENTRY_V1_BASELINE", exit_name)][0]
        for variant_name in ENTRY_VARIANT_NAMES:
            result, result_nc, fengine = raw[(variant_name, exit_name)]
            combinations.append(
                _build_variant_result(
                    variant_name,
                    exit_name,
                    result,
                    result_nc,
                    fengine,
                    standalone_initial,
                    baseline_result,
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

    Each (entry_variant, exit_config) combo carries its 2023 final equity
    forward as the 2024 initial capital, independently of other combos.

    Equity conventions:
      standalone  — each year uses the original initial_capital (e.g. 10 000 USDT)
      compounded  — 2023 final equity → 2024 initial capital (per combo)

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

    # All variants use the same V1 base engine.
    base_engine_23 = StrategyEngine(strat_config)
    base_engine_24 = StrategyEngine(strat_config)

    standalone_initial = config_2023.initial_capital  # e.g. 10 000 USDT

    # Run 2023 combos
    raw_23: dict[tuple[str, str], tuple] = {}
    bah_23 = _ZERO

    for exit_name, risk_cfg in [
        (_EXIT_STOP_ONLY, _RISK_STOP_ONLY),
        (_EXIT_STOP_TP, _RISK_STOP_TP),
    ]:
        for variant_name in ENTRY_VARIANT_NAMES:
            filter_cfg = _filter_cfg_for(variant_name)
            htf = htf_23 if variant_name == "ENTRY_V5_HIGHER_TIMEFRAME" else None

            result, result_nc, fengine = _run_combo(
                candles_2023,
                warmup_2023,
                config_2023,
                risk_cfg,
                ind_config,
                base_engine_23,
                filter_cfg,
                all_results_23,
                htf,
            )
            raw_23[(variant_name, exit_name)] = (result, result_nc, fengine)
            if bah_23 == _ZERO:
                bah_23 = result.buy_and_hold_return_pct

    # Run 2024 combos with capital compounding (2023 final equity → 2024 initial capital)
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
            risk_cfg = RiskExitConfig(
                use_take_profit=risk_cfg_base.use_take_profit,
                maximum_holding_candles=risk_cfg_base.maximum_holding_candles,
                use_bearish_crossover_exit=risk_cfg_base.use_bearish_crossover_exit,
                position_allocation_percentage=risk_cfg_base.position_allocation_percentage,
                atr_stop_multiplier=risk_cfg_base.atr_stop_multiplier,
                reward_to_risk_ratio=risk_cfg_base.reward_to_risk_ratio,
            )
            filter_cfg = _filter_cfg_for(variant_name)
            htf = htf_24 if variant_name == "ENTRY_V5_HIGHER_TIMEFRAME" else None

            result, result_nc, fengine = _run_combo(
                candles_2024,
                warmup_2024,
                cfg_24_compounded,
                risk_cfg,
                ind_config,
                base_engine_24,
                filter_cfg,
                all_results_24,
                htf,
            )
            raw_24[(variant_name, exit_name)] = (result, result_nc, fengine)
            if bah_24 == _ZERO:
                bah_24 = result.buy_and_hold_return_pct

    # Build per-year EntryComparisonReport
    def _build_report(
        raw: dict[tuple[str, str], tuple],
        bah: Decimal,
        period_label: str,
        standalone_init: Decimal,
    ) -> EntryComparisonReport:
        combinations: list[EntryVariantResult] = []
        for en in EXIT_CONFIG_NAMES:
            bl_result = raw[("ENTRY_V1_BASELINE", en)][0]
            for vn in ENTRY_VARIANT_NAMES:
                res, res_nc, feng = raw[(vn, en)]
                combinations.append(
                    _build_variant_result(vn, en, res, res_nc, feng, standalone_init, bl_result)
                )
        return EntryComparisonReport(
            combinations=combinations,
            bah_return_pct=bah,
            period_label=period_label,
        )

    report_23 = _build_report(raw_23, bah_23, "2023", standalone_initial)
    # For 2024 standalone, use config_2024.initial_capital (the original, not compounded)
    standalone_initial_24 = config_2024.initial_capital
    report_24 = _build_report(raw_24, bah_24, "2024", standalone_initial_24)

    # Build yearly summary with both standalone and compounded equity
    yearly: list[EntryYearlySummary] = []
    for exit_name in EXIT_CONFIG_NAMES:
        for variant_name in ENTRY_VARIANT_NAMES:
            r23 = raw_23[(variant_name, exit_name)][0]
            r24 = raw_24[(variant_name, exit_name)][0]

            ret_23 = r23.total_return_pct / _HUNDRED
            ret_24 = r24.total_return_pct / _HUNDRED

            # Standalone equity (independent years, both start with initial_capital)
            sa_init = standalone_initial
            sa_final_23 = sa_init * (Decimal("1") + ret_23)
            sa_final_24 = standalone_initial_24 * (Decimal("1") + ret_24)

            # Compounded equity
            comp_init_23 = sa_init
            comp_final_23 = sa_final_23  # same, since 2023 starts with sa_init
            comp_init_24 = comp_final_23
            comp_final_24 = comp_init_24 * (Decimal("1") + ret_24)

            combined = ((Decimal("1") + ret_23) * (Decimal("1") + ret_24) - Decimal("1")) * _HUNDRED
            positive = sum(1 for r in (r23.total_return_pct, r24.total_return_pct) if r > _ZERO)
            worst_dd = max(r23.max_drawdown_pct, r24.max_drawdown_pct)

            yearly.append(
                EntryYearlySummary(
                    entry_variant=variant_name,
                    exit_config_name=exit_name,
                    return_pct_2023=r23.total_return_pct,
                    return_pct_2024=r24.total_return_pct,
                    standalone_initial=sa_init,
                    standalone_final_2023=sa_final_23,
                    standalone_final_2024=sa_final_24,
                    compounded_initial_2023=comp_init_23,
                    compounded_final_2023=comp_final_23,
                    compounded_initial_2024=comp_init_24,
                    compounded_final_2024=comp_final_24,
                    combined_return_pct=combined,
                    positive_years=positive,
                    worst_drawdown_pct=worst_dd,
                    total_trades_2023=r23.total_trades,
                    total_trades_2024=r24.total_trades,
                    win_rate_pct_2023=r23.win_rate_pct,
                    win_rate_pct_2024=r24.win_rate_pct,
                    profit_factor_2023=r23.profit_factor,
                    profit_factor_2024=r24.profit_factor,
                    total_fees_2023=r23.total_fees,
                    total_fees_2024=r24.total_fees,
                )
            )

    return EntryMultiPeriodReport(
        period_2023=report_23,
        period_2024=report_24,
        yearly_summary=yearly,
    )
