"""Stage 6.0 — Donchian breakout family evaluation.

Evaluates 8 pre-registered configurations (A–D × 1h/4h) over 2021–2025.
Each year runs standalone (initial_capital = 10 000 per year) AND compounded
(final equity carried forward year-to-year).  Three cost scenarios per run.

Qualification: ALL 9 criteria must pass.
Selection order (if multiple qualified): years_positive DESC, worst_drawdown ASC,
best_year_pct ASC, profit_factor DESC.

Abort policy: if any year has zero eval candles for any timeframe → raise before
running any engine; show missing ranges; never present partial results as complete.

PAPER/TEST only. No real orders. Past results do NOT predict future performance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.donchian_engine import DonchianBreakoutEngine, DonchianConfig
from app.backtesting.exceptions import BacktestInsufficientDataError
from app.backtesting.schemas import BacktestResult
from app.backtesting.timeframe_aggregator import aggregate_candles, warmup_len_for
from app.models.candle import Candle

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")

# ---------------------------------------------------------------------------
# Pre-registered constants — DO NOT MODIFY after results are seen.
# ---------------------------------------------------------------------------

BREAKOUT_SYMBOL = "BTCUSDT"
BREAKOUT_SOURCE_INTERVAL = "15m"
BREAKOUT_SOURCE_INTERVAL_MS = 900_000

BREAKOUT_TIMEFRAMES = ("1h", "4h")
BREAKOUT_SCENARIO_NAMES = ("NO_COSTS", "BASE_COSTS", "CONSERVATIVE")
BREAKOUT_YEARS = ("2021", "2022", "2023", "2024", "2025")
BREAKOUT_INITIAL_CAPITAL = Decimal("10000")

BREAKOUT_SCENARIOS: dict[str, tuple[Decimal, Decimal]] = {
    "NO_COSTS": (Decimal("0"), Decimal("0")),
    "BASE_COSTS": (Decimal("0.1"), Decimal("0.05")),
    "CONSERVATIVE": (Decimal("0.1"), Decimal("0.10")),
}

BREAKOUT_CONFIGS: dict[str, DonchianConfig] = {
    "A": DonchianConfig(
        entry_lookback=20,
        exit_lookback=10,
        atr_period=14,
        atr_multiplier=Decimal("2.0"),
    ),
    "B": DonchianConfig(
        entry_lookback=20,
        exit_lookback=10,
        atr_period=14,
        atr_multiplier=Decimal("3.0"),
    ),
    "C": DonchianConfig(
        entry_lookback=55,
        exit_lookback=20,
        atr_period=14,
        atr_multiplier=Decimal("2.0"),
    ),
    "D": DonchianConfig(
        entry_lookback=55,
        exit_lookback=20,
        atr_period=14,
        atr_multiplier=Decimal("3.0"),
    ),
}

# Year boundary timestamps (ms, UTC midnight)
_YEAR_START_MS: dict[str, int] = {
    "2021": 1_609_459_200_000,
    "2022": 1_640_995_200_000,
    "2023": 1_672_531_200_000,
    "2024": 1_704_067_200_000,
    "2025": 1_735_689_600_000,
}
_YEAR_END_MS: dict[str, int] = {
    "2021": 1_640_995_200_000,
    "2022": 1_672_531_200_000,
    "2023": 1_704_067_200_000,
    "2024": 1_735_689_600_000,
    "2025": 1_767_225_600_000,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakoutYearlyResult:
    """Standalone per-year result (initial_capital = 10 000 each year)."""

    config_id: str
    year: str
    scenario: str
    initial_capital: Decimal
    final_equity: Decimal
    return_pct: Decimal
    total_trades: int
    win_rate_pct: Decimal | None
    profit_factor: Decimal | None
    max_drawdown_pct: Decimal
    total_fees: Decimal
    buy_and_hold_return_pct: Decimal


@dataclass(frozen=True)
class BreakoutCompoundedResult:
    """Compounded 5-year result: equity carries forward year-to-year."""

    config_id: str
    scenario: str
    initial_capital: Decimal
    final_equity: Decimal
    total_return_pct: Decimal
    profit_factor: Decimal | None
    max_drawdown_pct: Decimal
    total_trades: int
    years_with_data: int


@dataclass(frozen=True)
class QualificationCheck:
    name: str
    passed: bool
    value: str
    threshold: str


@dataclass(frozen=True)
class BreakoutRobustnessSummary:
    """Qualification result for one config_id."""

    config_id: str
    is_qualified: bool
    qualification_checks: tuple[QualificationCheck, ...]


@dataclass(frozen=True)
class BreakoutDataCoverage:
    config_id: str
    year: str
    eval_candles: int


@dataclass(frozen=True)
class BreakoutAudit:
    candles_15m_total: int
    symbol: str
    source_interval: str
    years_covered: tuple[str, ...]
    missing_data_years: tuple[str, ...]
    config_signal_hashes: dict[str, str]
    violations: tuple[str, ...]


@dataclass
class BreakoutFamilyReport:
    """Complete Stage 6.0 report. PAPER/TEST only."""

    symbol: str
    source_interval: str
    evaluation_years: tuple[str, ...]
    initial_capital: Decimal

    yearly_results: list[BreakoutYearlyResult]
    compounded_results: list[BreakoutCompoundedResult]
    robustness: list[BreakoutRobustnessSummary]
    data_coverage: list[BreakoutDataCoverage]

    # [config_id][year][scenario] → BacktestResult
    raw_results: dict[str, dict[str, dict[str, BacktestResult]]]

    audit: BreakoutAudit

    paper_test_disclaimer: str = (
        "PAPER/TEST only. No real money. Past results do NOT predict future performance."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_breakout_family(
    symbol: str,
    all_candles_15m: list[Candle],
    initial_capital: Decimal = BREAKOUT_INITIAL_CAPITAL,
    force_close_at_end: bool = True,
) -> BreakoutFamilyReport:
    """Evaluate all 8 Donchian configurations over 2021–2025.

    Aborts (raises BacktestInsufficientDataError) if any year × timeframe
    combination has zero evaluation candles.

    all_candles_15m: 15m candles with warmup prefix before 2021-01-01.
    """
    # Aggregate to 1h and 4h once
    candles_by_tf: dict[str, list[Candle]] = {
        "1h": aggregate_candles(all_candles_15m, "1h", BREAKOUT_SOURCE_INTERVAL_MS),
        "4h": aggregate_candles(all_candles_15m, "4h", BREAKOUT_SOURCE_INTERVAL_MS),
    }

    # Pre-flight: verify every (timeframe, year) has eval candles
    violations: list[str] = []
    missing_years: set[str] = set()
    for tf in BREAKOUT_TIMEFRAMES:
        candles = candles_by_tf[tf]
        for yr in BREAKOUT_YEARS:
            start_ms = _YEAR_START_MS[yr]
            end_ms = _YEAR_END_MS[yr]
            eval_count = sum(1 for c in candles if start_ms <= c.open_time < end_ms)
            if eval_count == 0:
                msg = f"No {tf} eval candles for {yr} (range [{start_ms}, {end_ms}))"
                violations.append(msg)
                missing_years.add(yr)

    if violations:
        raise BacktestInsufficientDataError(
            "Missing data — aborting before engine execution.\n"
            + "\n".join(f"  • {v}" for v in violations)
        )

    # Run all configs × years × scenarios
    yearly_results: list[BreakoutYearlyResult] = []
    compounded_results: list[BreakoutCompoundedResult] = []
    data_coverage: list[BreakoutDataCoverage] = []
    raw_results: dict[str, dict[str, dict[str, BacktestResult]]] = {}
    signal_hashes: dict[str, str] = {}

    for cfg_letter, dcfg in BREAKOUT_CONFIGS.items():
        for tf in BREAKOUT_TIMEFRAMES:
            config_id = f"{cfg_letter}_{tf}"
            candles_tf = candles_by_tf[tf]
            raw_results[config_id] = {}

            # ---- Standalone per-year runs ----
            for yr in BREAKOUT_YEARS:
                start_ms = _YEAR_START_MS[yr]
                end_ms = _YEAR_END_MS[yr]
                warmup_len = warmup_len_for(candles_tf, start_ms)
                yr_candles = [c for c in candles_tf if c.open_time < end_ms]

                eval_count = sum(1 for c in candles_tf if start_ms <= c.open_time < end_ms)
                data_coverage.append(
                    BreakoutDataCoverage(
                        config_id=config_id,
                        year=yr,
                        eval_candles=eval_count,
                    )
                )

                yr_raw: dict[str, BacktestResult] = {}
                for scenario in BREAKOUT_SCENARIO_NAMES:
                    fee_pct, slip_pct = BREAKOUT_SCENARIOS[scenario]
                    bt_cfg = BacktestConfig(
                        symbol=symbol,
                        interval=tf,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        initial_capital=initial_capital,
                        fee_percentage=fee_pct,
                        slippage_percentage=slip_pct,
                        force_close_at_end=force_close_at_end,
                    )
                    engine = DonchianBreakoutEngine(
                        config=bt_cfg,
                        donchian_config=dcfg,
                    )
                    result = engine.run(yr_candles, warmup_len)
                    yr_raw[scenario] = result

                    yearly_results.append(
                        BreakoutYearlyResult(
                            config_id=config_id,
                            year=yr,
                            scenario=scenario,
                            initial_capital=result.initial_capital,
                            final_equity=result.final_equity,
                            return_pct=result.total_return_pct,
                            total_trades=result.total_trades,
                            win_rate_pct=result.win_rate_pct,
                            profit_factor=result.profit_factor,
                            max_drawdown_pct=result.max_drawdown_pct,
                            total_fees=result.total_fees,
                            buy_and_hold_return_pct=result.buy_and_hold_return_pct,
                        )
                    )

                raw_results[config_id][yr] = yr_raw

            # Signal hash for BASE_COSTS (determinism audit)
            bc_entry_times: list[int] = []
            for yr in BREAKOUT_YEARS:
                bc_result = raw_results[config_id][yr].get("BASE_COSTS")
                if bc_result:
                    bc_entry_times.extend(t.entry_signal_time for t in bc_result.trades)
            digest = hashlib.sha256(
                ",".join(str(t) for t in sorted(bc_entry_times)).encode()
            ).hexdigest()[:16]
            signal_hashes[config_id] = digest

            # ---- Compounded runs (equity carries forward year-to-year) ----
            for scenario in BREAKOUT_SCENARIO_NAMES:
                fee_pct, slip_pct = BREAKOUT_SCENARIOS[scenario]
                running_capital = initial_capital
                total_trades_compound = 0
                max_dd_compound = _ZERO
                total_gross_wins = _ZERO
                total_gross_losses = _ZERO
                years_with_data = 0

                for yr in BREAKOUT_YEARS:
                    start_ms = _YEAR_START_MS[yr]
                    end_ms = _YEAR_END_MS[yr]
                    warmup_len = warmup_len_for(candles_tf, start_ms)
                    yr_candles = [c for c in candles_tf if c.open_time < end_ms]

                    bt_cfg = BacktestConfig(
                        symbol=symbol,
                        interval=tf,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        initial_capital=running_capital,
                        fee_percentage=fee_pct,
                        slippage_percentage=slip_pct,
                        force_close_at_end=force_close_at_end,
                    )
                    engine = DonchianBreakoutEngine(
                        config=bt_cfg,
                        donchian_config=dcfg,
                    )
                    yr_result = engine.run(yr_candles, warmup_len)

                    running_capital = yr_result.final_equity
                    total_trades_compound += yr_result.total_trades
                    max_dd_compound = max(max_dd_compound, yr_result.max_drawdown_pct)
                    years_with_data += 1

                    for t in yr_result.trades:
                        if t.net_pnl > _ZERO:
                            total_gross_wins += t.net_pnl
                        else:
                            total_gross_losses += abs(t.net_pnl)

                compound_pf: Decimal | None = (
                    total_gross_wins / total_gross_losses if total_gross_losses > _ZERO else None
                )
                compound_return_pct = (
                    (running_capital - initial_capital) / initial_capital * _HUNDRED
                )
                compounded_results.append(
                    BreakoutCompoundedResult(
                        config_id=config_id,
                        scenario=scenario,
                        initial_capital=initial_capital,
                        final_equity=running_capital,
                        total_return_pct=compound_return_pct,
                        profit_factor=compound_pf,
                        max_drawdown_pct=max_dd_compound,
                        total_trades=total_trades_compound,
                        years_with_data=years_with_data,
                    )
                )

    # Build qualification
    robustness = _build_robustness(yearly_results, compounded_results)

    audit = BreakoutAudit(
        candles_15m_total=len(all_candles_15m),
        symbol=symbol,
        source_interval=BREAKOUT_SOURCE_INTERVAL,
        years_covered=tuple(BREAKOUT_YEARS),
        missing_data_years=tuple(sorted(missing_years)),
        config_signal_hashes=signal_hashes,
        violations=(),
    )

    return BreakoutFamilyReport(
        symbol=symbol,
        source_interval=BREAKOUT_SOURCE_INTERVAL,
        evaluation_years=tuple(BREAKOUT_YEARS),
        initial_capital=initial_capital,
        yearly_results=yearly_results,
        compounded_results=compounded_results,
        robustness=robustness,
        data_coverage=data_coverage,
        raw_results=raw_results,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Qualification helpers
# ---------------------------------------------------------------------------


def _get_compounded(
    compounded_results: list[BreakoutCompoundedResult],
    config_id: str,
    scenario: str,
) -> BreakoutCompoundedResult | None:
    for r in compounded_results:
        if r.config_id == config_id and r.scenario == scenario:
            return r
    return None


def _get_yearly(
    yearly_results: list[BreakoutYearlyResult],
    config_id: str,
    scenario: str,
) -> list[BreakoutYearlyResult]:
    return [r for r in yearly_results if r.config_id == config_id and r.scenario == scenario]


def _build_robustness(
    yearly_results: list[BreakoutYearlyResult],
    compounded_results: list[BreakoutCompoundedResult],
) -> list[BreakoutRobustnessSummary]:
    result: list[BreakoutRobustnessSummary] = []
    config_ids = [f"{c}_{tf}" for c in BREAKOUT_CONFIGS for tf in BREAKOUT_TIMEFRAMES]

    for config_id in config_ids:
        checks = _compute_qualification_checks(config_id, yearly_results, compounded_results)
        is_qualified = all(c.passed for c in checks)
        result.append(
            BreakoutRobustnessSummary(
                config_id=config_id,
                is_qualified=is_qualified,
                qualification_checks=tuple(checks),
            )
        )

    return result


def _compute_qualification_checks(
    config_id: str,
    yearly_results: list[BreakoutYearlyResult],
    compounded_results: list[BreakoutCompoundedResult],
) -> list[QualificationCheck]:
    checks: list[QualificationCheck] = []

    bc_compound = _get_compounded(compounded_results, config_id, "BASE_COSTS")
    cons_compound = _get_compounded(compounded_results, config_id, "CONSERVATIVE")
    bc_yearly = _get_yearly(yearly_results, config_id, "BASE_COSTS")

    # 1. Compounded BASE_COSTS return > 0
    bc_return = bc_compound.total_return_pct if bc_compound else Decimal("-99999")
    checks.append(
        QualificationCheck(
            name="compounded_base_positive",
            passed=bc_return > _ZERO,
            value=str(bc_return),
            threshold="> 0",
        )
    )

    # 2. Compounded CONSERVATIVE return > 0
    cons_return = cons_compound.total_return_pct if cons_compound else Decimal("-99999")
    checks.append(
        QualificationCheck(
            name="compounded_conservative_positive",
            passed=cons_return > _ZERO,
            value=str(cons_return),
            threshold="> 0",
        )
    )

    # 3. ≥3 positive years (BASE_COSTS standalone)
    positive_years = sum(1 for r in bc_yearly if r.return_pct > _ZERO)
    checks.append(
        QualificationCheck(
            name="years_positive_ge3",
            passed=positive_years >= 3,
            value=str(positive_years),
            threshold=">= 3",
        )
    )

    # 4. Profit factor > 1.10 (compounded BASE_COSTS)
    pf = bc_compound.profit_factor if bc_compound else None
    pf_ok = pf is not None and pf > Decimal("1.10")
    checks.append(
        QualificationCheck(
            name="profit_factor_gt1_10",
            passed=pf_ok,
            value=str(pf) if pf is not None else "None",
            threshold="> 1.10",
        )
    )

    # 5. Max drawdown < 15% (compounded BASE_COSTS)
    max_dd = bc_compound.max_drawdown_pct if bc_compound else Decimal("99999")
    checks.append(
        QualificationCheck(
            name="max_drawdown_lt15",
            passed=max_dd < Decimal("15"),
            value=str(max_dd),
            threshold="< 15%",
        )
    )

    # 6. Total trades ≥ 30 (compounded BASE_COSTS across all years)
    total_trades = bc_compound.total_trades if bc_compound else 0
    checks.append(
        QualificationCheck(
            name="min_trades_30",
            passed=total_trades >= 30,
            value=str(total_trades),
            threshold=">= 30",
        )
    )

    # 7. No year worse than -10% (BASE_COSTS standalone)
    worst_year = min((r.return_pct for r in bc_yearly), default=_ZERO)
    checks.append(
        QualificationCheck(
            name="no_year_below_neg10",
            passed=worst_year >= Decimal("-10"),
            value=str(worst_year),
            threshold=">= -10%",
        )
    )

    # 8. Best year ≤ 70% of total gross profit (concentration check)
    gross_by_year: dict[str, Decimal] = {}
    for yr in BREAKOUT_YEARS:
        yr_result = next((r for r in bc_yearly if r.year == yr), None)
        # Use final_equity - initial_capital as proxy for gross per year
        # (standalone run resets capital each year; net profit = final - initial)
        if yr_result is not None and yr_result.final_equity > yr_result.initial_capital:
            gross_by_year[yr] = yr_result.final_equity - yr_result.initial_capital
        else:
            gross_by_year[yr] = _ZERO

    total_gross = sum(gross_by_year.values(), _ZERO)
    best_yr_gross = max(gross_by_year.values(), default=_ZERO)
    concentration_ok = total_gross <= _ZERO or best_yr_gross <= Decimal("0.70") * total_gross
    checks.append(
        QualificationCheck(
            name="best_year_le70pct_gross",
            passed=concentration_ok,
            value=str(best_yr_gross),
            threshold=f"<= 70% of {total_gross}",
        )
    )

    # 9. No audit violations (always passes here; violations abort before this point)
    checks.append(
        QualificationCheck(
            name="no_violations",
            passed=True,
            value="0",
            threshold="0",
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Selection helpers (exported for CLI use)
# ---------------------------------------------------------------------------


def select_best_qualified(robustness: list[BreakoutRobustnessSummary]) -> str | None:
    """Return the config_id of the best qualified config, or None if none qualify.

    Selection order (all descending by preference):
      1. More positive years → better
      2. Lower worst drawdown → better
      3. Lower best-year concentration → better
      4. Higher profit factor → better
    """
    qualified = [r for r in robustness if r.is_qualified]
    if not qualified:
        return None

    def _check_val(r: BreakoutRobustnessSummary, name: str) -> str:
        for c in r.qualification_checks:
            if c.name == name:
                return c.value
        return "0"

    def _sort_key(r: BreakoutRobustnessSummary) -> tuple[int, Decimal, Decimal, Decimal]:
        years_pos = int(_check_val(r, "years_positive_ge3"))
        try:
            dd = Decimal(_check_val(r, "max_drawdown_lt15"))
        except Exception:
            dd = Decimal("99999")
        try:
            best_yr = Decimal(_check_val(r, "best_year_le70pct_gross"))
        except Exception:
            best_yr = Decimal("99999")
        pf_str = _check_val(r, "profit_factor_gt1_10")
        try:
            pf = Decimal(pf_str) if pf_str not in ("None", "") else _ZERO
        except Exception:
            pf = _ZERO
        # Sort ascending: (-years_pos, dd, best_yr, -pf)
        return (-years_pos, dd, best_yr, -pf)

    qualified.sort(key=_sort_key)
    return qualified[0].config_id
