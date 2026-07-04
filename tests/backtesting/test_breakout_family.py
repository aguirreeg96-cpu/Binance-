"""Tests for Stage 6.0 — Donchian breakout family evaluation.

Covers:
  - Pre-flight abort on missing data
  - 1h and 4h timeframes both produce eval candles
  - Signal identity across cost scenarios (same entry times)
  - Cost monotonicity (NO_COSTS >= BASE_COSTS >= CONSERVATIVE)
  - Determinism across two identical runs
  - Qualification classification (QUALIFIED / REJECTED)
  - All 9 qualification criteria tested individually
  - Export file columns and JSON keys
  - select_best_qualified selection order
  - CLI compatibility (existing arguments unaffected)
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.backtesting.breakout_exporters import (
    export_breakout_audit_json,
    export_breakout_compounded_results_csv,
    export_breakout_equity_curves_csv,
    export_breakout_report_json,
    export_breakout_robustness_summary_csv,
    export_breakout_trades_csv,
    export_breakout_yearly_results_csv,
)
from app.backtesting.breakout_family import (
    BREAKOUT_YEARS,
    BreakoutAudit,
    BreakoutCompoundedResult,
    BreakoutDataCoverage,
    BreakoutFamilyReport,
    BreakoutRobustnessSummary,
    BreakoutYearlyResult,
    QualificationCheck,
    _compute_qualification_checks,
    run_breakout_family,
    select_best_qualified,
)
from app.backtesting.donchian_engine import DonchianConfig
from app.backtesting.exceptions import BacktestInsufficientDataError
from app.models.candle import Candle

_D = Decimal
_T_15M = 900_000  # 15m in ms

_YEAR_STARTS = {
    "2021": 1_609_459_200_000,
    "2022": 1_640_995_200_000,
    "2023": 1_672_531_200_000,
    "2024": 1_704_067_200_000,
    "2025": 1_735_689_600_000,
}

# ---------------------------------------------------------------------------
# Tiny pre-registered configs — used exclusively in tests (not real configs)
# ---------------------------------------------------------------------------

_TINY_CFG_A = DonchianConfig(
    entry_lookback=3,
    exit_lookback=2,
    atr_period=3,
    atr_multiplier=Decimal("1"),
    ema_period=5,
    ema_slope_lookback=2,
    allocation_pct=Decimal("25"),
)
_TINY_CFG_B = DonchianConfig(
    entry_lookback=3,
    exit_lookback=2,
    atr_period=3,
    atr_multiplier=Decimal("2"),
    ema_period=5,
    ema_slope_lookback=2,
    allocation_pct=Decimal("25"),
)
_TINY_CONFIGS = {"A": _TINY_CFG_A, "B": _TINY_CFG_B}
_TINY_TFS = ("1h", "4h")


# ---------------------------------------------------------------------------
# Synthetic candle factory
# ---------------------------------------------------------------------------


def _make_15m(t: int, close_val: int) -> Candle:
    cl = Decimal(close_val)
    return Candle(
        symbol="BTCUSDT",
        interval="15m",
        open_time=t,
        open=cl - Decimal("2"),
        high=cl + Decimal("1"),
        low=cl - Decimal("1"),
        close=cl,
        volume=Decimal("1000"),
        close_time=t + _T_15M - 1,
        quote_asset_volume=Decimal("100000"),
        trades=100,
        taker_buy_base_volume=Decimal("500"),
        taker_buy_quote_volume=Decimal("50000"),
        is_closed=True,
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )


def _make_dataset(
    warmup_count: int = 80,
    eval_count_per_year: int = 64,
    skip_year: str | None = None,
) -> list[Candle]:
    """Synthetic 15m candles: warmup prefix + 5 years.

    Prices rise monotonically to ensure Donchian breakout signals fire.
    All groups are strictly consecutive so 1h and 4h aggregation succeeds.
    """
    candles: list[Candle] = []
    close_val = 100

    # Warmup: strictly before 2021-01-01
    warmup_start = _YEAR_STARTS["2021"] - warmup_count * _T_15M
    for i in range(warmup_count):
        candles.append(_make_15m(warmup_start + i * _T_15M, close_val))
        close_val += 2

    # Per-year eval blocks
    for yr, start_ms in _YEAR_STARTS.items():
        if yr == skip_year:
            continue
        for i in range(eval_count_per_year):
            candles.append(_make_15m(start_ms + i * _T_15M, close_val))
            close_val += 2

    return candles


def _run_tiny_family(monkeypatch, skip_year: str | None = None):
    """Helper: patch BREAKOUT_CONFIGS/TIMEFRAMES and run with tiny dataset."""
    monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
    monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", _TINY_TFS)
    dataset = _make_dataset(skip_year=skip_year)
    return run_breakout_family("BTCUSDT", dataset)


# ---------------------------------------------------------------------------
# Minimal report for export tests (no engine runs needed)
# ---------------------------------------------------------------------------


def _make_minimal_report() -> BreakoutFamilyReport:
    """Build a minimal BreakoutFamilyReport for testing export structure."""
    from app.backtesting.config import BacktestConfig
    from app.backtesting.schemas import BacktestResult

    cfg = BacktestConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_ms=_YEAR_STARTS["2021"],
        end_ms=_YEAR_STARTS["2022"],
        initial_capital=Decimal("10000"),
        fee_percentage=Decimal("0"),
        slippage_percentage=Decimal("0"),
        force_close_at_end=True,
    )
    empty_result = BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=[],
        first_candle_open_time=_YEAR_STARTS["2021"],
        last_candle_open_time=_YEAR_STARTS["2021"] + 3_600_000,
        total_candles=1,
        evaluated_candles=1,
        initial_capital=Decimal("10000"),
        final_equity=Decimal("10000"),
        total_return_pct=Decimal("0"),
        buy_and_hold_return_pct=Decimal("0"),
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate_pct=None,
        avg_win_pct=None,
        avg_loss_pct=None,
        profit_factor=None,
        expectancy_pct=None,
        max_drawdown_pct=Decimal("0"),
        exposure_pct=Decimal("0"),
        max_win_streak=0,
        max_loss_streak=0,
        total_fees=Decimal("0"),
        has_open_position_at_end=False,
    )

    yr = BreakoutYearlyResult(
        config_id="A_1h",
        year="2021",
        scenario="BASE_COSTS",
        initial_capital=Decimal("10000"),
        final_equity=Decimal("10500"),
        return_pct=Decimal("5"),
        total_trades=2,
        win_rate_pct=Decimal("100"),
        profit_factor=Decimal("2"),
        max_drawdown_pct=Decimal("3"),
        total_fees=Decimal("10"),
        buy_and_hold_return_pct=Decimal("10"),
    )
    comp = BreakoutCompoundedResult(
        config_id="A_1h",
        scenario="BASE_COSTS",
        initial_capital=Decimal("10000"),
        final_equity=Decimal("11000"),
        total_return_pct=Decimal("10"),
        profit_factor=Decimal("1.5"),
        max_drawdown_pct=Decimal("5"),
        total_trades=5,
        years_with_data=5,
    )
    check = QualificationCheck(
        name="compounded_base_positive",
        passed=True,
        value="10",
        threshold="> 0",
    )
    rob = BreakoutRobustnessSummary(
        config_id="A_1h",
        is_qualified=True,
        qualification_checks=(check,),
    )
    cov = BreakoutDataCoverage(config_id="A_1h", year="2021", eval_candles=16)
    audit = BreakoutAudit(
        candles_15m_total=100,
        symbol="BTCUSDT",
        source_interval="15m",
        years_covered=("2021",),
        missing_data_years=(),
        config_signal_hashes={"A_1h": "abc123"},
        violations=(),
    )
    return BreakoutFamilyReport(
        symbol="BTCUSDT",
        source_interval="15m",
        evaluation_years=("2021",),
        initial_capital=Decimal("10000"),
        yearly_results=[yr],
        compounded_results=[comp],
        robustness=[rob],
        data_coverage=[cov],
        raw_results={"A_1h": {"2021": {"BASE_COSTS": empty_result}}},
        audit=audit,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestPreflightAbort:
    def test_missing_one_year_aborts(self, monkeypatch):
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", _TINY_TFS)
        dataset = _make_dataset(skip_year="2023")
        with pytest.raises(BacktestInsufficientDataError, match="2023"):
            run_breakout_family("BTCUSDT", dataset)

    def test_error_message_has_bullet_points(self, monkeypatch):
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", _TINY_TFS)
        dataset = _make_dataset(skip_year="2022")
        with pytest.raises(BacktestInsufficientDataError) as exc_info:
            run_breakout_family("BTCUSDT", dataset)
        assert "•" in str(exc_info.value)

    def test_all_years_present_no_error(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        assert report is not None

    def test_empty_dataset_aborts(self, monkeypatch):
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", _TINY_TFS)
        with pytest.raises(BacktestInsufficientDataError):
            run_breakout_family("BTCUSDT", [])


class TestDataCoverage:
    def test_both_timeframes_in_coverage(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        tfs_in_coverage = {c.config_id.split("_")[1] for c in report.data_coverage}
        assert "1h" in tfs_in_coverage
        assert "4h" in tfs_in_coverage

    def test_all_five_years_in_coverage(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        covered_years = {c.year for c in report.data_coverage}
        assert covered_years == set(BREAKOUT_YEARS)

    def test_all_config_ids_in_coverage(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        covered_configs = {c.config_id for c in report.data_coverage}
        expected = {f"{k}_{tf}" for k in _TINY_CONFIGS for tf in _TINY_TFS}
        assert covered_configs == expected

    def test_eval_candles_positive_for_all_years(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        assert all(c.eval_candles > 0 for c in report.data_coverage)

    def test_audit_symbol_matches(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        assert report.audit.symbol == "BTCUSDT"
        assert report.audit.source_interval == "15m"


class TestSignalIdentity:
    """Entry signals must be identical across all 3 cost scenarios for same config."""

    def test_entry_times_same_across_scenarios(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        for config_id, yr_map in report.raw_results.items():
            for yr, scenario_map in yr_map.items():
                nc_times = sorted(t.entry_signal_time for t in scenario_map["NO_COSTS"].trades)
                bc_times = sorted(t.entry_signal_time for t in scenario_map["BASE_COSTS"].trades)
                co_times = sorted(t.entry_signal_time for t in scenario_map["CONSERVATIVE"].trades)
                assert nc_times == bc_times, f"{config_id}/{yr}: NO_COSTS vs BASE_COSTS differ"
                assert bc_times == co_times, f"{config_id}/{yr}: BASE_COSTS vs CONSERVATIVE differ"

    def test_trade_count_same_across_scenarios(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        for config_id, yr_map in report.raw_results.items():
            for yr, scenario_map in yr_map.items():
                nc_n = len(scenario_map["NO_COSTS"].trades)
                bc_n = len(scenario_map["BASE_COSTS"].trades)
                co_n = len(scenario_map["CONSERVATIVE"].trades)
                assert nc_n == bc_n == co_n, f"{config_id}/{yr}: trade counts differ"

    def test_signal_hashes_present_for_all_configs(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        expected = {f"{k}_{tf}" for k in _TINY_CONFIGS for tf in _TINY_TFS}
        assert set(report.audit.config_signal_hashes.keys()) == expected

    def test_signal_hash_is_hex_string(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        for h in report.audit.config_signal_hashes.values():
            assert isinstance(h, str) and len(h) == 16


class TestCostMonotonicity:
    """NO_COSTS return >= BASE_COSTS return >= CONSERVATIVE return."""

    def test_yearly_return_monotone(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        for yr_nc in report.yearly_results:
            if yr_nc.scenario != "NO_COSTS":
                continue
            config_id = yr_nc.config_id
            yr = yr_nc.year

            nc_ret = yr_nc.return_pct
            bc = next(
                r
                for r in report.yearly_results
                if r.config_id == config_id and r.year == yr and r.scenario == "BASE_COSTS"
            )
            co = next(
                r
                for r in report.yearly_results
                if r.config_id == config_id and r.year == yr and r.scenario == "CONSERVATIVE"
            )
            assert nc_ret >= bc.return_pct, f"{config_id}/{yr}: NO_COSTS < BASE_COSTS"
            assert bc.return_pct >= co.return_pct, f"{config_id}/{yr}: BASE_COSTS < CONSERVATIVE"

    def test_compounded_return_monotone(self, monkeypatch):
        report = _run_tiny_family(monkeypatch)
        config_ids = {r.config_id for r in report.compounded_results}
        for config_id in config_ids:
            nc = next(
                r
                for r in report.compounded_results
                if r.config_id == config_id and r.scenario == "NO_COSTS"
            )
            bc = next(
                r
                for r in report.compounded_results
                if r.config_id == config_id and r.scenario == "BASE_COSTS"
            )
            co = next(
                r
                for r in report.compounded_results
                if r.config_id == config_id and r.scenario == "CONSERVATIVE"
            )
            assert nc.total_return_pct >= bc.total_return_pct, f"{config_id}: NC < BC"
            assert bc.total_return_pct >= co.total_return_pct, f"{config_id}: BC < CO"


class TestDeterminism:
    def test_two_runs_identical(self, monkeypatch):
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", _TINY_TFS)
        dataset = _make_dataset()

        r1 = run_breakout_family("BTCUSDT", dataset)
        r2 = run_breakout_family("BTCUSDT", dataset)

        for c1, c2 in zip(r1.yearly_results, r2.yearly_results, strict=True):
            assert c1.return_pct == c2.return_pct
            assert c1.final_equity == c2.final_equity
            assert c1.total_trades == c2.total_trades

    def test_signal_hashes_stable(self, monkeypatch):
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", _TINY_TFS)
        dataset = _make_dataset()

        r1 = run_breakout_family("BTCUSDT", dataset)
        r2 = run_breakout_family("BTCUSDT", dataset)

        assert r1.audit.config_signal_hashes == r2.audit.config_signal_hashes


class TestQualificationChecks:
    """Unit-test each of the 9 qualification criteria via _compute_qualification_checks."""

    _BASE_YEARLY = [
        BreakoutYearlyResult(
            config_id="X_1h",
            year=yr,
            scenario="BASE_COSTS",
            initial_capital=Decimal("10000"),
            final_equity=Decimal("10500"),
            return_pct=Decimal("5"),
            total_trades=8,
            win_rate_pct=Decimal("62"),
            profit_factor=Decimal("1.5"),
            max_drawdown_pct=Decimal("3"),
            total_fees=Decimal("10"),
            buy_and_hold_return_pct=Decimal("20"),
        )
        for yr in BREAKOUT_YEARS
    ]
    _BASE_COMPOUNDED = [
        BreakoutCompoundedResult(
            config_id="X_1h",
            scenario="BASE_COSTS",
            initial_capital=Decimal("10000"),
            final_equity=Decimal("15000"),
            total_return_pct=Decimal("50"),
            profit_factor=Decimal("1.5"),
            max_drawdown_pct=Decimal("10"),
            total_trades=40,
            years_with_data=5,
        ),
        BreakoutCompoundedResult(
            config_id="X_1h",
            scenario="CONSERVATIVE",
            initial_capital=Decimal("10000"),
            final_equity=Decimal("14000"),
            total_return_pct=Decimal("40"),
            profit_factor=Decimal("1.3"),
            max_drawdown_pct=Decimal("12"),
            total_trades=40,
            years_with_data=5,
        ),
    ]

    def test_all_checks_pass_qualifies(self):
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, self._BASE_COMPOUNDED)
        assert all(c.passed for c in checks)

    def test_compounded_base_negative_rejects(self):
        compounded = [
            BreakoutCompoundedResult(
                config_id="X_1h",
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("9000"),
                total_return_pct=Decimal("-10"),
                profit_factor=Decimal("0.9"),
                max_drawdown_pct=Decimal("10"),
                total_trades=40,
                years_with_data=5,
            ),
            self._BASE_COMPOUNDED[1],
        ]
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, compounded)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["compounded_base_positive"]

    def test_compounded_conservative_negative_rejects(self):
        compounded = [
            self._BASE_COMPOUNDED[0],
            BreakoutCompoundedResult(
                config_id="X_1h",
                scenario="CONSERVATIVE",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("9500"),
                total_return_pct=Decimal("-5"),
                profit_factor=None,
                max_drawdown_pct=Decimal("10"),
                total_trades=40,
                years_with_data=5,
            ),
        ]
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, compounded)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["compounded_conservative_positive"]

    def test_too_few_positive_years_rejects(self):
        # Only 2 positive years (< 3)
        yearly = [
            BreakoutYearlyResult(
                config_id="X_1h",
                year=yr,
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=(
                    Decimal("9500") if yr in ("2021", "2022", "2023") else Decimal("10500")
                ),
                return_pct=Decimal("-5") if yr in ("2021", "2022", "2023") else Decimal("5"),
                total_trades=8,
                win_rate_pct=None,
                profit_factor=None,
                max_drawdown_pct=Decimal("5"),
                total_fees=Decimal("10"),
                buy_and_hold_return_pct=Decimal("20"),
            )
            for yr in BREAKOUT_YEARS
        ]
        checks = _compute_qualification_checks("X_1h", yearly, self._BASE_COMPOUNDED)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["years_positive_ge3"]

    def test_profit_factor_too_low_rejects(self):
        compounded = [
            BreakoutCompoundedResult(
                config_id="X_1h",
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("10200"),
                total_return_pct=Decimal("2"),
                profit_factor=Decimal("1.05"),  # below 1.10
                max_drawdown_pct=Decimal("10"),
                total_trades=40,
                years_with_data=5,
            ),
            self._BASE_COMPOUNDED[1],
        ]
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, compounded)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["profit_factor_gt1_10"]

    def test_profit_factor_none_rejects(self):
        compounded = [
            BreakoutCompoundedResult(
                config_id="X_1h",
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("10200"),
                total_return_pct=Decimal("2"),
                profit_factor=None,
                max_drawdown_pct=Decimal("10"),
                total_trades=40,
                years_with_data=5,
            ),
            self._BASE_COMPOUNDED[1],
        ]
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, compounded)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["profit_factor_gt1_10"]

    def test_high_drawdown_rejects(self):
        compounded = [
            BreakoutCompoundedResult(
                config_id="X_1h",
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("15000"),
                total_return_pct=Decimal("50"),
                profit_factor=Decimal("2"),
                max_drawdown_pct=Decimal("20"),  # >= 15 → fails
                total_trades=40,
                years_with_data=5,
            ),
            self._BASE_COMPOUNDED[1],
        ]
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, compounded)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["max_drawdown_lt15"]

    def test_drawdown_exactly_15_rejects(self):
        compounded = [
            BreakoutCompoundedResult(
                config_id="X_1h",
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("15000"),
                total_return_pct=Decimal("50"),
                profit_factor=Decimal("2"),
                max_drawdown_pct=Decimal("15"),  # exactly 15 → < 15 fails
                total_trades=40,
                years_with_data=5,
            ),
            self._BASE_COMPOUNDED[1],
        ]
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, compounded)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["max_drawdown_lt15"]

    def test_too_few_trades_rejects(self):
        compounded = [
            BreakoutCompoundedResult(
                config_id="X_1h",
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("15000"),
                total_return_pct=Decimal("50"),
                profit_factor=Decimal("2"),
                max_drawdown_pct=Decimal("10"),
                total_trades=25,  # < 30
                years_with_data=5,
            ),
            self._BASE_COMPOUNDED[1],
        ]
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, compounded)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["min_trades_30"]

    def test_year_below_neg10_rejects(self):
        yearly = [
            BreakoutYearlyResult(
                config_id="X_1h",
                year=yr,
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("8900") if yr == "2022" else Decimal("10500"),
                return_pct=Decimal("-11") if yr == "2022" else Decimal("5"),
                total_trades=8,
                win_rate_pct=None,
                profit_factor=None,
                max_drawdown_pct=Decimal("5"),
                total_fees=Decimal("10"),
                buy_and_hold_return_pct=Decimal("20"),
            )
            for yr in BREAKOUT_YEARS
        ]
        checks = _compute_qualification_checks("X_1h", yearly, self._BASE_COMPOUNDED)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["no_year_below_neg10"]

    def test_concentration_check_rejects_when_one_year_dominates(self):
        # Best year contributes > 70% of total gross profit
        yearly = [
            BreakoutYearlyResult(
                config_id="X_1h",
                year="2021",
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("18000"),  # +8000 gross
                return_pct=Decimal("80"),
                total_trades=8,
                win_rate_pct=Decimal("75"),
                profit_factor=Decimal("3"),
                max_drawdown_pct=Decimal("3"),
                total_fees=Decimal("10"),
                buy_and_hold_return_pct=Decimal("20"),
            ),
        ] + [
            BreakoutYearlyResult(
                config_id="X_1h",
                year=yr,
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("10500"),  # +500 gross each
                return_pct=Decimal("5"),
                total_trades=4,
                win_rate_pct=Decimal("60"),
                profit_factor=Decimal("1.3"),
                max_drawdown_pct=Decimal("3"),
                total_fees=Decimal("10"),
                buy_and_hold_return_pct=Decimal("20"),
            )
            for yr in ("2022", "2023", "2024", "2025")
        ]
        # total gross = 8000 + 4*500 = 10000; best year = 8000 = 80% > 70%
        checks = _compute_qualification_checks("X_1h", yearly, self._BASE_COMPOUNDED)
        check_map = {c.name: c.passed for c in checks}
        assert not check_map["best_year_le70pct_gross"]

    def test_concentration_check_passes_when_balanced(self):
        yearly = [
            BreakoutYearlyResult(
                config_id="X_1h",
                year=yr,
                scenario="BASE_COSTS",
                initial_capital=Decimal("10000"),
                final_equity=Decimal("12000"),
                return_pct=Decimal("20"),
                total_trades=8,
                win_rate_pct=Decimal("62"),
                profit_factor=Decimal("1.5"),
                max_drawdown_pct=Decimal("3"),
                total_fees=Decimal("10"),
                buy_and_hold_return_pct=Decimal("20"),
            )
            for yr in BREAKOUT_YEARS
        ]
        # Each year contributes 2000, total = 10000, best = 2000/10000 = 20% ≤ 70%
        checks = _compute_qualification_checks("X_1h", yearly, self._BASE_COMPOUNDED)
        check_map = {c.name: c.passed for c in checks}
        assert check_map["best_year_le70pct_gross"]

    def test_no_violations_always_passes(self):
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, self._BASE_COMPOUNDED)
        check_map = {c.name: c.passed for c in checks}
        assert check_map["no_violations"]

    def test_nine_checks_total(self):
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, self._BASE_COMPOUNDED)
        assert len(checks) == 9

    def test_all_checks_have_required_fields(self):
        checks = _compute_qualification_checks("X_1h", self._BASE_YEARLY, self._BASE_COMPOUNDED)
        for c in checks:
            assert c.name
            assert isinstance(c.passed, bool)
            assert isinstance(c.value, str)
            assert isinstance(c.threshold, str)


class TestSelectBestQualified:
    def _make_rob(
        self,
        config_id: str,
        qualified: bool,
        years_pos: int = 3,
        dd: str = "10",
        pf: str = "1.5",
        best_yr: str = "3000",
    ) -> BreakoutRobustnessSummary:
        checks = (
            QualificationCheck("years_positive_ge3", True, str(years_pos), ">= 3"),
            QualificationCheck("max_drawdown_lt15", True, dd, "< 15%"),
            QualificationCheck("profit_factor_gt1_10", True, pf, "> 1.10"),
            QualificationCheck("best_year_le70pct_gross", True, best_yr, "<= 70%"),
        )
        return BreakoutRobustnessSummary(
            config_id=config_id,
            is_qualified=qualified,
            qualification_checks=checks,
        )

    def test_no_qualified_returns_none(self):
        rob = [self._make_rob("A_1h", False), self._make_rob("B_1h", False)]
        assert select_best_qualified(rob) is None

    def test_empty_list_returns_none(self):
        assert select_best_qualified([]) is None

    def test_single_qualified_returned(self):
        rob = [self._make_rob("A_1h", True)]
        assert select_best_qualified(rob) == "A_1h"

    def test_unqualified_not_selected(self):
        rob = [self._make_rob("A_1h", False), self._make_rob("B_1h", True)]
        assert select_best_qualified(rob) == "B_1h"

    def test_more_positive_years_wins(self):
        # A has 5 years positive, B has 3
        a = self._make_rob("A_1h", True, years_pos=5)
        b = self._make_rob("B_1h", True, years_pos=3)
        assert select_best_qualified([b, a]) == "A_1h"

    def test_lower_drawdown_wins_on_years_tiebreak(self):
        # Both have 4 years positive; A has lower drawdown
        a = self._make_rob("A_1h", True, years_pos=4, dd="8")
        b = self._make_rob("B_1h", True, years_pos=4, dd="12")
        assert select_best_qualified([b, a]) == "A_1h"

    def test_higher_profit_factor_wins_last_tiebreak(self):
        # Same years, same drawdown, same concentration — A has higher PF
        a = self._make_rob("A_1h", True, years_pos=4, dd="8", pf="2.0", best_yr="3000")
        b = self._make_rob("B_1h", True, years_pos=4, dd="8", pf="1.5", best_yr="3000")
        assert select_best_qualified([b, a]) == "A_1h"


class TestExportStructure:
    def test_yearly_csv_columns(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "yearly.csv"
        export_breakout_yearly_results_csv(report, out)
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        expected = {
            "config_id",
            "year",
            "scenario",
            "initial_capital",
            "final_equity",
            "return_pct",
            "total_trades",
            "win_rate_pct",
            "profit_factor",
            "max_drawdown_pct",
            "total_fees",
            "buy_and_hold_return_pct",
        }
        assert expected.issubset(set(fieldnames))

    def test_yearly_csv_has_data_rows(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "yearly.csv"
        export_breakout_yearly_results_csv(report, out)
        with out.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(report.yearly_results)

    def test_compounded_csv_columns(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "compounded.csv"
        export_breakout_compounded_results_csv(report, out)
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        expected = {
            "config_id",
            "scenario",
            "initial_capital",
            "final_equity",
            "total_return_pct",
            "profit_factor",
            "max_drawdown_pct",
            "total_trades",
            "years_with_data",
        }
        assert expected.issubset(set(fieldnames))

    def test_robustness_csv_columns(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "robustness.csv"
        export_breakout_robustness_summary_csv(report, out)
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "config_id" in fieldnames
        assert "is_qualified" in fieldnames
        assert "compounded_base_positive" in fieldnames

    def test_trades_csv_written(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "trades.csv"
        export_breakout_trades_csv(report, out)
        assert out.exists()
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "config_id" in fieldnames
        assert "trade_id" in fieldnames
        assert "entry_exec_price" in fieldnames
        assert "net_pnl" in fieldnames

    def test_equity_curves_csv_written(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "equity.csv"
        export_breakout_equity_curves_csv(report, out)
        assert out.exists()
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "config_id" in fieldnames
        assert "equity" in fieldnames
        assert "drawdown_pct" in fieldnames

    def test_audit_json_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.backtesting.breakout_family.BREAKOUT_CONFIGS",
            {"A": _TINY_CFG_A},
        )
        monkeypatch.setattr(
            "app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES",
            ("1h",),
        )
        report = _make_minimal_report()
        out = tmp_path / "audit.json"
        export_breakout_audit_json(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "candles_15m_total" in data
        assert "symbol" in data
        assert "frozen_configs" in data
        assert "paper_test_disclaimer" in data

    def test_report_json_keys(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "report.json"
        export_breakout_report_json(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "symbol" in data
        assert "yearly_results" in data
        assert "compounded_results" in data
        assert "robustness" in data
        assert "paper_test_disclaimer" in data

    def test_report_json_decimal_serialised_as_string(self, tmp_path):
        report = _make_minimal_report()
        out = tmp_path / "report.json"
        export_breakout_report_json(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        # initial_capital is a Decimal that must appear as a string
        assert isinstance(data["initial_capital"], str)

    def test_all_seven_exports_produce_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.backtesting.breakout_family.BREAKOUT_CONFIGS",
            {"A": _TINY_CFG_A},
        )
        monkeypatch.setattr(
            "app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES",
            ("1h",),
        )
        report = _make_minimal_report()
        paths = [
            tmp_path / "yearly.csv",
            tmp_path / "compounded.csv",
            tmp_path / "robustness.csv",
            tmp_path / "trades.csv",
            tmp_path / "equity.csv",
            tmp_path / "audit.json",
            tmp_path / "report.json",
        ]
        export_breakout_yearly_results_csv(report, paths[0])
        export_breakout_compounded_results_csv(report, paths[1])
        export_breakout_robustness_summary_csv(report, paths[2])
        export_breakout_trades_csv(report, paths[3])
        export_breakout_equity_curves_csv(report, paths[4])
        export_breakout_audit_json(report, paths[5])
        export_breakout_report_json(report, paths[6])
        for p in paths:
            assert p.exists(), f"{p.name} was not created"


class TestCLICompatibility:
    """Verify existing CLI arguments are unaffected by Stage 6.0 additions."""

    def _build_parser(self):
        from app.cli.run_backtest import _build_parser

        return _build_parser()

    def test_compare_breakout_family_arg_exists(self):
        p = self._build_parser()
        args = p.parse_args(
            [
                "--symbol",
                "BTCUSDT",
                "--interval",
                "15m",
                "--start",
                "2024-01-01",
                "--end",
                "2025-01-01",
                "--compare-breakout-family",
            ]
        )
        assert args.compare_breakout_family is True

    def test_compare_timeframes_costs_still_present(self):
        p = self._build_parser()
        args = p.parse_args(
            [
                "--symbol",
                "BTCUSDT",
                "--interval",
                "15m",
                "--start",
                "2024-01-01",
                "--end",
                "2025-01-01",
                "--compare-timeframes-costs",
            ]
        )
        assert args.compare_timeframes_costs is True

    def test_run_frozen_oos_2025_still_present(self):
        p = self._build_parser()
        args = p.parse_args(
            [
                "--symbol",
                "BTCUSDT",
                "--interval",
                "15m",
                "--start",
                "2024-01-01",
                "--end",
                "2025-01-01",
                "--run-frozen-oos-2025",
            ]
        )
        assert args.run_frozen_oos_2025 is True

    def test_compare_strategy_variants_still_present(self):
        p = self._build_parser()
        args = p.parse_args(
            [
                "--symbol",
                "BTCUSDT",
                "--interval",
                "15m",
                "--start",
                "2024-01-01",
                "--end",
                "2025-01-01",
                "--compare-strategy-variants",
            ]
        )
        assert args.compare_strategy_variants is True

    def test_compare_breakout_family_is_false_by_default(self):
        p = self._build_parser()
        args = p.parse_args(
            [
                "--symbol",
                "BTCUSDT",
                "--interval",
                "15m",
                "--start",
                "2024-01-01",
                "--end",
                "2025-01-01",
            ]
        )
        assert args.compare_breakout_family is False

    def test_existing_args_not_mutually_exclusive(self):
        p = self._build_parser()
        # Verify parser accepts standard args without error
        args = p.parse_args(
            [
                "--symbol",
                "BTCUSDT",
                "--interval",
                "15m",
                "--start",
                "2024-01-01",
                "--end",
                "2025-01-01",
                "--no-force-close",
                "--initial-capital",
                "5000",
                "--fee-percentage",
                "0.1",
                "--slippage-percentage",
                "0.05",
            ]
        )
        assert args.symbol == "BTCUSDT"
        assert args.no_force_close is True


class TestIncompletePeriodsHandling:
    """Verify that incomplete (partial) candle periods are rejected by the aggregator."""

    def test_incomplete_4h_group_not_included(self, monkeypatch):
        """Trailing 15m candles that don't form a complete 4h group are dropped."""
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", ("4h",))
        # Create 80 warmup + 17 candles per year (only 1 complete 4h = 16, 1 trailing dropped)
        dataset = _make_dataset(warmup_count=80, eval_count_per_year=17)
        report = run_breakout_family("BTCUSDT", dataset)
        # data_coverage eval_candles should reflect only the complete 4h groups
        for cov in report.data_coverage:
            # 17 15m → 1 complete 4h group → 1 eval 4h candle
            assert cov.eval_candles == 1, f"{cov.config_id}/{cov.year}: expected 1 4h eval candle"

    def test_warmup_excluded_from_evaluation(self, monkeypatch):
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_CONFIGS", _TINY_CONFIGS)
        monkeypatch.setattr("app.backtesting.breakout_family.BREAKOUT_TIMEFRAMES", ("1h",))
        dataset = _make_dataset(warmup_count=80, eval_count_per_year=64)
        report = run_breakout_family("BTCUSDT", dataset)
        # No trade should have entry_exec_time before the first 2021 candle open_time
        year_start_ms = 1_609_459_200_000
        for config_id, yr_map in report.raw_results.items():
            for yr, scenario_map in yr_map.items():
                for t in scenario_map["BASE_COSTS"].trades:
                    assert t.entry_exec_time >= year_start_ms, (
                        f"{config_id}/{yr}: trade executed before evaluation period"
                    )
