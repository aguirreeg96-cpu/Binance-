"""Tests for Stage 5.2C — timeframe_cost_comparison.

Covers:
  - Constants: TIMEFRAMES, COST_SCENARIO_NAMES, COST_SCENARIOS values
  - Frozen candidate constants (ENTRY_V3_ALIGNED_TREND, V2_STOP_ONLY, 25%)
  - Report structure: 24 results, 12 yearly_summary, 12 robustness
  - NO_COSTS has cost_drag=0 and slippage_cost=0
  - Compounded equity: 2024 initial_capital == 2023 final_equity per (timeframe, scenario)
  - Determinism: identical results on repeated calls
  - Export functions create files with correct column headers
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.backtesting.timeframe_cost_comparison import (
    COST_SCENARIO_NAMES,
    COST_SCENARIOS,
    FROZEN_ALLOCATION_PCT,
    FROZEN_ENTRY_VARIANT,
    FROZEN_EXIT_CONFIG,
    TIMEFRAMES,
    TimeframeCostReport,
    run_timeframe_cost_comparison,
)
from app.backtesting.timeframe_cost_exporters import (
    export_cost_sensitivity_csv,
    export_timeframe_cost_csv,
    export_timeframe_cost_json,
    export_yearly_timeframe_csv,
)
from app.indicators.schemas import IndicatorConfig
from app.models.candle import Candle

_INTERVAL_15M = 900_000

# Small indicator config: ema_short < ema_medium < ema_long, with warmup=4.
_IND_CFG = IndicatorConfig(
    ema_short_period=2,
    ema_medium_period=3,
    ema_long_period=4,
)

# 2023: 40 15m candles, first 20 are warmup.
_START_MS_2023 = 20 * _INTERVAL_15M
_END_MS_2023 = 40 * _INTERVAL_15M

# 2024: 40 15m candles at a higher timestamp, first 20 are warmup.
_BASE_2024 = 1000 * _INTERVAL_15M
_START_MS_2024 = _BASE_2024 + 20 * _INTERVAL_15M
_END_MS_2024 = _BASE_2024 + 40 * _INTERVAL_15M

_INITIAL_CAPITAL = Decimal("10000")
_FLAT_PRICE = Decimal("100")
_FLAT_VOLUME = Decimal("500")


def _make_flat_candle(open_time: int, symbol: str = "BTCUSDT") -> Candle:
    """Candle with constant price; produces no EMA crossovers → 0 trades."""
    return Candle(
        symbol=symbol,
        interval="15m",
        open_time=open_time,
        open=_FLAT_PRICE,
        high=_FLAT_PRICE,
        low=_FLAT_PRICE,
        close=_FLAT_PRICE,
        volume=_FLAT_VOLUME,
        close_time=open_time + _INTERVAL_15M - 1,
        quote_asset_volume=_FLAT_VOLUME * _FLAT_PRICE,
        trades=10,
        taker_buy_base_volume=_FLAT_VOLUME / Decimal("2"),
        taker_buy_quote_volume=_FLAT_VOLUME / Decimal("2") * _FLAT_PRICE,
        is_closed=True,
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )


def _make_sequence_2023() -> list[Candle]:
    return [_make_flat_candle(i * _INTERVAL_15M) for i in range(40)]


def _make_sequence_2024() -> list[Candle]:
    return [_make_flat_candle(_BASE_2024 + i * _INTERVAL_15M) for i in range(40)]


@pytest.fixture(scope="module")
def report() -> TimeframeCostReport:
    return run_timeframe_cost_comparison(
        symbol="BTCUSDT",
        initial_capital=_INITIAL_CAPITAL,
        candles_15m_2023=_make_sequence_2023(),
        candles_15m_2024=_make_sequence_2024(),
        start_ms_2023=_START_MS_2023,
        start_ms_2024=_START_MS_2024,
        end_ms_2023=_END_MS_2023,
        end_ms_2024=_END_MS_2024,
        indicator_config=_IND_CFG,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_timeframes_tuple(self):
        assert TIMEFRAMES == ("15m", "30m", "1h")

    def test_cost_scenario_names(self):
        assert COST_SCENARIO_NAMES == (
            "NO_COSTS",
            "BASE_COSTS",
            "LOW_SLIPPAGE",
            "CONSERVATIVE",
        )

    def test_cost_scenarios_no_costs(self):
        fee, slip = COST_SCENARIOS["NO_COSTS"]
        assert fee == Decimal("0")
        assert slip == Decimal("0")

    def test_cost_scenarios_base_costs(self):
        fee, slip = COST_SCENARIOS["BASE_COSTS"]
        assert fee == Decimal("0.1")
        assert slip == Decimal("0.05")

    def test_cost_scenarios_low_slippage(self):
        fee, slip = COST_SCENARIOS["LOW_SLIPPAGE"]
        assert fee == Decimal("0.1")
        assert slip == Decimal("0.02")

    def test_cost_scenarios_conservative(self):
        fee, slip = COST_SCENARIOS["CONSERVATIVE"]
        assert fee == Decimal("0.1")
        assert slip == Decimal("0.10")

    def test_frozen_entry_variant(self):
        assert FROZEN_ENTRY_VARIANT == "ENTRY_V3_ALIGNED_TREND"

    def test_frozen_exit_config(self):
        assert FROZEN_EXIT_CONFIG == "V2_STOP_ONLY"

    def test_frozen_allocation_pct(self):
        assert FROZEN_ALLOCATION_PCT == Decimal("25")


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


class TestReportStructure:
    def test_results_count(self, report: TimeframeCostReport):
        """3 timeframes × 4 scenarios × 2 years = 24."""
        assert len(report.results) == 24

    def test_yearly_summary_count(self, report: TimeframeCostReport):
        """3 timeframes × 4 scenarios = 12."""
        assert len(report.yearly_summary) == 12

    def test_robustness_count(self, report: TimeframeCostReport):
        """3 timeframes × 4 scenarios = 12."""
        assert len(report.robustness) == 12

    def test_all_timeframes_present_in_results(self, report: TimeframeCostReport):
        tfs = {r.timeframe for r in report.results}
        assert tfs == set(TIMEFRAMES)

    def test_all_scenarios_present_in_results(self, report: TimeframeCostReport):
        scenarios = {r.cost_scenario for r in report.results}
        assert scenarios == set(COST_SCENARIO_NAMES)

    def test_all_years_present(self, report: TimeframeCostReport):
        years = {r.year for r in report.results}
        assert years == {"2023", "2024"}

    def test_each_combination_appears_exactly_once(self, report: TimeframeCostReport):
        combos = [(r.timeframe, r.cost_scenario, r.year) for r in report.results]
        assert len(combos) == len(set(combos))

    def test_yearly_summary_combos_unique(self, report: TimeframeCostReport):
        combos = [(s.timeframe, s.cost_scenario) for s in report.yearly_summary]
        assert len(combos) == len(set(combos))

    def test_robustness_combos_unique(self, report: TimeframeCostReport):
        combos = [(r.timeframe, r.cost_scenario) for r in report.robustness]
        assert len(combos) == len(set(combos))


# ---------------------------------------------------------------------------
# NO_COSTS invariants
# ---------------------------------------------------------------------------


class TestNoCostsInvariants:
    def test_no_costs_cost_drag_is_zero(self, report: TimeframeCostReport):
        """NO_COSTS reference: cost_drag must be exactly 0."""
        for r in report.results:
            if r.cost_scenario == "NO_COSTS":
                assert r.cost_drag == Decimal("0"), (
                    f"{r.timeframe}/{r.year}: cost_drag={r.cost_drag}"
                )

    def test_no_costs_slippage_cost_is_zero(self, report: TimeframeCostReport):
        """NO_COSTS reference: slippage_cost must be exactly 0."""
        for r in report.results:
            if r.cost_scenario == "NO_COSTS":
                assert r.slippage_cost == Decimal("0"), (
                    f"{r.timeframe}/{r.year}: slippage_cost={r.slippage_cost}"
                )

    def test_no_costs_total_fees_zero(self, report: TimeframeCostReport):
        """NO_COSTS run charges no fees."""
        for r in report.results:
            if r.cost_scenario == "NO_COSTS":
                assert r.total_fees == Decimal("0"), (
                    f"{r.timeframe}/{r.year}: total_fees={r.total_fees}"
                )


# ---------------------------------------------------------------------------
# Compounded equity
# ---------------------------------------------------------------------------


class TestCompoundedEquity:
    def test_2024_initial_equals_2023_final_equity(self, report: TimeframeCostReport):
        """2024 initial_capital must equal 2023 final_equity for every (tf, scenario)."""
        for tf in TIMEFRAMES:
            for scenario in COST_SCENARIO_NAMES:
                r23 = next(
                    r
                    for r in report.results
                    if r.timeframe == tf and r.cost_scenario == scenario and r.year == "2023"
                )
                r24 = next(
                    r
                    for r in report.results
                    if r.timeframe == tf and r.cost_scenario == scenario and r.year == "2024"
                )
                assert r24.initial_capital == r23.final_equity, (
                    f"{tf}/{scenario}: "
                    f"r24.initial_capital={r24.initial_capital} "
                    f"r23.final_equity={r23.final_equity}"
                )

    def test_yearly_summary_compounded_chain(self, report: TimeframeCostReport):
        """YearlySummary compounded fields match actual 2023 final and 2024 initial."""
        for s in report.yearly_summary:
            r23 = next(
                r
                for r in report.results
                if r.timeframe == s.timeframe
                and r.cost_scenario == s.cost_scenario
                and r.year == "2023"
            )
            r24 = next(
                r
                for r in report.results
                if r.timeframe == s.timeframe
                and r.cost_scenario == s.cost_scenario
                and r.year == "2024"
            )
            assert s.compounded_final_2023 == r23.final_equity
            assert s.compounded_initial_2024 == r24.initial_capital


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_results_on_rerun(self):
        """Calling run_timeframe_cost_comparison twice yields same report."""
        common = {
            "symbol": "BTCUSDT",
            "initial_capital": _INITIAL_CAPITAL,
            "candles_15m_2023": _make_sequence_2023(),
            "candles_15m_2024": _make_sequence_2024(),
            "start_ms_2023": _START_MS_2023,
            "start_ms_2024": _START_MS_2024,
            "end_ms_2023": _END_MS_2023,
            "end_ms_2024": _END_MS_2024,
            "indicator_config": _IND_CFG,
        }
        r1 = run_timeframe_cost_comparison(**common)
        r2 = run_timeframe_cost_comparison(**common)

        for a, b in zip(r1.results, r2.results, strict=True):
            assert a.final_equity == b.final_equity, (
                f"{a.timeframe}/{a.cost_scenario}/{a.year}: {a.final_equity} != {b.final_equity}"
            )
            assert a.return_pct == b.return_pct


# ---------------------------------------------------------------------------
# Export files
# ---------------------------------------------------------------------------


_RESULT_FIELDNAMES = {
    "timeframe",
    "cost_scenario",
    "year",
    "initial_capital",
    "final_equity",
    "return_pct",
    "buy_and_hold_return_pct",
    "total_trades",
    "win_rate_pct",
    "profit_factor",
    "max_drawdown_pct",
    "total_fees",
    "slippage_cost",
    "exposure_pct",
    "cost_drag",
    "avg_trade_duration_candles",
    "median_net_pnl",
    "avg_gross_pnl",
    "avg_net_pnl",
    "avg_cost_per_trade",
}

_YEARLY_FIELDNAMES = {
    "timeframe",
    "cost_scenario",
    "return_pct_2023",
    "return_pct_2024",
    "combined_return_pct",
    "compounded_initial_2023",
    "compounded_final_2023",
    "compounded_initial_2024",
    "compounded_final_2024",
    "positive_years",
    "worst_drawdown_pct",
    "worst_profit_factor",
    "total_trades_2023",
    "total_trades_2024",
    "total_fees_2023",
    "total_fees_2024",
    "cost_drag_2023",
    "cost_drag_2024",
    "buy_and_hold_2023",
    "buy_and_hold_2024",
}

_ROBUSTNESS_FIELDNAMES = {
    "timeframe",
    "cost_scenario",
    "positive_both_years",
    "profit_factor_above_1_both_years",
    "depends_on_low_slippage",
    "fails_with_conservative",
    "low_trade_count",
}


class TestExports:
    def test_result_csv_columns(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "results.csv"
        export_timeframe_cost_csv(report, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert set(reader.fieldnames) == _RESULT_FIELDNAMES

    def test_result_csv_row_count(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "results.csv"
        export_timeframe_cost_csv(report, path)
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 24

    def test_yearly_csv_columns(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "yearly.csv"
        export_yearly_timeframe_csv(report, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert set(reader.fieldnames) == _YEARLY_FIELDNAMES

    def test_yearly_csv_row_count(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "yearly.csv"
        export_yearly_timeframe_csv(report, path)
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12

    def test_sensitivity_csv_columns(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "sensitivity.csv"
        export_cost_sensitivity_csv(report, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert set(reader.fieldnames) == _ROBUSTNESS_FIELDNAMES

    def test_sensitivity_csv_row_count(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "sensitivity.csv"
        export_cost_sensitivity_csv(report, path)
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12

    def test_json_export_structure(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "report.json"
        export_timeframe_cost_json(report, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert "results" in data
        assert "yearly_summary" in data
        assert "robustness" in data
        assert "warning" in data
        assert len(data["results"]) == 24
        assert len(data["yearly_summary"]) == 12
        assert len(data["robustness"]) == 12

    def test_json_result_fields(self, report: TimeframeCostReport, tmp_path: Path):
        path = tmp_path / "report_fields.json"
        export_timeframe_cost_json(report, path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        first = data["results"][0]
        assert set(first.keys()) == _RESULT_FIELDNAMES
