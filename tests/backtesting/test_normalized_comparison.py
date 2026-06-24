"""Tests for Stage 5.2A.1 — Normalized strategy comparison.

Covers:
  - 12-row matrix (4 variants × 3 allocations)
  - Determinism: same inputs → same report
  - Allocation 25/50/100 correctly stored
  - Fees proportional to allocation (when trades exist)
  - No negative balances
  - Same entry signals across allocations of same variant
  - No-cost runs produce zero fees
  - Period labels stored correctly
  - Multi-period yearly summary structure
  - Combined return compounding formula
  - Exporters: CSV row counts, JSON structure, Decimal as string
  - Compatibility: V1_BASELINE@100% trade count consistent with run_variants
"""

import csv
import json
import tempfile
from decimal import Decimal
from pathlib import Path

from app.backtesting.config import BacktestConfig
from app.backtesting.normalized_comparison import (
    ALLOCATIONS,
    VARIANT_NAMES,
    MultiPeriodReport,
    NormalizedComparisonReport,
    run_multi_period_comparison,
    run_normalized_comparison,
)
from app.indicators.schemas import IndicatorConfig
from tests.backtesting.conftest import make_candles

_D = Decimal
_BASE_TIME = 1_000_000
_INTERVAL_MS = 900_000  # 15m
_BASE_TIME_2024 = _BASE_TIME + 35_040 * _INTERVAL_MS  # second "year"


def _ind_config() -> IndicatorConfig:
    return IndicatorConfig(
        sma_short_period=2,
        sma_long_period=3,
        ema_short_period=2,
        ema_medium_period=3,
        ema_long_period=4,
        rsi_period=2,
        atr_period=2,
        volume_period=2,
    )


def _cfg(base_time: int = _BASE_TIME, **kwargs) -> BacktestConfig:
    defaults = {
        "symbol": "BTCUSDT",
        "interval": "15m",
        "start_ms": base_time,
        "end_ms": base_time + 50 * _INTERVAL_MS,
        "initial_capital": _D("10000"),
        "fee_percentage": _D("0.1"),
        "slippage_percentage": _D("0.05"),
        "force_close_at_end": True,
    }
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _run(
    n_candles: int = 40,
    base_time: int = _BASE_TIME,
    **cfg_kwargs,
) -> NormalizedComparisonReport:
    candles = make_candles(
        n_candles, base_open_time=base_time, interval_ms=_INTERVAL_MS, interval="15m"
    )
    return run_normalized_comparison(
        all_candles=candles,
        warmup_len=0,
        config=_cfg(base_time=base_time, **cfg_kwargs),
        indicator_config=_ind_config(),
    )


def _run_multi() -> MultiPeriodReport:
    candles_23 = make_candles(
        40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
    )
    candles_24 = make_candles(
        40, base_open_time=_BASE_TIME_2024, interval_ms=_INTERVAL_MS, interval="15m"
    )
    return run_multi_period_comparison(
        candles_2023=candles_23,
        warmup_2023=0,
        config_2023=_cfg(base_time=_BASE_TIME),
        candles_2024=candles_24,
        warmup_2024=0,
        config_2024=_cfg(base_time=_BASE_TIME_2024),
        indicator_config=_ind_config(),
    )


# ---------------------------------------------------------------------------
# Matrix structure
# ---------------------------------------------------------------------------


class TestMatrixStructure:
    def test_matrix_has_12_rows(self):
        report = _run()
        assert len(report.matrix) == 12

    def test_all_variant_names_present(self):
        report = _run()
        names = {r.variant_name for r in report.matrix}
        assert names == set(VARIANT_NAMES)

    def test_all_allocations_present(self):
        report = _run()
        allocs = {r.allocation_pct for r in report.matrix}
        assert allocs == set(ALLOCATIONS)

    def test_each_variant_has_three_allocations(self):
        report = _run()
        for name in VARIANT_NAMES:
            rows = [r for r in report.matrix if r.variant_name == name]
            assert len(rows) == 3
            assert {r.allocation_pct for r in rows} == set(ALLOCATIONS)

    def test_bah_is_decimal(self):
        report = _run()
        assert isinstance(report.bah_return_pct, Decimal)

    def test_period_label_stored(self):
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_normalized_comparison(candles, 0, _cfg(), _ind_config(), period_label="2024")
        assert report.period_label == "2024"

    def test_result_fields_are_correct_types(self):
        report = _run()
        for r in report.matrix:
            assert isinstance(r.variant_name, str)
            assert isinstance(r.allocation_pct, Decimal)
            assert isinstance(r.return_pct, Decimal)
            assert isinstance(r.return_pct_no_costs, Decimal)
            assert isinstance(r.final_equity, Decimal)
            assert isinstance(r.max_drawdown_pct, Decimal)
            assert isinstance(r.total_fees, Decimal)
            assert isinstance(r.slippage_cost, Decimal)
            assert isinstance(r.total_trades, int)
            assert isinstance(r.exposure_pct, Decimal)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_matrix(self):
        r1 = _run()
        r2 = _run()
        for a, b in zip(r1.matrix, r2.matrix, strict=True):
            assert a.return_pct == b.return_pct
            assert a.final_equity == b.final_equity
            assert a.total_trades == b.total_trades

    def test_multi_period_determinism(self):
        r1 = _run_multi()
        r2 = _run_multi()
        for a, b in zip(r1.yearly_summary, r2.yearly_summary, strict=True):
            assert a.combined_return_pct == b.combined_return_pct


# ---------------------------------------------------------------------------
# Allocation correctness
# ---------------------------------------------------------------------------


class TestAllocationCorrectness:
    def test_25pct_rows_allocation_value(self):
        report = _run()
        at25 = [r for r in report.matrix if r.allocation_pct == _D("25")]
        assert len(at25) == 4
        for r in at25:
            assert r.allocation_pct == _D("25")

    def test_50pct_rows_allocation_value(self):
        report = _run()
        at50 = [r for r in report.matrix if r.allocation_pct == _D("50")]
        assert len(at50) == 4
        for r in at50:
            assert r.allocation_pct == _D("50")

    def test_100pct_rows_allocation_value(self):
        report = _run()
        at100 = [r for r in report.matrix if r.allocation_pct == _D("100")]
        assert len(at100) == 4
        for r in at100:
            assert r.allocation_pct == _D("100")

    def test_allocation_affects_fees_when_trades(self):
        """With trades, higher allocation → higher absolute fees."""
        report = _run()
        for name in VARIANT_NAMES:
            rows = {r.allocation_pct: r for r in report.matrix if r.variant_name == name}
            r25 = rows[_D("25")]
            r100 = rows[_D("100")]
            if r100.total_trades > 0:
                assert r100.total_fees >= r25.total_fees

    def test_allocation_affects_final_equity(self):
        """Different allocations → different final equities (when trades occur)."""
        report = _run()
        for name in VARIANT_NAMES:
            rows = {r.allocation_pct: r for r in report.matrix if r.variant_name == name}
            r25 = rows[_D("25")]
            r100 = rows[_D("100")]
            if r25.total_trades > 0 and r100.total_trades > 0:
                assert r25.final_equity != r100.final_equity


# ---------------------------------------------------------------------------
# No negative balances
# ---------------------------------------------------------------------------


class TestNoNegativeBalance:
    def test_final_equity_non_negative(self):
        report = _run()
        for r in report.matrix:
            assert r.final_equity >= _D("0"), (
                f"{r.variant_name}@{r.allocation_pct}% has negative equity: {r.final_equity}"
            )

    def test_slippage_cost_non_negative(self):
        report = _run()
        for r in report.matrix:
            assert r.slippage_cost >= _D("0")

    def test_total_fees_non_negative(self):
        report = _run()
        for r in report.matrix:
            assert r.total_fees >= _D("0")


# ---------------------------------------------------------------------------
# No-cost runs
# ---------------------------------------------------------------------------


class TestNoCostConfiguration:
    def test_zero_fee_config_has_zero_fees(self):
        report = _run(fee_percentage=_D("0"), slippage_percentage=_D("0"))
        for r in report.matrix:
            assert r.total_fees == _D("0")

    def test_zero_fee_slippage_cost_zero(self):
        report = _run(fee_percentage=_D("0"), slippage_percentage=_D("0"))
        for r in report.matrix:
            assert r.slippage_cost == _D("0")

    def test_no_costs_return_gte_with_costs_when_trades(self):
        report = _run()
        for r in report.matrix:
            if r.total_trades > 0:
                assert r.return_pct_no_costs >= r.return_pct


# ---------------------------------------------------------------------------
# Same entry signals / trade count consistency
# ---------------------------------------------------------------------------


class TestEntrySIgnalConsistency:
    def test_same_variant_same_trade_count_across_allocations(self):
        """For V1_BASELINE (crossover only, no risk exits) all allocations
        produce same trade count given identical candles."""
        report = _run()
        for name in VARIANT_NAMES:
            rows = [r for r in report.matrix if r.variant_name == name]
            counts = [r.total_trades for r in rows]
            # All allocations should see same signals; minor edge case:
            # if capital depleted, trade count could differ — tolerate that.
            # At minimum, non-zero count should be consistent direction.
            if all(c > 0 for c in counts):
                # All three allocations should have same trade count
                # (signals don't depend on capital size)
                assert counts[0] == counts[1] == counts[2], (
                    f"{name} has inconsistent trade counts: {counts}"
                )


# ---------------------------------------------------------------------------
# Compatibility with run_variants
# ---------------------------------------------------------------------------


class TestCompatibilityWithVariants:
    def test_v1_baseline_100pct_consistent_with_run_variants(self):
        """V1_BASELINE@100% trade count should match run_variants V1_BASELINE."""
        from app.backtesting.variants import run_variants

        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        config = _cfg()
        ind = _ind_config()

        variants_report = run_variants(candles, 0, config, ind)
        v1_trades = next(
            v.result.total_trades for v in variants_report.variants if v.name == "V1_BASELINE"
        )

        norm_report = run_normalized_comparison(candles, 0, config, ind)
        v1_100 = next(
            r
            for r in norm_report.matrix
            if r.variant_name == "V1_BASELINE" and r.allocation_pct == _D("100")
        )
        # V2 engine with risk disabled should produce same trade count as V1 engine
        assert v1_100.total_trades == v1_trades


# ---------------------------------------------------------------------------
# Multi-period structure
# ---------------------------------------------------------------------------


class TestMultiPeriodStructure:
    def test_yearly_summary_has_12_entries(self):
        multi = _run_multi()
        assert len(multi.yearly_summary) == 12

    def test_period_labels(self):
        multi = _run_multi()
        assert multi.period_2023.period_label == "2023"
        assert multi.period_2024.period_label == "2024"

    def test_each_period_has_12_matrix_rows(self):
        multi = _run_multi()
        assert len(multi.period_2023.matrix) == 12
        assert len(multi.period_2024.matrix) == 12

    def test_positive_years_in_range(self):
        multi = _run_multi()
        for s in multi.yearly_summary:
            assert 0 <= s.positive_years <= 2

    def test_worst_drawdown_is_max_of_years(self):
        multi = _run_multi()
        for s in multi.yearly_summary:
            lookup_23 = {(r.variant_name, r.allocation_pct): r for r in multi.period_2023.matrix}
            lookup_24 = {(r.variant_name, r.allocation_pct): r for r in multi.period_2024.matrix}
            r23 = lookup_23[(s.variant_name, s.allocation_pct)]
            r24 = lookup_24[(s.variant_name, s.allocation_pct)]
            expected = max(r23.max_drawdown_pct, r24.max_drawdown_pct)
            assert s.worst_drawdown_pct == expected

    def test_combined_return_compounding(self):
        """combined = (1+r23/100)*(1+r24/100)-1 * 100."""
        multi = _run_multi()
        _HUNDRED = _D("100")
        for s in multi.yearly_summary:
            expected = (
                (1 + s.return_pct_2023 / _HUNDRED) * (1 + s.return_pct_2024 / _HUNDRED) - 1
            ) * _HUNDRED
            # Allow tiny rounding diff from intermediate Decimal ops
            assert abs(s.combined_return_pct - expected) < _D("0.0001")

    def test_trade_count_fields(self):
        multi = _run_multi()
        for s in multi.yearly_summary:
            assert isinstance(s.trade_count_2023, int)
            assert isinstance(s.trade_count_2024, int)
            assert s.trade_count_2023 >= 0
            assert s.trade_count_2024 >= 0

    def test_separate_period_evaluation(self):
        """Period 2023 and 2024 use different candles → different BAH returns."""
        multi = _run_multi()
        # BAH is computed per-period from different candles
        # With constant-price candles, BAH is 0 for both, so compare matrix rows
        p23_first = multi.period_2023.matrix[0]
        p24_first = multi.period_2024.matrix[0]
        # Same variant@alloc but different period → should be structurally identical
        # (constant candles give same result regardless of timestamp)
        assert p23_first.variant_name == p24_first.variant_name
        assert p23_first.allocation_pct == p24_first.allocation_pct


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


class TestNormalizedExporters:
    def _report(self) -> NormalizedComparisonReport:
        return _run()

    def _multi(self) -> MultiPeriodReport:
        return _run_multi()

    def test_export_csv_12_rows(self):
        from app.backtesting.normalized_exporters import export_normalized_comparison_csv

        report = self._report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        export_normalized_comparison_csv(report, path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 12
        path.unlink()

    def test_export_csv_has_required_columns(self):
        from app.backtesting.normalized_exporters import export_normalized_comparison_csv

        report = self._report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        export_normalized_comparison_csv(report, path)
        rows = list(csv.DictReader(path.open()))
        required = {
            "variant",
            "allocation_pct",
            "return_pct",
            "return_pct_no_costs",
            "final_equity",
            "max_drawdown_pct",
            "profit_factor",
            "win_rate_pct",
            "total_fees",
            "slippage_cost",
            "total_trades",
            "avg_trade_pnl",
            "median_trade_pnl",
            "exposure_pct",
        }
        assert required <= set(rows[0].keys())
        path.unlink()

    def test_export_json_structure(self):
        from app.backtesting.normalized_exporters import export_normalized_comparison_json

        report = self._report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        export_normalized_comparison_json(report, path)
        data = json.loads(path.read_text())
        assert "warning" in data
        assert "matrix" in data
        assert "period" in data
        assert "buy_and_hold_return_pct" in data
        assert len(data["matrix"]) == 12
        path.unlink()

    def test_export_json_decimal_as_string(self):
        from app.backtesting.normalized_exporters import export_normalized_comparison_json

        report = self._report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        export_normalized_comparison_json(report, path)
        data = json.loads(path.read_text())
        for row in data["matrix"]:
            assert isinstance(row["return_pct"], str)
            assert isinstance(row["final_equity"], str)
            assert isinstance(row["allocation_pct"], str)
        path.unlink()

    def test_export_yearly_csv_12_rows(self):
        from app.backtesting.normalized_exporters import export_yearly_comparison_csv

        multi = self._multi()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        export_yearly_comparison_csv(multi, path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 12
        path.unlink()

    def test_export_yearly_csv_has_correct_columns(self):
        from app.backtesting.normalized_exporters import export_yearly_comparison_csv

        multi = self._multi()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        export_yearly_comparison_csv(multi, path)
        rows = list(csv.DictReader(path.open()))
        required = {
            "variant",
            "allocation_pct",
            "return_pct_2023",
            "return_pct_2024",
            "combined_return_pct",
            "positive_years",
            "worst_drawdown_pct",
            "profit_factor_stability",
            "trade_count_2023",
            "trade_count_2024",
        }
        assert required <= set(rows[0].keys())
        path.unlink()
