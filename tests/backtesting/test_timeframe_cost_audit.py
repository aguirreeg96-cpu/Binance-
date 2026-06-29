"""Tests for Stage 5.2C.1 — timeframe_cost_audit.

Covers:
  - Candle count transparency: raw_15m == warmup_15m + eval_15m; agg_total == agg_warmup + agg_eval
  - Warmup not counted as period: trade_before_start is always False
  - Scenario signal identity: entry_signals_match == True for all pairs (indicator-based)
  - Slippage formula: entry/exit formula flags are True for all trades
  - Monotonicity: no violations with flat-price candles (0 trades, 0% return)
  - No trades before start_ms or after end_ms
  - Report structure: correct row counts
  - Export functions: correct CSV column names and row counts
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.backtesting.timeframe_cost_audit import (
    TimeframeCostAuditReport,
    export_audit_data_counts_csv,
    export_audit_json,
    export_audit_scenario_identity_csv,
    export_audit_slippage_csv,
    export_audit_warmup_boundaries_csv,
)
from app.backtesting.timeframe_cost_comparison import (
    COST_SCENARIO_NAMES,
    TIMEFRAMES,
    run_timeframe_cost_audit,
)
from app.indicators.schemas import IndicatorConfig
from app.models.candle import Candle

_INTERVAL_15M = 900_000
_IND_CFG = IndicatorConfig(ema_short_period=2, ema_medium_period=3, ema_long_period=4)

_START_MS_2023 = 20 * _INTERVAL_15M
_END_MS_2023 = 40 * _INTERVAL_15M
_BASE_2024 = 1000 * _INTERVAL_15M
_START_MS_2024 = _BASE_2024 + 20 * _INTERVAL_15M
_END_MS_2024 = _BASE_2024 + 40 * _INTERVAL_15M
_INITIAL_CAPITAL = Decimal("10000")
_FLAT_PRICE = Decimal("100")
_FLAT_VOLUME = Decimal("500")


def _make_flat_candle(open_time: int) -> Candle:
    return Candle(
        symbol="BTCUSDT",
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
def audit_result() -> tuple:
    return run_timeframe_cost_audit(
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


@pytest.fixture(scope="module")
def audit(audit_result: tuple) -> TimeframeCostAuditReport:
    return audit_result[1]


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


class TestAuditStructure:
    def test_candle_counts_row_count(self, audit: TimeframeCostAuditReport):
        """3 timeframes × 2 years = 6 rows."""
        assert len(audit.candle_counts) == 6

    def test_scenario_identity_row_count(self, audit: TimeframeCostAuditReport):
        """3 tf × 2 years × C(4,2)=6 pairs = 36 rows."""
        assert len(audit.scenario_identity) == 36

    def test_warmup_boundaries_row_count(self, audit: TimeframeCostAuditReport):
        """3 tf × 4 scenarios × 2 years = 24 rows."""
        assert len(audit.warmup_boundaries) == 24

    def test_all_timeframes_in_candle_counts(self, audit: TimeframeCostAuditReport):
        tfs = {r.timeframe for r in audit.candle_counts}
        assert tfs == set(TIMEFRAMES)

    def test_all_years_in_candle_counts(self, audit: TimeframeCostAuditReport):
        years = {r.year for r in audit.candle_counts}
        assert years == {"2023", "2024"}

    def test_all_scenarios_in_warmup_boundaries(self, audit: TimeframeCostAuditReport):
        scenarios = {r.cost_scenario for r in audit.warmup_boundaries}
        assert scenarios == set(COST_SCENARIO_NAMES)


# ---------------------------------------------------------------------------
# Candle count invariants
# ---------------------------------------------------------------------------


class TestCandleCountInvariants:
    def test_warmup_plus_eval_equals_raw_15m(self, audit: TimeframeCostAuditReport):
        """raw_15m_total == warmup_15m + eval_15m for every row."""
        for row in audit.candle_counts:
            total = row.warmup_15m + row.eval_15m
            assert (
                total == row.raw_15m_total
            ), f"{row.timeframe}/{row.year}: {row.warmup_15m}+{row.eval_15m}!={row.raw_15m_total}"

    def test_agg_warmup_plus_eval_equals_agg_total(self, audit: TimeframeCostAuditReport):
        """agg_warmup + agg_eval == agg_total for every row."""
        for row in audit.candle_counts:
            assert (
                row.agg_warmup + row.agg_eval == row.agg_total
            ), f"{row.timeframe}/{row.year}: {row.agg_warmup} + {row.agg_eval} != {row.agg_total}"

    def test_eval_candles_positive(self, audit: TimeframeCostAuditReport):
        """Every (tf, year) has at least one eval candle."""
        for row in audit.candle_counts:
            assert row.agg_eval > 0, f"{row.timeframe}/{row.year}: agg_eval={row.agg_eval}"

    def test_15m_raw_total(self, audit: TimeframeCostAuditReport):
        """We loaded 40 15m candles per year."""
        for row in audit.candle_counts:
            assert row.raw_15m_total == 40, f"{row.timeframe}/{row.year}"

    def test_15m_warmup_count(self, audit: TimeframeCostAuditReport):
        """First 20 of 40 candles are warmup (start_ms = 20 * 15m)."""
        for row in audit.candle_counts:
            assert row.warmup_15m == 20, f"{row.timeframe}/{row.year}"

    def test_30m_agg_total(self, audit: TimeframeCostAuditReport):
        """40 15m → 20 30m candles."""
        for row in audit.candle_counts:
            if row.timeframe == "30m":
                assert row.agg_total == 20, f"year={row.year}"

    def test_1h_agg_total(self, audit: TimeframeCostAuditReport):
        """40 15m → 10 1h candles."""
        for row in audit.candle_counts:
            if row.timeframe == "1h":
                assert row.agg_total == 10, f"year={row.year}"

    def test_30m_agg_warmup(self, audit: TimeframeCostAuditReport):
        """20 30m candles; first 10 are warmup (open_times 0..16.2m < 18m)."""
        for row in audit.candle_counts:
            if row.timeframe == "30m":
                assert row.agg_warmup == 10, f"year={row.year}"

    def test_1h_agg_warmup(self, audit: TimeframeCostAuditReport):
        """10 1h candles; first 5 are warmup (open_times 0..14.4m < 18m)."""
        for row in audit.candle_counts:
            if row.timeframe == "1h":
                assert row.agg_warmup == 5, f"year={row.year}"


# ---------------------------------------------------------------------------
# Scenario identity
# ---------------------------------------------------------------------------


class TestScenarioIdentity:
    def test_entry_signals_match_all_pairs(self, audit: TimeframeCostAuditReport):
        """Entry signal timestamps identical across all scenario pairs (indicator-based)."""
        for row in audit.scenario_identity:
            assert row.entry_signals_match, (
                f"{row.timeframe}/{row.year} {row.scenario_a} vs {row.scenario_b}: "
                f"entry hash mismatch"
            )

    def test_trade_counts_match_all_pairs(self, audit: TimeframeCostAuditReport):
        """With flat prices (0 trades), all scenarios produce the same trade count."""
        for row in audit.scenario_identity:
            assert row.trade_counts_match, (
                f"{row.timeframe}/{row.year} {row.scenario_a}({row.trade_count_a}) "
                f"vs {row.scenario_b}({row.trade_count_b})"
            )

    def test_no_trades_all_scenarios(self, audit: TimeframeCostAuditReport):
        """Flat-price candles produce 0 trades in every scenario."""
        for row in audit.scenario_identity:
            assert row.trade_count_a == 0
            assert row.trade_count_b == 0

    def test_eval_candle_hash_consistent_within_tf_year(self, audit: TimeframeCostAuditReport):
        """All rows for the same (tf, year) share the same eval_candle_hash."""
        by_tf_year: dict[tuple[str, str], str] = {}
        for row in audit.scenario_identity:
            key = (row.timeframe, row.year)
            if key in by_tf_year:
                assert row.eval_candle_hash == by_tf_year[key]
            else:
                by_tf_year[key] = row.eval_candle_hash


# ---------------------------------------------------------------------------
# Slippage trades
# ---------------------------------------------------------------------------


class TestSlippageTrades:
    def test_no_trades_with_flat_prices(self, audit: TimeframeCostAuditReport):
        """Flat candles produce no trades; slippage_trades list is empty."""
        assert audit.slippage_trades == []

    def test_slippage_formula_ok_when_trades_exist(self, audit: TimeframeCostAuditReport):
        """All formula flags must be True for every trade present."""
        for row in audit.slippage_trades:
            assert row.entry_slippage_formula_ok, f"trade {row.trade_id} entry slip"
            assert row.exit_slippage_formula_ok, f"trade {row.trade_id} exit slip"
            assert row.fee_formula_ok, f"trade {row.trade_id} fee"


# ---------------------------------------------------------------------------
# Warmup boundaries
# ---------------------------------------------------------------------------


class TestWarmupBoundaries:
    def test_no_trade_before_start(self, audit: TimeframeCostAuditReport):
        """No trade entry before start_ms (warmup boundary respected)."""
        for row in audit.warmup_boundaries:
            assert (
                not row.trade_before_start
            ), f"{row.timeframe}/{row.year}/{row.cost_scenario}: trade before start_ms"

    def test_no_trade_after_end(self, audit: TimeframeCostAuditReport):
        """No trade exit after end_ms."""
        for row in audit.warmup_boundaries:
            assert (
                not row.trade_after_end
            ), f"{row.timeframe}/{row.year}/{row.cost_scenario}: trade after end_ms"

    def test_first_trade_is_none_with_zero_trades(self, audit: TimeframeCostAuditReport):
        """No trades → first_trade_entry_exec_time is None."""
        for row in audit.warmup_boundaries:
            assert row.first_trade_entry_exec_time is None

    def test_first_eval_candle_at_start_ms(self, audit: TimeframeCostAuditReport):
        """First eval candle open_time == start_ms for all timeframes."""
        for row in audit.warmup_boundaries:
            expected = _START_MS_2023 if row.year == "2023" else _START_MS_2024
            assert row.first_eval_candle_open_time == expected, (
                f"{row.timeframe}/{row.year}: "
                f"first_eval_ot={row.first_eval_candle_open_time}, expected={expected}"
            )


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_no_violations_with_flat_prices(self, audit: TimeframeCostAuditReport):
        """Flat prices → 0% return for all scenarios → no monotonicity violations."""
        assert audit.monotonicity_violations == []

    def test_stop_price_varies_false_with_zero_trades(self, audit: TimeframeCostAuditReport):
        """No trades → no exit divergence → stop_price_varies_by_slippage is False."""
        assert audit.stop_price_varies_by_slippage is False


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

_COUNTS_FIELDNAMES = {
    "timeframe",
    "year",
    "raw_15m_total",
    "warmup_15m",
    "eval_15m",
    "agg_total",
    "agg_warmup",
    "agg_eval",
}

_IDENTITY_FIELDNAMES = {
    "timeframe",
    "year",
    "scenario_a",
    "scenario_b",
    "eval_candle_hash",
    "entry_signal_hash_a",
    "entry_signal_hash_b",
    "entry_signals_match",
    "trade_count_a",
    "trade_count_b",
    "trade_counts_match",
    "exit_reason_hash_a",
    "exit_reason_hash_b",
    "logical_exits_match",
}

_SLIPPAGE_FIELDNAMES = {
    "timeframe",
    "year",
    "cost_scenario",
    "trade_id",
    "entry_exec_time",
    "entry_exec_price",
    "entry_raw_price",
    "entry_slippage_paid",
    "entry_fee",
    "quantity",
    "capital_at_entry",
    "exit_exec_time",
    "exit_exec_price",
    "exit_raw_price",
    "exit_slippage_paid",
    "exit_fee",
    "gross_pnl",
    "net_pnl",
    "exit_reasons",
    "slippage_rate",
    "fee_rate",
    "entry_slippage_formula_ok",
    "exit_slippage_formula_ok",
    "fee_formula_ok",
}

_WARMUP_FIELDNAMES = {
    "timeframe",
    "year",
    "cost_scenario",
    "start_ms",
    "end_ms",
    "first_eval_candle_open_time",
    "first_trade_entry_exec_time",
    "last_trade_exit_exec_time",
    "trade_before_start",
    "trade_after_end",
}


class TestExports:
    def test_data_counts_csv_columns(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "counts.csv"
        export_audit_data_counts_csv(audit, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames or []) == _COUNTS_FIELDNAMES

    def test_data_counts_csv_row_count(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "counts2.csv"
        export_audit_data_counts_csv(audit, path)
        with path.open(encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 6

    def test_scenario_identity_csv_columns(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "identity.csv"
        export_audit_scenario_identity_csv(audit, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames or []) == _IDENTITY_FIELDNAMES

    def test_scenario_identity_csv_row_count(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "identity2.csv"
        export_audit_scenario_identity_csv(audit, path)
        with path.open(encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 36

    def test_slippage_csv_columns(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "slip.csv"
        export_audit_slippage_csv(audit, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames or []) == _SLIPPAGE_FIELDNAMES

    def test_slippage_csv_zero_rows_with_flat_prices(
        self, audit: TimeframeCostAuditReport, tmp_path: Path
    ):
        path = tmp_path / "slip2.csv"
        export_audit_slippage_csv(audit, path)
        with path.open(encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 0

    def test_warmup_boundaries_csv_columns(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "warmup.csv"
        export_audit_warmup_boundaries_csv(audit, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames or []) == _WARMUP_FIELDNAMES

    def test_warmup_boundaries_csv_row_count(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "warmup2.csv"
        export_audit_warmup_boundaries_csv(audit, path)
        with path.open(encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 24

    def test_json_structure(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "audit.json"
        export_audit_json(audit, path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert "warning" in data
        assert "candle_counts" in data
        assert "scenario_identity" in data
        assert "slippage_trades" in data
        assert "warmup_boundaries" in data
        assert "monotonicity_violations" in data
        assert "stop_price_varies_by_slippage" in data
        assert "cost_drag_compounding_note" in data

    def test_json_counts(self, audit: TimeframeCostAuditReport, tmp_path: Path):
        path = tmp_path / "audit2.json"
        export_audit_json(audit, path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["candle_counts"]) == 6
        assert len(data["scenario_identity"]) == 36
        assert len(data["warmup_boundaries"]) == 24
        assert len(data["slippage_trades"]) == 0
        assert len(data["monotonicity_violations"]) == 0
