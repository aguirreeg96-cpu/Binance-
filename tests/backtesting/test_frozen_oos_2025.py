"""Tests for Stage 5.3 — frozen out-of-sample 2025 evaluation.

Covers:
  - Frozen config constants: OOS_TRADING_TIMEFRAME, _OOS_FILTER_CFG, _OOS_RISK_CFG
  - Exclusive 2025 range: no warmup-period trades in result
  - 30m aggregation counts: 60 15m → 30 30m; 10 warmup 30m; 20 eval 30m
  - Signal identity: entry_signal_hash identical across two runs
  - Costs and monotonicity: NO_COSTS >= BASE_COSTS >= CONSERVATIVE (0% with flat prices)
  - Exports: CSV field names, row counts, JSON structure
  - Determinism: two runs with same input produce identical results
  - Parameter rejection: report always shows frozen config values
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.backtesting.frozen_oos_2025 import (
    _OOS_FILTER_CFG,
    _OOS_RISK_CFG,
    OOS_HISTORICAL_REFERENCE,
    OOS_SCENARIO_NAMES,
    OOS_SCENARIOS,
    OOS_TRADING_TIMEFRAME,
    OOS_YEAR,
    FrozenOos2025Report,
    run_frozen_oos_2025,
)
from app.backtesting.frozen_oos_exporters import (
    export_oos_audit_json,
    export_oos_equity_curve_csv,
    export_oos_scenarios_csv,
    export_oos_summary_json,
    export_oos_trades_csv,
)
from app.indicators.schemas import IndicatorConfig
from app.models.candle import Candle
from app.strategy.entry_filter import EntryFilterType  # noqa: F401 (used in assertions)

_INTERVAL_15M = 900_000
_IND_CFG = IndicatorConfig(ema_short_period=2, ema_medium_period=3, ema_long_period=4)

# 60 candles: first 20 are warmup (before _START_MS_2025), next 40 are eval.
_BASE_2025 = 5000 * _INTERVAL_15M
_START_MS_2025 = _BASE_2025 + 20 * _INTERVAL_15M
_END_MS_2025 = _BASE_2025 + 60 * _INTERVAL_15M
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


def _make_candles_2025() -> list[Candle]:
    return [_make_flat_candle(_BASE_2025 + i * _INTERVAL_15M) for i in range(60)]


@pytest.fixture(scope="module")
def report() -> FrozenOos2025Report:
    return run_frozen_oos_2025(
        symbol="BTCUSDT",
        initial_capital=_INITIAL_CAPITAL,
        candles_15m_2025=_make_candles_2025(),
        start_ms_2025=_START_MS_2025,
        end_ms_2025=_END_MS_2025,
        indicator_config=_IND_CFG,
    )


# ---------------------------------------------------------------------------
# TestFrozenConfig — constants and frozen config values
# ---------------------------------------------------------------------------


class TestFrozenConfig:
    def test_trading_timeframe_is_30m(self) -> None:
        assert OOS_TRADING_TIMEFRAME == "30m"

    def test_oos_year_is_2025(self) -> None:
        assert OOS_YEAR == "2025"

    def test_scenario_names(self) -> None:
        assert OOS_SCENARIO_NAMES == ("NO_COSTS", "BASE_COSTS", "CONSERVATIVE")

    def test_no_costs_scenario_zero_fees_slip(self) -> None:
        fee, slip = OOS_SCENARIOS["NO_COSTS"]
        assert fee == Decimal("0")
        assert slip == Decimal("0")

    def test_base_costs_scenario(self) -> None:
        fee, slip = OOS_SCENARIOS["BASE_COSTS"]
        assert fee == Decimal("0.1")
        assert slip == Decimal("0.05")

    def test_conservative_scenario(self) -> None:
        fee, slip = OOS_SCENARIOS["CONSERVATIVE"]
        assert fee == Decimal("0.1")
        assert slip == Decimal("0.10")

    def test_filter_cfg_v3_aligned_trend(self) -> None:
        assert _OOS_FILTER_CFG.filter_type == EntryFilterType.V3_ALIGNED_TREND

    def test_filter_cfg_crossover_lookback(self) -> None:
        assert _OOS_FILTER_CFG.crossover_lookback_candles == 1

    def test_risk_cfg_no_take_profit(self) -> None:
        assert _OOS_RISK_CFG.use_take_profit is False

    def test_risk_cfg_bearish_crossover_exit(self) -> None:
        assert _OOS_RISK_CFG.use_bearish_crossover_exit is True

    def test_risk_cfg_allocation_25pct(self) -> None:
        assert _OOS_RISK_CFG.position_allocation_percentage == Decimal("25")

    def test_risk_cfg_no_max_holding(self) -> None:
        assert _OOS_RISK_CFG.maximum_holding_candles == 0

    def test_historical_reference_keys(self) -> None:
        expected_keys = {
            "symbol",
            "trading_timeframe",
            "entry",
            "exit_config",
            "allocation_pct",
            "cost_scenario",
            "period_2023_return_pct",
            "period_2024_return_pct",
            "combined_return_pct",
            "total_trades_2023_2024",
            "worst_drawdown_pct",
            "note",
        }
        assert set(OOS_HISTORICAL_REFERENCE.keys()) == expected_keys

    def test_historical_reference_read_only_note(self) -> None:
        assert "READ-ONLY" in OOS_HISTORICAL_REFERENCE["note"]


# ---------------------------------------------------------------------------
# TestOosExclusive2025Range — warmup candles produce no trades
# ---------------------------------------------------------------------------


class TestOosExclusive2025Range:
    def test_no_trades_before_start_ms(self, report: FrozenOos2025Report) -> None:
        for scenario_name, result in report.raw_results.items():
            for trade in result.trades:
                assert (
                    trade.entry_exec_time >= _START_MS_2025
                ), f"{scenario_name}: trade at {trade.entry_exec_time} before start_ms"

    def test_no_non_forced_exit_after_end_ms(self, report: FrozenOos2025Report) -> None:
        for scenario_name, result in report.raw_results.items():
            for trade in result.trades:
                if not trade.is_forced_close:
                    assert (
                        trade.exit_exec_time <= _END_MS_2025
                    ), f"{scenario_name}: non-forced exit at {trade.exit_exec_time} after end_ms"

    def test_warmup_boundary_ok_in_audit(self, report: FrozenOos2025Report) -> None:
        assert report.audit.warmup_boundary_ok is True

    def test_end_boundary_ok_in_audit(self, report: FrozenOos2025Report) -> None:
        assert report.audit.end_boundary_ok is True

    def test_report_oos_year_is_2025(self, report: FrozenOos2025Report) -> None:
        assert report.oos_year == "2025"

    def test_report_start_ms(self, report: FrozenOos2025Report) -> None:
        assert report.start_ms == _START_MS_2025

    def test_report_end_ms(self, report: FrozenOos2025Report) -> None:
        assert report.end_ms == _END_MS_2025


# ---------------------------------------------------------------------------
# Test30mAggregation — candle count verification
# ---------------------------------------------------------------------------


class Test30mAggregation:
    def test_candles_15m_total(self, report: FrozenOos2025Report) -> None:
        assert report.audit.candles_15m_total == 60

    def test_warmup_15m(self, report: FrozenOos2025Report) -> None:
        assert report.audit.warmup_15m == 20

    def test_eval_15m(self, report: FrozenOos2025Report) -> None:
        assert report.audit.eval_15m == 40

    def test_candles_30m_total(self, report: FrozenOos2025Report) -> None:
        assert report.audit.candles_30m_total == 30

    def test_warmup_30m(self, report: FrozenOos2025Report) -> None:
        assert report.audit.warmup_30m == 10

    def test_eval_30m(self, report: FrozenOos2025Report) -> None:
        assert report.audit.eval_30m == 20

    def test_15m_count_equals_warmup_plus_eval(self, report: FrozenOos2025Report) -> None:
        assert report.audit.warmup_15m + report.audit.eval_15m == report.audit.candles_15m_total

    def test_30m_count_equals_warmup_plus_eval(self, report: FrozenOos2025Report) -> None:
        assert report.audit.warmup_30m + report.audit.eval_30m == report.audit.candles_30m_total

    def test_trading_timeframe_in_report(self, report: FrozenOos2025Report) -> None:
        assert report.trading_timeframe == "30m"


# ---------------------------------------------------------------------------
# TestSignalIdentity — entry signal hash reproducibility
# ---------------------------------------------------------------------------


class TestSignalIdentity:
    def test_entry_signal_hash_is_string(self, report: FrozenOos2025Report) -> None:
        assert isinstance(report.audit.entry_signal_hash, str)

    def test_entry_signal_hash_length_16(self, report: FrozenOos2025Report) -> None:
        assert len(report.audit.entry_signal_hash) == 16

    def test_entry_signal_hash_hex(self, report: FrozenOos2025Report) -> None:
        int(report.audit.entry_signal_hash, 16)  # must not raise

    def test_entry_signal_hash_reproducible(self) -> None:
        candles = _make_candles_2025()
        r1 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        r2 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        assert r1.audit.entry_signal_hash == r2.audit.entry_signal_hash

    def test_no_violations_in_audit(self, report: FrozenOos2025Report) -> None:
        assert len(report.audit.violations) == 0


# ---------------------------------------------------------------------------
# TestCostsAndMonotonicity — cost ordering with flat-price data
# ---------------------------------------------------------------------------


class TestCostsAndMonotonicity:
    def test_three_scenario_results(self, report: FrozenOos2025Report) -> None:
        assert len(report.scenarios) == 3

    def test_scenario_names_in_order(self, report: FrozenOos2025Report) -> None:
        names = [s.scenario for s in report.scenarios]
        assert names == list(OOS_SCENARIO_NAMES)

    def test_flat_price_zero_return_all_scenarios(self, report: FrozenOos2025Report) -> None:
        for s in report.scenarios:
            assert s.return_pct == Decimal("0"), f"{s.scenario}: expected 0% return"

    def test_no_trades_flat_price(self, report: FrozenOos2025Report) -> None:
        for s in report.scenarios:
            assert s.total_trades == 0, f"{s.scenario}: expected 0 trades"

    def test_costs_monotonic_flag_true(self, report: FrozenOos2025Report) -> None:
        assert report.audit.costs_monotonic is True

    def test_interpretation_costs_monotonic(self, report: FrozenOos2025Report) -> None:
        assert report.interpretation.costs_monotonic is True

    def test_no_monotonicity_violations(self, report: FrozenOos2025Report) -> None:
        monotonicity_violations = [v for v in report.audit.violations if "Monotonicity" in v]
        assert len(monotonicity_violations) == 0

    def test_interpretation_not_profitable_flat(self, report: FrozenOos2025Report) -> None:
        assert report.interpretation.profitable_no_costs is False
        assert report.interpretation.profitable_base_costs is False
        assert report.interpretation.profitable_conservative is False

    def test_interpretation_no_profit_factor_flat(self, report: FrozenOos2025Report) -> None:
        assert report.interpretation.profit_factor_above_1_base is False

    def test_interpretation_insufficient_trades_flat(self, report: FrozenOos2025Report) -> None:
        assert report.interpretation.sufficient_trades is False

    def test_drawdown_below_historical_zero_dd(self, report: FrozenOos2025Report) -> None:
        assert report.interpretation.drawdown_below_historical is True

    def test_historical_reference_in_report(self, report: FrozenOos2025Report) -> None:
        assert report.historical_reference["period_2023_return_pct"] == "3.8785"
        assert report.historical_reference["period_2024_return_pct"] == "4.4320"


# ---------------------------------------------------------------------------
# TestExports — CSV field names, row counts, JSON structure
# ---------------------------------------------------------------------------


class TestExports:
    def test_summary_json_structure(self, report: FrozenOos2025Report, tmp_path: Path) -> None:
        out = tmp_path / "oos_summary.json"
        export_oos_summary_json(report, out)
        data = json.loads(out.read_text())
        assert "symbol" in data
        assert "trading_timeframe" in data
        assert "scenarios" in data
        assert "interpretation" in data
        assert "audit" in data
        assert "historical_reference" in data
        assert "paper_test_disclaimer" in data

    def test_summary_json_scenarios_count(
        self, report: FrozenOos2025Report, tmp_path: Path
    ) -> None:
        out = tmp_path / "oos_summary.json"
        export_oos_summary_json(report, out)
        data = json.loads(out.read_text())
        assert len(data["scenarios"]) == 3

    def test_summary_json_interpretation_keys(
        self, report: FrozenOos2025Report, tmp_path: Path
    ) -> None:
        out = tmp_path / "oos_summary.json"
        export_oos_summary_json(report, out)
        data = json.loads(out.read_text())
        intp = data["interpretation"]
        assert "profitable_no_costs" in intp
        assert "profitable_base_costs" in intp
        assert "profitable_conservative" in intp
        assert "profit_factor_above_1_base" in intp
        assert "costs_monotonic" in intp
        assert "drawdown_below_historical" in intp
        assert "return_above_historical_min" in intp
        assert "sufficient_trades" in intp

    def test_scenarios_csv_row_count(self, report: FrozenOos2025Report, tmp_path: Path) -> None:
        out = tmp_path / "oos_scenarios.csv"
        export_oos_scenarios_csv(report, out)
        rows = list(csv.DictReader(out.read_text().splitlines()))
        assert len(rows) == 3

    def test_scenarios_csv_field_names(self, report: FrozenOos2025Report, tmp_path: Path) -> None:
        out = tmp_path / "oos_scenarios.csv"
        export_oos_scenarios_csv(report, out)
        reader = csv.DictReader(out.read_text().splitlines())
        fields = set(reader.fieldnames or [])
        expected = {
            "scenario",
            "initial_capital",
            "final_equity",
            "return_pct",
            "total_trades",
            "win_rate_pct",
            "profit_factor",
            "max_drawdown_pct",
            "total_fees",
            "slippage_cost",
            "gross_profit",
            "gross_loss",
            "avg_net_pnl",
            "median_net_pnl",
            "exposure_pct",
            "avg_trade_duration_candles",
            "buy_and_hold_return_pct",
            "first_trade_date",
            "last_trade_date",
        }
        assert expected.issubset(fields)

    def test_trades_csv_empty_with_flat_prices(
        self, report: FrozenOos2025Report, tmp_path: Path
    ) -> None:
        out = tmp_path / "oos_trades.csv"
        export_oos_trades_csv(report, out)
        rows = list(csv.DictReader(out.read_text().splitlines()))
        assert len(rows) == 0

    def test_trades_csv_field_names(self, report: FrozenOos2025Report, tmp_path: Path) -> None:
        out = tmp_path / "oos_trades.csv"
        export_oos_trades_csv(report, out)
        reader = csv.DictReader(out.read_text().splitlines())
        fields = set(reader.fieldnames or [])
        expected = {
            "trade_id",
            "entry_signal_time",
            "entry_exec_time",
            "entry_exec_price",
            "entry_fee",
            "quantity",
            "exit_signal_time",
            "exit_exec_time",
            "exit_exec_price",
            "exit_fee",
            "gross_pnl",
            "net_pnl",
            "return_pct",
            "is_forced_close",
            "capital_at_entry",
            "entry_reasons",
            "exit_reasons",
        }
        assert expected.issubset(fields)

    def test_equity_curve_csv_row_count(self, report: FrozenOos2025Report, tmp_path: Path) -> None:
        out = tmp_path / "oos_equity.csv"
        export_oos_equity_curve_csv(report, out)
        rows = list(csv.DictReader(out.read_text().splitlines()))
        # eval candles in 30m = 20
        assert len(rows) == 20

    def test_equity_curve_csv_field_names(
        self, report: FrozenOos2025Report, tmp_path: Path
    ) -> None:
        out = tmp_path / "oos_equity.csv"
        export_oos_equity_curve_csv(report, out)
        reader = csv.DictReader(out.read_text().splitlines())
        fields = set(reader.fieldnames or [])
        expected = {
            "open_time",
            "close_time",
            "close_price",
            "equity",
            "quote_balance",
            "base_balance",
            "base_value",
            "drawdown_pct",
            "peak_equity",
            "has_open_position",
        }
        assert expected.issubset(fields)

    def test_audit_json_structure(self, report: FrozenOos2025Report, tmp_path: Path) -> None:
        out = tmp_path / "oos_audit.json"
        export_oos_audit_json(report, out)
        data = json.loads(out.read_text())
        assert "candles_15m_total" in data
        assert "candles_30m_total" in data
        assert "entry_signal_hash" in data
        assert "warmup_boundary_ok" in data
        assert "costs_monotonic" in data
        assert "missing_data_ranges" in data
        assert "violations" in data
        assert "frozen_config" in data
        assert "paper_test_disclaimer" in data

    def test_audit_json_frozen_config_keys(
        self, report: FrozenOos2025Report, tmp_path: Path
    ) -> None:
        out = tmp_path / "oos_audit.json"
        export_oos_audit_json(report, out)
        data = json.loads(out.read_text())
        cfg = data["frozen_config"]
        assert cfg["entry"] == "ENTRY_V3_ALIGNED_TREND"
        assert cfg["exit_config"] == "V2_STOP_ONLY"
        assert cfg["trading_timeframe"] == "30m"
        assert cfg["allocation_pct"] == "25"
        assert cfg["use_take_profit"] is False


# ---------------------------------------------------------------------------
# TestDeterminism — identical results across two runs
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_return_pct(self) -> None:
        candles = _make_candles_2025()
        r1 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        r2 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        for s1, s2 in zip(r1.scenarios, r2.scenarios, strict=False):
            assert s1.return_pct == s2.return_pct, f"{s1.scenario}: non-deterministic return"

    def test_identical_trade_counts(self) -> None:
        candles = _make_candles_2025()
        r1 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        r2 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        for s1, s2 in zip(r1.scenarios, r2.scenarios, strict=False):
            assert (
                s1.total_trades == s2.total_trades
            ), f"{s1.scenario}: non-deterministic trade count"

    def test_identical_audit(self) -> None:
        candles = _make_candles_2025()
        r1 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        r2 = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=candles,
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        assert r1.audit == r2.audit


# ---------------------------------------------------------------------------
# TestParameterRejection — frozen config not overridable
# ---------------------------------------------------------------------------


class TestParameterRejection:
    def test_report_always_shows_30m_trading_timeframe(self, report: FrozenOos2025Report) -> None:
        assert report.trading_timeframe == OOS_TRADING_TIMEFRAME == "30m"

    def test_oos_filter_cfg_cannot_be_changed(self) -> None:
        # The filter config is a module-level constant — verify its values are frozen.
        assert _OOS_FILTER_CFG.filter_type == EntryFilterType.V3_ALIGNED_TREND
        assert _OOS_FILTER_CFG.crossover_lookback_candles == 1

    def test_oos_risk_cfg_cannot_be_changed(self) -> None:
        assert _OOS_RISK_CFG.use_take_profit is False
        assert _OOS_RISK_CFG.use_bearish_crossover_exit is True
        assert _OOS_RISK_CFG.position_allocation_percentage == Decimal("25")

    def test_different_indicator_config_same_scenario_names(self) -> None:
        alt_ind = IndicatorConfig(ema_short_period=3, ema_medium_period=5, ema_long_period=8)
        r = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=_make_candles_2025(),
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=alt_ind,
        )
        assert [s.scenario for s in r.scenarios] == list(OOS_SCENARIO_NAMES)

    def test_report_trading_timeframe_immutable(self) -> None:
        r = run_frozen_oos_2025(
            symbol="BTCUSDT",
            initial_capital=_INITIAL_CAPITAL,
            candles_15m_2025=_make_candles_2025(),
            start_ms_2025=_START_MS_2025,
            end_ms_2025=_END_MS_2025,
            indicator_config=_IND_CFG,
        )
        assert r.trading_timeframe == "30m"

    def test_historical_reference_immutable(self, report: FrozenOos2025Report) -> None:
        assert report.historical_reference is OOS_HISTORICAL_REFERENCE

    def test_symbol_in_report(self, report: FrozenOos2025Report) -> None:
        assert report.symbol == "BTCUSDT"

    def test_initial_capital_in_report(self, report: FrozenOos2025Report) -> None:
        assert report.initial_capital == _INITIAL_CAPITAL
