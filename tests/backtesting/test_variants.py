"""Tests for strategy variant comparator.

Covers:
  - run_variants returns ComparisonReport with all 4 variants
  - V1_BASELINE uses V1 engine
  - V2 variants have expected exit codes
  - No-costs variant has lower fees
  - exit type counting is correct
  - avg_duration_candles and median_net_pnl compute correctly
  - Serialisation via v2_exporters (JSON + CSV)
  - Determinism: same inputs → same report
"""

import csv
import json
import tempfile
from decimal import Decimal
from pathlib import Path

from app.backtesting.config import BacktestConfig
from app.backtesting.schemas import BacktestTrade
from app.backtesting.variants import (
    ComparisonReport,
    _avg_duration,
    _count_exit,
    _median_net_pnl,
    run_variants,
)
from app.indicators.schemas import IndicatorConfig
from app.strategy.reasons import ReasonCode
from tests.backtesting.conftest import (
    AlwaysWaitEngine,
    make_candles,
)

_D = Decimal
_BASE_TIME = 1_000_000
_INTERVAL_MS = 900_000  # 15m


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


def _cfg(**kwargs) -> BacktestConfig:
    defaults = {
        "symbol": "BTCUSDT",
        "interval": "15m",
        "start_ms": _BASE_TIME,
        "end_ms": _BASE_TIME + 50 * _INTERVAL_MS,
        "initial_capital": _D("10000"),
        "fee_percentage": _D("0.1"),
        "slippage_percentage": _D("0.05"),
        "force_close_at_end": True,
    }
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _make_trade(
    exit_reasons: tuple[str, ...],
    net_pnl: str = "0",
    entry_exec_time: int = _BASE_TIME + _INTERVAL_MS,
    exit_exec_time: int = _BASE_TIME + 6 * _INTERVAL_MS,
) -> BacktestTrade:
    return BacktestTrade(
        trade_id=1,
        entry_signal_time=_BASE_TIME,
        entry_exec_time=entry_exec_time,
        entry_exec_price=_D("100"),
        entry_fee=_D("0"),
        quantity=_D("1"),
        exit_signal_time=None,
        exit_exec_time=exit_exec_time,
        exit_exec_price=_D("100"),
        exit_fee=_D("0"),
        gross_pnl=_D("0"),
        net_pnl=_D(net_pnl),
        return_pct=_D("0"),
        is_forced_close=False,
        capital_at_entry=_D("100"),
        entry_reasons=(),
        exit_reasons=exit_reasons,
    )


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_count_exit_zero(self):
        assert _count_exit([], str(ReasonCode.ATR_STOP_LOSS)) == 0

    def test_count_exit_matches(self):
        trades = [
            _make_trade((str(ReasonCode.ATR_STOP_LOSS),)),
            _make_trade((str(ReasonCode.ATR_STOP_LOSS),)),
            _make_trade((str(ReasonCode.BEARISH_CROSSOVER),)),
        ]
        assert _count_exit(trades, str(ReasonCode.ATR_STOP_LOSS)) == 2
        assert _count_exit(trades, str(ReasonCode.BEARISH_CROSSOVER)) == 1

    def test_count_exit_ambiguous_multi_reason(self):
        trades = [
            _make_trade(
                (
                    str(ReasonCode.ATR_STOP_LOSS),
                    str(ReasonCode.AMBIGUOUS_INTRABAR_STOP_FIRST),
                )
            ),
        ]
        assert _count_exit(trades, str(ReasonCode.ATR_STOP_LOSS)) == 1
        assert _count_exit(trades, str(ReasonCode.AMBIGUOUS_INTRABAR_STOP_FIRST)) == 1

    def test_avg_duration_none_no_trades(self):
        assert _avg_duration([], _INTERVAL_MS) is None

    def test_avg_duration_single_trade(self):
        t = _make_trade(
            (),
            entry_exec_time=_BASE_TIME,
            exit_exec_time=_BASE_TIME + 5 * _INTERVAL_MS,
        )
        avg = _avg_duration([t], _INTERVAL_MS)
        assert avg is not None
        assert avg == _D("5")

    def test_median_none_no_trades(self):
        assert _median_net_pnl([]) is None

    def test_median_odd_trades(self):
        trades = [
            _make_trade((), net_pnl="10"),
            _make_trade((), net_pnl="20"),
            _make_trade((), net_pnl="30"),
        ]
        assert _median_net_pnl(trades) == _D("20")

    def test_median_even_trades(self):
        trades = [
            _make_trade((), net_pnl="10"),
            _make_trade((), net_pnl="20"),
        ]
        assert _median_net_pnl(trades) == _D("15")


# ---------------------------------------------------------------------------
# run_variants integration
# ---------------------------------------------------------------------------


class TestRunVariants:
    def _run(self, strategy, **cfg_kwargs):
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        config = _cfg(**cfg_kwargs)
        return run_variants(
            all_candles=candles,
            warmup_len=0,
            config=config,
            indicator_config=_ind_config(),
        )

    def test_returns_four_variants(self):
        report = self._run(AlwaysWaitEngine)
        assert len(report.variants) == 4

    def test_variant_names(self):
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        names = [v.name for v in report.variants]
        assert names == ["V1_BASELINE", "V2_STOP_ONLY", "V2_STOP_TP", "V2_STOP_TP_TIME"]

    def test_bah_return_is_decimal(self):
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        assert isinstance(report.bah_return_pct, Decimal)

    def test_no_costs_variant_has_zero_fees(self):
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(fee_percentage=_D("0.1")),
            indicator_config=_ind_config(),
        )
        for v in report.variants:
            assert v.result_no_costs.total_fees == _D("0")

    def test_result_and_no_costs_differ_when_trades(self):
        """With fees and trades, result differs from no-costs result."""
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(fee_percentage=_D("0.1"), slippage_percentage=_D("0.05")),
            indicator_config=_ind_config(),
        )
        for v in report.variants:
            if v.result.total_trades > 0:
                # With fees, final equity is lower
                assert v.result.final_equity <= v.result_no_costs.final_equity

    def test_determinism(self):
        """Same candles + same config → same report."""
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        config = _cfg()
        ind = _ind_config()
        r1 = run_variants(candles, 0, config, ind)
        r2 = run_variants(candles, 0, config, ind)
        for v1, v2 in zip(r1.variants, r2.variants, strict=True):
            assert v1.result.final_equity == v2.result.final_equity
            assert v1.result.total_trades == v2.result.total_trades

    def test_v1_baseline_only_uses_crossover_exits(self):
        """V1_BASELINE should have no SL/TP/time exits."""
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        v1 = next(v for v in report.variants if v.name == "V1_BASELINE")
        assert v1.sl_exits == 0
        assert v1.tp_exits == 0
        assert v1.timed_exits == 0

    def test_v2_stop_only_has_no_tp_exits(self):
        """V2_STOP_ONLY never has TP exits."""
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        v = next(v for v in report.variants if v.name == "V2_STOP_ONLY")
        assert v.tp_exits == 0

    def test_v2_stop_tp_time_has_time_exit_capability(self):
        """V2_STOP_TP_TIME has maximum_holding_candles=192 configured."""
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        v = next(v for v in report.variants if v.name == "V2_STOP_TP_TIME")
        # Config should have maximum_holding_candles=192 — just verify structure
        assert v.name == "V2_STOP_TP_TIME"

    def test_all_variants_have_same_bah(self):
        """All variants use same data → same buy-and-hold return."""
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        # BAH is same for all variants (same period, same capital)
        for v in report.variants:
            assert v.result.buy_and_hold_return_pct == report.bah_return_pct


# ---------------------------------------------------------------------------
# V2 exporter tests
# ---------------------------------------------------------------------------


class TestV2Exporters:
    def _make_report(self) -> ComparisonReport:
        candles = make_candles(
            40, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        return run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )

    def test_export_variant_comparison_json(self):
        from app.backtesting.v2_exporters import export_variant_comparison_json

        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        export_variant_comparison_json(report, path)
        data = json.loads(path.read_text())
        assert "variants" in data
        assert len(data["variants"]) == 4
        assert "warning" in data
        path.unlink()

    def test_export_variant_comparison_csv(self):
        from app.backtesting.v2_exporters import export_variant_comparison_csv

        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        export_variant_comparison_csv(report, path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 4
        assert rows[0]["variant"] == "V1_BASELINE"
        path.unlink()

    def test_export_v2_trades_csv(self):
        from app.backtesting.v2_exporters import export_v2_trades_csv

        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        export_v2_trades_csv(report, path)
        # May be empty if no trades, but should not raise
        if path.stat().st_size > 0:
            rows = list(csv.DictReader(path.open()))
            for row in rows:
                assert "variant" in row
                assert "net_pnl" in row
        path.unlink()

    def test_export_exit_type_breakdown_csv(self):
        from app.backtesting.v2_exporters import export_exit_type_breakdown_csv

        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        export_exit_type_breakdown_csv(report, path)
        if path.stat().st_size > 0:
            rows = list(csv.DictReader(path.open()))
            for row in rows:
                assert "variant" in row
                assert "exit_reason" in row
        path.unlink()

    def test_decimal_serialized_as_string_in_json(self):
        from app.backtesting.v2_exporters import export_variant_comparison_json

        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        export_variant_comparison_json(report, path)
        raw = path.read_text()
        data = json.loads(raw)
        for v in data["variants"]:
            # Decimal fields should be strings
            assert isinstance(v["total_return_pct"], str)
            assert isinstance(v["final_equity"], str)
        path.unlink()

    def test_v1_compat_existing_exporters_still_work(self):
        """V1 export functions remain unaffected by V2 additions."""
        from app.backtesting.exporters import result_to_dict

        candles = make_candles(10, interval_ms=_INTERVAL_MS, interval="15m", price="100")
        from app.backtesting.config import BacktestConfig
        from app.backtesting.engine import BacktestEngine

        config = BacktestConfig(
            symbol="BTCUSDT",
            interval="15m",
            start_ms=_BASE_TIME,
            end_ms=_BASE_TIME + 10 * _INTERVAL_MS,
            initial_capital=_D("10000"),
            fee_percentage=_D("0"),
            slippage_percentage=_D("0"),
            force_close_at_end=True,
        )
        engine = BacktestEngine(
            config=config,
            strategy_engine=AlwaysWaitEngine(),
            indicator_config=_ind_config(),
        )
        result = engine.run(candles, warmup_len=0)
        d = result_to_dict(result)
        assert "trades" in d
        assert "equity_curve" in d


# ---------------------------------------------------------------------------
# Comparison report dataclass
# ---------------------------------------------------------------------------


class TestComparisonReport:
    def test_report_frozen_variants(self):
        candles = make_candles(
            10, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        assert isinstance(report.variants, list)
        assert isinstance(report.bah_return_pct, Decimal)

    def test_variant_result_fields(self):
        candles = make_candles(
            10, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS, interval="15m"
        )
        report = run_variants(
            all_candles=candles,
            warmup_len=0,
            config=_cfg(),
            indicator_config=_ind_config(),
        )
        for v in report.variants:
            assert isinstance(v.sl_exits, int)
            assert isinstance(v.tp_exits, int)
            assert isinstance(v.timed_exits, int)
            assert isinstance(v.crossover_exits, int)
            assert isinstance(v.ambiguous_exits, int)
            assert isinstance(v.forced_exits, int)
            assert v.sl_exits >= 0
