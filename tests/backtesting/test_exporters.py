"""Tests for JSON and CSV exporters."""

import csv
import json
from decimal import Decimal

import pytest

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.exporters import (
    export_equity_csv,
    export_json,
    export_trades_csv,
    result_to_dict,
)
from tests.backtesting.conftest import (
    AlwaysWaitEngine,
    BuyThenSellEngine,
    make_candles,
)

_D = Decimal
_BASE_TIME = 1_000_000
_INTERVAL_MS = 3_600_000


def _minimal_ind():
    from app.indicators.schemas import IndicatorConfig

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


def _cfg(n: int) -> BacktestConfig:
    return BacktestConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_ms=_BASE_TIME,
        end_ms=_BASE_TIME + n * _INTERVAL_MS,
        initial_capital=_D("10000"),
        fee_percentage=_D("0.1"),
        slippage_percentage=_D("0.05"),
        force_close_at_end=True,
    )


def _make_result(engine_cls=AlwaysWaitEngine, n: int = 15):
    candles = make_candles(n)
    cfg = _cfg(n)
    engine = BacktestEngine(cfg, engine_cls(), _minimal_ind())
    return engine.run(candles, warmup_len=0)


class TestResultToDict:
    def test_has_required_top_level_keys(self):
        result = _make_result()
        d = result_to_dict(result)
        assert "config" in d
        assert "summary" in d
        assert "trades" in d
        assert "equity_curve" in d
        assert "warning" in d

    def test_warning_present(self):
        result = _make_result()
        d = result_to_dict(result)
        assert d["warning"]  # non-empty string

    def test_decimals_serialized_as_strings(self):
        result = _make_result()
        d = result_to_dict(result)
        assert isinstance(d["summary"]["initial_capital"], str)
        assert isinstance(d["summary"]["final_equity"], str)
        assert isinstance(d["config"]["initial_capital"], str)

    def test_equity_curve_items_are_dicts(self):
        result = _make_result()
        d = result_to_dict(result)
        for ep in d["equity_curve"]:
            assert isinstance(ep, dict)
            assert "equity" in ep
            assert "open_time" in ep

    def test_trades_items_are_dicts(self):
        result = _make_result(BuyThenSellEngine, n=20)
        d = result_to_dict(result)
        for trade in d["trades"]:
            assert isinstance(trade, dict)
            assert "trade_id" in trade
            assert "net_pnl" in trade

    def test_no_float_in_financial_fields(self):
        result = _make_result(BuyThenSellEngine, n=20)
        d = result_to_dict(result)
        for key in ("initial_capital", "final_equity", "total_return_pct"):
            assert isinstance(d["summary"][key], str), f"{key} should be str, not float"


class TestExportJson:
    def test_creates_valid_json_file(self, tmp_path):
        result = _make_result()
        path = tmp_path / "result.json"
        export_json(result, path)
        assert path.exists()
        with path.open() as f:
            data = json.load(f)
        assert "summary" in data

    def test_json_file_has_warning(self, tmp_path):
        result = _make_result()
        path = tmp_path / "result.json"
        export_json(result, path)
        with path.open() as f:
            data = json.load(f)
        assert "warning" in data
        assert data["warning"]

    def test_json_contains_equity_curve(self, tmp_path):
        n = 10
        result = _make_result(n=n)
        path = tmp_path / "equity.json"
        export_json(result, path)
        with path.open() as f:
            data = json.load(f)
        assert len(data["equity_curve"]) == n


class TestExportEquityCsv:
    def test_creates_csv_with_header(self, tmp_path):
        result = _make_result()
        path = tmp_path / "equity.csv"
        export_equity_csv(result, path)
        assert path.exists()
        with path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(result.equity_curve)

    def test_csv_has_expected_columns(self, tmp_path):
        result = _make_result()
        path = tmp_path / "equity.csv"
        export_equity_csv(result, path)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert "equity" in reader.fieldnames
            assert "open_time" in reader.fieldnames
            assert "drawdown_pct" in reader.fieldnames


class TestExportTradesCsv:
    def test_no_file_created_for_no_trades(self, tmp_path):
        result = _make_result()  # AlwaysWait = no trades
        path = tmp_path / "trades.csv"
        export_trades_csv(result, path)
        assert not path.exists()

    def test_creates_csv_when_trades_exist(self, tmp_path):
        result = _make_result(BuyThenSellEngine, n=20)
        path = tmp_path / "trades.csv"
        export_trades_csv(result, path)
        if result.trades:
            assert path.exists()
            with path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == len(result.trades)

    def test_trades_csv_has_pnl_columns(self, tmp_path):
        result = _make_result(BuyThenSellEngine, n=20)
        if not result.trades:
            pytest.skip("No trades generated")
        path = tmp_path / "trades.csv"
        export_trades_csv(result, path)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert "net_pnl" in reader.fieldnames
            assert "return_pct" in reader.fieldnames
