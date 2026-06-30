"""Stage 6.1 tests: forward exporters — CSV and JSON serialisation."""

from __future__ import annotations

import csv
import json
import types
from datetime import datetime
from decimal import Decimal

from app.forward.exporters import (
    export_forward_equity_curve_csv,
    export_forward_manifest_json,
    export_forward_orders_csv,
    export_forward_signals_csv,
    export_forward_system_events_csv,
    export_forward_trades_csv,
)
from app.forward.manifest import FROZEN_CONFIG_HASH


def _evaluation(**overrides):
    fields = {
        "candle_close_time": datetime(2024, 1, 1, 4, 0, 0),
        "evaluated_at": datetime(2024, 1, 1, 4, 0, 1),
        "signal": "WAIT",
        "reasons": ["NO_SIGNAL"],
        "raw_market_price": Decimal("100"),
        "planned_execution_time": None,
        "entry_donchian_level": None,
        "exit_donchian_level": None,
        "ema_200": Decimal("95"),
        "ema_slope": Decimal("0.01"),
        "atr": Decimal("5"),
        "initial_stop": None,
        "current_trailing_stop": None,
        "highest_high_since_entry": None,
        "position_quantity": None,
        "cash": Decimal("10000"),
        "equity": Decimal("10000"),
        "frozen_config_hash": FROZEN_CONFIG_HASH,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _order(**overrides):
    fields = {
        "id": 1,
        "client_order_id": "fwd-1-buy-123",
        "side": "BUY",
        "order_type": "MARKET",
        "status": "FILLED",
        "price": None,
        "quantity": Decimal("1"),
        "filled_quantity": Decimal("1"),
        "avg_fill_price": Decimal("100.05"),
        "stop_price": None,
        "commission": Decimal("0.1"),
        "position_id": 1,
        "created_at": datetime(2024, 1, 1),
        "updated_at": datetime(2024, 1, 1),
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _trade(**overrides):
    fields = {
        "id": 1,
        "entry_price": Decimal("100"),
        "exit_price": Decimal("120"),
        "quantity": Decimal("1"),
        "side": "BUY",
        "gross_pnl": Decimal("20"),
        "commission": Decimal("0.2"),
        "net_pnl": Decimal("19.8"),
        "planned_stop_loss": Decimal("90"),
        "planned_take_profit": Decimal("0"),
        "exit_reason": "DONCHIAN_CHANNEL_EXIT",
        "opened_at": datetime(2024, 1, 1),
        "closed_at": datetime(2024, 1, 2),
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _system_event(**overrides):
    fields = {
        "timestamp": datetime(2024, 1, 1, 0, 0, 0),
        "level": "ERROR",
        "source": "forward_engine",
        "message": "Test gap event",
        "details": {"launch_id": 1},
        "trading_mode": "PAPER",
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _launch(**overrides):
    fields = {
        "id": 1,
        "strategy_name": "donchian_breakout_B_4h",
        "strategy_version": "6.1.0",
        "symbol": "BTCUSDT",
        "launch_timestamp": datetime(2024, 1, 1),
        "initial_capital": Decimal("10000"),
        "status": "ACTIVE",
        "frozen_config_hash": FROZEN_CONFIG_HASH,
        "code_commit_hash": None,
        "last_evaluated_candle_close": None,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


class TestExportForwardSignalsCsv:
    def test_writes_header_and_data_row(self, tmp_path):
        path = tmp_path / "signals.csv"
        export_forward_signals_csv([_evaluation()], path)

        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["signal"] == "WAIT"
        assert row["candle_close_time"] == "2024-01-01T04:00:00"
        assert row["frozen_config_hash"] == FROZEN_CONFIG_HASH

    def test_reasons_joined_with_pipe(self, tmp_path):
        path = tmp_path / "signals.csv"
        ev = _evaluation(reasons=["R1", "R2"])
        export_forward_signals_csv([ev], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["reasons"] == "R1|R2"

    def test_decimal_fields_serialised_as_strings(self, tmp_path):
        path = tmp_path / "signals.csv"
        ev = _evaluation(raw_market_price=Decimal("99.12345678901234"))
        export_forward_signals_csv([ev], path)
        rows = list(csv.DictReader(path.open()))
        # Must be a string representation, not a float that could lose precision
        assert "." in rows[0]["raw_market_price"]
        float(rows[0]["raw_market_price"])  # parseable

    def test_empty_list_writes_header_only(self, tmp_path):
        path = tmp_path / "signals.csv"
        export_forward_signals_csv([], path)
        lines = [line for line in path.read_text().strip().splitlines() if line.strip()]
        assert len(lines) == 1  # only header

    def test_none_fields_written_as_empty(self, tmp_path):
        path = tmp_path / "signals.csv"
        export_forward_signals_csv([_evaluation(entry_donchian_level=None)], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["entry_donchian_level"] == ""


class TestExportForwardOrdersCsv:
    def test_writes_order_row(self, tmp_path):
        path = tmp_path / "orders.csv"
        export_forward_orders_csv([_order()], path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1
        assert rows[0]["side"] == "BUY"
        assert rows[0]["status"] == "FILLED"
        assert rows[0]["avg_fill_price"] == "100.05"

    def test_none_price_serialised_as_empty(self, tmp_path):
        path = tmp_path / "orders.csv"
        export_forward_orders_csv([_order(price=None)], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["price"] == ""


class TestExportForwardTradesCsv:
    def test_writes_trade_row(self, tmp_path):
        path = tmp_path / "trades.csv"
        export_forward_trades_csv([_trade()], path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1
        assert rows[0]["exit_reason"] == "DONCHIAN_CHANNEL_EXIT"
        assert rows[0]["net_pnl"] == "19.8"

    def test_empty_trades_writes_header_only(self, tmp_path):
        path = tmp_path / "trades.csv"
        export_forward_trades_csv([], path)
        lines = [line for line in path.read_text().strip().splitlines() if line.strip()]
        assert len(lines) == 1


class TestExportForwardEquityCurveCsv:
    def test_writes_equity_row(self, tmp_path):
        path = tmp_path / "equity.csv"
        export_forward_equity_curve_csv([_evaluation()], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["cash"] == "10000"
        assert rows[0]["equity"] == "10000"
        assert rows[0]["signal"] == "WAIT"

    def test_position_quantity_none_is_empty(self, tmp_path):
        path = tmp_path / "equity.csv"
        export_forward_equity_curve_csv([_evaluation(position_quantity=None)], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["position_quantity"] == ""


class TestExportForwardSystemEventsCsv:
    def test_writes_event_row(self, tmp_path):
        path = tmp_path / "events.csv"
        export_forward_system_events_csv([_system_event()], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["level"] == "ERROR"
        assert rows[0]["source"] == "forward_engine"
        assert rows[0]["trading_mode"] == "PAPER"

    def test_details_serialised_as_json_string(self, tmp_path):
        path = tmp_path / "events.csv"
        export_forward_system_events_csv([_system_event(details={"launch_id": 42})], path)
        rows = list(csv.DictReader(path.open()))
        details = json.loads(rows[0]["details"])
        assert details["launch_id"] == 42

    def test_none_details_written_as_empty(self, tmp_path):
        path = tmp_path / "events.csv"
        export_forward_system_events_csv([_system_event(details=None)], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["details"] == ""


class TestExportForwardManifestJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "manifest.json"
        export_forward_manifest_json(_launch(), path)
        doc = json.loads(path.read_text())
        assert doc["launch_id"] == 1
        assert doc["strategy_name"] == "donchian_breakout_B_4h"
        assert doc["symbol"] == "BTCUSDT"

    def test_includes_disclaimer(self, tmp_path):
        path = tmp_path / "manifest.json"
        export_forward_manifest_json(_launch(), path)
        doc = json.loads(path.read_text())
        assert "PAPER" in doc["paper_test_disclaimer"]
        assert "No real money" in doc["paper_test_disclaimer"]

    def test_decimal_initial_capital_serialised_as_string(self, tmp_path):
        path = tmp_path / "manifest.json"
        export_forward_manifest_json(_launch(initial_capital=Decimal("10000")), path)
        doc = json.loads(path.read_text())
        assert doc["initial_capital"] == "10000"

    def test_includes_frozen_configuration(self, tmp_path):
        path = tmp_path / "manifest.json"
        export_forward_manifest_json(_launch(), path)
        doc = json.loads(path.read_text())
        cfg = doc["frozen_configuration"]
        assert cfg["symbol"] == "BTCUSDT"
        assert cfg["entry_lookback"] == 20
        assert cfg["atr_multiplier"] == "3.0"

    def test_includes_future_evaluation_criteria(self, tmp_path):
        path = tmp_path / "manifest.json"
        export_forward_manifest_json(_launch(), path)
        doc = json.loads(path.read_text())
        criteria = doc["future_evaluation_criteria"]
        assert criteria["first_checkpoint_days"] == 90
        assert criteria["min_closed_trades_before_real_money"] == 20
