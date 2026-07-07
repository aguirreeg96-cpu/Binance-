"""Tests for Stage 6.2: web dashboard routes and dashboard-summary endpoint."""

from __future__ import annotations

from unittest.mock import patch as _patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_session(test_engine):
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def api_client(test_session):
    from app.main import create_app

    application = create_app()
    application.dependency_overrides[get_db] = lambda: test_session

    with (
        _patch("app.main.run_migrations"),
        _patch("app.main.verify_db_connection"),
    ):
        with TestClient(application) as client:
            yield client


# ---------------------------------------------------------------------------
# Dashboard HTML page
# ---------------------------------------------------------------------------


class TestDashboardPage:
    def test_dashboard_returns_200(self, api_client):
        resp = api_client.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_content_type_html(self, api_client):
        resp = api_client.get("/dashboard")
        assert "text/html" in resp.headers["content-type"]

    def test_dashboard_references_css(self, api_client):
        resp = api_client.get("/dashboard")
        assert "/dashboard/static/dashboard.css" in resp.text

    def test_dashboard_references_js(self, api_client):
        resp = api_client.get("/dashboard")
        assert "/dashboard/static/dashboard.js" in resp.text

    def test_dashboard_has_cache_control_no_store(self, api_client):
        resp = api_client.get("/dashboard")
        assert resp.headers.get("cache-control") == "no-store"

    def test_dashboard_has_x_content_type_nosniff(self, api_client):
        resp = api_client.get("/dashboard")
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_dashboard_has_csp_header(self, api_client):
        resp = api_client.get("/dashboard")
        csp = resp.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp

    def test_dashboard_contains_paper_trading_warning(self, api_client):
        resp = api_client.get("/dashboard")
        assert "PAPER" in resp.text

    def test_dashboard_does_not_expose_api_keys(self, api_client):
        resp = api_client.get("/dashboard")
        body = resp.text.lower()
        assert "api_key" not in body
        assert "secret" not in body

    def test_dashboard_has_required_element_ids(self, api_client):
        resp = api_client.get("/dashboard")
        required_ids = [
            "launch-status-badge",
            "update-indicator",
            "last-updated",
            "error-banner",
            "warnings-bar",
            "card-signal",
            "card-balance",
            "card-equity",
            "card-pnl",
            "card-pnl-pct",
            "card-position",
            "card-trades-count",
            "card-winrate",
            "current-eval",
            "position-container",
            "signal-filter",
            "signals-tbody",
            "trades-empty",
            "trades-table-wrapper",
            "trades-tbody",
            "equity-chart-container",
            "frozen-config-grid",
            "system-status-grid",
        ]
        for elem_id in required_ids:
            assert elem_id in resp.text, f"Missing element id: {elem_id}"

    def test_dashboard_has_export_buttons(self, api_client):
        resp = api_client.get("/dashboard")
        assert "exportSignals()" in resp.text
        assert "exportTrades()" in resp.text
        assert "exportEquity()" in resp.text
        assert "exportManifest()" in resp.text

    def test_dashboard_has_refresh_button(self, api_client):
        resp = api_client.get("/dashboard")
        assert "refreshDashboard()" in resp.text


# ---------------------------------------------------------------------------
# Static files served correctly
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_css_file_served(self, api_client):
        resp = api_client.get("/dashboard/static/dashboard.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    def test_js_file_served(self, api_client):
        resp = api_client.get("/dashboard/static/dashboard.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]

    def test_nonexistent_static_returns_404(self, api_client):
        resp = api_client.get("/dashboard/static/nonexistent.js")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Dashboard summary — no launch
# ---------------------------------------------------------------------------


class TestDashboardSummaryNoLaunch:
    def test_summary_200_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        assert resp.status_code == 200

    def test_summary_launch_exists_false(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["launch_exists"] is False

    def test_summary_launch_null_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["launch"] is None
        assert data["account"] is None
        assert data["position"] is None

    def test_summary_empty_collections_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["signals"] == []
        assert data["recent_trades"] == []
        assert data["equity_curve"] == []

    def test_summary_has_warnings_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert len(data["warnings"]) > 0

    def test_summary_frozen_config_present_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert isinstance(data["frozen_config"], dict)
        assert len(data["frozen_config"]) > 0

    def test_summary_system_status_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        sys = data["system"]
        assert sys["launch_exists"] is False
        assert sys["total_evaluations"] == 0

    def test_summary_trade_stats_zeros_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        stats = data["trade_stats"]
        assert stats["total_trades"] == 0
        assert stats["win_rate_pct"] is None

    def test_summary_has_generated_at(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert "generated_at" in data
        assert data["generated_at"] != ""

    def test_summary_cache_control_no_store(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        assert resp.headers.get("cache-control") == "no-store"

    def test_summary_x_content_type_nosniff(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        assert resp.headers.get("x-content-type-options") == "nosniff"


# ---------------------------------------------------------------------------
# Dashboard summary — with launch
# ---------------------------------------------------------------------------


@pytest.fixture
def launched_client(test_engine):
    """Client with a fresh DB that has a launch already created."""
    from app.main import create_app

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    application = create_app()
    Session = sessionmaker(bind=engine)
    session = Session()
    application.dependency_overrides[get_db] = lambda: session

    with (
        _patch("app.main.run_migrations"),
        _patch("app.main.verify_db_connection"),
    ):
        with TestClient(application) as client:
            # Create the launch
            resp = client.post(
                "/api/v1/paper-breakout/start",
                json={"initial_capital": "10000"},
            )
            assert resp.status_code == 200
            yield client

    session.close()
    engine.dispose()


class TestDecisionField:
    def test_decision_null_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["decision"] is None

    def test_decision_present_with_launch(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["decision"] is not None

    def test_decision_has_required_fields(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        dec = resp.json()["decision"]
        for field in (
            "current_decision",
            "action_label",
            "action_severity",
            "plain_language_explanation",
            "failed_conditions",
            "is_entry_signal",
            "is_exit_signal",
            "has_open_position",
        ):
            assert field in dec, f"Missing field: {field}"

    def test_decision_current_decision_is_valid_enum(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        dec = resp.json()["decision"]
        valid = {
            "NO_COMPRAR",
            "ENTRADA_DETECTADA",
            "POSICION_ABIERTA",
            "CERRAR_POSICION",
            "OPERACION_CERRADA",
            "REVISAR_SISTEMA",
        }
        assert dec["current_decision"] in valid

    def test_decision_severity_is_valid(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        dec = resp.json()["decision"]
        assert dec["action_severity"] in ("neutral", "success", "info", "warning", "danger")

    def test_decision_boolean_flags(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        dec = resp.json()["decision"]
        assert isinstance(dec["is_entry_signal"], bool)
        assert isinstance(dec["is_exit_signal"], bool)
        assert isinstance(dec["has_open_position"], bool)

    def test_decision_failed_conditions_is_list(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        dec = resp.json()["decision"]
        assert isinstance(dec["failed_conditions"], list)

    def test_decision_no_launch_gives_no_comprar_when_no_signals(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        dec = data["decision"]
        if not data["signals"]:
            assert dec["current_decision"] == "NO_COMPRAR"


class TestDashboardSummaryWithLaunch:
    def test_summary_200_with_launch(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        assert resp.status_code == 200

    def test_summary_launch_exists_true(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["launch_exists"] is True

    def test_summary_has_launch_object(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["launch"] is not None
        assert "status" in data["launch"]
        assert "initial_capital" in data["launch"]

    def test_summary_account_key_present(self, launched_client):
        """account key is always present; may be None before first cycle."""
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert "account" in data

    def test_summary_initial_capital_string(self, launched_client):
        """Decimal values must be serialized as strings, not floats."""
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        cap = data["launch"]["initial_capital"]
        assert isinstance(cap, str), "initial_capital must be a string (Decimal precision)"

    def test_summary_account_balance_string_if_present(self, launched_client):
        """If account exists, balance must be a string (Decimal precision)."""
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        if data["account"] is not None:
            assert isinstance(data["account"]["balance"], str)

    def test_summary_no_position_initially(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["position"] is None

    def test_summary_system_launch_exists_true(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["system"]["launch_exists"] is True

    def test_summary_signals_ordered_newest_first(self, launched_client):
        """Signals list must be ordered newest-first (reversed from DB order)."""
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        signals = data["signals"]
        if len(signals) >= 2:
            times = [s["candle_close_time"] for s in signals]
            assert times == sorted(times, reverse=True)

    def test_summary_frozen_config_has_expected_keys(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        cfg = data["frozen_config"]
        assert "strategy_name" in cfg
        assert "symbol" in cfg

    def test_summary_next_eligible_close_not_none(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        assert data["next_eligible_close_utc"] is not None

    def test_summary_trade_stats_present(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/dashboard-summary")
        data = resp.json()
        stats = data["trade_stats"]
        assert "total_trades" in stats
        assert "winning_trades" in stats
        assert "losing_trades" in stats
        assert "total_realized_pnl" in stats


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------


class TestExportEndpoints:
    def test_signals_export_404_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/signals/export")
        assert resp.status_code == 404

    def test_trades_export_404_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/trades/export")
        assert resp.status_code == 404

    def test_equity_export_404_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/equity/export")
        assert resp.status_code == 404

    def test_config_export_404_without_launch(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/config/export")
        assert resp.status_code == 404

    def test_signals_export_200_with_launch(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/signals/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_trades_export_200_with_launch(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/trades/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_equity_export_200_with_launch(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/equity/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_config_export_200_with_launch(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/config/export")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]

    def test_signals_export_filename_header(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/signals/export")
        cd = resp.headers.get("content-disposition", "")
        assert "forward_signals.csv" in cd

    def test_trades_export_filename_header(self, launched_client):
        resp = launched_client.get("/api/v1/paper-breakout/trades/export")
        cd = resp.headers.get("content-disposition", "")
        assert "forward_trades.csv" in cd


# ---------------------------------------------------------------------------
# Security — no Binance order endpoints on paper-breakout
# ---------------------------------------------------------------------------


class TestNoBinanceOrders:
    """Verify the paper-breakout API never exposes order-placement endpoints."""

    def test_no_place_order_endpoint(self, api_client):
        for method in ("post", "put", "patch"):
            resp = getattr(api_client, method)("/api/v1/paper-breakout/order")
            assert resp.status_code in (404, 405)

    def test_no_cancel_order_endpoint(self, api_client):
        resp = api_client.delete("/api/v1/paper-breakout/order/1")
        assert resp.status_code in (404, 405)

    def test_dashboard_summary_is_read_only(self, api_client):
        """Dashboard summary must be GET only."""
        for method in ("post", "put", "patch", "delete"):
            resp = getattr(api_client, method)("/api/v1/paper-breakout/dashboard-summary")
            assert resp.status_code in (404, 405)
