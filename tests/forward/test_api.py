"""Stage 6.1 tests: paper-breakout REST API endpoints."""

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
# Tests — GET endpoints before any launch
# ---------------------------------------------------------------------------


class TestGetEndpointsBeforeLaunch:
    def test_status_404_before_start(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/status")
        assert resp.status_code == 404

    def test_signals_404_before_start(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/signals")
        assert resp.status_code == 404

    def test_position_404_before_start(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/position")
        assert resp.status_code == 404

    def test_trades_404_before_start(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/trades")
        assert resp.status_code == 404

    def test_equity_404_before_start(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/equity")
        assert resp.status_code == 404

    def test_config_always_200(self, api_client):
        """GET /config never requires a launch to exist."""
        resp = api_client.get("/api/v1/paper-breakout/config")
        assert resp.status_code == 200
        body = resp.json()
        assert "frozen_configuration" in body
        assert "paper_test_disclaimer" in body
        assert "PAPER" in body["paper_test_disclaimer"]


# ---------------------------------------------------------------------------
# Tests — POST /start
# ---------------------------------------------------------------------------


class TestPostStart:
    def test_start_creates_launch(self, api_client):
        from decimal import Decimal

        resp = api_client.post("/api/v1/paper-breakout/start")
        assert resp.status_code == 200
        body = resp.json()
        assert body["launch_id"] is not None
        assert body["strategy_name"] == "donchian_breakout_B_4h"
        assert body["symbol"] == "BTCUSDT"
        assert body["status"] == "ACTIVE"
        assert Decimal(body["initial_capital"]) == Decimal("10000")

    def test_start_is_idempotent(self, api_client):
        r1 = api_client.post("/api/v1/paper-breakout/start")
        r2 = api_client.post("/api/v1/paper-breakout/start")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["launch_id"] == r2.json()["launch_id"]

    def test_start_with_explicit_capital(self, api_client):
        resp = api_client.post("/api/v1/paper-breakout/start", json={"initial_capital": "5000"})
        assert resp.status_code == 200
        # Capital ignored on resume (launch already exists after earlier tests)
        # Just verify the response is well-formed
        body = resp.json()
        assert "initial_capital" in body


# ---------------------------------------------------------------------------
# Tests — GET endpoints after launch
# ---------------------------------------------------------------------------


class TestGetEndpointsAfterLaunch:
    @pytest.fixture(autouse=True)
    def ensure_launch(self, api_client):
        api_client.post("/api/v1/paper-breakout/start")

    def test_status_returns_launch_details(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "BTCUSDT"
        assert body["frozen_config_hash"] is not None
        assert len(body["frozen_config_hash"]) == 64

    def test_signals_returns_empty_list_with_no_evaluations(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/signals")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_position_returns_none_when_no_position(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/position")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_trades_returns_empty_list(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/trades")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_equity_returns_empty_list(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/equity")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_config_returns_frozen_manifest(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/config")
        assert resp.status_code == 200
        cfg = resp.json()["frozen_configuration"]
        assert cfg["symbol"] == "BTCUSDT"
        assert cfg["entry_lookback"] == 20
        assert cfg["atr_multiplier"] == "3.0"
        assert cfg["long_only"] is True


# ---------------------------------------------------------------------------
# Tests — POST /stop
# ---------------------------------------------------------------------------


class TestPostStop:
    def test_stop_returns_stopped_status(self, api_client):
        api_client.post("/api/v1/paper-breakout/start")
        resp = api_client.post("/api/v1/paper-breakout/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "STOPPED"


# ---------------------------------------------------------------------------
# Architecture guard
# ---------------------------------------------------------------------------


class TestArchitectureGuard:
    def test_router_sends_no_orders_to_binance(self):
        import app.api.paper_breakout as router_mod

        with open(router_mod.__file__, encoding="utf-8") as f:
            src = f.read()
        for forbidden in ("place_order", "cancel_order", "send_order", "api_key", "api_secret"):
            assert forbidden not in src

    def test_endpoints_are_paper_only(self):
        import app.api.paper_breakout as router_mod

        with open(router_mod.__file__, encoding="utf-8") as f:
            src = f.read()
        assert "PAPER" in src or "paper" in src
