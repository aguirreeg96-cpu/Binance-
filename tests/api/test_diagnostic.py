"""Tests for the diagnostic endpoint."""

from __future__ import annotations

from unittest.mock import patch as _patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db


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


class TestDiagnosticEndpoint:
    def test_returns_200(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/diagnostic")
        assert resp.status_code == 200

    def test_response_schema(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/diagnostic")
        data = resp.json()
        assert "generated_at" in data
        assert "app_version" in data
        assert "python_version" in data
        assert "db_path" in data
        assert "launch_id" in data
        assert "total_evaluations" in data
        assert "total_trades" in data
        assert "total_events" in data
        assert "unread_events" in data
        assert "warning" in data

    def test_no_secrets_in_response(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/diagnostic")
        text = resp.text.lower()
        assert "telegram_bot_token" not in text
        assert "telegram_chat_id" not in text
        assert "api_key" not in text
        assert "api_secret" not in text
        assert "password" not in text

    def test_no_launch_gives_nulls(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/diagnostic")
        data = resp.json()
        assert data["launch_id"] is None
        assert data["launch_status"] is None
        assert data["account_balance"] is None
        assert data["open_position"] is False

    def test_event_counts_accurate(self, api_client, test_session):
        from app.services.alerting import emit_event

        emit_event(
            test_session,
            event_type="BUY_PENDING",
            message="diag_test",
            idempotency_key="diag_ev_1",
        )
        test_session.commit()

        resp = api_client.get("/api/v1/paper-breakout/diagnostic")
        data = resp.json()
        assert data["total_events"] >= 1
        assert data["unread_events"] >= 1

    def test_disk_free_present(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/diagnostic")
        data = resp.json()
        # disk_free_gb may be null in some environments but should not error
        assert "disk_free_gb" in data

    def test_warning_field_present(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/diagnostic")
        data = resp.json()
        assert data["warning"]
        assert "PAPER" in data["warning"] or "no real money" in data["warning"].lower()
