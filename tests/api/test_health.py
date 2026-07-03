"""Tests for the enhanced /health endpoint."""

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


class TestHealthEndpoint:
    def test_returns_200(self, api_client, test_session):
        with _patch("app.api.health.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = lambda s: test_session
            mock_sl.return_value.__exit__ = lambda s, *a: None
            resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_response_schema(self, api_client, test_session):
        with _patch("app.api.health.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = lambda s: test_session
            mock_sl.return_value.__exit__ = lambda s, *a: None
            resp = api_client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "db_status" in data
        assert "launch_status" in data
        assert "heartbeat_status" in data
        assert "data_freshness" in data
        assert "issues" in data
        assert "checked_at" in data
        assert "warning" in data
        assert data["status"] in ("HEALTHY", "DEGRADED", "ERROR")

    def test_degraded_without_launch(self, api_client, test_session):
        with _patch("app.api.health.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = lambda s: test_session
            mock_sl.return_value.__exit__ = lambda s, *a: None
            resp = api_client.get("/health")
        data = resp.json()
        # No launch exists → DEGRADED
        assert data["status"] == "DEGRADED"
        assert data["launch_status"] == "no_launch"

    def test_db_status_ok(self, api_client, test_session):
        with _patch("app.api.health.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = lambda s: test_session
            mock_sl.return_value.__exit__ = lambda s, *a: None
            resp = api_client.get("/health")
        data = resp.json()
        assert data["db_status"] == "ok"

    def test_warning_never_contains_credentials(self, api_client, test_session):
        with _patch("app.api.health.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = lambda s: test_session
            mock_sl.return_value.__exit__ = lambda s, *a: None
            resp = api_client.get("/health")
        data = resp.json()
        text = str(data)
        assert "token" not in text.lower()
        assert "secret" not in text.lower()
        assert "password" not in text.lower()

    def test_active_heartbeat_reflected(self, api_client, test_session):
        from app.services.heartbeat import record_heartbeat

        record_heartbeat(
            test_session,
            cycle_result="OK",
            evaluations_created=1,
        )
        test_session.commit()

        with _patch("app.api.health.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = lambda s: test_session
            mock_sl.return_value.__exit__ = lambda s, *a: None
            resp = api_client.get("/health")
        data = resp.json()
        # Heartbeat exists, so status is active or idle
        assert data["heartbeat_status"] in ("active", "idle")
        assert data["last_heartbeat_result"] == "OK"
