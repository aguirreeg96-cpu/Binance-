"""Tests for the paper events API endpoints."""

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


def _insert_event(session, event_type: str, ikey: str, message: str = "msg") -> None:
    from app.services.alerting import emit_event

    emit_event(session, event_type=event_type, message=message, idempotency_key=ikey)
    session.commit()


class TestListEvents:
    def test_empty_returns_list(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/events")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_returns_inserted_event(self, api_client, test_session):
        _insert_event(test_session, "BUY_PENDING", "list_test_1")
        resp = api_client.get("/api/v1/paper-breakout/events")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        types = [e["event_type"] for e in data]
        assert "BUY_PENDING" in types

    def test_event_schema(self, api_client, test_session):
        _insert_event(test_session, "LONG_OPENED", "schema_test_1", "Long opened")
        resp = api_client.get("/api/v1/paper-breakout/events")
        assert resp.status_code == 200
        ev = resp.json()[0]
        assert "id" in ev
        assert "event_type" in ev
        assert "severity" in ev
        assert "timestamp_utc" in ev
        assert "message" in ev
        assert "telegram_sent" in ev
        assert "browser_acked" in ev
        assert "idempotency_key" in ev

    def test_limit_param(self, api_client, test_session):
        for i in range(5):
            _insert_event(test_session, "WAIT", f"limit_test_{i}")
        resp = api_client.get("/api/v1/paper-breakout/events?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) <= 2

    def test_limit_too_large_rejected(self, api_client):
        resp = api_client.get("/api/v1/paper-breakout/events?limit=9999")
        assert resp.status_code == 422


class TestListUnreadEvents:
    def test_returns_unread_only(self, api_client, test_session):
        _insert_event(test_session, "SELL_PENDING", "unread_test_1")
        resp = api_client.get("/api/v1/paper-breakout/events/unread")
        assert resp.status_code == 200
        data = resp.json()
        assert all(not e["browser_acked"] for e in data)


class TestMarkEventRead:
    def test_marks_read(self, api_client, test_session):
        from app.services.alerting import emit_event

        ev = emit_event(
            test_session,
            event_type="POSITION_EXITED",
            message="exit",
            idempotency_key="mark_read_test_1",
        )
        test_session.commit()
        event_id = ev.id

        resp = api_client.post(f"/api/v1/paper-breakout/events/{event_id}/read")
        assert resp.status_code == 200
        data = resp.json()
        assert data["browser_acked"] is True
        assert data["read_at"] is not None

    def test_404_for_missing_event(self, api_client):
        resp = api_client.post("/api/v1/paper-breakout/events/999999/read")
        assert resp.status_code == 404

    def test_idempotent_read(self, api_client, test_session):
        from app.services.alerting import emit_event

        ev = emit_event(
            test_session,
            event_type="BUY_PENDING",
            message="buy",
            idempotency_key="idempotent_read_1",
        )
        test_session.commit()
        event_id = ev.id

        resp1 = api_client.post(f"/api/v1/paper-breakout/events/{event_id}/read")
        resp2 = api_client.post(f"/api/v1/paper-breakout/events/{event_id}/read")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp2.json()["browser_acked"] is True
