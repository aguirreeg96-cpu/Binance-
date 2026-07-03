"""Tests for the alerting service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.alerting import (
    emit_event,
    maybe_send_telegram,
    send_telegram,
)


class TestEmitEvent:
    def test_inserts_event(self, db_session):
        event = emit_event(
            db_session,
            event_type="BUY_PENDING",
            message="Test buy",
            idempotency_key="test_key_1",
        )
        db_session.flush()
        assert event is not None
        assert event.id is not None
        assert event.event_type == "BUY_PENDING"
        assert event.severity == "INFO"
        assert event.message == "Test buy"
        assert event.telegram_sent is False
        assert event.browser_acked is False

    def test_idempotency_returns_none_on_duplicate(self, db_session):
        emit_event(
            db_session,
            event_type="BUY_PENDING",
            message="First",
            idempotency_key="dup_key",
        )
        db_session.commit()

        result = emit_event(
            db_session,
            event_type="BUY_PENDING",
            message="Second",
            idempotency_key="dup_key",
        )
        assert result is None

    def test_severity_mapping(self, db_session):
        cases = [
            ("BUY_PENDING", "INFO"),
            ("ERROR_DATA_GAP", "WARNING"),
            ("INCONSISTENT_STATE", "ERROR"),
            ("PAPER_PROCESS_STOPPED", "WARNING"),
        ]
        for i, (et, expected_sev) in enumerate(cases):
            ev = emit_event(
                db_session,
                event_type=et,
                message="m",
                idempotency_key=f"sev_{i}",
            )
            db_session.commit()
            assert ev is not None
            assert ev.severity == expected_sev, f"{et} → expected {expected_sev}"

    def test_optional_fields(self, db_session):
        ev = emit_event(
            db_session,
            event_type="LONG_OPENED",
            message="Long",
            idempotency_key="opt_key",
            related_evaluation_id=7,
            related_trade_id=3,
        )
        db_session.flush()
        assert ev is not None
        assert ev.related_evaluation_id == 7
        assert ev.related_trade_id == 3

    def test_timestamp_naive_utc(self, db_session):
        ev = emit_event(
            db_session,
            event_type="SELL_PENDING",
            message="sell",
            idempotency_key="ts_key",
        )
        db_session.flush()
        assert ev is not None
        assert ev.timestamp_utc.tzinfo is None


class TestSendTelegram:
    @pytest.mark.asyncio
    async def test_returns_false_without_httpx(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("no httpx")
            return real_import(name, *args, **kwargs)

        event = MagicMock()
        with patch("builtins.__import__", side_effect=mock_import):
            result = await send_telegram(event, bot_token="tok", chat_id="cid")
        assert result is False

    @pytest.mark.asyncio
    async def test_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        event = MagicMock()
        event.id = 1
        event.event_type = "BUY_PENDING"
        event.timestamp_utc.isoformat.return_value = "2025-01-01T00:00:00"
        event.message = "msg"

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await send_telegram(event, bot_token="tok", chat_id="cid")
        assert result is True

    @pytest.mark.asyncio
    async def test_http_error_returns_false(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        event = MagicMock()
        event.id = 2
        event.event_type = "BUY_PENDING"
        event.timestamp_utc.isoformat.return_value = "2025-01-01T00:00:00"
        event.message = "msg"

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await send_telegram(event, bot_token="tok", chat_id="cid")
        assert result is False


class TestMaybeSendTelegram:
    @pytest.mark.asyncio
    async def test_skips_when_no_credentials(self, db_session):
        event = MagicMock()
        event.telegram_sent = False

        with patch("app.config.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = None
            mock_settings.return_value.telegram_chat_id = None
            await maybe_send_telegram(db_session, event)

        assert event.telegram_sent is False

    @pytest.mark.asyncio
    async def test_skips_when_already_sent(self, db_session):
        event = MagicMock()
        event.telegram_sent = True

        with patch("app.config.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = "tok"
            mock_settings.return_value.telegram_chat_id = "cid"
            await maybe_send_telegram(db_session, event)

        # Should not have called send_telegram — telegram_sent stays True
        assert event.telegram_sent is True

    @pytest.mark.asyncio
    async def test_sets_sent_on_success(self, db_session):
        event = MagicMock()
        event.telegram_sent = False

        with (
            patch("app.config.get_settings") as mock_settings,
            patch("app.services.alerting.send_telegram", new_callable=AsyncMock) as mock_send,
        ):
            mock_settings.return_value.telegram_bot_token = "tok"
            mock_settings.return_value.telegram_chat_id = "cid"
            mock_send.return_value = True
            await maybe_send_telegram(db_session, event)

        assert event.telegram_sent is True
