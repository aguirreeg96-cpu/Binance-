"""Alerting service — emits PaperEvent rows and optionally sends Telegram messages.

Design principles:
- Telegram credentials are NEVER logged, printed, or returned in any API response.
- Duplicate events are suppressed via idempotency_key (UNIQUE constraint).
- Telegram is fully optional: if the token/chat_id are absent the application
  runs normally without any alerting.
- httpx is used for the Telegram HTTP call; ImportError is caught so the
  dependency stays optional.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.forward.manifest import FORWARD_SYMBOL, STRATEGY_NAME
from app.models.paper_event import PaperEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_TYPES: frozenset[str] = frozenset(
    {
        "BUY_PENDING",
        "LONG_OPENED",
        "SELL_PENDING",
        "POSITION_EXITED",
        "ERROR_DATA_GAP",
        "INCONSISTENT_STATE",
        "PAPER_PROCESS_STOPPED",
        "MISSED_EXPECTED_EVALUATION",
    }
)

_SEVERITY: dict[str, str] = {
    "BUY_PENDING": "INFO",
    "LONG_OPENED": "INFO",
    "SELL_PENDING": "INFO",
    "POSITION_EXITED": "INFO",
    "ERROR_DATA_GAP": "WARNING",
    "INCONSISTENT_STATE": "ERROR",
    "PAPER_PROCESS_STOPPED": "WARNING",
    "MISSED_EXPECTED_EVALUATION": "WARNING",
}

# ---------------------------------------------------------------------------
# Emit event
# ---------------------------------------------------------------------------


def emit_event(
    session: Session,
    *,
    event_type: str,
    message: str,
    idempotency_key: str,
    related_evaluation_id: int | None = None,
    related_trade_id: int | None = None,
) -> PaperEvent | None:
    """Insert a PaperEvent row; return None if the idempotency_key already exists."""
    severity = _SEVERITY.get(event_type, "INFO")
    event = PaperEvent(
        event_type=event_type,
        severity=severity,
        timestamp_utc=_now_naive_utc(),
        message=message,
        related_evaluation_id=related_evaluation_id,
        related_trade_id=related_trade_id,
        telegram_sent=False,
        browser_acked=False,
        idempotency_key=idempotency_key,
        read_at=None,
    )
    try:
        session.add(event)
        session.flush()
    except IntegrityError:
        session.rollback()
        return None
    return event


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def _build_telegram_message(event: PaperEvent) -> str:
    lines = [
        "🤖 PAPER TRADING — NO REAL MONEY",
        f"Strategy: {STRATEGY_NAME}",
        f"Symbol: {FORWARD_SYMBOL}",
        f"Event: {event.event_type}",
        f"Time (UTC): {event.timestamp_utc.isoformat()}",
        event.message,
    ]
    return "\n".join(lines)


async def send_telegram(
    event: PaperEvent,
    *,
    bot_token: str,
    chat_id: str,
) -> bool:
    """POST a notification to Telegram. Returns True on success, False on any failure.

    bot_token and chat_id are NEVER logged.
    """
    try:
        import httpx  # optional dependency
    except ImportError:
        logger.debug("httpx not installed — Telegram notification skipped.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": _build_telegram_message(event)}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            return True
        logger.warning("Telegram returned HTTP %s (event_id=%s).", resp.status_code, event.id)
        return False
    except Exception:  # noqa: BLE001
        logger.warning("Telegram notification failed (event_id=%s).", event.id)
        return False


async def maybe_send_telegram(session: Session, event: PaperEvent) -> None:
    """Send Telegram if configured and not already sent. Updates event.telegram_sent."""
    from app.config import get_settings

    settings = get_settings()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return
    if event.telegram_sent:
        return

    success = await send_telegram(
        event,
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )
    if success:
        event.telegram_sent = True
        session.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_naive_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
