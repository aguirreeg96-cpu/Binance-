"""Heartbeat service — records one row per engine cycle for health monitoring."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.heartbeat import PaperHeartbeat

RESULT_OK = "OK"
RESULT_NO_NEW_CANDLE = "NO_NEW_CANDLE"
RESULT_ERROR = "ERROR"


def record_heartbeat(
    session: Session,
    *,
    cycle_result: str,
    evaluations_created: int = 0,
    last_candle_time: datetime | None = None,
    error_message: str | None = None,
    process_pid: int | None = None,
) -> PaperHeartbeat:
    """Insert a heartbeat row. Caller is responsible for committing the session."""
    hb = PaperHeartbeat(
        timestamp_utc=_now_naive_utc(),
        cycle_result=cycle_result,
        evaluations_created=evaluations_created,
        last_candle_time=last_candle_time,
        error_message=error_message,
        process_pid=process_pid,
    )
    session.add(hb)
    return hb


def get_latest_heartbeat(session: Session) -> PaperHeartbeat | None:
    """Return the most-recent heartbeat row or None."""
    return session.query(PaperHeartbeat).order_by(desc(PaperHeartbeat.timestamp_utc)).first()


def _now_naive_utc() -> datetime:
    from datetime import UTC

    return datetime.now(UTC).replace(tzinfo=None)
