"""Events API — list, query unread, and mark paper events as read.

GET  /api/v1/paper-breakout/events              — list events (newest first)
GET  /api/v1/paper-breakout/events/unread       — unread events (browser_acked=False)
POST /api/v1/paper-breakout/events/{id}/read    — mark one event as browser-acknowledged
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.paper_event import PaperEvent

router = APIRouter(prefix="/api/v1/paper-breakout", tags=["paper-breakout-events"])

_MAX_LIMIT = 500
_DEFAULT_LIMIT = 100


def _now_naive_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PaperEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    id: int
    event_type: str
    severity: str
    timestamp_utc: str
    message: str
    related_evaluation_id: int | None
    related_trade_id: int | None
    telegram_sent: bool
    browser_acked: bool
    idempotency_key: str
    read_at: str | None

    @classmethod
    def from_event(cls, e: PaperEvent) -> PaperEventResponse:
        return cls(
            id=e.id,
            event_type=e.event_type,
            severity=e.severity,
            timestamp_utc=e.timestamp_utc.isoformat(),
            message=e.message,
            related_evaluation_id=e.related_evaluation_id,
            related_trade_id=e.related_trade_id,
            telegram_sent=e.telegram_sent,
            browser_acked=e.browser_acked,
            idempotency_key=e.idempotency_key,
            read_at=e.read_at.isoformat() if e.read_at is not None else None,
        )


@router.get("/events", response_model=list[PaperEventResponse])
async def list_events(
    limit: Annotated[int, Query(ge=1, le=_MAX_LIMIT)] = _DEFAULT_LIMIT,
    db: Session = Depends(get_db),
) -> list[PaperEventResponse]:
    events = db.query(PaperEvent).order_by(PaperEvent.timestamp_utc.desc()).limit(limit).all()
    return [PaperEventResponse.from_event(e) for e in events]


@router.get("/events/unread", response_model=list[PaperEventResponse])
async def list_unread_events(
    db: Session = Depends(get_db),
) -> list[PaperEventResponse]:
    events = (
        db.query(PaperEvent)
        .filter(PaperEvent.browser_acked.is_(False))
        .order_by(PaperEvent.timestamp_utc.desc())
        .all()
    )
    return [PaperEventResponse.from_event(e) for e in events]


@router.post("/events/{event_id}/read", response_model=PaperEventResponse)
async def mark_event_read(
    event_id: int,
    db: Session = Depends(get_db),
) -> PaperEventResponse:
    event = db.query(PaperEvent).filter(PaperEvent.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found.",
        )
    if not event.browser_acked:
        event.browser_acked = True
        event.read_at = _now_naive_utc()
        db.commit()
        db.refresh(event)
    return PaperEventResponse.from_event(event)
