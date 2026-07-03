"""Diagnostic endpoint — returns system state snapshot with no secrets.

GET /api/v1/paper-breakout/diagnostic
"""

from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.forward.manifest import FORWARD_SYMBOL, STRATEGY_NAME
from app.models.paper_event import PaperEvent
from app.repositories.forward_repository import ForwardRepository
from app.services.heartbeat import get_latest_heartbeat

router = APIRouter(prefix="/api/v1/paper-breakout", tags=["paper-breakout-diagnostic"])

_APP_VERSION = "0.2.0"


def _now_naive_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DiagnosticResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    generated_at: str
    app_version: str
    python_version: str
    db_path: str
    db_size_bytes: int | None
    launch_id: int | None
    launch_status: str | None
    launch_timestamp: str | None
    strategy_name: str | None
    strategy_version: str | None
    last_evaluated_candle_close: str | None
    account_balance: str | None
    account_equity: str | None
    open_position: bool
    total_evaluations: int
    total_trades: int
    total_events: int
    unread_events: int
    latest_heartbeat_at: str | None
    latest_heartbeat_result: str | None
    latest_heartbeat_age_seconds: float | None
    disk_free_gb: float | None
    warning: str


@router.get("/diagnostic", response_model=DiagnosticResponse)
async def get_diagnostic(db: Session = Depends(get_db)) -> DiagnosticResponse:
    settings = get_settings()
    now = _now_naive_utc()

    db_path = settings.database_url.replace("sqlite:///", "")
    db_size: int | None = None
    try:
        db_size = Path(db_path).stat().st_size
    except Exception:  # noqa: BLE001
        pass

    disk_free_gb: float | None = None
    try:
        usage = shutil.disk_usage(".")
        disk_free_gb = round(usage.free / (1024**3), 2)
    except Exception:  # noqa: BLE001
        pass

    repo = ForwardRepository(db)
    launch = repo.get_launch(STRATEGY_NAME, FORWARD_SYMBOL)
    account = repo.get_account(launch) if launch else None
    position = repo.get_open_position(launch) if launch else None
    evaluations = repo.list_evaluations(launch) if launch else []
    trades = repo.list_trades(launch) if launch else []

    hb = get_latest_heartbeat(db)
    hb_age: float | None = None
    if hb is not None:
        hb_age = round((now - hb.timestamp_utc).total_seconds(), 1)

    total_events = db.query(PaperEvent).count()
    unread_events = db.query(PaperEvent).filter(PaperEvent.browser_acked.is_(False)).count()

    return DiagnosticResponse(
        generated_at=now.isoformat(),
        app_version=_APP_VERSION,
        python_version=sys.version.split()[0],
        db_path=db_path,
        db_size_bytes=db_size,
        launch_id=launch.id if launch else None,
        launch_status=launch.status if launch else None,
        launch_timestamp=launch.launch_timestamp.isoformat() if launch else None,
        strategy_name=launch.strategy_name if launch else None,
        strategy_version=launch.strategy_version if launch else None,
        last_evaluated_candle_close=(
            launch.last_evaluated_candle_close.isoformat()
            if launch and launch.last_evaluated_candle_close
            else None
        ),
        account_balance=str(account.balance) if account else None,
        account_equity=str(account.equity) if account else None,
        open_position=position is not None,
        total_evaluations=len(evaluations),
        total_trades=len(trades),
        total_events=total_events,
        unread_events=unread_events,
        latest_heartbeat_at=hb.timestamp_utc.isoformat() if hb else None,
        latest_heartbeat_result=hb.cycle_result if hb else None,
        latest_heartbeat_age_seconds=hb_age,
        disk_free_gb=disk_free_gb,
        warning=(
            "PAPER/TEST environment — no real money. Not investment advice. "
            "No secrets included in this response."
        ),
    )
