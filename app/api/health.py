"""Enhanced health-check endpoint.

GET /health returns HEALTHY, DEGRADED, or ERROR based on:
  - Database accessibility
  - ForwardLaunch existence and status
  - Heartbeat age (active<600s, idle<14400s, stale>=14400s)
  - Data freshness (fresh<6h, stale>=6h)
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.database import SessionLocal
from app.forward.manifest import FORWARD_SYMBOL, STRATEGY_NAME
from app.repositories.forward_repository import ForwardRepository
from app.services.heartbeat import get_latest_heartbeat

router = APIRouter(tags=["system"])

_HEALTHY = "HEALTHY"
_DEGRADED = "DEGRADED"
_ERROR = "ERROR"
_RANK = {_HEALTHY: 0, _DEGRADED: 1, _ERROR: 2}


def _worst(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


def _now_naive_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class HealthResponse(BaseModel):
    status: str
    db_status: str
    launch_status: str
    heartbeat_status: str
    data_freshness: str
    heartbeat_age_seconds: float | None
    last_candle_age_hours: float | None
    last_heartbeat_result: str | None
    issues: list[str]
    checked_at: str
    warning: str


def _build_response(
    status: str,
    db_status: str,
    launch_status: str,
    heartbeat_status: str,
    data_freshness: str,
    hb_age: float | None,
    candle_age_hours: float | None,
    last_hb_result: str | None,
    issues: list[str],
    now: datetime,
) -> HealthResponse:
    return HealthResponse(
        status=status,
        db_status=db_status,
        launch_status=launch_status,
        heartbeat_status=heartbeat_status,
        data_freshness=data_freshness,
        heartbeat_age_seconds=round(hb_age, 1) if hb_age is not None else None,
        last_candle_age_hours=round(candle_age_hours, 2) if candle_age_hours is not None else None,
        last_heartbeat_result=last_hb_result,
        issues=issues,
        checked_at=now.isoformat(),
        warning="PAPER/TEST environment — no real money. Not investment advice.",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    now = _now_naive_utc()
    overall = _HEALTHY
    issues: list[str] = []

    db_status = "ok"
    launch_status_str = "no_launch"
    heartbeat_status_str = "no_heartbeat"
    data_freshness_str = "no_data"
    hb_age: float | None = None
    candle_age_hours: float | None = None
    last_hb_result: str | None = None

    try:
        with SessionLocal() as session:
            # 1. DB check
            try:
                session.execute(text("SELECT 1"))
            except Exception as exc:  # noqa: BLE001
                db_status = "error"
                overall = _ERROR
                issues.append(f"Database inaccessible: {exc!s}")
                return _build_response(
                    overall,
                    db_status,
                    launch_status_str,
                    heartbeat_status_str,
                    data_freshness_str,
                    hb_age,
                    candle_age_hours,
                    last_hb_result,
                    issues,
                    now,
                )

            # 2. Launch
            repo = ForwardRepository(session)
            launch = repo.get_launch(STRATEGY_NAME, FORWARD_SYMBOL)
            if launch is None:
                launch_status_str = "no_launch"
                issues.append("No forward paper-trading launch exists yet.")
                overall = _worst(overall, _DEGRADED)
            else:
                launch_status_str = launch.status
                if launch.status == "STOPPED":
                    issues.append("Launch is STOPPED.")
                    overall = _worst(overall, _DEGRADED)
                elif launch.status not in ("ACTIVE",):
                    issues.append(f"Launch status: {launch.status}")
                    overall = _worst(overall, _DEGRADED)

            # 3. Heartbeat
            hb = get_latest_heartbeat(session)
            if hb is None:
                heartbeat_status_str = "no_heartbeat"
                if launch is not None:
                    issues.append("No heartbeat recorded — paper trader may not be running.")
                    overall = _worst(overall, _DEGRADED)
            else:
                hb_age = (now - hb.timestamp_utc).total_seconds()
                last_hb_result = hb.cycle_result

                if hb_age < 600:
                    heartbeat_status_str = "active"
                elif hb_age < 14400:
                    heartbeat_status_str = "idle"
                else:
                    heartbeat_status_str = "stale"
                    issues.append(f"Heartbeat stale: {hb_age / 3600:.1f}h since last cycle.")
                    overall = _worst(overall, _DEGRADED)

                if hb.cycle_result == "ERROR":
                    issues.append(f"Last cycle error: {hb.error_message or '(no message)'}")
                    overall = _worst(overall, _ERROR if hb_age < 600 else _DEGRADED)

                if hb.last_candle_time is not None:
                    candle_age_hours = (now - hb.last_candle_time).total_seconds() / 3600
                    if candle_age_hours < 6:
                        data_freshness_str = "fresh"
                    else:
                        data_freshness_str = "stale"
                        issues.append(f"Last candle data {candle_age_hours:.1f}h old.")
                        overall = _worst(overall, _DEGRADED)

            # Fallback: use launch.last_evaluated_candle_close if heartbeat had no candle time
            if (
                data_freshness_str == "no_data"
                and launch is not None
                and launch.last_evaluated_candle_close is not None
            ):
                candle_age_hours = (now - launch.last_evaluated_candle_close).total_seconds() / 3600
                if candle_age_hours < 6:
                    data_freshness_str = "fresh"
                else:
                    data_freshness_str = "stale"
                    issues.append(f"Last candle data {candle_age_hours:.1f}h old.")
                    overall = _worst(overall, _DEGRADED)

    except Exception as exc:  # noqa: BLE001
        overall = _ERROR
        db_status = "error"
        issues.append(f"Health check failed: {exc!s}")

    return _build_response(
        overall,
        db_status,
        launch_status_str,
        heartbeat_status_str,
        data_freshness_str,
        hb_age,
        candle_age_hours,
        last_hb_result,
        issues,
        now,
    )
