"""
Market data REST API.

GET  /api/v1/market-data/klines   — query stored candles
POST /api/v1/market-data/download — download from Binance and persist
"""

import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer, field_validator
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.market_data.client import MarketDataClient
from app.market_data.exceptions import (
    BannedError,
    InvalidIntervalError,
    InvalidSymbolError,
    MarketDataError,
    MaxRequestsError,
    PaginationStallError,
    RateLimitError,
)
from app.market_data.historical_service import HistoricalDataService
from app.market_data.interval_utils import VALID_INTERVALS
from app.repositories.candle_repository import CandleRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/market-data", tags=["market-data"])

_MAX_LIMIT = 5000
_MIN_LIMIT = 1


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_market_data_client(request: Request) -> MarketDataClient:
    return request.app.state.market_data_client


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class KlineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    interval: str
    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: int
    quote_asset_volume: Decimal
    trades: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    is_closed: bool

    @computed_field  # type: ignore[misc]
    @property
    def open_time_iso(self) -> str:
        return _ms_to_iso(self.open_time)

    @computed_field  # type: ignore[misc]
    @property
    def close_time_iso(self) -> str:
        return _ms_to_iso(self.close_time)

    @field_serializer(
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )
    def serialize_decimal(self, v: Decimal) -> str:
        return str(v)


class DownloadRequest(BaseModel):
    symbol: str = Field(..., min_length=2, max_length=20)
    interval: str
    start: datetime = Field(..., description="ISO 8601 UTC datetime (inclusive)")
    end: datetime = Field(..., description="ISO 8601 UTC datetime (exclusive)")
    include_open_candle: bool = False

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("interval", mode="before")
    @classmethod
    def validate_interval_field(cls, v: str) -> str:
        if v not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval {v!r}. Valid values: {sorted(VALID_INTERVALS)}")
        return v

    @field_validator("start", "end", mode="after")
    @classmethod
    def require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "Datetime must include timezone info (e.g. '2025-01-01T00:00:00Z'). "
                "Naive datetimes are rejected."
            )
        return v

    @field_validator("end", mode="after")
    @classmethod
    def end_after_start(cls, v: datetime, info: object) -> datetime:
        # Pydantic v2: use model_validator for cross-field; this is a pre-check
        return v


class DownloadResponse(BaseModel):
    request_id: str
    symbol: str
    interval: str
    requested_start: str
    requested_end: str
    requests_made: int
    received: int
    inserted: int
    updated: int
    ignored: int
    first_open_time: str | None
    last_open_time: str | None
    duration_ms: int
    warning: str = "PAPER/TEST environment — no real money"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/klines", response_model=list[KlineResponse])
async def get_klines(
    symbol: Annotated[str, Query(min_length=2, max_length=20)],
    interval: Annotated[str, Query()],
    start: Annotated[datetime | None, Query(description="ISO 8601 UTC, inclusive")] = None,
    end: Annotated[datetime | None, Query(description="ISO 8601 UTC, exclusive")] = None,
    source: Annotated[Literal["database", "binance"], Query()] = "database",
    include_open_candle: bool = False,
    limit: Annotated[int, Query(ge=_MIN_LIMIT, le=_MAX_LIMIT)] = 500,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: MarketDataClient = Depends(get_market_data_client),
) -> list[KlineResponse]:
    symbol = symbol.upper()

    if interval not in VALID_INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid interval {interval!r}. Valid: {sorted(VALID_INTERVALS)}",
        )

    # Validate timezone on query params
    for name, val in [("start", start), ("end", end)]:
        if val is not None and val.tzinfo is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must include timezone (e.g. 2025-01-01T00:00:00Z)",
            )

    start_ms: int | None = None
    end_ms: int | None = None
    if start:
        start_ms = int(start.astimezone(UTC).timestamp() * 1000)
    if end:
        end_ms = int(end.astimezone(UTC).timestamp() * 1000)

    server_time_ms: int | None = None
    if not include_open_candle:
        server_time_ms = await client.get_server_time()

    repo = CandleRepository(db)
    candles = repo.query(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
        include_open_candle=include_open_candle,
        server_time_ms=server_time_ms,
    )

    return [KlineResponse.model_validate(c) for c in candles]


@router.post(
    "/download",
    response_model=DownloadResponse,
    status_code=status.HTTP_200_OK,
)
async def download_klines(
    body: DownloadRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    client: MarketDataClient = Depends(get_market_data_client),
) -> DownloadResponse:
    request_id = str(uuid.uuid4())
    logger.info(
        "[%s] Download request: %s %s %s → %s",
        request_id,
        body.symbol,
        body.interval,
        body.start.isoformat(),
        body.end.isoformat(),
    )

    # Cross-field range validation
    if body.start >= body.end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start must be before end",
        )

    # Guard excessive range
    delta_days = (body.end - body.start).days
    if delta_days > settings.market_data_max_range_days:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Requested range {delta_days} days exceeds limit "
                f"{settings.market_data_max_range_days} days."
            ),
        )

    service = HistoricalDataService(
        client=client,
        max_requests=settings.market_data_max_requests,
    )

    try:
        result = await service.download(
            session=db,
            symbol=body.symbol,
            interval=body.interval,
            start=body.start,
            end=body.end,
            include_open_candle=body.include_open_candle,
        )
        db.commit()
    except (InvalidSymbolError, InvalidIntervalError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RateLimitError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limited by Binance. Retry after {exc.retry_after}s.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except BannedError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporarily unavailable. Please try again later.",
        ) from exc
    except (MaxRequestsError, PaginationStallError, MarketDataError) as exc:
        db.rollback()
        logger.error("[%s] Download failed: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data source temporarily unavailable.",
        ) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("[%s] Unexpected error during download", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from exc

    return DownloadResponse(
        request_id=request_id,
        symbol=result.symbol,
        interval=result.interval,
        requested_start=result.requested_start.isoformat(),
        requested_end=result.requested_end.isoformat(),
        requests_made=result.requests_made,
        received=result.received,
        inserted=result.inserted,
        updated=result.updated,
        ignored=result.ignored,
        first_open_time=result.first_open_time.isoformat() if result.first_open_time else None,
        last_open_time=result.last_open_time.isoformat() if result.last_open_time else None,
        duration_ms=result.duration_ms,
    )


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()
