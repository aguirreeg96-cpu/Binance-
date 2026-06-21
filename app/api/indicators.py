"""
Indicators REST API.

GET /api/v1/indicators        — compute indicators for a symbol/interval range
GET /api/v1/indicators/latest — latest indicator values for a symbol/interval

Data is read exclusively from SQLite.  No Binance calls are made.
No trading signals (BUY/SELL/WAIT) are generated or returned.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.database import get_db
from app.indicators.calculator import IndicatorCalculator
from app.indicators.exceptions import IndicatorError
from app.indicators.schemas import IndicatorConfig, IndicatorResult
from app.market_data.interval_utils import VALID_INTERVALS
from app.repositories.candle_repository import CandleRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/indicators", tags=["indicators"])

_MAX_LIMIT = 5000
_MIN_LIMIT = 1


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class IndicatorResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    symbol: str
    interval: str
    open_time: int
    open_time_iso: str
    close_time: int
    close_time_iso: str
    close: str
    sma_short: str | None
    sma_long: str | None
    ema_short: str | None
    ema_medium: str | None
    ema_long: str | None
    rsi: str | None
    atr: str | None
    volume_sma: str | None
    volume_ratio: str | None
    ema_short_medium_cross: str
    warmup_complete: bool
    source_candle_id: int | None
    calculated_at: str | None

    @classmethod
    def from_result(cls, r: IndicatorResult) -> "IndicatorResultResponse":
        return cls(
            symbol=r.symbol,
            interval=r.interval,
            open_time=r.open_time,
            open_time_iso=_ms_to_iso(r.open_time),
            close_time=r.close_time,
            close_time_iso=_ms_to_iso(r.close_time),
            close=str(r.close),
            sma_short=str(r.sma_short) if r.sma_short is not None else None,
            sma_long=str(r.sma_long) if r.sma_long is not None else None,
            ema_short=str(r.ema_short) if r.ema_short is not None else None,
            ema_medium=str(r.ema_medium) if r.ema_medium is not None else None,
            ema_long=str(r.ema_long) if r.ema_long is not None else None,
            rsi=str(r.rsi) if r.rsi is not None else None,
            atr=str(r.atr) if r.atr is not None else None,
            volume_sma=str(r.volume_sma) if r.volume_sma is not None else None,
            volume_ratio=str(r.volume_ratio) if r.volume_ratio is not None else None,
            ema_short_medium_cross=r.ema_short_medium_cross.value,
            warmup_complete=r.warmup_complete,
            source_candle_id=r.source_candle_id,
            calculated_at=r.calculated_at.isoformat() if r.calculated_at else None,
        )


class LatestIndicatorResponse(IndicatorResultResponse):
    candles_used: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()


def _parse_dt(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Datetime must include timezone (e.g. 2025-01-01T00:00:00Z).",
        )
    return int(value.astimezone(UTC).timestamp() * 1000)


def _build_config(
    sma_short_period: int,
    sma_long_period: int,
    ema_short_period: int,
    ema_medium_period: int,
    ema_long_period: int,
    rsi_period: int,
    atr_period: int,
    volume_period: int,
) -> IndicatorConfig:
    try:
        return IndicatorConfig(
            sma_short_period=sma_short_period,
            sma_long_period=sma_long_period,
            ema_short_period=ema_short_period,
            ema_medium_period=ema_medium_period,
            ema_long_period=ema_long_period,
            rsi_period=rsi_period,
            atr_period=atr_period,
            volume_period=volume_period,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid indicator configuration: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[IndicatorResultResponse])
async def get_indicators(
    symbol: Annotated[str, Query(min_length=2, max_length=20)],
    interval: Annotated[str, Query()],
    start: Annotated[datetime | None, Query(description="ISO 8601 UTC, inclusive")] = None,
    end: Annotated[datetime | None, Query(description="ISO 8601 UTC, exclusive")] = None,
    limit: Annotated[int, Query(ge=_MIN_LIMIT, le=_MAX_LIMIT)] = 500,
    include_warmup: bool = False,
    sma_short_period: int = 20,
    sma_long_period: int = 50,
    ema_short_period: int = 20,
    ema_medium_period: int = 50,
    ema_long_period: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_period: int = 20,
    db: Session = Depends(get_db),
) -> list[IndicatorResultResponse]:
    symbol = symbol.upper()

    if interval not in VALID_INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid interval {interval!r}. Valid: {sorted(VALID_INTERVALS)}",
        )

    start_ms = _parse_dt(start)
    end_ms = _parse_dt(end)

    config = _build_config(
        sma_short_period,
        sma_long_period,
        ema_short_period,
        ema_medium_period,
        ema_long_period,
        rsi_period,
        atr_period,
        volume_period,
    )

    repo = CandleRepository(db)
    candles, warmup_len = repo.query_for_indicators(
        symbol=symbol,
        interval=interval,
        warmup_count=config.warmup_candles,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
    )

    if not candles:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No candles found for {symbol} {interval}.",
        )

    try:
        calc = IndicatorCalculator(config)
        all_results = calc.calculate(candles)
    except IndicatorError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Return only the requested range (skip warmup prefix)
    results = all_results[warmup_len:]

    if not include_warmup:
        results = [r for r in results if r.warmup_complete]

    if not results:
        warmup_needed = config.warmup_candles
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No complete indicators available: {symbol} {interval} needs at least "
                f"{warmup_needed} candles for warm-up "
                f"(longest period: ema_long={ema_long_period}). "
                "Use include_warmup=true to see partial results, or download more history."
            ),
        )

    return [IndicatorResultResponse.from_result(r) for r in results]


@router.get("/latest", response_model=LatestIndicatorResponse)
async def get_latest_indicators(
    symbol: Annotated[str, Query(min_length=2, max_length=20)],
    interval: Annotated[str, Query()],
    sma_short_period: int = 20,
    sma_long_period: int = 50,
    ema_short_period: int = 20,
    ema_medium_period: int = 50,
    ema_long_period: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_period: int = 20,
    db: Session = Depends(get_db),
) -> LatestIndicatorResponse:
    symbol = symbol.upper()

    if interval not in VALID_INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid interval {interval!r}. Valid: {sorted(VALID_INTERVALS)}",
        )

    config = _build_config(
        sma_short_period,
        sma_long_period,
        ema_short_period,
        ema_medium_period,
        ema_long_period,
        rsi_period,
        atr_period,
        volume_period,
    )

    repo = CandleRepository(db)
    # For latest: use ALL available candles for maximum accuracy
    candles, _ = repo.query_for_indicators(
        symbol=symbol,
        interval=interval,
        warmup_count=0,
    )

    if not candles:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No candles found for {symbol} {interval}.",
        )

    try:
        calc = IndicatorCalculator(config)
        all_results = calc.calculate(candles)
    except IndicatorError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    last = all_results[-1]
    base = IndicatorResultResponse.from_result(last)
    return LatestIndicatorResponse(
        **base.model_dump(),
        candles_used=len(candles),
    )
