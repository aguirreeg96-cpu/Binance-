"""
Strategy REST API.

GET /api/v1/strategy/latest  — latest decision for a symbol/interval
GET /api/v1/strategy/history — sequential decisions over a date range

No Binance calls are made.
No orders are placed.
SELL signals represent exiting a Spot long position, never opening a short.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.database import get_db
from app.indicators.exceptions import IndicatorError
from app.indicators.schemas import IndicatorConfig
from app.market_data.interval_utils import VALID_INTERVALS
from app.strategy.config import StrategyEngineConfig
from app.strategy.exceptions import InsufficientDataForStrategyError, StrategyError
from app.strategy.schemas import IndicatorsSnapshot, PositionContext, StrategyDecision
from app.strategy.service import StrategyService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/strategy", tags=["strategy"])

_MAX_LIMIT = 5000
_MIN_LIMIT = 1


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class IndicatorsSnapshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    close: str
    ema_short: str | None
    ema_medium: str | None
    ema_long: str | None
    rsi: str | None
    atr: str | None
    volume_sma: str | None
    volume_ratio: str | None
    crossover: str

    @classmethod
    def from_snapshot(cls, s: IndicatorsSnapshot) -> "IndicatorsSnapshotResponse":
        return cls(
            close=str(s.close),
            ema_short=str(s.ema_short) if s.ema_short is not None else None,
            ema_medium=str(s.ema_medium) if s.ema_medium is not None else None,
            ema_long=str(s.ema_long) if s.ema_long is not None else None,
            rsi=str(s.rsi) if s.rsi is not None else None,
            atr=str(s.atr) if s.atr is not None else None,
            volume_sma=str(s.volume_sma) if s.volume_sma is not None else None,
            volume_ratio=str(s.volume_ratio) if s.volume_ratio is not None else None,
            crossover=s.crossover.value,
        )


class StrategyDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    action: str
    symbol: str
    interval: str
    candle_open_time: int
    candle_open_time_iso: str
    candle_close_time: int
    candle_close_time_iso: str
    close_price: str
    strategy_name: str
    strategy_version: str
    reasons: list[str]
    failed_conditions: list[str]
    indicators_snapshot: IndicatorsSnapshotResponse
    warmup_complete: bool
    has_open_position: bool
    generated_at: str

    @classmethod
    def from_decision(cls, d: StrategyDecision) -> "StrategyDecisionResponse":
        return cls(
            action=d.action.value,
            symbol=d.symbol,
            interval=d.interval,
            candle_open_time=d.candle_open_time,
            candle_open_time_iso=_ms_to_iso(d.candle_open_time),
            candle_close_time=d.candle_close_time,
            candle_close_time_iso=_ms_to_iso(d.candle_close_time),
            close_price=str(d.close_price),
            strategy_name=d.strategy_name,
            strategy_version=d.strategy_version,
            reasons=[r.value for r in d.reasons],
            failed_conditions=[r.value for r in d.failed_conditions],
            indicators_snapshot=IndicatorsSnapshotResponse.from_snapshot(d.indicators_snapshot),
            warmup_complete=d.warmup_complete,
            has_open_position=d.has_open_position,
            generated_at=d.generated_at.isoformat(),
        )


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


def _build_ind_config(
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


def _build_strat_config(
    buy_rsi_min: Decimal,
    buy_rsi_max: Decimal,
    sell_rsi_overbought: Decimal,
    minimum_volume_ratio: Decimal,
    require_bullish_crossover: bool,
    require_bearish_crossover_for_sell: bool,
    require_price_above_long_ema: bool,
) -> StrategyEngineConfig:
    try:
        return StrategyEngineConfig(
            buy_rsi_min=buy_rsi_min,
            buy_rsi_max=buy_rsi_max,
            sell_rsi_overbought=sell_rsi_overbought,
            minimum_volume_ratio=minimum_volume_ratio,
            require_bullish_crossover=require_bullish_crossover,
            require_bearish_crossover_for_sell=require_bearish_crossover_for_sell,
            require_price_above_long_ema=require_price_above_long_ema,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid strategy configuration: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/latest", response_model=StrategyDecisionResponse)
async def get_latest_strategy(
    symbol: Annotated[str, Query(min_length=2, max_length=20)],
    interval: Annotated[str, Query()],
    has_open_position: bool = False,
    # Indicator config
    sma_short_period: int = 20,
    sma_long_period: int = 50,
    ema_short_period: int = 20,
    ema_medium_period: int = 50,
    ema_long_period: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_period: int = 20,
    # Strategy config
    buy_rsi_min: Decimal = Decimal("50"),
    buy_rsi_max: Decimal = Decimal("65"),
    sell_rsi_overbought: Decimal = Decimal("75"),
    minimum_volume_ratio: Decimal = Decimal("1"),
    require_bullish_crossover: bool = True,
    require_bearish_crossover_for_sell: bool = True,
    require_price_above_long_ema: bool = True,
    db: Session = Depends(get_db),
) -> StrategyDecisionResponse:
    symbol = symbol.upper()
    if interval not in VALID_INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid interval {interval!r}. Valid: {sorted(VALID_INTERVALS)}",
        )

    ind_config = _build_ind_config(
        sma_short_period,
        sma_long_period,
        ema_short_period,
        ema_medium_period,
        ema_long_period,
        rsi_period,
        atr_period,
        volume_period,
    )
    strat_config = _build_strat_config(
        buy_rsi_min,
        buy_rsi_max,
        sell_rsi_overbought,
        minimum_volume_ratio,
        require_bullish_crossover,
        require_bearish_crossover_for_sell,
        require_price_above_long_ema,
    )
    position = PositionContext(has_open_long_position=has_open_position)

    svc = StrategyService(db)
    try:
        decision = svc.get_latest_decision(symbol, interval, position, ind_config, strat_config)
    except InsufficientDataForStrategyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (IndicatorError, StrategyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return StrategyDecisionResponse.from_decision(decision)


@router.get("/history", response_model=list[StrategyDecisionResponse])
async def get_strategy_history(
    symbol: Annotated[str, Query(min_length=2, max_length=20)],
    interval: Annotated[str, Query()],
    start: Annotated[datetime | None, Query(description="ISO 8601 UTC, inclusive")] = None,
    end: Annotated[datetime | None, Query(description="ISO 8601 UTC, exclusive")] = None,
    limit: Annotated[int, Query(ge=_MIN_LIMIT, le=_MAX_LIMIT)] = 500,
    include_wait: bool = False,
    initial_position_open: bool = False,
    # Indicator config
    sma_short_period: int = 20,
    sma_long_period: int = 50,
    ema_short_period: int = 20,
    ema_medium_period: int = 50,
    ema_long_period: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_period: int = 20,
    # Strategy config
    buy_rsi_min: Decimal = Decimal("50"),
    buy_rsi_max: Decimal = Decimal("65"),
    sell_rsi_overbought: Decimal = Decimal("75"),
    minimum_volume_ratio: Decimal = Decimal("1"),
    require_bullish_crossover: bool = True,
    require_bearish_crossover_for_sell: bool = True,
    require_price_above_long_ema: bool = True,
    db: Session = Depends(get_db),
) -> list[StrategyDecisionResponse]:
    symbol = symbol.upper()
    if interval not in VALID_INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid interval {interval!r}. Valid: {sorted(VALID_INTERVALS)}",
        )

    start_ms = _parse_dt(start)
    end_ms = _parse_dt(end)

    ind_config = _build_ind_config(
        sma_short_period,
        sma_long_period,
        ema_short_period,
        ema_medium_period,
        ema_long_period,
        rsi_period,
        atr_period,
        volume_period,
    )
    strat_config = _build_strat_config(
        buy_rsi_min,
        buy_rsi_max,
        sell_rsi_overbought,
        minimum_volume_ratio,
        require_bullish_crossover,
        require_bearish_crossover_for_sell,
        require_price_above_long_ema,
    )
    initial_position = PositionContext(has_open_long_position=initial_position_open)

    svc = StrategyService(db)
    try:
        decisions = svc.get_history(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
            initial_position=initial_position,
            include_wait=include_wait,
            ind_config=ind_config,
            strat_config=strat_config,
        )
    except InsufficientDataForStrategyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (IndicatorError, StrategyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return [StrategyDecisionResponse.from_decision(d) for d in decisions]
