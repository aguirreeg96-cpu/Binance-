"""Backtesting REST API.

POST /api/v1/backtests/run — run a deterministic backtest on stored candles.

Reads candles from the local SQLite DB only — no Binance API calls.
No real orders are placed.  PAPER/TEST environment only.
Decimal values are serialized as strings in all responses.

SECURITY: This endpoint never places orders, never accesses API keys,
and never connects to Binance or any external service.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from app.backtesting.config import BacktestConfig
from app.backtesting.exceptions import BacktestError, BacktestInsufficientDataError
from app.backtesting.exporters import result_to_dict
from app.backtesting.service import BacktestService
from app.database import get_db
from app.indicators.schemas import IndicatorConfig
from app.market_data.interval_utils import VALID_INTERVALS
from app.strategy.config import StrategyEngineConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/backtests", tags=["backtesting"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class RunBacktestRequest(BaseModel):
    """Request body for POST /api/v1/backtests/run.

    start and end are ISO 8601 UTC datetimes; they are converted to
    millisecond timestamps internally.  All financial values are Decimal.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=2, max_length=20)
    interval: str
    start: datetime = Field(description="Inclusive start datetime (ISO 8601 UTC)")
    end: datetime = Field(description="Exclusive end datetime (ISO 8601 UTC)")
    initial_capital: Decimal = Field(default=Decimal("10000"), gt=0)
    fee_percentage: Decimal = Field(default=Decimal("0.1"), ge=0, le=5)
    slippage_percentage: Decimal = Field(default=Decimal("0.05"), ge=0, le=5)
    force_close_at_end: bool = True

    # Indicator config
    sma_short_period: int = Field(default=20, gt=0)
    sma_long_period: int = Field(default=50, gt=0)
    ema_short_period: int = Field(default=20, gt=0)
    ema_medium_period: int = Field(default=50, gt=0)
    ema_long_period: int = Field(default=200, gt=0)
    rsi_period: int = Field(default=14, gt=0)
    atr_period: int = Field(default=14, gt=0)
    volume_period: int = Field(default=20, gt=0)

    # Strategy config
    buy_rsi_min: Decimal = Field(default=Decimal("50"))
    buy_rsi_max: Decimal = Field(default=Decimal("65"))
    sell_rsi_overbought: Decimal = Field(default=Decimal("75"))
    minimum_volume_ratio: Decimal = Field(default=Decimal("1"))
    require_bullish_crossover: bool = True
    require_bearish_crossover_for_sell: bool = True
    require_price_above_long_ema: bool = True

    @model_validator(mode="after")
    def _validate_datetimes(self) -> "RunBacktestRequest":
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("start and end must include timezone info (e.g. 2024-01-01T00:00:00Z)")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval {self.interval!r}. Valid: {sorted(VALID_INTERVALS)}"
            )
        return self

    def to_start_ms(self) -> int:
        return int(self.start.astimezone(UTC).timestamp() * 1000)

    def to_end_ms(self) -> int:
        return int(self.end.astimezone(UTC).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=None)
async def run_backtest(
    request: RunBacktestRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run a deterministic backtest on locally stored candles.

    Reads candles from SQLite — no external API calls.
    No real orders placed.  PAPER/TEST only.
    Returns a JSON object with config, summary, trades, and equity_curve.
    All Decimal values are strings.
    """
    try:
        ind_config = IndicatorConfig(
            sma_short_period=request.sma_short_period,
            sma_long_period=request.sma_long_period,
            ema_short_period=request.ema_short_period,
            ema_medium_period=request.ema_medium_period,
            ema_long_period=request.ema_long_period,
            rsi_period=request.rsi_period,
            atr_period=request.atr_period,
            volume_period=request.volume_period,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid indicator configuration: {exc}",
        ) from exc

    try:
        strat_config = StrategyEngineConfig(
            buy_rsi_min=request.buy_rsi_min,
            buy_rsi_max=request.buy_rsi_max,
            sell_rsi_overbought=request.sell_rsi_overbought,
            minimum_volume_ratio=request.minimum_volume_ratio,
            require_bullish_crossover=request.require_bullish_crossover,
            require_bearish_crossover_for_sell=request.require_bearish_crossover_for_sell,
            require_price_above_long_ema=request.require_price_above_long_ema,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid strategy configuration: {exc}",
        ) from exc

    try:
        config = BacktestConfig(
            symbol=request.symbol,
            interval=request.interval,
            start_ms=request.to_start_ms(),
            end_ms=request.to_end_ms(),
            initial_capital=request.initial_capital,
            fee_percentage=request.fee_percentage,
            slippage_percentage=request.slippage_percentage,
            force_close_at_end=request.force_close_at_end,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid backtest configuration: {exc}",
        ) from exc

    svc = BacktestService(db)
    try:
        result = svc.run(config, ind_config, strat_config)
    except BacktestInsufficientDataError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except BacktestError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return result_to_dict(result)
