"""Schemas for indicator configuration and results."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CrossSignal(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE = "none"


class IndicatorConfig(BaseModel):
    """Configuration for indicator periods.  All periods must be > 0.

    Period ordering constraints:
      sma_short_period < sma_long_period
      ema_short_period < ema_medium_period < ema_long_period
    """

    model_config = ConfigDict(frozen=True)

    sma_short_period: int = Field(default=20, gt=0)
    sma_long_period: int = Field(default=50, gt=0)
    ema_short_period: int = Field(default=20, gt=0)
    ema_medium_period: int = Field(default=50, gt=0)
    ema_long_period: int = Field(default=200, gt=0)
    rsi_period: int = Field(default=14, gt=0)
    atr_period: int = Field(default=14, gt=0)
    volume_period: int = Field(default=20, gt=0)
    max_candles: int = Field(default=5000, gt=0)

    @model_validator(mode="after")
    def validate_ordering(self) -> "IndicatorConfig":
        if self.sma_short_period >= self.sma_long_period:
            raise ValueError(
                f"sma_short_period ({self.sma_short_period}) must be "
                f"< sma_long_period ({self.sma_long_period})"
            )
        if not (self.ema_short_period < self.ema_medium_period < self.ema_long_period):
            raise ValueError(
                "EMA periods must be strictly ascending: "
                f"{self.ema_short_period} < {self.ema_medium_period} < {self.ema_long_period}"
            )
        return self

    @property
    def warmup_candles(self) -> int:
        """Minimum candles needed before any result can have warmup_complete=True."""
        return max(
            self.sma_long_period,
            self.ema_long_period,
            self.rsi_period + 1,  # RSI needs (period+1) closes to produce first value
            self.atr_period,
            self.volume_period,
        )


@dataclass
class IndicatorResult:
    """Indicators computed for a single candle.

    Fields are None when insufficient history exists (warm-up phase).
    warmup_complete is True only when ALL indicators have values.
    """

    symbol: str
    interval: str
    open_time: int
    close_time: int
    close: Decimal

    sma_short: Decimal | None
    sma_long: Decimal | None
    ema_short: Decimal | None
    ema_medium: Decimal | None
    ema_long: Decimal | None
    rsi: Decimal | None
    atr: Decimal | None
    volume_sma: Decimal | None
    volume_ratio: Decimal | None
    ema_short_medium_cross: CrossSignal

    warmup_complete: bool

    source_candle_id: int | None = None
    calculated_at: datetime | None = None
