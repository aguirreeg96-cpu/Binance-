"""BacktestConfig — immutable, validated backtest configuration.

PAPER/TEST only — no real money, no live orders, no derivatives, no borrowing.
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BacktestConfig(BaseModel):
    """Immutable configuration for a single backtest run.

    All financial parameters are Decimal to prevent float contamination.
    Symbol is normalised to uppercase on construction.
    fee_rate and slippage_rate properties expose fractional forms (e.g. 0.001).
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=2, max_length=20)
    interval: str
    start_ms: int = Field(gt=0, description="Inclusive lower bound on open_time (ms)")
    end_ms: int = Field(gt=0, description="Exclusive upper bound on open_time (ms)")
    initial_capital: Decimal = Field(gt=0, description="Initial quote balance")
    fee_percentage: Decimal = Field(
        default=Decimal("0.1"),
        ge=0,
        le=5,
        description="Fee per trade leg in percent (e.g. 0.1 = 0.1%)",
    )
    slippage_percentage: Decimal = Field(
        default=Decimal("0.05"),
        ge=0,
        le=5,
        description="Adverse slippage per trade leg in percent (e.g. 0.05 = 0.05%)",
    )
    force_close_at_end: bool = Field(
        default=True,
        description="Force close any open position at the last candle's close price",
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        return v.upper()

    @field_validator("interval")
    @classmethod
    def _validate_interval(cls, v: str) -> str:
        from app.market_data.interval_utils import VALID_INTERVALS

        if v not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval {v!r}. Valid: {sorted(VALID_INTERVALS)}")
        return v

    @model_validator(mode="after")
    def _validate_time_range(self) -> "BacktestConfig":
        if self.end_ms <= self.start_ms:
            raise ValueError(f"end_ms ({self.end_ms}) must be > start_ms ({self.start_ms})")
        return self

    @property
    def fee_rate(self) -> Decimal:
        """Fee as a decimal fraction (e.g. 0.001 for 0.1%)."""
        return self.fee_percentage / Decimal("100")

    @property
    def slippage_rate(self) -> Decimal:
        """Slippage as a decimal fraction (e.g. 0.0005 for 0.05%)."""
        return self.slippage_percentage / Decimal("100")
