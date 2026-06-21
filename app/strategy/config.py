"""StrategyEngineConfig — frozen, validated configuration for the strategy engine."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


class StrategyEngineConfig(BaseModel):
    """All parameters that govern buy/sell/wait decision logic.

    Frozen after construction — never mutate during a session.
    """

    model_config = ConfigDict(frozen=True)

    # Global guards
    require_warmup_complete: bool = True
    require_closed_candle: bool = True

    # BUY conditions
    require_price_above_long_ema: bool = True
    buy_rsi_min: Decimal = Field(default=Decimal("50"))
    buy_rsi_max: Decimal = Field(default=Decimal("65"))
    minimum_volume_ratio: Decimal = Field(default=Decimal("1"))
    require_bullish_crossover: bool = True

    # SELL conditions
    sell_rsi_overbought: Decimal = Field(default=Decimal("75"))
    require_bearish_crossover_for_sell: bool = True
    allow_sell_without_open_position: bool = False

    # Identity
    strategy_name: str = "ema-rsi-volume-v1"
    strategy_version: str = "1.0.0"

    @model_validator(mode="after")
    def _validate(self) -> "StrategyEngineConfig":
        if self.buy_rsi_min >= self.buy_rsi_max:
            raise ValueError(
                f"buy_rsi_min ({self.buy_rsi_min}) must be " f"< buy_rsi_max ({self.buy_rsi_max})"
            )
        for name, val in [
            ("buy_rsi_min", self.buy_rsi_min),
            ("buy_rsi_max", self.buy_rsi_max),
            ("sell_rsi_overbought", self.sell_rsi_overbought),
        ]:
            if not (_ZERO <= val <= _HUNDRED):
                raise ValueError(f"{name} must be in [0, 100], got {val}")
        if self.minimum_volume_ratio < _ZERO:
            raise ValueError(f"minimum_volume_ratio must be >= 0, got {self.minimum_volume_ratio}")
        if not self.strategy_name.strip():
            raise ValueError("strategy_name must not be empty")
        if not self.strategy_version.strip():
            raise ValueError("strategy_version must not be empty")
        return self
