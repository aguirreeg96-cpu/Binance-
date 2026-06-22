"""RiskExitConfig — configuration for the V2 risk-based exit system.

These parameters define exit behaviour only.  Entry rules are identical to V1.
No parameter values are declared optimal or guaranteed to be profitable.

PAPER/TEST only — no real money, no real orders.
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class RiskExitConfig(BaseModel):
    """Configuration for the V2 risk-based exit system.

    Spot long-only. No leverage. No short positions. PAPER/TEST only.
    None of these defaults are declared optimal or profitable.
    They serve as a starting point for comparative analysis only.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = True

    # ---- Stop-loss ----
    atr_stop_multiplier: Decimal = Field(
        default=Decimal("2.0"),
        gt=0,
        description="ATR multiplier for initial stop-loss distance from entry.",
    )

    # ---- Take-profit ----
    use_take_profit: bool = True
    reward_to_risk_ratio: Decimal = Field(
        default=Decimal("2.0"),
        gt=0,
        description="Take-profit distance as a multiple of initial risk (R).",
    )

    # ---- Trailing stop ----
    trailing_stop_enabled: bool = False
    trailing_activation_r: Decimal = Field(
        default=Decimal("1.0"),
        gt=0,
        description="Price must move this many R in favor before trailing stop activates.",
    )
    trailing_distance_atr: Decimal = Field(
        default=Decimal("1.5"),
        gt=0,
        description="Trailing stop distance below highest price seen, measured in ATR units.",
    )

    # ---- Time exit ----
    maximum_holding_candles: int = Field(
        default=192,
        ge=0,
        description="Force close after this many candles. 0 = disabled.",
    )

    # ---- Secondary crossover exit ----
    use_bearish_crossover_exit: bool = True

    # ---- Position sizing ----
    position_allocation_percentage: Decimal = Field(
        default=Decimal("25"),
        gt=0,
        le=100,
        description=(
            "Percentage of available quote_balance deployed per trade. "
            "Remaining capital stays undeployed. No leverage."
        ),
    )
