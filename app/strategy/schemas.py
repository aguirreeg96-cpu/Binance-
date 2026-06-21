"""Data types for strategy inputs and outputs.

No ORM models, no HTTP, no DB sessions.
All Decimal fields stay Decimal — serialisation happens in the API layer.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from app.indicators.schemas import CrossSignal
from app.strategy.reasons import ReasonCode


class StrategyAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


@dataclass
class PositionContext:
    """Snapshot of the current position state passed into the strategy engine.

    The engine is pure — it never queries the Position table itself.
    The caller is responsible for providing accurate state.

    Default: no open position.
    """

    has_open_long_position: bool = False
    entry_price: Decimal | None = None
    entry_time: int | None = None


@dataclass
class IndicatorsSnapshot:
    """Immutable copy of the indicator values used to reach the decision."""

    close: Decimal
    ema_short: Decimal | None
    ema_medium: Decimal | None
    ema_long: Decimal | None
    rsi: Decimal | None
    atr: Decimal | None
    volume_sma: Decimal | None
    volume_ratio: Decimal | None
    crossover: CrossSignal


@dataclass
class StrategyDecision:
    """The result of evaluating one IndicatorResult against the strategy rules.

    Spot-only: SELL means closing a long position, never opening a short.
    No orders are placed; this is a pure signal.
    """

    action: StrategyAction
    symbol: str
    interval: str
    candle_open_time: int
    candle_close_time: int
    close_price: Decimal
    strategy_name: str
    strategy_version: str
    reasons: list[ReasonCode]
    failed_conditions: list[ReasonCode]
    indicators_snapshot: IndicatorsSnapshot
    warmup_complete: bool
    has_open_position: bool
    generated_at: datetime
