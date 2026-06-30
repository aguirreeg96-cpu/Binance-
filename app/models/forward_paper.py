from datetime import datetime
from decimal import Decimal

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.types import ExactDecimal
from app.schemas.common import ForwardLaunchStatus, ForwardSignalState


class ForwardLaunch(Base):
    """A single forward paper-trading activation of a frozen strategy config.

    `launch_timestamp` is written once on first start and never changes —
    it is the permanent boundary between warm-up data and forward signals.
    """

    __tablename__ = "forward_launches"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    frozen_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    frozen_config_json: Mapped[str] = mapped_column(String, nullable=False)
    code_commit_hash: Mapped[str | None] = mapped_column(String(40), nullable=True)
    launch_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    initial_capital: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default=ForwardLaunchStatus.ACTIVE
    )
    last_evaluated_candle_close: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ForwardLaunch {self.strategy_name} {self.symbol} "
            f"launched={self.launch_timestamp} status={self.status}>"
        )


class ForwardSignalEvaluation(Base):
    """One per-closed-candle evaluation cycle of the forward engine.

    Unique on (launch_id, candle_close_time): exactly one evaluation per
    closed candle per launch, guaranteeing idempotency across restarts.
    """

    __tablename__ = "forward_signal_evaluations"
    __table_args__ = (
        UniqueConstraint("launch_id", "candle_close_time", name="uq_forward_eval_launch_candle"),
        Index("ix_forward_eval_launch_candle", "launch_id", "candle_close_time"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    launch_id: Mapped[int] = mapped_column(ForeignKey("forward_launches.id"), nullable=False)
    candle_close_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    signal: Mapped[str] = mapped_column(String(20), nullable=False, default=ForwardSignalState.WAIT)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    raw_market_price: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    planned_execution_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    entry_donchian_level: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    exit_donchian_level: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    ema_200: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    ema_slope: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    atr: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    initial_stop: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    current_trailing_stop: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    highest_high_since_entry: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    position_quantity: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    cash: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    equity: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    frozen_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<ForwardSignalEvaluation launch={self.launch_id} "
            f"candle={self.candle_close_time} signal={self.signal}>"
        )
