from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.types import ExactDecimal
from app.schemas.common import PositionStatus


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (Index("ix_position_symbol_status", "symbol", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default=PositionStatus.OPEN)
    entry_price: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    stop_loss: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    take_profit: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    trailing_stop_enabled: Mapped[bool] = mapped_column(default=False)
    trailing_stop_price: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    exit_price: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    realized_pnl: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    signal_id: Mapped[int | None] = mapped_column(nullable=True)
    # Forward paper-trading launch that opened this position, if any
    launch_id: Mapped[int | None] = mapped_column(ForeignKey("forward_launches.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<Position {self.symbol} {self.status} qty={self.quantity} entry={self.entry_price}>"
        )
