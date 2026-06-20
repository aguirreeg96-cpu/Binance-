from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.schemas.common import PositionStatus


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (
        Index("ix_position_symbol_status", "symbol", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default=PositionStatus.OPEN
    )
    entry_price: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    quantity: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    stop_loss: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    take_profit: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    trailing_stop_enabled: Mapped[bool] = mapped_column(default=False)
    trailing_stop_price: Mapped[Optional[str]] = mapped_column(
        Numeric(30, 10), nullable=True
    )
    exit_price: Mapped[Optional[str]] = mapped_column(
        Numeric(30, 10), nullable=True
    )
    realized_pnl: Mapped[Optional[str]] = mapped_column(
        Numeric(30, 10), nullable=True
    )
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    signal_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Position {self.symbol} {self.status} "
            f"qty={self.quantity} entry={self.entry_price}>"
        )
