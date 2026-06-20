from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.schemas.common import OrderSide, OrderStatus, OrderType


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_order_symbol_status", "symbol", "status"),
        Index("ix_order_client_order_id", "client_order_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=OrderStatus.PENDING_APPROVAL
    )
    price: Mapped[Optional[str]] = mapped_column(Numeric(30, 10), nullable=True)
    quantity: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    filled_quantity: Mapped[str] = mapped_column(
        Numeric(30, 10), nullable=False, default="0"
    )
    avg_fill_price: Mapped[Optional[str]] = mapped_column(
        Numeric(30, 10), nullable=True
    )
    stop_price: Mapped[Optional[str]] = mapped_column(Numeric(30, 10), nullable=True)
    commission: Mapped[str] = mapped_column(
        Numeric(30, 10), nullable=False, default="0"
    )
    # Mode under which the order was created
    trading_mode: Mapped[str] = mapped_column(String(10), nullable=False)
    position_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    # Approval tracking for DEMO mode
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    rejected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Order {self.side} {self.symbol} qty={self.quantity} "
            f"status={self.status}>"
        )
