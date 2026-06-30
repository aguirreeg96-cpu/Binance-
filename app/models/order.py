from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.types import ExactDecimal
from app.schemas.common import OrderStatus


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_order_symbol_status", "symbol", "status"),
        Index("ix_order_client_order_id", "client_order_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=OrderStatus.PENDING_APPROVAL
    )
    price: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    quantity: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(
        ExactDecimal(), nullable=False, default=Decimal("0")
    )
    avg_fill_price: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(ExactDecimal(), nullable=True)
    commission: Mapped[Decimal] = mapped_column(
        ExactDecimal(), nullable=False, default=Decimal("0")
    )
    # Mode under which the order was created
    trading_mode: Mapped[str] = mapped_column(String(10), nullable=False)
    position_id: Mapped[int | None] = mapped_column(nullable=True)
    # Forward paper-trading launch that created this order, if any
    launch_id: Mapped[int | None] = mapped_column(ForeignKey("forward_launches.id"), nullable=True)
    # Approval tracking for DEMO mode
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rejected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Order {self.side} {self.symbol} qty={self.quantity} status={self.status}>"
