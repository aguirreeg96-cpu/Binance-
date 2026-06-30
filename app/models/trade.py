from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.types import ExactDecimal


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (Index("ix_trade_symbol_opened", "symbol", "opened_at"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    exit_price: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    gross_pnl: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    commission: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    net_pnl: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    # Risk/reward realized
    planned_stop_loss: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    planned_take_profit: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(50), nullable=False)
    trading_mode: Mapped[str] = mapped_column(String(10), nullable=False)
    position_id: Mapped[int | None] = mapped_column(nullable=True)
    # Forward paper-trading launch that produced this trade, if any
    launch_id: Mapped[int | None] = mapped_column(ForeignKey("forward_launches.id"), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Trade {self.symbol} pnl={self.net_pnl} opened={self.opened_at:%Y-%m-%d}>"
