from datetime import datetime

from sqlalchemy import DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (Index("ix_trade_symbol_opened", "symbol", "opened_at"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_price: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    exit_price: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    quantity: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    gross_pnl: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    commission: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    net_pnl: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    # Risk/reward realized
    planned_stop_loss: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    planned_take_profit: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(50), nullable=False)
    trading_mode: Mapped[str] = mapped_column(String(10), nullable=False)
    position_id: Mapped[int | None] = mapped_column(nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Trade {self.symbol} pnl={self.net_pnl} opened={self.opened_at:%Y-%m-%d}>"
