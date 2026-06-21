from datetime import datetime

from sqlalchemy import DateTime, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PaperAccount(Base):
    __tablename__ = "paper_accounts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Quote asset balance (e.g. USDT)
    balance: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    # Base asset balance (e.g. BTC) — held in open positions
    asset_balance: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False, default="0")
    equity: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    realized_pnl: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False, default="0")
    total_fees_paid: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False, default="0")
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USDT")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<PaperAccount balance={self.balance} equity={self.equity}>"
