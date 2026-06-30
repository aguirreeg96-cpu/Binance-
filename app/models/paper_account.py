from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.types import ExactDecimal


class PaperAccount(Base):
    __tablename__ = "paper_accounts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Quote asset balance (e.g. USDT)
    balance: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    # Base asset balance (e.g. BTC) — held in open positions
    asset_balance: Mapped[Decimal] = mapped_column(
        ExactDecimal(), nullable=False, default=Decimal("0")
    )
    equity: Mapped[Decimal] = mapped_column(ExactDecimal(), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(
        ExactDecimal(), nullable=False, default=Decimal("0")
    )
    total_fees_paid: Mapped[Decimal] = mapped_column(
        ExactDecimal(), nullable=False, default=Decimal("0")
    )
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USDT")
    # Forward paper-trading launch that owns this account, if any
    launch_id: Mapped[int | None] = mapped_column(ForeignKey("forward_launches.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<PaperAccount balance={self.balance} equity={self.equity}>"
