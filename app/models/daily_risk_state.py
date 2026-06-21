from datetime import date, datetime

from sqlalchemy import Date, DateTime, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class DailyRiskState(Base):
    __tablename__ = "daily_risk_states"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)

    trades_count: Mapped[int] = mapped_column(default=0)
    consecutive_losses: Mapped[int] = mapped_column(default=0)
    daily_pnl: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False, default="0")
    daily_pnl_pct: Mapped[str] = mapped_column(Numeric(10, 6), nullable=False, default="0")
    starting_equity: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)

    # Kill switch state
    kill_switch_active: Mapped[bool] = mapped_column(default=False)
    kill_switch_reason: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    # Pause state (after N consecutive losses)
    trading_paused: Mapped[bool] = mapped_column(default=False)
    pause_reason: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<DailyRiskState {self.date} trades={self.trades_count} pnl={self.daily_pnl_pct}%>"
