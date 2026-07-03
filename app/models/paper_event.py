from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PaperEvent(Base):
    __tablename__ = "paper_events"
    __table_args__ = (Index("ix_paper_event_ts", "timestamp_utc"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # BUY_PENDING | LONG_OPENED | SELL_PENDING | POSITION_EXITED |
    # ERROR_DATA_GAP | INCONSISTENT_STATE | PAPER_PROCESS_STOPPED |
    # MISSED_EXPECTED_EVALUATION
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    # INFO | WARNING | ERROR
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    related_evaluation_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    related_trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    telegram_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    browser_acked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Prevents duplicate event creation (unique per business occurrence)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<PaperEvent id={self.id} type={self.event_type} at={self.timestamp_utc}>"
