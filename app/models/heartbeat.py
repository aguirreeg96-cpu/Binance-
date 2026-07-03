from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PaperHeartbeat(Base):
    __tablename__ = "paper_heartbeats"
    __table_args__ = (Index("ix_paper_heartbeat_ts", "timestamp_utc"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    # "OK" | "NO_NEW_CANDLE" | "ERROR"
    cycle_result: Mapped[str] = mapped_column(String(20), nullable=False)
    evaluations_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_candle_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    process_pid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<PaperHeartbeat id={self.id} result={self.cycle_result}"
            f" at={self.timestamp_utc.isoformat()}>"
        )
