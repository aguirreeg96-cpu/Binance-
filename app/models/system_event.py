from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.schemas.common import EventLevel


class SystemEvent(Base):
    __tablename__ = "system_events"
    __table_args__ = (Index("ix_system_event_level_ts", "level", "timestamp"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True, default=datetime.utcnow
    )
    level: Mapped[str] = mapped_column(String(10), nullable=False, default=EventLevel.INFO)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trading_mode: Mapped[str] = mapped_column(String(10), nullable=False)

    def __repr__(self) -> str:
        return f"<SystemEvent [{self.level}] {self.source}: {self.message[:50]}>"
