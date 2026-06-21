from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.schemas.common import SignalType


class Signal(Base):
    __tablename__ = "signals"
    __table_args__ = (Index("ix_signal_symbol_ts", "symbol", "timestamp"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    price: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(10), nullable=False, default=SignalType.WAIT)
    # JSON blob with all indicator values at signal time
    indicators: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Human-readable list of reasons behind the decision
    reasons: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    strategy_version: Mapped[str] = mapped_column(String(50), nullable=False)
    config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Signal {self.signal_type} {self.symbol} @{self.timestamp}>"
