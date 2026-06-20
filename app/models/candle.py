from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_candle"),
        Index("ix_candle_symbol_interval_time", "symbol", "interval", "open_time"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    open_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    open: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    high: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    low: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    close: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    volume: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    close_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    quote_volume: Mapped[str] = mapped_column(Numeric(30, 10), nullable=False)
    trades: Mapped[int] = mapped_column(BigInteger, nullable=False)
    is_closed: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Candle {self.symbol} {self.interval} @{self.open_time}>"
