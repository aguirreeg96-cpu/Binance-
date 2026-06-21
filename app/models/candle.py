from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, DateTime, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_candle"),
        Index("ix_candle_symbol_interval_time", "symbol", "interval", "open_time"),
        Index("ix_candle_symbol_interval", "symbol", "interval"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    open_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(30, 10, asdecimal=True), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(30, 10, asdecimal=True), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(30, 10, asdecimal=True), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(30, 10, asdecimal=True), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 10, asdecimal=True), nullable=False)
    close_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    quote_asset_volume: Mapped[Decimal] = mapped_column(
        Numeric(30, 10, asdecimal=True), nullable=False
    )
    trades: Mapped[int] = mapped_column(BigInteger, nullable=False)
    taker_buy_base_volume: Mapped[Decimal] = mapped_column(
        Numeric(30, 10, asdecimal=True), nullable=False
    )
    taker_buy_quote_volume: Mapped[Decimal] = mapped_column(
        Numeric(30, 10, asdecimal=True), nullable=False
    )
    is_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Candle {self.symbol} {self.interval} @{self.open_time}>"
