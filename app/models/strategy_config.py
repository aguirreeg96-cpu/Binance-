from datetime import datetime

from sqlalchemy import Boolean, DateTime, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class StrategyConfig(Base):
    __tablename__ = "strategy_configs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # EMA settings
    ema_fast: Mapped[int] = mapped_column(default=20)
    ema_slow: Mapped[int] = mapped_column(default=50)
    ema_trend: Mapped[int] = mapped_column(default=200)

    # RSI settings
    rsi_period: Mapped[int] = mapped_column(default=14)
    rsi_min: Mapped[str] = mapped_column(Numeric(10, 4), default="50")
    rsi_max: Mapped[str] = mapped_column(Numeric(10, 4), default="65")

    # ATR settings
    atr_period: Mapped[int] = mapped_column(default=14)
    atr_sl_multiplier: Mapped[str] = mapped_column(Numeric(10, 4), default="1.5")

    # Risk/reward
    rr_ratio: Mapped[str] = mapped_column(Numeric(10, 4), default="2.0")

    # Volume filter
    volume_ma_period: Mapped[int] = mapped_column(default=20)

    # Trailing stop
    trailing_stop_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<StrategyConfig version={self.version} active={self.is_active}>"
