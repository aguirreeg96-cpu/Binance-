from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.common import TradingMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Trading mode — defaults to paper (never real money)
    trading_mode: TradingMode = TradingMode.PAPER

    # Binance API — loaded from env, never hardcoded
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_base_url: str = "https://testnet.binance.vision"
    binance_ws_url: str = "wss://testnet.binance.vision/ws"

    # Trading parameters
    trading_symbol: str = "BTCUSDT"
    trading_interval: str = "1h"

    # Risk management
    risk_per_trade: Decimal = Decimal("0.005")
    daily_loss_limit: Decimal = Decimal("0.01")
    max_open_positions: int = 1
    max_trades_per_day: int = 3
    pause_after_losses: int = 2
    max_position_size_pct: Decimal = Decimal("0.20")
    trailing_stop_enabled: bool = False

    # Paper account
    paper_initial_balance: Decimal = Decimal("10000.00")

    # Strategy parameters
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_period: int = 14
    rsi_min: Decimal = Decimal("50")
    rsi_max: Decimal = Decimal("65")
    atr_period: int = 14
    atr_sl_multiplier: Decimal = Decimal("1.5")
    rr_ratio: Decimal = Decimal("2.0")
    volume_ma_period: int = 20

    # Backtesting
    backtest_commission: Decimal = Decimal("0.001")
    backtest_slippage: Decimal = Decimal("0.0005")
    backtest_train_split: float = 0.8

    # Demo mode confirmation
    demo_approval_timeout: int = 60

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    database_url: str = "sqlite:///./trading.db"

    @field_validator("trading_mode", mode="before")
    @classmethod
    def reject_live_mode(cls, v: str) -> str:
        if str(v).lower() == "live":
            raise ValueError(
                "LIVE mode is not implemented. "
                "Set TRADING_MODE to: signal, paper, or demo."
            )
        return v

    @field_validator("binance_base_url", "binance_ws_url", mode="before")
    @classmethod
    def reject_production_urls(cls, v: str) -> str:
        production_hosts = ["api.binance.com", "stream.binance.com"]
        for host in production_hosts:
            if host in str(v):
                raise ValueError(
                    f"Production Binance URL detected: {v!r}. "
                    "Only testnet/demo endpoints are permitted."
                )
        return v

    @model_validator(mode="after")
    def validate_demo_requires_credentials(self) -> "Settings":
        if self.trading_mode == TradingMode.DEMO:
            if not self.binance_api_key or not self.binance_api_secret:
                raise ValueError(
                    "DEMO mode requires BINANCE_API_KEY and BINANCE_API_SECRET "
                    "to be set in your .env file."
                )
        return self

    def masked_api_key(self) -> str:
        if not self.binance_api_key:
            return "(not set)"
        key = self.binance_api_key
        return key[:4] + "****" + key[-4:] if len(key) > 8 else "****"

    def masked_api_secret(self) -> str:
        return "****" if self.binance_api_secret else "(not set)"


@lru_cache
def get_settings() -> Settings:
    return Settings()
