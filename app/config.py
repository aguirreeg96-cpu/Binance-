from decimal import Decimal
from functools import lru_cache
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.common import TradingMode

# ---------------------------------------------------------------------------
# URL allowlists (checked by hostname, not string search)
# ---------------------------------------------------------------------------
_ALLOWED_MARKET_DATA_HOSTS = frozenset({"data-api.binance.vision"})
_ALLOWED_TRADING_HOSTS = frozenset({"demo-api.binance.com"})
_ALLOWED_MARKET_WS_HOSTS = frozenset({"data-stream.binance.vision"})
_REJECTED_ANYWHERE = frozenset({
    "api.binance.com",
    "ws-api.binance.com",
    "stream.binance.com",
})


def _parse_and_guard(
    url: str,
    field: str,
    allowed_hosts: frozenset[str],
    allowed_schemes: tuple[str, ...],
) -> str:
    parsed = urlparse(url)

    if parsed.username or parsed.password:
        raise ValueError(
            f"{field}: URL must not contain embedded credentials: {url!r}"
        )

    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            f"{field}: scheme must be one of {allowed_schemes}, "
            f"got {parsed.scheme!r} in {url!r}"
        )

    hostname = (parsed.hostname or "").lower()

    if hostname in _REJECTED_ANYWHERE:
        raise ValueError(
            f"{field}: host {hostname!r} is a production Binance endpoint "
            "and is not permitted in this application."
        )

    if hostname not in allowed_hosts:
        raise ValueError(
            f"{field}: host {hostname!r} is not in the allowlist "
            f"{set(allowed_hosts)}. "
            "Check for typos — subdomain attacks like "
            "'data-api.binance.vision.evil.com' are rejected."
        )

    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Trading mode — defaults to paper (never real money)
    trading_mode: TradingMode = TradingMode.PAPER

    # Binance API credentials — only required for DEMO mode
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # Market data (public REST — no auth required)
    binance_market_data_url: str = "https://data-api.binance.vision"
    # Order execution (DEMO only)
    binance_trading_url: str = "https://demo-api.binance.com"
    # WebSocket market data (future Stage 7)
    binance_market_ws_url: str = "wss://data-stream.binance.vision"

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

    # Market data HTTP client settings
    market_data_timeout: float = 30.0
    market_data_max_retries: int = 3
    market_data_max_retry_after: int = 60  # seconds cap on Retry-After
    market_data_max_requests: int = 500    # max pages per download job
    market_data_max_range_days: int = 365  # API validation guard

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    database_url: str = "sqlite:///./trading.db"

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @field_validator("trading_mode", mode="before")
    @classmethod
    def reject_live_mode(cls, v: str) -> str:
        if str(v).lower() == "live":
            raise ValueError(
                "LIVE mode is not implemented. "
                "Set TRADING_MODE to: signal, paper, or demo."
            )
        return v

    @field_validator("binance_market_data_url", mode="before")
    @classmethod
    def validate_market_data_url(cls, v: str) -> str:
        return _parse_and_guard(
            v, "BINANCE_MARKET_DATA_URL", _ALLOWED_MARKET_DATA_HOSTS, ("https",)
        )

    @field_validator("binance_trading_url", mode="before")
    @classmethod
    def validate_trading_url(cls, v: str) -> str:
        return _parse_and_guard(
            v, "BINANCE_TRADING_URL", _ALLOWED_TRADING_HOSTS, ("https",)
        )

    @field_validator("binance_market_ws_url", mode="before")
    @classmethod
    def validate_market_ws_url(cls, v: str) -> str:
        return _parse_and_guard(
            v, "BINANCE_MARKET_WS_URL", _ALLOWED_MARKET_WS_HOSTS, ("wss",)
        )

    @model_validator(mode="after")
    def validate_demo_requires_credentials(self) -> "Settings":
        if self.trading_mode == TradingMode.DEMO:
            if not self.binance_api_key or not self.binance_api_secret:
                raise ValueError(
                    "DEMO mode requires BINANCE_API_KEY and BINANCE_API_SECRET "
                    "to be set in your .env file."
                )
        return self

    # ---------------------------------------------------------------------------
    # Helpers — never expose raw secrets
    # ---------------------------------------------------------------------------

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
