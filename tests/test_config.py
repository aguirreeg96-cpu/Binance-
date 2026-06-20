"""Tests for app configuration and safety guards."""
import os

import pytest
from pydantic import ValidationError

from app.schemas.common import TradingMode


def _make_settings(**overrides):
    """Create a Settings instance with environment isolation.

    pydantic-settings v2 resolves init kwargs using field names (lowercase),
    not environment variable names. case_sensitive=False only applies to
    reading from the environment/dotenv file.
    """
    from app.config import Settings

    # Normalize keys: both UPPER_CASE and lower_case callers are supported
    normalized = {k.lower(): v for k, v in overrides.items()}

    base = {
        "trading_mode": "paper",
        "binance_api_key": "",
        "binance_api_secret": "",
        "binance_base_url": "https://testnet.binance.vision",
        "binance_ws_url": "wss://testnet.binance.vision/ws",
    }
    base.update(normalized)
    return Settings(**base)


class TestTradingModeDefaults:
    def test_default_mode_is_paper(self):
        s = _make_settings()
        assert s.trading_mode == TradingMode.PAPER

    def test_signal_mode_accepted(self):
        s = _make_settings(TRADING_MODE="signal")
        assert s.trading_mode == TradingMode.SIGNAL

    def test_demo_mode_accepted_with_credentials(self):
        s = _make_settings(
            TRADING_MODE="demo",
            BINANCE_API_KEY="testkey1234",
            BINANCE_API_SECRET="testsecret5678",
        )
        assert s.trading_mode == TradingMode.DEMO

    def test_live_mode_is_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(TRADING_MODE="live")
        assert "not implemented" in str(exc_info.value).lower()

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValidationError):
            _make_settings(TRADING_MODE="real")


class TestDemoModeCredentialGuard:
    def test_demo_without_api_key_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(
                TRADING_MODE="demo",
                BINANCE_API_KEY="",
                BINANCE_API_SECRET="secret",
            )
        assert "DEMO mode requires" in str(exc_info.value)

    def test_demo_without_api_secret_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(
                TRADING_MODE="demo",
                BINANCE_API_KEY="mykey",
                BINANCE_API_SECRET="",
            )
        assert "DEMO mode requires" in str(exc_info.value)

    def test_paper_mode_does_not_require_credentials(self):
        s = _make_settings(TRADING_MODE="paper", BINANCE_API_KEY="", BINANCE_API_SECRET="")
        assert s.trading_mode == TradingMode.PAPER


class TestProductionUrlGuard:
    def test_production_base_url_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(BINANCE_BASE_URL="https://api.binance.com")
        assert "production" in str(exc_info.value).lower()

    def test_production_ws_url_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(BINANCE_WS_URL="wss://stream.binance.com/ws")
        assert "production" in str(exc_info.value).lower()

    def test_testnet_url_accepted(self):
        s = _make_settings(BINANCE_BASE_URL="https://testnet.binance.vision")
        assert "testnet" in s.binance_base_url


class TestApiKeyMasking:
    def test_long_key_is_masked(self):
        s = _make_settings(BINANCE_API_KEY="ABCD1234EFGH5678")
        masked = s.masked_api_key()
        assert "ABCD" in masked
        assert "5678" in masked
        assert "1234EFGH" not in masked
        assert "****" in masked

    def test_empty_key_shows_not_set(self):
        s = _make_settings(BINANCE_API_KEY="")
        assert s.masked_api_key() == "(not set)"

    def test_secret_always_masked(self):
        s = _make_settings(BINANCE_API_SECRET="supersecretvalue")
        assert s.masked_api_secret() == "****"

    def test_empty_secret_shows_not_set(self):
        s = _make_settings(BINANCE_API_SECRET="")
        assert s.masked_api_secret() == "(not set)"


class TestRiskDefaults:
    def test_risk_per_trade_default(self):
        from decimal import Decimal

        s = _make_settings()
        assert s.risk_per_trade == Decimal("0.005")

    def test_daily_loss_limit_default(self):
        from decimal import Decimal

        s = _make_settings()
        assert s.daily_loss_limit == Decimal("0.01")

    def test_max_open_positions_default(self):
        s = _make_settings()
        assert s.max_open_positions == 1

    def test_max_trades_per_day_default(self):
        s = _make_settings()
        assert s.max_trades_per_day == 3

    def test_paper_initial_balance_default(self):
        from decimal import Decimal

        s = _make_settings()
        assert s.paper_initial_balance == Decimal("10000.00")
