"""Tests for app configuration and safety guards."""

import pytest
from pydantic import ValidationError

from app.schemas.common import TradingMode


def _make_settings(**overrides):
    """
    Create a Settings instance with environment isolation.

    pydantic-settings v2 uses field names (lowercase) for init kwargs.
    Keys are normalized to lowercase so callers can use either case.
    """
    from app.config import Settings

    normalized = {k.lower(): v for k, v in overrides.items()}
    base = {
        "trading_mode": "paper",
        "binance_api_key": "",
        "binance_api_secret": "",
        "binance_market_data_url": "https://data-api.binance.vision",
        "binance_trading_url": "https://demo-api.binance.com",
        "binance_market_ws_url": "wss://data-stream.binance.vision",
    }
    base.update(normalized)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Trading mode
# ---------------------------------------------------------------------------

class TestTradingModeDefaults:
    def test_default_mode_is_paper(self):
        s = _make_settings()
        assert s.trading_mode == TradingMode.PAPER

    def test_signal_mode_accepted(self):
        s = _make_settings(trading_mode="signal")
        assert s.trading_mode == TradingMode.SIGNAL

    def test_demo_mode_accepted_with_credentials(self):
        s = _make_settings(
            trading_mode="demo",
            binance_api_key="testkey1234",
            binance_api_secret="testsecret5678",
        )
        assert s.trading_mode == TradingMode.DEMO

    def test_live_mode_is_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(trading_mode="live")
        assert "not implemented" in str(exc_info.value).lower()

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValidationError):
            _make_settings(trading_mode="real")


# ---------------------------------------------------------------------------
# DEMO mode credential guard
# ---------------------------------------------------------------------------

class TestDemoModeCredentialGuard:
    def test_demo_without_api_key_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(
                trading_mode="demo",
                binance_api_key="",
                binance_api_secret="secret",
            )
        assert "DEMO mode requires" in str(exc_info.value)

    def test_demo_without_api_secret_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(
                trading_mode="demo",
                binance_api_key="mykey",
                binance_api_secret="",
            )
        assert "DEMO mode requires" in str(exc_info.value)

    def test_paper_mode_does_not_require_credentials(self):
        s = _make_settings(
            trading_mode="paper",
            binance_api_key="",
            binance_api_secret="",
        )
        assert s.trading_mode == TradingMode.PAPER


# ---------------------------------------------------------------------------
# URL allowlist — market data URL
# ---------------------------------------------------------------------------

class TestMarketDataUrlGuard:
    def test_default_market_data_url_accepted(self):
        s = _make_settings()
        assert s.binance_market_data_url == "https://data-api.binance.vision"

    def test_http_market_data_url_rejected(self):
        with pytest.raises(ValidationError, match="scheme"):
            _make_settings(binance_market_data_url="http://data-api.binance.vision")

    def test_production_api_url_rejected_as_market_data(self):
        with pytest.raises(ValidationError):
            _make_settings(binance_market_data_url="https://api.binance.com")

    def test_malicious_subdomain_rejected(self):
        """data-api.binance.vision.evil.com must be rejected (hostname check, not substring)."""
        with pytest.raises(ValidationError, match="allowlist"):
            _make_settings(
                binance_market_data_url="https://data-api.binance.vision.evil.com"
            )

    def test_embedded_credentials_rejected(self):
        with pytest.raises(ValidationError, match="credentials"):
            _make_settings(
                binance_market_data_url="https://user:pass@data-api.binance.vision"
            )

    def test_unknown_host_rejected(self):
        with pytest.raises(ValidationError, match="allowlist"):
            _make_settings(binance_market_data_url="https://other-api.binance.vision")


# ---------------------------------------------------------------------------
# URL allowlist — trading URL
# ---------------------------------------------------------------------------

class TestTradingUrlGuard:
    def test_default_trading_url_accepted(self):
        s = _make_settings()
        assert s.binance_trading_url == "https://demo-api.binance.com"

    def test_production_trading_url_rejected(self):
        with pytest.raises(ValidationError, match="production"):
            _make_settings(binance_trading_url="https://api.binance.com")

    def test_http_trading_url_rejected(self):
        with pytest.raises(ValidationError, match="scheme"):
            _make_settings(binance_trading_url="http://demo-api.binance.com")

    def test_embedded_credentials_in_trading_url_rejected(self):
        with pytest.raises(ValidationError, match="credentials"):
            _make_settings(
                binance_trading_url="https://key:secret@demo-api.binance.com"
            )

    def test_stream_binance_rejected_as_trading(self):
        with pytest.raises(ValidationError):
            _make_settings(binance_trading_url="https://stream.binance.com")


# ---------------------------------------------------------------------------
# URL allowlist — WebSocket URL
# ---------------------------------------------------------------------------

class TestMarketWsUrlGuard:
    def test_default_ws_url_accepted(self):
        s = _make_settings()
        assert s.binance_market_ws_url == "wss://data-stream.binance.vision"

    def test_ws_scheme_required(self):
        with pytest.raises(ValidationError, match="scheme"):
            _make_settings(binance_market_ws_url="https://data-stream.binance.vision")

    def test_malicious_ws_host_rejected(self):
        with pytest.raises(ValidationError, match="allowlist"):
            _make_settings(
                binance_market_ws_url="wss://data-stream.binance.vision.evil.com"
            )

    def test_stream_binance_rejected_as_ws(self):
        with pytest.raises(ValidationError):
            _make_settings(binance_market_ws_url="wss://stream.binance.com")


# ---------------------------------------------------------------------------
# API key masking
# ---------------------------------------------------------------------------

class TestApiKeyMasking:
    def test_long_key_is_masked(self):
        s = _make_settings(binance_api_key="ABCD1234EFGH5678")
        masked = s.masked_api_key()
        assert "ABCD" in masked
        assert "5678" in masked
        assert "1234EFGH" not in masked
        assert "****" in masked

    def test_empty_key_shows_not_set(self):
        s = _make_settings(binance_api_key="")
        assert s.masked_api_key() == "(not set)"

    def test_secret_always_masked(self):
        s = _make_settings(binance_api_secret="supersecretvalue")
        assert s.masked_api_secret() == "****"

    def test_empty_secret_shows_not_set(self):
        s = _make_settings(binance_api_secret="")
        assert s.masked_api_secret() == "(not set)"


# ---------------------------------------------------------------------------
# Risk defaults
# ---------------------------------------------------------------------------

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
