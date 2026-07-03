"""Tests for market data API endpoints (no real HTTP calls)."""

import inspect
from decimal import Decimal
from unittest.mock import AsyncMock
from unittest.mock import AsyncMock as _AsyncMock
from unittest.mock import patch as _patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.market_data.client import MarketDataClient
from app.market_data.exceptions import BannedError, MarketDataError, RateLimitError
from app.market_data.kline_parser import KlineData
from app.models import (  # noqa: F401  # noqa: F401
    candle,
    daily_risk_state,
    order,
    paper_account,
    position,
    signal,
    strategy_config,
    system_event,
    trade,
)
from app.repositories.candle_repository import CandleRepository

# ---------------------------------------------------------------------------
# App + fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_session(test_engine):
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def mock_market_client():
    client = AsyncMock(spec=MarketDataClient)
    client.get_server_time.return_value = 9_999_999_999_999
    client.get_exchange_info.return_value = {
        "symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "permissions": ["SPOT"]}]
    }
    client.get_klines.return_value = []
    return client


@pytest.fixture
def api_client(test_session, mock_market_client):
    from app.api.market_data import get_market_data_client
    from app.main import create_app

    application = create_app()
    application.dependency_overrides[get_db] = lambda: test_session
    application.dependency_overrides[get_market_data_client] = lambda: mock_market_client

    with (
        _patch("app.main.run_migrations"),
        _patch("app.main.verify_db_connection"),
        _patch("app.main.BinanceMarketDataClient") as MockClient,
    ):
        MockClient.return_value.close = _AsyncMock()
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client


def _seed_candle(session, open_time: int = 1_700_000_000_000) -> KlineData:
    kd = KlineData(
        symbol="BTCUSDT",
        interval="1h",
        open_time=open_time,
        open=Decimal("35000.00"),
        high=Decimal("35500.00"),
        low=Decimal("34800.00"),
        close=Decimal("35200.00"),
        volume=Decimal("100.5"),
        close_time=open_time + 3_600_000 - 1,
        quote_asset_volume=Decimal("3538100.00"),
        number_of_trades=1500,
        taker_buy_base_volume=Decimal("50.25"),
        taker_buy_quote_volume=Decimal("1769050.00"),
    )
    CandleRepository(session).upsert_batch([kd])
    session.commit()
    return kd


# ---------------------------------------------------------------------------
# GET /api/v1/market-data/klines
# ---------------------------------------------------------------------------


class TestGetKlines:
    def test_returns_empty_list_when_no_data(self, api_client):
        resp = api_client.get(
            "/api/v1/market-data/klines",
            params={"symbol": "ETHUSDT", "interval": "1h"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_stored_candles(self, api_client, test_session, mock_market_client):
        _seed_candle(test_session, 1_700_100_000_000)
        resp = api_client.get(
            "/api/v1/market-data/klines",
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "include_open_candle": "true",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["symbol"] == "BTCUSDT"

    def test_decimal_fields_are_strings(self, api_client, test_session, mock_market_client):
        _seed_candle(test_session, 1_700_200_000_000)
        resp = api_client.get(
            "/api/v1/market-data/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "include_open_candle": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        if data:
            candle_item = data[0]
            for field in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
            ):
                assert isinstance(
                    candle_item[field], str
                ), f"Field {field!r} should be string in JSON response"

    def test_timestamps_include_iso_string(self, api_client, test_session, mock_market_client):
        _seed_candle(test_session, 1_700_300_000_000)
        resp = api_client.get(
            "/api/v1/market-data/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "include_open_candle": "true"},
        )
        data = resp.json()
        if data:
            assert "open_time_iso" in data[0]
            assert "close_time_iso" in data[0]
            assert "Z" in data[0]["open_time_iso"] or "+" in data[0]["open_time_iso"]

    def test_invalid_interval_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/market-data/klines",
            params={"symbol": "BTCUSDT", "interval": "99x"},
        )
        assert resp.status_code == 422

    def test_limit_enforced(self, api_client):
        resp = api_client.get(
            "/api/v1/market-data/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 99999},
        )
        assert resp.status_code == 422  # exceeds _MAX_LIMIT

    def test_naive_datetime_rejected(self, api_client):
        resp = api_client.get(
            "/api/v1/market-data/klines",
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2025-01-01T00:00:00",  # no timezone
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/market-data/download
# ---------------------------------------------------------------------------


class TestDownloadKlines:
    def _body(self, **overrides) -> dict:
        base = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T06:00:00Z",
            "include_open_candle": False,
        }
        base.update(overrides)
        return base

    def test_download_success_returns_200(self, api_client, mock_market_client):
        mock_market_client.get_klines.return_value = []
        resp = api_client.post("/api/v1/market-data/download", json=self._body())
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BTCUSDT"
        assert "inserted" in data
        assert "requests_made" in data
        assert "warning" in data

    def test_start_after_end_returns_422(self, api_client):
        resp = api_client.post(
            "/api/v1/market-data/download",
            json=self._body(
                start="2025-01-02T00:00:00Z",
                end="2025-01-01T00:00:00Z",
            ),
        )
        assert resp.status_code == 422

    def test_invalid_interval_returns_422(self, api_client):
        resp = api_client.post("/api/v1/market-data/download", json=self._body(interval="99x"))
        assert resp.status_code == 422

    def test_naive_datetime_rejected(self, api_client):
        resp = api_client.post(
            "/api/v1/market-data/download",
            json=self._body(
                start="2025-01-01T00:00:00",  # no Z
            ),
        )
        assert resp.status_code == 422

    def test_rate_limit_error_returns_429(self, api_client, mock_market_client):
        mock_market_client.get_klines.side_effect = RateLimitError(30)
        resp = api_client.post("/api/v1/market-data/download", json=self._body())
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_banned_error_returns_503(self, api_client, mock_market_client):
        mock_market_client.get_klines.side_effect = BannedError(300)
        resp = api_client.post("/api/v1/market-data/download", json=self._body())
        assert resp.status_code == 503
        # Must not expose internal error details to client
        assert "banned" not in resp.text.lower()
        assert "418" not in resp.text

    def test_binance_error_returns_503(self, api_client, mock_market_client):
        mock_market_client.get_klines.side_effect = MarketDataError("Upstream error")
        resp = api_client.post("/api/v1/market-data/download", json=self._body())
        assert resp.status_code == 503

    def test_no_stack_trace_in_error_response(self, api_client, mock_market_client):
        mock_market_client.get_klines.side_effect = Exception("Unexpected!")
        resp = api_client.post("/api/v1/market-data/download", json=self._body())
        assert "Traceback" not in resp.text
        assert "Exception" not in resp.text


# ---------------------------------------------------------------------------
# Security / architecture checks
# ---------------------------------------------------------------------------


class TestArchitectureGuards:
    def test_market_data_client_has_no_order_methods(self):
        forbidden = {"order", "trade", "balance", "account", "withdraw", "cancel"}
        method_names = {
            name for name, _ in inspect.getmembers(MarketDataClient, predicate=inspect.isfunction)
        }
        for name in method_names:
            for keyword in forbidden:
                assert (
                    keyword not in name.lower()
                ), f"MarketDataClient has forbidden method {name!r} containing {keyword!r}"

    def test_no_api_key_required_for_download(self, api_client, mock_market_client):
        """Download endpoint must work without any API key."""
        mock_market_client.get_klines.return_value = []
        resp = api_client.post(
            "/api/v1/market-data/download",
            json={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T02:00:00Z",
            },
        )
        # Market data client was called without any credential
        assert resp.status_code == 200
