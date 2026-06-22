"""Tests for GET /api/v1/indicators and GET /api/v1/indicators/latest."""

from decimal import Decimal
from unittest.mock import AsyncMock as _AsyncMock
from unittest.mock import patch as _patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.market_data.kline_parser import KlineData
from app.models import (  # noqa: F401 — register all models
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

_BASE_TIME = 1_700_000_000_000
_INTERVAL_MS = 3_600_000  # 1h


# ---------------------------------------------------------------------------
# Fixtures
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
def api_client(test_session):
    from app.main import create_app

    application = create_app()
    application.dependency_overrides[get_db] = lambda: test_session

    with (
        _patch("app.main.run_migrations"),
        _patch("app.main.verify_db_connection"),
        _patch("app.main.BinanceMarketDataClient") as MockClient,
    ):
        MockClient.return_value.close = _AsyncMock()
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client


def _seed_candle(session, idx: int, close: str = "35000.00", symbol: str = "BTCUSDT") -> None:
    open_time = _BASE_TIME + idx * _INTERVAL_MS
    price = Decimal(close)
    delta = price * Decimal("0.01")  # 1% — keeps low > 0 for any price
    kd = KlineData(
        symbol=symbol,
        interval="1h",
        open_time=open_time,
        open=price,
        high=price + delta,
        low=price - delta,
        close=price,
        volume=Decimal("100"),
        close_time=open_time + _INTERVAL_MS - 1,
        quote_asset_volume=Decimal("3500000"),
        number_of_trades=1000,
        taker_buy_base_volume=Decimal("50"),
        taker_buy_quote_volume=Decimal("1750000"),
    )
    CandleRepository(session).upsert_batch([kd])
    session.commit()


def _seed_n(session, n: int, symbol: str = "INDICTEST") -> None:
    """Seed n candles with slightly rising prices."""
    for i in range(n):
        _seed_candle(session, idx=i + 10_000, close=str(10000 + i), symbol=symbol)


# ---------------------------------------------------------------------------
# /api/v1/indicators — 404 when no data
# ---------------------------------------------------------------------------


class TestGetIndicators404:
    def test_no_data_returns_404(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators",
            params={"symbol": "NODATA", "interval": "1h"},
        )
        assert resp.status_code == 404

    def test_wrong_interval_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators",
            params={"symbol": "BTCUSDT", "interval": "99x"},
        )
        assert resp.status_code == 422

    def test_invalid_period_config_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators",
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "sma_short_period": 50,
                "sma_long_period": 20,  # violates short < long
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/indicators — insufficient data (warmup)
# ---------------------------------------------------------------------------


class TestGetIndicatorsWarmup:
    @pytest.fixture(autouse=True)
    def seed(self, test_session):
        """Seed 5 candles — far fewer than the default warmup of 200."""
        _seed_n(test_session, 5, symbol="WARMUPTEST")

    def test_returns_422_when_warmup_incomplete(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators",
            params={"symbol": "WARMUPTEST", "interval": "1h"},
        )
        assert resp.status_code == 422
        assert "warm-up" in resp.json()["detail"].lower()

    def test_include_warmup_returns_200(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators",
            params={"symbol": "WARMUPTEST", "interval": "1h", "include_warmup": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5
        for item in data:
            assert item["warmup_complete"] is False

    def test_include_warmup_indicators_are_null_during_warmup(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators",
            params={"symbol": "WARMUPTEST", "interval": "1h", "include_warmup": "true"},
        )
        assert resp.status_code == 200
        first = resp.json()[0]
        assert first["sma_short"] is None
        assert first["sma_long"] is None
        assert first["rsi"] is None
        assert first["atr"] is None


# ---------------------------------------------------------------------------
# /api/v1/indicators — complete results with small config
# ---------------------------------------------------------------------------


class TestGetIndicatorsComplete:
    SYMBOL = "SMALLCFG"

    @pytest.fixture(autouse=True)
    def seed(self, test_session):
        _seed_n(test_session, 20, symbol=self.SYMBOL)

    def _small_params(self, **extra) -> dict:
        return {
            "symbol": self.SYMBOL,
            "interval": "1h",
            "sma_short_period": 2,
            "sma_long_period": 3,
            "ema_short_period": 2,
            "ema_medium_period": 3,
            "ema_long_period": 4,
            "rsi_period": 3,
            "atr_period": 3,
            "volume_period": 3,
            **extra,
        }

    def test_returns_200_with_small_periods(self, api_client):
        resp = api_client.get("/api/v1/indicators", params=self._small_params())
        assert resp.status_code == 200

    def test_response_is_list(self, api_client):
        resp = api_client.get("/api/v1/indicators", params=self._small_params())
        assert isinstance(resp.json(), list)
        assert len(resp.json()) > 0

    def test_all_returned_results_are_warmup_complete(self, api_client):
        resp = api_client.get("/api/v1/indicators", params=self._small_params())
        for item in resp.json():
            assert item["warmup_complete"] is True

    def test_decimal_fields_are_strings(self, api_client):
        resp = api_client.get("/api/v1/indicators", params=self._small_params())
        item = resp.json()[0]
        for field in (
            "close",
            "sma_short",
            "sma_long",
            "ema_short",
            "ema_medium",
            "ema_long",
            "rsi",
            "atr",
            "volume_sma",
        ):
            assert isinstance(item[field], str), (
                f"Field {field!r} should be str, got {type(item[field])}"
            )

    def test_iso_timestamps_present(self, api_client):
        resp = api_client.get("/api/v1/indicators", params=self._small_params())
        item = resp.json()[0]
        assert "open_time_iso" in item
        assert "close_time_iso" in item
        assert "T" in item["open_time_iso"]

    def test_cross_signal_field_is_string(self, api_client):
        resp = api_client.get("/api/v1/indicators", params=self._small_params())
        for item in resp.json():
            assert item["ema_short_medium_cross"] in ("bullish", "bearish", "none")

    def test_symbol_normalised_to_uppercase(self, api_client):
        params = self._small_params()
        params["symbol"] = self.SYMBOL.lower()
        resp = api_client.get("/api/v1/indicators", params=params)
        assert resp.status_code == 200
        for item in resp.json():
            assert item["symbol"] == self.SYMBOL.upper()

    def test_limit_parameter_respected(self, api_client):
        resp = api_client.get("/api/v1/indicators", params=self._small_params(limit=3))
        assert resp.status_code == 200
        # After excluding warmup, we may get fewer — just verify limit applied
        assert len(resp.json()) <= 3

    def test_no_trading_signals_in_response(self, api_client):
        """Verify that BUY/SELL/WAIT signals are absent from the response schema."""
        resp = api_client.get("/api/v1/indicators", params=self._small_params())
        item = resp.json()[0]
        for forbidden in ("signal", "buy", "sell", "wait", "action", "recommendation"):
            assert forbidden not in item, (
                f"Field {forbidden!r} must not appear in indicator response"
            )


# ---------------------------------------------------------------------------
# /api/v1/indicators/latest
# ---------------------------------------------------------------------------


class TestGetLatestIndicators:
    SYMBOL = "LATESTTEST"

    @pytest.fixture(autouse=True)
    def seed(self, test_session):
        _seed_n(test_session, 20, symbol=self.SYMBOL)

    def _small_params(self, **extra) -> dict:
        return {
            "symbol": self.SYMBOL,
            "interval": "1h",
            "sma_short_period": 2,
            "sma_long_period": 3,
            "ema_short_period": 2,
            "ema_medium_period": 3,
            "ema_long_period": 4,
            "rsi_period": 3,
            "atr_period": 3,
            "volume_period": 3,
            **extra,
        }

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/v1/indicators/latest", params=self._small_params())
        assert resp.status_code == 200

    def test_response_has_candles_used(self, api_client):
        resp = api_client.get("/api/v1/indicators/latest", params=self._small_params())
        data = resp.json()
        assert "candles_used" in data
        assert data["candles_used"] == 20

    def test_response_has_single_result(self, api_client):
        resp = api_client.get("/api/v1/indicators/latest", params=self._small_params())
        data = resp.json()
        assert "symbol" in data
        assert "close" in data

    def test_latest_returns_404_for_unknown_symbol(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators/latest",
            params={"symbol": "UNKNOWN99", "interval": "1h"},
        )
        assert resp.status_code == 404

    def test_latest_invalid_interval_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/indicators/latest",
            params={"symbol": self.SYMBOL, "interval": "bad"},
        )
        assert resp.status_code == 422

    def test_latest_returns_most_recent_candle(self, api_client):
        resp = api_client.get("/api/v1/indicators/latest", params=self._small_params())
        data = resp.json()
        # The last seeded candle has close = str(10000 + 19) = "10019"
        assert Decimal(data["close"]) == Decimal("10019")

    def test_latest_decimal_fields_are_strings(self, api_client):
        resp = api_client.get("/api/v1/indicators/latest", params=self._small_params())
        data = resp.json()
        for field in ("close", "sma_short"):
            assert isinstance(data[field], str), f"{field} should be str in JSON"

    def test_latest_has_no_trading_signals(self, api_client):
        resp = api_client.get("/api/v1/indicators/latest", params=self._small_params())
        data = resp.json()
        for forbidden in ("signal", "buy", "sell", "wait", "action"):
            assert forbidden not in data
