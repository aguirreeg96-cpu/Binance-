"""Tests for GET /api/v1/strategy/latest and GET /api/v1/strategy/history."""

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
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()
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
    delta = price * Decimal("0.01")
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
        quote_asset_volume=price * Decimal("100"),
        number_of_trades=1000,
        taker_buy_base_volume=Decimal("50"),
        taker_buy_quote_volume=price * Decimal("50"),
    )
    CandleRepository(session).upsert_batch([kd])
    session.commit()


def _seed_n(session, n: int, symbol: str = "STRATTEST") -> None:
    """Seed n candles with slightly rising prices (10000, 10001, …)."""
    for i in range(n):
        _seed_candle(session, idx=i + 10_000, close=str(10000 + i), symbol=symbol)


# Small-period params that produce warmup_complete=True with ~20 candles.
# warmup_candles = max(sma_long=3, ema_long=4, rsi+1=4, atr=3, volume=3) = 4
_SMALL_IND = {
    "sma_short_period": 2,
    "sma_long_period": 3,
    "ema_short_period": 2,
    "ema_medium_period": 3,
    "ema_long_period": 4,
    "rsi_period": 3,
    "atr_period": 3,
    "volume_period": 3,
}

# Strategy config that relaxes crossover/EMA constraints for predictable test outcomes.
_RELAXED_STRAT = {
    "require_bullish_crossover": "false",
    "require_price_above_long_ema": "false",
    "require_bearish_crossover_for_sell": "false",
    "buy_rsi_min": "0",
    "buy_rsi_max": "100",
    "minimum_volume_ratio": "0",
}


# ---------------------------------------------------------------------------
# GET /api/v1/strategy/latest — error cases
# ---------------------------------------------------------------------------


class TestLatest404:
    def test_no_data_returns_404(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/latest",
            params={"symbol": "NODATAXYZ", "interval": "1h"},
        )
        assert resp.status_code == 404

    def test_wrong_interval_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/latest",
            params={"symbol": "BTCUSDT", "interval": "99x"},
        )
        assert resp.status_code == 422

    def test_invalid_period_config_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/latest",
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "sma_short_period": 50,
                "sma_long_period": 20,  # violates short < long
            },
        )
        assert resp.status_code == 422

    def test_invalid_rsi_bounds_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/latest",
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "buy_rsi_min": "65",
                "buy_rsi_max": "50",  # min > max
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/strategy/latest — successful responses
# ---------------------------------------------------------------------------


class TestLatestSuccess:
    SYMBOL = "STRAT_LATEST"

    @pytest.fixture(autouse=True)
    def seed(self, test_session):
        _seed_n(test_session, 20, symbol=self.SYMBOL)

    def _params(self, **extra) -> dict:
        return {"symbol": self.SYMBOL, "interval": "1h", **_SMALL_IND, **extra}

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        assert resp.status_code == 200

    def test_response_has_action_field(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        assert "action" in data
        assert data["action"] in ("BUY", "SELL", "WAIT")

    def test_response_action_is_string(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        assert isinstance(resp.json()["action"], str)

    def test_close_price_is_string(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        assert isinstance(data["close_price"], str)
        Decimal(data["close_price"])  # must be valid decimal

    def test_reasons_is_list_of_strings(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        assert isinstance(data["reasons"], list)
        for r in data["reasons"]:
            assert isinstance(r, str)

    def test_failed_conditions_is_list(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        assert isinstance(resp.json()["failed_conditions"], list)

    def test_indicators_snapshot_present(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        assert "indicators_snapshot" in data
        snap = data["indicators_snapshot"]
        assert "close" in snap
        assert "crossover" in snap

    def test_snapshot_close_is_string_decimal(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        snap = resp.json()["indicators_snapshot"]
        Decimal(snap["close"])  # valid

    def test_snapshot_crossover_is_valid_string(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        snap = resp.json()["indicators_snapshot"]
        assert snap["crossover"] in ("bullish", "bearish", "none")

    def test_iso_timestamps_present(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        assert "candle_open_time_iso" in data
        assert "candle_close_time_iso" in data
        assert "T" in data["candle_open_time_iso"]

    def test_generated_at_is_iso_string(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        generated_at = resp.json()["generated_at"]
        assert isinstance(generated_at, str)
        assert "T" in generated_at

    def test_symbol_normalised_to_uppercase(self, api_client):
        params = self._params(**_RELAXED_STRAT)
        params["symbol"] = self.SYMBOL.lower()
        resp = api_client.get("/api/v1/strategy/latest", params=params)
        assert resp.status_code == 200
        assert resp.json()["symbol"] == self.SYMBOL.upper()

    def test_warmup_complete_is_bool(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        assert isinstance(data["warmup_complete"], bool)

    def test_has_open_position_flag_false_by_default(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        assert resp.json()["has_open_position"] is False

    def test_has_open_position_flag_true_when_passed(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/latest",
            params=self._params(**_RELAXED_STRAT, has_open_position="true"),
        )
        assert resp.json()["has_open_position"] is True

    def test_strategy_name_and_version_present(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        assert isinstance(data["strategy_name"], str)
        assert isinstance(data["strategy_version"], str)

    def test_no_order_execution_fields(self, api_client):
        """Confirm the response contains no order/trade execution fields."""
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        data = resp.json()
        for forbidden in ("order_id", "trade_id", "quantity", "fills", "executed_qty"):
            assert forbidden not in data, f"Forbidden field {forbidden!r} in response"

    def test_no_short_action_in_response(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        assert resp.json()["action"] != "SHORT"

    def test_relaxed_config_produces_buy(self, api_client):
        """With all constraints relaxed and no position, the latest signal is BUY."""
        resp = api_client.get(
            "/api/v1/strategy/latest",
            params=self._params(**_RELAXED_STRAT, has_open_position="false"),
        )
        assert resp.status_code == 200
        assert resp.json()["action"] == "BUY"

    def test_warmup_complete_true_with_enough_candles(self, api_client):
        resp = api_client.get("/api/v1/strategy/latest", params=self._params(**_RELAXED_STRAT))
        assert resp.json()["warmup_complete"] is True


# ---------------------------------------------------------------------------
# GET /api/v1/strategy/history — error cases
# ---------------------------------------------------------------------------


class TestHistoryErrors:
    def test_no_data_returns_404(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params={"symbol": "NODATAABC", "interval": "1h"},
        )
        assert resp.status_code == 404

    def test_wrong_interval_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params={"symbol": "BTCUSDT", "interval": "bad"},
        )
        assert resp.status_code == 422

    def test_invalid_period_config_returns_422(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "ema_short_period": 10,
                "ema_medium_period": 5,  # short > medium
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/strategy/history — successful responses
# ---------------------------------------------------------------------------


class TestHistorySuccess:
    SYMBOL = "STRAT_HIST"

    @pytest.fixture(autouse=True)
    def seed(self, test_session):
        _seed_n(test_session, 20, symbol=self.SYMBOL)

    def _params(self, **extra) -> dict:
        return {"symbol": self.SYMBOL, "interval": "1h", **_SMALL_IND, **extra}

    def test_returns_200_without_include_wait(self, api_client):
        resp = api_client.get("/api/v1/strategy/history", params=self._params(**_RELAXED_STRAT))
        assert resp.status_code == 200

    def test_response_is_list(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true"),
        )
        assert isinstance(resp.json(), list)

    def test_include_wait_false_filters_waits(self, api_client):
        """With relaxed rules and rising prices, all decisions should be BUY or SELL."""
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="false"),
        )
        data = resp.json()
        for item in data:
            assert item["action"] != "WAIT"

    def test_include_wait_true_may_return_waits(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true"),
        )
        data = resp.json()
        assert isinstance(data, list)
        # All items have valid action
        for item in data:
            assert item["action"] in ("BUY", "SELL", "WAIT")

    def test_history_items_have_required_fields(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true"),
        )
        data = resp.json()
        assert len(data) > 0
        item = data[0]
        for field in (
            "action",
            "symbol",
            "interval",
            "candle_open_time",
            "candle_open_time_iso",
            "close_price",
            "reasons",
            "failed_conditions",
            "indicators_snapshot",
            "warmup_complete",
            "has_open_position",
            "generated_at",
        ):
            assert field in item, f"Missing field {field!r}"

    def test_history_no_double_buy(self, api_client):
        """Position tracking: after a BUY, the next entry cannot also be BUY."""
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true"),
        )
        data = resp.json()
        prev_action = None
        for item in data:
            if prev_action == "BUY":
                assert item["action"] != "BUY", "Double BUY without intervening SELL"
            prev_action = item["action"]

    def test_history_ascending_candle_times(self, api_client):
        """Decisions must be returned in ascending candle open_time order."""
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true"),
        )
        data = resp.json()
        times = [item["candle_open_time"] for item in data]
        assert times == sorted(times)

    def test_limit_parameter_respected(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true", limit=3),
        )
        assert len(resp.json()) <= 3

    def test_history_symbols_are_uppercase(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true"),
        )
        for item in resp.json():
            assert item["symbol"] == self.SYMBOL.upper()

    def test_history_no_order_execution_fields(self, api_client):
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(**_RELAXED_STRAT, include_wait="true"),
        )
        for item in resp.json():
            for forbidden in ("order_id", "trade_id", "quantity", "fills"):
                assert forbidden not in item

    def test_initial_position_open_flag(self, api_client):
        """Starting with position open prevents BUY as first action."""
        resp = api_client.get(
            "/api/v1/strategy/history",
            params=self._params(
                **_RELAXED_STRAT,
                include_wait="true",
                initial_position_open="true",
            ),
        )
        data = resp.json()
        if data:
            assert data[0]["action"] != "BUY"
