"""Tests for POST /api/v1/backtests/run."""

from decimal import Decimal
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.backtesting.config import BacktestConfig
from app.backtesting.schemas import BacktestResult
from app.database import Base, get_db
from app.main import create_app
from app.models import candle  # noqa: F401 — register model

_D = Decimal


@pytest.fixture()
def client():
    """FastAPI test client with an isolated in-memory DB."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app, raise_server_exceptions=False)


def _valid_request() -> dict:
    return {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-06-01T00:00:00Z",
        "initial_capital": "10000",
        "fee_percentage": "0.1",
        "slippage_percentage": "0.05",
        "force_close_at_end": True,
    }


class TestBacktestingAPIValidation:
    def test_missing_symbol_returns_422(self, client):
        body = _valid_request()
        del body["symbol"]
        resp = client.post("/api/v1/backtests/run", json=body)
        assert resp.status_code == 422

    def test_invalid_interval_returns_422(self, client):
        body = _valid_request()
        body["interval"] = "99x"
        resp = client.post("/api/v1/backtests/run", json=body)
        assert resp.status_code == 422

    def test_end_before_start_returns_422(self, client):
        body = _valid_request()
        body["start"] = "2024-06-01T00:00:00Z"
        body["end"] = "2024-01-01T00:00:00Z"
        resp = client.post("/api/v1/backtests/run", json=body)
        assert resp.status_code == 422

    def test_negative_capital_returns_422(self, client):
        body = _valid_request()
        body["initial_capital"] = "-100"
        resp = client.post("/api/v1/backtests/run", json=body)
        assert resp.status_code == 422

    def test_fee_above_5_returns_422(self, client):
        body = _valid_request()
        body["fee_percentage"] = "10"
        resp = client.post("/api/v1/backtests/run", json=body)
        assert resp.status_code == 422


class TestBacktestingAPINotFound:
    def test_no_data_returns_404(self, client):
        """When no candles exist in DB, service raises BacktestInsufficientDataError → 404."""
        resp = client.post("/api/v1/backtests/run", json=_valid_request())
        assert resp.status_code == 404

    def test_404_body_has_detail(self, client):
        resp = client.post("/api/v1/backtests/run", json=_valid_request())
        data = resp.json()
        assert "detail" in data


class TestBacktestingAPIResponseStructure:
    def test_successful_run_returns_expected_keys(self, client):
        """Mock the service to return a minimal BacktestResult."""
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=1_000_000,
            end_ms=2_000_000,
            initial_capital=_D("10000"),
        )
        fake_result = BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=[],
            first_candle_open_time=1_000_000,
            last_candle_open_time=1_100_000,
            total_candles=10,
            evaluated_candles=5,
            initial_capital=_D("10000"),
            final_equity=_D("10000"),
            total_return_pct=_D("0"),
            buy_and_hold_return_pct=_D("0"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=None,
            avg_win_pct=None,
            avg_loss_pct=None,
            profit_factor=None,
            expectancy_pct=None,
            max_drawdown_pct=_D("0"),
            exposure_pct=_D("0"),
            max_win_streak=0,
            max_loss_streak=0,
            total_fees=_D("0"),
            has_open_position_at_end=False,
        )

        with patch("app.api.backtesting.BacktestService") as MockSvc:
            MockSvc.return_value.run.return_value = fake_result
            resp = client.post("/api/v1/backtests/run", json=_valid_request())

        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "summary" in data
        assert "trades" in data
        assert "equity_curve" in data
        assert "warning" in data

    def test_decimal_values_are_strings_in_response(self, client):
        cfg = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            start_ms=1_000_000,
            end_ms=2_000_000,
            initial_capital=_D("10000"),
        )
        fake_result = BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=[],
            first_candle_open_time=1_000_000,
            last_candle_open_time=1_100_000,
            total_candles=10,
            evaluated_candles=5,
            initial_capital=_D("10000"),
            final_equity=_D("10500"),
            total_return_pct=_D("5"),
            buy_and_hold_return_pct=_D("3"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=None,
            avg_win_pct=None,
            avg_loss_pct=None,
            profit_factor=None,
            expectancy_pct=None,
            max_drawdown_pct=_D("2"),
            exposure_pct=_D("30"),
            max_win_streak=0,
            max_loss_streak=0,
            total_fees=_D("10"),
            has_open_position_at_end=False,
        )

        with patch("app.api.backtesting.BacktestService") as MockSvc:
            MockSvc.return_value.run.return_value = fake_result
            resp = client.post("/api/v1/backtests/run", json=_valid_request())

        assert resp.status_code == 200
        data = resp.json()
        # Financial values must be strings, not floats
        assert isinstance(data["summary"]["final_equity"], str)
        assert isinstance(data["summary"]["total_return_pct"], str)
        assert isinstance(data["summary"]["total_fees"], str)


class TestBacktestingAPISecurityHeaders:
    def test_endpoint_does_not_place_orders(self, client):
        """The endpoint must never import or call order-related code."""
        import app.api.backtesting as module

        source = open(module.__file__).read()
        forbidden = [
            "BinanceMarketDataClient",
            "place_order",
            "create_order",
            "PaperAccount",
            "leverage_factor",
            "futures_position",
            "margin_call",
            "short_position",
            "sell_short",
            "open_short",
        ]
        for term in forbidden:
            assert term.lower() not in source.lower(), (
                f"Forbidden term {term!r} found in backtesting API module"
            )
