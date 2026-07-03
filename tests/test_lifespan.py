"""
Integration test: real FastAPI lifespan + real Alembic migrations + temp SQLite.

Verifies:
  - lifespan starts without error
  - Alembic runs real migrations against a temp DB file
  - /health responds correctly
  - GET /api/v1/market-data/klines returns 200 (queries the temp DB)
  - BinanceMarketDataClient.close() is called when TestClient exits
  - No external HTTP connections are made
"""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from alembic import command
from alembic.config import Config
from app.database import get_db


class TestLifespanIntegration:
    def test_real_lifespan_health_klines_and_client_close(self, tmp_path, monkeypatch):
        """
        End-to-end wiring test with no mocked components except:
          - The Alembic target URL (redirected to a temp file).
          - verify_db_connection (uses the patched engine).
          - get_db sessions (bound to the same temp engine).

        BinanceMarketDataClient is created for real but no methods are called
        (GET /klines with include_open_candle=True skips get_server_time()).
        """
        db_url = f"sqlite:///{tmp_path}/lifespan_integration.db"

        # ---- Point alembic at the temp DB (real migrations) ----
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        def _run_real_migrations_temp() -> None:
            command.upgrade(alembic_cfg, "head")

        monkeypatch.setattr("app.main.run_migrations", _run_real_migrations_temp)

        # ---- verify_db_connection uses our temp engine ----
        temp_engine = create_engine(db_url, connect_args={"check_same_thread": False})

        from sqlalchemy import text

        def _verify_temp() -> None:
            with temp_engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        monkeypatch.setattr("app.main.verify_db_connection", _verify_temp)

        # ---- get_db yields sessions from the temp engine ----
        TempSession = sessionmaker(bind=temp_engine)

        def _get_db_temp():
            db = TempSession()
            try:
                yield db
            finally:
                db.close()

        from app.main import create_app

        application = create_app()
        application.dependency_overrides[get_db] = _get_db_temp

        with TestClient(application, raise_server_exceptions=True) as tc:
            # -- /health --
            resp = tc.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] in ("HEALTHY", "DEGRADED", "ERROR")
            assert "db_status" in body
            assert "no real money" in body["warning"].lower()

            # -- GET /klines (empty DB, include_open_candle=True → no server-time call) --
            resp = tc.get(
                "/api/v1/market-data/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "include_open_candle": "true",
                },
            )
            assert resp.status_code == 200
            assert resp.json() == []

            # Capture the live client reference before lifespan teardown
            market_client = application.state.market_data_client
            assert not market_client._closed, "client must be open during lifespan"

        # After TestClient.__exit__, lifespan teardown has executed close()
        assert market_client._closed is True, "client must be closed after lifespan teardown"

        # Verify real migrations ran: all expected tables must exist in temp DB
        tables = set(inspect(temp_engine).get_table_names())
        expected = {
            "candles",
            "signals",
            "orders",
            "positions",
            "trades",
            "paper_accounts",
            "daily_risk_states",
            "strategy_configs",
            "system_events",
        }
        missing = expected - tables
        assert expected.issubset(tables), f"Missing tables after real migrations: {missing}"

        temp_engine.dispose()
