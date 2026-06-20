"""Tests for CandleRepository upsert logic and Alembic migration cycle."""

import os
import tempfile
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.market_data.kline_parser import KlineData
from app.models.candle import Candle
from app.repositories.candle_repository import CandleRepository, UpsertResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(alembic_session):
    return CandleRepository(alembic_session)


def _kline(
    open_time: int = 1_700_000_000_000,
    close: str = "35200.00",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> KlineData:
    close_time = open_time + 3_600_000 - 1
    return KlineData(
        symbol=symbol, interval=interval,
        open_time=open_time,
        open=Decimal("35000.00"), high=Decimal("35500.00"),
        low=Decimal("34800.00"), close=Decimal(close),
        volume=Decimal("100.5"), close_time=close_time,
        quote_asset_volume=Decimal("3538100.00"),
        number_of_trades=1500,
        taker_buy_base_volume=Decimal("50.25"),
        taker_buy_quote_volume=Decimal("1769050.00"),
    )


# ---------------------------------------------------------------------------
# Upsert tests
# ---------------------------------------------------------------------------

class TestUpsertBatch:
    def test_insert_new_candles(self, repo, alembic_session):
        klines = [_kline(1_700_000_000_000), _kline(1_700_003_600_000)]
        result = repo.upsert_batch(klines)

        assert result.inserted == 2
        assert result.updated == 0
        assert result.ignored == 0
        alembic_session.commit()

        count = alembic_session.execute(
            text("SELECT COUNT(*) FROM candles WHERE symbol='BTCUSDT'")
        ).scalar()
        assert count == 2

    def test_second_identical_download_all_ignored(self, repo, alembic_session):
        klines = [_kline(1_700_010_000_000)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        # Repeat exact same data
        result = repo.upsert_batch(klines)
        alembic_session.commit()

        assert result.inserted == 0
        assert result.updated == 0
        assert result.ignored == 1

    def test_changed_close_price_produces_update(self, repo, alembic_session):
        klines_v1 = [_kline(1_700_020_000_000, close="35200.00")]
        repo.upsert_batch(klines_v1)
        alembic_session.commit()

        klines_v2 = [_kline(1_700_020_000_000, close="35999.00")]
        result = repo.upsert_batch(klines_v2)
        alembic_session.commit()

        assert result.inserted == 0
        assert result.updated == 1
        assert result.ignored == 0

        row = alembic_session.query(Candle).filter_by(open_time=1_700_020_000_000).one()
        assert row.close == Decimal("35999.00")

    def test_empty_batch_returns_zeros(self, repo):
        result = repo.upsert_batch([])
        assert result == UpsertResult(0, 0, 0)

    def test_all_decimal_fields_returned_as_decimal(self, repo, alembic_session):
        klines = [_kline(1_700_030_000_000)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        row = alembic_session.query(Candle).filter_by(open_time=1_700_030_000_000).one()

        for field_name in ("open", "high", "low", "close", "volume",
                           "quote_asset_volume", "taker_buy_base_volume",
                           "taker_buy_quote_volume"):
            val = getattr(row, field_name)
            assert isinstance(val, Decimal), (
                f"Column {field_name!r} should return Decimal, got {type(val)}"
            )

    def test_mixed_insert_update_ignored(self, repo, alembic_session):
        t0 = 1_700_040_000_000
        t1 = t0 + 3_600_000
        t2 = t1 + 3_600_000

        # Insert t0 and t1
        repo.upsert_batch([_kline(t0), _kline(t1)])
        alembic_session.commit()

        # Second batch: t0 unchanged (ignored), t1 changed (updated), t2 new (inserted)
        result = repo.upsert_batch([
            _kline(t0, close="35200.00"),   # same → ignored
            _kline(t1, close="36000.00"),   # changed → updated
            _kline(t2),                      # new → inserted
        ])
        alembic_session.commit()

        assert result.inserted == 1
        assert result.updated == 1
        assert result.ignored == 1


class TestQuery:
    def test_query_ascending_order(self, repo, alembic_session):
        t0 = 1_700_050_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(5)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", start_ms=t0)
        times = [r.open_time for r in rows]
        assert times == sorted(times)

    def test_query_respects_start_boundary_inclusive(self, repo, alembic_session):
        # Use a t0 far past test_query_ascending_order's range (ends at 1_700_064_400_000)
        t0 = 1_700_150_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(3)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", start_ms=t0 + 3_600_000, end_ms=t0 + 3 * 3_600_000)
        assert len(rows) == 2
        assert rows[0].open_time == t0 + 3_600_000

    def test_query_respects_end_boundary_exclusive(self, repo, alembic_session):
        t0 = 1_700_200_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(3)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", start_ms=t0, end_ms=t0 + 3_600_000)
        assert len(rows) == 1
        assert rows[0].open_time == t0

    def test_query_limit(self, repo, alembic_session):
        t0 = 1_700_250_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(10)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", start_ms=t0, limit=3)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# Alembic migration cycle tests
# ---------------------------------------------------------------------------

class TestAlembicMigrations:
    def _make_cfg(self, db_url: str) -> Config:
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", db_url)
        return cfg

    def test_upgrade_head_creates_all_tables(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/migration_test.db"
        cfg = self._make_cfg(db_url)
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        tables = set(inspect(engine).get_table_names())
        expected = {
            "candles", "signals", "paper_accounts", "positions",
            "orders", "trades", "strategy_configs",
            "daily_risk_states", "system_events",
        }
        assert expected.issubset(tables), f"Missing: {expected - tables}"
        engine.dispose()

    def test_downgrade_base_drops_all_tables(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/downgrade_test.db"
        cfg = self._make_cfg(db_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")

        engine = create_engine(db_url)
        tables = [t for t in inspect(engine).get_table_names() if t != "alembic_version"]
        engine.dispose()
        assert tables == [], f"Tables still exist after downgrade: {tables}"

    def test_full_cycle_upgrade_downgrade_upgrade(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/cycle_test.db"
        cfg = self._make_cfg(db_url)

        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        tables = set(inspect(engine).get_table_names())
        engine.dispose()
        assert "candles" in tables

    def test_candles_table_has_all_columns(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/cols_test.db"
        cfg = self._make_cfg(db_url)
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        col_names = {c["name"] for c in inspect(engine).get_columns("candles")}
        engine.dispose()

        required = {
            "id", "symbol", "interval", "open_time",
            "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base_volume", "taker_buy_quote_volume",
            "is_closed", "created_at",
        }
        assert required.issubset(col_names), f"Missing columns: {required - col_names}"

    def test_candles_unique_constraint_exists(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/uc_test.db"
        cfg = self._make_cfg(db_url)
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        kline = _kline(1_800_000_000_000)
        from app.repositories.candle_repository import _candle_from_kline
        candle1 = _candle_from_kline(kline)
        candle2 = _candle_from_kline(kline)  # same key

        session.add(candle1)
        session.commit()
        session.add(candle2)

        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            session.commit()

        session.rollback()
        session.close()
        engine.dispose()
