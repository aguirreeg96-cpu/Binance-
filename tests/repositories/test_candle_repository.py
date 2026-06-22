"""Tests for CandleRepository upsert logic and Alembic migration cycle."""

from decimal import Decimal

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from alembic.config import Config
from app.market_data.kline_parser import KlineData
from app.models.candle import Candle
from app.models.types import normalize_decimal
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
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        open=Decimal("35000.00"),
        high=Decimal("35500.00"),
        low=Decimal("34800.00"),
        close=Decimal(close),
        volume=Decimal("100.5"),
        close_time=close_time,
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

        for field_name in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ):
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
        result = repo.upsert_batch(
            [
                _kline(t0, close="35200.00"),  # same → ignored
                _kline(t1, close="36000.00"),  # changed → updated
                _kline(t2),  # new → inserted
            ]
        )
        alembic_session.commit()

        assert result.inserted == 1
        assert result.updated == 1
        assert result.ignored == 1


class TestQuery:
    def test_query_ascending_order(self, repo, alembic_session):
        t0 = 1_700_000_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(5)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", start_ms=t0)
        times = [r.open_time for r in rows]
        assert times == sorted(times)

    def test_query_respects_start_boundary_inclusive(self, repo, alembic_session):
        t0 = 1_700_000_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(3)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", start_ms=t0 + 3_600_000)
        assert len(rows) == 2
        assert rows[0].open_time == t0 + 3_600_000

    def test_query_respects_end_boundary_exclusive(self, repo, alembic_session):
        t0 = 1_700_000_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(3)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", end_ms=t0 + 3_600_000)
        assert len(rows) == 1
        assert rows[0].open_time == t0

    def test_query_limit(self, repo, alembic_session):
        t0 = 1_700_000_000_000
        klines = [_kline(t0 + i * 3_600_000) for i in range(10)]
        repo.upsert_batch(klines)
        alembic_session.commit()

        rows = repo.query("BTCUSDT", "1h", limit=3)
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
            "candles",
            "signals",
            "paper_accounts",
            "positions",
            "orders",
            "trades",
            "strategy_configs",
            "daily_risk_states",
            "system_events",
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
            "id",
            "symbol",
            "interval",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "is_closed",
            "created_at",
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

    def test_002_upgrade_changes_candle_decimal_columns_to_varchar(self, tmp_path):
        """After running migration 002, candle decimal columns must be VARCHAR(50)."""
        db_url = f"sqlite:///{tmp_path}/pragma_test.db"
        cfg = self._make_cfg(db_url)
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        with engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(candles)")).fetchall()
        # PRAGMA columns: (cid, name, type, notnull, dflt_value, pk)
        col_type_map = {row[1]: row[2] for row in cols}
        decimal_cols = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        )
        for col in decimal_cols:
            assert "VARCHAR" in col_type_map[col].upper(), (
                f"Column {col!r} expected VARCHAR(50), got {col_type_map[col]!r}"
            )
        engine.dispose()

    def test_002_downgrade_reverts_columns_to_numeric(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/downgrade002_test.db"
        cfg = self._make_cfg(db_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "001")

        engine = create_engine(db_url)
        with engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(candles)")).fetchall()
        col_type_map = {row[1]: row[2] for row in cols}
        # After downgrade to 001, columns should be NUMERIC (not VARCHAR)
        assert "VARCHAR" not in col_type_map["open"].upper()
        engine.dispose()

    def test_002_upgrade_downgrade_upgrade_cycle(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/cycle002_test.db"
        cfg = self._make_cfg(db_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "001")
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        with engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(candles)")).fetchall()
        col_type_map = {row[1]: row[2] for row in cols}
        assert "VARCHAR" in col_type_map["open"].upper()
        engine.dispose()


# ---------------------------------------------------------------------------
# Idempotency regression tests — the core bug fix
# ---------------------------------------------------------------------------


def _large_decimal_kline(open_time: int = 1_900_000_000_000) -> KlineData:
    """KlineData with the exact values that triggered the idempotency bug."""
    return KlineData(
        symbol="BTCUSDT",
        interval="15m",
        open_time=open_time,
        open=Decimal("42150.32000000"),
        high=Decimal("42300.00000000"),
        low=Decimal("42100.00000000"),
        close=Decimal("42200.50000000"),
        volume=Decimal("1253.48700000"),
        close_time=open_time + 900_000 - 1,
        quote_asset_volume=Decimal("10753491.66553520"),  # the problematic field
        number_of_trades=8742,
        taker_buy_base_volume=Decimal("651.23400000"),
        taker_buy_quote_volume=Decimal("27481234.87654321"),  # also had float64 error
    )


class TestIdempotencyRegression:
    """Regression suite for the float64 round-trip idempotency bug.

    Root cause: SQLite's NUMERIC affinity converted Decimal strings to REAL
    (float64), introducing errors of ~4e-10 for values with >15 significant
    digits.  A re-download of identical candles would then detect spurious
    differences (e.g. stored=10753491.6655352004 vs incoming=10753491.66553520)
    and classify them as 'updated'.

    Fix: ExactDecimal stores values as VARCHAR(50) with TEXT affinity, and
    normalize_decimal() ensures insert, update, and comparison all use the
    same 10-decimal-place scale.
    """

    def test_second_identical_download_large_decimals_ignored(self, repo, alembic_session):
        klines = [_large_decimal_kline()]
        repo.upsert_batch(klines)
        alembic_session.commit()

        result = repo.upsert_batch(klines)
        alembic_session.commit()

        assert result.inserted == 0
        assert result.updated == 0
        assert result.ignored == 1

    def test_real_change_in_close_still_produces_update(self, repo, alembic_session):
        t0 = 1_900_100_000_000
        repo.upsert_batch([_large_decimal_kline(t0)])
        alembic_session.commit()

        changed = KlineData(
            symbol="BTCUSDT",
            interval="15m",
            open_time=t0,
            open=Decimal("42150.32000000"),
            high=Decimal("42300.00000000"),
            low=Decimal("42100.00000000"),
            close=Decimal("43000.00000000"),  # real change
            volume=Decimal("1253.48700000"),
            close_time=t0 + 900_000 - 1,
            quote_asset_volume=Decimal("10753491.66553520"),
            number_of_trades=8742,
            taker_buy_base_volume=Decimal("651.23400000"),
            taker_buy_quote_volume=Decimal("27481234.87654321"),
        )
        result = repo.upsert_batch([changed])
        alembic_session.commit()

        assert result.updated == 1
        assert result.ignored == 0

        row = alembic_session.query(Candle).filter_by(open_time=t0).one()
        assert row.close == normalize_decimal(Decimal("43000.00000000"))

    def test_real_change_in_quote_asset_volume_produces_update(self, repo, alembic_session):
        t0 = 1_900_200_000_000
        repo.upsert_batch([_large_decimal_kline(t0)])
        alembic_session.commit()

        changed = KlineData(
            symbol="BTCUSDT",
            interval="15m",
            open_time=t0,
            open=Decimal("42150.32000000"),
            high=Decimal("42300.00000000"),
            low=Decimal("42100.00000000"),
            close=Decimal("42200.50000000"),
            volume=Decimal("1253.48700000"),
            close_time=t0 + 900_000 - 1,
            quote_asset_volume=Decimal("10999999.99999999"),  # real change
            number_of_trades=8742,
            taker_buy_base_volume=Decimal("651.23400000"),
            taker_buy_quote_volume=Decimal("27481234.87654321"),
        )
        result = repo.upsert_batch([changed])
        alembic_session.commit()

        assert result.updated == 1

    def test_batch_288_identical_candles_all_ignored(self, repo, alembic_session):
        """Simulate the real-world two consecutive downloads of 288 BTCUSDT 15m candles."""
        t0 = 1_900_300_000_000
        klines = [_large_decimal_kline(t0 + i * 900_000) for i in range(288)]

        repo.upsert_batch(klines)
        alembic_session.commit()

        result = repo.upsert_batch(klines)
        alembic_session.commit()

        assert result.inserted == 0
        assert result.updated == 0
        assert result.ignored == 288

    def test_created_at_difference_does_not_cause_update(self, repo, alembic_session):
        """created_at is an internal timestamp — it must never drive update detection."""
        t0 = 1_900_400_000_000
        repo.upsert_batch([_large_decimal_kline(t0)])
        alembic_session.commit()

        row = alembic_session.query(Candle).filter_by(open_time=t0).one()
        original_created_at = row.created_at

        result = repo.upsert_batch([_large_decimal_kline(t0)])
        alembic_session.commit()

        assert result.ignored == 1
        assert result.updated == 0

        row2 = alembic_session.query(Candle).filter_by(open_time=t0).one()
        assert row2.created_at == original_created_at

    def test_fresh_session_read_back_returns_exact_decimal(self, tmp_path, alembic_session):
        """After commit, open a brand-new session and verify exact Decimal values."""
        t0 = 1_900_500_000_000
        klines = [_large_decimal_kline(t0)]
        repo = CandleRepository(alembic_session)
        repo.upsert_batch(klines)
        alembic_session.commit()

        # Expire all objects so the next access hits the DB
        alembic_session.expire_all()
        row = alembic_session.query(Candle).filter_by(open_time=t0).one()

        assert type(row.open) is Decimal
        assert type(row.quote_asset_volume) is Decimal
        assert row.quote_asset_volume == normalize_decimal(Decimal("10753491.66553520"))
        assert row.taker_buy_quote_volume == normalize_decimal(Decimal("27481234.87654321"))

    def test_upsert_inserts_exact_decimal_values(self, repo, alembic_session):
        t0 = 1_900_600_000_000
        klines = [_large_decimal_kline(t0)]
        result = repo.upsert_batch(klines)
        alembic_session.commit()

        assert result.inserted == 1
        assert result.updated == 0

        row = alembic_session.query(Candle).filter_by(open_time=t0).one()
        assert row.quote_asset_volume == normalize_decimal(Decimal("10753491.66553520"))

    def test_all_decimal_fields_are_decimal_type_after_upsert(self, repo, alembic_session):
        t0 = 1_900_700_000_000
        repo.upsert_batch([_large_decimal_kline(t0)])
        alembic_session.commit()
        alembic_session.expire_all()

        row = alembic_session.query(Candle).filter_by(open_time=t0).one()
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
            val = getattr(row, field)
            assert type(val) is Decimal, (
                f"Column {field!r} should be Decimal after reload, got {type(val)}"
            )

    def test_idempotency_pragma_confirms_varchar_storage(self, tmp_path):
        """PRAGMA table_info confirms candle decimal columns have VARCHAR/TEXT affinity."""
        db_url = f"sqlite:///{tmp_path}/idempotency_pragma.db"
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", db_url)
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        with engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(candles)")).fetchall()
        col_map = {row[1]: row[2] for row in cols}
        for col in ("open", "quote_asset_volume", "taker_buy_quote_volume"):
            assert "VARCHAR" in col_map[col].upper(), (
                f"Column {col!r} must be VARCHAR for exact TEXT storage, got {col_map[col]!r}"
            )
        engine.dispose()
