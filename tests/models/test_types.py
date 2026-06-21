"""Tests for ExactDecimal TypeDecorator and normalize_decimal()."""

from decimal import Decimal

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import sqlite as sqlite_module

from app.models.types import DECIMAL_QUANTUM, ExactDecimal, normalize_decimal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQLITE_DIALECT = sqlite_module.dialect()


def _make_engine_and_table(tmp_path):
    """Fresh SQLite engine with a single ExactDecimal column for round-trip tests."""
    db_url = f"sqlite:///{tmp_path}/types_test.db"
    engine = create_engine(db_url)
    meta = sa.MetaData()
    table = sa.Table(
        "test_exact",
        meta,
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("value", ExactDecimal(), nullable=False),
    )
    meta.create_all(engine)
    return engine, table


def _insert_and_fetch(engine, table, value) -> Decimal:
    with engine.connect() as conn:
        conn.execute(table.insert().values(value=value))
        conn.commit()
        val = conn.execute(sa.select(table.c.value)).scalar_one()
    return val  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# normalize_decimal()
# ---------------------------------------------------------------------------


class TestNormalizeDecimal:
    def test_fewer_than_10_places_padded(self):
        assert normalize_decimal(Decimal("10.5")) == Decimal("10.5000000000")

    def test_trailing_zeros_in_input_normalized_to_10(self):
        assert normalize_decimal(Decimal("35200.0")) == Decimal("35200.0000000000")

    def test_exactly_10_places_unchanged(self):
        d = Decimal("1.2345678901")
        assert normalize_decimal(d) == d

    def test_more_than_10_places_round_half_even_down(self):
        # 11th digit = 5, 10th digit = 2 (even) → banker's round → keep 2
        assert normalize_decimal(Decimal("1.00000000025")) == Decimal("1.0000000002")

    def test_more_than_10_places_round_half_even_up(self):
        # 11th digit = 5, 10th digit = 3 (odd) → banker's round → up to 4
        assert normalize_decimal(Decimal("1.00000000035")) == Decimal("1.0000000004")

    def test_problematic_real_value_normalizes_correctly(self):
        # The exact value that caused the idempotency bug.
        # float64 conversion of "10753491.66553520" produced "10753491.6655352004"
        # — the error is in the 10th decimal place (4 instead of 0).
        # normalize_decimal pads to 10 places but cannot fix a value that
        # already has 10 decimal digits with a corrupt trailing digit.
        # The fix is TEXT storage (ExactDecimal) which prevents the float64
        # conversion entirely: the stored string is always the normalized incoming.
        incoming = normalize_decimal(Decimal("10753491.66553520"))
        assert incoming == Decimal("10753491.6655352000")
        # The float64-corrupted value is detectably DIFFERENT at the 10th decimal:
        corrupted = Decimal("10753491.6655352004")
        assert corrupted != incoming  # 4 ≠ 0 at the 10th decimal place
        # With TEXT storage, the value is stored as "10753491.6655352000" and
        # round-trips exactly — see TestExactDecimalRoundTrip for the end-to-end proof.

    def test_zero(self):
        assert normalize_decimal(Decimal("0")) == Decimal("0.0000000000")

    def test_negative(self):
        assert normalize_decimal(Decimal("-1.5")) == Decimal("-1.5000000000")

    def test_quantum_constant(self):
        assert DECIMAL_QUANTUM == Decimal("0.0000000001")


# ---------------------------------------------------------------------------
# ExactDecimal.process_bind_param (unit tests using SQLite dialect directly)
# ---------------------------------------------------------------------------


class TestExactDecimalBindParam:
    def test_rejects_float(self):
        t = ExactDecimal()
        with pytest.raises(TypeError, match="float"):
            t.process_bind_param(1.5, _SQLITE_DIALECT)

    def test_rejects_float_zero(self):
        t = ExactDecimal()
        with pytest.raises(TypeError, match="float"):
            t.process_bind_param(0.0, _SQLITE_DIALECT)

    def test_accepts_decimal(self):
        t = ExactDecimal()
        result = t.process_bind_param(Decimal("100.123"), _SQLITE_DIALECT)
        assert result == "100.1230000000"

    def test_accepts_numeric_string(self):
        t = ExactDecimal()
        result = t.process_bind_param("99.99", _SQLITE_DIALECT)
        assert result == "99.9900000000"

    def test_none_passthrough(self):
        t = ExactDecimal()
        assert t.process_bind_param(None, _SQLITE_DIALECT) is None

    def test_no_scientific_notation_small_value(self):
        t = ExactDecimal()
        result = t.process_bind_param(Decimal("0.0000000001"), _SQLITE_DIALECT)
        assert "e" not in result.lower()
        assert result == "0.0000000001"

    def test_no_scientific_notation_large_value(self):
        t = ExactDecimal()
        result = t.process_bind_param(Decimal("10753491.66553520"), _SQLITE_DIALECT)
        assert "e" not in result.lower()
        assert result == "10753491.6655352000"

    def test_output_is_always_fixed_point_string_on_sqlite(self):
        t = ExactDecimal()
        for raw in ("0", "1", "100.5", "10753491.66553520", "0.0000000001"):
            result = t.process_bind_param(Decimal(raw), _SQLITE_DIALECT)
            assert isinstance(result, str)
            assert "e" not in result.lower()
            assert "." in result


# ---------------------------------------------------------------------------
# ExactDecimal round-trip (SQLite TEXT storage via Core API)
# ---------------------------------------------------------------------------


class TestExactDecimalRoundTrip:
    def test_large_decimal_exact_roundtrip(self, tmp_path):
        engine, table = _make_engine_and_table(tmp_path)
        original = Decimal("10753491.66553520")
        val = _insert_and_fetch(engine, table, original)
        assert val == normalize_decimal(original)
        assert type(val) is Decimal
        engine.dispose()

    def test_type_returned_is_always_decimal(self, tmp_path):
        engine, table = _make_engine_and_table(tmp_path)
        val = _insert_and_fetch(engine, table, Decimal("100.0"))
        assert type(val) is Decimal
        engine.dispose()

    def test_trailing_zeros_normalized_on_roundtrip(self, tmp_path):
        engine, table = _make_engine_and_table(tmp_path)
        val = _insert_and_fetch(engine, table, Decimal("35200.0"))
        assert val == Decimal("35200.0000000000")
        engine.dispose()

    def test_zero_value(self, tmp_path):
        engine, table = _make_engine_and_table(tmp_path)
        val = _insert_and_fetch(engine, table, Decimal("0"))
        assert val == Decimal("0.0000000000")
        assert type(val) is Decimal
        engine.dispose()

    def test_negative_value_stored_correctly(self, tmp_path):
        engine, table = _make_engine_and_table(tmp_path)
        val = _insert_and_fetch(engine, table, Decimal("-0.0000000001"))
        assert val == Decimal("-0.0000000001")
        engine.dispose()

    def test_more_than_10_decimals_normalized_on_insert(self, tmp_path):
        engine, table = _make_engine_and_table(tmp_path)
        # 11th digit = 5, 10th digit = 2 (even) → half-even → keep 2
        val = _insert_and_fetch(engine, table, Decimal("1.00000000025"))
        assert val == Decimal("1.0000000002")
        engine.dispose()

    def test_maximum_representable_value(self, tmp_path):
        # 18 integer digits + 10 decimal places = 28 significant digits,
        # which is Python's default decimal context precision.
        # The resulting string "999999999999999999.9999999999" is 29 chars — fits VARCHAR(50).
        engine, table = _make_engine_and_table(tmp_path)
        max_val = Decimal("9" * 18 + "." + "9" * 10)
        val = _insert_and_fetch(engine, table, max_val)
        assert val == max_val
        engine.dispose()

    def test_raw_sqlite_storage_is_text_no_scientific_notation(self, tmp_path):
        engine, table = _make_engine_and_table(tmp_path)
        test_values = [
            Decimal("0.0000000001"),
            Decimal("10753491.66553520"),
            Decimal("35200.0"),
        ]
        with engine.connect() as conn:
            for v in test_values:
                conn.execute(table.insert().values(value=v))
            conn.commit()
            raw_rows = conn.execute(text("SELECT value FROM test_exact")).fetchall()

        for (raw,) in raw_rows:
            assert isinstance(raw, str), f"Expected str in SQLite TEXT column, got {type(raw)}"
            assert "e" not in raw.lower(), f"Scientific notation found in stored value: {raw!r}"

        engine.dispose()

    def test_sqlite_pragma_column_type_is_varchar(self, tmp_path):
        engine, _ = _make_engine_and_table(tmp_path)
        with engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(test_exact)")).fetchall()
        # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
        col_map = {row[1]: row[2] for row in cols}
        assert "VARCHAR" in col_map["value"].upper()
        engine.dispose()

    def test_fresh_connection_returns_exact_decimal(self, tmp_path):
        """After commit, a new connection on the same table object returns Decimal.

        The TypeDecorator lives in the Python table definition, not in the SQLite
        schema — so we must query via the same table object (which carries
        ExactDecimal) rather than a reflected table (which would lose it).
        """
        engine, table = _make_engine_and_table(tmp_path)
        original = Decimal("10753491.66553520")
        with engine.connect() as conn:
            conn.execute(table.insert().values(value=original))
            conn.commit()

        # New connection through the same engine / table — TypeDecorator is applied
        with engine.connect() as fresh_conn:
            val = fresh_conn.execute(sa.select(table.c.value)).scalar_one()
        assert type(val) is Decimal
        assert val == normalize_decimal(original)
        engine.dispose()
