"""Custom SQLAlchemy column types."""

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

from sqlalchemy import Numeric, String
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, TypeEngine

DECIMAL_QUANTUM = Decimal("0.0000000001")  # 10 decimal places — matches Numeric(30, 10) scale


def normalize_decimal(value: Decimal) -> Decimal:
    """Quantize to 10 decimal places with ROUND_HALF_EVEN.

    Applied before every insert, update, and comparison so that the stored
    representation and the incoming KlineData value are always at the same
    scale — eliminating false 'updated' classifications on idempotent re-downloads.
    """
    return value.quantize(DECIMAL_QUANTUM, rounding=ROUND_HALF_EVEN)


class ExactDecimal(TypeDecorator[Decimal]):
    """Dialect-aware exact Decimal storage.

    SQLite: stored as VARCHAR(50) with TEXT affinity.  TEXT affinity means
    SQLite never converts the stored string to REAL (float64), so no
    precision is lost on the round-trip.  The Decimal is serialised to a
    fixed-point canonical string (10 decimal places, no scientific notation)
    and deserialised back to an exact Decimal.

    Other engines (PostgreSQL, MySQL, etc.): stored as Numeric(30, 10),
    which has native exact-decimal support at the database level.

    The Python type exposed by the ORM is always Decimal regardless of
    dialect — callers never need to branch on storage backend.

    Normalisation policy:
      All values are quantized to DECIMAL_QUANTUM = Decimal("0.0000000001")
      (10 decimal places) using ROUND_HALF_EVEN before every write.  The
      same function (normalize_decimal) is used for comparisons in the
      repository, so stored and incoming values are always compared at
      identical scale.

    Float inputs are rejected to prevent silent precision contamination.
    Pass Decimal or a numeric string instead.

    NOTE — SQLite column ordering:
      Because financial columns are TEXT on SQLite, ORDER BY, SUM, AVG,
      MIN, and MAX on those columns use lexicographic / text semantics, not
      numeric.  Always order candles by open_time (BigInteger) and perform
      financial aggregations in Python using Decimal arithmetic.
    """

    impl = Numeric(30, 10)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String(50))
        return dialect.type_descriptor(Numeric(30, 10))

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        if isinstance(value, float):
            raise TypeError(
                "ExactDecimal rejects float input to prevent silent precision loss. "
                f"Received {value!r}. Use Decimal(str(value)) or a quoted literal."
            )
        d = value if isinstance(value, Decimal) else Decimal(str(value))
        normalized = normalize_decimal(d)
        if dialect.name == "sqlite":
            return format(normalized, "f")  # fixed-point; no 'E' notation
        return normalized  # pass Decimal to Numeric's bind processor on other dialects

    def process_result_value(self, value: Any | None, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value  # Numeric (non-SQLite) already returns Decimal
        return Decimal(str(value))  # String (SQLite) returns a str
