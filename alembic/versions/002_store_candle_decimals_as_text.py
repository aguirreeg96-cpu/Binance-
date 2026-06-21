"""Change candle decimal columns from NUMERIC to VARCHAR(50) for exact SQLite storage.

Revision ID: 002
Revises: 001
Create Date: 2025-06-21 00:00:00.000000

ROOT CAUSE:
  SQLite's NUMERIC type affinity silently converts stored decimal strings to
  IEEE-754 REAL (float64), introducing rounding errors of ~1e-9 for values
  with more than 15 significant digits.  For example:

    quote_asset_volume = Decimal("10753491.66553520")
    stored as REAL   → 10753491.6655352004  (error ≈ 4e-10)
    read back        → Decimal("10753491.6655352004")

  This caused idempotent re-downloads to falsely classify unchanged candles
  as 'updated' (updated=287 instead of ignored=287 for 288-candle batches).

FIX:
  Change the 8 financial Decimal columns in `candles` to VARCHAR(50).
  VARCHAR has TEXT affinity in SQLite — the exact canonical string is stored
  and returned without any REAL conversion.

  Values are normalised to exactly 10 decimal places (ROUND_HALF_EVEN) by
  ExactDecimal.process_bind_param() before every write, so the stored string
  is always fixed-point (e.g. "10753491.6655352000") with no scientific
  notation.

NOTE ON EXISTING DATA:
  SQLite has already converted any previously stored decimal strings to
  float64 REAL.  The ALTER COLUMN (via batch table recreation) copies those
  values as-is into the new VARCHAR(50) column; the already-degraded float
  representations are stored as text but the lost digits cannot be recovered.

  After applying this migration, re-import any historical data to guarantee
  full precision.  The application's local btc_local.db should be recreated
  from scratch (delete the file and re-run the download).

NOTE ON DOWNGRADE:
  Reverting to NUMERIC in SQLite restores REAL affinity.  On the next write,
  decimal strings will again be converted to float64, reintroducing precision
  loss.  Data fidelity is not guaranteed after a downgrade.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The 8 candle columns that hold Binance market data as Decimal values.
_DECIMAL_COLS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)


def upgrade() -> None:
    with op.batch_alter_table("candles") as batch_op:
        for col in _DECIMAL_COLS:
            batch_op.alter_column(
                col,
                existing_type=sa.Numeric(30, 10),
                type_=sa.String(50),
                existing_nullable=False,
            )


def downgrade() -> None:
    # WARNING: reverting to Numeric(30, 10) on SQLite restores REAL affinity.
    # Existing text values are re-read as float64 on the next write, reintroducing
    # the precision loss that this migration was designed to prevent.
    with op.batch_alter_table("candles") as batch_op:
        for col in _DECIMAL_COLS:
            batch_op.alter_column(
                col,
                existing_type=sa.String(50),
                type_=sa.Numeric(30, 10),
                existing_nullable=False,
            )
