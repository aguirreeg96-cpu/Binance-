"""Change paper-trading decimal columns from NUMERIC to VARCHAR(50) for exact SQLite storage.

Revision ID: 003
Revises: 002
Create Date: 2025-06-30 00:00:00.000000

ROOT CAUSE:
  Same issue fixed for `candles` in migration 002: SQLite's NUMERIC type
  affinity silently converts stored decimal strings to IEEE-754 REAL
  (float64), introducing rounding errors for values with more than 15
  significant digits.

  `orders`, `positions`, `trades` and `paper_accounts` were not touched by
  migration 002 because they had no real consumers yet. Stage 6.1 (forward
  paper trading) is their first real consumer, so the same fix is applied
  here before any financial data is written.

FIX:
  Change the financial Decimal columns in `orders`, `positions`, `trades`
  and `paper_accounts` to VARCHAR(50). Values are normalised to exactly 10
  decimal places (ROUND_HALF_EVEN) by ExactDecimal.process_bind_param()
  before every write, so the stored string is always fixed-point with no
  scientific notation.

NOTE ON DOWNGRADE:
  Reverting to NUMERIC in SQLite restores REAL affinity, reintroducing
  precision loss on the next write. Data fidelity is not guaranteed after a
  downgrade.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ORDER_COLS: tuple[str, ...] = (
    "price",
    "quantity",
    "filled_quantity",
    "avg_fill_price",
    "stop_price",
    "commission",
)
_POSITION_COLS: tuple[str, ...] = (
    "entry_price",
    "quantity",
    "stop_loss",
    "take_profit",
    "trailing_stop_price",
    "exit_price",
    "realized_pnl",
)
_TRADE_COLS: tuple[str, ...] = (
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl",
    "commission",
    "net_pnl",
    "planned_stop_loss",
    "planned_take_profit",
)
_PAPER_ACCOUNT_COLS: tuple[str, ...] = (
    "balance",
    "asset_balance",
    "equity",
    "realized_pnl",
    "total_fees_paid",
)

# (table, columns, nullable_columns)
_TABLES: tuple[tuple[str, tuple[str, ...], frozenset[str]], ...] = (
    ("orders", _ORDER_COLS, frozenset({"price", "avg_fill_price", "stop_price"})),
    (
        "positions",
        _POSITION_COLS,
        frozenset({"trailing_stop_price", "exit_price", "realized_pnl"}),
    ),
    ("trades", _TRADE_COLS, frozenset()),
    ("paper_accounts", _PAPER_ACCOUNT_COLS, frozenset()),
)


def upgrade() -> None:
    for table, cols, nullable_cols in _TABLES:
        with op.batch_alter_table(table) as batch_op:
            for col in cols:
                batch_op.alter_column(
                    col,
                    existing_type=sa.Numeric(30, 10),
                    type_=sa.String(50),
                    existing_nullable=col in nullable_cols,
                )


def downgrade() -> None:
    # WARNING: reverting to Numeric(30, 10) on SQLite restores REAL affinity,
    # reintroducing precision loss on the next write.
    for table, cols, nullable_cols in reversed(_TABLES):
        with op.batch_alter_table(table) as batch_op:
            for col in cols:
                batch_op.alter_column(
                    col,
                    existing_type=sa.String(50),
                    type_=sa.Numeric(30, 10),
                    existing_nullable=col in nullable_cols,
                )
