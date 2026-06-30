"""Forward paper trading — launch record and per-candle signal evaluations.

Revision ID: 004
Revises: 003
Create Date: 2025-06-30 00:00:00.000000

Adds the two tables backing Etapa 6.1 (forward paper trading of the frozen
B_4h Donchian breakout config):

  forward_launches            — one permanent activation record per strategy
                                 launch; launch_timestamp is written once and
                                 never updated.
  forward_signal_evaluations  — one row per closed-candle evaluation cycle,
                                 unique on (launch_id, candle_close_time) so
                                 restarts cannot duplicate an evaluation.

Also adds a nullable `launch_id` FK column to orders, positions, trades and
paper_accounts so forward-engine recovery queries can be scoped to exactly
one launch instead of relying on the tables being globally single-tenant.

Financial columns use VARCHAR(50) directly (the ExactDecimal SQLite
representation) since these tables have no prior NUMERIC-typed data to
migrate — see migration 002 for the root-cause analysis of why NUMERIC is
unsafe for exact decimal storage on SQLite.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forward_launches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("strategy_version", sa.String(20), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("frozen_config_hash", sa.String(64), nullable=False),
        sa.Column("frozen_config_json", sa.String(), nullable=False),
        sa.Column("code_commit_hash", sa.String(40), nullable=True),
        sa.Column("launch_timestamp", sa.DateTime(), nullable=False),
        sa.Column("initial_capital", sa.String(50), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("last_evaluated_candle_close", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "forward_signal_evaluations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("launch_id", sa.Integer(), nullable=False),
        sa.Column("candle_close_time", sa.DateTime(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(), nullable=False),
        sa.Column("signal", sa.String(20), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("raw_market_price", sa.String(50), nullable=True),
        sa.Column("planned_execution_time", sa.DateTime(), nullable=True),
        sa.Column("entry_donchian_level", sa.String(50), nullable=True),
        sa.Column("exit_donchian_level", sa.String(50), nullable=True),
        sa.Column("ema_200", sa.String(50), nullable=True),
        sa.Column("ema_slope", sa.String(50), nullable=True),
        sa.Column("atr", sa.String(50), nullable=True),
        sa.Column("initial_stop", sa.String(50), nullable=True),
        sa.Column("current_trailing_stop", sa.String(50), nullable=True),
        sa.Column("highest_high_since_entry", sa.String(50), nullable=True),
        sa.Column("position_quantity", sa.String(50), nullable=True),
        sa.Column("cash", sa.String(50), nullable=False),
        sa.Column("equity", sa.String(50), nullable=False),
        sa.Column("frozen_config_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["launch_id"], ["forward_launches.id"]),
        sa.UniqueConstraint("launch_id", "candle_close_time", name="uq_forward_eval_launch_candle"),
    )
    op.create_index(
        "ix_forward_eval_launch_candle",
        "forward_signal_evaluations",
        ["launch_id", "candle_close_time"],
    )

    for table in ("orders", "positions", "trades", "paper_accounts"):
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(sa.Column("launch_id", sa.Integer(), nullable=True))
            batch_op.create_foreign_key(
                f"fk_{table}_launch_id",
                "forward_launches",
                ["launch_id"],
                ["id"],
            )


def downgrade() -> None:
    for table in ("paper_accounts", "trades", "positions", "orders"):
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_constraint(f"fk_{table}_launch_id", type_="foreignkey")
            batch_op.drop_column("launch_id")

    op.drop_table("forward_signal_evaluations")
    op.drop_table("forward_launches")
