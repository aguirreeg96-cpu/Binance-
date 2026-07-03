"""Stage 6.3 — paper heartbeats and paper events tables.

Revision ID: 005
Revises: 004
Create Date: 2026-07-03 00:00:00.000000

Adds two tables:

  paper_heartbeats  — one row per engine cycle; records whether a new candle
                      was evaluated, the PID, and any error message.  Used by
                      the health-check endpoint to distinguish an active process
                      (waiting for the next 4h candle) from a stopped or
                      blocked process.

  paper_events      — one row per business event (BUY_PENDING, LONG_OPENED,
                      etc.).  idempotency_key is UNIQUE so duplicate events
                      are never inserted.  Used by the events API and by the
                      optional Telegram alerting service.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "paper_heartbeats",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp_utc", sa.DateTime(), nullable=False),
        sa.Column("cycle_result", sa.String(20), nullable=False),
        sa.Column("evaluations_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_candle_time", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
        sa.Column("process_pid", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_paper_heartbeat_ts", "paper_heartbeats", ["timestamp_utc"])

    op.create_table(
        "paper_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_type", sa.String(40), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("timestamp_utc", sa.DateTime(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("related_evaluation_id", sa.Integer(), nullable=True),
        sa.Column("related_trade_id", sa.Integer(), nullable=True),
        sa.Column("telegram_sent", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("browser_acked", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("read_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_paper_event_idempotency_key"),
    )
    op.create_index("ix_paper_event_ts", "paper_events", ["timestamp_utc"])


def downgrade() -> None:
    op.drop_index("ix_paper_event_ts", table_name="paper_events")
    op.drop_table("paper_events")
    op.drop_index("ix_paper_heartbeat_ts", table_name="paper_heartbeats")
    op.drop_table("paper_heartbeats")
