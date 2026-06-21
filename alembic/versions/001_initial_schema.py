"""Initial schema — all tables from Etapa 1 + 2.

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------ candles
    op.create_table(
        "candles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False),
        sa.Column("open_time", sa.BigInteger(), nullable=False),
        sa.Column("open", sa.Numeric(30, 10), nullable=False),
        sa.Column("high", sa.Numeric(30, 10), nullable=False),
        sa.Column("low", sa.Numeric(30, 10), nullable=False),
        sa.Column("close", sa.Numeric(30, 10), nullable=False),
        sa.Column("volume", sa.Numeric(30, 10), nullable=False),
        sa.Column("close_time", sa.BigInteger(), nullable=False),
        sa.Column("quote_asset_volume", sa.Numeric(30, 10), nullable=False),
        sa.Column("trades", sa.BigInteger(), nullable=False),
        sa.Column("taker_buy_base_volume", sa.Numeric(30, 10), nullable=False),
        sa.Column("taker_buy_quote_volume", sa.Numeric(30, 10), nullable=False),
        sa.Column("is_closed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "interval", "open_time", name="uq_candle"),
    )
    op.create_index(
        "ix_candle_symbol_interval_time",
        "candles",
        ["symbol", "interval", "open_time"],
    )
    op.create_index(
        "ix_candle_symbol_interval",
        "candles",
        ["symbol", "interval"],
    )

    # ------------------------------------------------------------------ signals
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False),
        sa.Column("price", sa.Numeric(30, 10), nullable=False),
        sa.Column("signal_type", sa.String(10), nullable=False),
        sa.Column("indicators", sa.JSON(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("strategy_version", sa.String(50), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_signal_symbol_ts", "signals", ["symbol", "timestamp"])
    op.create_index("ix_signals_timestamp", "signals", ["timestamp"])

    # -------------------------------------------------------------- paper_accounts
    op.create_table(
        "paper_accounts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("balance", sa.Numeric(30, 10), nullable=False),
        sa.Column("asset_balance", sa.Numeric(30, 10), nullable=False),
        sa.Column("equity", sa.Numeric(30, 10), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(30, 10), nullable=False),
        sa.Column("total_fees_paid", sa.Numeric(30, 10), nullable=False),
        sa.Column("currency", sa.String(10), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # --------------------------------------------------------------- positions
    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("entry_price", sa.Numeric(30, 10), nullable=False),
        sa.Column("quantity", sa.Numeric(30, 10), nullable=False),
        sa.Column("stop_loss", sa.Numeric(30, 10), nullable=False),
        sa.Column("take_profit", sa.Numeric(30, 10), nullable=False),
        sa.Column("trailing_stop_enabled", sa.Boolean(), nullable=False),
        sa.Column("trailing_stop_price", sa.Numeric(30, 10), nullable=True),
        sa.Column("exit_price", sa.Numeric(30, 10), nullable=True),
        sa.Column("realized_pnl", sa.Numeric(30, 10), nullable=True),
        sa.Column("opened_at", sa.DateTime(), nullable=False),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("signal_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_position_symbol_status", "positions", ["symbol", "status"])

    # ------------------------------------------------------------------ orders
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("client_order_id", sa.String(64), nullable=False),
        sa.Column("exchange_order_id", sa.String(64), nullable=True),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("order_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("price", sa.Numeric(30, 10), nullable=True),
        sa.Column("quantity", sa.Numeric(30, 10), nullable=False),
        sa.Column("filled_quantity", sa.Numeric(30, 10), nullable=False),
        sa.Column("avg_fill_price", sa.Numeric(30, 10), nullable=True),
        sa.Column("stop_price", sa.Numeric(30, 10), nullable=True),
        sa.Column("commission", sa.Numeric(30, 10), nullable=False),
        sa.Column("trading_mode", sa.String(10), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("rejected_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_order_id"),
    )
    op.create_index("ix_order_symbol_status", "orders", ["symbol", "status"])
    op.create_index("ix_order_client_order_id", "orders", ["client_order_id"])

    # ------------------------------------------------------------------ trades
    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("entry_price", sa.Numeric(30, 10), nullable=False),
        sa.Column("exit_price", sa.Numeric(30, 10), nullable=False),
        sa.Column("quantity", sa.Numeric(30, 10), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("gross_pnl", sa.Numeric(30, 10), nullable=False),
        sa.Column("commission", sa.Numeric(30, 10), nullable=False),
        sa.Column("net_pnl", sa.Numeric(30, 10), nullable=False),
        sa.Column("planned_stop_loss", sa.Numeric(30, 10), nullable=False),
        sa.Column("planned_take_profit", sa.Numeric(30, 10), nullable=False),
        sa.Column("exit_reason", sa.String(50), nullable=False),
        sa.Column("trading_mode", sa.String(10), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("opened_at", sa.DateTime(), nullable=False),
        sa.Column("closed_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trade_symbol_opened", "trades", ["symbol", "opened_at"])

    # --------------------------------------------------------- strategy_configs
    op.create_table(
        "strategy_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("ema_fast", sa.Integer(), nullable=False),
        sa.Column("ema_slow", sa.Integer(), nullable=False),
        sa.Column("ema_trend", sa.Integer(), nullable=False),
        sa.Column("rsi_period", sa.Integer(), nullable=False),
        sa.Column("rsi_min", sa.Numeric(10, 4), nullable=False),
        sa.Column("rsi_max", sa.Numeric(10, 4), nullable=False),
        sa.Column("atr_period", sa.Integer(), nullable=False),
        sa.Column("atr_sl_multiplier", sa.Numeric(10, 4), nullable=False),
        sa.Column("rr_ratio", sa.Numeric(10, 4), nullable=False),
        sa.Column("volume_ma_period", sa.Integer(), nullable=False),
        sa.Column("trailing_stop_enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version"),
    )

    # -------------------------------------------------------- daily_risk_states
    op.create_table(
        "daily_risk_states",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("trades_count", sa.Integer(), nullable=False),
        sa.Column("consecutive_losses", sa.Integer(), nullable=False),
        sa.Column("daily_pnl", sa.Numeric(30, 10), nullable=False),
        sa.Column("daily_pnl_pct", sa.Numeric(10, 6), nullable=False),
        sa.Column("starting_equity", sa.Numeric(30, 10), nullable=False),
        sa.Column("kill_switch_active", sa.Boolean(), nullable=False),
        sa.Column("kill_switch_reason", sa.String(255), nullable=False),
        sa.Column("trading_paused", sa.Boolean(), nullable=False),
        sa.Column("pause_reason", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )
    op.create_index("ix_daily_risk_state_date", "daily_risk_states", ["date"])

    # --------------------------------------------------------------- system_events
    op.create_table(
        "system_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("level", sa.String(10), nullable=False),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("trading_mode", sa.String(10), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_system_event_level_ts", "system_events", ["level", "timestamp"])
    op.create_index("ix_system_events_timestamp", "system_events", ["timestamp"])


def downgrade() -> None:
    op.drop_table("system_events")
    op.drop_table("daily_risk_states")
    op.drop_table("strategy_configs")
    op.drop_table("trades")
    op.drop_table("orders")
    op.drop_table("positions")
    op.drop_table("paper_accounts")
    op.drop_table("signals")
    op.drop_table("candles")
