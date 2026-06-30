"""Stage 6.1 exporters — forward paper-trading report serialisation.

All Decimal values are serialised as strings to prevent precision loss.
Reads already-queried lists of model rows; performs no DB or network I/O
itself, and sends no orders anywhere.

PAPER/TEST only. No real money. Past results do NOT predict future
performance.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.forward.manifest import (
    FUTURE_EVALUATION_CRITERIA,
    HISTORICAL_DOCUMENTATION,
    build_frozen_manifest,
)

if TYPE_CHECKING:
    from app.models.forward_paper import ForwardLaunch, ForwardSignalEvaluation
    from app.models.order import Order
    from app.models.system_event import SystemEvent
    from app.models.trade import Trade

_DISCLAIMER = (
    "PAPER/TEST only. No real money was used or is at risk. "
    "Past results do NOT predict future performance and are not investment advice."
)


def _d(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _decimal_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def export_forward_signals_csv(evaluations: list[ForwardSignalEvaluation], path: Path) -> None:
    """CSV: one row per ForwardSignalEvaluation, in candle_close_time order."""
    fieldnames = [
        "candle_close_time",
        "evaluated_at",
        "signal",
        "reasons",
        "raw_market_price",
        "planned_execution_time",
        "entry_donchian_level",
        "exit_donchian_level",
        "ema_200",
        "ema_slope",
        "atr",
        "initial_stop",
        "current_trailing_stop",
        "highest_high_since_entry",
        "position_quantity",
        "cash",
        "equity",
        "frozen_config_hash",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in evaluations:
            writer.writerow(
                {
                    "candle_close_time": e.candle_close_time.isoformat(),
                    "evaluated_at": e.evaluated_at.isoformat(),
                    "signal": e.signal,
                    "reasons": "|".join(e.reasons),
                    "raw_market_price": _d(e.raw_market_price),
                    "planned_execution_time": (
                        e.planned_execution_time.isoformat() if e.planned_execution_time else None
                    ),
                    "entry_donchian_level": _d(e.entry_donchian_level),
                    "exit_donchian_level": _d(e.exit_donchian_level),
                    "ema_200": _d(e.ema_200),
                    "ema_slope": _d(e.ema_slope),
                    "atr": _d(e.atr),
                    "initial_stop": _d(e.initial_stop),
                    "current_trailing_stop": _d(e.current_trailing_stop),
                    "highest_high_since_entry": _d(e.highest_high_since_entry),
                    "position_quantity": _d(e.position_quantity),
                    "cash": str(e.cash),
                    "equity": str(e.equity),
                    "frozen_config_hash": e.frozen_config_hash,
                }
            )


def export_forward_orders_csv(orders: list[Order], path: Path) -> None:
    """CSV: one row per Order created by the forward engine."""
    fieldnames = [
        "id",
        "client_order_id",
        "side",
        "order_type",
        "status",
        "price",
        "quantity",
        "filled_quantity",
        "avg_fill_price",
        "stop_price",
        "commission",
        "position_id",
        "created_at",
        "updated_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for o in orders:
            writer.writerow(
                {
                    "id": o.id,
                    "client_order_id": o.client_order_id,
                    "side": o.side,
                    "order_type": o.order_type,
                    "status": o.status,
                    "price": _d(o.price),
                    "quantity": str(o.quantity),
                    "filled_quantity": str(o.filled_quantity),
                    "avg_fill_price": _d(o.avg_fill_price),
                    "stop_price": _d(o.stop_price),
                    "commission": str(o.commission),
                    "position_id": o.position_id,
                    "created_at": o.created_at.isoformat(),
                    "updated_at": o.updated_at.isoformat(),
                }
            )


def export_forward_trades_csv(trades: list[Trade], path: Path) -> None:
    """CSV: one row per closed Trade produced by the forward engine."""
    fieldnames = [
        "id",
        "entry_price",
        "exit_price",
        "quantity",
        "side",
        "gross_pnl",
        "commission",
        "net_pnl",
        "planned_stop_loss",
        "planned_take_profit",
        "exit_reason",
        "opened_at",
        "closed_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow(
                {
                    "id": t.id,
                    "entry_price": str(t.entry_price),
                    "exit_price": str(t.exit_price),
                    "quantity": str(t.quantity),
                    "side": t.side,
                    "gross_pnl": str(t.gross_pnl),
                    "commission": str(t.commission),
                    "net_pnl": str(t.net_pnl),
                    "planned_stop_loss": str(t.planned_stop_loss),
                    "planned_take_profit": str(t.planned_take_profit),
                    "exit_reason": t.exit_reason,
                    "opened_at": t.opened_at.isoformat(),
                    "closed_at": t.closed_at.isoformat(),
                }
            )


def export_forward_equity_curve_csv(evaluations: list[ForwardSignalEvaluation], path: Path) -> None:
    """CSV: equity/cash snapshot per evaluated candle, derived from evaluations.

    The forward engine does not persist a separate equity-curve table —
    cash and equity are recorded on every ForwardSignalEvaluation, which is
    the single source of truth for this report.
    """
    fieldnames = [
        "candle_close_time",
        "cash",
        "equity",
        "position_quantity",
        "signal",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in evaluations:
            writer.writerow(
                {
                    "candle_close_time": e.candle_close_time.isoformat(),
                    "cash": str(e.cash),
                    "equity": str(e.equity),
                    "position_quantity": _d(e.position_quantity),
                    "signal": e.signal,
                }
            )


def export_forward_system_events_csv(events: list[SystemEvent], path: Path) -> None:
    """CSV: one row per SystemEvent logged by the forward engine for this launch."""
    fieldnames = ["timestamp", "level", "source", "message", "details", "trading_mode"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow(
                {
                    "timestamp": ev.timestamp.isoformat(),
                    "level": ev.level,
                    "source": ev.source,
                    "message": ev.message,
                    "details": json.dumps(ev.details) if ev.details is not None else None,
                    "trading_mode": ev.trading_mode,
                }
            )


def export_forward_manifest_json(launch: ForwardLaunch, path: Path) -> None:
    """JSON: the frozen B_4h manifest plus this launch's identity and evaluation criteria.

    Documents the historical (pre-forward) backtest results for context
    only — they are never consumed by engine logic and never used to
    justify changing the frozen configuration.
    """
    doc: dict[str, Any] = {
        "launch_id": launch.id,
        "strategy_name": launch.strategy_name,
        "strategy_version": launch.strategy_version,
        "symbol": launch.symbol,
        "launch_timestamp": launch.launch_timestamp.isoformat(),
        "initial_capital": str(launch.initial_capital),
        "status": launch.status,
        "frozen_config_hash": launch.frozen_config_hash,
        "code_commit_hash": launch.code_commit_hash,
        "last_evaluated_candle_close": (
            launch.last_evaluated_candle_close.isoformat()
            if launch.last_evaluated_candle_close
            else None
        ),
        "frozen_configuration": build_frozen_manifest(),
        "historical_backtest_documentation": HISTORICAL_DOCUMENTATION,
        "future_evaluation_criteria": FUTURE_EVALUATION_CRITERIA,
        "paper_test_disclaimer": _DISCLAIMER,
    }
    path.write_text(json.dumps(doc, indent=2, default=_decimal_default), encoding="utf-8")
