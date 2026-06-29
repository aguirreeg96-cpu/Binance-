"""Stage 5.3 exporters — frozen OOS 2025 report serialisation.

Writes FrozenOos2025Report to CSV and JSON.
All Decimal values serialised as strings to prevent precision loss.

PAPER/TEST only. Past results do NOT predict future performance.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.backtesting.frozen_oos_2025 import FrozenOos2025Report
    from app.backtesting.schemas import EquityPoint


def _d(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _decimal_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def export_oos_summary_json(report: FrozenOos2025Report, path: Path) -> None:
    """Export the full OOS summary as JSON.

    File: BTCUSDT_30m_2025_oos_summary.json
    """
    doc: dict[str, Any] = {
        "symbol": report.symbol,
        "trading_timeframe": report.trading_timeframe,
        "oos_year": report.oos_year,
        "start_ms": report.start_ms,
        "end_ms": report.end_ms,
        "initial_capital": str(report.initial_capital),
        "scenarios": [
            {
                "scenario": s.scenario,
                "initial_capital": str(s.initial_capital),
                "final_equity": str(s.final_equity),
                "return_pct": str(s.return_pct),
                "total_trades": s.total_trades,
                "win_rate_pct": _d(s.win_rate_pct),
                "profit_factor": _d(s.profit_factor),
                "max_drawdown_pct": str(s.max_drawdown_pct),
                "total_fees": str(s.total_fees),
                "slippage_cost": str(s.slippage_cost),
                "gross_profit": str(s.gross_profit),
                "gross_loss": str(s.gross_loss),
                "avg_net_pnl": _d(s.avg_net_pnl),
                "median_net_pnl": _d(s.median_net_pnl),
                "exposure_pct": str(s.exposure_pct),
                "avg_trade_duration_candles": _d(s.avg_trade_duration_candles),
                "buy_and_hold_return_pct": str(s.buy_and_hold_return_pct),
                "first_trade_date": s.first_trade_date,
                "last_trade_date": s.last_trade_date,
                "exit_reason_counts": s.exit_reason_counts,
            }
            for s in report.scenarios
        ],
        "historical_reference": report.historical_reference,
        "interpretation": dataclasses.asdict(report.interpretation),
        "audit": {
            "candles_15m_total": report.audit.candles_15m_total,
            "warmup_15m": report.audit.warmup_15m,
            "eval_15m": report.audit.eval_15m,
            "candles_30m_total": report.audit.candles_30m_total,
            "warmup_30m": report.audit.warmup_30m,
            "eval_30m": report.audit.eval_30m,
            "entry_signal_hash": report.audit.entry_signal_hash,
            "warmup_boundary_ok": report.audit.warmup_boundary_ok,
            "end_boundary_ok": report.audit.end_boundary_ok,
            "costs_monotonic": report.audit.costs_monotonic,
            "missing_data_ranges": list(report.audit.missing_data_ranges),
            "violations": list(report.audit.violations),
        },
        "paper_test_disclaimer": (
            "PAPER/TEST only. No real money. Past results do NOT predict future performance."
        ),
    }
    path.write_text(json.dumps(doc, indent=2, default=_decimal_default), encoding="utf-8")


def export_oos_scenarios_csv(report: FrozenOos2025Report, path: Path) -> None:
    """Export per-scenario metrics as CSV (3 rows: NO_COSTS, BASE_COSTS, CONSERVATIVE).

    File: BTCUSDT_30m_2025_oos_scenarios.csv
    """
    fieldnames = [
        "scenario",
        "initial_capital",
        "final_equity",
        "return_pct",
        "total_trades",
        "win_rate_pct",
        "profit_factor",
        "max_drawdown_pct",
        "total_fees",
        "slippage_cost",
        "gross_profit",
        "gross_loss",
        "avg_net_pnl",
        "median_net_pnl",
        "exposure_pct",
        "avg_trade_duration_candles",
        "buy_and_hold_return_pct",
        "first_trade_date",
        "last_trade_date",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in report.scenarios:
            writer.writerow(
                {
                    "scenario": s.scenario,
                    "initial_capital": str(s.initial_capital),
                    "final_equity": str(s.final_equity),
                    "return_pct": str(s.return_pct),
                    "total_trades": s.total_trades,
                    "win_rate_pct": _d(s.win_rate_pct),
                    "profit_factor": _d(s.profit_factor),
                    "max_drawdown_pct": str(s.max_drawdown_pct),
                    "total_fees": str(s.total_fees),
                    "slippage_cost": str(s.slippage_cost),
                    "gross_profit": str(s.gross_profit),
                    "gross_loss": str(s.gross_loss),
                    "avg_net_pnl": _d(s.avg_net_pnl),
                    "median_net_pnl": _d(s.median_net_pnl),
                    "exposure_pct": str(s.exposure_pct),
                    "avg_trade_duration_candles": _d(s.avg_trade_duration_candles),
                    "buy_and_hold_return_pct": str(s.buy_and_hold_return_pct),
                    "first_trade_date": s.first_trade_date,
                    "last_trade_date": s.last_trade_date,
                }
            )


def export_oos_trades_csv(report: FrozenOos2025Report, path: Path) -> None:
    """Export BASE_COSTS trade list as CSV.

    File: BTCUSDT_30m_2025_oos_trades.csv
    """
    fieldnames = [
        "trade_id",
        "entry_signal_time",
        "entry_exec_time",
        "entry_exec_price",
        "entry_fee",
        "quantity",
        "exit_signal_time",
        "exit_exec_time",
        "exit_exec_price",
        "exit_fee",
        "gross_pnl",
        "net_pnl",
        "return_pct",
        "is_forced_close",
        "capital_at_entry",
        "entry_reasons",
        "exit_reasons",
    ]
    base_result = report.raw_results.get("BASE_COSTS")
    trades = base_result.trades if base_result else []

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow(
                {
                    "trade_id": t.trade_id,
                    "entry_signal_time": t.entry_signal_time,
                    "entry_exec_time": t.entry_exec_time,
                    "entry_exec_price": str(t.entry_exec_price),
                    "entry_fee": str(t.entry_fee),
                    "quantity": str(t.quantity),
                    "exit_signal_time": t.exit_signal_time,
                    "exit_exec_time": t.exit_exec_time,
                    "exit_exec_price": str(t.exit_exec_price),
                    "exit_fee": str(t.exit_fee),
                    "gross_pnl": str(t.gross_pnl),
                    "net_pnl": str(t.net_pnl),
                    "return_pct": str(t.return_pct),
                    "is_forced_close": t.is_forced_close,
                    "capital_at_entry": str(t.capital_at_entry),
                    "entry_reasons": "|".join(t.entry_reasons),
                    "exit_reasons": "|".join(t.exit_reasons),
                }
            )


def export_oos_equity_curve_csv(report: FrozenOos2025Report, path: Path) -> None:
    """Export BASE_COSTS equity curve as CSV.

    File: BTCUSDT_30m_2025_oos_equity_curve.csv
    """
    fieldnames = [
        "open_time",
        "close_time",
        "close_price",
        "equity",
        "quote_balance",
        "base_balance",
        "base_value",
        "drawdown_pct",
        "peak_equity",
        "has_open_position",
    ]
    base_result = report.raw_results.get("BASE_COSTS")
    equity_curve: list[EquityPoint] = base_result.equity_curve if base_result else []

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in equity_curve:
            writer.writerow(
                {
                    "open_time": ep.open_time,
                    "close_time": ep.close_time,
                    "close_price": str(ep.close_price),
                    "equity": str(ep.equity),
                    "quote_balance": str(ep.quote_balance),
                    "base_balance": str(ep.base_balance),
                    "base_value": str(ep.base_value),
                    "drawdown_pct": str(ep.drawdown_pct),
                    "peak_equity": str(ep.peak_equity),
                    "has_open_position": ep.has_open_position,
                }
            )


def export_oos_audit_json(report: FrozenOos2025Report, path: Path) -> None:
    """Export the OOS audit data as JSON.

    File: BTCUSDT_30m_2025_oos_audit.json
    """
    doc: dict[str, Any] = {
        "candles_15m_total": report.audit.candles_15m_total,
        "warmup_15m": report.audit.warmup_15m,
        "eval_15m": report.audit.eval_15m,
        "candles_30m_total": report.audit.candles_30m_total,
        "warmup_30m": report.audit.warmup_30m,
        "eval_30m": report.audit.eval_30m,
        "entry_signal_hash": report.audit.entry_signal_hash,
        "warmup_boundary_ok": report.audit.warmup_boundary_ok,
        "end_boundary_ok": report.audit.end_boundary_ok,
        "costs_monotonic": report.audit.costs_monotonic,
        "missing_data_ranges": list(report.audit.missing_data_ranges),
        "violations": list(report.audit.violations),
        "frozen_config": {
            "entry": "ENTRY_V3_ALIGNED_TREND",
            "exit_config": "V2_STOP_ONLY",
            "trading_timeframe": "30m",
            "source_interval": "15m",
            "allocation_pct": "25",
            "use_take_profit": False,
            "use_bearish_crossover_exit": True,
            "maximum_holding_candles": 0,
        },
        "paper_test_disclaimer": (
            "PAPER/TEST only. No real money. Past results do NOT predict future performance."
        ),
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
