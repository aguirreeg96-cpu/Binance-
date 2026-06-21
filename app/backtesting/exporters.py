"""Exporters — write BacktestResult to JSON or CSV files.

No DB, no network, no real orders.
All Decimal values are serialized as strings to prevent precision loss.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.backtesting.schemas import BacktestResult, BacktestTrade, EquityPoint


def _d(value: Decimal | None) -> str | None:
    """Serialize Decimal to string, or None."""
    return str(value) if value is not None else None


def _trade_to_dict(t: BacktestTrade) -> dict[str, Any]:
    return {
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
    }


def _equity_point_to_dict(ep: EquityPoint) -> dict[str, Any]:
    return {
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


def result_to_dict(result: BacktestResult) -> dict[str, Any]:
    """Convert BacktestResult to a JSON-serializable dict.

    All Decimal values are serialized as strings.
    """
    return {
        "warning": "PAPER/TEST only. Past results do NOT predict future performance.",
        "config": {
            "symbol": result.config.symbol,
            "interval": result.config.interval,
            "start_ms": result.config.start_ms,
            "end_ms": result.config.end_ms,
            "initial_capital": str(result.config.initial_capital),
            "fee_percentage": str(result.config.fee_percentage),
            "slippage_percentage": str(result.config.slippage_percentage),
            "force_close_at_end": result.config.force_close_at_end,
        },
        "summary": {
            "first_candle_open_time": result.first_candle_open_time,
            "last_candle_open_time": result.last_candle_open_time,
            "total_candles": result.total_candles,
            "evaluated_candles": result.evaluated_candles,
            "initial_capital": str(result.initial_capital),
            "final_equity": str(result.final_equity),
            "total_return_pct": str(result.total_return_pct),
            "buy_and_hold_return_pct": str(result.buy_and_hold_return_pct),
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate_pct": _d(result.win_rate_pct),
            "avg_win_pct": _d(result.avg_win_pct),
            "avg_loss_pct": _d(result.avg_loss_pct),
            "profit_factor": _d(result.profit_factor),
            "expectancy_pct": _d(result.expectancy_pct),
            "max_drawdown_pct": str(result.max_drawdown_pct),
            "exposure_pct": str(result.exposure_pct),
            "max_win_streak": result.max_win_streak,
            "max_loss_streak": result.max_loss_streak,
            "total_fees": str(result.total_fees),
            "has_open_position_at_end": result.has_open_position_at_end,
        },
        "trades": [_trade_to_dict(t) for t in result.trades],
        "equity_curve": [_equity_point_to_dict(ep) for ep in result.equity_curve],
    }


def export_json(result: BacktestResult, path: Path) -> None:
    """Write BacktestResult to a JSON file."""
    data = result_to_dict(result)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_equity_csv(result: BacktestResult, path: Path) -> None:
    """Write the equity curve to a CSV file."""
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in result.equity_curve:
            writer.writerow(_equity_point_to_dict(ep))


def export_trades_csv(result: BacktestResult, path: Path) -> None:
    """Write the trades list to a CSV file."""
    if not result.trades:
        return
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
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in result.trades:
            writer.writerow(_trade_to_dict(t))
