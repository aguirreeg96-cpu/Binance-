"""V2 exporters — write strategy variant comparison results to CSV and JSON.

No DB, no network, no real orders.
All Decimal values are serialised as strings to prevent precision loss.

PAPER/TEST only. Past results do NOT predict future performance.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.backtesting.schemas import BacktestTrade

if TYPE_CHECKING:
    from app.backtesting.variants import ComparisonReport, VariantResult


def _d(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _variant_summary(v: "VariantResult") -> dict[str, Any]:
    """Flat dict for one variant (with-costs columns)."""
    r = v.result
    nc = v.result_no_costs
    return {
        "variant": v.name,
        "description": v.description,
        # With costs
        "initial_capital": str(r.initial_capital),
        "final_equity": str(r.final_equity),
        "total_return_pct": str(r.total_return_pct),
        "total_return_pct_no_costs": str(nc.total_return_pct),
        "buy_and_hold_return_pct": str(r.buy_and_hold_return_pct),
        "total_trades": r.total_trades,
        "winning_trades": r.winning_trades,
        "losing_trades": r.losing_trades,
        "win_rate_pct": _d(r.win_rate_pct),
        "win_rate_pct_no_costs": _d(nc.win_rate_pct),
        "profit_factor": _d(r.profit_factor),
        "profit_factor_no_costs": _d(nc.profit_factor),
        "max_drawdown_pct": str(r.max_drawdown_pct),
        "exposure_pct": str(r.exposure_pct),
        "total_fees": str(r.total_fees),
        "avg_win_pct": _d(r.avg_win_pct),
        "avg_loss_pct": _d(r.avg_loss_pct),
        "expectancy_pct": _d(r.expectancy_pct),
        "max_win_streak": r.max_win_streak,
        "max_loss_streak": r.max_loss_streak,
        "has_open_position_at_end": r.has_open_position_at_end,
        # Exit counts
        "sl_exits": v.sl_exits,
        "tp_exits": v.tp_exits,
        "timed_exits": v.timed_exits,
        "crossover_exits": v.crossover_exits,
        "ambiguous_exits": v.ambiguous_exits,
        "forced_exits": v.forced_exits,
        # Duration / distribution
        "avg_duration_candles": _d(v.avg_duration_candles),
        "median_net_pnl": _d(v.median_net_pnl),
    }


def export_variant_comparison_json(report: "ComparisonReport", path: Path) -> None:
    """Write the full ComparisonReport to a JSON file."""
    data: dict[str, Any] = {
        "warning": "PAPER/TEST only. Past results do NOT predict future performance.",
        "buy_and_hold_return_pct": str(report.bah_return_pct),
        "variants": [_variant_summary(v) for v in report.variants],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_variant_comparison_csv(report: "ComparisonReport", path: Path) -> None:
    """Write a flat CSV row per variant for easy spreadsheet comparison."""
    if not report.variants:
        return
    fieldnames = list(_variant_summary(report.variants[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in report.variants:
            writer.writerow(_variant_summary(v))


def _trade_to_v2_dict(t: BacktestTrade, variant_name: str) -> dict[str, Any]:
    return {
        "variant": variant_name,
        "trade_id": t.trade_id,
        "entry_signal_time": t.entry_signal_time,
        "entry_exec_time": t.entry_exec_time,
        "entry_exec_price": str(t.entry_exec_price),
        "entry_fee": str(t.entry_fee),
        "quantity": str(t.quantity),
        "capital_at_entry": str(t.capital_at_entry),
        "exit_signal_time": t.exit_signal_time,
        "exit_exec_time": t.exit_exec_time,
        "exit_exec_price": str(t.exit_exec_price),
        "exit_fee": str(t.exit_fee),
        "gross_pnl": str(t.gross_pnl),
        "net_pnl": str(t.net_pnl),
        "return_pct": str(t.return_pct),
        "is_forced_close": t.is_forced_close,
        "entry_reasons": "|".join(t.entry_reasons),
        "exit_reasons": "|".join(t.exit_reasons),
    }


def export_v2_trades_csv(report: "ComparisonReport", path: Path) -> None:
    """Write all trades from all variants to a single CSV file."""
    all_rows = []
    for v in report.variants:
        for t in v.result.trades:
            all_rows.append(_trade_to_v2_dict(t, v.name))

    if not all_rows:
        return

    fieldnames = list(all_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def _exit_type_rows(report: "ComparisonReport") -> list[dict[str, Any]]:
    """One row per (variant, exit_reason) with counts and P&L."""
    rows = []
    for v in report.variants:
        # Aggregate by primary exit reason
        buckets: dict[str, list[BacktestTrade]] = {}
        for t in v.result.trades:
            primary = t.exit_reasons[0] if t.exit_reasons else "UNKNOWN"
            buckets.setdefault(primary, []).append(t)

        for reason, trades in sorted(buckets.items()):
            wins = [t for t in trades if t.net_pnl > Decimal("0")]
            total_net = sum((t.net_pnl for t in trades), Decimal("0"))
            win_rate = (
                Decimal(str(len(wins))) / Decimal(str(len(trades))) * Decimal("100")
                if trades
                else None
            )
            rows.append(
                {
                    "variant": v.name,
                    "exit_reason": reason,
                    "trade_count": len(trades),
                    "win_count": len(wins),
                    "win_rate_pct": _d(win_rate),
                    "total_net_pnl": str(total_net),
                }
            )
    return rows


def export_exit_type_breakdown_csv(report: "ComparisonReport", path: Path) -> None:
    """Write per-variant exit-reason breakdown to CSV."""
    rows = _exit_type_rows(report)
    if not rows:
        return
    fieldnames = [
        "variant",
        "exit_reason",
        "trade_count",
        "win_count",
        "win_rate_pct",
        "total_net_pnl",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
