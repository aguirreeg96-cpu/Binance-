"""Entry variant comparison exporters.

Writes EntryComparisonReport and EntryMultiPeriodReport to CSV and JSON.
All Decimal values serialized as strings to prevent precision loss.
PAPER/TEST only. Past results do NOT predict future performance.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.backtesting.entry_comparison import (
        EntryComparisonReport,
        EntryMultiPeriodReport,
        EntryVariantResult,
        EntryYearlySummary,
    )


def _d(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _combo_row(r: "EntryVariantResult") -> dict[str, Any]:
    fa = r.filter_analysis
    sc = r.signal_counts
    return {
        "entry_variant": r.entry_variant,
        "exit_config": r.exit_config_name,
        "return_pct": str(r.result.total_return_pct),
        "return_pct_no_costs": str(r.result_nc.total_return_pct),
        "final_equity": str(r.result.final_equity),
        "max_drawdown_pct": str(r.result.max_drawdown_pct),
        "profit_factor": _d(r.result.profit_factor),
        "win_rate_pct": _d(r.result.win_rate_pct),
        "total_trades": r.result.total_trades,
        "total_fees": str(r.result.total_fees),
        "exposure_pct": str(r.result.exposure_pct),
        "sl_exits": r.sl_exits,
        "tp_exits": r.tp_exits,
        "crossover_exits": r.crossover_exits,
        "median_net_pnl": _d(r.median_net_pnl),
        "avg_trade_pnl": _d(r.avg_trade_pnl),
        "signals_detected": sc.signals_detected,
        "buys_executed": sc.buys_executed,
        "blocked_by_filter": sc.blocked_by_filter,
        "entries_eliminated": fa.entries_eliminated,
        "entries_added": fa.entries_added,
        "eliminated_pnl": str(fa.eliminated_pnl),
        "eliminated_winners": fa.eliminated_winners,
        "eliminated_losers": fa.eliminated_losers,
        "return_pct_vs_baseline": str(fa.return_pct_vs_baseline),
        "max_dd_vs_baseline": str(fa.max_dd_vs_baseline),
        "trade_count_vs_baseline": fa.trade_count_vs_baseline,
        "exposure_vs_baseline": str(fa.exposure_vs_baseline),
    }


def _yearly_row(s: "EntryYearlySummary") -> dict[str, Any]:
    return {
        "entry_variant": s.entry_variant,
        "exit_config": s.exit_config_name,
        "return_pct_2023": str(s.return_pct_2023),
        "return_pct_2024": str(s.return_pct_2024),
        "combined_return_pct": str(s.combined_return_pct),
        "capital_2023_end": str(s.capital_2023_end),
        "capital_2024_end": str(s.capital_2024_end),
        "positive_years": s.positive_years,
        "worst_drawdown_pct": str(s.worst_drawdown_pct),
        "total_trades_2023": s.total_trades_2023,
        "total_trades_2024": s.total_trades_2024,
    }


def export_entry_comparison_csv(
    report: "EntryComparisonReport",
    path: Path,
) -> None:
    """Write the 10-combo entry comparison matrix to CSV."""
    if not report.combinations:
        return
    fieldnames = list(_combo_row(report.combinations[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.combinations:
            writer.writerow(_combo_row(r))


def export_entry_comparison_json(
    report: "EntryComparisonReport",
    path: Path,
) -> None:
    """Write the full EntryComparisonReport to JSON."""
    data: dict[str, Any] = {
        "warning": "PAPER/TEST only. Past results do NOT predict future performance.",
        "period": report.period_label,
        "buy_and_hold_return_pct": str(report.bah_return_pct),
        "combinations": [_combo_row(r) for r in report.combinations],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_yearly_entry_comparison_csv(
    multi_report: "EntryMultiPeriodReport",
    path: Path,
) -> None:
    """Write the per-(entry_variant, exit_config) yearly summary to CSV."""
    if not multi_report.yearly_summary:
        return
    fieldnames = list(_yearly_row(multi_report.yearly_summary[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in multi_report.yearly_summary:
            writer.writerow(_yearly_row(s))


def export_filter_analysis_csv(
    report: "EntryComparisonReport",
    path: Path,
) -> None:
    """Write filter analysis rows (entries_eliminated, eliminated_pnl, etc.) to CSV."""
    rows: list[dict[str, Any]] = []
    for r in report.combinations:
        fa = r.filter_analysis
        rows.append(
            {
                "entry_variant": r.entry_variant,
                "exit_config": r.exit_config_name,
                "entries_eliminated": fa.entries_eliminated,
                "entries_added": fa.entries_added,
                "eliminated_pnl": str(fa.eliminated_pnl),
                "eliminated_winners": fa.eliminated_winners,
                "eliminated_losers": fa.eliminated_losers,
                "return_pct_vs_baseline": str(fa.return_pct_vs_baseline),
                "max_dd_vs_baseline": str(fa.max_dd_vs_baseline),
                "trade_count_vs_baseline": fa.trade_count_vs_baseline,
                "exposure_vs_baseline": str(fa.exposure_vs_baseline),
            }
        )
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_signal_counts_csv(
    report: "EntryComparisonReport",
    path: Path,
) -> None:
    """Write signal count statistics to CSV."""
    rows: list[dict[str, Any]] = []
    for r in report.combinations:
        sc = r.signal_counts
        rows.append(
            {
                "entry_variant": r.entry_variant,
                "exit_config": r.exit_config_name,
                "signals_detected": sc.signals_detected,
                "buys_executed": sc.buys_executed,
                "blocked_by_filter": sc.blocked_by_filter,
            }
        )
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
