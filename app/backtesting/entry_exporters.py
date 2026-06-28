"""Entry variant comparison exporters (Stage 5.2B.1 — audited).

Writes EntryComparisonReport and EntryMultiPeriodReport to CSV and JSON.
All Decimal values serialized as strings to prevent precision loss.

Equity conventions (see entry_comparison.py for full definitions):
  standalone_*  — each year starts from the original initial_capital
  compounded_*  — 2023 final equity → 2024 initial capital (per combo)

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
        # Returns
        "return_pct": str(r.result.total_return_pct),
        "return_pct_no_costs": str(r.result_nc.total_return_pct),
        # Standalone equity (always uses original initial_capital)
        "standalone_initial_capital": str(r.standalone_initial_capital),
        "standalone_final_equity": str(r.standalone_final_equity),
        # Actual result equity (may be compounded for 2024 in multi-period run)
        "compounded_final_equity": str(r.result.final_equity),
        # Risk metrics
        "max_drawdown_pct": str(r.result.max_drawdown_pct),
        "profit_factor": _d(r.result.profit_factor),
        "win_rate_pct": _d(r.result.win_rate_pct),
        "total_trades": r.result.total_trades,
        "total_fees": str(r.result.total_fees),
        "exposure_pct": str(r.result.exposure_pct),
        # Exit breakdown
        "sl_exits": r.sl_exits,
        "tp_exits": r.tp_exits,
        "crossover_exits": r.crossover_exits,
        # Trade-level statistics
        "median_net_pnl": _d(r.median_net_pnl),
        "avg_trade_pnl": _d(r.avg_trade_pnl),
        # Signal pipeline counts
        "baseline_buy_candidates": sc.baseline_buy_candidates,
        "filter_passed_candidates": sc.filter_passed_candidates,
        "filter_rejected_candidates": sc.filter_rejected_candidates,
        "executed_buys": sc.executed_buys,
        "blocked_by_open_position": sc.blocked_by_open_position,
        # Filter analysis vs V1 baseline
        "trades_conserved": fa.trades_conserved,
        "trades_eliminated": fa.trades_eliminated,
        "trades_added": fa.trades_added,
        "pnl_conserved": str(fa.pnl_conserved),
        "pnl_eliminated": str(fa.pnl_eliminated),
        "return_pct_vs_baseline": str(fa.return_pct_vs_baseline),
        "max_dd_vs_baseline": str(fa.max_dd_vs_baseline),
        "trade_count_vs_baseline": fa.trade_count_vs_baseline,
        "exposure_vs_baseline": str(fa.exposure_vs_baseline),
    }


def _yearly_row(s: "EntryYearlySummary") -> dict[str, Any]:
    return {
        "entry_variant": s.entry_variant,
        "exit_config": s.exit_config_name,
        # Per-year returns
        "return_pct_2023": str(s.return_pct_2023),
        "return_pct_2024": str(s.return_pct_2024),
        "combined_return_pct": str(s.combined_return_pct),
        # Standalone equity (both years start from standalone_initial)
        "standalone_initial": str(s.standalone_initial),
        "standalone_final_2023": str(s.standalone_final_2023),
        "standalone_final_2024": str(s.standalone_final_2024),
        # Compounded equity (2023 final → 2024 initial)
        "compounded_initial_2023": str(s.compounded_initial_2023),
        "compounded_final_2023": str(s.compounded_final_2023),
        "compounded_initial_2024": str(s.compounded_initial_2024),
        "compounded_final_2024": str(s.compounded_final_2024),
        # Summary stats
        "positive_years": s.positive_years,
        "worst_drawdown_pct": str(s.worst_drawdown_pct),
        "total_trades_2023": s.total_trades_2023,
        "total_trades_2024": s.total_trades_2024,
        "win_rate_pct_2023": _d(s.win_rate_pct_2023),
        "win_rate_pct_2024": _d(s.win_rate_pct_2024),
        "profit_factor_2023": _d(s.profit_factor_2023),
        "profit_factor_2024": _d(s.profit_factor_2024),
        "total_fees_2023": str(s.total_fees_2023),
        "total_fees_2024": str(s.total_fees_2024),
    }


def export_entry_comparison_csv(
    report: "EntryComparisonReport",
    path: Path,
) -> None:
    """Write the 14-combo entry comparison matrix to CSV."""
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
    """Write filter analysis rows to CSV (signal pipeline + trade-level comparison)."""
    rows: list[dict[str, Any]] = []
    for r in report.combinations:
        fa = r.filter_analysis
        sc = r.signal_counts
        rows.append(
            {
                "entry_variant": r.entry_variant,
                "exit_config": r.exit_config_name,
                "baseline_candidates": fa.baseline_candidates,
                "passed_candidates": fa.passed_candidates,
                "rejected_candidates": fa.rejected_candidates,
                "executed_buys": fa.executed_buys,
                "blocked_by_open_position": fa.blocked_by_open_position,
                "trades_conserved": fa.trades_conserved,
                "trades_eliminated": fa.trades_eliminated,
                "trades_added": fa.trades_added,
                "pnl_conserved": str(fa.pnl_conserved),
                "pnl_eliminated": str(fa.pnl_eliminated),
                "return_pct_vs_baseline": str(fa.return_pct_vs_baseline),
                "max_dd_vs_baseline": str(fa.max_dd_vs_baseline),
                "trade_count_vs_baseline": fa.trade_count_vs_baseline,
                "exposure_vs_baseline": str(fa.exposure_vs_baseline),
                "invariant_passed_plus_rejected_eq_baseline": (
                    sc.filter_passed_candidates + sc.filter_rejected_candidates
                    == sc.baseline_buy_candidates
                ),
                "invariant_executed_plus_blocked_eq_passed": (
                    sc.executed_buys + sc.blocked_by_open_position == sc.filter_passed_candidates
                ),
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
    """Write signal pipeline statistics to CSV."""
    rows: list[dict[str, Any]] = []
    for r in report.combinations:
        sc = r.signal_counts
        rows.append(
            {
                "entry_variant": r.entry_variant,
                "exit_config": r.exit_config_name,
                "baseline_buy_candidates": sc.baseline_buy_candidates,
                "filter_passed_candidates": sc.filter_passed_candidates,
                "filter_rejected_candidates": sc.filter_rejected_candidates,
                "executed_buys": sc.executed_buys,
                "blocked_by_open_position": sc.blocked_by_open_position,
            }
        )
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
