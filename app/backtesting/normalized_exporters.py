"""Normalized comparison exporters.

Writes NormalizedComparisonReport and MultiPeriodReport to CSV and JSON.
All Decimal values serialized as strings to prevent precision loss.
PAPER/TEST only. Past results do NOT predict future performance.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.backtesting.normalized_comparison import (
        MultiPeriodReport,
        NormalizedComparisonReport,
        NormalizedVariantResult,
        YearlyVariantSummary,
    )


def _d(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _normalized_row(r: "NormalizedVariantResult") -> dict[str, Any]:
    return {
        "variant": r.variant_name,
        "allocation_pct": str(r.allocation_pct),
        "return_pct": str(r.return_pct),
        "return_pct_no_costs": str(r.return_pct_no_costs),
        "final_equity": str(r.final_equity),
        "max_drawdown_pct": str(r.max_drawdown_pct),
        "profit_factor": _d(r.profit_factor),
        "win_rate_pct": _d(r.win_rate_pct),
        "total_fees": str(r.total_fees),
        "slippage_cost": str(r.slippage_cost),
        "total_trades": r.total_trades,
        "avg_trade_pnl": _d(r.avg_trade_pnl),
        "median_trade_pnl": _d(r.median_trade_pnl),
        "exposure_pct": str(r.exposure_pct),
    }


def _yearly_row(s: "YearlyVariantSummary") -> dict[str, Any]:
    return {
        "variant": s.variant_name,
        "allocation_pct": str(s.allocation_pct),
        "return_pct_2023": str(s.return_pct_2023),
        "return_pct_2024": str(s.return_pct_2024),
        "combined_return_pct": str(s.combined_return_pct),
        "positive_years": s.positive_years,
        "worst_drawdown_pct": str(s.worst_drawdown_pct),
        "profit_factor_stability": _d(s.profit_factor_stability),
        "trade_count_2023": s.trade_count_2023,
        "trade_count_2024": s.trade_count_2024,
    }


def export_normalized_comparison_csv(
    report: "NormalizedComparisonReport",
    path: Path,
) -> None:
    """Write the 12-row allocation matrix to CSV."""
    if not report.matrix:
        return
    fieldnames = list(_normalized_row(report.matrix[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.matrix:
            writer.writerow(_normalized_row(r))


def export_normalized_comparison_json(
    report: "NormalizedComparisonReport",
    path: Path,
) -> None:
    """Write the full NormalizedComparisonReport to JSON."""
    data: dict[str, Any] = {
        "warning": "PAPER/TEST only. Past results do NOT predict future performance.",
        "period": report.period_label,
        "buy_and_hold_return_pct": str(report.bah_return_pct),
        "matrix": [_normalized_row(r) for r in report.matrix],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_yearly_comparison_csv(
    multi_report: "MultiPeriodReport",
    path: Path,
) -> None:
    """Write the per-(variant, allocation) yearly summary to CSV."""
    if not multi_report.yearly_summary:
        return
    fieldnames = list(_yearly_row(multi_report.yearly_summary[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in multi_report.yearly_summary:
            writer.writerow(_yearly_row(s))
