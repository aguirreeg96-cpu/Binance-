"""Stage 5.2C exporters — timeframe × cost robustness report serialisation.

Writes TimeframeCostReport to CSV and JSON.
All Decimal values serialised as strings to prevent precision loss.

PAPER/TEST only. Past results do NOT predict future performance.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.backtesting.timeframe_cost_comparison import (
        TimeframeCostReport,
        TimeframeCostResult,
        TimeframeCostRobustness,
        TimeframeCostYearlySummary,
    )


def _d(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _result_row(r: "TimeframeCostResult") -> dict[str, Any]:
    return {
        "timeframe": r.timeframe,
        "cost_scenario": r.cost_scenario,
        "year": r.year,
        "initial_capital": str(r.initial_capital),
        "final_equity": str(r.final_equity),
        "return_pct": str(r.return_pct),
        "buy_and_hold_return_pct": str(r.buy_and_hold_return_pct),
        "total_trades": r.total_trades,
        "win_rate_pct": _d(r.win_rate_pct),
        "profit_factor": _d(r.profit_factor),
        "max_drawdown_pct": str(r.max_drawdown_pct),
        "total_fees": str(r.total_fees),
        "slippage_cost": str(r.slippage_cost),
        "exposure_pct": str(r.exposure_pct),
        "cost_drag": str(r.cost_drag),
        "avg_trade_duration_candles": _d(r.avg_trade_duration_candles),
        "median_net_pnl": _d(r.median_net_pnl),
        "avg_gross_pnl": _d(r.avg_gross_pnl),
        "avg_net_pnl": _d(r.avg_net_pnl),
        "avg_cost_per_trade": _d(r.avg_cost_per_trade),
    }


def _yearly_row(s: "TimeframeCostYearlySummary") -> dict[str, Any]:
    return {
        "timeframe": s.timeframe,
        "cost_scenario": s.cost_scenario,
        "return_pct_2023": str(s.return_pct_2023),
        "return_pct_2024": str(s.return_pct_2024),
        "combined_return_pct": str(s.combined_return_pct),
        "compounded_initial_2023": str(s.compounded_initial_2023),
        "compounded_final_2023": str(s.compounded_final_2023),
        "compounded_initial_2024": str(s.compounded_initial_2024),
        "compounded_final_2024": str(s.compounded_final_2024),
        "positive_years": s.positive_years,
        "worst_drawdown_pct": str(s.worst_drawdown_pct),
        "worst_profit_factor": _d(s.worst_profit_factor),
        "total_trades_2023": s.total_trades_2023,
        "total_trades_2024": s.total_trades_2024,
        "total_fees_2023": str(s.total_fees_2023),
        "total_fees_2024": str(s.total_fees_2024),
        "cost_drag_2023": str(s.cost_drag_2023),
        "cost_drag_2024": str(s.cost_drag_2024),
        "buy_and_hold_2023": str(s.buy_and_hold_2023),
        "buy_and_hold_2024": str(s.buy_and_hold_2024),
    }


def _robustness_row(r: "TimeframeCostRobustness") -> dict[str, Any]:
    return {
        "timeframe": r.timeframe,
        "cost_scenario": r.cost_scenario,
        "positive_both_years": r.positive_both_years,
        "profit_factor_above_1_both_years": r.profit_factor_above_1_both_years,
        "depends_on_low_slippage": r.depends_on_low_slippage,
        "fails_with_conservative": r.fails_with_conservative,
        "low_trade_count": r.low_trade_count,
    }


def export_timeframe_cost_csv(report: "TimeframeCostReport", path: Path) -> None:
    """Write all 24 (timeframe × scenario × year) result rows to CSV."""
    if not report.results:
        return
    fieldnames = list(_result_row(report.results[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.results:
            writer.writerow(_result_row(r))


def export_timeframe_cost_json(report: "TimeframeCostReport", path: Path) -> None:
    """Write the full TimeframeCostReport to JSON."""
    data: dict[str, Any] = {
        "warning": "PAPER/TEST only. Past results do NOT predict future performance.",
        "frozen_candidate": "ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY @ 25% allocation",
        "results": [_result_row(r) for r in report.results],
        "yearly_summary": [_yearly_row(s) for s in report.yearly_summary],
        "robustness": [_robustness_row(r) for r in report.robustness],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_yearly_timeframe_csv(report: "TimeframeCostReport", path: Path) -> None:
    """Write the 12 yearly summary rows to CSV."""
    if not report.yearly_summary:
        return
    fieldnames = list(_yearly_row(report.yearly_summary[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in report.yearly_summary:
            writer.writerow(_yearly_row(s))


def export_cost_sensitivity_csv(report: "TimeframeCostReport", path: Path) -> None:
    """Write the 12 robustness flag rows to CSV."""
    if not report.robustness:
        return
    fieldnames = list(_robustness_row(report.robustness[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.robustness:
            writer.writerow(_robustness_row(r))
