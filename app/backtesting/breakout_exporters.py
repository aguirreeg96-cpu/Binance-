"""Stage 6.0 exporters — Donchian breakout family report serialisation.

All Decimal values serialised as strings to prevent precision loss.
PAPER/TEST only. Past results do NOT predict future performance.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.backtesting.breakout_family import BreakoutFamilyReport

_DISCLAIMER = "PAPER/TEST only. No real money. Past results do NOT predict future performance."


def _d(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _decimal_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def export_breakout_yearly_results_csv(report: BreakoutFamilyReport, path: Path) -> None:
    """CSV: one row per (config_id × year × scenario). 12 columns."""
    fieldnames = [
        "config_id",
        "year",
        "scenario",
        "initial_capital",
        "final_equity",
        "return_pct",
        "total_trades",
        "win_rate_pct",
        "profit_factor",
        "max_drawdown_pct",
        "total_fees",
        "buy_and_hold_return_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.yearly_results:
            writer.writerow(
                {
                    "config_id": r.config_id,
                    "year": r.year,
                    "scenario": r.scenario,
                    "initial_capital": str(r.initial_capital),
                    "final_equity": str(r.final_equity),
                    "return_pct": str(r.return_pct),
                    "total_trades": r.total_trades,
                    "win_rate_pct": _d(r.win_rate_pct),
                    "profit_factor": _d(r.profit_factor),
                    "max_drawdown_pct": str(r.max_drawdown_pct),
                    "total_fees": str(r.total_fees),
                    "buy_and_hold_return_pct": str(r.buy_and_hold_return_pct),
                }
            )


def export_breakout_compounded_results_csv(report: BreakoutFamilyReport, path: Path) -> None:
    """CSV: one row per (config_id × scenario) with 5-year compounded metrics."""
    fieldnames = [
        "config_id",
        "scenario",
        "initial_capital",
        "final_equity",
        "total_return_pct",
        "profit_factor",
        "max_drawdown_pct",
        "total_trades",
        "years_with_data",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.compounded_results:
            writer.writerow(
                {
                    "config_id": r.config_id,
                    "scenario": r.scenario,
                    "initial_capital": str(r.initial_capital),
                    "final_equity": str(r.final_equity),
                    "total_return_pct": str(r.total_return_pct),
                    "profit_factor": _d(r.profit_factor),
                    "max_drawdown_pct": str(r.max_drawdown_pct),
                    "total_trades": r.total_trades,
                    "years_with_data": r.years_with_data,
                }
            )


def export_breakout_robustness_summary_csv(report: BreakoutFamilyReport, path: Path) -> None:
    """CSV: one row per config_id with QUALIFIED/REJECTED and check details."""
    fieldnames = [
        "config_id",
        "is_qualified",
        "compounded_base_positive",
        "compounded_conservative_positive",
        "years_positive_ge3",
        "profit_factor_gt1_10",
        "max_drawdown_lt15",
        "min_trades_30",
        "no_year_below_neg10",
        "best_year_le70pct_gross",
        "no_violations",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rob in report.robustness:
            check_map = {c.name: c.passed for c in rob.qualification_checks}
            writer.writerow(
                {
                    "config_id": rob.config_id,
                    "is_qualified": rob.is_qualified,
                    "compounded_base_positive": check_map.get("compounded_base_positive", False),
                    "compounded_conservative_positive": check_map.get(
                        "compounded_conservative_positive", False
                    ),
                    "years_positive_ge3": check_map.get("years_positive_ge3", False),
                    "profit_factor_gt1_10": check_map.get("profit_factor_gt1_10", False),
                    "max_drawdown_lt15": check_map.get("max_drawdown_lt15", False),
                    "min_trades_30": check_map.get("min_trades_30", False),
                    "no_year_below_neg10": check_map.get("no_year_below_neg10", False),
                    "best_year_le70pct_gross": check_map.get("best_year_le70pct_gross", False),
                    "no_violations": check_map.get("no_violations", False),
                }
            )


def export_breakout_trades_csv(report: BreakoutFamilyReport, path: Path) -> None:
    """CSV: BASE_COSTS trades for all config_ids and all years."""
    fieldnames = [
        "config_id",
        "year",
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config_id, yr_map in report.raw_results.items():
            for yr, scenario_map in yr_map.items():
                bc = scenario_map.get("BASE_COSTS")
                if bc is None:
                    continue
                for t in bc.trades:
                    writer.writerow(
                        {
                            "config_id": config_id,
                            "year": yr,
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


def export_breakout_equity_curves_csv(report: BreakoutFamilyReport, path: Path) -> None:
    """CSV: BASE_COSTS equity curves for all config_ids and all years."""
    fieldnames = [
        "config_id",
        "year",
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
        for config_id, yr_map in report.raw_results.items():
            for yr, scenario_map in yr_map.items():
                bc = scenario_map.get("BASE_COSTS")
                if bc is None:
                    continue
                for ep in bc.equity_curve:
                    writer.writerow(
                        {
                            "config_id": config_id,
                            "year": yr,
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


def export_breakout_audit_json(report: BreakoutFamilyReport, path: Path) -> None:
    """JSON: full audit with data coverage and signal hashes."""
    from app.backtesting.breakout_family import BREAKOUT_CONFIGS, BREAKOUT_TIMEFRAMES

    frozen_configs: dict[str, Any] = {}
    for cfg_letter, dcfg in BREAKOUT_CONFIGS.items():
        for tf in BREAKOUT_TIMEFRAMES:
            cfg_id = f"{cfg_letter}_{tf}"
            frozen_configs[cfg_id] = {
                "entry_lookback": dcfg.entry_lookback,
                "exit_lookback": dcfg.exit_lookback,
                "atr_period": dcfg.atr_period,
                "atr_multiplier": str(dcfg.atr_multiplier),
                "ema_period": dcfg.ema_period,
                "ema_slope_lookback": dcfg.ema_slope_lookback,
                "allocation_pct": str(dcfg.allocation_pct),
            }

    doc: dict[str, Any] = {
        "candles_15m_total": report.audit.candles_15m_total,
        "symbol": report.audit.symbol,
        "source_interval": report.audit.source_interval,
        "years_covered": list(report.audit.years_covered),
        "missing_data_years": list(report.audit.missing_data_years),
        "config_signal_hashes": report.audit.config_signal_hashes,
        "violations": list(report.audit.violations),
        "data_coverage": [
            {
                "config_id": dc.config_id,
                "year": dc.year,
                "eval_candles": dc.eval_candles,
            }
            for dc in report.data_coverage
        ],
        "frozen_configs": frozen_configs,
        "paper_test_disclaimer": _DISCLAIMER,
    }
    path.write_text(json.dumps(doc, indent=2, default=_decimal_default), encoding="utf-8")


def export_breakout_report_json(report: BreakoutFamilyReport, path: Path) -> None:
    """JSON: full report summary (no equity curves or full trade lists)."""
    doc: dict[str, Any] = {
        "symbol": report.symbol,
        "source_interval": report.source_interval,
        "evaluation_years": list(report.evaluation_years),
        "initial_capital": str(report.initial_capital),
        "yearly_results": [
            {
                "config_id": r.config_id,
                "year": r.year,
                "scenario": r.scenario,
                "initial_capital": str(r.initial_capital),
                "final_equity": str(r.final_equity),
                "return_pct": str(r.return_pct),
                "total_trades": r.total_trades,
                "win_rate_pct": _d(r.win_rate_pct),
                "profit_factor": _d(r.profit_factor),
                "max_drawdown_pct": str(r.max_drawdown_pct),
                "total_fees": str(r.total_fees),
                "buy_and_hold_return_pct": str(r.buy_and_hold_return_pct),
            }
            for r in report.yearly_results
        ],
        "compounded_results": [
            {
                "config_id": r.config_id,
                "scenario": r.scenario,
                "initial_capital": str(r.initial_capital),
                "final_equity": str(r.final_equity),
                "total_return_pct": str(r.total_return_pct),
                "profit_factor": _d(r.profit_factor),
                "max_drawdown_pct": str(r.max_drawdown_pct),
                "total_trades": r.total_trades,
                "years_with_data": r.years_with_data,
            }
            for r in report.compounded_results
        ],
        "robustness": [
            {
                "config_id": rob.config_id,
                "is_qualified": rob.is_qualified,
                "qualification_checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "value": c.value,
                        "threshold": c.threshold,
                    }
                    for c in rob.qualification_checks
                ],
            }
            for rob in report.robustness
        ],
        "paper_test_disclaimer": _DISCLAIMER,
    }
    path.write_text(json.dumps(doc, indent=2, default=_decimal_default), encoding="utf-8")
