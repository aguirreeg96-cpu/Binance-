"""Exporters — write BacktestResult and BacktestDiagnostics to JSON or CSV.

No DB, no network, no real orders.
All Decimal values are serialized as strings to prevent precision loss.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.backtesting.schemas import BacktestResult, BacktestTrade, EquityPoint

if TYPE_CHECKING:
    from app.backtesting.diagnostics import BacktestDiagnostics


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
        "entry_reasons": "|".join(t.entry_reasons),
        "exit_reasons": "|".join(t.exit_reasons),
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
        "entry_reasons",
        "exit_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in result.trades:
            writer.writerow(_trade_to_dict(t))


# ---------------------------------------------------------------------------
# Diagnostic exporters
# ---------------------------------------------------------------------------


def export_diagnostics_json(diag: "BacktestDiagnostics", path: Path) -> None:
    """Write BacktestDiagnostics to a JSON file."""

    def _d(v: Decimal | None) -> str | None:
        return str(v) if v is not None else None

    b = diag.benchmark
    cb = diag.cost_breakdown
    td = diag.trade_distribution
    eb = diag.entry_blockers

    data: dict[str, Any] = {
        "warning": "PAPER/TEST only. Past results do NOT predict future performance.",
        "benchmark": {
            "full_period_start_time": b.full_period_start_time,
            "full_period_start_price": str(b.full_period_start_price),
            "effective_start_time": b.effective_start_time,
            "effective_start_price": str(b.effective_start_price),
            "last_candle_time": b.last_candle_time,
            "last_candle_price": str(b.last_candle_price),
            "buy_and_hold_full_period_pct": str(b.buy_and_hold_full_period_pct),
            "buy_and_hold_effective_period_pct": str(b.buy_and_hold_effective_period_pct),
        },
        "cost_breakdown": {
            "gross_profit_before_costs": str(cb.gross_profit_before_costs),
            "gross_loss_before_costs": str(cb.gross_loss_before_costs),
            "net_profit_after_costs": str(cb.net_profit_after_costs),
            "average_gross_trade": _d(cb.average_gross_trade),
            "average_net_trade": _d(cb.average_net_trade),
            "average_entry_fee": _d(cb.average_entry_fee),
            "average_exit_fee": _d(cb.average_exit_fee),
            "average_total_cost_per_trade": _d(cb.average_total_cost_per_trade),
            "costs_as_pct_of_initial_capital": str(cb.costs_as_pct_of_initial_capital),
            "costs_as_pct_of_gross_profit": _d(cb.costs_as_pct_of_gross_profit),
            "profitable_before_costs_but_losing_after": (
                cb.profitable_before_costs_but_losing_after
            ),
            "trades_losing_before_costs": cb.trades_losing_before_costs,
            "trades_losing_after_costs": cb.trades_losing_after_costs,
        },
        "cost_scenarios": [
            {
                "name": s.name,
                "fee_percentage": str(s.fee_percentage),
                "slippage_percentage": str(s.slippage_percentage),
                "final_equity": str(s.final_equity),
                "return_pct": str(s.return_pct),
                "total_fees": str(s.total_fees),
                "estimated_slippage_cost": str(s.estimated_slippage_cost),
                "max_drawdown_pct": str(s.max_drawdown_pct),
                "total_trades": s.total_trades,
                "win_rate_pct": _d(s.win_rate_pct),
                "profit_factor": _d(s.profit_factor),
            }
            for s in diag.cost_scenarios
        ],
        "exit_reason_breakdown": [
            {
                "exit_reason": er.exit_reason,
                "trade_count": er.trade_count,
                "win_count": er.win_count,
                "win_rate_pct": _d(er.win_rate_pct),
                "gross_pnl": str(er.gross_pnl),
                "net_pnl": str(er.net_pnl),
                "avg_net_pnl": _d(er.avg_net_pnl),
                "avg_duration_candles": _d(er.avg_duration_candles),
                "max_loss": str(er.max_loss),
                "max_gain": str(er.max_gain),
            }
            for er in diag.exit_reason_breakdown
        ],
        "monthly_breakdown": [
            {
                "year": m.year,
                "month": m.month,
                "trade_count": m.trade_count,
                "net_pnl": str(m.net_pnl),
                "total_fees": str(m.total_fees),
                "win_count": m.win_count,
                "win_rate_pct": _d(m.win_rate_pct),
            }
            for m in diag.monthly_breakdown
        ],
        "quarterly_breakdown": [
            {
                "year": q.year,
                "quarter": q.quarter,
                "trade_count": q.trade_count,
                "net_pnl": str(q.net_pnl),
                "total_fees": str(q.total_fees),
                "win_count": q.win_count,
                "win_rate_pct": _d(q.win_rate_pct),
            }
            for q in diag.quarterly_breakdown
        ],
        "trade_distribution": {
            "median_net_pnl": _d(td.median_net_pnl),
            "p25_net_pnl": _d(td.p25_net_pnl),
            "p75_net_pnl": _d(td.p75_net_pnl),
            "median_duration_candles": _d(td.median_duration_candles),
            "p25_duration_candles": _d(td.p25_duration_candles),
            "p75_duration_candles": _d(td.p75_duration_candles),
            "best_5_trade_ids": list(td.best_5_trade_ids),
            "worst_5_trade_ids": list(td.worst_5_trade_ids),
            "pct_result_from_top_5": _d(td.pct_result_from_top_5),
            "pct_loss_from_bottom_5": _d(td.pct_loss_from_bottom_5),
        },
        "entry_blockers": {
            "warmup_incomplete": eb.warmup_incomplete,
            "no_bullish_crossover": eb.no_bullish_crossover,
            "price_below_long_ema": eb.price_below_long_ema,
            "rsi_outside_buy_range": eb.rsi_outside_buy_range,
            "insufficient_volume": eb.insufficient_volume,
            "position_already_open": eb.position_already_open,
            "bearish_crossover": eb.bearish_crossover,
            "rsi_overbought": eb.rsi_overbought,
            "price_below_ema_for_sell": eb.price_below_ema_for_sell,
            "total_evaluated": eb.total_evaluated,
        },
        "trade_excursions": [
            {
                "trade_id": ex.trade_id,
                "entry_exec_price": str(ex.entry_exec_price),
                "max_favorable_excursion_pct": str(ex.max_favorable_excursion_pct),
                "max_adverse_excursion_pct": str(ex.max_adverse_excursion_pct),
                "highest_price": str(ex.highest_price),
                "lowest_price": str(ex.lowest_price),
                "candles_until_mfe": ex.candles_until_mfe,
                "candles_until_mae": ex.candles_until_mae,
                "candles_in_trade": ex.candles_in_trade,
            }
            for ex in diag.trade_excursions
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_monthly_csv(diag: "BacktestDiagnostics", path: Path) -> None:
    """Write monthly breakdown to CSV."""
    fieldnames = [
        "year",
        "month",
        "trade_count",
        "net_pnl",
        "total_fees",
        "win_count",
        "win_rate_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in diag.monthly_breakdown:
            writer.writerow(
                {
                    "year": m.year,
                    "month": m.month,
                    "trade_count": m.trade_count,
                    "net_pnl": str(m.net_pnl),
                    "total_fees": str(m.total_fees),
                    "win_count": m.win_count,
                    "win_rate_pct": str(m.win_rate_pct) if m.win_rate_pct is not None else "",
                }
            )


def export_exit_reason_csv(diag: "BacktestDiagnostics", path: Path) -> None:
    """Write exit reason breakdown to CSV."""
    if not diag.exit_reason_breakdown:
        return
    fieldnames = [
        "exit_reason",
        "trade_count",
        "win_count",
        "win_rate_pct",
        "gross_pnl",
        "net_pnl",
        "avg_net_pnl",
        "avg_duration_candles",
        "max_loss",
        "max_gain",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for er in diag.exit_reason_breakdown:
            writer.writerow(
                {
                    "exit_reason": er.exit_reason,
                    "trade_count": er.trade_count,
                    "win_count": er.win_count,
                    "win_rate_pct": str(er.win_rate_pct) if er.win_rate_pct is not None else "",
                    "gross_pnl": str(er.gross_pnl),
                    "net_pnl": str(er.net_pnl),
                    "avg_net_pnl": str(er.avg_net_pnl) if er.avg_net_pnl is not None else "",
                    "avg_duration_candles": (
                        str(er.avg_duration_candles) if er.avg_duration_candles is not None else ""
                    ),
                    "max_loss": str(er.max_loss),
                    "max_gain": str(er.max_gain),
                }
            )


def export_cost_scenarios_csv(diag: "BacktestDiagnostics", path: Path) -> None:
    """Write cost scenario comparison to CSV."""
    fieldnames = [
        "name",
        "fee_percentage",
        "slippage_percentage",
        "final_equity",
        "return_pct",
        "total_fees",
        "estimated_slippage_cost",
        "max_drawdown_pct",
        "total_trades",
        "win_rate_pct",
        "profit_factor",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in diag.cost_scenarios:
            writer.writerow(
                {
                    "name": s.name,
                    "fee_percentage": str(s.fee_percentage),
                    "slippage_percentage": str(s.slippage_percentage),
                    "final_equity": str(s.final_equity),
                    "return_pct": str(s.return_pct),
                    "total_fees": str(s.total_fees),
                    "estimated_slippage_cost": str(s.estimated_slippage_cost),
                    "max_drawdown_pct": str(s.max_drawdown_pct),
                    "total_trades": s.total_trades,
                    "win_rate_pct": str(s.win_rate_pct) if s.win_rate_pct is not None else "",
                    "profit_factor": str(s.profit_factor) if s.profit_factor is not None else "",
                }
            )


def export_entry_blockers_csv(diag: "BacktestDiagnostics", path: Path) -> None:
    """Write entry blocker counts to CSV."""
    eb = diag.entry_blockers
    rows = [
        ("warmup_incomplete", eb.warmup_incomplete),
        ("no_bullish_crossover", eb.no_bullish_crossover),
        ("price_below_long_ema", eb.price_below_long_ema),
        ("rsi_outside_buy_range", eb.rsi_outside_buy_range),
        ("insufficient_volume", eb.insufficient_volume),
        ("position_already_open", eb.position_already_open),
        ("bearish_crossover", eb.bearish_crossover),
        ("rsi_overbought", eb.rsi_overbought),
        ("price_below_ema_for_sell", eb.price_below_ema_for_sell),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "count"])
        writer.writeheader()
        for condition, count in rows:
            writer.writerow({"condition": condition, "count": count})


def export_trade_excursions_csv(diag: "BacktestDiagnostics", path: Path) -> None:
    """Write per-trade MFE/MAE excursions to CSV."""
    if not diag.trade_excursions:
        return
    fieldnames = [
        "trade_id",
        "entry_exec_price",
        "max_favorable_excursion_pct",
        "max_adverse_excursion_pct",
        "highest_price",
        "lowest_price",
        "candles_until_mfe",
        "candles_until_mae",
        "candles_in_trade",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in diag.trade_excursions:
            writer.writerow(
                {
                    "trade_id": ex.trade_id,
                    "entry_exec_price": str(ex.entry_exec_price),
                    "max_favorable_excursion_pct": str(ex.max_favorable_excursion_pct),
                    "max_adverse_excursion_pct": str(ex.max_adverse_excursion_pct),
                    "highest_price": str(ex.highest_price),
                    "lowest_price": str(ex.lowest_price),
                    "candles_until_mfe": ex.candles_until_mfe,
                    "candles_until_mae": ex.candles_until_mae,
                    "candles_in_trade": ex.candles_in_trade,
                }
            )
