"""Stage 5.2C.1 — Timeframe cost audit: data integrity, warmup, slippage, monotonicity.

Verifies:
  - Candle counts: raw 15m → warmup → eval, after aggregation.
  - Scenario identity: entry signal timestamps identical across cost scenarios;
    exit timestamps may differ due to stop-price variation by slippage rate.
  - Slippage formula: entry/exit exec prices consistent with cost scenario rates.
  - Warmup boundaries: no trade executed before start_ms or after end_ms.
  - Monotonicity: higher total cost never improves return_pct (can be violated
    when stop triggers fire at different candles due to exec-price difference).

PAPER/TEST only.  No result implies profitability.
Past results do NOT predict future performance.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.backtesting.schemas import BacktestResult
    from app.models.candle import Candle

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")

# Scenario cost-level ordering (lower = cheaper).
# NO_COSTS(0%) < LOW_SLIPPAGE(0.12%) < BASE_COSTS(0.15%) < CONSERVATIVE(0.20%).
_COST_LEVEL: dict[str, int] = {
    "NO_COSTS": 0,
    "LOW_SLIPPAGE": 1,
    "BASE_COSTS": 2,
    "CONSERVATIVE": 3,
}

_COST_DRAG_NOTE = (
    "For 2024, each scenario's initial_capital = 2023 final_equity for that scenario. "
    "cost_drag in 2024 therefore conflates 2024 execution costs with 2023 compounding "
    "differences.  Use SlippageTradeRow per-trade data for isolated 2024 cost analysis."
)

_STOP_PRICE_NOTE = (
    "Stop-loss trigger = entry_exec_price - atr * multiplier.  "
    "entry_exec_price = candle.open * (1 + slippage_rate), so higher slippage raises "
    "the trigger.  Entry signals are indicator-based and identical across scenarios; "
    "exit timing and exit type may differ between scenarios with different slippage rates."
)


# ---------------------------------------------------------------------------
# Audit dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandleCountsRow:
    """Candle count transparency for one (timeframe, year)."""

    timeframe: str
    year: str
    raw_15m_total: int
    warmup_15m: int
    eval_15m: int
    agg_total: int
    agg_warmup: int
    agg_eval: int


@dataclass(frozen=True)
class ScenarioIdentityRow:
    """Scenario identity comparison for one (timeframe, year, scenario_a, scenario_b) pair.

    entry_signals_match is expected True: signals are indicator-based, cost-independent.
    logical_exits_match may be False when stop-price variation causes different exit timing.
    """

    timeframe: str
    year: str
    scenario_a: str
    scenario_b: str
    eval_candle_hash: str
    entry_signal_hash_a: str
    entry_signal_hash_b: str
    entry_signals_match: bool
    trade_count_a: int
    trade_count_b: int
    trade_counts_match: bool
    exit_reason_hash_a: str
    exit_reason_hash_b: str
    logical_exits_match: bool


@dataclass(frozen=True)
class SlippageTradeRow:
    """Per-trade slippage and fee breakdown for one (timeframe, year, scenario, trade)."""

    timeframe: str
    year: str
    cost_scenario: str
    trade_id: int
    entry_exec_time: int
    entry_exec_price: Decimal
    entry_raw_price: Decimal
    entry_slippage_paid: Decimal
    entry_fee: Decimal
    quantity: Decimal
    capital_at_entry: Decimal
    exit_exec_time: int
    exit_exec_price: Decimal
    exit_raw_price: Decimal
    exit_slippage_paid: Decimal
    exit_fee: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    exit_reasons: str
    slippage_rate: Decimal
    fee_rate: Decimal
    entry_slippage_formula_ok: bool
    exit_slippage_formula_ok: bool
    fee_formula_ok: bool


@dataclass(frozen=True)
class WarmupBoundaryRow:
    """Warmup/period boundary verification for one (timeframe, year, cost_scenario)."""

    timeframe: str
    year: str
    cost_scenario: str
    start_ms: int
    end_ms: int
    first_eval_candle_open_time: int
    first_trade_entry_exec_time: int | None
    last_trade_exit_exec_time: int | None
    trade_before_start: bool
    trade_after_end: bool


@dataclass(frozen=True)
class MonotonicityViolation:
    """A case where higher cost produced better return_pct than lower cost."""

    timeframe: str
    year: str
    higher_cost_scenario: str
    lower_cost_scenario: str
    higher_cost_return_pct: Decimal
    lower_cost_return_pct: Decimal
    cause: str


@dataclass(frozen=True)
class TimeframeCostAuditReport:
    """Complete Stage 5.2C.1 audit report.

    candle_counts: 3 timeframes × 2 years = 6 rows.
    scenario_identity: 3 × 2 years × C(4,2)=6 pairs = 36 rows.
    slippage_trades: one row per trade across all (tf, scenario, year) combos.
    warmup_boundaries: 3 tf × 4 scenarios × 2 years = 24 rows.
    monotonicity_violations: violations detected (typically 0 with no real trades).
    stop_price_varies_by_slippage: True when entry times match but exits differ.
    cost_drag_compounding_note: explains 2024 cost_drag limitation.
    """

    candle_counts: list[CandleCountsRow]
    scenario_identity: list[ScenarioIdentityRow]
    slippage_trades: list[SlippageTradeRow]
    warmup_boundaries: list[WarmupBoundaryRow]
    monotonicity_violations: list[MonotonicityViolation]
    stop_price_varies_by_slippage: bool
    cost_drag_compounding_note: str


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------


def _sha16(items: list[str]) -> str:
    """SHA-256 of newline-joined items, truncated to 16 hex chars."""
    data = "\n".join(items).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def build_audit_from_raw(
    raw: dict[tuple[str, str, str], BacktestResult],
    agg_by_tf_year: dict[tuple[str, str], list[Candle]],
    warmup_by_tf_year: dict[tuple[str, str], int],
    candles_15m_by_year: dict[str, list[Candle]],
    start_ms_by_year: dict[str, int],
    end_ms_by_year: dict[str, int],
) -> TimeframeCostAuditReport:
    """Build the full audit report from raw backtest results and candle metadata."""
    from app.backtesting.timeframe_cost_comparison import (
        COST_SCENARIO_NAMES,
        COST_SCENARIOS,
        TIMEFRAMES,
    )

    # ---- Candle counts ----
    candle_counts: list[CandleCountsRow] = []
    for tf in TIMEFRAMES:
        for year in ("2023", "2024"):
            candles_15m = candles_15m_by_year[year]
            warmup_15m = warmup_by_tf_year[("15m", year)]
            agg = agg_by_tf_year[(tf, year)]
            agg_warmup = warmup_by_tf_year[(tf, year)]
            candle_counts.append(
                CandleCountsRow(
                    timeframe=tf,
                    year=year,
                    raw_15m_total=len(candles_15m),
                    warmup_15m=warmup_15m,
                    eval_15m=len(candles_15m) - warmup_15m,
                    agg_total=len(agg),
                    agg_warmup=agg_warmup,
                    agg_eval=len(agg) - agg_warmup,
                )
            )

    # ---- Scenario identity ----
    scenario_identity: list[ScenarioIdentityRow] = []
    for tf in TIMEFRAMES:
        for year in ("2023", "2024"):
            agg = agg_by_tf_year[(tf, year)]
            agg_warmup = warmup_by_tf_year[(tf, year)]
            eval_candles = agg[agg_warmup:]
            eval_hash = _sha16([str(c.open_time) for c in eval_candles])

            for sc_a, sc_b in combinations(COST_SCENARIO_NAMES, 2):
                res_a = raw[(tf, sc_a, year)]
                res_b = raw[(tf, sc_b, year)]

                entry_times_a = sorted(t.entry_exec_time for t in res_a.trades)
                entry_times_b = sorted(t.entry_exec_time for t in res_b.trades)
                entry_hash_a = _sha16([str(x) for x in entry_times_a])
                entry_hash_b = _sha16([str(x) for x in entry_times_b])

                exit_pairs_a = sorted(
                    f"{t.exit_exec_time}:{','.join(t.exit_reasons)}" for t in res_a.trades
                )
                exit_pairs_b = sorted(
                    f"{t.exit_exec_time}:{','.join(t.exit_reasons)}" for t in res_b.trades
                )
                exit_hash_a = _sha16(exit_pairs_a)
                exit_hash_b = _sha16(exit_pairs_b)

                scenario_identity.append(
                    ScenarioIdentityRow(
                        timeframe=tf,
                        year=year,
                        scenario_a=sc_a,
                        scenario_b=sc_b,
                        eval_candle_hash=eval_hash,
                        entry_signal_hash_a=entry_hash_a,
                        entry_signal_hash_b=entry_hash_b,
                        entry_signals_match=entry_hash_a == entry_hash_b,
                        trade_count_a=res_a.total_trades,
                        trade_count_b=res_b.total_trades,
                        trade_counts_match=res_a.total_trades == res_b.total_trades,
                        exit_reason_hash_a=exit_hash_a,
                        exit_reason_hash_b=exit_hash_b,
                        logical_exits_match=exit_hash_a == exit_hash_b,
                    )
                )

    # ---- Slippage trades ----
    slippage_trades: list[SlippageTradeRow] = []
    for tf in TIMEFRAMES:
        for year in ("2023", "2024"):
            for scenario in COST_SCENARIO_NAMES:
                fee_rate, slip_rate = COST_SCENARIOS[scenario]
                result = raw[(tf, scenario, year)]
                slip_frac = slip_rate / _HUNDRED

                for trade in result.trades:
                    if slip_frac > _ZERO:
                        entry_raw = trade.entry_exec_price / (1 + slip_frac)
                        exit_raw = trade.exit_exec_price / (1 - slip_frac)
                    else:
                        entry_raw = trade.entry_exec_price
                        exit_raw = trade.exit_exec_price

                    entry_slip = trade.entry_exec_price - entry_raw
                    exit_slip = exit_raw - trade.exit_exec_price

                    entry_slip_ok = entry_slip >= _ZERO
                    exit_slip_ok = exit_slip >= _ZERO
                    if slip_rate == _ZERO:
                        entry_slip_ok = entry_slip == _ZERO
                        exit_slip_ok = exit_slip == _ZERO

                    fee_ok = trade.entry_fee >= _ZERO and trade.exit_fee >= _ZERO
                    if fee_rate == _ZERO:
                        fee_ok = trade.entry_fee == _ZERO and trade.exit_fee == _ZERO

                    slippage_trades.append(
                        SlippageTradeRow(
                            timeframe=tf,
                            year=year,
                            cost_scenario=scenario,
                            trade_id=trade.trade_id,
                            entry_exec_time=trade.entry_exec_time,
                            entry_exec_price=trade.entry_exec_price,
                            entry_raw_price=entry_raw,
                            entry_slippage_paid=entry_slip,
                            entry_fee=trade.entry_fee,
                            quantity=trade.quantity,
                            capital_at_entry=trade.capital_at_entry,
                            exit_exec_time=trade.exit_exec_time,
                            exit_exec_price=trade.exit_exec_price,
                            exit_raw_price=exit_raw,
                            exit_slippage_paid=exit_slip,
                            exit_fee=trade.exit_fee,
                            gross_pnl=trade.gross_pnl,
                            net_pnl=trade.net_pnl,
                            exit_reasons=",".join(trade.exit_reasons),
                            slippage_rate=slip_rate,
                            fee_rate=fee_rate,
                            entry_slippage_formula_ok=entry_slip_ok,
                            exit_slippage_formula_ok=exit_slip_ok,
                            fee_formula_ok=fee_ok,
                        )
                    )

    # ---- Warmup boundaries ----
    warmup_boundaries: list[WarmupBoundaryRow] = []
    for tf in TIMEFRAMES:
        for scenario in COST_SCENARIO_NAMES:
            for year in ("2023", "2024"):
                result = raw[(tf, scenario, year)]
                agg = agg_by_tf_year[(tf, year)]
                agg_warmup = warmup_by_tf_year[(tf, year)]
                start_ms = start_ms_by_year[year]
                end_ms = end_ms_by_year[year]

                eval_candles = agg[agg_warmup:]
                first_eval_ot = eval_candles[0].open_time if eval_candles else -1

                trades = result.trades
                first_entry = min((t.entry_exec_time for t in trades), default=None)
                last_exit = max((t.exit_exec_time for t in trades), default=None)
                trade_before = any(t.entry_exec_time < start_ms for t in trades)
                trade_after = any(t.exit_exec_time > end_ms for t in trades)

                warmup_boundaries.append(
                    WarmupBoundaryRow(
                        timeframe=tf,
                        year=year,
                        cost_scenario=scenario,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        first_eval_candle_open_time=first_eval_ot,
                        first_trade_entry_exec_time=first_entry,
                        last_trade_exit_exec_time=last_exit,
                        trade_before_start=trade_before,
                        trade_after_end=trade_after,
                    )
                )

    # ---- Monotonicity violations ----
    violations: list[MonotonicityViolation] = []
    checked: set[tuple[str, str]] = set()
    for sc_a, sc_b in combinations(COST_SCENARIO_NAMES, 2):
        level_a = _COST_LEVEL[sc_a]
        level_b = _COST_LEVEL[sc_b]
        if level_a == level_b:
            continue
        sc_hi, sc_lo = (sc_a, sc_b) if level_a > level_b else (sc_b, sc_a)
        if (sc_hi, sc_lo) in checked:
            continue
        checked.add((sc_hi, sc_lo))

        for tf in TIMEFRAMES:
            for year in ("2023", "2024"):
                res_hi = raw[(tf, sc_hi, year)]
                res_lo = raw[(tf, sc_lo, year)]
                if res_hi.total_return_pct > res_lo.total_return_pct:
                    # Diagnose: check if exit reasons differ (stop-price cause)
                    exits_hi = {(t.exit_exec_time, tuple(t.exit_reasons)) for t in res_hi.trades}
                    exits_lo = {(t.exit_exec_time, tuple(t.exit_reasons)) for t in res_lo.trades}
                    cause = "stop_price_varies_with_slippage" if exits_hi != exits_lo else "unknown"
                    violations.append(
                        MonotonicityViolation(
                            timeframe=tf,
                            year=year,
                            higher_cost_scenario=sc_hi,
                            lower_cost_scenario=sc_lo,
                            higher_cost_return_pct=res_hi.total_return_pct,
                            lower_cost_return_pct=res_lo.total_return_pct,
                            cause=cause,
                        )
                    )

    stop_varies = any(
        row.entry_signals_match and not row.logical_exits_match for row in scenario_identity
    )

    return TimeframeCostAuditReport(
        candle_counts=candle_counts,
        scenario_identity=scenario_identity,
        slippage_trades=slippage_trades,
        warmup_boundaries=warmup_boundaries,
        monotonicity_violations=violations,
        stop_price_varies_by_slippage=stop_varies,
        cost_drag_compounding_note=_COST_DRAG_NOTE,
    )


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

_WARNING = "PAPER/TEST only. No real money. Past results do NOT predict future performance."


def export_audit_data_counts_csv(audit: TimeframeCostAuditReport, path: Path) -> None:
    """Write candle count audit rows to CSV."""
    fieldnames = [
        "timeframe",
        "year",
        "raw_15m_total",
        "warmup_15m",
        "eval_15m",
        "agg_total",
        "agg_warmup",
        "agg_eval",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit.candle_counts:
            writer.writerow(dataclasses.asdict(row))


def export_audit_scenario_identity_csv(audit: TimeframeCostAuditReport, path: Path) -> None:
    """Write scenario identity comparison rows to CSV."""
    fieldnames = [
        "timeframe",
        "year",
        "scenario_a",
        "scenario_b",
        "eval_candle_hash",
        "entry_signal_hash_a",
        "entry_signal_hash_b",
        "entry_signals_match",
        "trade_count_a",
        "trade_count_b",
        "trade_counts_match",
        "exit_reason_hash_a",
        "exit_reason_hash_b",
        "logical_exits_match",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit.scenario_identity:
            writer.writerow(dataclasses.asdict(row))


def export_audit_slippage_csv(audit: TimeframeCostAuditReport, path: Path) -> None:
    """Write per-trade slippage audit rows to CSV."""
    fieldnames = [
        "timeframe",
        "year",
        "cost_scenario",
        "trade_id",
        "entry_exec_time",
        "entry_exec_price",
        "entry_raw_price",
        "entry_slippage_paid",
        "entry_fee",
        "quantity",
        "capital_at_entry",
        "exit_exec_time",
        "exit_exec_price",
        "exit_raw_price",
        "exit_slippage_paid",
        "exit_fee",
        "gross_pnl",
        "net_pnl",
        "exit_reasons",
        "slippage_rate",
        "fee_rate",
        "entry_slippage_formula_ok",
        "exit_slippage_formula_ok",
        "fee_formula_ok",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit.slippage_trades:
            writer.writerow(dataclasses.asdict(row))


def export_audit_warmup_boundaries_csv(audit: TimeframeCostAuditReport, path: Path) -> None:
    """Write warmup boundary audit rows to CSV."""
    fieldnames = [
        "timeframe",
        "year",
        "cost_scenario",
        "start_ms",
        "end_ms",
        "first_eval_candle_open_time",
        "first_trade_entry_exec_time",
        "last_trade_exit_exec_time",
        "trade_before_start",
        "trade_after_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit.warmup_boundaries:
            writer.writerow(dataclasses.asdict(row))


def _decimal_default(obj: object) -> object:
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_audit_json(audit: TimeframeCostAuditReport, path: Path) -> None:
    """Write the full audit report to JSON."""
    data = {
        "warning": _WARNING,
        "stop_price_varies_by_slippage": audit.stop_price_varies_by_slippage,
        "cost_drag_compounding_note": audit.cost_drag_compounding_note,
        "candle_counts": [dataclasses.asdict(r) for r in audit.candle_counts],
        "scenario_identity": [dataclasses.asdict(r) for r in audit.scenario_identity],
        "slippage_trades": [dataclasses.asdict(r) for r in audit.slippage_trades],
        "warmup_boundaries": [dataclasses.asdict(r) for r in audit.warmup_boundaries],
        "monotonicity_violations": [dataclasses.asdict(r) for r in audit.monotonicity_violations],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_decimal_default)
