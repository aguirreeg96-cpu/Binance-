"""CLI for running backtests.

Usage:
    python -m app.cli.run_backtest \\
        --symbol BTCUSDT \\
        --interval 15m \\
        --start 2024-01-01T00:00:00Z \\
        --end 2025-01-01T00:00:00Z \\
        --initial-capital 10000 \\
        --fee-percentage 0.1 \\
        --slippage-percentage 0.05 \\
        --export-dir ./backtest_results

PAPER/TEST only — no real money, no real orders.
Past results do NOT predict future performance.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.backtesting.diagnostics import BacktestDiagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_WARNING = (
    "\n"
    "╔══════════════════════════════════════════════════════════════╗\n"
    "║       ⚠  PAPER / TEST — NO REAL MONEY — EDUCATIONAL  ⚠      ║\n"
    "║  Past backtest results do NOT predict future performance.    ║\n"
    "╚══════════════════════════════════════════════════════════════╝\n"
)


def _parse_dt(value: str, name: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue
    print(f"ERROR: cannot parse {name}={value!r}. Use ISO 8601 (e.g. 2024-01-01T00:00:00Z).")
    sys.exit(1)


def _parse_decimal(value: str, name: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation:
        print(f"ERROR: {name}={value!r} is not a valid decimal number.")
        sys.exit(1)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.run_backtest",
        description="Run a deterministic backtest on locally stored Binance candles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", required=True, help="Trading pair (e.g. BTCUSDT)")
    p.add_argument("--interval", required=True, help="Candle interval (e.g. 15m, 1h)")
    p.add_argument("--start", required=True, help="Start datetime ISO 8601 UTC (inclusive)")
    p.add_argument("--end", required=True, help="End datetime ISO 8601 UTC (exclusive)")
    p.add_argument("--initial-capital", default="10000", help="Initial quote balance (Decimal)")
    p.add_argument("--fee-percentage", default="0.1", help="Fee per leg in percent (e.g. 0.1)")
    p.add_argument(
        "--slippage-percentage", default="0.05", help="Slippage per leg in percent (e.g. 0.05)"
    )
    p.add_argument(
        "--no-force-close",
        action="store_true",
        help="Do NOT force close open position at end of backtest",
    )
    p.add_argument(
        "--export-dir",
        default=None,
        help="Directory to export results (JSON + CSV). Created if absent.",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Compute extended diagnostics: cost scenarios, MFE/MAE excursions, "
            "entry blockers, monthly/quarterly breakdown. "
            "Exports additional CSV and JSON files when --export-dir is set."
        ),
    )
    return p


def main() -> None:
    print(_WARNING)
    parser = _build_parser()
    args = parser.parse_args()

    start_dt = _parse_dt(args.start, "--start")
    end_dt = _parse_dt(args.end, "--end")
    initial_capital = _parse_decimal(args.initial_capital, "--initial-capital")
    fee_pct = _parse_decimal(args.fee_percentage, "--fee-percentage")
    slip_pct = _parse_decimal(args.slippage_percentage, "--slippage-percentage")

    from app.backtesting.config import BacktestConfig
    from app.backtesting.exceptions import BacktestError, BacktestInsufficientDataError
    from app.backtesting.exporters import export_equity_csv, export_json, export_trades_csv
    from app.backtesting.service import BacktestService
    from app.database import SessionLocal

    try:
        config = BacktestConfig(
            symbol=args.symbol,
            interval=args.interval,
            start_ms=_ms(start_dt),
            end_ms=_ms(end_dt),
            initial_capital=initial_capital,
            fee_percentage=fee_pct,
            slippage_percentage=slip_pct,
            force_close_at_end=not args.no_force_close,
        )
    except Exception as exc:
        print(f"ERROR: Invalid configuration: {exc}")
        sys.exit(1)

    db = SessionLocal()
    try:
        svc = BacktestService(db)
        if args.diagnostics:
            result, all_candles, warmup_len, strategy_engine, ind_config = svc.run_with_context(
                config
            )
        else:
            result = svc.run(config)
            all_candles = []
            warmup_len = 0
            strategy_engine = None
            ind_config = None
    except BacktestInsufficientDataError as exc:
        print(f"ERROR: Insufficient data — {exc}")
        print("Hint: download historical data first with:  python -m app.cli.download_klines")
        sys.exit(1)
    except BacktestError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        db.close()

    # Print summary
    r = result
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS: {config.symbol} {config.interval}")
    print(f"{'=' * 60}")
    print(f"  Period         : {args.start} → {args.end}")
    print(f"  Total candles  : {r.total_candles}")
    print(f"  Evaluated      : {r.evaluated_candles}")
    print(f"  Initial capital: {r.initial_capital}")
    print(f"  Final equity   : {r.final_equity:.8f}")
    print(f"  Total return   : {r.total_return_pct:.4f}%")
    print(f"  Buy & Hold     : {r.buy_and_hold_return_pct:.4f}%")
    print(f"  Total trades   : {r.total_trades}")
    if r.win_rate_pct is not None:
        print(f"  Win rate       : {r.win_rate_pct:.2f}%")
    else:
        print("  Win rate       : N/A")
    if r.profit_factor is not None:
        print(f"  Profit factor  : {r.profit_factor:.4f}")
    else:
        print("  Profit factor  : N/A")
    print(f"  Max drawdown   : {r.max_drawdown_pct:.4f}%")
    print(f"  Exposure       : {r.exposure_pct:.2f}%")
    print(f"  Total fees     : {r.total_fees:.8f}")
    print(f"  Open at end    : {r.has_open_position_at_end}")
    print(f"{'=' * 60}")

    if args.export_dir:
        export_path = Path(args.export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        slug = f"{config.symbol}_{config.interval}_{args.start[:10]}_{args.end[:10]}"

        json_path = export_path / f"{slug}_result.json"
        equity_path = export_path / f"{slug}_equity.csv"
        trades_path = export_path / f"{slug}_trades.csv"

        export_json(result, json_path)
        export_equity_csv(result, equity_path)
        export_trades_csv(result, trades_path)

        print("\nExported:")
        print(f"  JSON   : {json_path}")
        print(f"  Equity : {equity_path}")
        print(f"  Trades : {trades_path}")

    if args.diagnostics and strategy_engine is not None and ind_config is not None:
        from app.backtesting.diagnostics import compute_diagnostics
        from app.backtesting.exporters import (
            export_cost_scenarios_csv,
            export_diagnostics_json,
            export_entry_blockers_csv,
            export_exit_reason_csv,
            export_monthly_csv,
            export_trade_excursions_csv,
        )

        print("\nComputing diagnostics …")
        diag = compute_diagnostics(
            result=result,
            all_candles=all_candles,
            warmup_len=warmup_len,
            strategy_engine=strategy_engine,
            indicator_config=ind_config,
        )
        _print_diagnostics(diag)

        if args.export_dir:
            slug = f"{config.symbol}_{config.interval}_{args.start[:10]}_{args.end[:10]}"
            export_path = Path(args.export_dir)
            diag_json = export_path / f"{slug}_diagnostics.json"
            monthly_csv = export_path / f"{slug}_monthly.csv"
            exit_csv = export_path / f"{slug}_exit_reasons.csv"
            cost_csv = export_path / f"{slug}_cost_scenarios.csv"
            blockers_csv = export_path / f"{slug}_entry_blockers.csv"
            excursions_csv = export_path / f"{slug}_trade_excursions.csv"

            export_diagnostics_json(diag, diag_json)
            export_monthly_csv(diag, monthly_csv)
            export_exit_reason_csv(diag, exit_csv)
            export_cost_scenarios_csv(diag, cost_csv)
            export_entry_blockers_csv(diag, blockers_csv)
            export_trade_excursions_csv(diag, excursions_csv)

            print("\nDiagnostic exports:")
            print(f"  Diagnostics JSON : {diag_json}")
            print(f"  Monthly CSV      : {monthly_csv}")
            print(f"  Exit reasons CSV : {exit_csv}")
            print(f"  Cost scenarios   : {cost_csv}")
            print(f"  Entry blockers   : {blockers_csv}")
            print(f"  Trade excursions : {excursions_csv}")

    print(_WARNING)


def _print_diagnostics(diag: "BacktestDiagnostics") -> None:
    """Print a human-readable diagnostic summary."""
    b = diag.benchmark
    cb = diag.cost_breakdown
    td = diag.trade_distribution
    eb = diag.entry_blockers

    sep = "=" * 60
    print(f"\n{sep}")
    print("  DIAGNOSTICS")
    print(sep)

    print("\n--- Benchmark (Buy & Hold) ---")
    print(f"  Full period B&H      : {b.buy_and_hold_full_period_pct:.4f}%")
    print(f"  Effective period B&H : {b.buy_and_hold_effective_period_pct:.4f}%")

    print("\n--- Cost Breakdown ---")
    costs_pct = cb.costs_as_pct_of_initial_capital
    print(f"  Total costs / initial capital : {costs_pct:.4f}%")
    if cb.costs_as_pct_of_gross_profit is not None:
        print(f"  Total costs / gross profit    : {cb.costs_as_pct_of_gross_profit:.4f}%")
    n_flip = cb.profitable_before_costs_but_losing_after
    print(f"  Profitable before costs, losing after : {n_flip}")
    print(f"  Trades losing before costs    : {cb.trades_losing_before_costs}")
    print(f"  Trades losing after costs     : {cb.trades_losing_after_costs}")

    print("\n--- Cost Scenarios ---")
    for s in diag.cost_scenarios:
        wr = f"{s.win_rate_pct:.2f}%" if s.win_rate_pct is not None else "N/A"
        pf = f"{s.profit_factor:.4f}" if s.profit_factor is not None else "N/A"
        print(
            f"  [{s.name:<20}] return={s.return_pct:.4f}%  "
            f"fees={s.total_fees:.2f}  wr={wr}  pf={pf}"
        )

    print("\n--- Exit Reason Breakdown ---")
    for er in diag.exit_reason_breakdown:
        wr = f"{er.win_rate_pct:.1f}%" if er.win_rate_pct is not None else "N/A"
        print(
            f"  {er.exit_reason:<30} trades={er.trade_count:>4}  "
            f"win_rate={wr}  net_pnl={er.net_pnl:.2f}"
        )

    print("\n--- Monthly P&L ---")
    for m in diag.monthly_breakdown:
        wr = f"{m.win_rate_pct:.1f}%" if m.win_rate_pct is not None else "N/A"
        print(
            f"  {m.year}-{m.month:02d}  trades={m.trade_count:>3}  "
            f"net_pnl={m.net_pnl:.2f}  wr={wr}"
        )

    print("\n--- Trade Distribution ---")
    if td.median_net_pnl is not None:
        p25 = td.p25_net_pnl
        med = td.median_net_pnl
        p75 = td.p75_net_pnl
        print(f"  P25 / Median / P75 net P&L : {p25:.2f} / {med:.2f} / {p75:.2f}")
    if td.best_5_trade_ids:
        print(f"  Best 5 trade IDs  : {list(td.best_5_trade_ids)}")
    if td.worst_5_trade_ids:
        print(f"  Worst 5 trade IDs : {list(td.worst_5_trade_ids)}")

    print("\n--- Entry Blockers (all eval candles, no-position context) ---")
    total = eb.total_evaluated or 1

    def _pct(n: int) -> str:
        return f"{n / total * 100:.1f}%"

    print(f"  Warmup incomplete    : {eb.warmup_incomplete:>6}  ({_pct(eb.warmup_incomplete)})")
    print(
        f"  No bullish crossover : {eb.no_bullish_crossover:>6}  "
        f"({_pct(eb.no_bullish_crossover)})"
    )
    print(
        f"  Price below long EMA : {eb.price_below_long_ema:>6}  "
        f"({_pct(eb.price_below_long_ema)})"
    )
    print(
        f"  RSI outside buy range: {eb.rsi_outside_buy_range:>6}  "
        f"({_pct(eb.rsi_outside_buy_range)})"
    )
    print(f"  Volume insufficient  : {eb.insufficient_volume:>6}  ({_pct(eb.insufficient_volume)})")
    print(f"  Position already open: {eb.position_already_open:>6}")
    print(f"  Total evaluated      : {eb.total_evaluated:>6}")
    print(sep)


if __name__ == "__main__":
    main()
