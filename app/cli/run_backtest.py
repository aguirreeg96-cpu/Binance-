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
    from app.backtesting.normalized_comparison import MultiPeriodReport
    from app.backtesting.variants import ComparisonReport

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
    p.add_argument(
        "--strategy-version",
        choices=["v1", "v2"],
        default="v1",
        help="Strategy version to run: v1 (bearish-crossover only) or v2 (risk-based exits).",
    )
    p.add_argument(
        "--compare-strategy-variants",
        action="store_true",
        help=(
            "Run all four controlled variants (V1_BASELINE, V2_STOP_ONLY, "
            "V2_STOP_TP, V2_STOP_TP_TIME) and print a comparison table. "
            "Exports strategy_variants.csv/.json, v2_trades.csv, "
            "exit_type_breakdown.csv when --export-dir is set."
        ),
    )
    p.add_argument(
        "--normalized-comparison",
        action="store_true",
        help=(
            "Run all 4 variants at 25%%, 50%% and 100%% allocation for 2023 and 2024 "
            "to produce a fair normalized comparison matrix. "
            "Exports normalized_strategy_comparison.csv/.json and "
            "yearly_strategy_comparison.csv when --export-dir is set."
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

        if args.normalized_comparison:
            # ---- Normalized multi-period comparison mode ----

            _MS_2023_START = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)
            _MS_2024_START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
            _MS_2025_START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
            cfg_2023 = BacktestConfig(
                symbol=args.symbol,
                interval=args.interval,
                start_ms=_MS_2023_START,
                end_ms=_MS_2024_START,
                initial_capital=initial_capital,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                force_close_at_end=not args.no_force_close,
            )
            cfg_2024 = BacktestConfig(
                symbol=args.symbol,
                interval=args.interval,
                start_ms=_MS_2024_START,
                end_ms=_MS_2025_START,
                initial_capital=initial_capital,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                force_close_at_end=not args.no_force_close,
            )
            multi_report = svc.run_multi_period_comparison(cfg_2023, cfg_2024)
            _print_normalized_comparison(multi_report)
            if args.export_dir:
                from app.backtesting.normalized_exporters import (
                    export_normalized_comparison_csv,
                    export_normalized_comparison_json,
                    export_yearly_comparison_csv,
                )

                export_path = Path(args.export_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                slug = f"{args.symbol}_{args.interval}_2023_2024"

                nc_csv = export_path / f"{slug}_normalized_strategy_comparison.csv"
                nc_json = export_path / f"{slug}_normalized_strategy_comparison.json"
                yr_csv = export_path / f"{slug}_yearly_strategy_comparison.csv"

                # Export 2024 matrix as the primary period CSV/JSON
                export_normalized_comparison_csv(multi_report.period_2024, nc_csv)
                export_normalized_comparison_json(multi_report.period_2024, nc_json)
                export_yearly_comparison_csv(multi_report, yr_csv)

                print("\nNormalized exports:")
                print(f"  Normalized CSV  : {nc_csv}")
                print(f"  Normalized JSON : {nc_json}")
                print(f"  Yearly CSV      : {yr_csv}")
            print(_WARNING)
            return

        elif args.compare_strategy_variants:
            # ---- Variant comparison mode ----
            report = svc.run_variants(config)
            result = report.variants[0].result  # V1_BASELINE for the standard summary
            all_candles = []  # type: list
            warmup_len = 0
            strategy_engine = None
            ind_config = None
        elif args.diagnostics:
            result, all_candles, warmup_len, strategy_engine, ind_config = svc.run_with_context(
                config
            )
        elif args.strategy_version == "v2":
            result = svc.run_v2(config)
            all_candles = []
            warmup_len = 0
            strategy_engine = None
            ind_config = None
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

    # Print comparison table if requested
    if args.compare_strategy_variants:
        _print_comparison(report)
        if args.export_dir:
            from app.backtesting.v2_exporters import (
                export_exit_type_breakdown_csv,
                export_v2_trades_csv,
                export_variant_comparison_csv,
                export_variant_comparison_json,
            )

            export_path = Path(args.export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            slug = f"{config.symbol}_{config.interval}_{args.start[:10]}_{args.end[:10]}"

            variants_json = export_path / f"{slug}_strategy_variants.json"
            variants_csv = export_path / f"{slug}_strategy_variants.csv"
            v2_trades_csv = export_path / f"{slug}_v2_trades.csv"
            exit_csv = export_path / f"{slug}_exit_type_breakdown.csv"

            export_variant_comparison_json(report, variants_json)
            export_variant_comparison_csv(report, variants_csv)
            export_v2_trades_csv(report, v2_trades_csv)
            export_exit_type_breakdown_csv(report, exit_csv)

            print("\nVariant exports:")
            print(f"  Variants JSON : {variants_json}")
            print(f"  Variants CSV  : {variants_csv}")
            print(f"  V2 trades CSV : {v2_trades_csv}")
            print(f"  Exit types CSV: {exit_csv}")
        print(_WARNING)
        return

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


def _print_comparison(report: "ComparisonReport") -> None:
    """Print a human-readable strategy variant comparison table."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  STRATEGY VARIANT COMPARISON  (PAPER/TEST only)")
    print(sep)
    print(f"  Buy & Hold benchmark : {report.bah_return_pct:.4f}%")
    print()

    hdr = (
        f"  {'Variant':<22} {'Return':>9} {'NoCost':>9} "
        f"{'Trades':>7} {'WinRate':>8} {'MaxDD':>8} {'Fees':>8}"
    )
    print(hdr)
    print(f"  {'-' * 70}")

    for v in report.variants:
        r = v.result
        nc = v.result_no_costs
        wr = f"{r.win_rate_pct:.1f}%" if r.win_rate_pct is not None else "N/A"
        print(
            f"  {v.name:<22} "
            f"{r.total_return_pct:>8.4f}% "
            f"{nc.total_return_pct:>8.4f}% "
            f"{r.total_trades:>7} "
            f"{wr:>8} "
            f"{r.max_drawdown_pct:>7.4f}% "
            f"{r.total_fees:>8.2f}"
        )

    print(f"\n  {'-' * 70}")
    print("  Exit type breakdown:")
    hdr2 = (
        f"  {'Variant':<22} {'SL':>5} {'TP':>5} {'Time':>5} {'Cross':>6} {'Ambig':>6} {'Force':>6}"
    )
    print(hdr2)
    print(f"  {'-' * 70}")
    for v in report.variants:
        print(
            f"  {v.name:<22} "
            f"{v.sl_exits:>5} "
            f"{v.tp_exits:>5} "
            f"{v.timed_exits:>5} "
            f"{v.crossover_exits:>6} "
            f"{v.ambiguous_exits:>6} "
            f"{v.forced_exits:>6}"
        )

    print(f"\n  {'-' * 70}")
    print("  Duration & distribution:")
    hdr3 = f"  {'Variant':<22} {'AvgDur(c)':>10} {'MedianPnL':>12} {'ProfFact':>10}"
    print(hdr3)
    print(f"  {'-' * 70}")
    for v in report.variants:
        r = v.result
        avg_dur = f"{v.avg_duration_candles:.1f}" if v.avg_duration_candles is not None else "N/A"
        med_pnl = f"{v.median_net_pnl:.2f}" if v.median_net_pnl is not None else "N/A"
        pf = f"{r.profit_factor:.4f}" if r.profit_factor is not None else "N/A"
        print(f"  {v.name:<22} {avg_dur:>10} {med_pnl:>12} {pf:>10}")

    print(sep)


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
            f"  {m.year}-{m.month:02d}  trades={m.trade_count:>3}  net_pnl={m.net_pnl:.2f}  wr={wr}"
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
        f"  No bullish crossover : {eb.no_bullish_crossover:>6}  ({_pct(eb.no_bullish_crossover)})"
    )
    print(
        f"  Price below long EMA : {eb.price_below_long_ema:>6}  ({_pct(eb.price_below_long_ema)})"
    )
    print(
        f"  RSI outside buy range: {eb.rsi_outside_buy_range:>6}  "
        f"({_pct(eb.rsi_outside_buy_range)})"
    )
    print(f"  Volume insufficient  : {eb.insufficient_volume:>6}  ({_pct(eb.insufficient_volume)})")
    print(f"  Position already open: {eb.position_already_open:>6}")
    print(f"  Total evaluated      : {eb.total_evaluated:>6}")
    print(sep)


def _print_normalized_comparison(multi_report: "MultiPeriodReport") -> None:
    """Print normalized strategy comparison matrix and multi-year summary."""
    sep = "=" * 80

    for period_report in (multi_report.period_2023, multi_report.period_2024):
        label = period_report.period_label
        print(f"\n{sep}")
        print(f"  NORMALIZED COMPARISON {label}  (PAPER/TEST only)")
        print(sep)
        print(f"  Buy & Hold {label}: {period_report.bah_return_pct:.4f}%")
        print()

        # --- 25 % focused block ---
        at25 = [r for r in period_report.matrix if r.allocation_pct == 25]
        print("  ── All variants at 25 % allocation ──")
        hdr = (
            f"  {'Variant':<18} {'Return':>8} {'NoCost':>8} {'Equity':>10} "
            f"{'MaxDD':>7} {'WinR':>6} {'Fees':>8} {'Trades':>7} {'Avg':>9} {'Exp':>6}"
        )
        print(hdr)
        print(f"  {'-' * 78}")
        for r in at25:
            wr = f"{r.win_rate_pct:.1f}%" if r.win_rate_pct is not None else "N/A"
            avg = f"{r.avg_trade_pnl:.2f}" if r.avg_trade_pnl is not None else "N/A"
            print(
                f"  {r.variant_name:<18} "
                f"{r.return_pct:>7.4f}% "
                f"{r.return_pct_no_costs:>7.4f}% "
                f"{r.final_equity:>10.2f} "
                f"{r.max_drawdown_pct:>6.4f}% "
                f"{wr:>6} "
                f"{r.total_fees:>8.2f} "
                f"{r.total_trades:>7} "
                f"{avg:>9} "
                f"{r.exposure_pct:>5.1f}%"
            )

        # --- Full 12-row matrix ---
        print()
        print("  ── Full matrix (all variants × all allocations) ──")
        hdr2 = (
            f"  {'Variant':<18} {'Alloc':>6} {'Return':>8} {'NoCost':>8} "
            f"{'Equity':>10} {'MaxDD':>7} {'ProfFact':>9} {'Fees':>8} {'Slippage':>9}"
        )
        print(hdr2)
        print(f"  {'-' * 78}")
        for r in period_report.matrix:
            pf = f"{r.profit_factor:.4f}" if r.profit_factor is not None else "N/A"
            print(
                f"  {r.variant_name:<18} "
                f"{r.allocation_pct:>5.0f}% "
                f"{r.return_pct:>7.4f}% "
                f"{r.return_pct_no_costs:>7.4f}% "
                f"{r.final_equity:>10.2f} "
                f"{r.max_drawdown_pct:>6.4f}% "
                f"{pf:>9} "
                f"{r.total_fees:>8.2f} "
                f"{r.slippage_cost:>9.2f}"
            )

    # --- Multi-year summary ---
    print(f"\n{sep}")
    print("  MULTI-YEAR SUMMARY (2023 + 2024 combined, same parameters)")
    print(sep)
    hdr3 = (
        f"  {'Variant':<18} {'Alloc':>6} {'2023%':>8} {'2024%':>8} "
        f"{'Combined%':>10} {'PosYrs':>7} {'WrstDD':>7} {'PFStab':>8} "
        f"{'Tr23':>5} {'Tr24':>5}"
    )
    print(hdr3)
    print(f"  {'-' * 78}")
    for s in multi_report.yearly_summary:
        pf_stab = (
            f"{s.profit_factor_stability:.4f}" if s.profit_factor_stability is not None else "N/A"
        )
        print(
            f"  {s.variant_name:<18} "
            f"{s.allocation_pct:>5.0f}% "
            f"{s.return_pct_2023:>7.4f}% "
            f"{s.return_pct_2024:>7.4f}% "
            f"{s.combined_return_pct:>9.4f}% "
            f"{s.positive_years:>7} "
            f"{s.worst_drawdown_pct:>6.4f}% "
            f"{pf_stab:>8} "
            f"{s.trade_count_2023:>5} "
            f"{s.trade_count_2024:>5}"
        )
    print(sep)
    print("  No variant is declared optimal or expected to be profitable.")


if __name__ == "__main__":
    main()
