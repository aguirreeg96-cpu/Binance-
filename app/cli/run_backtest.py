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
    from app.backtesting.breakout_family import BreakoutFamilyReport
    from app.backtesting.diagnostics import BacktestDiagnostics
    from app.backtesting.entry_comparison import EntryMultiPeriodReport
    from app.backtesting.frozen_oos_2025 import FrozenOos2025Report
    from app.backtesting.normalized_comparison import MultiPeriodReport
    from app.backtesting.timeframe_cost_comparison import TimeframeCostReport
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
    p.add_argument(
        "--compare-entry-variants",
        action="store_true",
        help=(
            "Run 5 entry variants × 2 exit configs = 10 combos for 2023 and 2024 "
            "with capital compounding to compare entry filter effectiveness. "
            "2025 data is untouched. Exports 5 CSV/JSON files when --export-dir is set."
        ),
    )
    p.add_argument(
        "--compare-timeframes-costs",
        action="store_true",
        help=(
            "Stage 5.2C: run frozen ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY @ 25%% "
            "across 3 timeframes (15m, 30m, 1h) and 4 cost scenarios (NO_COSTS, "
            "BASE_COSTS, LOW_SLIPPAGE, CONSERVATIVE) for 2023 and 2024. "
            "30m and 1h are built from local 15m candles — no new downloads. "
            "Exports timeframe_cost_comparison.csv/.json, "
            "yearly_timeframe_comparison.csv, cost_sensitivity.csv "
            "when --export-dir is set."
        ),
    )
    p.add_argument(
        "--run-frozen-oos-2025",
        action="store_true",
        help=(
            "Stage 5.3: run the frozen candidate (ENTRY_V3_ALIGNED_TREND + "
            "V2_STOP_ONLY @ 25%%) on 2025 out-of-sample data exclusively. "
            "Source: local BTCUSDT 15m candles aggregated to 30m. "
            "Requires --interval 15m and local 2025 candles. "
            "3 scenarios: NO_COSTS, BASE_COSTS, CONSERVATIVE. "
            "Exports BTCUSDT_30m_2025_oos_summary.json, _oos_scenarios.csv, "
            "_oos_trades.csv, _oos_equity_curve.csv, _oos_audit.json "
            "when --export-dir is set. "
            "Do NOT modify strategy parameters after seeing the result."
        ),
    )
    p.add_argument(
        "--compare-breakout-family",
        action="store_true",
        help=(
            "Stage 6.0: evaluate 8 pre-registered Donchian breakout configurations "
            "(A–D × 1h/4h) over 2021–2025 with 3 cost scenarios each. "
            "Requires local BTCUSDT 15m candles for 2021–2025 with warmup before 2021. "
            "Aborts if any year has missing data — no partial results. "
            "Exports BTCUSDT_donchian_yearly.csv, _compounded.csv, "
            "_robustness.csv, _trades.csv, _equity_curves.csv, "
            "_audit.json, _report.json when --export-dir is set. "
            "Do NOT add or modify configurations after seeing the result."
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

        if args.compare_breakout_family:
            # ---- Stage 6.0: Donchian breakout family ----
            bf_report = svc.run_breakout_family(
                symbol=args.symbol,
                initial_capital_str=args.initial_capital,
                force_close_at_end=not args.no_force_close,
            )
            _print_breakout_family(bf_report)
            if args.export_dir:
                from app.backtesting.breakout_exporters import (
                    export_breakout_audit_json,
                    export_breakout_compounded_results_csv,
                    export_breakout_equity_curves_csv,
                    export_breakout_report_json,
                    export_breakout_robustness_summary_csv,
                    export_breakout_trades_csv,
                    export_breakout_yearly_results_csv,
                )

                export_path = Path(args.export_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                slug = f"{args.symbol}_donchian"

                bf_yearly = export_path / f"{slug}_yearly.csv"
                bf_compound = export_path / f"{slug}_compounded.csv"
                bf_robust = export_path / f"{slug}_robustness.csv"
                bf_trades = export_path / f"{slug}_trades.csv"
                bf_equity = export_path / f"{slug}_equity_curves.csv"
                bf_audit = export_path / f"{slug}_audit.json"
                bf_report_json = export_path / f"{slug}_report.json"

                export_breakout_yearly_results_csv(bf_report, bf_yearly)
                export_breakout_compounded_results_csv(bf_report, bf_compound)
                export_breakout_robustness_summary_csv(bf_report, bf_robust)
                export_breakout_trades_csv(bf_report, bf_trades)
                export_breakout_equity_curves_csv(bf_report, bf_equity)
                export_breakout_audit_json(bf_report, bf_audit)
                export_breakout_report_json(bf_report, bf_report_json)

                print("\nBreakout family exports:")
                print(f"  Yearly CSV      : {bf_yearly}")
                print(f"  Compounded CSV  : {bf_compound}")
                print(f"  Robustness CSV  : {bf_robust}")
                print(f"  Trades CSV      : {bf_trades}")
                print(f"  Equity CSV      : {bf_equity}")
                print(f"  Audit JSON      : {bf_audit}")
                print(f"  Report JSON     : {bf_report_json}")
            print(_WARNING)
            return

        elif args.compare_timeframes_costs:
            # ---- Timeframe × cost robustness mode (Stage 5.2C) ----

            _MS_2023_START = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)
            _MS_2024_START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
            _MS_2025_START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
            cfg_tc_23 = BacktestConfig(
                symbol=args.symbol,
                interval="15m",
                start_ms=_MS_2023_START,
                end_ms=_MS_2024_START,
                initial_capital=initial_capital,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                force_close_at_end=not args.no_force_close,
            )
            cfg_tc_24 = BacktestConfig(
                symbol=args.symbol,
                interval="15m",
                start_ms=_MS_2024_START,
                end_ms=_MS_2025_START,
                initial_capital=initial_capital,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                force_close_at_end=not args.no_force_close,
            )
            tc_report, tc_audit = svc.run_timeframe_cost_audit(cfg_tc_23, cfg_tc_24)
            _print_timeframe_cost_comparison(tc_report)
            if args.export_dir:
                from app.backtesting.timeframe_cost_audit import (
                    export_audit_data_counts_csv,
                    export_audit_json,
                    export_audit_scenario_identity_csv,
                    export_audit_slippage_csv,
                    export_audit_warmup_boundaries_csv,
                )
                from app.backtesting.timeframe_cost_exporters import (
                    export_cost_sensitivity_csv,
                    export_timeframe_cost_csv,
                    export_timeframe_cost_json,
                    export_yearly_timeframe_csv,
                )

                export_path = Path(args.export_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                slug = f"{args.symbol}_15m_2023_2024"

                tc_csv = export_path / f"{slug}_timeframe_cost_comparison.csv"
                tc_json = export_path / f"{slug}_timeframe_cost_comparison.json"
                yr_csv = export_path / f"{slug}_yearly_timeframe_comparison.csv"
                cs_csv = export_path / f"{slug}_cost_sensitivity.csv"
                aud_counts = export_path / f"{slug}_audit_data_counts.csv"
                aud_identity = export_path / f"{slug}_audit_scenario_trade_identity.csv"
                aud_slip = export_path / f"{slug}_audit_slippage_by_trade.csv"
                aud_warmup = export_path / f"{slug}_audit_warmup_boundaries.csv"
                aud_json = export_path / f"{slug}_audit_report.json"

                export_timeframe_cost_csv(tc_report, tc_csv)
                export_timeframe_cost_json(tc_report, tc_json)
                export_yearly_timeframe_csv(tc_report, yr_csv)
                export_cost_sensitivity_csv(tc_report, cs_csv)
                export_audit_data_counts_csv(tc_audit, aud_counts)
                export_audit_scenario_identity_csv(tc_audit, aud_identity)
                export_audit_slippage_csv(tc_audit, aud_slip)
                export_audit_warmup_boundaries_csv(tc_audit, aud_warmup)
                export_audit_json(tc_audit, aud_json)

                print("\nTimeframe/cost exports:")
                print(f"  Comparison CSV  : {tc_csv}")
                print(f"  Comparison JSON : {tc_json}")
                print(f"  Yearly CSV      : {yr_csv}")
                print(f"  Sensitivity CSV : {cs_csv}")
                print("\nAudit exports:")
                print(f"  Data counts CSV : {aud_counts}")
                print(f"  Identity CSV    : {aud_identity}")
                print(f"  Slippage CSV    : {aud_slip}")
                print(f"  Warmup CSV      : {aud_warmup}")
                print(f"  Audit JSON      : {aud_json}")
            print(_WARNING)
            return

        elif args.run_frozen_oos_2025:
            # ---- Stage 5.3: frozen OOS 2025 evaluation ----
            _MS_2025_START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
            _MS_2026_START = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
            cfg_oos = BacktestConfig(
                symbol=args.symbol,
                interval="15m",
                start_ms=_MS_2025_START,
                end_ms=_MS_2026_START,
                initial_capital=initial_capital,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                force_close_at_end=not args.no_force_close,
            )
            oos_report = svc.run_frozen_oos_2025(cfg_oos)
            _print_frozen_oos_2025(oos_report)
            if args.export_dir:
                from app.backtesting.frozen_oos_exporters import (
                    export_oos_audit_json,
                    export_oos_equity_curve_csv,
                    export_oos_scenarios_csv,
                    export_oos_summary_json,
                    export_oos_trades_csv,
                )

                export_path = Path(args.export_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                slug = "BTCUSDT_30m_2025"

                oos_summary = export_path / f"{slug}_oos_summary.json"
                oos_scenarios = export_path / f"{slug}_oos_scenarios.csv"
                oos_trades = export_path / f"{slug}_oos_trades.csv"
                oos_equity = export_path / f"{slug}_oos_equity_curve.csv"
                oos_audit = export_path / f"{slug}_oos_audit.json"

                export_oos_summary_json(oos_report, oos_summary)
                export_oos_scenarios_csv(oos_report, oos_scenarios)
                export_oos_trades_csv(oos_report, oos_trades)
                export_oos_equity_curve_csv(oos_report, oos_equity)
                export_oos_audit_json(oos_report, oos_audit)

                print("\nOOS 2025 exports:")
                print(f"  Summary JSON    : {oos_summary}")
                print(f"  Scenarios CSV   : {oos_scenarios}")
                print(f"  Trades CSV      : {oos_trades}")
                print(f"  Equity curve CSV: {oos_equity}")
                print(f"  Audit JSON      : {oos_audit}")
            print(_WARNING)
            return

        elif args.compare_entry_variants:
            # ---- Entry variant comparison mode ----

            _MS_2023_START = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)
            _MS_2024_START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
            _MS_2025_START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
            cfg_entry_23 = BacktestConfig(
                symbol=args.symbol,
                interval=args.interval,
                start_ms=_MS_2023_START,
                end_ms=_MS_2024_START,
                initial_capital=initial_capital,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                force_close_at_end=not args.no_force_close,
            )
            cfg_entry_24 = BacktestConfig(
                symbol=args.symbol,
                interval=args.interval,
                start_ms=_MS_2024_START,
                end_ms=_MS_2025_START,
                initial_capital=initial_capital,
                fee_percentage=fee_pct,
                slippage_percentage=slip_pct,
                force_close_at_end=not args.no_force_close,
            )
            entry_report = svc.run_entry_multi_period(cfg_entry_23, cfg_entry_24)
            _print_entry_comparison(entry_report)
            if args.export_dir:
                from app.backtesting.entry_exporters import (
                    export_entry_comparison_csv,
                    export_entry_comparison_json,
                    export_filter_analysis_csv,
                    export_signal_counts_csv,
                    export_yearly_entry_comparison_csv,
                )

                export_path = Path(args.export_dir)
                export_path.mkdir(parents=True, exist_ok=True)
                slug = f"{args.symbol}_{args.interval}_2023_2024"

                ec_csv = export_path / f"{slug}_entry_comparison.csv"
                ec_json = export_path / f"{slug}_entry_comparison.json"
                yr_csv = export_path / f"{slug}_yearly_entry_comparison.csv"
                fa_csv = export_path / f"{slug}_filter_analysis.csv"
                sc_csv = export_path / f"{slug}_signal_counts.csv"

                export_entry_comparison_csv(entry_report.period_2024, ec_csv)
                export_entry_comparison_json(entry_report.period_2024, ec_json)
                export_yearly_entry_comparison_csv(entry_report, yr_csv)
                export_filter_analysis_csv(entry_report.period_2024, fa_csv)
                export_signal_counts_csv(entry_report.period_2024, sc_csv)

                print("\nEntry variant exports:")
                print(f"  Comparison CSV   : {ec_csv}")
                print(f"  Comparison JSON  : {ec_json}")
                print(f"  Yearly CSV       : {yr_csv}")
                print(f"  Filter analysis  : {fa_csv}")
                print(f"  Signal counts    : {sc_csv}")
            print(_WARNING)
            return

        elif args.normalized_comparison:
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


def _print_entry_comparison(multi_report: "EntryMultiPeriodReport") -> None:
    """Print entry variant comparison table and multi-year summary."""
    sep = "=" * 110

    for period_report in (multi_report.period_2023, multi_report.period_2024):
        label = period_report.period_label
        print(f"\n{sep}")
        print(f"  ENTRY VARIANT COMPARISON {label}  (PAPER/TEST only — 25 % allocation)")
        print(sep)
        print(f"  Buy & Hold {label}: {period_report.bah_return_pct:.4f}%")
        print(
            "  Equity shown = standalone (initial 10 000 USDT per year); "
            "signal counts shared across all variants (same V1 base engine)."
        )

        for exit_name in ("V2_STOP_ONLY", "V2_STOP_TP"):
            combos = [c for c in period_report.combinations if c.exit_config_name == exit_name]
            print(f"\n  ── Exit config: {exit_name} ──")
            hdr = (
                f"  {'Entry Variant':<26} {'Ret%':>8} {'NoCost%':>8} "
                f"{'StdAlone$':>10} {'MaxDD%':>7} "
                f"{'Base':>5} {'Pass':>5} {'Rej':>4} {'PosBlk':>6} {'NoNext':>6} "
                f"{'Buys':>5} {'Trades':>6}"
            )
            print(hdr)
            print(f"  {'-' * 108}")
            for c in combos:
                r = c.result
                sc = c.signal_counts
                print(
                    f"  {c.entry_variant:<26} "
                    f"{r.total_return_pct:>7.4f}% "
                    f"{c.result_nc.total_return_pct:>7.4f}% "
                    f"{c.standalone_final_equity:>10.2f} "
                    f"{r.max_drawdown_pct:>6.4f}% "
                    f"{sc.baseline_buy_candidates:>5} "
                    f"{sc.filter_passed_candidates:>5} "
                    f"{sc.filter_rejected_candidates:>4} "
                    f"{sc.blocked_by_open_position:>6} "
                    f"{sc.passed_without_next_candle:>6} "
                    f"{sc.executed_buys:>5} "
                    f"{r.total_trades:>6}"
                )

    # Multi-year summary (compounded equity)
    print(f"\n{sep}")
    print("  MULTI-YEAR SUMMARY  (compounded: 2023 final → 2024 initial | standalone: 10 000/yr)")
    print(sep)
    for exit_name in ("V2_STOP_ONLY", "V2_STOP_TP"):
        rows = [s for s in multi_report.yearly_summary if s.exit_config_name == exit_name]
        print(f"\n  ── Exit config: {exit_name} ──")
        hdr3 = (
            f"  {'Entry Variant':<26} {'2023%':>8} {'2024%':>8} "
            f"{'Combined%':>10} {'Comp23End':>10} {'Comp24End':>10} "
            f"{'SA23End':>9} {'SA24End':>9} {'WrstDD%':>8} {'Tr23':>5} {'Tr24':>5}"
        )
        print(hdr3)
        print(f"  {'-' * 108}")
        for s in rows:
            print(
                f"  {s.entry_variant:<26} "
                f"{s.return_pct_2023:>7.4f}% "
                f"{s.return_pct_2024:>7.4f}% "
                f"{s.combined_return_pct:>9.4f}% "
                f"{s.compounded_final_2023:>10.2f} "
                f"{s.compounded_final_2024:>10.2f} "
                f"{s.standalone_final_2023:>9.2f} "
                f"{s.standalone_final_2024:>9.2f} "
                f"{s.worst_drawdown_pct:>7.4f}% "
                f"{s.total_trades_2023:>5} "
                f"{s.total_trades_2024:>5}"
            )
    print(sep)
    print("  No entry variant is declared optimal or expected to be profitable.")


def _print_timeframe_cost_comparison(report: "TimeframeCostReport") -> None:
    """Print timeframe × cost scenario robustness comparison table."""
    from app.backtesting.timeframe_cost_comparison import COST_SCENARIO_NAMES, TIMEFRAMES

    sep = "=" * 105
    print(f"\n{sep}")
    print("  TIMEFRAME × COST ROBUSTNESS  (frozen: ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY @ 25 %)")
    print(sep)
    print(
        "  Indicator periods NOT scaled across timeframes "
        "(EMA-200 = 200 candles of that timeframe)."
    )
    print("  PAPER/TEST only. No timeframe or scenario declared optimal.")

    for tf in TIMEFRAMES:
        print(f"\n  ── Timeframe: {tf} ──")
        hdr = (
            f"  {'Scenario':<16} {'Year':>4} {'Ret%':>8} {'B&H%':>8} "
            f"{'Trades':>7} {'WinR%':>7} {'ProfFact':>9} {'MaxDD%':>8} "
            f"{'Fees':>8} {'SlipCost':>9} {'CostDrag%':>10} {'Exp%':>6}"
        )
        print(hdr)
        print(f"  {'-' * 103}")
        for scenario in COST_SCENARIO_NAMES:
            for year in ("2023", "2024"):
                r = next(
                    x
                    for x in report.results
                    if x.timeframe == tf and x.cost_scenario == scenario and x.year == year
                )
                wr = f"{r.win_rate_pct:.1f}%" if r.win_rate_pct is not None else "N/A"
                pf = f"{r.profit_factor:.4f}" if r.profit_factor is not None else "N/A"
                print(
                    f"  {scenario:<16} {year:>4} "
                    f"{r.return_pct:>7.4f}% "
                    f"{r.buy_and_hold_return_pct:>7.4f}% "
                    f"{r.total_trades:>7} "
                    f"{wr:>7} "
                    f"{pf:>9} "
                    f"{r.max_drawdown_pct:>7.4f}% "
                    f"{r.total_fees:>8.2f} "
                    f"{r.slippage_cost:>9.2f} "
                    f"{r.cost_drag:>9.4f}% "
                    f"{r.exposure_pct:>5.1f}%"
                )

    print(f"\n{sep}")
    print("  MULTI-YEAR SUMMARY (2023→2024 compounded per scenario)")
    print(sep)
    hdr2 = (
        f"  {'TF':<4} {'Scenario':<16} {'2023%':>8} {'2024%':>8} "
        f"{'Combined%':>10} {'Comp23End':>10} {'Comp24End':>10} "
        f"{'PosYrs':>7} {'WrstDD%':>8} {'Tr23':>5} {'Tr24':>5}"
    )
    print(hdr2)
    print(f"  {'-' * 103}")
    for tf in TIMEFRAMES:
        for scenario in COST_SCENARIO_NAMES:
            s = next(
                x
                for x in report.yearly_summary
                if x.timeframe == tf and x.cost_scenario == scenario
            )
            print(
                f"  {tf:<4} {scenario:<16} "
                f"{s.return_pct_2023:>7.4f}% "
                f"{s.return_pct_2024:>7.4f}% "
                f"{s.combined_return_pct:>9.4f}% "
                f"{s.compounded_final_2023:>10.2f} "
                f"{s.compounded_final_2024:>10.2f} "
                f"{s.positive_years:>7} "
                f"{s.worst_drawdown_pct:>7.4f}% "
                f"{s.total_trades_2023:>5} "
                f"{s.total_trades_2024:>5}"
            )

    print(f"\n{sep}")
    print("  ROBUSTNESS FLAGS")
    print(sep)
    hdr3 = (
        f"  {'TF':<4} {'Scenario':<16} "
        f"{'Pos2Yrs':>8} {'PF>1(2Y)':>9} {'DepLS':>6} {'FailCons':>9} {'<20Tr':>6}"
    )
    print(hdr3)
    print(f"  {'-' * 62}")
    for tf in TIMEFRAMES:
        for scenario in COST_SCENARIO_NAMES:
            rob = next(
                x for x in report.robustness if x.timeframe == tf and x.cost_scenario == scenario
            )
            print(
                f"  {tf:<4} {scenario:<16} "
                f"{'YES' if rob.positive_both_years else 'NO':>8} "
                f"{'YES' if rob.profit_factor_above_1_both_years else 'NO':>9} "
                f"{'YES' if rob.depends_on_low_slippage else 'NO':>6} "
                f"{'YES' if rob.fails_with_conservative else 'NO':>9} "
                f"{'YES' if rob.low_trade_count else 'NO':>6}"
            )
    print(sep)
    print("  No timeframe or cost scenario is declared optimal.")
    print("  PAPER/TEST only. Past results do NOT predict future performance.")


def _print_frozen_oos_2025(report: "FrozenOos2025Report") -> None:
    """Print the frozen OOS 2025 result table."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("  STAGE 5.3 — FROZEN OUT-OF-SAMPLE 2025  (ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY @ 25 %)")
    print(sep)
    print("  Source: BTCUSDT 15m → 30m aggregated.  Period: 2025-01-01 to 2026-01-01 exclusively.")
    print("  PAPER/TEST only. No parameter modification after viewing this result.")

    print(
        f"\n  {'Scenario':<16} {'Ret%':>9} {'B&H%':>9} {'Trades':>7} "
        f"{'WinR%':>7} {'ProfFact':>9} {'MaxDD%':>8} {'Fees':>9} {'SlipCost':>9} {'Exp%':>6}"
    )
    print(f"  {'-' * 88}")
    for s in report.scenarios:
        wr = f"{s.win_rate_pct:.1f}%" if s.win_rate_pct is not None else "N/A"
        pf = f"{s.profit_factor:.4f}" if s.profit_factor is not None else "N/A"
        print(
            f"  {s.scenario:<16} "
            f"{s.return_pct:>8.4f}% "
            f"{s.buy_and_hold_return_pct:>8.4f}% "
            f"{s.total_trades:>7} "
            f"{wr:>7} "
            f"{pf:>9} "
            f"{s.max_drawdown_pct:>7.4f}% "
            f"{s.total_fees:>9.4f} "
            f"{s.slippage_cost:>9.4f} "
            f"{s.exposure_pct:>5.1f}%"
        )

    print(f"\n{sep}")
    print("  INTERPRETATION FLAGS")
    print(sep)
    intp = report.interpretation
    flags = [
        ("profitable_no_costs", intp.profitable_no_costs),
        ("profitable_base_costs", intp.profitable_base_costs),
        ("profitable_conservative", intp.profitable_conservative),
        ("profit_factor_above_1_base", intp.profit_factor_above_1_base),
        ("costs_monotonic", intp.costs_monotonic),
        ("drawdown_below_historical", intp.drawdown_below_historical),
        ("return_above_historical_min", intp.return_above_historical_min),
        ("sufficient_trades (>=5)", intp.sufficient_trades),
    ]
    for name, value in flags:
        print(f"  {'YES' if value else 'NO':>4}  {name}")

    ref = report.historical_reference
    print(f"\n{sep}")
    print("  HISTORICAL IN-SAMPLE REFERENCE (2023–2024, BASE_COSTS, read-only)")
    print(sep)
    print(f"  2023 return  : {ref['period_2023_return_pct']}%")
    print(f"  2024 return  : {ref['period_2024_return_pct']}%")
    print(f"  Combined     : {ref['combined_return_pct']}%")
    print(f"  Total trades : {ref['total_trades_2023_2024']}")
    print(f"  Worst DD     : {ref['worst_drawdown_pct']}%")
    print(f"  Note         : {ref['note']}")

    aud = report.audit
    print(f"\n{sep}")
    print("  AUDIT SUMMARY")
    print(sep)
    print(
        f"  15m candles  : {aud.candles_15m_total} total "
        f"({aud.warmup_15m} warmup + {aud.eval_15m} eval)"
    )
    print(
        f"  30m candles  : {aud.candles_30m_total} total "
        f"({aud.warmup_30m} warmup + {aud.eval_30m} eval)"
    )
    print(f"  Signal hash  : {aud.entry_signal_hash}")
    print(f"  Warmup ok    : {aud.warmup_boundary_ok}")
    print(f"  End ok       : {aud.end_boundary_ok}")
    print(f"  Monotonic    : {aud.costs_monotonic}")
    if aud.missing_data_ranges:
        print("  Missing data :")
        for gap in aud.missing_data_ranges:
            print(f"    {gap}")
    else:
        print("  Missing data : none detected")
    if aud.violations:
        print("  Violations   :")
        for v in aud.violations:
            print(f"    {v}")
    else:
        print("  Violations   : none")
    print(sep)
    print("  No result constitutes a trading recommendation.")
    print("  PAPER/TEST only. Past results do NOT predict future performance.")


def _print_breakout_family(report: "BreakoutFamilyReport") -> None:
    """Print Stage 6.0 Donchian breakout family results."""
    from app.backtesting.breakout_family import (
        BREAKOUT_YEARS,
        select_best_qualified,
    )

    sep = "=" * 100
    print(f"\n{sep}")
    print("  STAGE 6.0 — DONCHIAN BREAKOUT FAMILY  (8 configs × 3 scenarios × 2021–2025)")
    print(sep)
    print("  PAPER/TEST only. No real money. Do NOT modify configurations after viewing results.")

    # ---- Per-config compounded summary ----
    print(f"\n{sep}")
    print("  COMPOUNDED 5-YEAR RESULTS (initial capital carried forward year-to-year)")
    print(sep)
    hdr = (
        f"  {'Config':>8} {'Scenario':<16} {'TotalRet%':>10} "
        f"{'FinalEquity':>12} {'ProfFact':>9} {'MaxDD%':>8} "
        f"{'Trades':>7} {'YrsData':>8}"
    )
    print(hdr)
    print(f"  {'-' * 98}")
    for r in report.compounded_results:
        pf = f"{r.profit_factor:.4f}" if r.profit_factor is not None else "N/A"
        print(
            f"  {r.config_id:>8} {r.scenario:<16} "
            f"{r.total_return_pct:>9.4f}% "
            f"{r.final_equity:>12.2f} "
            f"{pf:>9} "
            f"{r.max_drawdown_pct:>7.4f}% "
            f"{r.total_trades:>7} "
            f"{r.years_with_data:>8}"
        )

    # ---- Standalone per-year summary (BASE_COSTS only) ----
    print(f"\n{sep}")
    print("  STANDALONE PER-YEAR RETURNS (BASE_COSTS, initial capital = 10 000 each year)")
    print(sep)
    hdr2 = f"  {'Config':>8} " + " ".join(f"{yr:>8}" for yr in BREAKOUT_YEARS)
    print(hdr2)
    print(f"  {'-' * 98}")
    config_ids = sorted({r.config_id for r in report.yearly_results})
    for config_id in config_ids:
        yr_returns = {}
        for yr in BREAKOUT_YEARS:
            yr_r = next(
                (
                    r
                    for r in report.yearly_results
                    if r.config_id == config_id and r.year == yr and r.scenario == "BASE_COSTS"
                ),
                None,
            )
            yr_returns[yr] = f"{yr_r.return_pct:>7.2f}%" if yr_r else "   N/A"
        row = f"  {config_id:>8} " + " ".join(yr_returns[yr] for yr in BREAKOUT_YEARS)
        print(row)

    # ---- Qualification summary ----
    print(f"\n{sep}")
    print("  QUALIFICATION SUMMARY  (all 9 criteria must pass)")
    print(sep)
    hdr3 = (
        f"  {'Config':>8} {'Status':>10} "
        f"{'BCPos':>6} {'CPos':>5} {'Yrs≥3':>6} "
        f"{'PF>1.1':>7} {'DD<15':>6} {'Tr≥30':>6} "
        f"{'NoNeg':>6} {'Conc':>5} {'NoVio':>6}"
    )
    print(hdr3)
    print(f"  {'-' * 98}")
    for rob in report.robustness:
        check_map = {c.name: c.passed for c in rob.qualification_checks}
        status = "QUALIFIED" if rob.is_qualified else "REJECTED"
        print(
            f"  {rob.config_id:>8} {status:>10} "
            f"{'Y' if check_map.get('compounded_base_positive') else 'N':>6} "
            f"{'Y' if check_map.get('compounded_conservative_positive') else 'N':>5} "
            f"{'Y' if check_map.get('years_positive_ge3') else 'N':>6} "
            f"{'Y' if check_map.get('profit_factor_gt1_10') else 'N':>7} "
            f"{'Y' if check_map.get('max_drawdown_lt15') else 'N':>6} "
            f"{'Y' if check_map.get('min_trades_30') else 'N':>6} "
            f"{'Y' if check_map.get('no_year_below_neg10') else 'N':>6} "
            f"{'Y' if check_map.get('best_year_le70pct_gross') else 'N':>5} "
            f"{'Y' if check_map.get('no_violations') else 'N':>6}"
        )

    best = select_best_qualified(report.robustness)
    print(f"\n  Best qualified config : {best if best else 'NONE — no config qualified'}")

    # ---- Audit ----
    aud = report.audit
    print(f"\n{sep}")
    print("  AUDIT")
    print(sep)
    print(f"  15m candles total : {aud.candles_15m_total}")
    print(f"  Years covered     : {', '.join(aud.years_covered)}")
    if aud.missing_data_years:
        print(f"  Missing years     : {', '.join(aud.missing_data_years)}")
    else:
        print("  Missing years     : none")
    if aud.violations:
        print("  Violations        :")
        for v in aud.violations:
            print(f"    {v}")
    else:
        print("  Violations        : none")
    print(sep)
    print("  No configuration is declared optimal or expected to be profitable.")
    print("  PAPER/TEST only. Past results do NOT predict future performance.")


if __name__ == "__main__":
    main()
