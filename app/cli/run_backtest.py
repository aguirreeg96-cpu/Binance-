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
        result = svc.run(config)
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

    print(_WARNING)


if __name__ == "__main__":
    main()
