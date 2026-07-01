"""CLI for Stage 6.1 forward paper trading of the frozen B_4h config.

Usage:
    python -m app.cli.run_paper_breakout --once
    python -m app.cli.run_paper_breakout --continuous
    python -m app.cli.run_paper_breakout --continuous --poll-interval-seconds 300

Public Binance market data only — no private keys, no real orders, no
exchange order placement of any kind. PAPER/TEST only.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_WARNING = (
    "\n"
    "╔══════════════════════════════════════════════════════════════╗\n"
    "║   ⚠  FORWARD PAPER TRADING — NO REAL MONEY — PAPER/TEST  ⚠   ║\n"
    "║  No private keys used. No orders ever sent to Binance.        ║\n"
    "║  Past results do NOT predict future performance.              ║\n"
    "╚══════════════════════════════════════════════════════════════╝\n"
)


def _now_naive_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def _run_one_cycle(args: argparse.Namespace) -> int:
    """Run one sync+evaluate cycle.

    DB migrations are intentionally omitted here — they are applied once at
    process startup inside main(), before the event loop begins, to avoid
    Alembic noise on every polling iteration.
    """
    from app.config import get_settings
    from app.database import SessionLocal
    from app.forward.engine import ForwardPaperEngine, start_or_resume_launch
    from app.forward.exceptions import ForwardConfigMismatchError
    from app.forward.sync import sync_forward_candles
    from app.market_data.client import BinanceMarketDataClient

    settings = get_settings()

    with SessionLocal() as session:
        try:
            launch = start_or_resume_launch(
                session,
                now=_now_naive_utc(),
                initial_capital=settings.forward_initial_capital,
            )
        except ForwardConfigMismatchError as exc:
            print(f"ERROR: {exc}")
            return 1

        async with BinanceMarketDataClient(
            base_url=settings.binance_market_data_url,
            timeout=settings.market_data_timeout,
            max_retries=settings.market_data_max_retries,
            max_retry_after=settings.market_data_max_retry_after,
        ) as client:
            sync_result = await sync_forward_candles(
                session,
                client,
                max_requests=settings.market_data_max_requests,
            )

        if sync_result is not None:
            logger.info(
                "Sync | inserted=%d updated=%d ignored=%d requests=%d | range=[%s, %s]",
                sync_result.inserted,
                sync_result.updated,
                sync_result.ignored,
                sync_result.requests_made,
                sync_result.first_open_time.isoformat() if sync_result.first_open_time else "n/a",
                sync_result.last_open_time.isoformat() if sync_result.last_open_time else "n/a",
            )
        else:
            logger.info("Sync: already up to date — no new candles fetched.")

        engine = ForwardPaperEngine(
            session, data_gap_grace_seconds=settings.forward_data_gap_grace_seconds
        )
        outcomes = engine.run_cycle(launch, _now_naive_utc())
        report = engine.last_report
        session.refresh(launch)

    print(f"Launch id={launch.id} status={launch.status} evaluations_this_cycle={len(outcomes)}")
    for outcome in outcomes:
        print(f"  {outcome.candle_close_time.isoformat()}  {outcome.signal}  {outcome.reasons}")

    if report is not None:
        if report.skipped_partial_launch_candle:
            print(
                f"  skipped_partial_launch_candle=true"
                f" (launch {report.launch_timestamp_utc.isoformat()}Z is not at a 4h UTC"
                f" boundary; the partially-elapsed candle is excluded)"
            )
        if report.next_eligible_4h_close_utc is not None:
            print(
                f"  next_eligible_4h_close={report.next_eligible_4h_close_utc.isoformat()}Z"
                f" (approx)"
            )
        print(
            f"  candles_15m={report.candles_15m_available}"
            f" | candles_4h={report.candles_4h_complete}"
        )

    return 0


async def _run_continuous(args: argparse.Namespace) -> int:
    from app.config import get_settings

    settings = get_settings()
    poll_seconds = args.poll_interval_seconds or settings.forward_poll_interval_seconds
    logger.info("Starting continuous forward paper trading loop (poll=%ds).", poll_seconds)

    try:
        while True:
            rc = await _run_one_cycle(args)
            if rc != 0:
                return rc
            await asyncio.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("Continuous mode stopped (KeyboardInterrupt).")
        return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.run_paper_breakout",
        description=(
            "Run the frozen B_4h Donchian breakout forward paper-trading engine. "
            "Downloads closed BTCUSDT 15m candles from Binance public data, "
            "aggregates to 4h, and evaluates the frozen strategy — no real "
            "orders are ever placed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--once", action="store_true", help="Sync candles and run exactly one evaluation cycle."
    )
    mode.add_argument(
        "--continuous",
        action="store_true",
        help="Repeat the sync-and-evaluate cycle forever at --poll-interval-seconds.",
    )
    p.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=None,
        help="Continuous mode polling interval (defaults to FORWARD_POLL_INTERVAL_SECONDS).",
    )
    return p


def main() -> None:
    print(_WARNING)
    parser = _build_parser()
    args = parser.parse_args()

    # Run DB migrations exactly ONCE at startup, before the event loop starts.
    # Continuous mode polls _run_one_cycle() repeatedly; running migrations on
    # every poll would produce Alembic noise and slow down each cycle.
    from app.database import run_migrations

    logger.info("Running DB migrations...")
    run_migrations()

    if args.once:
        rc = asyncio.run(_run_one_cycle(args))
    else:
        rc = asyncio.run(_run_continuous(args))

    print(_WARNING)
    sys.exit(rc)


if __name__ == "__main__":
    main()
