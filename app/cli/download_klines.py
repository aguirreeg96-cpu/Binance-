"""
CLI entry point for downloading historical klines.

Usage:
    python -m app.cli.download_klines \\
        --symbol BTCUSDT \\
        --interval 15m \\
        --start 2025-01-01T00:00:00Z \\
        --end 2025-02-01T00:00:00Z
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_dt(s: str) -> datetime:
    """Parse ISO 8601 string → UTC-aware datetime. Reject naive."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime {s!r}. Use ISO 8601 with timezone, e.g. 2025-01-01T00:00:00Z"
        ) from exc
    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"Datetime {s!r} has no timezone. Append 'Z' for UTC."
        )
    return dt.astimezone(timezone.utc)


async def _run(args: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.database import SessionLocal, run_migrations
    from app.market_data.client import BinanceMarketDataClient
    from app.market_data.historical_service import HistoricalDataService

    settings = get_settings()

    print(
        "\n⚠  PAPER / TEST ENVIRONMENT — downloading public market data only.\n"
        "   No API key required. No real money involved.\n"
    )

    logger.info("Running DB migrations...")
    run_migrations()

    async with BinanceMarketDataClient(
        base_url=settings.binance_market_data_url,
        timeout=settings.market_data_timeout,
        max_retries=settings.market_data_max_retries,
        max_retry_after=settings.market_data_max_retry_after,
    ) as client:
        service = HistoricalDataService(
            client=client,
            max_requests=settings.market_data_max_requests,
        )

        with SessionLocal() as session:
            result = await service.download(
                session=session,
                symbol=args.symbol,
                interval=args.interval,
                start=args.start,
                end=args.end,
                include_open_candle=args.include_open_candle,
            )
            session.commit()

    output = {
        "symbol": result.symbol,
        "interval": result.interval,
        "requested_start": result.requested_start.isoformat(),
        "requested_end": result.requested_end.isoformat(),
        "requests_made": result.requests_made,
        "received": result.received,
        "inserted": result.inserted,
        "updated": result.updated,
        "ignored": result.ignored,
        "first_open_time": result.first_open_time.isoformat() if result.first_open_time else None,
        "last_open_time": result.last_open_time.isoformat() if result.last_open_time else None,
        "duration_ms": result.duration_ms,
    }
    print(json.dumps(output, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical klines from Binance public API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python -m app.cli.download_klines "
               "--symbol BTCUSDT --interval 15m "
               "--start 2025-01-01T00:00:00Z --end 2025-02-01T00:00:00Z",
    )
    parser.add_argument("--symbol", required=True, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", required=True, help="Kline interval, e.g. 15m")
    parser.add_argument("--start", required=True, type=_parse_dt, help="Start datetime (UTC ISO 8601)")
    parser.add_argument("--end", required=True, type=_parse_dt, help="End datetime (UTC ISO 8601)")
    parser.add_argument(
        "--include-open-candle", action="store_true", default=False,
        help="Include the current (unclosed) candle",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
