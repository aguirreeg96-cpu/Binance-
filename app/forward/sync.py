"""Stage 6.1 — sync closed BTCUSDT 15m candles from Binance public data.

Public market data only: no private keys, no order placement. Bridges the
async Binance download (HistoricalDataService) with the synchronous
ForwardPaperEngine, which only ever reads candles already persisted in the
DB. Downloads are idempotent (CandleRepository upserts by symbol/interval/
open_time), so re-running this after a crash or restart never duplicates
candles.

Never fetches the still-open candle: HistoricalDataService excludes any
candle whose close_time is at or after the live Binance server time, and
the `end` boundary passed here is our own clock as a second, independent
upper bound.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.forward.manifest import FORWARD_SOURCE_INTERVAL, FORWARD_SYMBOL
from app.market_data.client import MarketDataClient
from app.market_data.historical_service import DownloadResult, HistoricalDataService
from app.market_data.interval_utils import interval_to_ms
from app.models.candle import Candle

# Generous cold-start warm-up window: comfortably more than the 200-period
# 4h EMA (~33.3 days) plus the Donchian entry/exit/ATR lookbacks.
WARMUP_LOOKBACK_DAYS = 400

_SOURCE_INTERVAL_MS = interval_to_ms(FORWARD_SOURCE_INTERVAL)


def _latest_stored_open_time(session: Session) -> int | None:
    return session.scalars(
        select(Candle.open_time)
        .where(Candle.symbol == FORWARD_SYMBOL)
        .where(Candle.interval == FORWARD_SOURCE_INTERVAL)
        .order_by(Candle.open_time.desc())
        .limit(1)
    ).first()


async def sync_forward_candles(
    session: Session,
    client: MarketDataClient,
    *,
    now: datetime | None = None,
    max_requests: int = 500,
) -> DownloadResult | None:
    """Download and idempotently store closed 15m candles up to `now`.

    Returns None if already caught up (nothing new to fetch yet) instead of
    calling the API with an empty/invalid range.
    """
    now_utc = now.astimezone(UTC) if now is not None else datetime.now(UTC)

    latest_open_ms = _latest_stored_open_time(session)
    start = (
        datetime.fromtimestamp((latest_open_ms + _SOURCE_INTERVAL_MS) / 1000.0, tz=UTC)
        if latest_open_ms is not None
        else now_utc - timedelta(days=WARMUP_LOOKBACK_DAYS)
    )

    if start >= now_utc:
        return None

    service = HistoricalDataService(client=client, max_requests=max_requests)
    result = await service.download(
        session=session,
        symbol=FORWARD_SYMBOL,
        interval=FORWARD_SOURCE_INTERVAL,
        start=start,
        end=now_utc,
        include_open_candle=False,
    )
    session.commit()
    return result
