"""
Download and persist historical klines from Binance.

Semantics:
  start — inclusive (open_time >= start_ms)
  end   — exclusive (open_time < end_ms)
  Naive datetimes are rejected; callers must pass UTC-aware datetimes.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.market_data.client import MarketDataClient
from app.market_data.exceptions import (
    InvalidSymbolError,
    MarketDataError,
    MaxRequestsError,
    PaginationStallError,
)
from app.market_data.interval_utils import interval_to_ms, validate_interval
from app.market_data.kline_parser import KlineData

if TYPE_CHECKING:
    from app.repositories.candle_repository import CandleRepository

logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")


@dataclass
class DownloadResult:
    symbol: str
    interval: str
    requested_start: datetime
    requested_end: datetime
    requests_made: int = 0
    received: int = 0
    inserted: int = 0
    updated: int = 0
    ignored: int = 0
    first_open_time: datetime | None = None
    last_open_time: datetime | None = None
    duration_ms: int = 0


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _require_utc(dt: datetime, name: str) -> datetime:
    if dt.tzinfo is None:
        raise ValueError(
            f"{name} must be a timezone-aware datetime. "
            "Naive datetimes are rejected to prevent UTC/local ambiguity."
        )
    return dt.astimezone(timezone.utc)


class HistoricalDataService:
    def __init__(
        self,
        client: MarketDataClient,
        max_requests: int = 500,
    ) -> None:
        self.client = client
        self.max_requests = max_requests

    async def download(
        self,
        session: Session,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        include_open_candle: bool = False,
    ) -> DownloadResult:
        from app.repositories.candle_repository import CandleRepository

        t0 = time.monotonic()

        # --- Input validation ---
        symbol = symbol.upper().strip()
        if not _SYMBOL_RE.match(symbol):
            raise InvalidSymbolError(symbol, "must be 2-20 uppercase alphanumeric chars")

        validate_interval(interval)

        start = _require_utc(start, "start")
        end = _require_utc(end, "end")
        if start >= end:
            raise ValueError(f"start ({start.isoformat()}) must be before end ({end.isoformat()})")

        start_ms = _dt_to_ms(start)
        end_ms = _dt_to_ms(end)
        interval_ms = interval_to_ms(interval)

        # --- Validate symbol once ---
        await self._validate_symbol(symbol)

        # --- Server time for open-candle filter ---
        server_time_ms: int | None = None
        if not include_open_candle:
            server_time_ms = await self.client.get_server_time()

        result = DownloadResult(
            symbol=symbol,
            interval=interval,
            requested_start=start,
            requested_end=end,
        )

        repo = CandleRepository(session)
        current_start_ms = start_ms
        prev_last_open_time: int | None = None

        while current_start_ms < end_ms:
            if result.requests_made >= self.max_requests:
                raise MaxRequestsError(self.max_requests)

            logger.debug(
                "Fetching %s %s from %s (page %d)",
                symbol, interval,
                _ms_to_dt(current_start_ms).isoformat(),
                result.requests_made + 1,
            )

            raw_batch = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                start_time_ms=current_start_ms,
                end_time_ms=end_ms - 1,  # end is exclusive in our API
                limit=1000,
            )
            result.requests_made += 1

            if not raw_batch:
                logger.debug("Empty page — download complete.")
                break

            # --- Batch integrity checks ---
            _validate_raw_batch(raw_batch, current_start_ms)

            last_open_time_in_batch = int(raw_batch[-1][0])

            # Stall detection
            if prev_last_open_time is not None:
                if last_open_time_in_batch <= prev_last_open_time:
                    raise PaginationStallError(
                        f"Pagination stalled: last open_time {last_open_time_in_batch} "
                        f"<= previous {prev_last_open_time}"
                    )

            next_start_ms = last_open_time_in_batch + interval_ms
            if next_start_ms <= current_start_ms:
                raise PaginationStallError(
                    f"next_start ({next_start_ms}) <= current_start ({current_start_ms})"
                )

            # --- Parse ---
            klines = [KlineData.from_raw(raw, symbol, interval) for raw in raw_batch]

            # --- Filter: respect end boundary (open_time < end_ms) ---
            klines = [k for k in klines if k.open_time < end_ms]

            # --- Filter: exclude unclosed candle ---
            if not include_open_candle and server_time_ms is not None:
                klines = [k for k in klines if k.close_time < server_time_ms]

            # --- Validate OHLCV invariants ---
            for kline in klines:
                kline.validate()

            # --- Upsert ---
            if klines:
                upsert = repo.upsert_batch(klines)
                result.received += len(klines)
                result.inserted += upsert.inserted
                result.updated += upsert.updated
                result.ignored += upsert.ignored

                if result.first_open_time is None:
                    result.first_open_time = _ms_to_dt(klines[0].open_time)
                result.last_open_time = _ms_to_dt(klines[-1].open_time)

            prev_last_open_time = last_open_time_in_batch
            current_start_ms = next_start_ms

            if len(raw_batch) < 1000:
                logger.debug("Last page (received %d < 1000).", len(raw_batch))
                break

        result.duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Download complete: %s %s — %d requests, %d candles "
            "(inserted=%d updated=%d ignored=%d) in %dms",
            symbol, interval, result.requests_made, result.received,
            result.inserted, result.updated, result.ignored, result.duration_ms,
        )
        return result

    async def _validate_symbol(self, symbol: str) -> None:
        """Validate symbol is in TRADING status on Spot."""
        try:
            info = await self.client.get_exchange_info(symbol)
        except MarketDataError as exc:
            raise InvalidSymbolError(symbol, str(exc)) from exc

        symbols = info.get("symbols", [])
        if not symbols:
            raise InvalidSymbolError(symbol, "not found in exchangeInfo")

        sym_info = symbols[0]
        if sym_info.get("symbol") != symbol:
            raise InvalidSymbolError(symbol, "symbol mismatch in exchangeInfo response")
        if sym_info.get("status") != "TRADING":
            raise InvalidSymbolError(
                symbol,
                f"status is {sym_info.get('status')!r}, expected 'TRADING'",
            )
        if "SPOT" not in sym_info.get("permissions", []):
            raise InvalidSymbolError(symbol, "SPOT permission not available for this symbol")


def _validate_raw_batch(batch: list[list], expected_start_ms: int) -> None:
    """Check ascending order and no duplicate open_times within a batch."""
    seen: set[int] = set()
    prev: int | None = None

    for i, raw in enumerate(batch):
        open_time = int(raw[0])

        if open_time in seen:
            raise PaginationStallError(
                f"Duplicate open_time {open_time} at index {i} in batch"
            )
        seen.add(open_time)

        if prev is not None and open_time <= prev:
            raise PaginationStallError(
                f"Batch not in ascending order: index {i} open_time {open_time} "
                f"<= index {i-1} open_time {prev}"
            )
        prev = open_time
