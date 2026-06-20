"""
CandleRepository — upsert and query candles in SQLite.

Upsert strategy (avoids rowcount ambiguity):
  1. Fetch existing rows by (symbol, interval, open_time) for the batch.
  2. Compare field-by-field to classify: inserted / updated / ignored.
  3. Execute INSERT for new, UPDATE for changed; skip ignored rows.
  4. All operations run in the session's current transaction — the caller
     controls commit/rollback.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.market_data.kline_parser import KlineData
from app.models.candle import Candle

logger = logging.getLogger(__name__)

_COMPARE_FIELDS = (
    "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume",
    "trades", "taker_buy_base_volume", "taker_buy_quote_volume",
)


@dataclass
class UpsertResult:
    inserted: int
    updated: int
    ignored: int


class CandleRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_batch(self, candles: Sequence[KlineData]) -> UpsertResult:
        """
        Upsert a batch of KlineData rows.

        All candles in a batch are expected to share the same symbol+interval
        (as produced by HistoricalDataService). If the batch is empty, returns
        zeros immediately without touching the DB.

        Runs entirely within the caller's transaction — no commit here.
        """
        if not candles:
            return UpsertResult(inserted=0, updated=0, ignored=0)

        symbol = candles[0].symbol
        interval = candles[0].interval
        open_times = [c.open_time for c in candles]

        # 1. Fetch existing rows
        existing_rows: dict[int, Candle] = {
            row.open_time: row
            for row in self.session.scalars(
                select(Candle)
                .where(Candle.symbol == symbol)
                .where(Candle.interval == interval)
                .where(Candle.open_time.in_(open_times))
            ).all()
        }

        inserted = updated = ignored = 0

        for kd in candles:
            row = existing_rows.get(kd.open_time)

            if row is None:
                self.session.add(_candle_from_kline(kd))
                inserted += 1
            elif _has_changes(row, kd):
                _apply_update(row, kd)
                updated += 1
            else:
                ignored += 1

        if inserted or updated:
            self.session.flush()

        logger.debug(
            "upsert_batch %s %s: inserted=%d updated=%d ignored=%d",
            symbol, interval, inserted, updated, ignored,
        )
        return UpsertResult(inserted=inserted, updated=updated, ignored=ignored)

    def query(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
        include_open_candle: bool = True,
        server_time_ms: int | None = None,
    ) -> list[Candle]:
        """
        Query stored candles in ascending open_time order.

        start_ms — inclusive lower bound on open_time (ms).
        end_ms   — exclusive upper bound on open_time (ms).
        include_open_candle=False requires server_time_ms to filter.
        """
        stmt = (
            select(Candle)
            .where(Candle.symbol == symbol.upper())
            .where(Candle.interval == interval)
            .order_by(Candle.open_time.asc())
        )
        if start_ms is not None:
            stmt = stmt.where(Candle.open_time >= start_ms)
        if end_ms is not None:
            stmt = stmt.where(Candle.open_time < end_ms)
        if not include_open_candle and server_time_ms is not None:
            stmt = stmt.where(Candle.close_time < server_time_ms)
        if limit is not None:
            stmt = stmt.limit(limit)

        return list(self.session.scalars(stmt).all())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _candle_from_kline(kd: KlineData) -> Candle:
    return Candle(
        symbol=kd.symbol,
        interval=kd.interval,
        open_time=kd.open_time,
        open=kd.open,
        high=kd.high,
        low=kd.low,
        close=kd.close,
        volume=kd.volume,
        close_time=kd.close_time,
        quote_asset_volume=kd.quote_asset_volume,
        trades=kd.number_of_trades,
        taker_buy_base_volume=kd.taker_buy_base_volume,
        taker_buy_quote_volume=kd.taker_buy_quote_volume,
        is_closed=True,
    )


def _apply_update(row: Candle, kd: KlineData) -> None:
    row.open = kd.open
    row.high = kd.high
    row.low = kd.low
    row.close = kd.close
    row.volume = kd.volume
    row.close_time = kd.close_time
    row.quote_asset_volume = kd.quote_asset_volume
    row.trades = kd.number_of_trades
    row.taker_buy_base_volume = kd.taker_buy_base_volume
    row.taker_buy_quote_volume = kd.taker_buy_quote_volume


def _has_changes(row: Candle, kd: KlineData) -> bool:
    return (
        row.open != kd.open
        or row.high != kd.high
        or row.low != kd.low
        or row.close != kd.close
        or row.volume != kd.volume
        or row.close_time != kd.close_time
        or row.quote_asset_volume != kd.quote_asset_volume
        or row.trades != kd.number_of_trades
        or row.taker_buy_base_volume != kd.taker_buy_base_volume
        or row.taker_buy_quote_volume != kd.taker_buy_quote_volume
    )
