"""
CandleRepository — upsert and query candles in SQLite.

Upsert strategy (avoids rowcount ambiguity):
  1. Fetch existing rows by (symbol, interval, open_time) for the batch.
  2. Compare field-by-field to classify: inserted / updated / ignored.
  3. Execute INSERT for new, UPDATE for changed; skip ignored rows.
  4. All operations run in the session's current transaction — the caller
     controls commit/rollback.

Decimal normalisation:
  All Decimal fields are quantized to 10 decimal places (ROUND_HALF_EVEN)
  via normalize_decimal() before insert, update, and comparison.  This
  ensures that the stored representation and the incoming KlineData value
  are always compared at the same scale, preventing false 'updated'
  results on idempotent re-downloads.

SQL ordering note:
  Financial columns (open, high, close, …) are stored as VARCHAR(50) on
  SQLite (ExactDecimal / TEXT affinity).  Never use ORDER BY, SUM, AVG,
  MIN, or MAX directly on those columns in SQL; use open_time (BigInteger)
  for ordering and perform financial aggregations in Python with Decimal.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.market_data.kline_parser import KlineData
from app.models.candle import Candle
from app.models.types import normalize_decimal

logger = logging.getLogger(__name__)

# Explicit immutable tuple of Binance-sourced fields used for change detection.
# Excludes: id, created_at, and any internally generated metadata — a
# difference in those fields must never trigger an 'updated' classification.
_BINANCE_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
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
            symbol,
            interval,
            inserted,
            updated,
            ignored,
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

        Always order by open_time (BigInteger). Financial columns are
        TEXT on SQLite and must not be used for SQL ordering or aggregation.
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

    def query_for_indicators(
        self,
        symbol: str,
        interval: str,
        warmup_count: int = 0,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> tuple[list[Candle], int]:
        """Query closed candles for indicator calculation, prepending warmup history.

        Returns (all_candles, warmup_len) where:
        - all_candles = warmup_prefix + requested_range (ascending open_time)
        - warmup_len  = number of leading warmup candles

        Callers should run IndicatorCalculator on all_candles and return
        only results[warmup_len:] to avoid leaking warmup-only data.

        limit vs start_ms behaviour:
        - start_ms given: ascending from start_ms, limit applied going forward.
        - start_ms absent + limit given: the most recent `limit` closed candles
          (descending fetch, reversed) so the user sees the freshest data.
        - Neither: all closed candles in ascending order (apply max_candles
          guard at the API layer).
        """
        sym = symbol.upper()

        if start_ms is not None:
            # Anchor on start: ascending from start_ms
            stmt = (
                select(Candle)
                .where(Candle.symbol == sym)
                .where(Candle.interval == interval)
                .where(Candle.is_closed == True)  # noqa: E712
                .where(Candle.open_time >= start_ms)
                .order_by(Candle.open_time.asc())
            )
            if end_ms is not None:
                stmt = stmt.where(Candle.open_time < end_ms)
            if limit is not None:
                stmt = stmt.limit(limit)
            requested = list(self.session.scalars(stmt).all())

        elif limit is not None:
            # No anchor: fetch the most recent `limit` candles
            desc_stmt = (
                select(Candle)
                .where(Candle.symbol == sym)
                .where(Candle.interval == interval)
                .where(Candle.is_closed == True)  # noqa: E712
                .order_by(Candle.open_time.desc())
                .limit(limit)
            )
            if end_ms is not None:
                desc_stmt = desc_stmt.where(Candle.open_time < end_ms)
            requested = list(reversed(list(self.session.scalars(desc_stmt).all())))

        else:
            # No anchor, no limit: all closed candles
            stmt = (
                select(Candle)
                .where(Candle.symbol == sym)
                .where(Candle.interval == interval)
                .where(Candle.is_closed == True)  # noqa: E712
                .order_by(Candle.open_time.asc())
            )
            if end_ms is not None:
                stmt = stmt.where(Candle.open_time < end_ms)
            requested = list(self.session.scalars(stmt).all())

        if not requested or warmup_count <= 0:
            return requested, 0

        # Fetch warmup candles strictly before the first requested candle
        first_open_time = requested[0].open_time
        warmup_stmt = (
            select(Candle)
            .where(Candle.symbol == sym)
            .where(Candle.interval == interval)
            .where(Candle.is_closed == True)  # noqa: E712
            .where(Candle.open_time < first_open_time)
            .order_by(Candle.open_time.desc())
            .limit(warmup_count)
        )
        warmup = list(self.session.scalars(warmup_stmt).all())
        warmup.reverse()  # restore ascending order

        return warmup + requested, len(warmup)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _candle_from_kline(kd: KlineData) -> Candle:
    return Candle(
        symbol=kd.symbol,
        interval=kd.interval,
        open_time=kd.open_time,
        open=normalize_decimal(kd.open),
        high=normalize_decimal(kd.high),
        low=normalize_decimal(kd.low),
        close=normalize_decimal(kd.close),
        volume=normalize_decimal(kd.volume),
        close_time=kd.close_time,
        quote_asset_volume=normalize_decimal(kd.quote_asset_volume),
        trades=kd.number_of_trades,
        taker_buy_base_volume=normalize_decimal(kd.taker_buy_base_volume),
        taker_buy_quote_volume=normalize_decimal(kd.taker_buy_quote_volume),
        is_closed=True,
    )


def _apply_update(row: Candle, kd: KlineData) -> None:
    row.open = normalize_decimal(kd.open)
    row.high = normalize_decimal(kd.high)
    row.low = normalize_decimal(kd.low)
    row.close = normalize_decimal(kd.close)
    row.volume = normalize_decimal(kd.volume)
    row.close_time = kd.close_time
    row.quote_asset_volume = normalize_decimal(kd.quote_asset_volume)
    row.trades = kd.number_of_trades
    row.taker_buy_base_volume = normalize_decimal(kd.taker_buy_base_volume)
    row.taker_buy_quote_volume = normalize_decimal(kd.taker_buy_quote_volume)


def _has_changes(row: Candle, kd: KlineData) -> bool:
    """Return True if any Binance-sourced field differs from the stored value.

    All Decimal comparisons normalise the incoming KlineData value to the
    same 10-decimal-place scale used on write, so a re-download of identical
    data never produces a false positive.

    Fields excluded from comparison: id, created_at, and any internally
    generated metadata (see _BINANCE_FIELDS for the complete authoritative list).
    """
    return (
        row.open != normalize_decimal(kd.open)
        or row.high != normalize_decimal(kd.high)
        or row.low != normalize_decimal(kd.low)
        or row.close != normalize_decimal(kd.close)
        or row.volume != normalize_decimal(kd.volume)
        or row.close_time != kd.close_time
        or row.quote_asset_volume != normalize_decimal(kd.quote_asset_volume)
        or row.trades != kd.number_of_trades
        or row.taker_buy_base_volume != normalize_decimal(kd.taker_buy_base_volume)
        or row.taker_buy_quote_volume != normalize_decimal(kd.taker_buy_quote_volume)
    )
