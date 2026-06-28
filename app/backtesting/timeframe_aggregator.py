"""Timeframe aggregator for Stage 5.2C — builds 30m and 1h candles from 15m source data.

Aggregation rules per the Stage 5.2C specification:
  open                 = first open in group
  high                 = max high across group
  low                  = min low across group
  close                = last close in group
  volume               = sum across group
  quote_asset_volume   = sum across group
  trades               = sum across group
  taker_buy_base_volume  = sum across group
  taker_buy_quote_volume = sum across group

Candle-completeness guarantee:
  A candle is emitted ONLY when ALL source candles in the group are present
  and strictly consecutive (no intra-group gaps).  Trailing incomplete groups
  are silently dropped.  Real market gaps are preserved — no candle is
  invented to fill missing data.

NOTE on indicator periods (Stage 5.2C design decision):
  Indicator periods are NOT scaled when changing timeframe.  The same
  numerical period values (e.g. EMA-200) are used for all timeframes so
  results represent a literal parameter comparison.  This means a 200-period
  EMA on 15m spans 50h, on 30m spans 100h, and on 1h spans 200h.  The
  spec explicitly documents this choice to avoid concealed optimisation.

PAPER/TEST only.  No live orders, no real capital.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from app.models.candle import Candle

_ZERO = Decimal("0")


def aggregate_candles(
    candles: list[Candle],
    target_interval: str,
    source_interval_ms: int = 900_000,
) -> list[Candle]:
    """Aggregate source candles into coarser-grained candles.

    Parameters
    ----------
    candles:
        Source candles in ascending open_time order.  All must share the same
        symbol; symbol is carried through to the output.
    target_interval:
        Destination Binance interval string, e.g. ``"30m"`` or ``"1h"``.
        Must be a positive integer multiple of *source_interval_ms*.
    source_interval_ms:
        Duration of each source candle in milliseconds (default 900 000 = 15m).

    Returns
    -------
    list[Candle]
        Aggregated candles in ascending order.  Only complete, gap-free groups
        are included; partial trailing groups are dropped.

    Raises
    ------
    ValueError
        If *target_interval* is not a positive integer multiple of
        *source_interval_ms*, or if *target_interval* is not in INTERVAL_MS.
    """
    from app.market_data.interval_utils import INTERVAL_MS

    if target_interval not in INTERVAL_MS:
        raise ValueError(f"Unknown target_interval {target_interval!r}")

    target_ms = INTERVAL_MS[target_interval]
    if target_ms % source_interval_ms != 0:
        raise ValueError(
            f"target_interval {target_interval!r} ({target_ms} ms) is not a "
            f"whole multiple of source_interval_ms {source_interval_ms} ms"
        )

    factor = target_ms // source_interval_ms
    if factor <= 1:
        # Already at or finer than target — return a shallow copy
        return list(candles)

    if not candles:
        return []

    result: list[Candle] = []
    symbol = candles[0].symbol
    n = len(candles)
    i = 0

    while i + factor <= n:
        group = candles[i : i + factor]

        # Verify strict consecutive order within the group.
        # If candle j is not exactly source_interval_ms after candle j-1,
        # there is a real market gap.  We advance i to re-anchor at the
        # gap-causing candle (group[j]) and try again.
        gap_at: int = -1
        for j in range(1, factor):
            if group[j].open_time != group[j - 1].open_time + source_interval_ms:
                gap_at = j
                break

        if gap_at >= 0:
            # Skip to the candle that broke the sequence
            i += gap_at
            continue

        first = group[0]
        last = group[-1]

        agg = Candle(
            symbol=symbol,
            interval=target_interval,
            open_time=first.open_time,
            open=first.open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=last.close,
            volume=sum((c.volume for c in group), _ZERO),
            close_time=last.close_time,
            quote_asset_volume=sum((c.quote_asset_volume for c in group), _ZERO),
            trades=sum(c.trades for c in group),
            taker_buy_base_volume=sum((c.taker_buy_base_volume for c in group), _ZERO),
            taker_buy_quote_volume=sum((c.taker_buy_quote_volume for c in group), _ZERO),
            is_closed=True,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        result.append(agg)
        i += factor

    return result


def warmup_len_for(candles: list[Candle], start_ms: int) -> int:
    """Count candles with open_time strictly before *start_ms*.

    This gives the warmup prefix length for a candle list that was built
    with extra history prepended before the evaluation period.
    """
    return sum(1 for c in candles if c.open_time < start_ms)
