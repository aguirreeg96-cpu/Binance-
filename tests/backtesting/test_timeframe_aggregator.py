"""Tests for Stage 5.2C — timeframe_aggregator.

Covers:
  - 15m → 30m aggregation: OHLCV correctness
  - 15m → 1h aggregation: OHLCV correctness
  - Incomplete trailing group dropped
  - Gap detection within a group: group is skipped
  - Real market gap preserved: no candle invented to fill it
  - No look-ahead: each aggregated candle uses only its own source group
  - Factor ≤ 1 returns shallow copy
  - Errors: unknown interval, non-multiple source interval
  - warmup_len_for: boundary conditions
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.backtesting.timeframe_aggregator import aggregate_candles, warmup_len_for
from app.models.candle import Candle

_INTERVAL_15M = 900_000
_INTERVAL_30M = 1_800_000
_INTERVAL_1H = 3_600_000
_BASE_TIME = 0


def _make_candle(
    open_time: int,
    open_p: str = "100",
    high_p: str = "110",
    low_p: str = "90",
    close_p: str = "105",
    volume: str = "1000",
    symbol: str = "BTCUSDT",
    interval: str = "15m",
) -> Candle:
    close_time = open_time + _INTERVAL_15M - 1
    p_close = Decimal(close_p)
    p_vol = Decimal(volume)
    return Candle(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        open=Decimal(open_p),
        high=Decimal(high_p),
        low=Decimal(low_p),
        close=p_close,
        volume=p_vol,
        close_time=close_time,
        quote_asset_volume=p_vol * p_close,
        trades=100,
        taker_buy_base_volume=p_vol / Decimal("2"),
        taker_buy_quote_volume=p_vol / Decimal("2") * p_close,
        is_closed=True,
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )


def _make_15m_sequence(n: int, base_open_time: int = _BASE_TIME) -> list[Candle]:
    return [
        _make_candle(
            open_time=base_open_time + i * _INTERVAL_15M,
            open_p=str(100 + i),
            high_p=str(110 + i),
            low_p=str(90 + i),
            close_p=str(105 + i),
            volume=str(1000 + i * 10),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 15m → 30m
# ---------------------------------------------------------------------------


class TestAggregateTo30m:
    def test_basic_ohlc_open_is_first(self):
        """Aggregated open = first source candle's open."""
        candles = [
            _make_candle(0, open_p="100", high_p="115", low_p="95", close_p="108"),
            _make_candle(_INTERVAL_15M, open_p="109", high_p="120", low_p="100", close_p="112"),
        ]
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert len(result) == 1
        assert result[0].open == Decimal("100")

    def test_basic_ohlc_high_is_max(self):
        candles = [
            _make_candle(0, high_p="115"),
            _make_candle(_INTERVAL_15M, high_p="130"),
        ]
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].high == Decimal("130")

    def test_basic_ohlc_low_is_min(self):
        candles = [
            _make_candle(0, low_p="88"),
            _make_candle(_INTERVAL_15M, low_p="92"),
        ]
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].low == Decimal("88")

    def test_basic_ohlc_close_is_last(self):
        candles = [
            _make_candle(0, close_p="108"),
            _make_candle(_INTERVAL_15M, close_p="112"),
        ]
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].close == Decimal("112")

    def test_volume_summed(self):
        candles = _make_15m_sequence(2)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].volume == candles[0].volume + candles[1].volume

    def test_trades_summed(self):
        candles = _make_15m_sequence(2)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].trades == candles[0].trades + candles[1].trades

    def test_taker_buy_base_volume_summed(self):
        candles = _make_15m_sequence(2)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        expected = candles[0].taker_buy_base_volume + candles[1].taker_buy_base_volume
        assert result[0].taker_buy_base_volume == expected

    def test_taker_buy_quote_volume_summed(self):
        candles = _make_15m_sequence(2)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        expected = candles[0].taker_buy_quote_volume + candles[1].taker_buy_quote_volume
        assert result[0].taker_buy_quote_volume == expected

    def test_quote_asset_volume_summed(self):
        candles = _make_15m_sequence(2)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        expected = candles[0].quote_asset_volume + candles[1].quote_asset_volume
        assert result[0].quote_asset_volume == expected

    def test_interval_set_to_target(self):
        candles = _make_15m_sequence(2)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].interval == "30m"

    def test_open_time_is_first_source_open_time(self):
        candles = _make_15m_sequence(2, base_open_time=5_000_000)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].open_time == 5_000_000

    def test_is_closed_true(self):
        candles = _make_15m_sequence(2)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].is_closed is True

    def test_symbol_carried_through(self):
        candles = [
            _make_candle(0, symbol="ETHUSDT"),
            _make_candle(_INTERVAL_15M, symbol="ETHUSDT"),
        ]
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result[0].symbol == "ETHUSDT"

    def test_incomplete_trailing_group_dropped(self):
        """3 candles → 1 complete 30m group + 1 orphan (dropped)."""
        candles = _make_15m_sequence(3)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert len(result) == 1

    def test_four_candles_two_groups(self):
        candles = _make_15m_sequence(4)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert len(result) == 2

    def test_single_candle_produces_no_output(self):
        candles = _make_15m_sequence(1)
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert result == []

    def test_empty_input(self):
        result = aggregate_candles([], "30m", _INTERVAL_15M)
        assert result == []

    def test_gap_within_group_skipped(self):
        """If the second candle in a pair has wrong open_time, the pair is skipped."""
        candles = [
            _make_candle(0),
            _make_candle(_INTERVAL_15M * 2),  # gap: should be at _INTERVAL_15M
        ]
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert len(result) == 0

    def test_real_market_gap_between_groups_preserved(self):
        """Gap between complete groups → gap in output (no candle invented)."""
        before_gap = _make_15m_sequence(4)  # two 30m groups: [0,15m] and [30m,45m]
        gap_start = 6 * _INTERVAL_15M  # skip one 30m slot
        after_gap = [
            _make_candle(gap_start),
            _make_candle(gap_start + _INTERVAL_15M),
        ]
        all_candles = before_gap + after_gap
        result = aggregate_candles(all_candles, "30m", _INTERVAL_15M)
        assert len(result) == 3
        open_times = [c.open_time for c in result]
        assert open_times[0] == 0
        assert open_times[1] == _INTERVAL_30M
        assert open_times[2] == gap_start  # no candle invented for the gap

    def test_gap_at_start_of_sequence_handled(self):
        """When first group has gap, advance and try next pair."""
        candles = [
            _make_candle(0),
            _make_candle(_INTERVAL_15M * 2),  # gap in first pair → skip c0, retry from c1
            _make_candle(_INTERVAL_15M * 3),  # c1=2*15m, c2=3*15m → consecutive → OK
        ]
        # After gap_at=1, i becomes 1. Group [c1(2*15m), c2(3*15m)]: consecutive → emit.
        result = aggregate_candles(candles, "30m", _INTERVAL_15M)
        assert len(result) == 1
        assert result[0].open_time == _INTERVAL_15M * 2


# ---------------------------------------------------------------------------
# 15m → 1h
# ---------------------------------------------------------------------------


class TestAggregateTo1h:
    def test_four_candles_one_group(self):
        candles = _make_15m_sequence(4)
        result = aggregate_candles(candles, "1h", _INTERVAL_15M)
        assert len(result) == 1

    def test_ohlc_correctness(self):
        candles = [
            _make_candle(0 * _INTERVAL_15M, open_p="100", high_p="115", low_p="85", close_p="101"),
            _make_candle(1 * _INTERVAL_15M, open_p="101", high_p="112", low_p="90", close_p="108"),
            _make_candle(2 * _INTERVAL_15M, open_p="108", high_p="125", low_p="92", close_p="118"),
            _make_candle(3 * _INTERVAL_15M, open_p="118", high_p="119", low_p="95", close_p="109"),
        ]
        result = aggregate_candles(candles, "1h", _INTERVAL_15M)
        assert len(result) == 1
        c = result[0]
        assert c.open == Decimal("100")  # first open
        assert c.high == Decimal("125")  # max high
        assert c.low == Decimal("85")  # min low
        assert c.close == Decimal("109")  # last close

    def test_volume_sum_four_candles(self):
        candles = _make_15m_sequence(4)
        result = aggregate_candles(candles, "1h", _INTERVAL_15M)
        expected = sum(c.volume for c in candles)
        assert result[0].volume == expected

    def test_seven_candles_one_complete_group(self):
        """7 candles → 1 complete 1h group + 3 orphans (dropped)."""
        candles = _make_15m_sequence(7)
        result = aggregate_candles(candles, "1h", _INTERVAL_15M)
        assert len(result) == 1

    def test_eight_candles_two_groups(self):
        candles = _make_15m_sequence(8)
        result = aggregate_candles(candles, "1h", _INTERVAL_15M)
        assert len(result) == 2

    def test_interval_label(self):
        candles = _make_15m_sequence(4)
        result = aggregate_candles(candles, "1h", _INTERVAL_15M)
        assert result[0].interval == "1h"

    def test_gap_within_1h_group_skipped(self):
        """Gap in position 2 of a 1h group causes advance by 2, skipping partial group."""
        candles = [
            _make_candle(0 * _INTERVAL_15M),
            _make_candle(1 * _INTERVAL_15M),
            _make_candle(3 * _INTERVAL_15M),  # gap at position 2 (should be 2*15m)
            _make_candle(4 * _INTERVAL_15M),
            _make_candle(5 * _INTERVAL_15M),
            _make_candle(6 * _INTERVAL_15M),
        ]
        # group [0,1,3,4]: gap at j=2 → i += 2; group [3,4,5,6] consecutive → emit
        result = aggregate_candles(candles, "1h", _INTERVAL_15M)
        assert len(result) == 1
        assert result[0].open_time == 3 * _INTERVAL_15M


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_first_group_unaffected_by_second_group_data(self):
        """Aggregated candle uses only data from its source group."""
        group1 = [
            _make_candle(0 * _INTERVAL_15M, high_p="110"),
            _make_candle(1 * _INTERVAL_15M, high_p="108"),
        ]
        group2 = [
            _make_candle(2 * _INTERVAL_15M, high_p="200"),
            _make_candle(3 * _INTERVAL_15M, high_p="205"),
        ]
        result = aggregate_candles(group1 + group2, "30m", _INTERVAL_15M)
        assert len(result) == 2
        assert result[0].high == Decimal("110")  # no contamination from group 2
        assert result[1].high == Decimal("205")

    def test_factor_one_returns_shallow_copy(self):
        """When target_interval == source, factor=1 → shallow copy returned."""
        candles = _make_15m_sequence(3)
        result = aggregate_candles(candles, "15m", _INTERVAL_15M)
        assert len(result) == 3
        assert result is not candles  # shallow copy
        assert result[0] is candles[0]  # same object references


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_interval_raises(self):
        with pytest.raises(ValueError, match="Unknown target_interval"):
            aggregate_candles(_make_15m_sequence(4), "INVALID", _INTERVAL_15M)

    def test_non_multiple_source_raises(self):
        """Source interval that doesn't divide target raises ValueError."""
        with pytest.raises(ValueError):
            # 1h (3_600_000) % 700_000 != 0
            aggregate_candles(_make_15m_sequence(8), "1h", 700_000)


# ---------------------------------------------------------------------------
# warmup_len_for
# ---------------------------------------------------------------------------


class TestWarmupLenFor:
    def test_all_candles_before_start(self):
        candles = _make_15m_sequence(5)
        assert warmup_len_for(candles, 5 * _INTERVAL_15M) == 5

    def test_no_candles_before_start(self):
        candles = _make_15m_sequence(5)
        assert warmup_len_for(candles, 0) == 0

    def test_partial_warmup(self):
        candles = _make_15m_sequence(5)
        assert warmup_len_for(candles, 3 * _INTERVAL_15M) == 3

    def test_boundary_candle_not_warmup(self):
        """Candle with open_time == start_ms is NOT warmup (strictly before)."""
        candles = _make_15m_sequence(3)
        # candle[1].open_time == _INTERVAL_15M; start_ms == _INTERVAL_15M → not warmup
        assert warmup_len_for(candles, _INTERVAL_15M) == 1

    def test_empty_candles(self):
        assert warmup_len_for([], 1_000_000) == 0

    def test_aggregated_candles_warmup(self):
        """warmup_len_for works correctly on aggregated 1h candles."""
        candles_15m = _make_15m_sequence(12)  # 3 complete 1h groups
        agg_1h = aggregate_candles(candles_15m, "1h", _INTERVAL_15M)
        assert len(agg_1h) == 3
        # start_ms = 2 * _INTERVAL_1H → first 2 candles are warmup
        start_ms = 2 * _INTERVAL_1H
        assert warmup_len_for(agg_1h, start_ms) == 2
