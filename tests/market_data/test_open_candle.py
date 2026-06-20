"""Tests for open-candle filtering logic."""

from datetime import datetime, timezone

import pytest

from app.market_data.kline_parser import KlineData

# A candle's close_time is the LAST millisecond of the interval.
# A candle is closed when server_time_ms > close_time  (strictly greater).
# At exact equality (server_time_ms == close_time) we treat it as still open
# because Binance may not have finalised it yet.

_OPEN_TIME = 1_700_000_000_000
_CLOSE_TIME = 1_700_003_599_999  # 1-hour candle, last ms inclusive


def _make_kline(open_time=_OPEN_TIME, close_time=_CLOSE_TIME) -> KlineData:
    return KlineData(
        symbol="BTCUSDT", interval="1h",
        open_time=open_time,
        open=__import__("decimal").Decimal("35000"),
        high=__import__("decimal").Decimal("35500"),
        low=__import__("decimal").Decimal("34800"),
        close=__import__("decimal").Decimal("35200"),
        volume=__import__("decimal").Decimal("100"),
        close_time=close_time,
        quote_asset_volume=__import__("decimal").Decimal("3500000"),
        number_of_trades=1500,
        taker_buy_base_volume=__import__("decimal").Decimal("50"),
        taker_buy_quote_volume=__import__("decimal").Decimal("1750000"),
    )


def _is_closed(kline: KlineData, server_time_ms: int) -> bool:
    """Filter used in historical_service: closed when close_time < server_time_ms."""
    return kline.close_time < server_time_ms


class TestOpenCandleFilter:
    def test_candle_closed_when_server_time_after(self):
        kline = _make_kline(close_time=_CLOSE_TIME)
        server_time = _CLOSE_TIME + 1  # 1ms past close
        assert _is_closed(kline, server_time) is True

    def test_candle_open_when_server_time_before(self):
        kline = _make_kline(close_time=_CLOSE_TIME)
        server_time = _CLOSE_TIME - 1000  # still 1s to go
        assert _is_closed(kline, server_time) is False

    def test_boundary_close_time_equals_server_time_treated_as_open(self):
        """At exact equality the candle is NOT yet considered closed."""
        kline = _make_kline(close_time=_CLOSE_TIME)
        server_time = _CLOSE_TIME  # exactly at boundary → still open
        assert _is_closed(kline, server_time) is False

    def test_candle_closed_one_ms_after_boundary(self):
        kline = _make_kline(close_time=_CLOSE_TIME)
        server_time = _CLOSE_TIME + 1
        assert _is_closed(kline, server_time) is True

    def test_uses_close_time_not_open_time(self):
        """Filter looks at close_time, not open_time."""
        kline = _make_kline(open_time=_OPEN_TIME, close_time=_CLOSE_TIME)
        # server_time past open but before close → still open
        server_time = _OPEN_TIME + 1000
        assert _is_closed(kline, server_time) is False

    def test_future_candle_is_open(self):
        """A candle whose close_time is in the far future is open."""
        far_future = _CLOSE_TIME + 86_400_000  # +1 day
        kline = _make_kline(close_time=far_future)
        server_time = _CLOSE_TIME
        assert _is_closed(kline, server_time) is False

    def test_very_old_candle_is_always_closed(self):
        past_close = 1_000_000_000_000  # year 2001
        kline = _make_kline(close_time=past_close)
        server_time = 1_700_000_000_000  # now
        assert _is_closed(kline, server_time) is True

    def test_filter_batch_mixed(self):
        """Simulate filtering a batch containing closed and open candles."""
        server_time = _CLOSE_TIME + 1

        closed_kline = _make_kline(close_time=_CLOSE_TIME)
        open_kline = _make_kline(
            open_time=_CLOSE_TIME + 1,
            close_time=_CLOSE_TIME + 3_600_000,
        )

        batch = [closed_kline, open_kline]
        filtered = [k for k in batch if _is_closed(k, server_time)]

        assert len(filtered) == 1
        assert filtered[0].close_time == _CLOSE_TIME
