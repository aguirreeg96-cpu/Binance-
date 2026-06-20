"""Tests for download pagination logic (mocked client, mocked DB)."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.market_data.exceptions import (
    InvalidIntervalError,
    InvalidSymbolError,
    MaxRequestsError,
    PaginationStallError,
)
from app.market_data.historical_service import HistoricalDataService, _validate_raw_batch
from app.market_data.kline_parser import KlineData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTERVAL_MS = 900_000  # 15m in ms
_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
_END = datetime(2025, 1, 2, tzinfo=timezone.utc)
_START_MS = int(_START.timestamp() * 1000)
_END_MS = int(_END.timestamp() * 1000)


def _make_raw(open_time: int) -> list:
    """Create a minimal valid raw kline with open_time."""
    close_time = open_time + _INTERVAL_MS - 1
    return [
        open_time, "35000.00", "35500.00", "34800.00", "35200.00",
        "100.00", close_time, "3500000.00", 1000,
        "50.00", "1750000.00", "0",
    ]


def _make_page(start_ms: int, count: int) -> list[list]:
    """Generate `count` sequential klines starting at start_ms."""
    return [_make_raw(start_ms + i * _INTERVAL_MS) for i in range(count)]


def _mock_client(pages: list[list[list]], server_time: int = _END_MS + 1_000_000) -> AsyncMock:
    """Build a mock MarketDataClient that returns pages sequentially."""
    client = AsyncMock()
    client.get_server_time.return_value = server_time
    client.get_exchange_info.return_value = {
        "symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "permissions": ["SPOT"]}]
    }
    client.get_klines = AsyncMock(side_effect=pages)
    return client


def _mock_session() -> MagicMock:
    """Minimal session mock with a working flush."""
    session = MagicMock()
    session.scalars.return_value.all.return_value = []
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyAndSinglePage:
    @pytest.mark.asyncio
    async def test_empty_first_page_returns_zero(self):
        client = _mock_client([[]])
        service = HistoricalDataService(client=client, max_requests=10)
        session = _mock_session()

        result = await service.download(
            session=session, symbol="BTCUSDT", interval="15m",
            start=_START, end=_END,
        )
        assert result.received == 0
        assert result.requests_made == 1

    @pytest.mark.asyncio
    async def test_single_page_less_than_1000(self):
        page = _make_page(_START_MS, 96)  # 96 candles for 24h of 15m
        client = _mock_client([page, []])
        service = HistoricalDataService(client=client, max_requests=10)
        session = _mock_session()

        result = await service.download(
            session=session, symbol="BTCUSDT", interval="15m",
            start=_START, end=_END,
        )
        assert result.received == 96
        assert result.requests_made == 1  # stopped after < 1000


class TestMultiPagePagination:
    @pytest.mark.asyncio
    async def test_two_full_pages_then_partial(self):
        page1 = _make_page(_START_MS, 1000)
        page2_start = _START_MS + 1000 * _INTERVAL_MS
        page2 = _make_page(page2_start, 500)
        client = _mock_client([page1, page2])
        service = HistoricalDataService(client=client, max_requests=10)
        session = _mock_session()

        # End must be after page2's last candle
        end = datetime.fromtimestamp(
            (page2_start + 600 * _INTERVAL_MS) / 1000, tz=timezone.utc
        )
        result = await service.download(
            session=session, symbol="BTCUSDT", interval="15m",
            start=_START, end=end,
        )
        assert result.requests_made == 2

    @pytest.mark.asyncio
    async def test_max_requests_exceeded_raises(self):
        # Return full pages endlessly
        def endless_pages(**_kwargs):
            start = getattr(endless_pages, "_start", _START_MS)
            page = _make_page(start, 1000)
            endless_pages._start = start + 1000 * _INTERVAL_MS
            return page

        client = AsyncMock()
        client.get_server_time.return_value = _END_MS + 10_000_000
        client.get_exchange_info.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "permissions": ["SPOT"]}]
        }
        client.get_klines = AsyncMock(side_effect=endless_pages)

        service = HistoricalDataService(client=client, max_requests=3)
        far_end = datetime(2030, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(MaxRequestsError):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=_START, end=far_end,
            )


class TestStallDetection:
    @pytest.mark.asyncio
    async def test_repeated_last_open_time_raises(self):
        page1 = _make_page(_START_MS, 1000)
        # Single candle with same open_time as page1's last — triggers stall detection
        page2 = [_make_raw(_START_MS + 999 * _INTERVAL_MS)]

        client = _mock_client([page1, page2])
        service = HistoricalDataService(client=client, max_requests=10)

        with pytest.raises(PaginationStallError, match="stalled"):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=_START, end=datetime(2030, 1, 1, tzinfo=timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_full_page_repeated_raises(self):
        page = _make_page(_START_MS, 1000)
        # Both calls return exactly the same page
        client = _mock_client([page, page])
        service = HistoricalDataService(client=client, max_requests=10)

        with pytest.raises(PaginationStallError):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=_START, end=datetime(2030, 1, 1, tzinfo=timezone.utc),
            )


class TestBatchValidation:
    def test_duplicate_open_times_rejected(self):
        batch = [
            _make_raw(_START_MS),
            _make_raw(_START_MS),  # duplicate
        ]
        with pytest.raises(PaginationStallError, match="Duplicate"):
            _validate_raw_batch(batch, _START_MS)

    def test_descending_order_rejected(self):
        batch = [
            _make_raw(_START_MS + _INTERVAL_MS),
            _make_raw(_START_MS),  # earlier candle after later one
        ]
        with pytest.raises(PaginationStallError, match="ascending"):
            _validate_raw_batch(batch, _START_MS)

    def test_valid_batch_passes(self):
        batch = _make_page(_START_MS, 5)
        _validate_raw_batch(batch, _START_MS)  # no exception


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_start_equals_end_raises(self):
        client = _mock_client([[]])
        service = HistoricalDataService(client=client, max_requests=10)
        with pytest.raises(ValueError, match="before end"):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=_START, end=_START,
            )

    @pytest.mark.asyncio
    async def test_start_after_end_raises(self):
        client = _mock_client([[]])
        service = HistoricalDataService(client=client, max_requests=10)
        with pytest.raises(ValueError, match="before end"):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=_END, end=_START,
            )

    @pytest.mark.asyncio
    async def test_naive_start_datetime_rejected(self):
        client = _mock_client([[]])
        service = HistoricalDataService(client=client, max_requests=10)
        naive = datetime(2025, 1, 1)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=naive, end=_END,
            )

    @pytest.mark.asyncio
    async def test_naive_end_datetime_rejected(self):
        client = _mock_client([[]])
        service = HistoricalDataService(client=client, max_requests=10)
        naive = datetime(2025, 1, 2)
        with pytest.raises(ValueError, match="timezone-aware"):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=_START, end=naive,
            )

    @pytest.mark.asyncio
    async def test_invalid_interval_raises(self):
        client = _mock_client([[]])
        service = HistoricalDataService(client=client, max_requests=10)
        with pytest.raises(InvalidIntervalError):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="99x",
                start=_START, end=_END,
            )

    @pytest.mark.asyncio
    async def test_invalid_symbol_raises(self):
        client = AsyncMock()
        client.get_server_time.return_value = _END_MS + 1_000_000
        client.get_exchange_info.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "status": "HALTED", "permissions": ["SPOT"]}]
        }
        service = HistoricalDataService(client=client, max_requests=10)
        with pytest.raises(InvalidSymbolError):
            await service.download(
                session=_mock_session(), symbol="BTCUSDT", interval="15m",
                start=_START, end=_END,
            )

    @pytest.mark.asyncio
    async def test_include_open_candle_skips_server_time(self):
        """When include_open_candle=True, server time is not fetched."""
        page = _make_page(_START_MS, 5)
        client = _mock_client([page])
        service = HistoricalDataService(client=client, max_requests=10)

        await service.download(
            session=_mock_session(), symbol="BTCUSDT", interval="15m",
            start=_START, end=_END, include_open_candle=True,
        )
        client.get_server_time.assert_not_called()
