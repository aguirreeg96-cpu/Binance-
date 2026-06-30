"""Stage 6.1 tests: sync_forward_candles — candle download scheduling and guards."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.forward.manifest import FORWARD_SOURCE_INTERVAL, FORWARD_SYMBOL
from app.forward.sync import WARMUP_LOOKBACK_DAYS, sync_forward_candles
from app.market_data.client import MarketDataClient
from app.market_data.historical_service import DownloadResult
from app.market_data.interval_utils import INTERVAL_MS
from app.models.candle import Candle

_15M_MS = INTERVAL_MS["15m"]
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _fake_result(inserted: int = 5) -> DownloadResult:
    return DownloadResult(
        symbol=FORWARD_SYMBOL,
        interval=FORWARD_SOURCE_INTERVAL,
        requested_start=_NOW - timedelta(days=1),
        requested_end=_NOW,
        inserted=inserted,
        requests_made=1,
    )


def _stored_candle(open_time_ms: int) -> Candle:
    return Candle(
        symbol=FORWARD_SYMBOL,
        interval=FORWARD_SOURCE_INTERVAL,
        open_time=open_time_ms,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("1"),
        close_time=open_time_ms + _15M_MS - 1,
        quote_asset_volume=Decimal("100"),
        trades=10,
        taker_buy_base_volume=Decimal("0.5"),
        taker_buy_quote_volume=Decimal("50"),
        is_closed=True,
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )


class TestSyncForwardCandles:
    @pytest.mark.asyncio
    async def test_cold_start_fetches_from_warmup_lookback(self, db_session):
        mock_client = MagicMock(spec=MarketDataClient)

        with patch("app.forward.sync.HistoricalDataService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.download = AsyncMock(return_value=_fake_result(10))

            result = await sync_forward_candles(db_session, mock_client, now=_NOW)

        assert result is not None
        assert result.inserted == 10

        call_kwargs = mock_instance.download.call_args.kwargs
        expected_start = _NOW - timedelta(days=WARMUP_LOOKBACK_DAYS)
        assert abs((call_kwargs["start"] - expected_start).total_seconds()) < 1
        assert call_kwargs["symbol"] == FORWARD_SYMBOL
        assert call_kwargs["interval"] == FORWARD_SOURCE_INTERVAL
        assert call_kwargs["include_open_candle"] is False

    @pytest.mark.asyncio
    async def test_incremental_start_from_after_latest_stored(self, db_session):
        candle_open_ms = int(_NOW.timestamp() * 1000) - 2 * _15M_MS
        db_session.add(_stored_candle(candle_open_ms))
        db_session.commit()

        mock_client = MagicMock(spec=MarketDataClient)

        with patch("app.forward.sync.HistoricalDataService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.download = AsyncMock(return_value=_fake_result(1))

            result = await sync_forward_candles(db_session, mock_client, now=_NOW)

        assert result is not None
        call_kwargs = mock_instance.download.call_args.kwargs
        expected_start_ms = candle_open_ms + _15M_MS
        expected_start = datetime.fromtimestamp(expected_start_ms / 1000.0, tz=UTC)
        assert call_kwargs["start"] == expected_start

    @pytest.mark.asyncio
    async def test_returns_none_when_already_caught_up(self, db_session):
        now_ms = int(_NOW.timestamp() * 1000)
        # Latest stored candle's "next open" == now → start >= now → None
        db_session.add(_stored_candle(now_ms - _15M_MS))
        db_session.commit()

        mock_client = MagicMock(spec=MarketDataClient)

        with patch("app.forward.sync.HistoricalDataService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.download = AsyncMock(return_value=_fake_result())
            result = await sync_forward_candles(db_session, mock_client, now=_NOW)

        assert result is None
        mock_instance.download.assert_not_called()

    @pytest.mark.asyncio
    async def test_end_is_bounded_by_now(self, db_session):
        mock_client = MagicMock(spec=MarketDataClient)

        with patch("app.forward.sync.HistoricalDataService") as MockService:
            mock_instance = MockService.return_value
            mock_instance.download = AsyncMock(return_value=_fake_result())

            await sync_forward_candles(db_session, mock_client, now=_NOW)

        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["end"] == _NOW


class TestArchitectureGuard:
    def test_sync_module_does_not_send_orders(self):
        import app.forward.sync as sync_mod

        with open(sync_mod.__file__, encoding="utf-8") as f:
            src = f.read()
        for forbidden in ("place_order", "cancel_order", "api_key", "api_secret"):
            assert forbidden not in src

    def test_sync_uses_only_public_market_data_client(self):
        import app.forward.sync as sync_mod

        with open(sync_mod.__file__, encoding="utf-8") as f:
            src = f.read()
        assert "BinanceMarketDataClient" not in src
