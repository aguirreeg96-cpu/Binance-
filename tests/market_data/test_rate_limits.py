"""Tests for HTTP 429, 418, 5xx and network error handling in BinanceMarketDataClient."""

import httpx
import pytest
import respx

from app.market_data.client import BinanceMarketDataClient
from app.market_data.exceptions import BannedError, MaxRetriesError, RateLimitError

_BASE = "https://data-api.binance.vision"
_TIME_PATH = "/api/v3/time"
_TIME_RESPONSE = {"serverTime": 1700000000000}


def _make_client(
    max_retries=2, max_retry_after=10, **kwargs
) -> tuple[BinanceMarketDataClient, list[float]]:
    """Create client with injectable sleep and jitter (no real waiting in tests)."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    client = BinanceMarketDataClient(
        base_url=_BASE,
        max_retries=max_retries,
        max_retry_after=max_retry_after,
        _sleep_fn=fake_sleep,
        _jitter_fn=lambda: 0.0,  # deterministic backoff
        **kwargs,
    )
    return client, slept


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_429_retries_after_delay(self):
        client, slept = _make_client(max_retries=2, max_retry_after=10)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).side_effect = [
                httpx.Response(429, headers={"Retry-After": "5"}),
                httpx.Response(200, json=_TIME_RESPONSE),
            ]
            result = await client.get_server_time()

        assert result == 1700000000000
        assert len(slept) == 1
        assert slept[0] == 5.0

    @pytest.mark.asyncio
    async def test_429_retry_after_capped_at_max(self):
        client, slept = _make_client(max_retries=2, max_retry_after=10)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).side_effect = [
                httpx.Response(429, headers={"Retry-After": "9999"}),
                httpx.Response(200, json=_TIME_RESPONSE),
            ]
            await client.get_server_time()

        assert slept[0] == 10.0  # capped at max_retry_after

    @pytest.mark.asyncio
    async def test_429_retry_after_negative_becomes_zero(self):
        client, slept = _make_client(max_retries=2, max_retry_after=60)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).side_effect = [
                httpx.Response(429, headers={"Retry-After": "-5"}),
                httpx.Response(200, json=_TIME_RESPONSE),
            ]
            await client.get_server_time()

        assert slept[0] == 0.0  # max(0, -5) = 0

    @pytest.mark.asyncio
    async def test_429_exhausts_retries_raises_rate_limit_error(self):
        client, slept = _make_client(max_retries=1, max_retry_after=5)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).return_value = httpx.Response(429, headers={"Retry-After": "3"})
            with pytest.raises(RateLimitError) as exc_info:
                await client.get_server_time()

        assert exc_info.value.retry_after == 3
        assert len(slept) == 1  # retried once, then raised


class TestBanned:
    @pytest.mark.asyncio
    async def test_418_raises_banned_immediately(self):
        client, slept = _make_client(max_retries=3)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).return_value = httpx.Response(418, headers={"Retry-After": "300"})
            with pytest.raises(BannedError) as exc_info:
                await client.get_server_time()

        assert exc_info.value.retry_after == 300
        assert len(slept) == 0  # no sleep on 418

    @pytest.mark.asyncio
    async def test_418_without_retry_after_header(self):
        client, _ = _make_client(max_retries=3)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).return_value = httpx.Response(418)
            with pytest.raises(BannedError) as exc_info:
                await client.get_server_time()

        assert exc_info.value.retry_after is None


class TestServerErrors:
    @pytest.mark.asyncio
    async def test_503_retries_with_exponential_backoff(self):
        client, slept = _make_client(max_retries=2)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).side_effect = [
                httpx.Response(503),
                httpx.Response(503),
                httpx.Response(200, json=_TIME_RESPONSE),
            ]
            result = await client.get_server_time()

        assert result == 1700000000000
        assert len(slept) == 2
        # With jitter=0: 2^0=1, 2^1=2
        assert slept[0] == 1.0
        assert slept[1] == 2.0

    @pytest.mark.asyncio
    async def test_5xx_max_retries_exceeded_raises(self):
        client, slept = _make_client(max_retries=2)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).return_value = httpx.Response(500)
            with pytest.raises(MaxRetriesError) as exc_info:
                await client.get_server_time()

        assert exc_info.value.attempts == 3  # 1 original + 2 retries

    @pytest.mark.asyncio
    async def test_network_error_retried(self):
        client, slept = _make_client(max_retries=2)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).side_effect = [
                httpx.NetworkError("Connection reset"),
                httpx.Response(200, json=_TIME_RESPONSE),
            ]
            result = await client.get_server_time()

        assert result == 1700000000000
        assert len(slept) == 1

    @pytest.mark.asyncio
    async def test_timeout_retried(self):
        client, slept = _make_client(max_retries=1)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).side_effect = [
                httpx.TimeoutException("Timeout"),
                httpx.Response(200, json=_TIME_RESPONSE),
            ]
            result = await client.get_server_time()

        assert result == 1700000000000
        assert len(slept) == 1


class TestClientErrors:
    @pytest.mark.asyncio
    async def test_400_not_retried(self):
        client, slept = _make_client(max_retries=3)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).return_value = httpx.Response(
                400, json={"code": -1121, "msg": "Invalid symbol"}
            )
            from app.market_data.exceptions import MarketDataError

            with pytest.raises(MarketDataError, match="HTTP 400"):
                await client.get_server_time()

        assert len(slept) == 0  # no retries for 4xx

    @pytest.mark.asyncio
    async def test_401_not_retried(self):
        client, slept = _make_client(max_retries=3)

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).return_value = httpx.Response(401)
            from app.market_data.exceptions import MarketDataError

            with pytest.raises(MarketDataError, match="HTTP 401"):
                await client.get_server_time()

        assert len(slept) == 0


class TestRateLimitHeader:
    @pytest.mark.asyncio
    async def test_used_weight_tracked(self):
        client, _ = _make_client()

        with respx.mock(base_url=_BASE) as mock:
            mock.get(_TIME_PATH).return_value = httpx.Response(
                200,
                json=_TIME_RESPONSE,
                headers={"X-MBX-USED-WEIGHT-1M": "42"},
            )
            await client.get_server_time()

        assert client.used_weight == 42


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_closed_client_raises(self):
        client, _ = _make_client()
        await client.close()

        from app.market_data.exceptions import ClientClosedError

        with pytest.raises(ClientClosedError):
            await client.get_server_time()

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self):
        client, _ = _make_client()
        async with client:
            pass
        assert client._closed is True


class TestForbiddenPaths:
    @pytest.mark.asyncio
    async def test_order_path_blocked(self):
        client, _ = _make_client()
        from app.market_data.exceptions import MarketDataError

        with pytest.raises(MarketDataError, match="forbidden"):
            await client._request("POST", "/api/v3/order")

    @pytest.mark.asyncio
    async def test_account_path_blocked(self):
        client, _ = _make_client()
        from app.market_data.exceptions import MarketDataError

        with pytest.raises(MarketDataError, match="forbidden"):
            await client._request("GET", "/api/v3/account")

    @pytest.mark.asyncio
    async def test_klines_path_allowed(self):
        client, _ = _make_client()
        with respx.mock(base_url=_BASE) as mock:
            mock.get("/api/v3/klines").return_value = httpx.Response(200, json=[])
            result = await client.get_klines("BTCUSDT", "1h")
        assert result == []
