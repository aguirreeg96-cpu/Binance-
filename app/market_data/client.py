"""
Market data HTTP client.

MarketDataClient — abstract interface (read-only, no order methods).
BinanceMarketDataClient — httpx implementation with retry / rate-limit logic.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx

from app.market_data.exceptions import (
    BannedError,
    ClientClosedError,
    MarketDataError,
    MaxRetriesError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Forbidden path fragments — belt-and-suspenders guard so this client can
# never accidentally call trading endpoints even via misconfiguration.
_FORBIDDEN_PATH_FRAGMENTS = frozenset(
    {
        "order",
        "trade",
        "balance",
        "account",
        "withdraw",
        "cancel",
        "deposit",
        "asset",
        "margin",
        "futures",
        "leveraged",
    }
)


class MarketDataClient(ABC):
    """
    Read-only market data interface.

    Deliberately contains NO methods related to orders, trades, balances,
    accounts, withdrawals, or cancellations.
    """

    @abstractmethod
    async def get_server_time(self) -> int:
        """Return Binance server time in milliseconds UTC."""

    @abstractmethod
    async def get_exchange_info(self, symbol: str) -> dict:
        """Return exchangeInfo for a single symbol (uses short TTL cache)."""

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        """Return raw kline arrays from Binance."""

    @abstractmethod
    async def close(self) -> None:
        """Release underlying HTTP resources."""

    async def __aenter__(self) -> "MarketDataClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


class BinanceMarketDataClient(MarketDataClient):
    """
    Binance public market data client.

    - Uses a single shared httpx.AsyncClient (created lazily).
    - Reads X-MBX-USED-WEIGHT-1M on every response.
    - 429: waits Retry-After (capped), retries.
    - 418: raises BannedError immediately, no retry.
    - 5xx / network errors: exponential backoff with jitter.
    - 4xx (other): raises immediately, no retry.
    - Injectable _sleep_fn and _jitter_fn for deterministic tests.
    """

    EXCHANGE_INFO_TTL = 300.0  # seconds

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_retry_after: int = 60,
        _sleep_fn: Callable[[float], Any] | None = None,
        _jitter_fn: Callable[[], float] | None = None,
        _now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_retry_after = max_retry_after
        self._sleep = _sleep_fn or asyncio.sleep
        self._jitter = _jitter_fn or (lambda: random.uniform(0, 1))
        self._now = _now_fn or time.monotonic

        self._client: httpx.AsyncClient | None = None
        self._closed = False
        self._used_weight: int = 0
        self._exchange_info_cache: dict[str, tuple[float, dict]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise ClientClosedError(
                "BinanceMarketDataClient has been closed. Create a new instance."
            )
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    def _guard_path(self, path: str) -> None:
        path_lower = path.lower()
        for fragment in _FORBIDDEN_PATH_FRAGMENTS:
            if fragment in path_lower:
                raise MarketDataError(
                    f"Path {path!r} contains forbidden fragment {fragment!r}. "
                    "MarketDataClient cannot call trading endpoints."
                )

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        self._guard_path(path)
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await client.request(method, path, **kwargs)

                # Track API weight consumption
                weight_header = response.headers.get("X-MBX-USED-WEIGHT-1M")
                if weight_header:
                    try:
                        self._used_weight = int(weight_header)
                    except ValueError:
                        pass

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 429:
                    raw_after = response.headers.get("Retry-After", "60")
                    try:
                        retry_after = max(0, min(int(raw_after), self._max_retry_after))
                    except ValueError:
                        retry_after = min(60, self._max_retry_after)

                    logger.warning(
                        "Rate limited (HTTP 429), waiting %ds (attempt %d/%d)",
                        retry_after,
                        attempt + 1,
                        self._max_retries + 1,
                    )
                    if attempt < self._max_retries:
                        await self._sleep(float(retry_after))
                        continue
                    raise RateLimitError(retry_after)

                if response.status_code == 418:
                    raw_after = response.headers.get("Retry-After")
                    banned_after: int | None = int(raw_after) if raw_after else None
                    logger.error("IP banned by Binance (HTTP 418)")
                    raise BannedError(banned_after)

                if 500 <= response.status_code < 600:
                    last_exc = MarketDataError(f"Binance server error HTTP {response.status_code}")
                    if attempt < self._max_retries:
                        wait = (2.0**attempt) + self._jitter()
                        logger.warning(
                            "Binance 5xx (attempt %d/%d), retrying in %.2fs",
                            attempt + 1,
                            self._max_retries + 1,
                            wait,
                        )
                        await self._sleep(wait)
                        continue
                    raise MaxRetriesError(attempt + 1, last_exc)

                # 4xx other than 429/418 — do not retry
                try:
                    body = response.json()
                    msg = body.get("msg", response.text[:200])
                except Exception:
                    msg = response.text[:200]
                raise MarketDataError(f"HTTP {response.status_code}: {msg}")

            except (BannedError, RateLimitError, MarketDataError):
                raise
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = (2.0**attempt) + self._jitter()
                    logger.warning(
                        "Network error (attempt %d/%d), retrying in %.2fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        wait,
                        exc,
                    )
                    await self._sleep(wait)
                    continue
                raise MaxRetriesError(attempt + 1, exc) from exc

        raise MaxRetriesError(self._max_retries + 1, last_exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def get_server_time(self) -> int:
        data = await self._request("GET", "/api/v3/time")
        return int(data["serverTime"])

    async def get_exchange_info(self, symbol: str) -> dict:
        symbol = symbol.upper()
        cached = self._exchange_info_cache.get(symbol)
        if cached is not None:
            ts, data = cached
            if self._now() - ts < self.EXCHANGE_INFO_TTL:
                logger.debug("exchange_info cache hit for %s", symbol)
                return data

        logger.debug("Fetching exchange_info for %s", symbol)
        data = await self._request("GET", "/api/v3/exchangeInfo", params={"symbol": symbol})
        self._exchange_info_cache[symbol] = (self._now(), data)
        return data

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
            "timeZone": "0",
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        return await self._request("GET", "/api/v3/klines", params=params)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._closed = True

    @property
    def used_weight(self) -> int:
        return self._used_weight
