"""Typed exceptions for the market data layer."""


class MarketDataError(Exception):
    """Base for all market data errors."""


class KlineParseError(MarketDataError):
    """Raw kline response could not be parsed."""


class KlineValidationError(MarketDataError):
    """Parsed kline failed OHLCV invariant checks."""


class RateLimitError(MarketDataError):
    """HTTP 429 — too many requests. retry_after is in seconds."""

    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limited by Binance. Retry after {retry_after}s.")


class BannedError(MarketDataError):
    """HTTP 418 — IP banned by Binance."""

    def __init__(self, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        msg = "IP banned by Binance (HTTP 418)."
        if retry_after is not None:
            msg += f" Retry after {retry_after}s."
        super().__init__(msg)


class MaxRetriesError(MarketDataError):
    """Exhausted retry budget for a single request."""

    def __init__(self, attempts: int, cause: Exception | None = None) -> None:
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"Request failed after {attempts} attempts: {cause}")


class PaginationStallError(MarketDataError):
    """Pagination is not advancing — likely an API or logic bug."""


class InvalidSymbolError(MarketDataError):
    """Symbol is not valid or not in TRADING status."""

    def __init__(self, symbol: str, reason: str = "") -> None:
        self.symbol = symbol
        super().__init__(f"Invalid symbol {symbol!r}: {reason}" if reason else f"Invalid symbol {symbol!r}")


class InvalidIntervalError(MarketDataError):
    """Interval string is not a recognised Binance kline interval."""

    def __init__(self, interval: str) -> None:
        self.interval = interval
        super().__init__(f"Invalid interval {interval!r}")


class MaxRequestsError(MarketDataError):
    """Download job exceeded the configured max_requests limit."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"Download exceeded max_requests limit of {limit}.")


class ClientClosedError(MarketDataError):
    """Attempted to use a client that has already been closed."""
