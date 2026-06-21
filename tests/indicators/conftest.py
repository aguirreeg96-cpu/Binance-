"""Shared fixtures for indicator tests."""

from decimal import Decimal

from app.models.candle import Candle

_BASE_TIME = 1_700_000_000_000
_INTERVAL_MS = 3_600_000  # 1h


def make_candle(
    idx: int = 0,
    open_: str = "100",
    high: str = "105",
    low: str = "95",
    close: str = "102",
    volume: str = "1000",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    is_closed: bool = True,
) -> Candle:
    """Build a Candle ORM object suitable for indicator tests (not persisted)."""
    c = Candle()
    c.id = None  # type: ignore[assignment]  # not persisted
    c.symbol = symbol
    c.interval = interval
    c.open_time = _BASE_TIME + idx * _INTERVAL_MS
    c.close_time = c.open_time + _INTERVAL_MS - 1
    c.open = Decimal(open_)
    c.high = Decimal(high)
    c.low = Decimal(low)
    c.close = Decimal(close)
    c.volume = Decimal(volume)
    c.quote_asset_volume = Decimal("0")
    c.trades = 0
    c.taker_buy_base_volume = Decimal("0")
    c.taker_buy_quote_volume = Decimal("0")
    c.is_closed = is_closed
    return c


def make_candles(closes: list[str], **kwargs: str) -> list[Candle]:
    """Build a series of candles where only the close price varies."""
    candles = []
    for i, close in enumerate(closes):
        price = Decimal(close)
        # Ensure OHLC invariants: high >= max(open,close), low <= min(open,close)
        high = str(price + Decimal("5"))
        low = str(price - Decimal("5"))
        candles.append(make_candle(idx=i, open_=close, high=high, low=low, close=close, **kwargs))
    return candles
