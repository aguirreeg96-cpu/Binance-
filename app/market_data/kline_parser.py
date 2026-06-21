"""Parse raw Binance kline arrays into typed KlineData objects."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.market_data.exceptions import KlineParseError, KlineValidationError

_MIN_FIELDS = 11  # Binance returns 12, index 11 is unused


@dataclass(frozen=True)
class KlineData:
    """Immutable, validated representation of a single Binance kline."""

    symbol: str
    interval: str
    open_time: int  # milliseconds UTC (inclusive open)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal  # base asset volume
    close_time: int  # milliseconds UTC (inclusive close)
    quote_asset_volume: Decimal
    number_of_trades: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: list, symbol: str, interval: str) -> "KlineData":
        """
        Parse a raw Binance kline list.

        Binance format (indices):
          0  open_time (int ms)
          1  open (str)
          2  high (str)
          3  low (str)
          4  close (str)
          5  volume / base asset volume (str)
          6  close_time (int ms)
          7  quote_asset_volume (str)
          8  number_of_trades (int)
          9  taker_buy_base_volume (str)
          10 taker_buy_quote_volume (str)
          11 ignored
        """
        if len(raw) < _MIN_FIELDS:
            raise KlineParseError(
                f"Expected at least {_MIN_FIELDS} fields in kline, got {len(raw)}"
            )
        try:
            return cls(
                symbol=symbol.upper(),
                interval=interval,
                open_time=int(raw[0]),
                open=_to_decimal(raw[1], "open"),
                high=_to_decimal(raw[2], "high"),
                low=_to_decimal(raw[3], "low"),
                close=_to_decimal(raw[4], "close"),
                volume=_to_decimal(raw[5], "volume"),
                close_time=int(raw[6]),
                quote_asset_volume=_to_decimal(raw[7], "quote_asset_volume"),
                number_of_trades=int(raw[8]),
                taker_buy_base_volume=_to_decimal(raw[9], "taker_buy_base_volume"),
                taker_buy_quote_volume=_to_decimal(raw[10], "taker_buy_quote_volume"),
            )
        except KlineParseError:
            raise
        except Exception as exc:
            raise KlineParseError(f"Failed to parse kline: {exc}") from exc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise KlineValidationError if OHLCV invariants are violated."""
        errors: list[str] = []

        for name, val in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
            ("quote_asset_volume", self.quote_asset_volume),
            ("taker_buy_base_volume", self.taker_buy_base_volume),
            ("taker_buy_quote_volume", self.taker_buy_quote_volume),
        ]:
            if val < 0:
                errors.append(f"{name} is negative ({val})")

        if self.high < self.open:
            errors.append(f"high ({self.high}) < open ({self.open})")
        if self.high < self.close:
            errors.append(f"high ({self.high}) < close ({self.close})")
        if self.high < self.low:
            errors.append(f"high ({self.high}) < low ({self.low})")
        if self.low > self.open:
            errors.append(f"low ({self.low}) > open ({self.open})")
        if self.low > self.close:
            errors.append(f"low ({self.low}) > close ({self.close})")
        if self.open_time >= self.close_time:
            errors.append(f"open_time ({self.open_time}) >= close_time ({self.close_time})")

        if errors:
            raise KlineValidationError(
                f"Kline {self.symbol} @{self.open_time} failed validation: " + "; ".join(errors)
            )


def _to_decimal(value: object, field: str) -> Decimal:
    """Convert value → Decimal without float intermediary."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise KlineParseError(f"Cannot convert field {field!r} to Decimal: {value!r}") from exc
