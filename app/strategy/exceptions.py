"""Exceptions for the strategy engine."""


class StrategyError(Exception):
    """Base class for all strategy errors."""


class InvalidStrategyConfigError(StrategyError):
    """Strategy configuration is structurally invalid."""


class InsufficientDataForStrategyError(StrategyError):
    """Not enough candle data to run the strategy."""

    def __init__(self, symbol: str, interval: str) -> None:
        super().__init__(
            f"No closed candles found for {symbol} {interval}. "
            "Download historical data before running the strategy."
        )
        self.symbol = symbol
        self.interval = interval
