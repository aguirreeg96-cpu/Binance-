"""Backtesting-specific exceptions."""


class BacktestError(Exception):
    """Base class for all backtesting errors."""


class BacktestInsufficientDataError(BacktestError):
    """Raised when there are not enough candles to run the backtest."""


class BacktestConfigError(BacktestError):
    """Raised for invalid backtest configuration."""
