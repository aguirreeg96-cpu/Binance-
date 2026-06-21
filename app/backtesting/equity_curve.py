"""EquityCurveBuilder — accumulates EquityPoint objects during a backtest run."""

from decimal import Decimal

from app.backtesting.portfolio import PortfolioState
from app.backtesting.schemas import EquityPoint
from app.models.candle import Candle

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


class EquityCurveBuilder:
    """Accumulate equity points during a backtest run, then build the final list."""

    def __init__(self) -> None:
        self._points: list[EquityPoint] = []

    def append(self, candle: Candle, portfolio: PortfolioState, equity: Decimal) -> None:
        """Record an equity point for this candle.

        Call AFTER updating portfolio.peak_equity to ensure correct drawdown.
        """
        drawdown_pct = (
            (portfolio.peak_equity - equity) / portfolio.peak_equity * _HUNDRED
            if portfolio.peak_equity > _ZERO
            else _ZERO
        )
        self._points.append(
            EquityPoint(
                open_time=candle.open_time,
                close_time=candle.close_time,
                close_price=candle.close,
                equity=equity,
                quote_balance=portfolio.quote_balance,
                base_balance=portfolio.base_balance,
                base_value=portfolio.base_balance * candle.close,
                drawdown_pct=drawdown_pct,
                peak_equity=portfolio.peak_equity,
                has_open_position=portfolio.open_position is not None,
            )
        )

    def build(self) -> list[EquityPoint]:
        """Return the accumulated equity curve."""
        return list(self._points)
