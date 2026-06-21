"""Shared fixtures for strategy tests."""

from decimal import Decimal

from app.indicators.schemas import CrossSignal, IndicatorResult
from app.strategy.config import StrategyEngineConfig

_BASE_TIME = 1_700_000_000_000
_INTERVAL_MS = 3_600_000  # 1h


def make_indicator_result(
    idx: int = 0,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    close: str = "35000",
    ema_short: str | None = "34000",
    ema_medium: str | None = "33000",
    ema_long: str | None = "30000",
    rsi: str | None = "55",
    atr: str | None = "500",
    volume_sma: str | None = "200",
    volume_ratio: str | None = "1.5",
    crossover: CrossSignal = CrossSignal.NONE,
    warmup_complete: bool = True,
) -> IndicatorResult:
    """Build an IndicatorResult suitable for strategy tests (not from DB)."""
    open_time = _BASE_TIME + idx * _INTERVAL_MS
    return IndicatorResult(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=open_time + _INTERVAL_MS - 1,
        close=Decimal(close),
        sma_short=None,
        sma_long=None,
        ema_short=Decimal(ema_short) if ema_short is not None else None,
        ema_medium=Decimal(ema_medium) if ema_medium is not None else None,
        ema_long=Decimal(ema_long) if ema_long is not None else None,
        rsi=Decimal(rsi) if rsi is not None else None,
        atr=Decimal(atr) if atr is not None else None,
        volume_sma=Decimal(volume_sma) if volume_sma is not None else None,
        volume_ratio=Decimal(volume_ratio) if volume_ratio is not None else None,
        ema_short_medium_cross=crossover,
        warmup_complete=warmup_complete,
        source_candle_id=None,
        calculated_at=None,
    )


def small_strategy_config() -> StrategyEngineConfig:
    """StrategyEngineConfig with default values (all rules enabled)."""
    return StrategyEngineConfig()


def buy_ready_result(idx: int = 0) -> IndicatorResult:
    """An IndicatorResult where ALL default BUY conditions are satisfied."""
    return make_indicator_result(
        idx=idx,
        close="35000",
        ema_short="34500",  # above ema_medium → crossed up
        ema_medium="34000",
        ema_long="30000",  # close > ema_long ✓
        rsi="57",  # in [50, 65] ✓
        volume_ratio="1.5",  # >= 1.0 ✓
        crossover=CrossSignal.BULLISH,
        warmup_complete=True,
    )


def sell_ready_result(idx: int = 0) -> IndicatorResult:
    """An IndicatorResult where the default SELL condition (bearish cross) is met."""
    return make_indicator_result(
        idx=idx,
        close="29000",
        ema_short="28000",
        ema_medium="29500",
        ema_long="30000",
        rsi="45",
        volume_ratio="1.2",
        crossover=CrossSignal.BEARISH,
        warmup_complete=True,
    )
