"""IndicatorCalculator — orchestrates all technical indicators.

Contract:
  - Receives a list of closed, sorted Candle ORM objects.
  - Returns one IndicatorResult per candle, in the same order.
  - Does NOT query Binance, open HTTP connections, create DB sessions,
    write to any table, or generate trading signals.
  - All computation is O(n) over the candle series.
"""

from datetime import UTC, datetime

from app.indicators.atr import compute_atr
from app.indicators.crossovers import compute_crossovers
from app.indicators.ema import compute_ema
from app.indicators.rsi import compute_rsi
from app.indicators.schemas import IndicatorConfig, IndicatorResult
from app.indicators.sma import compute_sma
from app.indicators.validation import validate_candle_series
from app.indicators.volume import compute_volume_indicators
from app.models.candle import Candle


class IndicatorCalculator:
    """Calculate all configured indicators for a validated candle series."""

    def __init__(self, config: IndicatorConfig | None = None) -> None:
        self.config = config if config is not None else IndicatorConfig()

    def calculate(self, candles: list[Candle]) -> list[IndicatorResult]:
        """Return one IndicatorResult per input candle.

        Validates the series first.  Results for candles in the warm-up phase
        have None for all indicators and warmup_complete=False.
        The output list is always the same length as the input list.
        """
        validate_candle_series(candles)
        cfg = self.config

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        sma_short = compute_sma(closes, cfg.sma_short_period)
        sma_long = compute_sma(closes, cfg.sma_long_period)
        ema_short = compute_ema(closes, cfg.ema_short_period)
        ema_medium = compute_ema(closes, cfg.ema_medium_period)
        ema_long = compute_ema(closes, cfg.ema_long_period)
        rsi = compute_rsi(closes, cfg.rsi_period)
        atr = compute_atr(highs, lows, closes, cfg.atr_period)
        volume_sma, volume_ratio = compute_volume_indicators(volumes, cfg.volume_period)
        crosses = compute_crossovers(ema_short, ema_medium)

        now = datetime.now(UTC)
        results: list[IndicatorResult] = []

        for i, candle in enumerate(candles):
            warmup_complete = (
                sma_short[i] is not None
                and sma_long[i] is not None
                and ema_short[i] is not None
                and ema_medium[i] is not None
                and ema_long[i] is not None
                and rsi[i] is not None
                and atr[i] is not None
                and volume_sma[i] is not None
            )
            results.append(
                IndicatorResult(
                    symbol=candle.symbol,
                    interval=candle.interval,
                    open_time=candle.open_time,
                    close_time=candle.close_time,
                    close=candle.close,
                    sma_short=sma_short[i],
                    sma_long=sma_long[i],
                    ema_short=ema_short[i],
                    ema_medium=ema_medium[i],
                    ema_long=ema_long[i],
                    rsi=rsi[i],
                    atr=atr[i],
                    volume_sma=volume_sma[i],
                    volume_ratio=volume_ratio[i],
                    ema_short_medium_cross=crosses[i],
                    warmup_complete=warmup_complete,
                    source_candle_id=candle.id,
                    calculated_at=now,
                )
            )

        return results
