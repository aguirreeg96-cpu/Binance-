"""Pure rule evaluation functions.

Each function is stateless and deterministic.
No DB access, no HTTP, no side effects.
"""

from app.indicators.schemas import CrossSignal, IndicatorResult
from app.strategy.config import StrategyEngineConfig
from app.strategy.reasons import ReasonCode


def has_missing_critical_indicators(result: IndicatorResult) -> bool:
    """Return True if any indicator required by the strategy rules is None."""
    return result.ema_long is None or result.rsi is None or result.volume_ratio is None


def evaluate_buy_conditions(
    result: IndicatorResult,
    cfg: StrategyEngineConfig,
) -> tuple[bool, list[ReasonCode], list[ReasonCode]]:
    """Evaluate all BUY conditions.

    Returns (all_conditions_met, met_reasons, failed_reasons).
    ALL conditions must pass for should_buy=True.
    """
    met: list[ReasonCode] = []
    failed: list[ReasonCode] = []

    # Condition 1: bullish EMA crossover (if required)
    if cfg.require_bullish_crossover:
        if result.ema_short_medium_cross == CrossSignal.BULLISH:
            met.append(ReasonCode.BULLISH_CROSSOVER)
        else:
            failed.append(ReasonCode.NO_NEW_CROSSOVER)

    # Condition 2: price above long-term EMA (if required)
    if cfg.require_price_above_long_ema:
        if result.ema_long is not None and result.close > result.ema_long:
            met.append(ReasonCode.PRICE_ABOVE_LONG_EMA)
        else:
            failed.append(ReasonCode.PRICE_BELOW_LONG_EMA)

    # Condition 3: RSI in buy range
    if result.rsi is not None:
        if cfg.buy_rsi_min <= result.rsi <= cfg.buy_rsi_max:
            met.append(ReasonCode.RSI_IN_BUY_RANGE)
        else:
            failed.append(ReasonCode.RSI_OUTSIDE_BUY_RANGE)
    else:
        failed.append(ReasonCode.MISSING_INDICATOR)

    # Condition 4: volume ratio sufficient
    if result.volume_ratio is not None:
        if result.volume_ratio >= cfg.minimum_volume_ratio:
            met.append(ReasonCode.VOLUME_CONFIRMED)
        else:
            failed.append(ReasonCode.VOLUME_INSUFFICIENT)
    else:
        failed.append(ReasonCode.VOLUME_INSUFFICIENT)

    return len(failed) == 0, met, failed


def evaluate_sell_conditions(
    result: IndicatorResult,
    cfg: StrategyEngineConfig,
) -> tuple[bool, list[ReasonCode], list[ReasonCode]]:
    """Evaluate SELL exit conditions.

    Returns (any_condition_met, met_reasons, failed_reasons).

    require_bearish_crossover_for_sell=True: bearish crossover alone is
      the required trigger; price/RSI reasons are collected but crossover
      must be present for should_sell=True.
    require_bearish_crossover_for_sell=False: any single exit condition
      (crossover, price below EMA, RSI overbought) is sufficient.

    SELL on Spot means closing a long position, never opening a short.
    """
    met: list[ReasonCode] = []
    failed: list[ReasonCode] = []

    has_bearish_cross = result.ema_short_medium_cross == CrossSignal.BEARISH
    has_price_exit = (
        cfg.require_price_above_long_ema
        and result.ema_long is not None
        and result.close < result.ema_long
    )
    has_rsi_exit = result.rsi is not None and result.rsi >= cfg.sell_rsi_overbought

    if has_bearish_cross:
        met.append(ReasonCode.BEARISH_CROSSOVER)
    elif cfg.require_bearish_crossover_for_sell:
        failed.append(ReasonCode.NO_NEW_CROSSOVER)

    if has_price_exit:
        met.append(ReasonCode.PRICE_BELOW_LONG_EMA)

    if has_rsi_exit:
        met.append(ReasonCode.RSI_OVERBOUGHT)

    if cfg.require_bearish_crossover_for_sell:
        # Bearish crossover is required; other conditions add context
        should_sell = has_bearish_cross
    else:
        # Any exit condition is enough
        should_sell = has_bearish_cross or has_price_exit or has_rsi_exit

    return should_sell, met, failed
