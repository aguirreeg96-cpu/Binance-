"""Typed reason codes for strategy decisions.

Each code is a stable string identifier; never change existing values
because they may be stored in serialised form.
"""

from enum import StrEnum


class ReasonCode(StrEnum):
    # Pre-condition failures
    WARMUP_INCOMPLETE = "WARMUP_INCOMPLETE"
    MISSING_INDICATOR = "MISSING_INDICATOR"

    # Crossover signals
    BULLISH_CROSSOVER = "BULLISH_CROSSOVER"
    BEARISH_CROSSOVER = "BEARISH_CROSSOVER"
    NO_NEW_CROSSOVER = "NO_NEW_CROSSOVER"

    # Price vs EMA long
    PRICE_ABOVE_LONG_EMA = "PRICE_ABOVE_LONG_EMA"
    PRICE_BELOW_LONG_EMA = "PRICE_BELOW_LONG_EMA"

    # RSI checks
    RSI_IN_BUY_RANGE = "RSI_IN_BUY_RANGE"
    RSI_OUTSIDE_BUY_RANGE = "RSI_OUTSIDE_BUY_RANGE"
    RSI_OVERBOUGHT = "RSI_OVERBOUGHT"

    # Volume checks
    VOLUME_CONFIRMED = "VOLUME_CONFIRMED"
    VOLUME_INSUFFICIENT = "VOLUME_INSUFFICIENT"

    # Position state
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    NO_POSITION_TO_CLOSE = "NO_POSITION_TO_CLOSE"

    # Composite outcomes
    BUY_CONDITIONS_MET = "BUY_CONDITIONS_MET"
    SELL_CONDITIONS_MET = "SELL_CONDITIONS_MET"


# Human-readable descriptions for logging / UI
REASON_DESCRIPTIONS: dict[ReasonCode, str] = {
    ReasonCode.WARMUP_INCOMPLETE: "Not enough historical data for all indicators.",
    ReasonCode.MISSING_INDICATOR: "One or more required indicators returned None.",
    ReasonCode.BULLISH_CROSSOVER: "EMA-short crossed above EMA-medium this candle.",
    ReasonCode.BEARISH_CROSSOVER: "EMA-short crossed below EMA-medium this candle.",
    ReasonCode.NO_NEW_CROSSOVER: "No new EMA crossover detected on this candle.",
    ReasonCode.PRICE_ABOVE_LONG_EMA: "Close price is above the long-term EMA.",
    ReasonCode.PRICE_BELOW_LONG_EMA: "Close price is below the long-term EMA.",
    ReasonCode.RSI_IN_BUY_RANGE: "RSI is within the configured buy range.",
    ReasonCode.RSI_OUTSIDE_BUY_RANGE: "RSI is outside the configured buy range.",
    ReasonCode.RSI_OVERBOUGHT: "RSI has reached or exceeded the overbought threshold.",
    ReasonCode.VOLUME_CONFIRMED: "Volume ratio meets or exceeds the minimum threshold.",
    ReasonCode.VOLUME_INSUFFICIENT: "Volume ratio is below the minimum threshold.",
    ReasonCode.POSITION_ALREADY_OPEN: "A long position is already open; BUY skipped.",
    ReasonCode.NO_POSITION_TO_CLOSE: "No open position to close; SELL skipped.",
    ReasonCode.BUY_CONDITIONS_MET: "All configured BUY conditions were satisfied.",
    ReasonCode.SELL_CONDITIONS_MET: "At least one configured SELL condition was satisfied.",
}
