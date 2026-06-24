"""Entry filter variants for Stage 5.2B.

Each filter adds conditions on top of the V1 baseline entry logic.
Filters are stateless, deterministic, and introduce no look-ahead.
PAPER/TEST only. No filter is declared optimal or expected to improve performance.
Past results do NOT predict future performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from app.indicators.schemas import CrossSignal

if TYPE_CHECKING:
    from app.indicators.schemas import IndicatorResult

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


class EntryFilterType(StrEnum):
    V1_BASELINE = "ENTRY_V1_BASELINE"
    V2_TREND_SLOPE = "ENTRY_V2_TREND_SLOPE"
    V3_ALIGNED_TREND = "ENTRY_V3_ALIGNED_TREND"
    V4_CROSSOVER_STRENGTH = "ENTRY_V4_CROSSOVER_STRENGTH"
    V5_HIGHER_TIMEFRAME = "ENTRY_V5_HIGHER_TIMEFRAME"


@dataclass(frozen=True)
class HtfBarData:
    """Pre-computed higher-timeframe bar indicators (close + EMA50 + EMA200)."""

    close_htf: Decimal
    ema50_htf: Decimal | None
    ema200_htf: Decimal | None


@dataclass(frozen=True)
class EntryFilterConfig:
    """Configuration for one entry variant filter.

    PAPER/TEST only — not a trading recommendation.
    """

    filter_type: EntryFilterType = EntryFilterType.V1_BASELINE
    # V2/V3: EMA slope lookback windows (number of candles in base timeframe)
    ema_medium_slope_candles: int = 8
    ema_long_slope_candles: int = 16
    # V3: bullish crossover must have occurred within this many candles (inclusive)
    crossover_lookback_candles: int = 1
    # V4: minimum (EMA_short − EMA_medium) / EMA_medium × 100
    ema_separation_min_pct: Decimal = Decimal("0.05")
    # V5: base-timeframe candles that form one HTF bar (e.g. 4 for 15m→1h)
    candles_per_htf_bar: int = 4


def build_htf_bars(
    closes: list[Decimal],
    candles_per_htf_bar: int = 4,
    ema_medium_period: int = 50,
    ema_long_period: int = 200,
) -> list[HtfBarData]:
    """Build HTF bar indicators from a base-timeframe close series.

    Groups every `candles_per_htf_bar` closes into one HTF bar (last close
    of each complete group is used as the HTF bar's close).  Only complete
    groups are included.  Computes EMA50 and EMA200 on the HTF close series.

    Returns one HtfBarData per complete HTF bar in chronological order.
    PAPER/TEST only — no look-ahead (bar j uses candles j·N … j·N+N−1).
    """
    from app.indicators.ema import compute_ema

    n = len(closes)
    complete_bars = n // candles_per_htf_bar
    if complete_bars == 0:
        return []

    htf_closes: list[Decimal] = [
        closes[(j + 1) * candles_per_htf_bar - 1] for j in range(complete_bars)
    ]

    ema50 = compute_ema(htf_closes, ema_medium_period)
    ema200 = compute_ema(htf_closes, ema_long_period)

    return [
        HtfBarData(
            close_htf=htf_closes[j],
            ema50_htf=ema50[j],
            ema200_htf=ema200[j],
        )
        for j in range(complete_bars)
    ]


def passes_entry_filter(
    idx: int,
    all_results: list[IndicatorResult],
    config: EntryFilterConfig,
    htf_bars: list[HtfBarData] | None = None,
) -> bool:
    """Return True if the entry signal at all_results[idx] passes the additional filter.

    idx is the absolute index in all_results (includes warmup prefix).
    PAPER/TEST only. Does not predict future performance.
    """
    ft = config.filter_type
    if ft == EntryFilterType.V1_BASELINE:
        return True
    if ft == EntryFilterType.V2_TREND_SLOPE:
        return _check_trend_slope(idx, all_results, config)
    if ft == EntryFilterType.V3_ALIGNED_TREND:
        return _check_aligned_trend(idx, all_results, config)
    if ft == EntryFilterType.V4_CROSSOVER_STRENGTH:
        return _check_crossover_strength(idx, all_results, config)
    if ft == EntryFilterType.V5_HIGHER_TIMEFRAME:
        return _check_htf_confirmation(idx, config, htf_bars)
    return True  # unknown type → pass through


def _check_trend_slope(
    idx: int,
    all_results: list[IndicatorResult],
    config: EntryFilterConfig,
) -> bool:
    """V2: EMA50 ascending over N candles AND EMA200 ascending over M candles."""
    r = all_results[idx]
    if r.ema_medium is None or r.ema_long is None:
        return False
    med_lb = config.ema_medium_slope_candles
    long_lb = config.ema_long_slope_candles
    if idx < med_lb or idx < long_lb:
        return False
    prev_med = all_results[idx - med_lb].ema_medium
    prev_long = all_results[idx - long_lb].ema_long
    if prev_med is None or prev_long is None:
        return False
    return r.ema_medium > prev_med and r.ema_long > prev_long


def _check_aligned_trend(
    idx: int,
    all_results: list[IndicatorResult],
    config: EntryFilterConfig,
) -> bool:
    """V3: EMA20 > EMA50 > EMA200 (aligned) AND both ascending AND crossover within N candles."""
    r = all_results[idx]
    if r.ema_short is None or r.ema_medium is None or r.ema_long is None:
        return False
    if not (r.ema_short > r.ema_medium > r.ema_long):
        return False

    med_lb = config.ema_medium_slope_candles
    long_lb = config.ema_long_slope_candles
    if idx < long_lb:
        return False

    if idx >= med_lb:
        prev_med = all_results[idx - med_lb].ema_medium
        if prev_med is None or r.ema_medium <= prev_med:
            return False

    if idx >= long_lb:
        prev_long = all_results[idx - long_lb].ema_long
        if prev_long is None or r.ema_long <= prev_long:
            return False

    # Bullish crossover must have occurred within the last N candles (inclusive)
    n = config.crossover_lookback_candles
    start = max(0, idx - n + 1)
    return any(
        all_results[j].ema_short_medium_cross == CrossSignal.BULLISH for j in range(start, idx + 1)
    )


def _check_crossover_strength(
    idx: int,
    all_results: list[IndicatorResult],
    config: EntryFilterConfig,
) -> bool:
    """V4: (EMA20 − EMA50) / EMA50 × 100 >= threshold."""
    r = all_results[idx]
    if r.ema_short is None or r.ema_medium is None or r.ema_medium <= _ZERO:
        return False
    sep_pct = (r.ema_short - r.ema_medium) / r.ema_medium * _HUNDRED
    return sep_pct >= config.ema_separation_min_pct


def _check_htf_confirmation(
    idx: int,
    config: EntryFilterConfig,
    htf_bars: list[HtfBarData] | None,
) -> bool:
    """V5: Last COMPLETE 1h bar has close>EMA200_1h, EMA50_1h>EMA200_1h, EMA200_1h ascending."""
    if not htf_bars:
        return False
    current_group = idx // config.candles_per_htf_bar
    last_complete = current_group - 1  # current group is in progress → not usable
    if last_complete < 1:
        return False
    if last_complete >= len(htf_bars):
        return False
    bar = htf_bars[last_complete]
    if bar.ema50_htf is None or bar.ema200_htf is None:
        return False
    if not (bar.close_htf > bar.ema200_htf and bar.ema50_htf > bar.ema200_htf):
        return False
    # EMA200 must be ascending vs previous complete bar
    prev_bar = htf_bars[last_complete - 1]
    if prev_bar.ema200_htf is None or bar.ema200_htf <= prev_bar.ema200_htf:
        return False
    return True
