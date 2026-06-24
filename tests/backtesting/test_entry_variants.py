"""Tests for Stage 5.2B — Entry variant comparison.

Covers:
  - EntryFilterType.V1_BASELINE: always passes
  - EntryFilterType.V2_TREND_SLOPE: EMA50 slope (8c) + EMA200 slope (16c)
  - EntryFilterType.V3_ALIGNED_TREND: alignment + ascending + crossover lookback 1/2/4
  - EntryFilterType.V4_CROSSOVER_STRENGTH: EMA separation 0/0.05/0.10 %
  - EntryFilterType.V5_HIGHER_TIMEFRAME: 1h confirmation from 15m candles
  - build_htf_bars: 15m→1h aggregation, incomplete group excluded
  - No look-ahead: current HTF group never used
  - FilteredStrategyEngine: detected_signals and blocked_count tracking
  - run_entry_comparison: 10 combos, correct structure
  - run_entry_multi_period: capital compounding 2023→2024
  - FilterAnalysis: entries_eliminated, entries_added, eliminated_pnl
  - SignalCounts: signals_detected = buys_executed + blocked_by_filter
  - Determinism: identical inputs → identical outputs
  - Zero-trade scenarios
  - All 5 exporters: files created, correct columns
  - Decimal serialization in JSON exporter
"""

import json
from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.entry_comparison import (
    ENTRY_VARIANT_NAMES,
    EXIT_CONFIG_NAMES,
    EntryComparisonReport,
    EntryMultiPeriodReport,
    run_entry_comparison,
    run_entry_multi_period,
)
from app.backtesting.entry_exporters import (
    export_entry_comparison_csv,
    export_entry_comparison_json,
    export_filter_analysis_csv,
    export_signal_counts_csv,
    export_yearly_entry_comparison_csv,
)
from app.indicators.schemas import IndicatorConfig, IndicatorResult
from app.strategy.entry_filter import (
    EntryFilterConfig,
    EntryFilterType,
    HtfBarData,
    build_htf_bars,
    passes_entry_filter,
)
from tests.backtesting.conftest import make_candle, make_candles

_D = Decimal
_BASE_TIME = 1_000_000
_INTERVAL_MS = 900_000  # 15m


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ind_config(**kwargs) -> IndicatorConfig:
    defaults = {
        "sma_short_period": 2,
        "sma_long_period": 3,
        "ema_short_period": 2,
        "ema_medium_period": 3,
        "ema_long_period": 4,
        "rsi_period": 2,
        "atr_period": 2,
        "volume_period": 2,
    }
    defaults.update(kwargs)
    return IndicatorConfig(**defaults)


def _cfg(
    start_offset: int = 0,
    n_candles: int = 60,
    capital: str = "10000",
    **kwargs,
) -> BacktestConfig:
    defaults = {
        "symbol": "BTCUSDT",
        "interval": "15m",
        "start_ms": _BASE_TIME + start_offset * _INTERVAL_MS,
        "end_ms": _BASE_TIME + (start_offset + n_candles) * _INTERVAL_MS,
        "initial_capital": _D(capital),
        "fee_percentage": _D("0"),
        "slippage_percentage": _D("0"),
        "force_close_at_end": True,
    }
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _candles_15m(n: int, price: str = "100") -> list:
    return make_candles(
        n,
        base_open_time=_BASE_TIME,
        interval_ms=_INTERVAL_MS,
        price=price,
        symbol="BTCUSDT",
        interval="15m",
    )


def _fake_result(
    all_results: list[IndicatorResult],
    idx: int,
    ema_short: Decimal | None = None,
    ema_medium: Decimal | None = None,
    ema_long: Decimal | None = None,
) -> IndicatorResult:
    """Return a copy of all_results[idx] with overridden EMA values."""
    from dataclasses import replace

    r = all_results[idx]
    return replace(
        r,
        ema_short=ema_short if ema_short is not None else r.ema_short,
        ema_medium=ema_medium if ema_medium is not None else r.ema_medium,
        ema_long=ema_long if ema_long is not None else r.ema_long,
    )


# ---------------------------------------------------------------------------
# build_htf_bars
# ---------------------------------------------------------------------------


class TestBuildHtfBars:
    def test_empty_input(self):
        assert build_htf_bars([], candles_per_htf_bar=4) == []

    def test_fewer_than_one_bar(self):
        closes = [_D("100")] * 3
        result = build_htf_bars(closes, candles_per_htf_bar=4)
        assert result == []

    def test_exactly_one_complete_bar(self):
        closes = [_D("100"), _D("101"), _D("102"), _D("103")]
        bars = build_htf_bars(closes, candles_per_htf_bar=4, ema_medium_period=1, ema_long_period=1)
        assert len(bars) == 1
        assert bars[0].close_htf == _D("103")  # last close of the group

    def test_incomplete_trailing_group_excluded(self):
        closes = [_D(str(i)) for i in range(9)]  # 9 closes → 2 complete groups of 4 + 1 leftover
        bars = build_htf_bars(closes, candles_per_htf_bar=4, ema_medium_period=1, ema_long_period=1)
        assert len(bars) == 2
        assert bars[0].close_htf == _D("3")  # index 3 (end of group 0)
        assert bars[1].close_htf == _D("7")  # index 7 (end of group 1)

    def test_ema_warmup_respected(self):
        # 12 closes → 3 bars; EMA200 needs 200 bars → all ema200_htf should be None
        closes = [_D("100")] * 12
        bars = build_htf_bars(
            closes, candles_per_htf_bar=4, ema_medium_period=50, ema_long_period=200
        )
        assert len(bars) == 3
        for b in bars:
            assert b.ema200_htf is None

    def test_ema_values_present_after_warmup(self):
        # With period=1, every bar gets an EMA value
        closes = [_D(str(100 + i)) for i in range(8)]
        bars = build_htf_bars(closes, candles_per_htf_bar=4, ema_medium_period=1, ema_long_period=1)
        assert len(bars) == 2
        assert all(b.ema50_htf is not None for b in bars)
        assert all(b.ema200_htf is not None for b in bars)


# ---------------------------------------------------------------------------
# passes_entry_filter — V1_BASELINE
# ---------------------------------------------------------------------------


class TestFilterV1Baseline:
    def test_always_passes(self):
        candles = _candles_15m(30)
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        cfg = EntryFilterConfig(filter_type=EntryFilterType.V1_BASELINE)
        for idx in range(len(all_results)):
            assert passes_entry_filter(idx, all_results, cfg) is True


# ---------------------------------------------------------------------------
# passes_entry_filter — V2_TREND_SLOPE
# ---------------------------------------------------------------------------


class TestFilterV2TrendSlope:
    def _results(self, n: int, price: str = "100") -> list[IndicatorResult]:
        from app.indicators.calculator import IndicatorCalculator

        return IndicatorCalculator(_ind_config()).calculate(_candles_15m(n, price))

    def test_passes_when_both_emas_rising(self):
        # Build a rising-price candle series so EMA50 and EMA200 slope up
        candles = []
        for i in range(50):
            candles.append(
                make_candle(
                    open_time=_BASE_TIME + i * _INTERVAL_MS,
                    open_p=str(100 + i),
                    high_p=str(101 + i),
                    low_p=str(99 + i),
                    close_p=str(100 + i),
                    interval="15m",
                )
            )
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V2_TREND_SLOPE,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=3,
        )
        # At sufficiently late indices (warmup + lookback), should pass
        last_idx = len(all_results) - 1
        result = passes_entry_filter(last_idx, all_results, cfg)
        assert result is True

    def test_fails_when_insufficient_history(self):
        all_results = self._results(5)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V2_TREND_SLOPE,
            ema_medium_slope_candles=8,
            ema_long_slope_candles=16,
        )
        # idx=0 has no lookback history
        assert passes_entry_filter(0, all_results, cfg) is False

    def test_fails_when_ema_none(self):
        all_results = self._results(20)
        cfg = EntryFilterConfig(filter_type=EntryFilterType.V2_TREND_SLOPE)
        # First few results have None EMAs due to warmup
        none_indices = [i for i, r in enumerate(all_results) if r.ema_medium is None]
        if none_indices:
            assert passes_entry_filter(none_indices[0], all_results, cfg) is False

    def test_fails_when_emas_flat(self):
        # Flat price → EMA slope is zero (not ascending) → filter fails
        all_results = self._results(50, price="100")
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V2_TREND_SLOPE,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=3,
        )
        # With constant price, EMAs are constant → slope = 0 → should fail
        last_idx = len(all_results) - 1
        r = all_results[last_idx]
        if r.ema_medium is not None and r.ema_long is not None:
            result = passes_entry_filter(last_idx, all_results, cfg)
            assert result is False


# ---------------------------------------------------------------------------
# passes_entry_filter — V3_ALIGNED_TREND
# ---------------------------------------------------------------------------


class TestFilterV3AlignedTrend:
    def _make_result_with_emas(
        self,
        idx: int,
        all_results: list[IndicatorResult],
        ema_short: Decimal,
        ema_medium: Decimal,
        ema_long: Decimal,
        cross=None,
    ) -> list[IndicatorResult]:
        """Return a copy of all_results with the ema values at idx overridden."""

        modified = list(all_results)
        r = all_results[idx]
        from dataclasses import replace

        cross_val = cross if cross is not None else r.ema_short_medium_cross
        modified[idx] = replace(
            r,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            ema_short_medium_cross=cross_val,
        )
        return modified

    def test_fails_when_not_aligned(self):
        from app.indicators.calculator import IndicatorCalculator

        candles = _candles_15m(50)
        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=3,
            crossover_lookback_candles=1,
        )
        # Constant price → EMAs are equal → not strictly aligned
        last_idx = len(all_results) - 1
        assert passes_entry_filter(last_idx, all_results, cfg) is False

    def test_passes_with_alignment_slope_and_crossover(self):
        from app.indicators.calculator import IndicatorCalculator
        from app.indicators.schemas import CrossSignal

        candles = _candles_15m(50)
        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=3,
            crossover_lookback_candles=1,
        )
        last_idx = len(all_results) - 1

        # Manually override EMAs at last_idx and lookback positions to satisfy filter
        # Set ascending EMAs: ema_short > ema_medium > ema_long, both slope positive
        med_lb = cfg.ema_medium_slope_candles
        long_lb = cfg.ema_long_slope_candles

        from dataclasses import replace

        modified = list(all_results)
        # prev positions (ascending from earlier lower values)
        if last_idx >= long_lb:
            modified[last_idx - long_lb] = replace(
                modified[last_idx - long_lb],
                ema_medium=_D("90"),
                ema_long=_D("80"),
            )
        if last_idx >= med_lb:
            modified[last_idx - med_lb] = replace(
                modified[last_idx - med_lb],
                ema_medium=_D("92"),
                ema_long=_D("81"),
            )
        modified[last_idx] = replace(
            modified[last_idx],
            ema_short=_D("105"),
            ema_medium=_D("100"),
            ema_long=_D("95"),
            ema_short_medium_cross=CrossSignal.BULLISH,
        )
        assert passes_entry_filter(last_idx, modified, cfg) is True

    def test_crossover_lookback_2_allows_prior_candle(self):
        from app.indicators.calculator import IndicatorCalculator
        from app.indicators.schemas import CrossSignal

        candles = _candles_15m(50)
        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        last_idx = len(all_results) - 1
        med_lb = 2
        long_lb = 3

        from dataclasses import replace

        modified = list(all_results)
        if last_idx >= long_lb:
            modified[last_idx - long_lb] = replace(
                modified[last_idx - long_lb], ema_medium=_D("88"), ema_long=_D("78")
            )
        if last_idx >= med_lb:
            modified[last_idx - med_lb] = replace(
                modified[last_idx - med_lb], ema_medium=_D("91"), ema_long=_D("80")
            )
        # Crossover at (last_idx - 1), NOT at last_idx
        modified[last_idx - 1] = replace(
            modified[last_idx - 1],
            ema_short_medium_cross=CrossSignal.BULLISH,
        )
        modified[last_idx] = replace(
            modified[last_idx],
            ema_short=_D("105"),
            ema_medium=_D("100"),
            ema_long=_D("95"),
            ema_short_medium_cross=CrossSignal.NONE,  # no crossover at current candle
        )
        cfg_lb1 = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=med_lb,
            ema_long_slope_candles=long_lb,
            crossover_lookback_candles=1,
        )
        cfg_lb2 = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=med_lb,
            ema_long_slope_candles=long_lb,
            crossover_lookback_candles=2,
        )
        # With lookback=1: crossover not on current candle → fails
        assert passes_entry_filter(last_idx, modified, cfg_lb1) is False
        # With lookback=2: crossover 1 candle ago → passes
        assert passes_entry_filter(last_idx, modified, cfg_lb2) is True

    def test_crossover_lookback_4(self):
        from app.indicators.calculator import IndicatorCalculator
        from app.indicators.schemas import CrossSignal

        candles = _candles_15m(60)
        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        last_idx = len(all_results) - 1
        med_lb = 2
        long_lb = 3

        from dataclasses import replace

        modified = list(all_results)
        if last_idx >= long_lb:
            modified[last_idx - long_lb] = replace(
                modified[last_idx - long_lb], ema_medium=_D("85"), ema_long=_D("75")
            )
        if last_idx >= med_lb:
            modified[last_idx - med_lb] = replace(
                modified[last_idx - med_lb], ema_medium=_D("90"), ema_long=_D("79")
            )
        # Crossover 3 candles ago
        modified[last_idx - 3] = replace(
            modified[last_idx - 3],
            ema_short_medium_cross=CrossSignal.BULLISH,
        )
        modified[last_idx] = replace(
            modified[last_idx],
            ema_short=_D("105"),
            ema_medium=_D("100"),
            ema_long=_D("95"),
            ema_short_medium_cross=CrossSignal.NONE,
        )
        cfg_lb4 = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=med_lb,
            ema_long_slope_candles=long_lb,
            crossover_lookback_candles=4,
        )
        cfg_lb3 = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=med_lb,
            ema_long_slope_candles=long_lb,
            crossover_lookback_candles=3,
        )
        assert passes_entry_filter(last_idx, modified, cfg_lb4) is True
        assert passes_entry_filter(last_idx, modified, cfg_lb3) is False


# ---------------------------------------------------------------------------
# passes_entry_filter — V4_CROSSOVER_STRENGTH
# ---------------------------------------------------------------------------


class TestFilterV4CrossoverStrength:
    def _results(self) -> list[IndicatorResult]:
        from app.indicators.calculator import IndicatorCalculator

        return IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))

    def test_threshold_zero_always_passes_when_ema_short_geq_medium(self):
        from dataclasses import replace

        all_results = self._results()
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0"),
        )
        last_idx = len(all_results) - 1
        modified = list(all_results)
        modified[last_idx] = replace(
            modified[last_idx],
            ema_short=_D("100"),
            ema_medium=_D("100"),  # equal → sep=0 >= 0 → passes
        )
        assert passes_entry_filter(last_idx, modified, cfg) is True

    def test_threshold_0_05_passes_when_sep_sufficient(self):
        from dataclasses import replace

        all_results = self._results()
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0.05"),
        )
        last_idx = len(all_results) - 1
        modified = list(all_results)
        # (101 - 100) / 100 * 100 = 1% >= 0.05% → passes
        modified[last_idx] = replace(modified[last_idx], ema_short=_D("101"), ema_medium=_D("100"))
        assert passes_entry_filter(last_idx, modified, cfg) is True

    def test_threshold_0_10_fails_when_sep_insufficient(self):
        from dataclasses import replace

        all_results = self._results()
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0.10"),
        )
        last_idx = len(all_results) - 1
        modified = list(all_results)
        # (100.05 - 100) / 100 * 100 = 0.05% < 0.10% → fails
        modified[last_idx] = replace(
            modified[last_idx],
            ema_short=_D("100.05"),
            ema_medium=_D("100"),
        )
        assert passes_entry_filter(last_idx, modified, cfg) is False

    def test_fails_when_ema_none(self):
        all_results = self._results()
        cfg = EntryFilterConfig(filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH)
        none_idx = next((i for i, r in enumerate(all_results) if r.ema_medium is None), None)
        if none_idx is not None:
            assert passes_entry_filter(none_idx, all_results, cfg) is False


# ---------------------------------------------------------------------------
# passes_entry_filter — V5_HIGHER_TIMEFRAME
# ---------------------------------------------------------------------------


class TestFilterV5HigherTimeframe:
    def _make_htf_bars(self, n_bars: int) -> list[HtfBarData]:
        return [
            HtfBarData(
                close_htf=_D("210"),
                ema50_htf=_D("200"),
                ema200_htf=_D(str(148 + i)),  # ascending EMA200
            )
            for i in range(n_bars)
        ]

    def test_no_htf_bars_fails(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(20))
        cfg = EntryFilterConfig(filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME)
        assert passes_entry_filter(5, all_results, cfg, htf_bars=None) is False

    def test_empty_htf_fails(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(20))
        cfg = EntryFilterConfig(filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME)
        assert passes_entry_filter(5, all_results, cfg, htf_bars=[]) is False

    def test_no_look_ahead_current_group_not_used(self):
        # At 15m idx=4 (start of group 1), the current group is group 1
        # Last complete group is 0; but we also need group_idx-1 ≥ 1 (prev bar for slope)
        # → last_complete=0, last_complete-1=-1 < 0 → fails → no look-ahead
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(20))
        htf_bars = self._make_htf_bars(5)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME,
            candles_per_htf_bar=4,
        )
        # idx=4: current_group=1, last_complete=0, last_complete-1=-1 < 0 → fail
        assert passes_entry_filter(4, all_results, cfg, htf_bars=htf_bars) is False

    def test_passes_when_htf_conditions_met(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(40))
        # idx=8: current_group=2, last_complete=1, prev=0 → both exist
        htf_bars = self._make_htf_bars(5)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME,
            candles_per_htf_bar=4,
        )
        # htf_bars[1]: close=210 > ema200=149, ema50=200 > ema200=149, ascending
        assert passes_entry_filter(8, all_results, cfg, htf_bars=htf_bars) is True

    def test_fails_when_ema200_not_ascending(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(40))
        # Descending EMA200
        htf_bars = [
            HtfBarData(close_htf=_D("210"), ema50_htf=_D("200"), ema200_htf=_D("152")),
            HtfBarData(close_htf=_D("210"), ema50_htf=_D("200"), ema200_htf=_D("150")),  # lower
            HtfBarData(close_htf=_D("210"), ema50_htf=_D("200"), ema200_htf=_D("148")),
        ]
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME,
            candles_per_htf_bar=4,
        )
        # idx=8, last_complete=1: ema200[1]=150 <= ema200[0]=152 → fails
        assert passes_entry_filter(8, all_results, cfg, htf_bars=htf_bars) is False

    def test_fails_when_close_below_ema200(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(40))
        # close=140 < ema200=150 → fails
        htf_bars = [
            HtfBarData(close_htf=_D("140"), ema50_htf=_D("200"), ema200_htf=_D("148")),
            HtfBarData(close_htf=_D("140"), ema50_htf=_D("200"), ema200_htf=_D("150")),
        ]
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME,
            candles_per_htf_bar=4,
        )
        assert passes_entry_filter(8, all_results, cfg, htf_bars=htf_bars) is False


# ---------------------------------------------------------------------------
# build_htf_bars integration with 15m candles
# ---------------------------------------------------------------------------


class TestBuildHtfBarsIntegration:
    def test_15m_to_1h_grouping(self):
        # 16 15m closes → 4 complete 1h bars
        closes = [_D(str(i * 10 + 10)) for i in range(16)]
        bars = build_htf_bars(closes, candles_per_htf_bar=4, ema_medium_period=2, ema_long_period=3)
        assert len(bars) == 4
        # Bar 0: closes 0-3, bar close = closes[3] = 40
        assert bars[0].close_htf == _D("40")
        # Bar 1: closes 4-7, bar close = closes[7] = 80
        assert bars[1].close_htf == _D("80")
        # Bar 3: closes 12-15, bar close = closes[15] = 160
        assert bars[3].close_htf == _D("160")

    def test_ema_period_on_htf_bars(self):
        # period=2, 2 bars → first EMA at index period-1=1 (0-based); bars[0] is in warmup
        closes = [_D("100")] * 8
        bars = build_htf_bars(closes, candles_per_htf_bar=4, ema_medium_period=2, ema_long_period=2)
        assert len(bars) == 2
        assert bars[0].ema50_htf is None  # still in warmup (seed needs period=2 bars)
        assert bars[1].ema50_htf == _D("100")  # constant series → EMA = 100


# ---------------------------------------------------------------------------
# run_entry_comparison — structure tests
# ---------------------------------------------------------------------------


class TestRunEntryComparisonStructure:
    def _run(self, n: int = 60) -> EntryComparisonReport:
        candles = _candles_15m(n)
        warmup = _ind_config().warmup_candles
        return run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=n),
            indicator_config=_ind_config(),
        )

    def test_exactly_10_combinations(self):
        report = self._run()
        assert len(report.combinations) == 10

    def test_5_entry_variants_present(self):
        report = self._run()
        variants = {c.entry_variant for c in report.combinations}
        assert variants == set(ENTRY_VARIANT_NAMES)

    def test_2_exit_configs_present(self):
        report = self._run()
        exits = {c.exit_config_name for c in report.combinations}
        assert exits == set(EXIT_CONFIG_NAMES)

    def test_all_variants_cross_all_exits(self):
        report = self._run()
        pairs = {(c.entry_variant, c.exit_config_name) for c in report.combinations}
        assert len(pairs) == 10

    def test_bah_return_set(self):
        report = self._run()
        assert isinstance(report.bah_return_pct, Decimal)

    def test_period_label_propagated(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        report = run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
            period_label="2023",
        )
        assert report.period_label == "2023"

    def test_signal_counts_consistent(self):
        report = self._run()
        for c in report.combinations:
            sc = c.signal_counts
            # signals_detected = buys_executed + blocked_by_filter
            assert sc.signals_detected == sc.buys_executed + sc.blocked_by_filter

    def test_v1_baseline_zero_blocked(self):
        report = self._run()
        for c in report.combinations:
            if c.entry_variant == "ENTRY_V1_BASELINE":
                assert c.signal_counts.blocked_by_filter == 0

    def test_v1_baseline_zero_filter_analysis_eliminated(self):
        report = self._run()
        for c in report.combinations:
            if c.entry_variant == "ENTRY_V1_BASELINE":
                assert c.filter_analysis.entries_eliminated == 0
                assert c.filter_analysis.entries_added == 0


# ---------------------------------------------------------------------------
# run_entry_comparison — no negative balance
# ---------------------------------------------------------------------------


class TestNoNegativeBalance:
    def test_no_negative_equity(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        report = run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
        )
        for combo in report.combinations:
            assert combo.result.final_equity >= Decimal("0")


# ---------------------------------------------------------------------------
# run_entry_comparison — determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_runs(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg = _cfg(n_candles=60)
        ind = _ind_config()

        r1 = run_entry_comparison(candles, warmup, cfg, ind)
        r2 = run_entry_comparison(candles, warmup, cfg, ind)

        for c1, c2 in zip(r1.combinations, r2.combinations, strict=True):
            assert c1.result.total_return_pct == c2.result.total_return_pct
            assert c1.result.total_trades == c2.result.total_trades
            assert c1.signal_counts.signals_detected == c2.signal_counts.signals_detected


# ---------------------------------------------------------------------------
# run_entry_multi_period — capital compounding
# ---------------------------------------------------------------------------


class TestCapitalCompounding:
    def test_structure(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg_23 = _cfg(n_candles=60, capital="10000")
        cfg_24 = _cfg(start_offset=60, n_candles=60, capital="10000")

        report = run_entry_multi_period(
            candles_2023=candles,
            warmup_2023=warmup,
            config_2023=cfg_23,
            candles_2024=candles,
            warmup_2024=warmup,
            config_2024=cfg_24,
            indicator_config=_ind_config(),
        )
        assert isinstance(report, EntryMultiPeriodReport)
        assert len(report.period_2023.combinations) == 10
        assert len(report.period_2024.combinations) == 10
        assert len(report.yearly_summary) == 10

    def test_2024_initial_capital_equals_2023_final(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg_23 = _cfg(n_candles=60, capital="10000")
        cfg_24 = _cfg(start_offset=60, n_candles=60, capital="10000")

        report = run_entry_multi_period(
            candles_2023=candles,
            warmup_2023=warmup,
            config_2023=cfg_23,
            candles_2024=candles,
            warmup_2024=warmup,
            config_2024=cfg_24,
            indicator_config=_ind_config(),
        )
        # For each combo, 2024 initial capital = 2023 final equity (compounding)
        for s in report.yearly_summary:
            matching_23 = next(
                c
                for c in report.period_2023.combinations
                if c.entry_variant == s.entry_variant and c.exit_config_name == s.exit_config_name
            )
            matching_24 = next(
                c
                for c in report.period_2024.combinations
                if c.entry_variant == s.entry_variant and c.exit_config_name == s.exit_config_name
            )
            # 2024 result's initial capital should be 2023's final_equity
            assert matching_24.result.initial_capital == matching_23.result.final_equity

    def test_combined_return_formula(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg_23 = _cfg(n_candles=60, capital="10000")
        cfg_24 = _cfg(start_offset=60, n_candles=60, capital="10000")

        report = run_entry_multi_period(
            candles_2023=candles,
            warmup_2023=warmup,
            config_2023=cfg_23,
            candles_2024=candles,
            warmup_2024=warmup,
            config_2024=cfg_24,
            indicator_config=_ind_config(),
        )
        _HUNDRED = Decimal("100")
        for s in report.yearly_summary:
            r23 = s.return_pct_2023 / _HUNDRED
            r24 = s.return_pct_2024 / _HUNDRED
            expected = ((1 + r23) * (1 + r24) - 1) * _HUNDRED
            assert abs(s.combined_return_pct - expected) < Decimal("0.0001")

    def test_yearly_summary_all_variants(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg_23 = _cfg(n_candles=60)
        cfg_24 = _cfg(start_offset=60, n_candles=60)

        report = run_entry_multi_period(
            candles_2023=candles,
            warmup_2023=warmup,
            config_2023=cfg_23,
            candles_2024=candles,
            warmup_2024=warmup,
            config_2024=cfg_24,
            indicator_config=_ind_config(),
        )
        pairs = {(s.entry_variant, s.exit_config_name) for s in report.yearly_summary}
        assert len(pairs) == 10


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


class TestEntryExporters:
    def _report(self) -> EntryComparisonReport:
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        return run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
        )

    def _multi_report(self) -> EntryMultiPeriodReport:
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg_23 = _cfg(n_candles=60)
        cfg_24 = _cfg(start_offset=60, n_candles=60)
        return run_entry_multi_period(
            candles_2023=candles,
            warmup_2023=warmup,
            config_2023=cfg_23,
            candles_2024=candles,
            warmup_2024=warmup,
            config_2024=cfg_24,
            indicator_config=_ind_config(),
        )

    def test_export_entry_comparison_csv(self, tmp_path):
        import csv

        report = self._report()
        path = tmp_path / "ec.csv"
        export_entry_comparison_csv(report, path)
        assert path.exists()
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 10
        assert "entry_variant" in rows[0]
        assert "return_pct" in rows[0]

    def test_export_entry_comparison_json(self, tmp_path):
        report = self._report()
        path = tmp_path / "ec.json"
        export_entry_comparison_json(report, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "warning" in data
        assert "combinations" in data
        assert len(data["combinations"]) == 10

    def test_json_decimal_as_string(self, tmp_path):
        report = self._report()
        path = tmp_path / "ec.json"
        export_entry_comparison_json(report, path)
        data = json.loads(path.read_text())
        # return_pct should be a string (Decimal serialized)
        assert isinstance(data["combinations"][0]["return_pct"], str)

    def test_export_yearly_entry_comparison_csv(self, tmp_path):
        import csv

        multi = self._multi_report()
        path = tmp_path / "yr.csv"
        export_yearly_entry_comparison_csv(multi, path)
        assert path.exists()
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 10
        assert "return_pct_2023" in rows[0]
        assert "combined_return_pct" in rows[0]
        assert "capital_2023_end" in rows[0]

    def test_export_filter_analysis_csv(self, tmp_path):
        import csv

        report = self._report()
        path = tmp_path / "fa.csv"
        export_filter_analysis_csv(report, path)
        assert path.exists()
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 10
        assert "entries_eliminated" in rows[0]
        assert "eliminated_pnl" in rows[0]

    def test_export_signal_counts_csv(self, tmp_path):
        import csv

        report = self._report()
        path = tmp_path / "sc.csv"
        export_signal_counts_csv(report, path)
        assert path.exists()
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 10
        assert "signals_detected" in rows[0]
        assert "blocked_by_filter" in rows[0]


# ---------------------------------------------------------------------------
# Filter analysis correctness
# ---------------------------------------------------------------------------


class TestFilterAnalysis:
    def test_v1_self_comparison_all_zeros(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        report = run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
        )
        for c in report.combinations:
            if c.entry_variant == "ENTRY_V1_BASELINE":
                fa = c.filter_analysis
                assert fa.entries_eliminated == 0
                assert fa.entries_added == 0
                assert fa.return_pct_vs_baseline == Decimal("0")
                assert fa.trade_count_vs_baseline == 0
