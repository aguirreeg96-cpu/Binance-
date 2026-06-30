"""Tests for Stage 5.2B.1/5.2B.2 — Audited entry variant comparison + filter-execution fix.

Covers:
  - EntryFilterType.V1_BASELINE: always passes
  - EntryFilterType.V2_TREND_SLOPE: EMA50 slope (8c) + EMA200 slope (16c)
  - EntryFilterType.V3_ALIGNED_TREND: alignment + ascending + crossover lookback
  - EntryFilterType.V4_CROSSOVER_STRENGTH: EMA separation 0 / 0.05 / 0.10 %
  - EntryFilterType.V5_HIGHER_TIMEFRAME: 1h confirmation from 15m candles
  - build_htf_bars: 15m→1h aggregation, incomplete group excluded
  - No look-ahead: current HTF group never used
  - _FilteredStrategyEngine: correct baseline / passed / rejected / blocked counting
  - Signal invariants: passed + rejected = baseline; executed + blocked = passed
  - run_entry_comparison: 14 combos (7 variants × 2 exits), correct structure
  - run_entry_multi_period: standalone vs compounded equity separation
  - V4 monotonicity: passed(0%) >= passed(0.05%) >= passed(0.10%)
  - No trades_added for standard filters (filters only reduce, never add)
  - FilterAnalysis: trades_conserved + trades_eliminated = V1 total trades
  - Standalone equity formula: standalone_initial * (1 + return_pct / 100)
  - Compounded equity: 2024 initial = 2023 final (per combo)
  - Combined return formula: ((1+r23)*(1+r24) - 1) * 100
  - Trade identifiers: deterministic digest, signal_timestamp matching
  - All 5 exporters: files created, correct columns
  - Decimal serialization in JSON exporter
  - Determinism: identical inputs → identical outputs
  - Zero-trade scenarios
"""

import json
from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.entry_comparison import (
    ENTRY_VARIANT_NAMES,
    EXIT_CONFIG_NAMES,
    EntryComparisonReport,
    EntryMultiPeriodReport,
    TradeIdentifier,
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
_UNSET = object()


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
    ema_short: object = _UNSET,
    ema_medium: object = _UNSET,
    ema_long: object = _UNSET,
) -> IndicatorResult:
    """Return a copy of all_results[idx] with overridden EMA values (pass None to force None)."""
    from dataclasses import replace

    r = all_results[idx]
    return replace(
        r,
        ema_short=r.ema_short if ema_short is _UNSET else ema_short,
        ema_medium=r.ema_medium if ema_medium is _UNSET else ema_medium,
        ema_long=r.ema_long if ema_long is _UNSET else ema_long,
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
        closes = [_D(str(i)) for i in range(9)]  # 9 closes → 2 complete groups + 1 leftover
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
            ema_long_slope_candles=2,
        )
        # At an index with enough history, EMA should be rising
        idx = 30
        assert passes_entry_filter(idx, all_results, cfg) is True

    def test_fails_when_insufficient_history(self):
        all_results = self._results(30)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V2_TREND_SLOPE,
            ema_medium_slope_candles=8,
            ema_long_slope_candles=16,
        )
        # idx < ema_long_slope_candles → not enough history
        assert passes_entry_filter(5, all_results, cfg) is False

    def test_fails_when_ema_medium_is_none(self):
        all_results = self._results(30)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V2_TREND_SLOPE,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=2,
        )
        # Force the current ema_medium to None
        patched = _fake_result(all_results, 10, ema_medium=None)
        modified = list(all_results)
        modified[10] = patched
        assert passes_entry_filter(10, modified, cfg) is False

    def test_fails_when_ema_not_ascending(self):
        # Descending price series → EMAs should decline
        candles = []
        for i in range(40):
            price = str(200 - i * 2)
            candles.append(
                make_candle(
                    open_time=_BASE_TIME + i * _INTERVAL_MS,
                    open_p=price,
                    high_p=str(201 - i * 2),
                    low_p=str(199 - i * 2),
                    close_p=price,
                    interval="15m",
                )
            )
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V2_TREND_SLOPE,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=2,
        )
        assert passes_entry_filter(30, all_results, cfg) is False


# ---------------------------------------------------------------------------
# passes_entry_filter — V3_ALIGNED_TREND
# ---------------------------------------------------------------------------


class TestFilterV3AlignedTrend:
    def test_fails_when_ema_not_aligned(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=2,
            crossover_lookback_candles=2,
        )
        # Flat price → EMA short ≈ EMA medium → not aligned (short > medium fails or ==)
        # Most constant-price results will fail EMA alignment
        for idx in range(5, 25):
            r = all_results[idx]
            if r.ema_short is None or r.ema_medium is None or r.ema_long is None:
                continue
            # For constant price, short ≈ medium ≈ long → alignment check fails
            if r.ema_short <= r.ema_medium:
                assert passes_entry_filter(idx, all_results, cfg) is False

    def test_crossover_lookback_candles_default(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            crossover_lookback_candles=1,
        )
        # Should not crash
        for idx in range(len(all_results)):
            passes_entry_filter(idx, all_results, cfg)

    def test_requires_crossover_within_lookback(self):
        from app.indicators.calculator import IndicatorCalculator
        from app.indicators.schemas import CrossSignal

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=2,
            ema_long_slope_candles=2,
            crossover_lookback_candles=1,
        )
        # Find an index with no recent crossover and aligned EMAs — should fail V3
        for idx in range(5, 25):
            r = all_results[idx]
            if r.ema_short is None or r.ema_medium is None or r.ema_long is None:
                continue
            if not (r.ema_short > r.ema_medium > r.ema_long):
                continue
            # If no BULLISH crossover at this exact candle, with lookback=1 it fails
            if r.ema_short_medium_cross != CrossSignal.BULLISH:
                assert passes_entry_filter(idx, all_results, cfg) is False
                break

    def test_fails_insufficient_history(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V3_ALIGNED_TREND,
            ema_medium_slope_candles=8,
            ema_long_slope_candles=16,
        )
        assert passes_entry_filter(5, all_results, cfg) is False


# ---------------------------------------------------------------------------
# passes_entry_filter — V4_CROSSOVER_STRENGTH
# ---------------------------------------------------------------------------


class TestFilterV4CrossoverStrength:
    def test_passes_with_zero_threshold(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0"),
        )
        # With threshold=0, any result where ema_short >= ema_medium passes
        for idx in range(len(all_results)):
            r = all_results[idx]
            if r.ema_short is None or r.ema_medium is None or r.ema_medium <= _D("0"):
                continue
            expected = r.ema_short >= r.ema_medium
            assert passes_entry_filter(idx, all_results, cfg) == expected

    def test_fails_when_ema_is_none(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0"),
        )
        patched = _fake_result(all_results, 10, ema_short=None)
        modified = list(all_results)
        modified[10] = patched
        assert passes_entry_filter(10, modified, cfg) is False

    def test_higher_threshold_more_restrictive(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        cfg_0 = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0"),
        )
        cfg_005 = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0.05"),
        )
        cfg_010 = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("0.10"),
        )
        for idx in range(len(all_results)):
            p0 = passes_entry_filter(idx, all_results, cfg_0)
            p005 = passes_entry_filter(idx, all_results, cfg_005)
            p010 = passes_entry_filter(idx, all_results, cfg_010)
            # Monotonicity: higher threshold → equal or fewer passes
            assert p0 >= p005  # type: ignore[operator]
            assert p005 >= p010  # type: ignore[operator]

    def test_exact_separation_calculation(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(30))
        # Manually set EMA values to control separation
        for idx in range(len(all_results)):
            r = all_results[idx]
            if r.ema_medium is not None and r.ema_medium > _D("0"):
                ema_med = r.ema_medium
                # Set ema_short so separation = exactly 0.06%
                ema_short_006 = ema_med * _D("1.0006")
                modified = list(all_results)
                modified[idx] = _fake_result(all_results, idx, ema_short=ema_short_006)
                # Should pass 0% and 0.05% but pass 0.10% only if 0.06 >= 0.10 (it doesn't)
                cfg_005 = EntryFilterConfig(
                    filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
                    ema_separation_min_pct=_D("0.05"),
                )
                cfg_010 = EntryFilterConfig(
                    filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
                    ema_separation_min_pct=_D("0.10"),
                )
                assert passes_entry_filter(idx, modified, cfg_005) is True
                assert passes_entry_filter(idx, modified, cfg_010) is False
                break


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

    def test_fails_without_htf_bars(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(20))
        cfg = EntryFilterConfig(filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME)
        assert passes_entry_filter(8, all_results, cfg, htf_bars=None) is False

    def test_fails_with_empty_htf_bars(self):
        from app.indicators.calculator import IndicatorCalculator

        all_results = IndicatorCalculator(_ind_config()).calculate(_candles_15m(20))
        cfg = EntryFilterConfig(filter_type=EntryFilterType.V5_HIGHER_TIMEFRAME)
        assert passes_entry_filter(8, all_results, cfg, htf_bars=[]) is False

    def test_fails_when_not_enough_complete_groups(self):
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

    def test_exactly_14_combinations(self):
        report = self._run()
        assert len(report.combinations) == 14

    def test_7_entry_variants_present(self):
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
        assert len(pairs) == 14

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

    def test_standalone_equity_formula(self):
        """standalone_final_equity = standalone_initial * (1 + return_pct / 100)."""
        report = self._run()
        for c in report.combinations:
            expected = c.standalone_initial_capital * (
                Decimal("1") + c.result.total_return_pct / Decimal("100")
            )
            assert abs(c.standalone_final_equity - expected) < Decimal("0.0001")

    def test_standalone_initial_equals_config_capital(self):
        report = self._run()
        for c in report.combinations:
            assert c.standalone_initial_capital == _D("10000")


# ---------------------------------------------------------------------------
# Signal count invariants
# ---------------------------------------------------------------------------


class TestSignalCountInvariants:
    def _run(self, n: int = 60) -> EntryComparisonReport:
        candles = _candles_15m(n)
        warmup = _ind_config().warmup_candles
        return run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=n),
            indicator_config=_ind_config(),
        )

    def test_passed_plus_rejected_equals_baseline(self):
        """filter_passed_candidates + filter_rejected_candidates = baseline_buy_candidates."""
        report = self._run()
        for c in report.combinations:
            sc = c.signal_counts
            assert (
                sc.filter_passed_candidates + sc.filter_rejected_candidates
                == sc.baseline_buy_candidates
            )

    def test_executed_plus_blocked_equals_passed(self):
        """executed_buys + blocked_by_open_position = filter_passed_candidates."""
        report = self._run()
        for c in report.combinations:
            sc = c.signal_counts
            assert sc.executed_buys + sc.blocked_by_open_position == sc.filter_passed_candidates

    def test_executed_equals_total_trades(self):
        """executed_buys must equal result.total_trades."""
        report = self._run()
        for c in report.combinations:
            assert c.signal_counts.executed_buys == c.result.total_trades

    def test_v1_baseline_zero_rejected(self):
        """V1 filter never rejects; all baseline signals pass."""
        report = self._run()
        for c in report.combinations:
            if c.entry_variant == "ENTRY_V1_BASELINE":
                sc = c.signal_counts
                assert sc.filter_rejected_candidates == 0
                assert sc.filter_passed_candidates == sc.baseline_buy_candidates

    def test_baseline_candidates_same_across_variants(self):
        """All variants share the same V1 base engine → same baseline_buy_candidates."""
        report = self._run()
        for exit_name in EXIT_CONFIG_NAMES:
            combos = [c for c in report.combinations if c.exit_config_name == exit_name]
            baseline_counts = {c.signal_counts.baseline_buy_candidates for c in combos}
            assert len(baseline_counts) == 1, (
                f"Expected all variants to share the same baseline_buy_candidates "
                f"for exit={exit_name}, got {baseline_counts}"
            )

    def test_filter_analysis_invariants(self):
        """FilterAnalysis signal-count fields match SignalCounts."""
        report = self._run()
        for c in report.combinations:
            fa = c.filter_analysis
            sc = c.signal_counts
            assert fa.baseline_candidates == sc.baseline_buy_candidates
            assert fa.passed_candidates == sc.filter_passed_candidates
            assert fa.rejected_candidates == sc.filter_rejected_candidates
            assert fa.executed_buys == sc.executed_buys
            assert fa.blocked_by_open_position == sc.blocked_by_open_position


# ---------------------------------------------------------------------------
# V4 monotonicity
# ---------------------------------------------------------------------------


class TestV4Monotonicity:
    def _run(self) -> EntryComparisonReport:
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        return run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
        )

    def test_passed_monotone_per_exit(self):
        """Raising EMA separation threshold must not increase passed_candidates."""
        report = self._run()
        for exit_name in EXIT_CONFIG_NAMES:

            def _passed(vn: str, _exit: str = exit_name) -> int:
                for c in report.combinations:
                    if c.entry_variant == vn and c.exit_config_name == _exit:
                        return c.signal_counts.filter_passed_candidates
                raise AssertionError(f"variant {vn} not found")

            p0 = _passed("ENTRY_V4_SEPARATION_0")
            p005 = _passed("ENTRY_V4_SEPARATION_005")
            p010 = _passed("ENTRY_V4_SEPARATION_010")
            assert p0 >= p005, f"exit={exit_name}: 0% passed={p0} but 0.05% passed={p005}"
            assert p005 >= p010, f"exit={exit_name}: 0.05% passed={p005} but 0.10% passed={p010}"

    def test_trades_monotone_per_exit(self):
        """Raising EMA separation threshold must not increase executed_buys."""
        report = self._run()
        for exit_name in EXIT_CONFIG_NAMES:

            def _trades(vn: str, _exit: str = exit_name) -> int:
                for c in report.combinations:
                    if c.entry_variant == vn and c.exit_config_name == _exit:
                        return c.result.total_trades
                raise AssertionError(f"variant {vn} not found")

            t0 = _trades("ENTRY_V4_SEPARATION_0")
            t005 = _trades("ENTRY_V4_SEPARATION_005")
            t010 = _trades("ENTRY_V4_SEPARATION_010")
            assert t0 >= t005
            assert t005 >= t010


# ---------------------------------------------------------------------------
# Filter analysis correctness
# ---------------------------------------------------------------------------


class TestFilterAnalysis:
    def _run(self) -> EntryComparisonReport:
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        return run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
        )

    def test_v1_self_comparison_all_zeros(self):
        report = self._run()
        for c in report.combinations:
            if c.entry_variant == "ENTRY_V1_BASELINE":
                fa = c.filter_analysis
                assert fa.trades_eliminated == 0
                assert fa.trades_added == 0
                assert fa.return_pct_vs_baseline == Decimal("0")
                assert fa.trade_count_vs_baseline == 0

    def test_no_trades_added_for_standard_filters(self):
        """Standard filters (V2-V5) can only eliminate V1 trades, never add new ones."""
        report = self._run()
        for c in report.combinations:
            assert c.filter_analysis.trades_added == 0, (
                f"{c.entry_variant}/{c.exit_config_name}: expected 0 added trades, "
                f"got {c.filter_analysis.trades_added}"
            )

    def test_conserved_plus_eliminated_equals_v1_trades(self):
        """For each exit config: conserved + eliminated = V1 baseline total_trades."""
        report = self._run()
        for exit_name in EXIT_CONFIG_NAMES:
            v1_trades = next(
                c.result.total_trades
                for c in report.combinations
                if c.entry_variant == "ENTRY_V1_BASELINE" and c.exit_config_name == exit_name
            )
            for c in report.combinations:
                if c.exit_config_name != exit_name:
                    continue
                fa = c.filter_analysis
                assert fa.trades_conserved + fa.trades_eliminated == v1_trades, (
                    f"{c.entry_variant}/{exit_name}: "
                    f"conserved({fa.trades_conserved}) + eliminated({fa.trades_eliminated}) "
                    f"!= v1_trades({v1_trades})"
                )


# ---------------------------------------------------------------------------
# Trade identifiers
# ---------------------------------------------------------------------------


class TestTradeIdentifiers:
    def test_identifier_structure(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        report = run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
        )
        for c in report.combinations:
            for tid in c.trade_identifiers:
                assert isinstance(tid, TradeIdentifier)
                assert tid.symbol == "BTCUSDT"
                assert tid.interval == "15m"
                assert isinstance(tid.signal_timestamp, int)
                assert isinstance(tid.entry_price, Decimal)

    def test_digest_is_deterministic(self):
        tid = TradeIdentifier(
            symbol="BTCUSDT",
            interval="15m",
            signal_timestamp=1_000_000,
            execution_timestamp=1_900_000,
            entry_price=_D("50000"),
            exit_timestamp=2_800_000,
            exit_reason="ATR_STOP_LOSS",
        )
        assert tid.digest() == tid.digest()
        assert len(tid.digest()) == 16

    def test_different_trades_different_digest(self):
        base = TradeIdentifier(
            symbol="BTCUSDT",
            interval="15m",
            signal_timestamp=1_000_000,
            execution_timestamp=1_900_000,
            entry_price=_D("50000"),
            exit_timestamp=2_800_000,
            exit_reason="ATR_STOP_LOSS",
        )
        other = TradeIdentifier(
            symbol="BTCUSDT",
            interval="15m",
            signal_timestamp=2_000_000,  # different signal time
            execution_timestamp=1_900_000,
            entry_price=_D("50000"),
            exit_timestamp=2_800_000,
            exit_reason="ATR_STOP_LOSS",
        )
        assert base.digest() != other.digest()

    def test_identifiers_count_equals_total_trades(self):
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        report = run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=60),
            indicator_config=_ind_config(),
        )
        for c in report.combinations:
            assert len(c.trade_identifiers) == c.result.total_trades


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
            assert (
                c1.signal_counts.baseline_buy_candidates == c2.signal_counts.baseline_buy_candidates
            )


# ---------------------------------------------------------------------------
# run_entry_multi_period — equity separation and capital compounding
# ---------------------------------------------------------------------------


class TestCapitalCompounding:
    def _run_multi(self, capital: str = "10000") -> EntryMultiPeriodReport:
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg_23 = _cfg(n_candles=60, capital=capital)
        cfg_24 = _cfg(start_offset=60, n_candles=60, capital=capital)
        return run_entry_multi_period(
            candles_2023=candles,
            warmup_2023=warmup,
            config_2023=cfg_23,
            candles_2024=candles,
            warmup_2024=warmup,
            config_2024=cfg_24,
            indicator_config=_ind_config(),
        )

    def test_structure(self):
        report = self._run_multi()
        assert isinstance(report, EntryMultiPeriodReport)
        assert len(report.period_2023.combinations) == 14
        assert len(report.period_2024.combinations) == 14
        assert len(report.yearly_summary) == 14

    def test_2024_initial_capital_equals_2023_final(self):
        """Compounding: 2024 result.initial_capital = 2023 result.final_equity (per combo)."""
        report = self._run_multi()
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
            assert matching_24.result.initial_capital == matching_23.result.final_equity

    def test_standalone_equity_2023(self):
        """Standalone 2023: final = initial * (1 + ret_23 / 100)."""
        report = self._run_multi()
        _HUNDRED = Decimal("100")
        for s in report.yearly_summary:
            expected = s.standalone_initial * (Decimal("1") + s.return_pct_2023 / _HUNDRED)
            assert abs(s.standalone_final_2023 - expected) < Decimal("0.0001"), (
                f"{s.entry_variant}/{s.exit_config_name}: "
                f"standalone_final_2023={s.standalone_final_2023} vs expected={expected}"
            )

    def test_standalone_equity_2024(self):
        """Standalone 2024: final = original_initial * (1 + ret_24 / 100)."""
        report = self._run_multi()
        _HUNDRED = Decimal("100")
        for s in report.yearly_summary:
            # standalone_final_2024 uses the per-year original capital (config_2024.initial_capital)
            expected = s.standalone_initial * (Decimal("1") + s.return_pct_2024 / _HUNDRED)
            assert abs(s.standalone_final_2024 - expected) < Decimal("0.0001"), (
                f"{s.entry_variant}/{s.exit_config_name}: "
                f"standalone_final_2024={s.standalone_final_2024} vs expected={expected}"
            )

    def test_compounded_equity_chain(self):
        """Compounded: final_23 = initial_24; final_24 = initial_24 * (1 + ret_24/100)."""
        report = self._run_multi()
        _HUNDRED = Decimal("100")
        for s in report.yearly_summary:
            assert s.compounded_final_2023 == s.compounded_initial_2024, (
                f"{s.entry_variant}: compounded_final_2023 != compounded_initial_2024"
            )
            expected_24 = s.compounded_initial_2024 * (Decimal("1") + s.return_pct_2024 / _HUNDRED)
            assert abs(s.compounded_final_2024 - expected_24) < Decimal("0.0001")

    def test_compounded_initial_2023_equals_standalone(self):
        """compounded_initial_2023 = standalone_initial (same start for both)."""
        report = self._run_multi()
        for s in report.yearly_summary:
            assert s.compounded_initial_2023 == s.standalone_initial

    def test_combined_return_formula(self):
        """combined = ((1+ret_23)*(1+ret_24) - 1) * 100."""
        report = self._run_multi()
        _HUNDRED = Decimal("100")
        for s in report.yearly_summary:
            r23 = s.return_pct_2023 / _HUNDRED
            r24 = s.return_pct_2024 / _HUNDRED
            expected = ((Decimal("1") + r23) * (Decimal("1") + r24) - Decimal("1")) * _HUNDRED
            assert abs(s.combined_return_pct - expected) < Decimal("0.0001")

    def test_standalone_differs_from_compounded_when_2023_nonzero(self):
        """
        When 2023 has a non-zero return, standalone_final_2024 != compounded_final_2024
        because they use different starting capitals.
        """
        report = self._run_multi()
        _HUNDRED = Decimal("100")
        # Find any combo where 2023 return != 0
        for s in report.yearly_summary:
            if s.return_pct_2023 != Decimal("0"):
                # standalone uses original capital; compounded uses 2023 final
                if s.return_pct_2024 != Decimal("0"):
                    assert s.standalone_final_2024 != s.compounded_final_2024
                    return
        # If all returns happen to be 0 (flat price), skip this assertion

    def test_yearly_summary_all_14_combos(self):
        report = self._run_multi()
        pairs = {(s.entry_variant, s.exit_config_name) for s in report.yearly_summary}
        assert len(pairs) == 14


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
        assert len(rows) == 14
        assert "entry_variant" in rows[0]
        assert "return_pct" in rows[0]
        assert "standalone_final_equity" in rows[0]
        assert "baseline_buy_candidates" in rows[0]

    def test_export_entry_comparison_json(self, tmp_path):
        report = self._report()
        path = tmp_path / "ec.json"
        export_entry_comparison_json(report, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "warning" in data
        assert "combinations" in data
        assert len(data["combinations"]) == 14

    def test_json_decimal_as_string(self, tmp_path):
        report = self._report()
        path = tmp_path / "ec.json"
        export_entry_comparison_json(report, path)
        data = json.loads(path.read_text())
        assert isinstance(data["combinations"][0]["return_pct"], str)

    def test_export_yearly_entry_comparison_csv(self, tmp_path):
        import csv

        multi = self._multi_report()
        path = tmp_path / "yr.csv"
        export_yearly_entry_comparison_csv(multi, path)
        assert path.exists()
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 14
        assert "return_pct_2023" in rows[0]
        assert "combined_return_pct" in rows[0]
        assert "standalone_final_2023" in rows[0]
        assert "compounded_final_2024" in rows[0]

    def test_export_filter_analysis_csv(self, tmp_path):
        import csv

        report = self._report()
        path = tmp_path / "fa.csv"
        export_filter_analysis_csv(report, path)
        assert path.exists()
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 14
        assert "baseline_candidates" in rows[0]
        assert "passed_candidates" in rows[0]
        assert "trades_eliminated" in rows[0]
        assert "pnl_eliminated" in rows[0]

    def test_export_signal_counts_csv(self, tmp_path):
        import csv

        report = self._report()
        path = tmp_path / "sc.csv"
        export_signal_counts_csv(report, path)
        assert path.exists()
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 14
        assert "baseline_buy_candidates" in rows[0]
        assert "filter_passed_candidates" in rows[0]
        assert "blocked_by_open_position" in rows[0]

    def test_filter_analysis_csv_invariant_columns(self, tmp_path):
        """export_filter_analysis_csv includes pre-computed invariant check columns."""
        import csv

        report = self._report()
        path = tmp_path / "fa.csv"
        export_filter_analysis_csv(report, path)
        rows = list(csv.DictReader(path.open()))
        assert "invariant_passed_plus_rejected_eq_baseline" in rows[0]
        assert "invariant_executed_plus_blocked_plus_no_next_eq_passed" in rows[0]
        # All invariants should be True
        for row in rows:
            assert row["invariant_passed_plus_rejected_eq_baseline"] == "True"
            assert row["invariant_executed_plus_blocked_plus_no_next_eq_passed"] == "True"


# ---------------------------------------------------------------------------
# Stage 5.2B.2 — Filter-execution integration tests
# ---------------------------------------------------------------------------


class TestFilterExecutionCorrectness:
    """Verify that a rejected BUY signal is never executed, never queued, never creates a trade.

    Uses _FilteredStrategyEngine directly with AlwaysBuyEngine as the base and a V4 filter
    with an impossible threshold (100%) so every signal is rejected.
    PAPER/TEST only.
    """

    def _run_filtered(
        self,
        n: int = 30,
        sep_pct: str = "100",
        fee_pct: str = "0.1",
    ):
        from app.backtesting.entry_comparison import (
            _RISK_STOP_ONLY,
            _FilteredStrategyEngine,
        )
        from app.backtesting.v2_engine import V2BacktestEngine
        from app.indicators.calculator import IndicatorCalculator
        from app.strategy.config import StrategyEngineConfig
        from tests.backtesting.conftest import AlwaysBuyEngine

        candles = _candles_15m(n)
        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        warmup = _ind_config().warmup_candles

        base = AlwaysBuyEngine(StrategyEngineConfig())
        filter_cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D(sep_pct),
        )
        fengine = _FilteredStrategyEngine(base, all_results, filter_cfg, None)

        config = _cfg(n_candles=n, fee_percentage=fee_pct)
        v2 = V2BacktestEngine(
            config=config,
            risk_exit_config=_RISK_STOP_ONLY,
            strategy_engine=fengine,
            indicator_config=_ind_config(),
        )
        return v2.run(candles, warmup), fengine, config

    def test_all_rejected_filter_produces_zero_trades(self):
        result, _, _ = self._run_filtered()
        assert result.total_trades == 0

    def test_all_rejected_filter_produces_zero_fees(self):
        result, _, _ = self._run_filtered(fee_pct="0.1")
        assert result.total_fees == _D("0")

    def test_all_rejected_filter_keeps_initial_equity(self):
        result, _, config = self._run_filtered()
        assert result.final_equity == config.initial_capital

    def test_zero_pass_variant_has_no_drawdown(self):
        result, _, _ = self._run_filtered()
        assert result.total_trades == 0
        assert result.total_fees == _D("0")
        assert result.max_drawdown_pct == _D("0")

    def test_baseline_buy_candidates_nonzero_when_rejected(self):
        """AlwaysBuyEngine produces BUY candidates even though the filter rejects them all."""
        _, fengine, _ = self._run_filtered(n=30)
        assert fengine.baseline_buy_candidates > 0

    def test_rejected_equals_baseline_when_all_rejected(self):
        _, fengine, _ = self._run_filtered()
        assert fengine.filter_rejected_candidates == fengine.baseline_buy_candidates
        assert fengine.filter_passed_candidates == 0

    def test_baseline_pass_through_produces_trades(self):
        """V1_BASELINE filter passes all signals → at least 1 trade with AlwaysBuyEngine."""
        from app.backtesting.entry_comparison import (
            _RISK_STOP_ONLY,
            _FilteredStrategyEngine,
        )
        from app.backtesting.v2_engine import V2BacktestEngine
        from app.indicators.calculator import IndicatorCalculator
        from app.strategy.config import StrategyEngineConfig
        from tests.backtesting.conftest import AlwaysBuyEngine

        n = 30
        candles = _candles_15m(n)
        all_results = IndicatorCalculator(_ind_config()).calculate(candles)
        warmup = _ind_config().warmup_candles

        base = AlwaysBuyEngine(StrategyEngineConfig())
        filter_cfg = EntryFilterConfig(filter_type=EntryFilterType.V1_BASELINE)
        fengine = _FilteredStrategyEngine(base, all_results, filter_cfg, None)

        config = _cfg(n_candles=n, fee_percentage="0")
        v2 = V2BacktestEngine(
            config=config,
            risk_exit_config=_RISK_STOP_ONLY,
            strategy_engine=fengine,
            indicator_config=_ind_config(),
        )
        result = v2.run(candles, warmup)
        assert result.total_trades >= 1


# ---------------------------------------------------------------------------
# _FilteredStrategyEngine unit tests
# ---------------------------------------------------------------------------


class TestFilteredEngineUnit:
    """Direct unit tests for _FilteredStrategyEngine.evaluate() call-by-call."""

    def _results(self, n: int = 30) -> list[IndicatorResult]:
        from app.indicators.calculator import IndicatorCalculator

        return IndicatorCalculator(_ind_config()).calculate(_candles_15m(n))

    def test_returns_wait_when_rejected_no_position(self):
        from app.backtesting.entry_comparison import _FilteredStrategyEngine
        from app.strategy.config import StrategyEngineConfig
        from app.strategy.schemas import PositionContext, StrategyAction
        from tests.backtesting.conftest import AlwaysBuyEngine

        all_results = self._results()
        base = AlwaysBuyEngine(StrategyEngineConfig())
        filter_cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("100"),
        )
        fengine = _FilteredStrategyEngine(base, all_results, filter_cfg, None)
        pos_no = PositionContext(has_open_long_position=False)

        warmup_results = [r for r in all_results if r.warmup_complete]
        assert warmup_results, "need at least one warmup-complete candle"
        for r in warmup_results:
            decision = fengine.evaluate(r, pos_no)
            assert decision.action == StrategyAction.WAIT

    def test_returns_buy_when_passed_no_position(self):
        from app.backtesting.entry_comparison import _FilteredStrategyEngine
        from app.strategy.config import StrategyEngineConfig
        from app.strategy.schemas import PositionContext, StrategyAction
        from tests.backtesting.conftest import AlwaysBuyEngine

        all_results = self._results()
        base = AlwaysBuyEngine(StrategyEngineConfig())
        filter_cfg = EntryFilterConfig(filter_type=EntryFilterType.V1_BASELINE)
        fengine = _FilteredStrategyEngine(base, all_results, filter_cfg, None)
        pos_no = PositionContext(has_open_long_position=False)

        decisions = [fengine.evaluate(r, pos_no) for r in all_results if r.warmup_complete]
        assert any(d.action == StrategyAction.BUY for d in decisions)

    def test_sell_passes_through_when_position_open(self):
        from app.backtesting.entry_comparison import _FilteredStrategyEngine
        from app.strategy.config import StrategyEngineConfig
        from app.strategy.schemas import PositionContext, StrategyAction
        from tests.backtesting.conftest import AlwaysSellEngine

        all_results = self._results()
        sell_base = AlwaysSellEngine(StrategyEngineConfig())
        filter_cfg = EntryFilterConfig(
            filter_type=EntryFilterType.V4_CROSSOVER_STRENGTH,
            ema_separation_min_pct=_D("100"),
        )
        fengine = _FilteredStrategyEngine(sell_base, all_results, filter_cfg, None)
        pos_open = PositionContext(has_open_long_position=True)

        warmup_results = [r for r in all_results if r.warmup_complete]
        assert warmup_results
        for r in warmup_results:
            decision = fengine.evaluate(r, pos_open)
            assert decision.action == StrategyAction.SELL


# ---------------------------------------------------------------------------
# Candidate ID list tests
# ---------------------------------------------------------------------------


class TestCandidateIds:
    def _run(self, n: int = 60) -> EntryComparisonReport:
        candles = _candles_15m(n)
        warmup = _ind_config().warmup_candles
        return run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=n),
            indicator_config=_ind_config(),
        )

    def test_executed_buy_ids_subset_of_baseline_candidate_ids(self):
        report = self._run()
        for c in report.combinations:
            base_set = set(c.baseline_candidate_ids)
            exec_set = set(c.executed_buy_ids)
            assert exec_set.issubset(base_set), (
                f"{c.entry_variant}/{c.exit_config_name}: "
                "executed_buy_ids not a subset of baseline_candidate_ids"
            )

    def test_passed_union_rejected_ids_equals_base_ids(self):
        report = self._run()
        for c in report.combinations:
            combined = set(c.passed_candidate_ids) | set(c.rejected_candidate_ids)
            base_set = set(c.baseline_candidate_ids)
            assert combined == base_set

    def test_rejected_ids_disjoint_from_passed_ids(self):
        report = self._run()
        for c in report.combinations:
            assert set(c.passed_candidate_ids).isdisjoint(set(c.rejected_candidate_ids))

    def test_id_list_lengths_match_signal_counts(self):
        report = self._run()
        for c in report.combinations:
            sc = c.signal_counts
            assert len(c.baseline_candidate_ids) == sc.baseline_buy_candidates
            assert len(c.passed_candidate_ids) == sc.filter_passed_candidates
            assert len(c.rejected_candidate_ids) == sc.filter_rejected_candidates
            assert len(c.executed_buy_ids) == sc.executed_buys


# ---------------------------------------------------------------------------
# Extended signal-count invariants (Stage 5.2B.2)
# ---------------------------------------------------------------------------


class TestExtendedInvariants:
    def _run(self, n: int = 60) -> EntryComparisonReport:
        candles = _candles_15m(n)
        warmup = _ind_config().warmup_candles
        return run_entry_comparison(
            all_candles=candles,
            warmup_len=warmup,
            config=_cfg(n_candles=n),
            indicator_config=_ind_config(),
        )

    def test_passed_plus_rejected_equals_baseline(self):
        report = self._run()
        for c in report.combinations:
            sc = c.signal_counts
            assert (
                sc.filter_passed_candidates + sc.filter_rejected_candidates
                == sc.baseline_buy_candidates
            )

    def test_executed_plus_blocked_plus_no_next_equals_passed(self):
        """Extended invariant: executed + blocked_by_open + passed_without_next = passed."""
        report = self._run()
        for c in report.combinations:
            sc = c.signal_counts
            assert (
                sc.executed_buys + sc.blocked_by_open_position + sc.passed_without_next_candle
                == sc.filter_passed_candidates
            ), (
                f"{c.entry_variant}/{c.exit_config_name}: "
                f"{sc.executed_buys} + {sc.blocked_by_open_position} + "
                f"{sc.passed_without_next_candle} != {sc.filter_passed_candidates}"
            )

    def test_passed_without_next_candle_is_always_zero(self):
        """V2BacktestEngine never evaluates the last candle → always 0."""
        report = self._run()
        for c in report.combinations:
            assert c.signal_counts.passed_without_next_candle == 0

    def test_v1_baseline_has_empty_rejected_ids(self):
        report = self._run()
        for c in report.combinations:
            if c.entry_variant == "ENTRY_V1_BASELINE":
                assert c.rejected_candidate_ids == []
                assert len(c.passed_candidate_ids) == len(c.baseline_candidate_ids)


# ---------------------------------------------------------------------------
# Determinism — candidate IDs and filter effect on trade timestamps
# ---------------------------------------------------------------------------


class TestDeterminismExtended:
    def test_deterministic_rerun_candidate_ids(self):
        """Same inputs → identical candidate ID lists on repeated runs."""
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg = _cfg(n_candles=60)
        ind = _ind_config()

        r1 = run_entry_comparison(candles, warmup, cfg, ind)
        r2 = run_entry_comparison(candles, warmup, cfg, ind)

        for c1, c2 in zip(r1.combinations, r2.combinations, strict=True):
            assert c1.baseline_candidate_ids == c2.baseline_candidate_ids
            assert c1.passed_candidate_ids == c2.passed_candidate_ids
            assert c1.executed_buy_ids == c2.executed_buy_ids

    def test_v4_separation_010_has_leq_trades_than_v1(self):
        """Stricter filter → fewer or equal executed trades vs V1_BASELINE."""
        candles = _candles_15m(60)
        warmup = _ind_config().warmup_candles
        cfg = _cfg(n_candles=60)
        ind = _ind_config()

        report = run_entry_comparison(candles, warmup, cfg, ind)
        for exit_name in EXIT_CONFIG_NAMES:
            v1 = next(
                c
                for c in report.combinations
                if c.entry_variant == "ENTRY_V1_BASELINE" and c.exit_config_name == exit_name
            )
            v4_010 = next(
                c
                for c in report.combinations
                if c.entry_variant == "ENTRY_V4_SEPARATION_010" and c.exit_config_name == exit_name
            )
            assert len(v4_010.executed_buy_ids) <= len(v1.executed_buy_ids), (
                f"exit={exit_name}: V4_010 has more executed trades than V1"
            )
