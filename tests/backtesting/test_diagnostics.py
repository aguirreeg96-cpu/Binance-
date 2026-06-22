"""Tests for BacktestDiagnostics computation — Stage 5.1."""

from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.diagnostics import (
    BacktestDiagnostics,
    _percentile,
    _trade_candle_indices,
    compute_diagnostics,
)
from app.backtesting.engine import BacktestEngine
from app.backtesting.schemas import BacktestTrade
from app.indicators.schemas import IndicatorConfig
from tests.backtesting.conftest import (
    AlwaysBuyEngine,
    AlwaysWaitEngine,
    BuyThenSellEngine,
    make_candle,
    make_candles,
)

_D = Decimal
_INTERVAL_MS = 3_600_000  # 1 h
_BASE_TIME = 1_700_000_000_000  # arbitrary fixed timestamp

# Dec 31, 2023 18:00:00 UTC — 5 candles later crosses into Jan 2024
_MONTH_BASE = 1_704_045_600_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ind_config(**kw: int) -> IndicatorConfig:
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
    defaults.update(kw)
    return IndicatorConfig(**defaults)


def _cfg(n: int = 20, **kw: object) -> BacktestConfig:
    defaults: dict[str, object] = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start_ms": _BASE_TIME,
        "end_ms": _BASE_TIME + n * _INTERVAL_MS,
        "initial_capital": _D("10000"),
        "fee_percentage": _D("0"),
        "slippage_percentage": _D("0"),
        "force_close_at_end": True,
    }
    defaults.update(kw)
    return BacktestConfig(**defaults)


def _run_and_diagnose(
    candles: list,
    strategy: object,
    cfg: BacktestConfig,
    ind: IndicatorConfig | None = None,
    warmup_len: int = 0,
) -> tuple:
    ind = ind or _ind_config()
    result = BacktestEngine(cfg, strategy, ind).run(candles, warmup_len)
    diag = compute_diagnostics(
        result=result,
        all_candles=candles,
        warmup_len=warmup_len,
        strategy_engine=strategy,
        indicator_config=ind,
    )
    return result, diag


def _make_trade(
    entry_exec_time: int = _BASE_TIME,
    exit_exec_time: int = _BASE_TIME + _INTERVAL_MS * 2,
    is_forced_close: bool = False,
    net_pnl: str = "0",
    gross_pnl: str = "0",
    exit_reasons: tuple[str, ...] = (),
    trade_id: int = 1,
) -> BacktestTrade:
    return BacktestTrade(
        trade_id=trade_id,
        entry_signal_time=entry_exec_time,
        entry_exec_time=entry_exec_time,
        entry_exec_price=_D("100"),
        entry_fee=_D("0"),
        quantity=_D("1"),
        exit_signal_time=None if is_forced_close else exit_exec_time,
        exit_exec_time=exit_exec_time,
        exit_exec_price=_D("100"),
        exit_fee=_D("0"),
        gross_pnl=_D(gross_pnl),
        net_pnl=_D(net_pnl),
        return_pct=_D("0"),
        is_forced_close=is_forced_close,
        capital_at_entry=_D("10000"),
        exit_reasons=exit_reasons,
    )


def _price_candles(n: int, open_price: str = "100", last_close: str | None = None) -> list:
    """Make n candles with constant price, optionally different last close."""
    p = _D(open_price)
    out = []
    for i in range(n):
        cp = last_close if (last_close is not None and i == n - 1) else open_price
        out.append(
            make_candle(
                open_time=_BASE_TIME + i * _INTERVAL_MS,
                open_p=open_price,
                high_p=str(p * _D("1.01")),
                low_p=str(p * _D("0.99")),
                close_p=cp,
            )
        )
    return out


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_returns_none(self) -> None:
        assert _percentile([], 0.5) is None

    def test_single_element(self) -> None:
        assert _percentile([_D("5")], 0.5) == _D("5")

    def test_p0_returns_min(self) -> None:
        data = [_D("1"), _D("2"), _D("3")]
        assert _percentile(data, 0.0) == _D("1")

    def test_p1_returns_max(self) -> None:
        data = [_D("1"), _D("2"), _D("3")]
        assert _percentile(data, 1.0) == _D("3")

    def test_median_of_odd_list(self) -> None:
        # [1, 3, 5] → k=(3-1)*0.5=1, lo=hi=1 → s[1]=3
        assert _percentile([_D("1"), _D("3"), _D("5")], 0.5) == _D("3")

    def test_median_of_even_list(self) -> None:
        # [1, 2, 3, 4] → k=(4-1)*0.5=1.5 → lo=1, hi=2 → 2*0.5 + 3*0.5 = 2.5
        data = [_D("1"), _D("2"), _D("3"), _D("4")]
        assert _percentile(data, 0.5) == _D("2.5")

    def test_unsorted_input_is_sorted(self) -> None:
        data = [_D("5"), _D("1"), _D("3")]
        assert _percentile(data, 0.0) == _D("1")
        assert _percentile(data, 1.0) == _D("5")


# ---------------------------------------------------------------------------
# _trade_candle_indices
# ---------------------------------------------------------------------------


class TestTradeCandleIndices:
    def test_all_candles_in_range(self) -> None:
        candles = make_candles(5, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS)
        t = _make_trade(
            entry_exec_time=_BASE_TIME,
            exit_exec_time=_BASE_TIME + 4 * _INTERVAL_MS,
        )
        assert _trade_candle_indices(t, candles) == [0, 1, 2, 3, 4]

    def test_partial_range(self) -> None:
        candles = make_candles(5, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS)
        t = _make_trade(
            entry_exec_time=_BASE_TIME + _INTERVAL_MS,
            exit_exec_time=_BASE_TIME + 3 * _INTERVAL_MS,
        )
        assert _trade_candle_indices(t, candles) == [1, 2, 3]

    def test_no_overlap(self) -> None:
        candles = make_candles(3, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS)
        t = _make_trade(
            entry_exec_time=_BASE_TIME + 100 * _INTERVAL_MS,
            exit_exec_time=_BASE_TIME + 200 * _INTERVAL_MS,
        )
        assert _trade_candle_indices(t, candles) == []

    def test_forced_close_uses_close_time(self) -> None:
        candles = make_candles(5, base_open_time=_BASE_TIME, interval_ms=_INTERVAL_MS)
        # Forced close: close_time = open_time + 3_599_999
        t = _make_trade(
            entry_exec_time=_BASE_TIME,
            exit_exec_time=_BASE_TIME + 4 * _INTERVAL_MS + 3_599_999,
            is_forced_close=True,
        )
        # in_range uses c.close_time <= exit_exec_time
        indices = _trade_candle_indices(t, candles)
        assert len(indices) > 0


# ---------------------------------------------------------------------------
# No-trade scenario
# ---------------------------------------------------------------------------


class TestNoTrades:
    def test_returns_diagnostics_object(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        assert isinstance(diag, BacktestDiagnostics)

    def test_empty_breakdowns(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        assert result.total_trades == 0
        assert len(diag.exit_reason_breakdown) == 0
        assert len(diag.monthly_breakdown) == 0
        assert len(diag.trade_excursions) == 0

    def test_trade_distribution_none_when_no_trades(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        td = diag.trade_distribution
        assert td.median_net_pnl is None
        assert td.best_5_trade_ids == ()
        assert td.worst_5_trade_ids == ()

    def test_cost_breakdown_zeros_when_no_trades(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        cb = diag.cost_breakdown
        assert cb.gross_profit_before_costs == _D("0")
        assert cb.profitable_before_costs_but_losing_after == 0
        assert cb.average_gross_trade is None

    def test_cost_scenarios_still_four(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        assert len(diag.cost_scenarios) == 4


# ---------------------------------------------------------------------------
# Cost scenarios
# ---------------------------------------------------------------------------


class TestCostScenarios:
    def test_four_named_scenarios(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        names = {s.name for s in diag.cost_scenarios}
        assert names == {"no_costs", "fee_only", "slippage_only", "fee_and_slippage"}

    def test_no_costs_has_zero_fees(self) -> None:
        orig_fee = _D("0.1")
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=orig_fee, slippage_percentage=_D("0.05"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        no_costs = next(s for s in diag.cost_scenarios if s.name == "no_costs")
        assert no_costs.total_fees == _D("0")
        assert no_costs.fee_percentage == _D("0")
        assert no_costs.slippage_percentage == _D("0")

    def test_fee_only_has_zero_slippage_pct(self) -> None:
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=_D("0.1"), slippage_percentage=_D("0.05"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        fee_only = next(s for s in diag.cost_scenarios if s.name == "fee_only")
        assert fee_only.slippage_percentage == _D("0")
        assert fee_only.estimated_slippage_cost == _D("0")

    def test_slippage_only_has_zero_fees(self) -> None:
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=_D("0.1"), slippage_percentage=_D("0.05"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        slip_only = next(s for s in diag.cost_scenarios if s.name == "slippage_only")
        assert slip_only.fee_percentage == _D("0")
        assert slip_only.total_fees == _D("0")

    def test_fee_and_slippage_uses_original_params(self) -> None:
        orig_fee = _D("0.1")
        orig_slip = _D("0.05")
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=orig_fee, slippage_percentage=orig_slip)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        both = next(s for s in diag.cost_scenarios if s.name == "fee_and_slippage")
        assert both.fee_percentage == orig_fee
        assert both.slippage_percentage == orig_slip

    def test_no_costs_return_higher_than_with_fees(self) -> None:
        """Removing costs can only improve or maintain returns."""
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=_D("0.1"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        if result.total_trades > 0:
            no_costs = next(s for s in diag.cost_scenarios if s.name == "no_costs")
            both = next(s for s in diag.cost_scenarios if s.name == "fee_and_slippage")
            assert no_costs.return_pct >= both.return_pct


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------


class TestCostBreakdown:
    def test_profitable_before_costs_but_losing_after(self) -> None:
        """Tiny gross gain eaten by high fee → counted as flipped trade."""
        candles = _price_candles(6, open_price="100", last_close="100.1")
        cfg = _cfg(6, fee_percentage=_D("2"), force_close_at_end=True)
        result, diag = _run_and_diagnose(candles, AlwaysBuyEngine(), cfg)
        assert result.total_trades == 1
        t = result.trades[0]
        assert t.gross_pnl > _D("0")
        assert t.net_pnl < _D("0")
        assert diag.cost_breakdown.profitable_before_costs_but_losing_after == 1

    def test_no_trades_all_counts_zero(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        cb = diag.cost_breakdown
        assert cb.trades_losing_before_costs == 0
        assert cb.trades_losing_after_costs == 0
        assert cb.profitable_before_costs_but_losing_after == 0

    def test_total_cost_pct_of_capital_correct(self) -> None:
        """Costs as pct of initial capital should match result.total_fees / initial_capital."""
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=_D("0.1"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        expected = result.total_fees / result.initial_capital * _D("100")
        assert abs(diag.cost_breakdown.costs_as_pct_of_initial_capital - expected) < _D("1e-6")

    def test_average_fees_none_when_no_trades(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        assert diag.cost_breakdown.average_entry_fee is None
        assert diag.cost_breakdown.average_exit_fee is None
        assert diag.cost_breakdown.average_total_cost_per_trade is None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_has_correct_candle_times(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        b = diag.benchmark
        assert b.full_period_start_time == candles[0].open_time
        assert b.last_candle_time == candles[-1].open_time

    def test_benchmark_prices_are_positive(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        b = diag.benchmark
        assert b.full_period_start_price > _D("0")
        assert b.last_candle_price > _D("0")

    def test_constant_price_bah_negative_with_fees(self) -> None:
        """Buy and hold on constant price with fees → negative return."""
        candles = make_candles(10, price="100")
        cfg = _cfg(10, fee_percentage=_D("0.1"))
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), cfg)
        b = diag.benchmark
        assert b.buy_and_hold_full_period_pct < _D("0")

    def test_bah_both_are_decimal(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        b = diag.benchmark
        assert isinstance(b.buy_and_hold_full_period_pct, Decimal)
        assert isinstance(b.buy_and_hold_effective_period_pct, Decimal)


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------


class TestMonthlyBreakdown:
    def test_single_month_for_short_sequence(self) -> None:
        """10 candles (10h) → all trades in same hour-band month."""
        candles = make_candles(10, base_open_time=_BASE_TIME)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(10))
        if result.total_trades > 0:
            assert len(diag.monthly_breakdown) == 1

    def test_two_months_when_crossing_dec_jan(self) -> None:
        """Candles spanning Dec 31 18:00 → Jan 1 trades split into 2 months."""
        n = 20
        candles = make_candles(n, base_open_time=_MONTH_BASE)
        cfg = _cfg(n)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        months = {(m.year, m.month) for m in diag.monthly_breakdown}
        assert (2023, 12) in months
        assert (2024, 1) in months

    def test_monthly_net_pnl_sums_to_total(self) -> None:
        """Sum of monthly net P&L equals total trade net P&L."""
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        if result.total_trades > 0:
            monthly_total = sum(m.net_pnl for m in diag.monthly_breakdown)
            trade_total = sum(t.net_pnl for t in result.trades)
            assert monthly_total == trade_total

    def test_monthly_trade_counts_sum_to_total(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        if result.total_trades > 0:
            assert sum(m.trade_count for m in diag.monthly_breakdown) == result.total_trades


# ---------------------------------------------------------------------------
# Exit reason breakdown
# ---------------------------------------------------------------------------


class TestExitReasonBreakdown:
    def test_forced_close_reason_group(self) -> None:
        n = 6
        candles = make_candles(n)
        cfg = _cfg(n, force_close_at_end=True)
        result, diag = _run_and_diagnose(candles, AlwaysBuyEngine(), cfg)
        if result.total_trades > 0:
            reasons = {er.exit_reason for er in diag.exit_reason_breakdown}
            assert "FORCED_CLOSE" in reasons

    def test_trade_counts_sum_to_total(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        if result.total_trades > 0:
            total = sum(er.trade_count for er in diag.exit_reason_breakdown)
            assert total == result.total_trades

    def test_win_count_le_trade_count(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        for er in diag.exit_reason_breakdown:
            assert er.win_count <= er.trade_count

    def test_net_pnl_sums_match_total(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        if result.total_trades > 0:
            reason_total = sum(er.net_pnl for er in diag.exit_reason_breakdown)
            trade_total = sum(t.net_pnl for t in result.trades)
            assert reason_total == trade_total


# ---------------------------------------------------------------------------
# Trade distribution
# ---------------------------------------------------------------------------


class TestTradeDistribution:
    def test_empty_when_no_trades(self) -> None:
        candles = make_candles(10)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10))
        td = diag.trade_distribution
        assert td.median_net_pnl is None
        assert td.best_5_trade_ids == ()
        assert td.worst_5_trade_ids == ()
        assert td.pct_result_from_top_5 is None

    def test_single_trade_appears_in_both_best_and_worst(self) -> None:
        n = 6
        candles = make_candles(n)
        cfg = _cfg(n)
        result, diag = _run_and_diagnose(candles, AlwaysBuyEngine(), cfg)
        if result.total_trades == 1:
            td = diag.trade_distribution
            assert len(td.best_5_trade_ids) == 1
            assert len(td.worst_5_trade_ids) == 1
            assert td.best_5_trade_ids[0] == td.worst_5_trade_ids[0]

    def test_best5_worst5_no_overlap_when_enough_trades(self) -> None:
        candles = make_candles(30)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(30))
        if len(result.trades) >= 10:
            best = set(diag.trade_distribution.best_5_trade_ids)
            worst = set(diag.trade_distribution.worst_5_trade_ids)
            assert best.isdisjoint(worst)

    def test_median_within_min_max(self) -> None:
        candles = make_candles(30)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(30))
        if result.total_trades >= 1:
            td = diag.trade_distribution
            if td.median_net_pnl is not None:
                min_pnl = min(t.net_pnl for t in result.trades)
                max_pnl = max(t.net_pnl for t in result.trades)
                assert min_pnl <= td.median_net_pnl <= max_pnl

    def test_constant_price_zero_fee_has_zero_median(self) -> None:
        """With zero fees and constant price, all trades P&L = 0 → median = 0."""
        candles = make_candles(20, price="100")
        cfg = _cfg(20, fee_percentage=_D("0"), slippage_percentage=_D("0"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        if result.total_trades > 0:
            assert diag.trade_distribution.median_net_pnl == _D("0")


# ---------------------------------------------------------------------------
# Excursions (MFE / MAE)
# ---------------------------------------------------------------------------


class TestExcursions:
    def test_count_equals_trade_count(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        assert len(diag.trade_excursions) == result.total_trades

    def test_mfe_and_mae_non_negative(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        for ex in diag.trade_excursions:
            assert ex.max_favorable_excursion_pct >= _D("0")
            assert ex.max_adverse_excursion_pct >= _D("0")

    def test_highest_gte_lowest(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        for ex in diag.trade_excursions:
            assert ex.highest_price >= ex.lowest_price

    def test_mfe_approximately_one_pct_for_constant_price(self) -> None:
        """make_candles high=price*1.01 → MFE ≈ 1% when entry at open."""
        n = 20
        candles = make_candles(n, price="100")
        cfg = _cfg(n, fee_percentage=_D("0"), slippage_percentage=_D("0"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        for ex in diag.trade_excursions:
            assert abs(ex.max_favorable_excursion_pct - _D("1")) < _D("0.01")

    def test_mae_approximately_one_pct_for_constant_price(self) -> None:
        """make_candles low=price*0.99 → MAE ≈ 1% when entry at open."""
        n = 20
        candles = make_candles(n, price="100")
        cfg = _cfg(n, fee_percentage=_D("0"), slippage_percentage=_D("0"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        for ex in diag.trade_excursions:
            assert abs(ex.max_adverse_excursion_pct - _D("1")) < _D("0.01")

    def test_candles_in_trade_positive(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        for ex in diag.trade_excursions:
            assert ex.candles_in_trade >= 0


# ---------------------------------------------------------------------------
# Entry blockers
# ---------------------------------------------------------------------------


class TestEntryBlockers:
    def test_total_evaluated_equals_eval_len(self) -> None:
        n = 20
        candles = make_candles(n)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(n))
        assert diag.entry_blockers.total_evaluated == n

    def test_all_counts_non_negative(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(20))
        eb = diag.entry_blockers
        assert eb.warmup_incomplete >= 0
        assert eb.no_bullish_crossover >= 0
        assert eb.price_below_long_ema >= 0
        assert eb.rsi_outside_buy_range >= 0
        assert eb.insufficient_volume >= 0
        assert eb.position_already_open >= 0
        assert eb.bearish_crossover >= 0
        assert eb.rsi_overbought >= 0
        assert eb.price_below_ema_for_sell >= 0

    def test_position_already_open_matches_equity_curve(self) -> None:
        """position_already_open == candles in equity_curve with open position."""
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        expected = sum(1 for ep in result.equity_curve if ep.has_open_position)
        assert diag.entry_blockers.position_already_open == expected

    def test_total_evaluated_with_warmup(self) -> None:
        """With warmup_len>0, total_evaluated still covers only eval candles."""
        candles = make_candles(15)
        result, diag = _run_and_diagnose(candles, AlwaysWaitEngine(), _cfg(10), warmup_len=5)
        assert diag.entry_blockers.total_evaluated == 10


# ---------------------------------------------------------------------------
# Engine audit (invariants verified through diagnostics)
# ---------------------------------------------------------------------------


class TestEngineAudit:
    def test_fees_charged_exactly_once(self) -> None:
        """result.total_fees == sum of per-trade entry + exit fees."""
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=_D("0.1"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        manual = sum(t.entry_fee + t.exit_fee for t in result.trades)
        assert result.total_fees == manual

    def test_no_negative_equity(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        for ep in result.equity_curve:
            assert ep.equity >= _D("0")
            assert ep.quote_balance >= _D("0")
            assert ep.base_balance >= _D("0")

    def test_exec_time_after_signal_time(self) -> None:
        """No look-ahead: entry_exec_time >= entry_signal_time for all trades."""
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        for t in result.trades:
            assert t.entry_exec_time > t.entry_signal_time
            if t.exit_signal_time is not None:
                assert t.exit_exec_time > t.exit_signal_time

    def test_trade_ids_sequential(self) -> None:
        candles = make_candles(20)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), _cfg(20))
        ids = [t.trade_id for t in result.trades]
        assert ids == list(range(1, len(ids) + 1))

    def test_capital_conserved_zero_costs(self) -> None:
        """With zero fees + slippage on constant price, equity stays at 10000."""
        candles = make_candles(20, price="100")
        cfg = _cfg(20, fee_percentage=_D("0"), slippage_percentage=_D("0"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        assert result.final_equity == _D("10000")

    def test_slippage_applied_once_per_leg(self) -> None:
        """Slippage-only scenario: cost_scenarios show correct estimated cost."""
        candles = make_candles(20, price="100")
        cfg = _cfg(20, fee_percentage=_D("0"), slippage_percentage=_D("0.1"))
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        slip_only = next(s for s in diag.cost_scenarios if s.name == "slippage_only")
        assert slip_only.estimated_slippage_cost > _D("0")
        # No fees in slippage-only scenario
        assert slip_only.total_fees == _D("0")

    def test_determinism(self) -> None:
        """Same candles and config produce identical diagnostics."""
        candles = make_candles(20)
        cfg = _cfg(20, fee_percentage=_D("0.1"))
        result1, diag1 = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        result2, diag2 = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        assert result1.total_trades == result2.total_trades
        assert result1.total_fees == result2.total_fees
        assert diag1.cost_scenarios == diag2.cost_scenarios
        assert diag1.monthly_breakdown == diag2.monthly_breakdown
        assert len(diag1.trade_excursions) == len(diag2.trade_excursions)

    def test_quarterly_breakdown_sums_match_monthly(self) -> None:
        """Sum of quarterly trade counts should equal sum of monthly trade counts."""
        n = 20
        candles = make_candles(n, base_open_time=_MONTH_BASE)
        cfg = _cfg(n)
        result, diag = _run_and_diagnose(candles, BuyThenSellEngine(), cfg)
        monthly_count = sum(m.trade_count for m in diag.monthly_breakdown)
        quarterly_count = sum(q.trade_count for q in diag.quarterly_breakdown)
        assert monthly_count == quarterly_count == result.total_trades
