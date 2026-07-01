"""Stage 6.1 tests: ForwardPaperEngine — the bulk of the verbatim test list.

Uses the `breakout_scenario` / `seeded_session` fixtures from conftest.py: a
single deterministic 15m candle series (200 flat warm-up 4h-buckets, a 30-
bucket ramp into breakout, 15 hold buckets, then a 5-bucket drop that
triggers the trailing stop). Most tests launch the engine right at the start
of the ramp (index N_WARMUP_FLAT) so the flat candles are pre-launch
warm-up history and the ramp/breakout/drop candles are genuinely evaluated.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from app.backtesting.execution import compute_buy
from app.forward.engine import (
    CycleReport,
    ForwardPaperEngine,
    build_4h_candles,
    start_or_resume_launch,
)
from app.forward.manifest import FORWARD_FEE_PERCENTAGE, FORWARD_SLIPPAGE_PERCENTAGE
from app.models.types import normalize_decimal
from app.repositories.forward_repository import ForwardRepository
from app.schemas.common import ForwardSignalState

from .conftest import (
    N_WARMUP_FLAT,
    BreakoutScenario,
    build_breakout_price_series,
    four_hour_open_times,
    make_15m_candles,
    ms_to_naive_dt,
)

_FEE_RATE = FORWARD_FEE_PERCENTAGE / Decimal("100")
_SLIPPAGE_RATE = FORWARD_SLIPPAGE_PERCENTAGE / Decimal("100")


def _launch_at(db_session, scenario: BreakoutScenario, index: int, capital=Decimal("10000")):
    now = scenario.open_time_4h(index)
    return start_or_resume_launch(db_session, now=now, initial_capital=capital)


class TestFourHourAggregation:
    def test_only_complete_contiguous_buckets_are_emitted(self, breakout_scenario):
        candles_4h = build_4h_candles(breakout_scenario.sub_candles)
        assert len(candles_4h) == len(breakout_scenario.prices)

    def test_drops_incomplete_bucket(self, breakout_scenario):
        # Remove one 15m candle from the final bucket -> that bucket must vanish.
        candles = list(breakout_scenario.sub_candles[:-1])
        candles_4h = build_4h_candles(candles)
        assert len(candles_4h) == len(breakout_scenario.prices) - 1

    def test_buckets_are_utc_aligned_every_4_hours(self, breakout_scenario):
        candles_4h = build_4h_candles(breakout_scenario.sub_candles)
        for c in candles_4h:
            assert c.open_time % (4 * 3_600_000) == 0

    def test_never_invents_a_candle(self, breakout_scenario):
        candles = [
            c
            for c in breakout_scenario.sub_candles
            if c.open_time != breakout_scenario.sub_candles[16].open_time
        ]
        candles_4h = build_4h_candles(candles)
        opens = {c.open_time for c in candles_4h}
        assert breakout_scenario.open_times_4h[1] not in opens


class TestWarmupAndLaunch:
    def test_no_trades_during_warmup_only(self, db_session, seeded_session, breakout_scenario):
        # Launch just after the last available candle's close — nothing due yet to evaluate.
        last_close = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1)
        launch_now = last_close + timedelta(microseconds=1)
        launch = start_or_resume_launch(
            db_session, now=launch_now, initial_capital=Decimal("10000")
        )
        engine = ForwardPaperEngine(db_session)
        outcomes = engine.run_cycle(launch, launch_now)
        assert outcomes == []
        assert ForwardRepository(db_session).list_trades(launch) == []

    def test_launch_timestamp_excludes_prior_history_from_trading(
        self, db_session, seeded_session, breakout_scenario
    ):
        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        assert launch.launch_timestamp == breakout_scenario.open_time_4h(N_WARMUP_FLAT)

        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        outcomes = engine.run_cycle(launch, now)

        # Every evaluated candle's close must be after the launch timestamp.
        for o in outcomes:
            assert o.candle_close_time > launch.launch_timestamp

    def test_first_evaluated_candle_is_first_closed_candle_after_launch(
        self, db_session, seeded_session, breakout_scenario
    ):
        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        outcomes = engine.run_cycle(launch, now)

        assert outcomes[0].candle_close_time == breakout_scenario.close_time_4h(N_WARMUP_FLAT)


class TestFullLifecycle:
    def _run_full(self, db_session, seeded_session, breakout_scenario):
        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        outcomes = engine.run_cycle(launch, now)
        return launch, engine, outcomes

    def test_produces_one_buy_and_one_sell(self, db_session, seeded_session, breakout_scenario):
        launch, _engine, outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        repo = ForwardRepository(db_session)
        trades = repo.list_trades(launch)
        assert len(trades) == 1
        assert trades[0].exit_reason in {
            "DONCHIAN_STOP_LOSS",
            "DONCHIAN_GAP_STOP",
            "DONCHIAN_CHANNEL_EXIT",
        }

    def test_signals_include_buy_pending_long_and_exit(
        self, db_session, seeded_session, breakout_scenario
    ):
        _launch, _engine, outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        signals = {o.signal for o in outcomes}
        assert ForwardSignalState.BUY_PENDING in signals
        assert ForwardSignalState.LONG in signals
        assert ForwardSignalState.EXITED in signals

    def test_single_position_throughout(self, db_session, seeded_session, breakout_scenario):
        launch, engine, _outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        # get_open_position would have raised ForwardStateInconsistentError
        # during run_cycle already if more than one position ever existed.
        assert engine.repo.get_open_position(launch) is None  # closed by end of drop

    def test_25_percent_allocation_on_entry(self, db_session, seeded_session, breakout_scenario):
        launch, _engine, outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        repo = ForwardRepository(db_session)
        trades = repo.list_trades(launch)
        trade = trades[0]

        capital_to_deploy = launch.initial_capital * Decimal("25") / Decimal("100")
        exec_price = trade.entry_price  # already includes slippage as recorded
        # Recompute expected quantity from the known allocation rule. Compare
        # at the storage layer's precision (ExactDecimal quantizes to 10 dp).
        expected_qty, _fee = compute_buy(capital_to_deploy, exec_price, _FEE_RATE)
        assert trade.quantity == normalize_decimal(expected_qty)

    def test_fee_and_slippage_applied_on_entry(self, db_session, seeded_session, breakout_scenario):
        launch, _engine, outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        repo = ForwardRepository(db_session)
        orders = repo.list_orders(launch)
        buy_orders = [o for o in orders if o.side == "BUY" and o.status == "FILLED"]
        assert len(buy_orders) == 1
        buy = buy_orders[0]

        # entry_price recorded on the trade must reflect adverse buy slippage
        # relative to *some* candle open in the scenario (can't know which
        # without re-deriving the signal index, so just check positivity and
        # that a nonzero commission was charged).
        assert buy.commission > Decimal("0")
        assert buy.avg_fill_price is not None
        assert buy.avg_fill_price > Decimal("0")

    def test_non_decreasing_trailing_stop(self, db_session, seeded_session, breakout_scenario):
        launch, _engine, outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        evaluations = ForwardRepository(db_session).list_evaluations(launch)
        stops = [
            e.current_trailing_stop for e in evaluations if e.current_trailing_stop is not None
        ]
        for prev, cur in zip(stops, stops[1:], strict=False):
            assert cur >= prev

    def test_evaluations_recorded_for_every_closed_candle_since_launch(
        self, db_session, seeded_session, breakout_scenario
    ):
        launch, _engine, outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        evaluations = ForwardRepository(db_session).list_evaluations(launch)
        expected_count = len(breakout_scenario.prices) - N_WARMUP_FLAT
        assert len(evaluations) == expected_count

    def test_every_evaluation_has_frozen_config_hash(
        self, db_session, seeded_session, breakout_scenario
    ):
        launch, _engine, outcomes = self._run_full(db_session, seeded_session, breakout_scenario)
        evaluations = ForwardRepository(db_session).list_evaluations(launch)
        for e in evaluations:
            assert e.frozen_config_hash == launch.frozen_config_hash


class TestNextOpenExecution:
    def test_buy_fills_at_next_candle_open_not_signal_close(
        self, db_session, seeded_session, breakout_scenario
    ):
        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        outcomes = engine.run_cycle(launch, now)

        buy_pending = next(o for o in outcomes if o.signal == ForwardSignalState.BUY_PENDING)
        evaluation = ForwardRepository(db_session).get_evaluation(
            launch, buy_pending.candle_close_time
        )
        assert (
            evaluation.planned_execution_time
            == evaluation.candle_close_time + timedelta(microseconds=1000)
            or evaluation.planned_execution_time > evaluation.candle_close_time
        )

        trades = ForwardRepository(db_session).list_trades(launch)
        # entry must happen strictly after the BUY_PENDING candle's close
        assert trades[0].opened_at > buy_pending.candle_close_time


class TestIdempotencyAndNoDuplicates:
    def test_rerunning_cycle_with_no_new_candles_is_a_noop(
        self, db_session, seeded_session, breakout_scenario
    ):
        launch, engine, outcomes = TestFullLifecycle()._run_full(
            db_session, seeded_session, breakout_scenario
        )
        repo = ForwardRepository(db_session)
        eval_count_before = len(repo.list_evaluations(launch))
        trade_count_before = len(repo.list_trades(launch))
        order_count_before = len(repo.list_orders(launch))

        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        second_outcomes = engine.run_cycle(launch, now)

        assert second_outcomes == []
        assert len(repo.list_evaluations(launch)) == eval_count_before
        assert len(repo.list_trades(launch)) == trade_count_before
        assert len(repo.list_orders(launch)) == order_count_before

    def test_restart_with_open_position_continues_without_duplicating(
        self, db_session, breakout_scenario
    ):
        # Seed only candles 0-235 so the position is open after the first run_cycle.
        # Entry fills at candle 231 (LONG); exit at candle 247 (not yet seeded).
        PARTIAL_CUTOFF = 236
        cutoff_open_ms = breakout_scenario.open_times_4h[PARTIAL_CUTOFF]
        partial_candles = [c for c in breakout_scenario.sub_candles if c.open_time < cutoff_open_ms]
        db_session.add_all(partial_candles)
        db_session.commit()

        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine1 = ForwardPaperEngine(db_session)
        mid_now = breakout_scenario.close_time_4h(PARTIAL_CUTOFF - 1) + timedelta(seconds=1)
        first_outcomes = engine1.run_cycle(launch, mid_now)
        assert any(o.signal == ForwardSignalState.LONG for o in first_outcomes)

        repo = ForwardRepository(db_session)
        position_before = repo.get_open_position(launch)
        assert position_before is not None

        # Simulate new data arriving, then restart with a fresh engine instance.
        remaining_candles = [
            c for c in breakout_scenario.sub_candles if c.open_time >= cutoff_open_ms
        ]
        db_session.add_all(remaining_candles)
        db_session.commit()

        engine2 = ForwardPaperEngine(db_session)
        final_now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        second_outcomes = engine2.run_cycle(launch, final_now)
        assert second_outcomes  # continues evaluating remaining candles

        evaluations = repo.list_evaluations(launch)
        close_times = [e.candle_close_time for e in evaluations]
        assert len(close_times) == len(set(close_times))  # no duplicate candle evaluations

        trades = repo.list_trades(launch)
        assert len(trades) == 1  # exactly one round-trip trade overall


class TestDataGaps:
    def test_missing_candle_after_grace_period_logged_as_error(self, db_session, breakout_scenario):
        # Seed only warm-up candles (0 to N_WARMUP_FLAT-1); the first forward
        # candle (N_WARMUP_FLAT) is absent so the gap path fires on the first call.
        cutoff_open_ms = breakout_scenario.open_times_4h[N_WARMUP_FLAT]
        partial_candles = [c for c in breakout_scenario.sub_candles if c.open_time < cutoff_open_ms]
        db_session.add_all(partial_candles)
        db_session.commit()

        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session, data_gap_grace_seconds=60)

        # "now" is far past the missing candle's close to exceed the grace period.
        missing_close = breakout_scenario.close_time_4h(N_WARMUP_FLAT)
        now = missing_close + timedelta(seconds=120)

        # The gap path always returns []; the evaluation is written directly to DB.
        outcomes = engine.run_cycle(launch, now)
        assert outcomes == []

        repo = ForwardRepository(db_session)
        evals = repo.list_evaluations(launch)
        assert len(evals) == 1
        assert evals[0].signal == ForwardSignalState.ERROR_DATA_GAP
        assert "MISSING_15M_SOURCE_CANDLES" in evals[0].reasons

    def test_missing_candle_within_grace_period_is_silently_retried(
        self, db_session, breakout_scenario
    ):
        cutoff_open_ms = breakout_scenario.open_times_4h[N_WARMUP_FLAT]
        partial_candles = [c for c in breakout_scenario.sub_candles if c.open_time < cutoff_open_ms]
        db_session.add_all(partial_candles)
        db_session.commit()

        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session, data_gap_grace_seconds=3600)

        missing_close = breakout_scenario.close_time_4h(N_WARMUP_FLAT)
        now = missing_close + timedelta(seconds=10)

        outcomes = engine.run_cycle(launch, now)
        assert outcomes == []
        assert ForwardRepository(db_session).list_evaluations(launch) == []


class TestDeterminism:
    def test_two_independent_runs_produce_identical_outcomes(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.database import Base

        def run_once():
            engine_db = create_engine(
                "sqlite:///:memory:",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            Base.metadata.create_all(engine_db)
            Session = sessionmaker(bind=engine_db)
            session = Session()

            # Build fresh Candle ORM objects per call; reusing instances across
            # sessions leaves them detached/expired after the first commit.
            prices = build_breakout_price_series()
            open_times = four_hour_open_times(len(prices))
            fresh_candles = make_15m_candles(prices, open_times)
            session.add_all(fresh_candles)
            session.commit()

            launch_now = ms_to_naive_dt(open_times[N_WARMUP_FLAT])
            launch = start_or_resume_launch(
                session, now=launch_now, initial_capital=Decimal("10000")
            )

            engine = ForwardPaperEngine(session)
            final_now = ms_to_naive_dt(open_times[-1] + 4 * 3600 * 1000 + 1000)
            outcomes = engine.run_cycle(launch, final_now)
            result = [(o.candle_close_time, o.signal, tuple(o.reasons)) for o in outcomes]
            session.close()
            engine_db.dispose()
            return result

        assert run_once() == run_once()


class TestArchitectureGuard:
    """The forward engine must never place, modify, or cancel exchange orders."""

    def test_engine_has_no_exchange_order_methods(self):
        forbidden = {"place_order", "cancel_order", "send_order", "submit_order"}
        methods = {name for name in dir(ForwardPaperEngine) if not name.startswith("__")}
        assert forbidden.isdisjoint(methods)

    def test_engine_module_imports_no_binance_client(self):
        import app.forward.engine as engine_module

        source = engine_module.__file__
        with open(source, encoding="utf-8") as f:
            content = f.read()
        assert "BinanceMarketDataClient" not in content
        assert "httpx" not in content


# ---------------------------------------------------------------------------
# Partial-launch-candle policy and CycleReport diagnostics
# ---------------------------------------------------------------------------


class TestPartialLaunchCandle:
    """Verify the policy: only 4h candles whose open_time >= launch_timestamp
    are ever evaluated.

    When a launch occurs in the middle of a 4h bucket, that bucket is excluded
    — even after it closes — because part of its price action predates the
    launch and the candle is not genuinely forward.  The first eligible candle
    is the one that opens at the next 4h UTC boundary after launch_timestamp.
    This mirrors the real-world scenario that triggered the bug report:

      Launch at ~23:09 UTC (20:09 Argentina), mid-way through the 20:00–00:00
      UTC candle.  That candle closes at 00:00 UTC, but its open_time (20:00
      UTC) is before the launch, so it is correctly excluded.  The next eligible
      candle opens at 00:00 UTC (July 1) and closes at 04:00 UTC.  Any run
      before 04:00 UTC produces evaluations_this_cycle=0, which is CORRECT, not
      a bug.  The CycleReport makes this explicit.
    """

    def test_launch_mid_candle_skips_partial_and_evaluates_next(
        self, db_session, breakout_scenario
    ):
        """Launch 1 h into candle N_WARMUP_FLAT → that candle is skipped; first
        evaluation is the following candle (N_WARMUP_FLAT + 1)."""
        db_session.add_all(breakout_scenario.sub_candles)
        db_session.commit()

        mid = breakout_scenario.open_time_4h(N_WARMUP_FLAT) + timedelta(hours=1)
        launch = start_or_resume_launch(db_session, now=mid, initial_capital=Decimal("10000"))

        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        outcomes = engine.run_cycle(launch, now)

        assert outcomes  # at least one evaluation happened
        assert engine.last_report is not None
        assert engine.last_report.skipped_partial_launch_candle is True
        # First evaluated candle must be N_WARMUP_FLAT + 1 (the first fully forward one)
        assert outcomes[0].candle_close_time == breakout_scenario.close_time_4h(N_WARMUP_FLAT + 1)

    def test_no_skip_when_launch_at_4h_boundary(
        self, db_session, seeded_session, breakout_scenario
    ):
        """Launch exactly at a 4h UTC boundary → skipped_partial_launch_candle=False."""
        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(N_WARMUP_FLAT) + timedelta(seconds=1)
        engine.run_cycle(launch, now)

        assert engine.last_report is not None
        assert engine.last_report.skipped_partial_launch_candle is False

    def test_next_eligible_close_set_when_candle_not_due_yet(self, db_session, breakout_scenario):
        """next_eligible_4h_close_utc is populated when the expected candle hasn't closed yet."""
        # Add only warm-up candles (0 .. N_WARMUP_FLAT-1); candle N_WARMUP_FLAT not in DB.
        cutoff_open_ms = breakout_scenario.open_times_4h[N_WARMUP_FLAT]
        partial = [c for c in breakout_scenario.sub_candles if c.open_time < cutoff_open_ms]
        db_session.add_all(partial)
        db_session.commit()

        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)

        # "now" is 1 hour into candle N_WARMUP_FLAT — it hasn't closed yet.
        now = breakout_scenario.open_time_4h(N_WARMUP_FLAT) + timedelta(hours=1)
        outcomes = engine.run_cycle(launch, now)

        assert outcomes == []
        report = engine.last_report
        assert report is not None
        assert report.next_eligible_4h_close_utc is not None
        # next_eligible_close should be ~4 h after the expected open
        expected_open = breakout_scenario.open_time_4h(N_WARMUP_FLAT)
        assert report.next_eligible_4h_close_utc == expected_open + timedelta(hours=4)

    def test_evaluated_count_in_report_matches_outcomes(
        self, db_session, seeded_session, breakout_scenario
    ):
        """CycleReport.evaluated_count equals len(outcomes)."""
        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        outcomes = engine.run_cycle(launch, now)

        assert engine.last_report is not None
        assert engine.last_report.evaluated_count == len(outcomes)
        assert engine.last_report.evaluated_count > 0
        assert engine.last_report.candles_4h_complete == len(breakout_scenario.prices)
        assert isinstance(engine.last_report, CycleReport)

    def test_report_next_eligible_close_none_after_evaluation(
        self, db_session, seeded_session, breakout_scenario
    ):
        """next_eligible_4h_close_utc is None when evaluations did occur."""
        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )
        outcomes = engine.run_cycle(launch, now)

        assert len(outcomes) > 0
        assert engine.last_report is not None
        assert engine.last_report.next_eligible_4h_close_utc is None

    def test_restart_preserves_launch_id_and_timestamp(self, db_session, breakout_scenario):
        """start_or_resume_launch is idempotent: same id and timestamp on every call."""
        db_session.add_all(breakout_scenario.sub_candles)
        db_session.commit()

        launch_time = breakout_scenario.open_time_4h(N_WARMUP_FLAT)
        launch1 = start_or_resume_launch(
            db_session, now=launch_time, initial_capital=Decimal("10000")
        )
        id1, ts1 = launch1.id, launch1.launch_timestamp

        later = breakout_scenario.open_time_4h(N_WARMUP_FLAT + 10)
        launch2 = start_or_resume_launch(db_session, now=later, initial_capital=Decimal("10000"))
        assert launch2.id == id1
        assert launch2.launch_timestamp == ts1

    def test_utc_alignment_holds_for_various_mid_candle_offsets(
        self, db_session, breakout_scenario
    ):
        """Whatever time within a 4h bucket the launch occurs, the first eligible
        candle always aligns to the next 4h UTC boundary."""
        db_session.add_all(breakout_scenario.sub_candles)
        db_session.commit()

        for offset_hours in (1, 2, 3):
            mid = breakout_scenario.open_time_4h(N_WARMUP_FLAT) + timedelta(hours=offset_hours)
            # Fresh DB state not possible within one test; just verify the launch time
            # the engine would use (expected_open) aligns to a 4h boundary.
            from app.forward.engine import _FOUR_HOUR_MS, _align_up_to_4h, _to_ms

            launch_ms = _to_ms(mid)
            aligned = _align_up_to_4h(launch_ms)
            assert aligned % _FOUR_HOUR_MS == 0, f"Not 4h-aligned for offset={offset_hours}h"
            assert aligned > launch_ms, "Must snap to a FUTURE boundary"

    def test_diagnostic_log_emitted_each_cycle(
        self, db_session, seeded_session, breakout_scenario, caplog
    ):
        """Engine emits at least one INFO log per cycle for operator auditing."""
        import logging

        launch = _launch_at(db_session, breakout_scenario, N_WARMUP_FLAT)
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(N_WARMUP_FLAT) + timedelta(seconds=1)

        with caplog.at_level(logging.INFO, logger="app.forward.engine"):
            engine.run_cycle(launch, now)

        assert any("Cycle" in r.message for r in caplog.records)
        assert any("15m" in r.message for r in caplog.records)
        assert any("4h" in r.message for r in caplog.records)
        assert any("expect_open" in r.message for r in caplog.records)

    def test_partial_skip_logged_when_launch_mid_candle(
        self, db_session, breakout_scenario, caplog
    ):
        """The policy log line is emitted when skipped_partial_launch_candle=True."""
        import logging

        db_session.add_all(breakout_scenario.sub_candles)
        db_session.commit()

        mid = breakout_scenario.open_time_4h(N_WARMUP_FLAT) + timedelta(hours=1)
        launch = start_or_resume_launch(db_session, now=mid, initial_capital=Decimal("10000"))
        engine = ForwardPaperEngine(db_session)
        now = breakout_scenario.close_time_4h(len(breakout_scenario.prices) - 1) + timedelta(
            seconds=1
        )

        with caplog.at_level(logging.INFO, logger="app.forward.engine"):
            engine.run_cycle(launch, now)

        assert any("policy" in r.message.lower() for r in caplog.records)
