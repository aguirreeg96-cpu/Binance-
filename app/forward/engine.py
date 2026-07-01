"""ForwardPaperEngine — Stage 6.1 forward paper trading evaluation cycle.

Mirrors DonchianBreakoutEngine.run()'s exact per-candle ordering (execute
pending order at open -> conservative intrabar stop check -> record
equity -> generate new signal at close), but operates on DB-persisted state
recovered via ForwardRepository instead of an in-memory portfolio, so a
crash or restart between any two steps never duplicates an order, a
position, a trade, or a signal evaluation.

Indicator continuity (ATR/EMA) and "highest high since entry" are never
persisted as transient values — they are recomputed each cycle from the
canonical stored candle history, which is the only source of truth.

4h candles are built directly from closed 15m candles using fixed UTC
boundaries (00:00, 04:00, 08:00, ...). A bucket is only emitted once all 16
expected 15m candles are present and contiguous; this guarantees every
forward 4h candle is genuinely boundary-aligned even across data gaps,
which the general-purpose (and frozen) Stage 5.2C aggregator does not
guarantee by itself.

PAPER/TEST only. No real orders, no real capital, no private keys.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtesting.breakout_family import BREAKOUT_SOURCE_INTERVAL_MS
from app.backtesting.donchian_engine import _compute_atr, _compute_ema
from app.backtesting.execution import buy_exec_price, compute_buy, compute_sell, sell_exec_price
from app.forward.exceptions import ForwardStateInconsistentError
from app.forward.manifest import (
    FORWARD_DONCHIAN_CONFIG,
    FORWARD_FEE_PERCENTAGE,
    FORWARD_SLIPPAGE_PERCENTAGE,
    FORWARD_SOURCE_INTERVAL,
    FORWARD_SYMBOL,
    FORWARD_TRADING_INTERVAL,
    FROZEN_CONFIG_HASH,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    build_frozen_manifest,
    get_code_commit_hash,
)
from app.market_data.interval_utils import INTERVAL_MS
from app.models.candle import Candle
from app.models.forward_paper import ForwardLaunch
from app.models.order import Order
from app.models.position import Position
from app.models.system_event import SystemEvent
from app.models.trade import Trade
from app.repositories.candle_repository import CandleRepository
from app.repositories.forward_repository import ForwardRepository
from app.schemas.common import (
    EventLevel,
    ForwardLaunchStatus,
    ForwardSignalState,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionStatus,
    TradingMode,
)

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_FOUR_HOUR_MS = INTERVAL_MS["4h"]
_FEE_RATE = FORWARD_FEE_PERCENTAGE / _HUNDRED
_SLIPPAGE_RATE = FORWARD_SLIPPAGE_PERCENTAGE / _HUNDRED

DEFAULT_DATA_GAP_GRACE_SECONDS = 1800


# ---------------------------------------------------------------------------
# Time helpers — all DateTime columns in this codebase store naive UTC.
# ---------------------------------------------------------------------------


def _to_ms(dt: datetime) -> int:
    aware = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return int(aware.timestamp() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).replace(tzinfo=None)


def _align_up_to_4h(ms: int) -> int:
    rem = ms % _FOUR_HOUR_MS
    return ms if rem == 0 else ms + (_FOUR_HOUR_MS - rem)


# ---------------------------------------------------------------------------
# UTC-aligned 4h aggregation
# ---------------------------------------------------------------------------


def build_4h_candles(candles_15m: list[Candle]) -> list[Candle]:
    """Aggregate closed 15m candles into strictly UTC-aligned 4h candles.

    Buckets are keyed by the 4h boundary each candle's open_time falls
    into. A bucket is emitted only once all 16 expected 15m candles are
    present and strictly contiguous; incomplete or gappy buckets are
    silently dropped. No candle is ever invented.
    """
    factor = _FOUR_HOUR_MS // BREAKOUT_SOURCE_INTERVAL_MS
    buckets: dict[int, list[Candle]] = {}
    for c in candles_15m:
        bucket_open = c.open_time - (c.open_time % _FOUR_HOUR_MS)
        buckets.setdefault(bucket_open, []).append(c)

    result: list[Candle] = []
    for bucket_open in sorted(buckets):
        group = sorted(buckets[bucket_open], key=lambda c: c.open_time)
        if len(group) != factor:
            continue
        if any(
            group[i].open_time != bucket_open + i * BREAKOUT_SOURCE_INTERVAL_MS
            for i in range(factor)
        ):
            continue

        first, last = group[0], group[-1]
        result.append(
            Candle(
                symbol=first.symbol,
                interval=FORWARD_TRADING_INTERVAL,
                open_time=first.open_time,
                open=first.open,
                high=max(g.high for g in group),
                low=min(g.low for g in group),
                close=last.close,
                volume=sum((g.volume for g in group), _ZERO),
                close_time=last.close_time,
                quote_asset_volume=sum((g.quote_asset_volume for g in group), _ZERO),
                trades=sum(g.trades for g in group),
                taker_buy_base_volume=sum((g.taker_buy_base_volume for g in group), _ZERO),
                taker_buy_quote_volume=sum((g.taker_buy_quote_volume for g in group), _ZERO),
                is_closed=True,
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Launch lifecycle
# ---------------------------------------------------------------------------


def start_or_resume_launch(
    session: Session, *, now: datetime, initial_capital: Decimal
) -> ForwardLaunch:
    """Get-or-create the permanent ForwardLaunch row for the frozen B_4h config.

    Raises ForwardConfigMismatchError (via ForwardRepository) if a prior
    launch's frozen config hash no longer matches what the current code
    computes — the app must refuse to start rather than silently trade
    under a different configuration. requested_initial_capital is ignored
    once a launch already exists.
    """
    repo = ForwardRepository(session)
    manifest = build_frozen_manifest()
    launch = repo.get_or_create_launch(
        strategy_name=STRATEGY_NAME,
        strategy_version=STRATEGY_VERSION,
        symbol=FORWARD_SYMBOL,
        frozen_config_hash=FROZEN_CONFIG_HASH,
        frozen_config_json=json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        code_commit_hash=get_code_commit_hash(),
        now=now,
        requested_initial_capital=initial_capital,
    )
    session.commit()
    return launch


# ---------------------------------------------------------------------------
# Evaluation outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationOutcome:
    candle_close_time: datetime
    signal: str
    reasons: list[str]


@dataclass(frozen=True)
class CycleReport:
    """Diagnostic snapshot from one engine cycle.

    Always stored as ``engine.last_report`` after every ``run_cycle()`` call
    (except when ``launch.status != ACTIVE``).  Callers — primarily the CLI —
    use this to surface operator-facing info without re-querying the DB.

    Policy documented here:
    A 4h candle is eligible when ALL of the following are true:
      1. Formed by exactly 16 closed, contiguous 15m candles.
      2. ``open_time >= launch_timestamp``  ← fully forward; no pre-launch price action.
      3. ``close_time <= now``              ← already closed at the time of evaluation.
      4. No existing ForwardSignalEvaluation for this candle's close_time.

    When the launch timestamp falls *inside* a 4h bucket (not at a UTC 4h
    boundary), that bucket is intentionally excluded even if it has already
    closed: part of its price action predates the launch and it is not a
    genuinely forward candle.  ``skipped_partial_launch_candle`` records this.
    The first eligible candle is the one that opens at the next 4h UTC boundary
    after ``launch_timestamp``.
    """

    now_utc: datetime
    """The ``now`` value passed to run_cycle(); naive UTC."""
    launch_timestamp_utc: datetime
    """The permanent launch timestamp; naive UTC."""
    candles_15m_available: int
    """Closed 15m candles in the DB for this symbol/interval."""
    first_15m_open_utc: datetime | None
    """Open time of the earliest stored 15m candle; naive UTC."""
    last_15m_open_utc: datetime | None
    """Open time of the most recent stored 15m candle; naive UTC."""
    candles_4h_complete: int
    """Fully-formed 4h buckets (16 contiguous closed 15m candles each)."""
    last_4h_open_time: datetime | None
    """Open time of the most recently assembled 4h candle; naive UTC."""
    last_4h_close_time: datetime | None
    """Close time of the most recently assembled 4h candle; naive UTC."""
    last_evaluated_close: datetime | None
    """close_time of the last 4h candle already in the DB; naive UTC."""
    expected_next_open_utc: datetime | None
    """open_time of the next expected 4h candle; naive UTC."""
    skipped_partial_launch_candle: bool
    """True when launch_timestamp is not at a 4h UTC boundary and no
    evaluations have occurred yet.  The candle whose open_time < launch is
    excluded on policy grounds (see class docstring)."""
    next_eligible_4h_close_utc: datetime | None
    """Approximate UTC close time of the next candle we are waiting for.
    Set when evaluated_count == 0 because the next candle hasn't closed yet.
    None when evaluations did occur this cycle."""
    evaluated_count: int
    """4h candles evaluated (and persisted to the DB) in this cycle."""


def _resolve_signal_state(
    *, position_closed: bool, position_open: bool, buy_queued: bool, sell_queued: bool
) -> str:
    """Precedence: EXITED > BUY_PENDING > SELL_PENDING > LONG > WAIT.

    ERROR_DATA_GAP is resolved separately by the gap-handling path, never
    through this function.
    """
    if position_closed:
        return ForwardSignalState.EXITED
    if buy_queued:
        return ForwardSignalState.BUY_PENDING
    if sell_queued:
        return ForwardSignalState.SELL_PENDING
    if position_open:
        return ForwardSignalState.LONG
    return ForwardSignalState.WAIT


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ForwardPaperEngine:
    """Forward paper-trading evaluation engine for the frozen B_4h config.

    Reads candles from the DB only; never calls Binance directly. Callers
    are responsible for syncing new closed 15m candles before calling
    run_cycle (see app.forward.sync).
    """

    def __init__(
        self,
        session: Session,
        *,
        data_gap_grace_seconds: int = DEFAULT_DATA_GAP_GRACE_SECONDS,
    ) -> None:
        self.session = session
        self.repo = ForwardRepository(session)
        self.candle_repo = CandleRepository(session)
        self.data_gap_grace_seconds = data_gap_grace_seconds
        self.last_report: CycleReport | None = None

    def run_cycle(self, launch: ForwardLaunch, now: datetime) -> list[EvaluationOutcome]:
        """Evaluate every closed 4h candle due since the last cycle, in order.

        Each candle's evaluation commits before the next one starts, so a
        crash mid-cycle resumes exactly where it left off on restart.

        Eligibility policy (see CycleReport docstring for the full spec):
          - open_time >= launch_timestamp  (fully forward; no pre-launch candles)
          - close_time <= now              (already closed)
          - not yet persisted in ForwardSignalEvaluation

        When the launch timestamp falls inside a 4h bucket, that bucket is
        excluded even after it closes.  The first eligible candle is the one
        whose open_time equals the next 4h UTC boundary after launch_timestamp.
        This decision is logged at INFO level and recorded in last_report so
        operators can distinguish "waiting correctly" from a silent failure.
        """
        # Reset on each call so stale data is never read after an inactive cycle.
        self.last_report = None

        if launch.status != ForwardLaunchStatus.ACTIVE:
            return []

        # --- Compute expected next candle BEFORE querying the DB ---
        launch_ms = _to_ms(launch.launch_timestamp)
        last_ms = (
            _to_ms(launch.last_evaluated_candle_close)
            if launch.last_evaluated_candle_close is not None
            else None
        )
        expected_open_ms = (last_ms + 1) if last_ms is not None else _align_up_to_4h(launch_ms)

        # Partial-launch-candle policy: when there are no evaluations yet AND
        # the launch did not start exactly at a 4h UTC boundary, the candle
        # whose open_time < launch_timestamp is excluded.  Its price action
        # is not genuinely forward.  We skip to the next 4h boundary instead.
        skipped_partial = last_ms is None and launch_ms % _FOUR_HOUR_MS != 0

        try:
            self.repo.recover(launch)
        except ForwardStateInconsistentError as exc:
            self._handle_state_inconsistent(launch, exc)
            return []

        candles_15m = self.candle_repo.query(FORWARD_SYMBOL, FORWARD_SOURCE_INTERVAL)
        candles_4h = build_4h_candles(candles_15m)

        first_15m_ms = min((c.open_time for c in candles_15m), default=None)
        last_15m_ms = max((c.open_time for c in candles_15m), default=None)
        last_4h_open = _ms_to_dt(candles_4h[-1].open_time) if candles_4h else None
        last_4h_close = _ms_to_dt(candles_4h[-1].close_time) if candles_4h else None
        expected_next_open = _ms_to_dt(expected_open_ms)
        next_eligible_close = _ms_to_dt(expected_open_ms + _FOUR_HOUR_MS)

        logger.info(
            "=== Forward Cycle | now=%s | launch=%s"
            " | 15m: count=%d first=%s last=%s"
            " | 4h: count=%d last=[%s → %s]"
            " | last_eval=%s | expect_open=%s | skipped_partial=%s ===",
            now.isoformat(),
            launch.launch_timestamp.isoformat(),
            len(candles_15m),
            _ms_to_dt(first_15m_ms).isoformat() if first_15m_ms is not None else "none",
            _ms_to_dt(last_15m_ms).isoformat() if last_15m_ms is not None else "none",
            len(candles_4h),
            last_4h_open.isoformat() if last_4h_open else "none",
            last_4h_close.isoformat() if last_4h_close else "none",
            launch.last_evaluated_candle_close.isoformat()
            if launch.last_evaluated_candle_close
            else "none",
            expected_next_open.isoformat(),
            skipped_partial,
        )

        if skipped_partial:
            partial_open_ms = launch_ms - (launch_ms % _FOUR_HOUR_MS)
            logger.info(
                "  [policy] Partial launch candle excluded: open=%s < launch=%s."
                " First eligible open=%s."
                " That candle started before the launch — its price action is not"
                " genuinely forward and is excluded from evaluation.",
                _ms_to_dt(partial_open_ms).isoformat(),
                launch.launch_timestamp.isoformat(),
                expected_next_open.isoformat(),
            )

        def _build_report(evaluated_count: int, *, with_next_close: bool) -> CycleReport:
            return CycleReport(
                now_utc=now,
                launch_timestamp_utc=launch.launch_timestamp,
                candles_15m_available=len(candles_15m),
                first_15m_open_utc=_ms_to_dt(first_15m_ms) if first_15m_ms is not None else None,
                last_15m_open_utc=_ms_to_dt(last_15m_ms) if last_15m_ms is not None else None,
                candles_4h_complete=len(candles_4h),
                last_4h_open_time=last_4h_open,
                last_4h_close_time=last_4h_close,
                last_evaluated_close=launch.last_evaluated_candle_close,
                expected_next_open_utc=expected_next_open,
                skipped_partial_launch_candle=skipped_partial,
                next_eligible_4h_close_utc=next_eligible_close if with_next_close else None,
                evaluated_count=evaluated_count,
            )

        if not candles_4h:
            logger.info("  No complete 4h candles available yet — waiting for data.")
            self.last_report = _build_report(0, with_next_close=True)
            return []

        index_by_open = {c.open_time: i for i, c in enumerate(candles_4h)}
        start_index = index_by_open.get(expected_open_ms)

        if start_index is None:
            logger.info(
                "  No new 4h candle to evaluate."
                " expected_open=%s not yet in DB."
                " skipped_partial_launch_candle=%s."
                " next_eligible_4h_close=%s",
                expected_next_open.isoformat(),
                skipped_partial,
                next_eligible_close.isoformat(),
            )
            self._check_for_data_gap(launch, expected_open_ms, now)
            self.last_report = _build_report(0, with_next_close=True)
            return []

        eligible_count = len(candles_4h) - start_index
        logger.info(
            "  Evaluating %d 4h candle(s) starting at open=%s",
            eligible_count,
            _ms_to_dt(candles_4h[start_index].open_time).isoformat(),
        )

        atrs = _compute_atr(candles_4h, FORWARD_DONCHIAN_CONFIG.atr_period)
        emas = _compute_ema([c.close for c in candles_4h], FORWARD_DONCHIAN_CONFIG.ema_period)

        outcomes: list[EvaluationOutcome] = []
        for all_i in range(start_index, len(candles_4h)):
            candle = candles_4h[all_i]
            logger.info(
                "  [%d/%d] open=%s close=%s",
                all_i - start_index + 1,
                eligible_count,
                _ms_to_dt(candle.open_time).isoformat(),
                _ms_to_dt(candle.close_time).isoformat(),
            )
            try:
                outcome = self._evaluate_candle(launch, candles_4h, atrs, emas, all_i)
            except ForwardStateInconsistentError as exc:
                self._handle_state_inconsistent(launch, exc)
                break
            outcomes.append(outcome)
            logger.info(
                "  [%d/%d] → signal=%s reasons=%s",
                all_i - start_index + 1,
                eligible_count,
                outcome.signal,
                outcome.reasons,
            )
            self.session.commit()

        self.last_report = _build_report(len(outcomes), with_next_close=False)
        return outcomes

    def _handle_state_inconsistent(
        self, launch: ForwardLaunch, exc: ForwardStateInconsistentError
    ) -> None:
        """No abrir ni cerrar posiciones; marcar ERROR; registrar SystemEvent.

        Requires manual intervention before evaluation can resume.
        """
        self.session.rollback()
        self.repo.mark_launch_status(launch, ForwardLaunchStatus.ERROR)
        self.session.add(
            SystemEvent(
                level=EventLevel.ERROR,
                source="forward_engine",
                message=f"Forward state inconsistent: {exc}",
                details={"launch_id": launch.id},
                trading_mode=TradingMode.PAPER,
            )
        )
        self.session.commit()

    # ------------------------------------------------------------ data gaps

    def _check_for_data_gap(
        self, launch: ForwardLaunch, expected_open_ms: int, now: datetime
    ) -> None:
        expected_close_ms = expected_open_ms + _FOUR_HOUR_MS
        now_ms = _to_ms(now)
        if now_ms < expected_close_ms:
            logger.info(
                "  Candle open=%s is not due yet (closes ~%s, now=%s) — waiting.",
                _ms_to_dt(expected_open_ms).isoformat(),
                _ms_to_dt(expected_close_ms).isoformat(),
                _ms_to_dt(now_ms).isoformat(),
            )
            return

        elapsed_seconds = (now_ms - expected_close_ms) / 1000
        if elapsed_seconds <= self.data_gap_grace_seconds:
            logger.warning(
                "Forward engine: 4h candle due at open_time=%d not yet available "
                "(retrying within %ds grace period).",
                expected_open_ms,
                self.data_gap_grace_seconds,
            )
            return

        self._record_data_gap(launch, expected_open_ms, expected_close_ms)

    def _record_data_gap(
        self, launch: ForwardLaunch, expected_open_ms: int, expected_close_ms: int
    ) -> None:
        account = self.repo.get_or_create_account(launch)
        position = self.repo.get_open_position(launch)

        self.repo.create_evaluation(
            launch,
            candle_close_time=_ms_to_dt(expected_close_ms - 1),
            signal=ForwardSignalState.ERROR_DATA_GAP,
            reasons=["MISSING_15M_SOURCE_CANDLES"],
            raw_market_price=None,
            planned_execution_time=None,
            entry_donchian_level=None,
            exit_donchian_level=None,
            ema_200=None,
            ema_slope=None,
            atr=None,
            initial_stop=position.stop_loss if position is not None else None,
            current_trailing_stop=(position.trailing_stop_price if position is not None else None),
            highest_high_since_entry=None,
            position_quantity=position.quantity if position is not None else None,
            cash=account.balance,
            equity=account.equity,
        )
        self.session.add(
            SystemEvent(
                level=EventLevel.ERROR,
                source="forward_engine",
                message=(
                    f"Data gap: expected 4h candle open_time={expected_open_ms} still "
                    f"missing after the {self.data_gap_grace_seconds}s grace period. "
                    "Skipping past it to avoid blocking forward evaluation forever."
                ),
                details={"launch_id": launch.id, "expected_open_ms": expected_open_ms},
                trading_mode=TradingMode.PAPER,
            )
        )
        self.session.commit()

    # ------------------------------------------------------------ per-candle

    def _evaluate_candle(
        self,
        launch: ForwardLaunch,
        candles_4h: list[Candle],
        atrs: list[Decimal | None],
        emas: list[Decimal | None],
        all_i: int,
    ) -> EvaluationOutcome:
        dcfg = FORWARD_DONCHIAN_CONFIG
        candle = candles_4h[all_i]
        atr_val = atrs[all_i]
        ema_val = emas[all_i]
        ema_prev = (
            emas[all_i - dcfg.ema_slope_lookback] if all_i >= dcfg.ema_slope_lookback else None
        )

        reasons: list[str] = []
        position_closed = False
        buy_queued = False
        sell_queued = False

        # ---- 1. Execute pending order at THIS candle's open ----
        pending_order = self.repo.get_pending_order(launch)
        if pending_order is not None:
            if pending_order.side == OrderSide.BUY:
                self._fill_buy_order(launch, pending_order, candle, atrs, all_i)
                reasons.append("ENTRY_FILLED_AT_OPEN")
            else:
                self._fill_sell_order(launch, pending_order, candle, "DONCHIAN_CHANNEL_EXIT")
                reasons.append("EXIT_FILLED_AT_OPEN")
                position_closed = True

        # ---- 2. Conservative intrabar stop check ----
        position = self.repo.get_open_position(launch)
        if position is not None:
            stop_reason = self._check_stop_exit(launch, position, candle, candles_4h, atr_val)
            if stop_reason is not None:
                position_closed = True
                reasons.append(stop_reason)

        # ---- 3. Snapshot account/equity ----
        account = self.repo.get_or_create_account(launch)
        position = self.repo.get_open_position(launch)
        position_value = position.quantity * candle.close if position is not None else _ZERO
        equity = account.balance + position_value
        account.equity = equity
        self.session.flush()

        # ---- 4. Generate new signal at CLOSE (queued for next open) ----
        entry_level: Decimal | None = None
        exit_level: Decimal | None = None
        if position is None:
            if ema_val is not None and atr_val is not None and all_i >= dcfg.entry_lookback:
                entry_level = max(c.high for c in candles_4h[all_i - dcfg.entry_lookback : all_i])
                slope_ok = ema_prev is not None and ema_val > ema_prev
                if candle.close > entry_level and candle.close > ema_val and slope_ok:
                    self._queue_buy_order(launch, candle)
                    buy_queued = True
                    reasons.append("DONCHIAN_BREAKOUT_SIGNAL")
        else:
            if all_i >= dcfg.exit_lookback:
                exit_level = min(c.low for c in candles_4h[all_i - dcfg.exit_lookback : all_i])
                if candle.close < exit_level:
                    self._queue_sell_order(launch, position, candle)
                    sell_queued = True
                    reasons.append("DONCHIAN_CHANNEL_EXIT_SIGNAL")

        if not reasons:
            reasons.append("NO_SIGNAL")

        position = self.repo.get_open_position(launch)
        signal = _resolve_signal_state(
            position_closed=position_closed,
            position_open=position is not None,
            buy_queued=buy_queued,
            sell_queued=sell_queued,
        )

        highest_high = (
            self._highest_high_since_entry(candles_4h, position, candle.close_time)
            if position is not None
            else None
        )

        evaluation = self.repo.create_evaluation(
            launch,
            candle_close_time=_ms_to_dt(candle.close_time),
            signal=signal,
            reasons=reasons,
            raw_market_price=candle.close,
            planned_execution_time=(
                _ms_to_dt(candle.close_time + 1) if (buy_queued or sell_queued) else None
            ),
            entry_donchian_level=entry_level,
            exit_donchian_level=exit_level,
            ema_200=ema_val,
            ema_slope=(
                (ema_val - ema_prev) if (ema_val is not None and ema_prev is not None) else None
            ),
            atr=atr_val,
            initial_stop=position.stop_loss if position is not None else None,
            current_trailing_stop=(position.trailing_stop_price if position is not None else None),
            highest_high_since_entry=highest_high,
            position_quantity=position.quantity if position is not None else None,
            cash=account.balance,
            equity=account.equity,
        )
        self.session.flush()

        return EvaluationOutcome(
            candle_close_time=evaluation.candle_close_time,
            signal=evaluation.signal,
            reasons=list(evaluation.reasons),
        )

    # ------------------------------------------------------------ execution

    def _fill_buy_order(
        self,
        launch: ForwardLaunch,
        order: Order,
        candle: Candle,
        atrs: list[Decimal | None],
        all_i: int,
    ) -> None:
        dcfg = FORWARD_DONCHIAN_CONFIG
        account = self.repo.get_or_create_account(launch)
        exec_price = buy_exec_price(candle.open, _SLIPPAGE_RATE)
        alloc_fraction = dcfg.allocation_pct / _HUNDRED
        capital_to_deploy = account.balance * alloc_fraction
        quantity, fee = compute_buy(capital_to_deploy, exec_price, _FEE_RATE)

        entry_atr = atrs[all_i - 1] if all_i > 0 else None
        atr = entry_atr if (entry_atr is not None and entry_atr > _ZERO) else _ZERO
        initial_stop = (exec_price - dcfg.atr_multiplier * atr) if atr > _ZERO else _ZERO

        account.balance -= capital_to_deploy
        account.asset_balance = quantity
        account.total_fees_paid += fee

        position = Position(
            symbol=FORWARD_SYMBOL,
            status=PositionStatus.OPEN,
            entry_price=exec_price,
            quantity=quantity,
            stop_loss=initial_stop,
            take_profit=_ZERO,  # no take-profit target: trailing-stop/channel-exit only
            trailing_stop_enabled=True,
            trailing_stop_price=initial_stop,
            opened_at=_ms_to_dt(candle.open_time),
            launch_id=launch.id,
        )
        self.session.add(position)
        self.session.flush()

        order.status = OrderStatus.FILLED
        order.quantity = quantity
        order.filled_quantity = quantity
        order.avg_fill_price = exec_price
        order.commission = fee
        order.position_id = position.id
        self.session.flush()

    def _fill_sell_order(
        self, launch: ForwardLaunch, order: Order, candle: Candle, exit_reason: str
    ) -> None:
        position = self.repo.get_open_position(launch)
        if position is None:
            raise ForwardStateInconsistentError(
                f"Launch {launch.id}: pending SELL order {order.id} has no open position to close."
            )
        exec_price = sell_exec_price(candle.open, _SLIPPAGE_RATE)
        self._close_position(launch, position, order, exec_price, exit_reason, candle.open_time)

    def _check_stop_exit(
        self,
        launch: ForwardLaunch,
        position: Position,
        candle: Candle,
        candles_4h: list[Candle],
        atr_val: Decimal | None,
    ) -> str | None:
        """Conservative: check the stop level set BEFORE this candle, then update."""
        dcfg = FORWARD_DONCHIAN_CONFIG
        stop = (
            position.trailing_stop_price
            if position.trailing_stop_price is not None
            else position.stop_loss
        )
        if stop <= _ZERO:
            return None

        if candle.open < stop:
            exec_price = sell_exec_price(candle.open, _SLIPPAGE_RATE)
            order = self._create_market_order(launch, OrderSide.SELL, position.quantity, candle)
            self._close_position(
                launch, position, order, exec_price, "DONCHIAN_GAP_STOP", candle.open_time
            )
            return "DONCHIAN_GAP_STOP"

        if candle.low <= stop:
            exec_price = sell_exec_price(stop, _SLIPPAGE_RATE)
            order = self._create_market_order(launch, OrderSide.SELL, position.quantity, candle)
            self._close_position(
                launch, position, order, exec_price, "DONCHIAN_STOP_LOSS", candle.open_time
            )
            return "DONCHIAN_STOP_LOSS"

        # Stop not hit — update trailing (never decreases)
        highest_so_far = self._highest_high_since_entry(candles_4h, position, candle.close_time)
        if atr_val is not None and atr_val > _ZERO:
            new_stop = highest_so_far - dcfg.atr_multiplier * atr_val
            if new_stop > stop:
                position.trailing_stop_price = new_stop
                self.session.flush()
        return None

    def _close_position(
        self,
        launch: ForwardLaunch,
        position: Position,
        order: Order,
        exec_price: Decimal,
        exit_reason: str,
        exec_time_ms: int,
    ) -> None:
        account = self.repo.get_or_create_account(launch)
        net_proceeds, fee = compute_sell(position.quantity, exec_price, _FEE_RATE)

        entry_order = self.session.scalars(
            select(Order).where(Order.position_id == position.id).where(Order.side == OrderSide.BUY)
        ).first()
        entry_fee = entry_order.commission if entry_order is not None else _ZERO

        gross_pnl = (exec_price - position.entry_price) * position.quantity
        net_pnl = gross_pnl - entry_fee - fee

        order.status = OrderStatus.FILLED
        order.quantity = position.quantity
        order.filled_quantity = position.quantity
        order.avg_fill_price = exec_price
        order.commission = fee
        order.position_id = position.id

        account.balance += net_proceeds
        account.asset_balance = _ZERO
        account.total_fees_paid += fee
        account.realized_pnl += net_pnl

        position.status = PositionStatus.CLOSED
        position.exit_price = exec_price
        position.realized_pnl = net_pnl
        position.closed_at = _ms_to_dt(exec_time_ms)

        trade = Trade(
            symbol=FORWARD_SYMBOL,
            entry_price=position.entry_price,
            exit_price=exec_price,
            quantity=position.quantity,
            side=OrderSide.BUY,
            gross_pnl=gross_pnl,
            commission=entry_fee + fee,
            net_pnl=net_pnl,
            planned_stop_loss=position.stop_loss,
            planned_take_profit=position.take_profit,
            exit_reason=exit_reason,
            trading_mode=TradingMode.PAPER,
            position_id=position.id,
            launch_id=launch.id,
            opened_at=position.opened_at,
            closed_at=position.closed_at,
        )
        self.session.add(trade)
        self.session.flush()

    def _create_market_order(
        self, launch: ForwardLaunch, side: str, quantity: Decimal, candle: Candle
    ) -> Order:
        order = Order(
            symbol=FORWARD_SYMBOL,
            client_order_id=f"fwd-{launch.id}-{side.lower()}-{candle.open_time}-stop",
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.NEW,
            quantity=quantity,
            trading_mode=TradingMode.PAPER,
            launch_id=launch.id,
        )
        self.session.add(order)
        self.session.flush()
        return order

    def _queue_buy_order(self, launch: ForwardLaunch, candle: Candle) -> None:
        order = Order(
            symbol=FORWARD_SYMBOL,
            client_order_id=f"fwd-{launch.id}-buy-{candle.close_time}",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.NEW,
            quantity=_ZERO,  # sized at fill time: market buy spends a % of available balance
            trading_mode=TradingMode.PAPER,
            launch_id=launch.id,
        )
        self.session.add(order)
        self.session.flush()

    def _queue_sell_order(self, launch: ForwardLaunch, position: Position, candle: Candle) -> None:
        order = Order(
            symbol=FORWARD_SYMBOL,
            client_order_id=f"fwd-{launch.id}-sell-{candle.close_time}",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.NEW,
            quantity=position.quantity,
            trading_mode=TradingMode.PAPER,
            position_id=position.id,
            launch_id=launch.id,
        )
        self.session.add(order)
        self.session.flush()

    @staticmethod
    def _highest_high_since_entry(
        candles_4h: list[Candle], position: Position | None, upto_close_time_ms: int
    ) -> Decimal:
        assert position is not None
        opened_ms = _to_ms(position.opened_at)
        highest = position.entry_price
        for c in candles_4h:
            if c.open_time < opened_ms:
                continue
            if c.close_time > upto_close_time_ms:
                break
            if c.high > highest:
                highest = c.high
        return highest
