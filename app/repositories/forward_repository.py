"""ForwardRepository — DB access and crash-safe recovery for forward paper trading.

Responsibilities:
  - Get-or-create the single permanent ForwardLaunch row for a strategy,
    refusing to start if the frozen config hash has changed.
  - Idempotent creation of ForwardSignalEvaluation rows: at most one per
    (launch_id, candle_close_time), so a restart between persisting an
    evaluation and committing never duplicates it.
  - Recovery queries: open position, pending order, latest evaluation
    (source of highest-high / trailing-stop continuity), paper account.
  - Consistency checks: raise ForwardStateInconsistentError rather than
    silently opening/closing a position when recovered state is ambiguous.

PAPER/TEST only. No real orders are placed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.forward.exceptions import ForwardConfigMismatchError, ForwardStateInconsistentError
from app.models.forward_paper import ForwardLaunch, ForwardSignalEvaluation
from app.models.order import Order
from app.models.paper_account import PaperAccount
from app.models.position import Position
from app.models.system_event import SystemEvent
from app.models.trade import Trade
from app.schemas.common import ForwardLaunchStatus, OrderSide, OrderStatus, PositionStatus

_FORWARD_ENGINE_SOURCE = "forward_engine"

_PENDING_ORDER_STATUSES = (OrderStatus.PENDING_APPROVAL, OrderStatus.NEW)


@dataclass
class RecoveredState:
    launch: ForwardLaunch
    account: PaperAccount
    open_position: Position | None
    pending_order: Order | None
    latest_evaluation: ForwardSignalEvaluation | None


class ForwardRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------ launch

    def get_launch(self, strategy_name: str, symbol: str) -> ForwardLaunch | None:
        return self.session.scalars(
            select(ForwardLaunch)
            .where(ForwardLaunch.strategy_name == strategy_name)
            .where(ForwardLaunch.symbol == symbol)
        ).first()

    def get_or_create_launch(
        self,
        *,
        strategy_name: str,
        strategy_version: str,
        symbol: str,
        frozen_config_hash: str,
        frozen_config_json: str,
        code_commit_hash: str | None,
        now: datetime,
        requested_initial_capital: Decimal,
    ) -> ForwardLaunch:
        """Return the permanent launch record, creating it on first call.

        launch_timestamp is set exactly once, on creation, and never
        updated on subsequent calls — restarts always return the same row.
        If a launch already exists, requested_initial_capital is ignored
        (the spec forbids changing capital after launch); only the
        frozen_config_hash is verified to guard against silent drift.
        """
        existing = self.get_launch(strategy_name, symbol)
        if existing is not None:
            if existing.frozen_config_hash != frozen_config_hash:
                raise ForwardConfigMismatchError(
                    f"Frozen config hash mismatch for {strategy_name}/{symbol}: "
                    f"launch recorded {existing.frozen_config_hash}, "
                    f"current code computes {frozen_config_hash}. "
                    "Refusing to start — the frozen configuration must never change."
                )
            return existing

        launch = ForwardLaunch(
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            symbol=symbol,
            frozen_config_hash=frozen_config_hash,
            frozen_config_json=frozen_config_json,
            code_commit_hash=code_commit_hash,
            launch_timestamp=now,
            initial_capital=requested_initial_capital,
            status=ForwardLaunchStatus.ACTIVE,
        )
        self.session.add(launch)
        self.session.flush()
        return launch

    def mark_launch_status(self, launch: ForwardLaunch, status: ForwardLaunchStatus) -> None:
        launch.status = status
        self.session.flush()

    def mark_last_evaluated_candle(
        self, launch: ForwardLaunch, candle_close_time: datetime
    ) -> None:
        launch.last_evaluated_candle_close = candle_close_time
        self.session.flush()

    # ----------------------------------------------------------------- account

    def get_account(self, launch: ForwardLaunch) -> PaperAccount | None:
        return self.session.scalars(
            select(PaperAccount).where(PaperAccount.launch_id == launch.id)
        ).first()

    def get_or_create_account(self, launch: ForwardLaunch) -> PaperAccount:
        account = self.get_account(launch)
        if account is not None:
            return account
        account = PaperAccount(
            balance=launch.initial_capital,
            asset_balance=Decimal("0"),
            equity=launch.initial_capital,
            realized_pnl=Decimal("0"),
            total_fees_paid=Decimal("0"),
            currency="USDT",
            launch_id=launch.id,
        )
        self.session.add(account)
        self.session.flush()
        return account

    # ---------------------------------------------------------------- position

    def get_open_position(self, launch: ForwardLaunch) -> Position | None:
        rows = list(
            self.session.scalars(
                select(Position)
                .where(Position.launch_id == launch.id)
                .where(Position.status == PositionStatus.OPEN)
            ).all()
        )
        if len(rows) > 1:
            raise ForwardStateInconsistentError(
                f"Launch {launch.id} has {len(rows)} open positions; expected at most 1."
            )
        return rows[0] if rows else None

    # ------------------------------------------------------------------- order

    def get_pending_order(self, launch: ForwardLaunch) -> Order | None:
        rows = list(
            self.session.scalars(
                select(Order)
                .where(Order.launch_id == launch.id)
                .where(Order.status.in_(_PENDING_ORDER_STATUSES))
            ).all()
        )
        if len(rows) > 1:
            raise ForwardStateInconsistentError(
                f"Launch {launch.id} has {len(rows)} pending orders; expected at most 1."
            )
        return rows[0] if rows else None

    # -------------------------------------------------------------- evaluation

    def get_evaluation(
        self, launch: ForwardLaunch, candle_close_time: datetime
    ) -> ForwardSignalEvaluation | None:
        return self.session.scalars(
            select(ForwardSignalEvaluation)
            .where(ForwardSignalEvaluation.launch_id == launch.id)
            .where(ForwardSignalEvaluation.candle_close_time == candle_close_time)
        ).first()

    def get_latest_evaluation(self, launch: ForwardLaunch) -> ForwardSignalEvaluation | None:
        return self.session.scalars(
            select(ForwardSignalEvaluation)
            .where(ForwardSignalEvaluation.launch_id == launch.id)
            .order_by(ForwardSignalEvaluation.candle_close_time.desc())
            .limit(1)
        ).first()

    def list_evaluations(
        self, launch: ForwardLaunch, limit: int | None = None
    ) -> list[ForwardSignalEvaluation]:
        stmt = (
            select(ForwardSignalEvaluation)
            .where(ForwardSignalEvaluation.launch_id == launch.id)
            .order_by(ForwardSignalEvaluation.candle_close_time.asc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt).all())

    def create_evaluation(
        self,
        launch: ForwardLaunch,
        *,
        candle_close_time: datetime,
        signal: str,
        reasons: list[str],
        raw_market_price: Decimal | None,
        planned_execution_time: datetime | None,
        entry_donchian_level: Decimal | None,
        exit_donchian_level: Decimal | None,
        ema_200: Decimal | None,
        ema_slope: Decimal | None,
        atr: Decimal | None,
        initial_stop: Decimal | None,
        current_trailing_stop: Decimal | None,
        highest_high_since_entry: Decimal | None,
        position_quantity: Decimal | None,
        cash: Decimal,
        equity: Decimal,
    ) -> ForwardSignalEvaluation:
        """Idempotent insert: returns the existing row if this candle was already evaluated."""
        existing = self.get_evaluation(launch, candle_close_time)
        if existing is not None:
            return existing

        evaluation = ForwardSignalEvaluation(
            launch_id=launch.id,
            candle_close_time=candle_close_time,
            signal=signal,
            reasons=reasons,
            raw_market_price=raw_market_price,
            planned_execution_time=planned_execution_time,
            entry_donchian_level=entry_donchian_level,
            exit_donchian_level=exit_donchian_level,
            ema_200=ema_200,
            ema_slope=ema_slope,
            atr=atr,
            initial_stop=initial_stop,
            current_trailing_stop=current_trailing_stop,
            highest_high_since_entry=highest_high_since_entry,
            position_quantity=position_quantity,
            cash=cash,
            equity=equity,
            frozen_config_hash=launch.frozen_config_hash,
        )
        self.session.add(evaluation)
        self.session.flush()
        launch.last_evaluated_candle_close = candle_close_time
        return evaluation

    # --------------------------------------------------------------- recovery

    def recover(self, launch: ForwardLaunch) -> RecoveredState:
        """Recover full forward-trading state for a launch.

        Raises ForwardStateInconsistentError if the open position and
        pending order disagree about whether the account should be flat or
        long — in that case the caller must not open or close positions
        and must require manual intervention.
        """
        account = self.get_or_create_account(launch)
        position = self.get_open_position(launch)
        order = self.get_pending_order(launch)
        evaluation = self.get_latest_evaluation(launch)

        if position is not None and order is not None and order.side == OrderSide.BUY:
            raise ForwardStateInconsistentError(
                f"Launch {launch.id}: an open position exists alongside a pending "
                "BUY order. Cannot determine true state — manual intervention required."
            )
        if position is None and order is not None and order.side == OrderSide.SELL:
            raise ForwardStateInconsistentError(
                f"Launch {launch.id}: a pending SELL order exists with no open "
                "position. Cannot determine true state — manual intervention required."
            )

        return RecoveredState(
            launch=launch,
            account=account,
            open_position=position,
            pending_order=order,
            latest_evaluation=evaluation,
        )

    # ----------------------------------------------------------------- export

    def list_orders(self, launch: ForwardLaunch) -> list[Order]:
        return list(
            self.session.scalars(
                select(Order).where(Order.launch_id == launch.id).order_by(Order.created_at.asc())
            ).all()
        )

    def list_trades(self, launch: ForwardLaunch) -> list[Trade]:
        return list(
            self.session.scalars(
                select(Trade).where(Trade.launch_id == launch.id).order_by(Trade.closed_at.asc())
            ).all()
        )

    def list_system_events(self, launch: ForwardLaunch) -> list[SystemEvent]:
        """SystemEvent has no launch_id column; filter by source + details payload."""
        rows = list(
            self.session.scalars(
                select(SystemEvent)
                .where(SystemEvent.source == _FORWARD_ENGINE_SOURCE)
                .order_by(SystemEvent.timestamp.asc())
            ).all()
        )
        return [
            r for r in rows if r.details is not None and r.details.get("launch_id") == launch.id
        ]
