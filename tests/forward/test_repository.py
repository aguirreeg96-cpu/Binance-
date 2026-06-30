"""Stage 6.1 tests: ForwardRepository — launch lifecycle, idempotency, recovery."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from app.forward.exceptions import ForwardConfigMismatchError, ForwardStateInconsistentError
from app.forward.manifest import FORWARD_SYMBOL, FROZEN_CONFIG_HASH, STRATEGY_NAME, STRATEGY_VERSION
from app.models.order import Order
from app.models.position import Position
from app.repositories.forward_repository import ForwardRepository
from app.schemas.common import (
    ForwardLaunchStatus,
    OrderSide,
    OrderStatus,
    PositionStatus,
    TradingMode,
)

NOW = datetime(2024, 1, 1, 0, 0, 0)


def _open_position(launch_id: int, **overrides) -> Position:
    fields = {
        "symbol": FORWARD_SYMBOL,
        "status": PositionStatus.OPEN,
        "entry_price": Decimal("100"),
        "quantity": Decimal("1"),
        "stop_loss": Decimal("90"),
        "take_profit": Decimal("0"),
        "opened_at": NOW,
        "launch_id": launch_id,
    }
    fields.update(overrides)
    return Position(**fields)


def _pending_order(launch_id: int, side: str, **overrides) -> Order:
    fields = {
        "symbol": FORWARD_SYMBOL,
        "client_order_id": f"test-{side.lower()}-{id(overrides)}",
        "side": side,
        "order_type": "MARKET",
        "status": OrderStatus.NEW,
        "quantity": Decimal("0"),
        "trading_mode": TradingMode.PAPER,
        "launch_id": launch_id,
    }
    fields.update(overrides)
    return Order(**fields)


def _make_launch(repo: ForwardRepository, now: datetime = NOW, capital: Decimal = Decimal("10000")):
    return repo.get_or_create_launch(
        strategy_name=STRATEGY_NAME,
        strategy_version=STRATEGY_VERSION,
        symbol=FORWARD_SYMBOL,
        frozen_config_hash=FROZEN_CONFIG_HASH,
        frozen_config_json="{}",
        code_commit_hash="deadbeef",
        now=now,
        requested_initial_capital=capital,
    )


class TestLaunchLifecycle:
    def test_creates_launch_on_first_call(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        assert launch.id is not None
        assert launch.launch_timestamp == NOW
        assert launch.initial_capital == Decimal("10000")
        assert launch.status == ForwardLaunchStatus.ACTIVE
        assert launch.frozen_config_hash == FROZEN_CONFIG_HASH

    def test_launch_timestamp_is_permanent_across_resumes(self, db_session):
        repo = ForwardRepository(db_session)
        first = _make_launch(repo, now=NOW)
        db_session.commit()

        later = datetime(2024, 6, 1, 0, 0, 0)
        second = repo.get_or_create_launch(
            strategy_name=STRATEGY_NAME,
            strategy_version=STRATEGY_VERSION,
            symbol=FORWARD_SYMBOL,
            frozen_config_hash=FROZEN_CONFIG_HASH,
            frozen_config_json="{}",
            code_commit_hash="deadbeef",
            now=later,
            requested_initial_capital=Decimal("10000"),
        )
        db_session.commit()

        assert second.id == first.id
        assert second.launch_timestamp == NOW  # unchanged

    def test_initial_capital_ignored_on_resume(self, db_session):
        repo = ForwardRepository(db_session)
        _make_launch(repo, capital=Decimal("10000"))
        db_session.commit()

        resumed = repo.get_or_create_launch(
            strategy_name=STRATEGY_NAME,
            strategy_version=STRATEGY_VERSION,
            symbol=FORWARD_SYMBOL,
            frozen_config_hash=FROZEN_CONFIG_HASH,
            frozen_config_json="{}",
            code_commit_hash="deadbeef",
            now=NOW,
            requested_initial_capital=Decimal("999999"),
        )
        assert resumed.initial_capital == Decimal("10000")

    def test_refuses_to_resume_on_hash_mismatch(self, db_session):
        repo = ForwardRepository(db_session)
        _make_launch(repo)
        db_session.commit()

        with pytest.raises(ForwardConfigMismatchError):
            repo.get_or_create_launch(
                strategy_name=STRATEGY_NAME,
                strategy_version=STRATEGY_VERSION,
                symbol=FORWARD_SYMBOL,
                frozen_config_hash="0" * 64,
                frozen_config_json="{}",
                code_commit_hash="deadbeef",
                now=NOW,
                requested_initial_capital=Decimal("10000"),
            )

    def test_mark_launch_status(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        repo.mark_launch_status(launch, ForwardLaunchStatus.STOPPED)
        assert launch.status == ForwardLaunchStatus.STOPPED


class TestAccount:
    def test_get_or_create_account_seeds_from_initial_capital(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo, capital=Decimal("5000"))
        db_session.commit()

        account = repo.get_or_create_account(launch)
        assert account.balance == Decimal("5000")
        assert account.equity == Decimal("5000")
        assert account.asset_balance == Decimal("0")

    def test_get_or_create_account_idempotent(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        a1 = repo.get_or_create_account(launch)
        a2 = repo.get_or_create_account(launch)
        assert a1.id == a2.id


class TestSinglePositionAndOrderEnforcement:
    def test_get_open_position_raises_if_more_than_one(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        for _ in range(2):
            db_session.add(_open_position(launch.id))
        db_session.commit()

        with pytest.raises(ForwardStateInconsistentError):
            repo.get_open_position(launch)

    def test_get_pending_order_raises_if_more_than_one(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        for i in range(2):
            db_session.add(_pending_order(launch.id, OrderSide.BUY, client_order_id=f"dup-{i}"))
        db_session.commit()

        with pytest.raises(ForwardStateInconsistentError):
            repo.get_pending_order(launch)


class TestEvaluationIdempotency:
    def test_create_evaluation_is_idempotent_per_candle(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        close_time = datetime(2024, 1, 1, 4, 0, 0)
        kwargs = {
            "candle_close_time": close_time,
            "signal": "WAIT",
            "reasons": ["NO_SIGNAL"],
            "raw_market_price": Decimal("100"),
            "planned_execution_time": None,
            "entry_donchian_level": None,
            "exit_donchian_level": None,
            "ema_200": None,
            "ema_slope": None,
            "atr": None,
            "initial_stop": None,
            "current_trailing_stop": None,
            "highest_high_since_entry": None,
            "position_quantity": None,
            "cash": Decimal("10000"),
            "equity": Decimal("10000"),
        }
        first = repo.create_evaluation(launch, **kwargs)
        second = repo.create_evaluation(launch, **kwargs)
        assert first.id == second.id

        all_evals = repo.list_evaluations(launch)
        assert len(all_evals) == 1

    def test_create_evaluation_updates_last_evaluated_candle(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        close_time = datetime(2024, 1, 1, 4, 0, 0)
        repo.create_evaluation(
            launch,
            candle_close_time=close_time,
            signal="WAIT",
            reasons=["NO_SIGNAL"],
            raw_market_price=Decimal("100"),
            planned_execution_time=None,
            entry_donchian_level=None,
            exit_donchian_level=None,
            ema_200=None,
            ema_slope=None,
            atr=None,
            initial_stop=None,
            current_trailing_stop=None,
            highest_high_since_entry=None,
            position_quantity=None,
            cash=Decimal("10000"),
            equity=Decimal("10000"),
        )
        assert launch.last_evaluated_candle_close == close_time


class TestRecovery:
    def test_recover_raises_on_open_position_with_pending_buy(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        db_session.add(_open_position(launch.id))
        db_session.add(_pending_order(launch.id, OrderSide.BUY, client_order_id="buy-1"))
        db_session.commit()

        with pytest.raises(ForwardStateInconsistentError):
            repo.recover(launch)

    def test_recover_raises_on_pending_sell_without_position(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        db_session.add(
            _pending_order(
                launch.id, OrderSide.SELL, client_order_id="sell-1", quantity=Decimal("1")
            )
        )
        db_session.commit()

        with pytest.raises(ForwardStateInconsistentError):
            repo.recover(launch)

    def test_recover_succeeds_with_consistent_flat_state(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        state = repo.recover(launch)
        assert state.open_position is None
        assert state.pending_order is None
        assert state.latest_evaluation is None
        assert state.account is not None

    def test_recover_succeeds_with_open_position_and_no_pending_order(self, db_session):
        repo = ForwardRepository(db_session)
        launch = _make_launch(repo)
        db_session.commit()

        db_session.add(_open_position(launch.id))
        db_session.commit()

        state = repo.recover(launch)
        assert state.open_position is not None


class TestArchitectureGuard:
    """The forward repository must never place, modify, or cancel exchange orders."""

    def test_repository_has_no_exchange_order_methods(self):
        forbidden = {"place_order", "cancel_order", "send_order", "submit_order"}
        methods = {name for name in dir(ForwardRepository) if not name.startswith("_")}
        assert forbidden.isdisjoint(methods)
