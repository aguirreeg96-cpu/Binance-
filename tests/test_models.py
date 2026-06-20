"""Tests for SQLAlchemy models — structure and creation."""
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.candle import Candle
from app.models.daily_risk_state import DailyRiskState
from app.models.order import Order
from app.models.paper_account import PaperAccount
from app.models.position import Position
from app.models.signal import Signal
from app.models.strategy_config import StrategyConfig
from app.models.system_event import SystemEvent
from app.models.trade import Trade
from app.schemas.common import (
    EventLevel,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionStatus,
    SignalType,
    TradingMode,
)


@pytest.fixture(scope="module")
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


class TestSchemaCreation:
    """Verify all tables are created correctly."""

    def test_all_tables_exist(self, db_session):
        engine = db_session.get_bind()
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        expected = {
            "candles",
            "signals",
            "paper_accounts",
            "positions",
            "orders",
            "trades",
            "strategy_configs",
            "daily_risk_states",
            "system_events",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"


class TestCandleModel:
    def test_create_candle(self, db_session):
        candle = Candle(
            symbol="BTCUSDT",
            interval="1h",
            open_time=1700000000000,
            open="35000.00",
            high="35500.00",
            low="34800.00",
            close="35200.00",
            volume="100.5",
            close_time=1700003600000,
            quote_volume="3538100.00",
            trades=1500,
            is_closed=True,
        )
        db_session.add(candle)
        db_session.commit()
        db_session.refresh(candle)
        assert candle.id is not None
        assert candle.symbol == "BTCUSDT"
        assert candle.is_closed is True


class TestSignalModel:
    def test_create_signal(self, db_session):
        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="BTCUSDT",
            interval="1h",
            price="35200.00",
            signal_type=SignalType.BUY,
            indicators={"ema20": 35100, "rsi": 55},
            reasons=["EMA crossover above EMA50", "RSI in range"],
            strategy_version="v0.1.0",
            config_snapshot={"ema_fast": 20, "ema_slow": 50},
        )
        db_session.add(signal)
        db_session.commit()
        db_session.refresh(signal)
        assert signal.id is not None
        assert signal.signal_type == SignalType.BUY
        assert len(signal.reasons) == 2


class TestPaperAccountModel:
    def test_create_paper_account(self, db_session):
        account = PaperAccount(
            balance="10000.00",
            equity="10000.00",
        )
        db_session.add(account)
        db_session.commit()
        db_session.refresh(account)
        assert account.id is not None
        assert account.currency == "USDT"


class TestPositionModel:
    def test_create_position(self, db_session):
        position = Position(
            symbol="BTCUSDT",
            status=PositionStatus.OPEN,
            entry_price="35200.00",
            quantity="0.00142",
            stop_loss="34800.00",
            take_profit="36000.00",
            opened_at=datetime.utcnow(),
        )
        db_session.add(position)
        db_session.commit()
        db_session.refresh(position)
        assert position.id is not None
        assert position.status == PositionStatus.OPEN
        assert position.trailing_stop_enabled is False


class TestOrderModel:
    def test_create_order(self, db_session):
        order = Order(
            symbol="BTCUSDT",
            client_order_id="test-order-001",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING_APPROVAL,
            quantity="0.00142",
            trading_mode=TradingMode.PAPER,
        )
        db_session.add(order)
        db_session.commit()
        db_session.refresh(order)
        assert order.id is not None
        assert order.status == OrderStatus.PENDING_APPROVAL


class TestTradeModel:
    def test_create_trade(self, db_session):
        trade = Trade(
            symbol="BTCUSDT",
            entry_price="35200.00",
            exit_price="36000.00",
            quantity="0.00142",
            side=OrderSide.BUY,
            gross_pnl="1.136",
            commission="0.07",
            net_pnl="1.066",
            planned_stop_loss="34800.00",
            planned_take_profit="36000.00",
            exit_reason="take_profit",
            trading_mode=TradingMode.PAPER,
            opened_at=datetime.utcnow(),
            closed_at=datetime.utcnow(),
        )
        db_session.add(trade)
        db_session.commit()
        db_session.refresh(trade)
        assert trade.id is not None
        assert trade.exit_reason == "take_profit"


class TestStrategyConfigModel:
    def test_create_strategy_config(self, db_session):
        config = StrategyConfig(version="v0.1.0")
        db_session.add(config)
        db_session.commit()
        db_session.refresh(config)
        assert config.id is not None
        assert config.ema_fast == 20
        assert config.ema_slow == 50
        assert config.is_active is True


class TestDailyRiskStateModel:
    def test_create_daily_risk_state(self, db_session):
        state = DailyRiskState(
            date=date.today(),
            starting_equity="10000.00",
        )
        db_session.add(state)
        db_session.commit()
        db_session.refresh(state)
        assert state.id is not None
        assert state.kill_switch_active is False
        assert state.trading_paused is False
        assert state.trades_count == 0


class TestSystemEventModel:
    def test_create_system_event(self, db_session):
        event = SystemEvent(
            level=EventLevel.INFO,
            source="test_suite",
            message="Application started in PAPER mode.",
            trading_mode=TradingMode.PAPER,
        )
        db_session.add(event)
        db_session.commit()
        db_session.refresh(event)
        assert event.id is not None
        assert event.level == EventLevel.INFO


class TestEnums:
    def test_trading_mode_values(self):
        assert set(TradingMode) == {
            TradingMode.SIGNAL,
            TradingMode.PAPER,
            TradingMode.DEMO,
        }
        assert "live" not in [m.value for m in TradingMode]

    def test_signal_type_values(self):
        assert SignalType.BUY == "BUY"
        assert SignalType.SELL == "SELL"
        assert SignalType.WAIT == "WAIT"
