"""Tests for SQLAlchemy models — structure, types, and creation."""

from datetime import date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import inspect

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


class TestSchemaCreation:
    def test_all_tables_exist(self, db_session):
        engine = db_session.get_bind()
        tables = set(inspect(engine).get_table_names())
        expected = {
            "candles", "signals", "paper_accounts", "positions",
            "orders", "trades", "strategy_configs",
            "daily_risk_states", "system_events",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"


class TestCandleModel:
    def test_create_candle(self, db_session):
        candle = Candle(
            symbol="BTCUSDT",
            interval="1h",
            open_time=1700000000000,
            open=Decimal("35000.00"),
            high=Decimal("35500.00"),
            low=Decimal("34800.00"),
            close=Decimal("35200.00"),
            volume=Decimal("100.5"),
            close_time=1700003599999,
            quote_asset_volume=Decimal("3538100.00"),
            trades=1500,
            taker_buy_base_volume=Decimal("50.25"),
            taker_buy_quote_volume=Decimal("1769050.00"),
            is_closed=True,
        )
        db_session.add(candle)
        db_session.commit()
        db_session.refresh(candle)

        assert candle.id is not None
        assert candle.symbol == "BTCUSDT"
        assert candle.is_closed is True

    def test_all_numeric_columns_return_decimal(self, db_session):
        candle = Candle(
            symbol="ETHUSDT", interval="15m",
            open_time=1700010000000,
            open=Decimal("2000.00"), high=Decimal("2050.00"),
            low=Decimal("1980.00"), close=Decimal("2020.00"),
            volume=Decimal("500.0"), close_time=1700010899999,
            quote_asset_volume=Decimal("1000000.00"),
            trades=800,
            taker_buy_base_volume=Decimal("250.0"),
            taker_buy_quote_volume=Decimal("500000.00"),
            is_closed=True,
        )
        db_session.add(candle)
        db_session.commit()
        db_session.expire(candle)  # force reload from DB

        row = db_session.query(Candle).filter_by(
            symbol="ETHUSDT", interval="15m"
        ).first()

        for field_name in (
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
        ):
            val = getattr(row, field_name)
            assert isinstance(val, Decimal), (
                f"Column {field_name!r} should return Decimal after DB roundtrip, "
                f"got {type(val).__name__}: {val}"
            )


class TestSignalModel:
    def test_create_signal(self, db_session):
        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="BTCUSDT",
            interval="1h",
            price=Decimal("35200.00"),
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
            balance=Decimal("10000.00"),
            equity=Decimal("10000.00"),
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
            entry_price=Decimal("35200.00"),
            quantity=Decimal("0.00142"),
            stop_loss=Decimal("34800.00"),
            take_profit=Decimal("36000.00"),
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
            quantity=Decimal("0.00142"),
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
            entry_price=Decimal("35200.00"),
            exit_price=Decimal("36000.00"),
            quantity=Decimal("0.00142"),
            side=OrderSide.BUY,
            gross_pnl=Decimal("1.136"),
            commission=Decimal("0.07"),
            net_pnl=Decimal("1.066"),
            planned_stop_loss=Decimal("34800.00"),
            planned_take_profit=Decimal("36000.00"),
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
            starting_equity=Decimal("10000.00"),
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
