"""Shared fixtures for backtesting tests."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.backtesting.config import BacktestConfig
from app.database import Base
from app.indicators.schemas import IndicatorConfig
from app.models import candle  # noqa: F401 — register model
from app.models.candle import Candle
from app.strategy.engine import StrategyEngine
from app.strategy.schemas import StrategyAction, StrategyDecision

# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# Candle factory
# ---------------------------------------------------------------------------


def make_candle(
    open_time: int,
    open_p: str = "100",
    high_p: str = "110",
    low_p: str = "90",
    close_p: str = "105",
    volume: str = "1000",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    trades: int = 100,
    is_closed: bool = True,
) -> Candle:
    """Create a Candle ORM object with sensible defaults."""
    close_time = open_time + 3_599_999
    return Candle(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        open=Decimal(open_p),
        high=Decimal(high_p),
        low=Decimal(low_p),
        close=Decimal(close_p),
        volume=Decimal(volume),
        close_time=close_time,
        quote_asset_volume=Decimal(volume) * Decimal(close_p),
        trades=trades,
        taker_buy_base_volume=Decimal(volume) / Decimal("2"),
        taker_buy_quote_volume=Decimal(volume) / Decimal("2") * Decimal(close_p),
        is_closed=is_closed,
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )


def make_candles(
    n: int,
    base_open_time: int = 1_000_000,
    interval_ms: int = 3_600_000,
    price: str = "100",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> list[Candle]:
    """Create a sequence of n candles with constant price."""
    return [
        make_candle(
            open_time=base_open_time + i * interval_ms,
            open_p=price,
            high_p=str(Decimal(price) * Decimal("1.01")),
            low_p=str(Decimal(price) * Decimal("0.99")),
            close_p=price,
            symbol=symbol,
            interval=interval,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Deterministic strategy stubs
# ---------------------------------------------------------------------------


class AlwaysBuyEngine(StrategyEngine):
    """Emits BUY on every warmup-complete candle (no position), WAIT otherwise."""

    def evaluate(self, result, position=None):  # type: ignore[override]
        from app.strategy.schemas import PositionContext, StrategyAction

        pos = position if position is not None else PositionContext()
        if result.warmup_complete and not pos.has_open_long_position:
            return _make_decision(result, StrategyAction.BUY)
        return _make_decision(result, StrategyAction.WAIT)


class AlwaysSellEngine(StrategyEngine):
    """Emits SELL on every warmup-complete candle (if position open), WAIT otherwise."""

    def evaluate(self, result, position=None):  # type: ignore[override]
        from app.strategy.schemas import PositionContext, StrategyAction

        pos = position if position is not None else PositionContext()
        if result.warmup_complete and pos.has_open_long_position:
            return _make_decision(result, StrategyAction.SELL)
        return _make_decision(result, StrategyAction.WAIT)


class AlwaysWaitEngine(StrategyEngine):
    """Never generates a BUY or SELL signal."""

    def evaluate(self, result, position=None):  # type: ignore[override]
        return _make_decision(result, StrategyAction.WAIT)


class BuyThenSellEngine(StrategyEngine):
    """Buys on first warmup-complete candle (no position), sells immediately on the next."""

    def evaluate(self, result, position=None):  # type: ignore[override]
        from app.strategy.schemas import PositionContext, StrategyAction

        pos = position if position is not None else PositionContext()
        if not result.warmup_complete:
            return _make_decision(result, StrategyAction.WAIT)
        if not pos.has_open_long_position:
            return _make_decision(result, StrategyAction.BUY)
        return _make_decision(result, StrategyAction.SELL)


def _make_decision(result, action: StrategyAction) -> StrategyDecision:
    from app.indicators.schemas import CrossSignal
    from app.strategy.schemas import IndicatorsSnapshot

    return StrategyDecision(
        action=action,
        symbol=result.symbol,
        interval=result.interval,
        candle_open_time=result.open_time,
        candle_close_time=result.close_time,
        close_price=result.close,
        strategy_name="stub",
        strategy_version="0.0.0",
        reasons=[],
        failed_conditions=[],
        indicators_snapshot=IndicatorsSnapshot(
            close=result.close,
            ema_short=None,
            ema_medium=None,
            ema_long=None,
            rsi=None,
            atr=None,
            volume_sma=None,
            volume_ratio=None,
            crossover=CrossSignal.NONE,
        ),
        warmup_complete=result.warmup_complete,
        has_open_position=False,
        generated_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_config() -> BacktestConfig:
    return BacktestConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_ms=1_000_000,
        end_ms=1_000_000 + 100 * 3_600_000,
        initial_capital=Decimal("10000"),
        fee_percentage=Decimal("0"),
        slippage_percentage=Decimal("0"),
        force_close_at_end=True,
    )


@pytest.fixture()
def minimal_ind_config() -> IndicatorConfig:
    """Small period indicator config to minimise warmup."""
    return IndicatorConfig(
        sma_short_period=2,
        sma_long_period=3,
        ema_short_period=2,
        ema_medium_period=3,
        ema_long_period=4,
        rsi_period=2,
        atr_period=2,
        volume_period=2,
    )
