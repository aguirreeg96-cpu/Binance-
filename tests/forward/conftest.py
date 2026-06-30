"""Shared fixtures for Stage 6.1 forward paper-trading tests.

`breakout_scenario` builds a single deterministic 15m candle series — 200
flat warm-up candles, a ramp into a Donchian/EMA breakout, a hold period,
then a sharp drop that triggers the trailing stop / channel exit. Most
engine tests launch partway through this series so the flat warm-up
candles are pre-launch history and the ramp/breakout/drop candles are the
genuinely "forward" candles evaluated by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.forward.manifest import FORWARD_SOURCE_INTERVAL, FORWARD_SYMBOL
from app.models.candle import Candle

FOUR_HOUR_MS = 14_400_000
FIFTEEN_MIN_MS = 900_000
SUB_CANDLES_PER_4H = FOUR_HOUR_MS // FIFTEEN_MIN_MS

START_OPEN_MS = 1_700_000_000_000 - (1_700_000_000_000 % FOUR_HOUR_MS)

N_WARMUP_FLAT = 200
N_RAMP = 30
N_HOLD = 15
N_DROP = 5


def build_breakout_price_series(
    n_warmup: int = N_WARMUP_FLAT,
    n_ramp: int = N_RAMP,
    n_hold: int = N_HOLD,
    n_drop: int = N_DROP,
) -> list[Decimal]:
    """One 4h-candle close price per element: flat -> ramp -> breakout -> hold -> drop."""
    prices: list[Decimal] = []
    p = Decimal("100")
    for _ in range(n_warmup):
        prices.append(p)
    for i in range(n_ramp):
        p = Decimal("100") + Decimal("0.5") * Decimal(i + 1)
        prices.append(p)
    breakout_price = prices[-1] + Decimal("10")
    prices.append(breakout_price)
    for i in range(n_hold):
        p = breakout_price + Decimal("0.3") * Decimal(i + 1)
        prices.append(p)
    for i in range(n_drop):
        p = prices[-1] - Decimal("5") * Decimal(i + 1)
        prices.append(p)
    return prices


def four_hour_open_times(n: int, start_open_ms: int = START_OPEN_MS) -> list[int]:
    return [start_open_ms + i * FOUR_HOUR_MS for i in range(n)]


def make_15m_candles(
    prices: list[Decimal],
    open_times_4h: list[int],
    symbol: str = FORWARD_SYMBOL,
) -> list[Candle]:
    """Expand one close-per-4h-bucket price series into 16 contiguous 15m candles each."""
    sub_candles: list[Candle] = []
    for idx, close in enumerate(prices):
        bucket_open = open_times_4h[idx]
        open_price = prices[idx - 1] if idx > 0 else close
        high = max(open_price, close) + Decimal("1")
        low = min(open_price, close) - Decimal("1")
        for j in range(SUB_CANDLES_PER_4H):
            sub_open = bucket_open + j * FIFTEEN_MIN_MS
            sub_candles.append(
                Candle(
                    symbol=symbol,
                    interval=FORWARD_SOURCE_INTERVAL,
                    open_time=sub_open,
                    open=open_price if j == 0 else close,
                    high=high,
                    low=low,
                    close=close,
                    volume=Decimal("1"),
                    close_time=sub_open + FIFTEEN_MIN_MS - 1,
                    quote_asset_volume=Decimal("100"),
                    trades=10,
                    taker_buy_base_volume=Decimal("0.5"),
                    taker_buy_quote_volume=Decimal("50"),
                    is_closed=True,
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )
    return sub_candles


def ms_to_naive_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).replace(tzinfo=None)


@dataclass
class BreakoutScenario:
    prices: list[Decimal]
    open_times_4h: list[int]
    sub_candles: list[Candle]

    def close_time_4h(self, index: int) -> datetime:
        return ms_to_naive_dt(self.open_times_4h[index] + FOUR_HOUR_MS - 1)

    def open_time_4h(self, index: int) -> datetime:
        return ms_to_naive_dt(self.open_times_4h[index])


@pytest.fixture
def breakout_scenario() -> BreakoutScenario:
    prices = build_breakout_price_series()
    open_times = four_hour_open_times(len(prices))
    sub_candles = make_15m_candles(prices, open_times)
    return BreakoutScenario(prices=prices, open_times_4h=open_times, sub_candles=sub_candles)


@pytest.fixture
def seeded_session(db_session, breakout_scenario: BreakoutScenario):
    """db_session (from tests/conftest.py) pre-loaded with the breakout scenario's 15m candles."""
    db_session.add_all(breakout_scenario.sub_candles)
    db_session.commit()
    return db_session
