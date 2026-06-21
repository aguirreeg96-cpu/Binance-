"""StrategyEngine — pure, deterministic signal generator.

Contract:
  - Receives IndicatorResult + optional PositionContext.
  - Returns StrategyDecision without any side effects.
  - Never queries Binance, never opens DB sessions, never places orders.
  - Never generates short-sell signals (Spot-only).
  - Inject a `clock` callable for deterministic tests.

Priority order:
  1. Warmup incomplete OR missing critical indicator → WAIT
  2. Position open → evaluate SELL rules → SELL or WAIT
  3. No position → evaluate BUY rules → BUY or WAIT
"""

from collections.abc import Callable
from datetime import UTC, datetime

from app.indicators.schemas import IndicatorResult
from app.strategy.config import StrategyEngineConfig
from app.strategy.reasons import ReasonCode
from app.strategy.rules import (
    evaluate_buy_conditions,
    evaluate_sell_conditions,
    has_missing_critical_indicators,
)
from app.strategy.schemas import (
    IndicatorsSnapshot,
    PositionContext,
    StrategyAction,
    StrategyDecision,
)


def _snapshot(result: IndicatorResult) -> IndicatorsSnapshot:
    return IndicatorsSnapshot(
        close=result.close,
        ema_short=result.ema_short,
        ema_medium=result.ema_medium,
        ema_long=result.ema_long,
        rsi=result.rsi,
        atr=result.atr,
        volume_sma=result.volume_sma,
        volume_ratio=result.volume_ratio,
        crossover=result.ema_short_medium_cross,
    )


class StrategyEngine:
    """Evaluate indicator results and produce trading decisions.

    The engine is stateless between calls: all context is passed explicitly.
    The same inputs always produce the same output (deterministic).
    """

    def __init__(
        self,
        config: StrategyEngineConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config if config is not None else StrategyEngineConfig()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def evaluate(
        self,
        result: IndicatorResult,
        position: PositionContext | None = None,
    ) -> StrategyDecision:
        """Evaluate a single IndicatorResult and return a StrategyDecision.

        position defaults to PositionContext() — no open position.
        """
        pos = position if position is not None else PositionContext()
        cfg = self.config
        now = self._clock()

        def _make(
            action: StrategyAction,
            reasons: list[ReasonCode],
            failed: list[ReasonCode],
        ) -> StrategyDecision:
            return StrategyDecision(
                action=action,
                symbol=result.symbol,
                interval=result.interval,
                candle_open_time=result.open_time,
                candle_close_time=result.close_time,
                close_price=result.close,
                strategy_name=cfg.strategy_name,
                strategy_version=cfg.strategy_version,
                reasons=reasons,
                failed_conditions=failed,
                indicators_snapshot=_snapshot(result),
                warmup_complete=result.warmup_complete,
                has_open_position=pos.has_open_long_position,
                generated_at=now,
            )

        # --- Priority 1: warmup gate ---
        if cfg.require_warmup_complete and not result.warmup_complete:
            return _make(
                StrategyAction.WAIT,
                reasons=[ReasonCode.WARMUP_INCOMPLETE],
                failed=[],
            )

        # --- Priority 1b: missing critical indicators ---
        if has_missing_critical_indicators(result):
            return _make(
                StrategyAction.WAIT,
                reasons=[ReasonCode.MISSING_INDICATOR],
                failed=[],
            )

        # --- Priority 2: open position → try SELL ---
        if pos.has_open_long_position:
            should_sell, sell_met, sell_failed = evaluate_sell_conditions(result, cfg)
            if should_sell:
                return _make(
                    StrategyAction.SELL,
                    reasons=sell_met + [ReasonCode.SELL_CONDITIONS_MET],
                    failed=sell_failed,
                )
            # No exit trigger — hold
            return _make(
                StrategyAction.WAIT,
                reasons=[ReasonCode.POSITION_ALREADY_OPEN],
                failed=sell_failed,
            )

        # --- Priority 3: no position → try BUY ---
        should_buy, buy_met, buy_failed = evaluate_buy_conditions(result, cfg)
        if should_buy:
            return _make(
                StrategyAction.BUY,
                reasons=buy_met + [ReasonCode.BUY_CONDITIONS_MET],
                failed=buy_failed,
            )

        # --- Priority 4: nothing fires → WAIT ---
        return _make(
            StrategyAction.WAIT,
            reasons=buy_met,
            failed=buy_failed,
        )

    def evaluate_series(
        self,
        results: list[IndicatorResult],
        initial_position: PositionContext | None = None,
    ) -> list[StrategyDecision]:
        """Evaluate a series of IndicatorResults sequentially without look-ahead.

        Logical position state transitions:
          BUY  → position opens  (entry_price = result.close)
          SELL → position closes
          WAIT → no state change

        This is a logical simulation for strategy analysis only.
        No orders are placed, no P&L is calculated.
        """
        decisions: list[StrategyDecision] = []
        current_pos = initial_position if initial_position is not None else PositionContext()

        for result in results:
            decision = self.evaluate(result, current_pos)
            decisions.append(decision)

            # Update logical position state
            if decision.action == StrategyAction.BUY and not current_pos.has_open_long_position:
                current_pos = PositionContext(
                    has_open_long_position=True,
                    entry_price=result.close,
                    entry_time=result.open_time,
                )
            elif decision.action == StrategyAction.SELL and current_pos.has_open_long_position:
                current_pos = PositionContext(has_open_long_position=False)

        return decisions
