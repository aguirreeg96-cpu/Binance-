"""StrategyService — bridges DB data and the pure strategy engine.

Responsibilities:
  - Query closed candles from SQLite.
  - Compute indicators via IndicatorCalculator.
  - Run StrategyEngine (pure, no DB access).
  - Return StrategyDecision(s) to the API layer.

Does NOT:
  - Call Binance.
  - Place orders.
  - Modify positions or balances.
  - Persist signals.
"""

import logging

from sqlalchemy.orm import Session

from app.indicators.calculator import IndicatorCalculator
from app.indicators.schemas import IndicatorConfig
from app.repositories.candle_repository import CandleRepository
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine
from app.strategy.exceptions import InsufficientDataForStrategyError
from app.strategy.schemas import PositionContext, StrategyAction, StrategyDecision

logger = logging.getLogger(__name__)


class StrategyService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_latest_decision(
        self,
        symbol: str,
        interval: str,
        position: PositionContext,
        ind_config: IndicatorConfig,
        strat_config: StrategyEngineConfig,
    ) -> StrategyDecision:
        """Compute indicators for ALL available candles and evaluate the latest one.

        Using all candles (not just the warmup window) maximises indicator accuracy.
        """
        repo = CandleRepository(self._session)
        candles, _ = repo.query_for_indicators(
            symbol=symbol.upper(),
            interval=interval,
            warmup_count=0,
        )
        if not candles:
            raise InsufficientDataForStrategyError(symbol, interval)

        calc = IndicatorCalculator(ind_config)
        results = calc.calculate(candles)

        engine = StrategyEngine(strat_config)
        decision = engine.evaluate(results[-1], position)
        logger.debug(
            "Latest decision for %s %s: %s (reasons=%s)",
            symbol,
            interval,
            decision.action,
            [r.value for r in decision.reasons],
        )
        return decision

    def get_history(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None,
        end_ms: int | None,
        limit: int | None,
        initial_position: PositionContext,
        include_wait: bool,
        ind_config: IndicatorConfig,
        strat_config: StrategyEngineConfig,
    ) -> list[StrategyDecision]:
        """Compute indicator history and evaluate each candle sequentially.

        Warmup candles are used internally for accurate indicators but the
        decisions returned start from the first non-warmup candle.
        Position context is threaded through sequentially (no look-ahead).
        """
        repo = CandleRepository(self._session)
        candles, warmup_len = repo.query_for_indicators(
            symbol=symbol.upper(),
            interval=interval,
            warmup_count=ind_config.warmup_candles,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )
        if not candles:
            raise InsufficientDataForStrategyError(symbol, interval)

        calc = IndicatorCalculator(ind_config)
        all_results = calc.calculate(candles)
        results = all_results[warmup_len:]

        engine = StrategyEngine(strat_config)
        decisions = engine.evaluate_series(results, initial_position)

        if not include_wait:
            decisions = [d for d in decisions if d.action != StrategyAction.WAIT]

        return decisions
