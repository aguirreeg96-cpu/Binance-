"""BacktestService — orchestrates data retrieval and backtest execution.

Reads historical candles from the local SQLite database via CandleRepository.
No Binance API calls, no real orders. PAPER/TEST only.
"""

import logging

from sqlalchemy.orm import Session

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.exceptions import BacktestInsufficientDataError
from app.backtesting.schemas import BacktestResult
from app.indicators.schemas import IndicatorConfig
from app.repositories.candle_repository import CandleRepository
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine

logger = logging.getLogger(__name__)


class BacktestService:
    """Coordinate data loading and backtesting engine execution.

    Reads historical candles from the local SQLite database.
    No network access. No real orders. PAPER/TEST mode only.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def run(
        self,
        config: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> BacktestResult:
        """Run a backtest and return the result.

        Fetches candles from the DB (with warmup prefix), runs the engine,
        and returns a BacktestResult.  The result is not persisted.
        """
        ind_config = indicator_config or IndicatorConfig()
        strat_config = strategy_config or StrategyEngineConfig()

        warmup_count = ind_config.warmup_candles
        repo = CandleRepository(self.db)

        all_candles, warmup_len = repo.query_for_indicators(
            symbol=config.symbol,
            interval=config.interval,
            warmup_count=warmup_count,
            start_ms=config.start_ms,
            end_ms=config.end_ms,
        )

        eval_candles = all_candles[warmup_len:]
        if not eval_candles:
            raise BacktestInsufficientDataError(
                f"No candles found for {config.symbol} {config.interval} "
                f"in range [{config.start_ms}, {config.end_ms}). "
                "Download historical data first using the CLI."
            )

        logger.info(
            "Backtest %s %s: %d eval candles + %d warmup candles",
            config.symbol,
            config.interval,
            len(eval_candles),
            warmup_len,
        )

        engine = BacktestEngine(
            config=config,
            strategy_engine=StrategyEngine(strat_config),
            indicator_config=ind_config,
        )
        return engine.run(all_candles, warmup_len)
