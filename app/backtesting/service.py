"""BacktestService — orchestrates data retrieval and backtest execution.

Reads historical candles from the local SQLite database via CandleRepository.
No Binance API calls, no real orders. PAPER/TEST only.
"""

import logging
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.exceptions import BacktestInsufficientDataError
from app.backtesting.schemas import BacktestResult
from app.indicators.schemas import IndicatorConfig
from app.models.candle import Candle
from app.repositories.candle_repository import CandleRepository
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine

if TYPE_CHECKING:
    from app.backtesting.risk_exit_config import RiskExitConfig
    from app.backtesting.variants import ComparisonReport

logger = logging.getLogger(__name__)


class BacktestService:
    """Coordinate data loading and backtesting engine execution.

    Reads historical candles from the local SQLite database.
    No network access. No real orders. PAPER/TEST mode only.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def run_with_context(
        self,
        config: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> tuple[BacktestResult, list[Candle], int, StrategyEngine, IndicatorConfig]:
        """Run a backtest and return both result and diagnostic context.

        Returns (result, all_candles, warmup_len, strategy_engine, indicator_config).
        Pass these directly to compute_diagnostics() for extended analysis.
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

        strategy_engine = StrategyEngine(strat_config)
        engine = BacktestEngine(
            config=config,
            strategy_engine=strategy_engine,
            indicator_config=ind_config,
        )
        result = engine.run(all_candles, warmup_len)
        return result, all_candles, warmup_len, strategy_engine, ind_config

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
        result, _, _, _, _ = self.run_with_context(config, indicator_config, strategy_config)
        return result

    def run_v2(
        self,
        config: BacktestConfig,
        risk_exit_config: "RiskExitConfig | None" = None,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> BacktestResult:
        """Run a V2 backtest with risk-based exits and return the result.

        Uses default RiskExitConfig when risk_exit_config is None.
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.risk_exit_config import RiskExitConfig as _RiskExitConfig
        from app.backtesting.v2_engine import V2BacktestEngine

        ind_config = indicator_config or IndicatorConfig()
        strat_config = strategy_config or StrategyEngineConfig()
        risk_cfg = risk_exit_config if risk_exit_config is not None else _RiskExitConfig()

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
            "V2 backtest %s %s: %d eval candles + %d warmup candles",
            config.symbol,
            config.interval,
            len(eval_candles),
            warmup_len,
        )

        strategy_engine = StrategyEngine(strat_config)
        engine = V2BacktestEngine(
            config=config,
            risk_exit_config=risk_cfg,
            strategy_engine=strategy_engine,
            indicator_config=ind_config,
        )
        return engine.run(all_candles, warmup_len)

    def run_variants(
        self,
        config: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> "ComparisonReport":
        """Run all four controlled strategy variants and return a ComparisonReport.

        Fetches candles once from the DB and passes the same data to all variants.
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.variants import run_variants

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
            "Variant comparison %s %s: %d eval candles + %d warmup candles",
            config.symbol,
            config.interval,
            len(eval_candles),
            warmup_len,
        )

        return run_variants(
            all_candles=all_candles,
            warmup_len=warmup_len,
            config=config,
            indicator_config=ind_config,
            strategy_config=strat_config,
        )
