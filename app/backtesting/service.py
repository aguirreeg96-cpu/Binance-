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
    from app.backtesting.normalized_comparison import MultiPeriodReport, NormalizedComparisonReport
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

    def run_normalized_comparison(
        self,
        config: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
        period_label: str = "",
    ) -> "NormalizedComparisonReport":
        """Run all 12 variant×allocation combinations for one period.

        Fetches candles from DB, runs V1_BASELINE/V2_STOP_ONLY/V2_STOP_TP/
        V2_STOP_TP_TIME each at 25 %/50 %/100 % allocation (24 engine runs total).
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.normalized_comparison import (
            run_normalized_comparison as _run_nc,
        )

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

        label = period_label or "?"
        logger.info(
            "Normalized comparison %s %s [%s]: %d eval + %d warmup candles",
            config.symbol,
            config.interval,
            label,
            len(eval_candles),
            warmup_len,
        )

        return _run_nc(
            all_candles=all_candles,
            warmup_len=warmup_len,
            config=config,
            indicator_config=ind_config,
            strategy_config=strat_config,
            period_label=period_label,
        )

    def run_multi_period_comparison(
        self,
        config_2023: BacktestConfig,
        config_2024: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> "MultiPeriodReport":
        """Run normalized comparison for 2023 and 2024 and return a MultiPeriodReport.

        Both years use identical variant configurations and parameters.
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.normalized_comparison import (
            run_multi_period_comparison as _run_mp,
        )

        ind_config = indicator_config or IndicatorConfig()
        strat_config = strategy_config or StrategyEngineConfig()
        warmup_count = ind_config.warmup_candles
        repo = CandleRepository(self.db)

        all_candles_23, warmup_23 = repo.query_for_indicators(
            symbol=config_2023.symbol,
            interval=config_2023.interval,
            warmup_count=warmup_count,
            start_ms=config_2023.start_ms,
            end_ms=config_2023.end_ms,
        )
        eval_23 = all_candles_23[warmup_23:]
        if not eval_23:
            raise BacktestInsufficientDataError(
                f"No 2023 candles for {config_2023.symbol} {config_2023.interval}. "
                "Download historical data first using the CLI."
            )

        all_candles_24, warmup_24 = repo.query_for_indicators(
            symbol=config_2024.symbol,
            interval=config_2024.interval,
            warmup_count=warmup_count,
            start_ms=config_2024.start_ms,
            end_ms=config_2024.end_ms,
        )
        eval_24 = all_candles_24[warmup_24:]
        if not eval_24:
            raise BacktestInsufficientDataError(
                f"No 2024 candles for {config_2024.symbol} {config_2024.interval}. "
                "Download historical data first using the CLI."
            )

        logger.info(
            "Multi-period comparison %s %s: 2023=%d candles, 2024=%d candles",
            config_2023.symbol,
            config_2023.interval,
            len(eval_23),
            len(eval_24),
        )

        return _run_mp(
            candles_2023=all_candles_23,
            warmup_2023=warmup_23,
            config_2023=config_2023,
            candles_2024=all_candles_24,
            warmup_2024=warmup_24,
            config_2024=config_2024,
            indicator_config=ind_config,
            strategy_config=strat_config,
        )
