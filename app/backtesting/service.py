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
    from app.backtesting.breakout_family import BreakoutFamilyReport
    from app.backtesting.entry_comparison import EntryComparisonReport, EntryMultiPeriodReport
    from app.backtesting.frozen_oos_2025 import FrozenOos2025Report
    from app.backtesting.normalized_comparison import MultiPeriodReport, NormalizedComparisonReport
    from app.backtesting.risk_exit_config import RiskExitConfig
    from app.backtesting.timeframe_cost_audit import TimeframeCostAuditReport
    from app.backtesting.timeframe_cost_comparison import TimeframeCostReport
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

    def run_entry_comparison(
        self,
        config: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
        period_label: str = "",
    ) -> "EntryComparisonReport":
        """Run all 10 entry-variant × exit-config combinations for one period.

        Fetches candles from DB, runs each combo twice (with/without costs).
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.entry_comparison import (
            run_entry_comparison as _run_ec,
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
            "Entry comparison %s %s [%s]: %d eval + %d warmup candles",
            config.symbol,
            config.interval,
            label,
            len(eval_candles),
            warmup_len,
        )

        return _run_ec(
            all_candles=all_candles,
            warmup_len=warmup_len,
            config=config,
            indicator_config=ind_config,
            strategy_config=strat_config,
            period_label=period_label,
        )

    def run_entry_multi_period(
        self,
        config_2023: BacktestConfig,
        config_2024: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> "EntryMultiPeriodReport":
        """Run entry variant comparison for 2023 and 2024 with capital compounding.

        2023 final equity is carried forward as 2024 initial capital per combo.
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.entry_comparison import (
            run_entry_multi_period as _run_emp,
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
            "Entry multi-period %s %s: 2023=%d candles, 2024=%d candles",
            config_2023.symbol,
            config_2023.interval,
            len(eval_23),
            len(eval_24),
        )

        return _run_emp(
            candles_2023=all_candles_23,
            warmup_2023=warmup_23,
            config_2023=config_2023,
            candles_2024=all_candles_24,
            warmup_2024=warmup_24,
            config_2024=config_2024,
            indicator_config=ind_config,
            strategy_config=strat_config,
        )

    def run_timeframe_cost_comparison(
        self,
        config_2023: BacktestConfig,
        config_2024: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> "TimeframeCostReport":
        """Run frozen ENTRY_V3_ALIGNED_TREND + V2_STOP_ONLY across 3 timeframes × 4 cost scenarios.

        Loads 15m candles with 4× warmup count so that all timeframes have adequate
        warmup candles after aggregation (15m→30m→1h).
        2024 initial capital is compounded from 2023 final equity per (timeframe, scenario).
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.timeframe_cost_comparison import (
            run_timeframe_cost_comparison as _run_tcc,
        )

        ind_config = indicator_config or IndicatorConfig()
        strat_config = strategy_config or StrategyEngineConfig()
        # 4× warmup so that 1h aggregated series has ≥ warmup_candles 1h warmup candles.
        warmup_count = ind_config.warmup_candles * 4
        repo = CandleRepository(self.db)

        all_candles_23, warmup_len_23 = repo.query_for_indicators(
            symbol=config_2023.symbol,
            interval="15m",
            warmup_count=warmup_count,
            start_ms=config_2023.start_ms,
            end_ms=config_2023.end_ms,
        )
        if not any(c.open_time >= config_2023.start_ms for c in all_candles_23):
            raise BacktestInsufficientDataError(
                f"No 2023 15m candles for {config_2023.symbol}. "
                "Download historical data first using the CLI."
            )

        all_candles_24, warmup_len_24 = repo.query_for_indicators(
            symbol=config_2024.symbol,
            interval="15m",
            warmup_count=warmup_count,
            start_ms=config_2024.start_ms,
            end_ms=config_2024.end_ms,
        )
        if not any(c.open_time >= config_2024.start_ms for c in all_candles_24):
            raise BacktestInsufficientDataError(
                f"No 2024 15m candles for {config_2024.symbol}. "
                "Download historical data first using the CLI."
            )

        logger.info(
            "Timeframe cost comparison %s: "
            "2023=%d 15m candles (%d warmup + %d eval), "
            "2024=%d 15m candles (%d warmup + %d eval)",
            config_2023.symbol,
            len(all_candles_23),
            warmup_len_23,
            len(all_candles_23) - warmup_len_23,
            len(all_candles_24),
            warmup_len_24,
            len(all_candles_24) - warmup_len_24,
        )

        return _run_tcc(
            symbol=config_2023.symbol,
            initial_capital=config_2023.initial_capital,
            candles_15m_2023=all_candles_23,
            candles_15m_2024=all_candles_24,
            start_ms_2023=config_2023.start_ms,
            start_ms_2024=config_2024.start_ms,
            end_ms_2023=config_2023.end_ms,
            end_ms_2024=config_2024.end_ms,
            force_close_at_end=config_2023.force_close_at_end,
            indicator_config=ind_config,
            strategy_config=strat_config,
        )

    def run_timeframe_cost_audit(
        self,
        config_2023: BacktestConfig,
        config_2024: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> "tuple[TimeframeCostReport, TimeframeCostAuditReport]":
        """Run Stage 5.2C.1 audit: cost comparison report + data integrity audit.

        Loads 15m candles with 4× warmup count so all timeframes have adequate warmup
        after aggregation.  Runs all backtests once; builds both reports from the same
        raw results.
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.timeframe_cost_comparison import (
            run_timeframe_cost_audit as _run_tca,
        )

        ind_config = indicator_config or IndicatorConfig()
        strat_config = strategy_config or StrategyEngineConfig()
        warmup_count = ind_config.warmup_candles * 4
        repo = CandleRepository(self.db)

        all_candles_23, warmup_len_23 = repo.query_for_indicators(
            symbol=config_2023.symbol,
            interval="15m",
            warmup_count=warmup_count,
            start_ms=config_2023.start_ms,
            end_ms=config_2023.end_ms,
        )
        if not any(c.open_time >= config_2023.start_ms for c in all_candles_23):
            raise BacktestInsufficientDataError(
                f"No 2023 15m candles for {config_2023.symbol}. "
                "Download historical data first using the CLI."
            )

        all_candles_24, warmup_len_24 = repo.query_for_indicators(
            symbol=config_2024.symbol,
            interval="15m",
            warmup_count=warmup_count,
            start_ms=config_2024.start_ms,
            end_ms=config_2024.end_ms,
        )
        if not any(c.open_time >= config_2024.start_ms for c in all_candles_24):
            raise BacktestInsufficientDataError(
                f"No 2024 15m candles for {config_2024.symbol}. "
                "Download historical data first using the CLI."
            )

        logger.info(
            "Timeframe cost audit %s: "
            "2023=%d 15m candles (%d warmup + %d eval), "
            "2024=%d 15m candles (%d warmup + %d eval)",
            config_2023.symbol,
            len(all_candles_23),
            warmup_len_23,
            len(all_candles_23) - warmup_len_23,
            len(all_candles_24),
            warmup_len_24,
            len(all_candles_24) - warmup_len_24,
        )

        return _run_tca(
            symbol=config_2023.symbol,
            initial_capital=config_2023.initial_capital,
            candles_15m_2023=all_candles_23,
            candles_15m_2024=all_candles_24,
            start_ms_2023=config_2023.start_ms,
            start_ms_2024=config_2024.start_ms,
            end_ms_2023=config_2023.end_ms,
            end_ms_2024=config_2024.end_ms,
            force_close_at_end=config_2023.force_close_at_end,
            indicator_config=ind_config,
            strategy_config=strat_config,
        )

    def run_frozen_oos_2025(
        self,
        config_2025: BacktestConfig,
        indicator_config: IndicatorConfig | None = None,
        strategy_config: StrategyEngineConfig | None = None,
    ) -> "FrozenOos2025Report":
        """Run Stage 5.3 frozen OOS evaluation on 2025 data.

        Loads 15m candles with 2× warmup count (30m = 2 × 15m candles per warmup slot).
        Aggregates to 30m and evaluates 3 cost scenarios (NO_COSTS, BASE_COSTS, CONSERVATIVE).
        Strategy filter and risk config are frozen — passed indicator/strategy configs affect
        indicator calculation and base strategy only, not filter/risk parameters.
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from app.backtesting.frozen_oos_2025 import run_frozen_oos_2025 as _run_oos

        ind_config = indicator_config or IndicatorConfig()
        strat_config = strategy_config or StrategyEngineConfig()
        # 2× warmup: each 30m warmup candle requires 2 source 15m candles.
        warmup_count = ind_config.warmup_candles * 2
        repo = CandleRepository(self.db)

        all_candles_2025, warmup_len_2025 = repo.query_for_indicators(
            symbol=config_2025.symbol,
            interval="15m",
            warmup_count=warmup_count,
            start_ms=config_2025.start_ms,
            end_ms=config_2025.end_ms,
        )
        if not any(c.open_time >= config_2025.start_ms for c in all_candles_2025):
            raise BacktestInsufficientDataError(
                f"No 2025 15m candles for {config_2025.symbol}. "
                "Download 2025 historical data first using the CLI."
            )

        logger.info(
            "Frozen OOS 2025 %s: %d 15m candles (%d warmup + %d eval)",
            config_2025.symbol,
            len(all_candles_2025),
            warmup_len_2025,
            len(all_candles_2025) - warmup_len_2025,
        )

        return _run_oos(
            symbol=config_2025.symbol,
            initial_capital=config_2025.initial_capital,
            candles_15m_2025=all_candles_2025,
            start_ms_2025=config_2025.start_ms,
            end_ms_2025=config_2025.end_ms,
            force_close_at_end=config_2025.force_close_at_end,
            indicator_config=ind_config,
            strategy_config=strat_config,
        )

    def run_breakout_family(
        self,
        symbol: str = "BTCUSDT",
        initial_capital_str: str = "10000",
        force_close_at_end: bool = True,
    ) -> "BreakoutFamilyReport":
        """Run Stage 6.0 Donchian breakout family evaluation (2021–2025).

        Loads 15m candles with 3 500 warmup candles before 2021-01-01 to ensure
        adequate indicator history for all timeframes (including 4h EMA-200).
        Aborts if any (year × timeframe) has zero eval candles.
        PAPER/TEST only — no real orders, no real capital at risk.
        """
        from decimal import Decimal

        from app.backtesting.breakout_family import (
            _YEAR_END_MS,
            _YEAR_START_MS,
            BREAKOUT_YEARS,
        )
        from app.backtesting.breakout_family import (
            run_breakout_family as _run_bf,
        )

        initial_capital = Decimal(initial_capital_str)
        start_ms = _YEAR_START_MS[BREAKOUT_YEARS[0]]  # 2021-01-01
        end_ms = _YEAR_END_MS[BREAKOUT_YEARS[-1]]  # 2026-01-01

        warmup_count = 3_500  # sufficient for EMA-200 on 4h (200 × 16 = 3200 + buffer)
        repo = CandleRepository(self.db)

        all_candles_15m, warmup_len = repo.query_for_indicators(
            symbol=symbol,
            interval="15m",
            warmup_count=warmup_count,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        if not any(c.open_time >= start_ms for c in all_candles_15m):
            raise BacktestInsufficientDataError(
                f"No 2021–2025 15m candles for {symbol}. "
                "Download historical data first using the CLI."
            )

        logger.info(
            "Breakout family %s: %d 15m candles (%d warmup + %d eval)",
            symbol,
            len(all_candles_15m),
            warmup_len,
            len(all_candles_15m) - warmup_len,
        )

        return _run_bf(
            symbol=symbol,
            all_candles_15m=all_candles_15m,
            initial_capital=initial_capital,
            force_close_at_end=force_close_at_end,
        )
