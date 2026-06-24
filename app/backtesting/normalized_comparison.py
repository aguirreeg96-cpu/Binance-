"""Stage 5.2A.1 — Normalized strategy comparison.

Runs V1_BASELINE, V2_STOP_ONLY, V2_STOP_TP and V2_STOP_TP_TIME each at
25 %, 50 % and 100 % capital allocation to produce a fair 12-row matrix.
V1_BASELINE is emulated via the V2 engine with risk exits disabled so
allocation can be controlled identically across all variants.

No variant is declared optimal.
PAPER/TEST only. No real orders. Past results do NOT predict future performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from app.backtesting.config import BacktestConfig
from app.backtesting.risk_exit_config import RiskExitConfig
from app.backtesting.v2_engine import V2BacktestEngine
from app.backtesting.variants import _median_net_pnl
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine

if TYPE_CHECKING:
    from app.backtesting.schemas import BacktestResult
    from app.indicators.schemas import IndicatorConfig
    from app.models.candle import Candle

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")

ALLOCATIONS: tuple[Decimal, ...] = (Decimal("25"), Decimal("50"), Decimal("100"))

VARIANT_NAMES: tuple[str, ...] = (
    "V1_BASELINE",
    "V2_STOP_ONLY",
    "V2_STOP_TP",
    "V2_STOP_TP_TIME",
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedVariantResult:
    """Performance metrics for one (variant, allocation) combination.

    No result implies profitability or is declared optimal.
    PAPER/TEST only.
    """

    variant_name: str
    allocation_pct: Decimal
    return_pct: Decimal
    return_pct_no_costs: Decimal
    final_equity: Decimal
    max_drawdown_pct: Decimal
    profit_factor: Decimal | None
    win_rate_pct: Decimal | None
    total_fees: Decimal
    slippage_cost: Decimal  # approximated from no-cost delta minus tracked fees
    total_trades: int
    avg_trade_pnl: Decimal | None
    median_trade_pnl: Decimal | None
    exposure_pct: Decimal


@dataclass(frozen=True)
class NormalizedComparisonReport:
    """Full allocation-normalized comparison matrix for one time period.

    matrix has len(VARIANT_NAMES) × len(ALLOCATIONS) rows = 12 total.
    PAPER/TEST only. No variant is declared the best.
    """

    matrix: list[NormalizedVariantResult]
    bah_return_pct: Decimal
    period_label: str


@dataclass(frozen=True)
class YearlyVariantSummary:
    """Cross-year statistics for one (variant, allocation) combination."""

    variant_name: str
    allocation_pct: Decimal
    return_pct_2023: Decimal
    return_pct_2024: Decimal
    combined_return_pct: Decimal
    positive_years: int
    worst_drawdown_pct: Decimal
    profit_factor_stability: Decimal | None  # |pf_2023 - pf_2024|; None if either undefined
    trade_count_2023: int
    trade_count_2024: int


@dataclass(frozen=True)
class MultiPeriodReport:
    """Combined multi-year normalized comparison.

    Contains per-year reports and a cross-year summary.
    PAPER/TEST only. Past results do NOT predict future performance.
    """

    period_2023: NormalizedComparisonReport
    period_2024: NormalizedComparisonReport
    yearly_summary: list[YearlyVariantSummary]


# ---------------------------------------------------------------------------
# Variant configuration factory
# ---------------------------------------------------------------------------


def _risk_cfg_for(variant_name: str, allocation: Decimal) -> RiskExitConfig:
    """Return RiskExitConfig for the named variant at the requested allocation."""
    if variant_name == "V1_BASELINE":
        # Risk exits disabled → crossover-only exits, same logic as V1 engine.
        return RiskExitConfig(
            enabled=False,
            use_bearish_crossover_exit=True,
            position_allocation_percentage=allocation,
        )
    if variant_name == "V2_STOP_ONLY":
        return RiskExitConfig(
            use_take_profit=False,
            maximum_holding_candles=0,
            use_bearish_crossover_exit=True,
            position_allocation_percentage=allocation,
        )
    if variant_name == "V2_STOP_TP":
        return RiskExitConfig(
            use_take_profit=True,
            maximum_holding_candles=0,
            use_bearish_crossover_exit=True,
            position_allocation_percentage=allocation,
        )
    if variant_name == "V2_STOP_TP_TIME":
        return RiskExitConfig(
            use_take_profit=True,
            maximum_holding_candles=192,
            use_bearish_crossover_exit=True,
            position_allocation_percentage=allocation,
        )
    raise ValueError(f"Unknown variant name: {variant_name!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zero_cost_config(config: BacktestConfig) -> BacktestConfig:
    return BacktestConfig(
        symbol=config.symbol,
        interval=config.interval,
        start_ms=config.start_ms,
        end_ms=config.end_ms,
        initial_capital=config.initial_capital,
        fee_percentage=_ZERO,
        slippage_percentage=_ZERO,
        force_close_at_end=config.force_close_at_end,
    )


def _run_one(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    risk_cfg: RiskExitConfig,
    strategy_engine: StrategyEngine,
    ind_config: IndicatorConfig,
) -> BacktestResult:
    engine = V2BacktestEngine(
        config=config,
        risk_exit_config=risk_cfg,
        strategy_engine=strategy_engine,
        indicator_config=ind_config,
    )
    return engine.run(all_candles, warmup_len)


def _build_normalized(
    variant_name: str,
    allocation: Decimal,
    result: BacktestResult,
    result_nc: BacktestResult,
) -> NormalizedVariantResult:
    """Build a NormalizedVariantResult from with-costs and no-costs BacktestResults."""
    trades = result.trades
    total_trades = result.total_trades

    avg_pnl: Decimal | None = None
    if total_trades > 0:
        total_net = sum((t.net_pnl for t in trades), _ZERO)
        avg_pnl = total_net / Decimal(str(total_trades))

    # Slippage approximation: total cost impact minus tracked fee charges.
    cost_delta = result_nc.final_equity - result.final_equity
    slippage_approx = max(_ZERO, cost_delta - result.total_fees)

    return NormalizedVariantResult(
        variant_name=variant_name,
        allocation_pct=allocation,
        return_pct=result.total_return_pct,
        return_pct_no_costs=result_nc.total_return_pct,
        final_equity=result.final_equity,
        max_drawdown_pct=result.max_drawdown_pct,
        profit_factor=result.profit_factor,
        win_rate_pct=result.win_rate_pct,
        total_fees=result.total_fees,
        slippage_cost=slippage_approx,
        total_trades=total_trades,
        avg_trade_pnl=avg_pnl,
        median_trade_pnl=_median_net_pnl(trades),
        exposure_pct=result.exposure_pct,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_normalized_comparison(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
    period_label: str = "",
) -> NormalizedComparisonReport:
    """Run all 12 (variant × allocation) combinations and return NormalizedComparisonReport.

    Each combination runs twice: once with configured costs, once with zero costs.
    All variants share identical candles and warmup period (no look-ahead).
    PAPER/TEST only. No variant is declared optimal.
    """
    from app.indicators.schemas import IndicatorConfig as _IC

    ind_config = indicator_config or _IC()
    strat_config = strategy_config or StrategyEngineConfig()
    strategy_engine = StrategyEngine(strat_config)
    zero_cfg = _zero_cost_config(config)

    matrix: list[NormalizedVariantResult] = []
    bah = _ZERO

    for variant_name in VARIANT_NAMES:
        for allocation in ALLOCATIONS:
            risk_cfg = _risk_cfg_for(variant_name, allocation)
            result = _run_one(
                all_candles, warmup_len, config, risk_cfg, strategy_engine, ind_config
            )
            result_nc = _run_one(
                all_candles, warmup_len, zero_cfg, risk_cfg, strategy_engine, ind_config
            )
            if bah == _ZERO:
                bah = result.buy_and_hold_return_pct
            matrix.append(_build_normalized(variant_name, allocation, result, result_nc))

    return NormalizedComparisonReport(
        matrix=matrix,
        bah_return_pct=bah,
        period_label=period_label,
    )


def run_multi_period_comparison(
    candles_2023: list[Candle],
    warmup_2023: int,
    config_2023: BacktestConfig,
    candles_2024: list[Candle],
    warmup_2024: int,
    config_2024: BacktestConfig,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
) -> MultiPeriodReport:
    """Run normalized comparison for 2023 and 2024 and build a cross-year summary.

    Both years use the same variant configurations and parameters.
    PAPER/TEST only. Past results do NOT predict future performance.
    """
    from app.indicators.schemas import IndicatorConfig as _IC

    ind_config = indicator_config or _IC()
    strat_config = strategy_config or StrategyEngineConfig()

    report_2023 = run_normalized_comparison(
        candles_2023,
        warmup_2023,
        config_2023,
        ind_config,
        strat_config,
        period_label="2023",
    )
    report_2024 = run_normalized_comparison(
        candles_2024,
        warmup_2024,
        config_2024,
        ind_config,
        strat_config,
        period_label="2024",
    )

    lookup_23: dict[tuple[str, Decimal], NormalizedVariantResult] = {
        (r.variant_name, r.allocation_pct): r for r in report_2023.matrix
    }
    lookup_24: dict[tuple[str, Decimal], NormalizedVariantResult] = {
        (r.variant_name, r.allocation_pct): r for r in report_2024.matrix
    }

    yearly: list[YearlyVariantSummary] = []
    for variant_name in VARIANT_NAMES:
        for allocation in ALLOCATIONS:
            r23 = lookup_23[(variant_name, allocation)]
            r24 = lookup_24[(variant_name, allocation)]

            ret_23 = r23.return_pct / _HUNDRED
            ret_24 = r24.return_pct / _HUNDRED
            combined = ((1 + ret_23) * (1 + ret_24) - 1) * _HUNDRED

            positive = sum(1 for r in (r23.return_pct, r24.return_pct) if r > _ZERO)
            worst_dd = max(r23.max_drawdown_pct, r24.max_drawdown_pct)

            pf_stability: Decimal | None = None
            if r23.profit_factor is not None and r24.profit_factor is not None:
                pf_stability = abs(r23.profit_factor - r24.profit_factor)

            yearly.append(
                YearlyVariantSummary(
                    variant_name=variant_name,
                    allocation_pct=allocation,
                    return_pct_2023=r23.return_pct,
                    return_pct_2024=r24.return_pct,
                    combined_return_pct=combined,
                    positive_years=positive,
                    worst_drawdown_pct=worst_dd,
                    profit_factor_stability=pf_stability,
                    trade_count_2023=r23.total_trades,
                    trade_count_2024=r24.total_trades,
                )
            )

    return MultiPeriodReport(
        period_2023=report_2023,
        period_2024=report_2024,
        yearly_summary=yearly,
    )
