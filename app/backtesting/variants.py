"""Strategy variant comparator for V1 vs V2 exits.

Runs four controlled variants over identical data:
  V1_BASELINE    — bearish crossover only, 100% allocation (V1 engine)
  V2_STOP_ONLY   — ATR stop 2.0, no TP, crossover secondary, 25% allocation
  V2_STOP_TP     — ATR stop 2.0, TP 2R, crossover secondary, 25% allocation
  V2_STOP_TP_TIME— ATR stop 2.0, TP 2R, max 192 candles, crossover secondary, 25%

No variant is declared optimal or profitable.
PAPER/TEST only — no real orders, no real capital at risk.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.backtesting.config import BacktestConfig
from app.backtesting.engine import BacktestEngine
from app.backtesting.risk_exit_config import RiskExitConfig
from app.backtesting.schemas import BacktestResult, BacktestTrade
from app.backtesting.v2_engine import V2BacktestEngine
from app.indicators.schemas import IndicatorConfig
from app.market_data.interval_utils import interval_to_ms
from app.models.candle import Candle
from app.strategy.config import StrategyEngineConfig
from app.strategy.engine import StrategyEngine
from app.strategy.reasons import ReasonCode

_ZERO = Decimal("0")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantResult:
    """Results for one strategy variant, with and without transaction costs.

    PAPER/TEST only. No result implies profitability or is declared optimal.
    """

    name: str
    description: str
    result: BacktestResult  # with configured fees and slippage
    result_no_costs: BacktestResult  # fee=0%, slippage=0%

    # Exit type counts from result.trades
    sl_exits: int
    tp_exits: int
    timed_exits: int
    crossover_exits: int
    ambiguous_exits: int
    forced_exits: int

    # Summary stats
    avg_duration_candles: Decimal | None
    median_net_pnl: Decimal | None


@dataclass(frozen=True)
class ComparisonReport:
    """Side-by-side comparison of all strategy variants.

    PAPER/TEST only. No variant is declared the best or expected to be profitable.
    Past backtest results do NOT predict future performance.
    """

    variants: list[VariantResult]
    bah_return_pct: Decimal  # buy-and-hold return over the eval period


# ---------------------------------------------------------------------------
# Fixed variant definitions
# ---------------------------------------------------------------------------

_V1_BASELINE_DESC = (
    "V1: bearish crossover exit only, 100% capital allocation. "
    "No stop-loss, no take-profit, no time limit."
)
_V2_STOP_ONLY_DESC = (
    "V2: ATR stop-loss (2×ATR), no take-profit, bearish crossover secondary, "
    "25% capital allocation."
)
_V2_STOP_TP_DESC = (
    "V2: ATR stop-loss (2×ATR), take-profit (2R), bearish crossover secondary, "
    "25% capital allocation."
)
_V2_STOP_TP_TIME_DESC = (
    "V2: ATR stop-loss (2×ATR), take-profit (2R), max 192 candles, "
    "bearish crossover secondary, 25% capital allocation."
)

_V2_STOP_ONLY_CFG = RiskExitConfig(
    use_take_profit=False,
    maximum_holding_candles=0,
)
_V2_STOP_TP_CFG = RiskExitConfig(
    use_take_profit=True,
    maximum_holding_candles=0,
)
_V2_STOP_TP_TIME_CFG = RiskExitConfig(
    use_take_profit=True,
    maximum_holding_candles=192,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_variants(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    indicator_config: IndicatorConfig | None = None,
    strategy_config: StrategyEngineConfig | None = None,
) -> ComparisonReport:
    """Run all four controlled variants and return a ComparisonReport.

    The same all_candles / warmup_len are used for every variant (identical data).
    A zero-cost version (fee=0%, slip=0%) is also computed for each variant.
    """
    ind_config = indicator_config or IndicatorConfig()
    strat_config = strategy_config or StrategyEngineConfig()
    interval_ms = interval_to_ms(config.interval)
    zero_cost_config = _zero_cost(config)

    # Shared strategy engine instance (stateless, safe to reuse)
    strategy_engine = StrategyEngine(strat_config)

    variants: list[VariantResult] = []

    # V1_BASELINE
    v1_result = _run_v1(all_candles, warmup_len, config, ind_config, strategy_engine)
    v1_nc = _run_v1(all_candles, warmup_len, zero_cost_config, ind_config, strategy_engine)
    variants.append(
        _build_variant(
            "V1_BASELINE",
            _V1_BASELINE_DESC,
            v1_result,
            v1_nc,
            interval_ms,
        )
    )

    # V2 variants
    for name, desc, risk_cfg in [
        ("V2_STOP_ONLY", _V2_STOP_ONLY_DESC, _V2_STOP_ONLY_CFG),
        ("V2_STOP_TP", _V2_STOP_TP_DESC, _V2_STOP_TP_CFG),
        ("V2_STOP_TP_TIME", _V2_STOP_TP_TIME_DESC, _V2_STOP_TP_TIME_CFG),
    ]:
        v2_result = _run_v2(all_candles, warmup_len, config, risk_cfg, ind_config, strategy_engine)
        v2_nc = _run_v2(
            all_candles, warmup_len, zero_cost_config, risk_cfg, ind_config, strategy_engine
        )
        variants.append(_build_variant(name, desc, v2_result, v2_nc, interval_ms))

    bah = v1_result.buy_and_hold_return_pct
    return ComparisonReport(variants=variants, bah_return_pct=bah)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zero_cost(config: BacktestConfig) -> BacktestConfig:
    return BacktestConfig(
        symbol=config.symbol,
        interval=config.interval,
        start_ms=config.start_ms,
        end_ms=config.end_ms,
        initial_capital=config.initial_capital,
        fee_percentage=Decimal("0"),
        slippage_percentage=Decimal("0"),
        force_close_at_end=config.force_close_at_end,
    )


def _run_v1(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    ind_config: IndicatorConfig,
    strategy_engine: StrategyEngine,
) -> BacktestResult:
    engine = BacktestEngine(
        config=config,
        strategy_engine=strategy_engine,
        indicator_config=ind_config,
    )
    return engine.run(all_candles, warmup_len)


def _run_v2(
    all_candles: list[Candle],
    warmup_len: int,
    config: BacktestConfig,
    risk_cfg: RiskExitConfig,
    ind_config: IndicatorConfig,
    strategy_engine: StrategyEngine,
) -> BacktestResult:
    engine = V2BacktestEngine(
        config=config,
        risk_exit_config=risk_cfg,
        strategy_engine=strategy_engine,
        indicator_config=ind_config,
    )
    return engine.run(all_candles, warmup_len)


def _build_variant(
    name: str,
    description: str,
    result: BacktestResult,
    result_no_costs: BacktestResult,
    interval_ms: int,
) -> VariantResult:
    trades = result.trades
    return VariantResult(
        name=name,
        description=description,
        result=result,
        result_no_costs=result_no_costs,
        sl_exits=_count_exit(trades, str(ReasonCode.ATR_STOP_LOSS)),
        tp_exits=_count_exit(trades, str(ReasonCode.RISK_REWARD_TAKE_PROFIT)),
        timed_exits=_count_exit(trades, str(ReasonCode.MAX_HOLDING_TIME)),
        crossover_exits=_count_exit(trades, str(ReasonCode.BEARISH_CROSSOVER)),
        ambiguous_exits=_count_exit(trades, str(ReasonCode.AMBIGUOUS_INTRABAR_STOP_FIRST)),
        forced_exits=_count_exit(trades, str(ReasonCode.FORCED_END_OF_BACKTEST)),
        avg_duration_candles=_avg_duration(trades, interval_ms),
        median_net_pnl=_median_net_pnl(trades),
    )


def _count_exit(trades: list[BacktestTrade], code: str) -> int:
    return sum(1 for t in trades if code in t.exit_reasons)


def _avg_duration(trades: list[BacktestTrade], interval_ms: int) -> Decimal | None:
    if not trades:
        return None
    # Round to nearest candle: add half-interval before integer division
    total = sum(
        (t.exit_exec_time - t.entry_exec_time + interval_ms // 2) // interval_ms for t in trades
    )
    return Decimal(str(total)) / Decimal(str(len(trades)))


def _median_net_pnl(trades: list[BacktestTrade]) -> Decimal | None:
    if not trades:
        return None
    sorted_pnls = sorted(t.net_pnl for t in trades)
    n = len(sorted_pnls)
    mid = n // 2
    if n % 2 == 1:
        return sorted_pnls[mid]
    return (sorted_pnls[mid - 1] + sorted_pnls[mid]) / _TWO
