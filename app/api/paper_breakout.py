"""
Forward paper-trading REST API for the frozen B_4h Donchian breakout config.

GET  /api/v1/paper-breakout/status             — launch identity, status, timestamps
GET  /api/v1/paper-breakout/signals            — per-closed-candle signal evaluations
GET  /api/v1/paper-breakout/position           — current open position, if any
GET  /api/v1/paper-breakout/trades             — closed trades
GET  /api/v1/paper-breakout/equity             — equity-curve points derived from evaluations
GET  /api/v1/paper-breakout/config             — the frozen, immutable B_4h manifest
GET  /api/v1/paper-breakout/dashboard-summary  — aggregate snapshot for the dashboard UI
GET  /api/v1/paper-breakout/signals/export     — download forward_signals.csv
GET  /api/v1/paper-breakout/trades/export      — download forward_trades.csv
GET  /api/v1/paper-breakout/equity/export      — download forward_equity_curve.csv
GET  /api/v1/paper-breakout/config/export      — download forward_manifest.json
POST /api/v1/paper-breakout/start              — create or resume the single permanent launch
POST /api/v1/paper-breakout/stop               — stop the engine from evaluating further candles

No Binance calls are made by this router. No orders are ever sent to
Binance. POST /start and POST /stop only ever create or update local,
database-only paper-trading state — they never place, modify, or cancel
any exchange order. PAPER/TEST only. Past results do NOT predict future
performance.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from app.config import get_settings
from app.database import get_db
from app.forward.engine import start_or_resume_launch
from app.forward.exceptions import ForwardConfigMismatchError
from app.forward.exporters import (
    export_forward_equity_curve_csv,
    export_forward_manifest_json,
    export_forward_signals_csv,
    export_forward_trades_csv,
)
from app.forward.manifest import (
    FORWARD_SYMBOL,
    FUTURE_EVALUATION_CRITERIA,
    HISTORICAL_DOCUMENTATION,
    STRATEGY_NAME,
    build_frozen_manifest,
)
from app.models.forward_paper import ForwardLaunch, ForwardSignalEvaluation
from app.models.paper_account import PaperAccount
from app.models.position import Position
from app.models.trade import Trade
from app.repositories.forward_repository import ForwardRepository
from app.schemas.common import ForwardLaunchStatus

router = APIRouter(prefix="/api/v1/paper-breakout", tags=["paper-breakout"])

_MAX_SIGNALS_LIMIT = 5000
_DASHBOARD_SIGNALS_LIMIT = 50
_DASHBOARD_TRADES_LIMIT = 100

_DISCLAIMER = (
    "PAPER/TEST only. No real money was used or is at risk. "
    "Past results do NOT predict future performance and are not investment advice."
)

_FOUR_HOUR_MS = 14_400_000
_EPOCH = datetime(1970, 1, 1)
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


def _dt_to_ms(dt: datetime) -> int:
    return int((dt - _EPOCH).total_seconds() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    return _EPOCH + timedelta(milliseconds=ms)


def _next_eligible_4h_close(launch: ForwardLaunch) -> str | None:
    """Approximate UTC ISO timestamp of the next 4h candle close to evaluate."""
    if launch.last_evaluated_candle_close is not None:
        return (launch.last_evaluated_candle_close + timedelta(hours=4)).isoformat()
    launch_ms = _dt_to_ms(launch.launch_timestamp)
    aligned_ms = ((launch_ms + _FOUR_HOUR_MS - 1) // _FOUR_HOUR_MS) * _FOUR_HOUR_MS
    next_close_ms = aligned_ms + _FOUR_HOUR_MS
    return _ms_to_dt(next_close_ms).isoformat()


def _now_naive_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _get_launch_or_404(db: Session) -> ForwardLaunch:
    launch = ForwardRepository(db).get_launch(STRATEGY_NAME, FORWARD_SYMBOL)
    if launch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No forward paper-trading launch exists yet. POST /start first.",
        )
    return launch


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    balance: str
    asset_balance: str
    equity: str
    realized_pnl: str
    total_fees_paid: str
    currency: str

    @classmethod
    def from_account(cls, a: PaperAccount) -> AccountResponse:
        return cls(
            balance=str(a.balance),
            asset_balance=str(a.asset_balance),
            equity=str(a.equity),
            realized_pnl=str(a.realized_pnl),
            total_fees_paid=str(a.total_fees_paid),
            currency=a.currency,
        )


class StatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    launch_id: int
    strategy_name: str
    strategy_version: str
    symbol: str
    launch_timestamp: str
    status: str
    frozen_config_hash: str
    code_commit_hash: str | None
    initial_capital: str
    last_evaluated_candle_close: str | None
    account: AccountResponse | None

    @classmethod
    def from_launch(cls, launch: ForwardLaunch, account: PaperAccount | None) -> StatusResponse:
        return cls(
            launch_id=launch.id,
            strategy_name=launch.strategy_name,
            strategy_version=launch.strategy_version,
            symbol=launch.symbol,
            launch_timestamp=launch.launch_timestamp.isoformat(),
            status=launch.status,
            frozen_config_hash=launch.frozen_config_hash,
            code_commit_hash=launch.code_commit_hash,
            initial_capital=str(launch.initial_capital),
            last_evaluated_candle_close=(
                launch.last_evaluated_candle_close.isoformat()
                if launch.last_evaluated_candle_close
                else None
            ),
            account=AccountResponse.from_account(account) if account is not None else None,
        )


class SignalEvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    candle_close_time: str
    evaluated_at: str
    signal: str
    reasons: list[str]
    raw_market_price: str | None
    planned_execution_time: str | None
    entry_donchian_level: str | None
    exit_donchian_level: str | None
    ema_200: str | None
    ema_slope: str | None
    atr: str | None
    initial_stop: str | None
    current_trailing_stop: str | None
    highest_high_since_entry: str | None
    position_quantity: str | None
    cash: str
    equity: str
    frozen_config_hash: str

    @classmethod
    def from_evaluation(cls, e: ForwardSignalEvaluation) -> SignalEvaluationResponse:
        def d(v: Decimal | None) -> str | None:
            return str(v) if v is not None else None

        return cls(
            candle_close_time=e.candle_close_time.isoformat(),
            evaluated_at=e.evaluated_at.isoformat(),
            signal=e.signal,
            reasons=list(e.reasons),
            raw_market_price=d(e.raw_market_price),
            planned_execution_time=(
                e.planned_execution_time.isoformat() if e.planned_execution_time else None
            ),
            entry_donchian_level=d(e.entry_donchian_level),
            exit_donchian_level=d(e.exit_donchian_level),
            ema_200=d(e.ema_200),
            ema_slope=d(e.ema_slope),
            atr=d(e.atr),
            initial_stop=d(e.initial_stop),
            current_trailing_stop=d(e.current_trailing_stop),
            highest_high_since_entry=d(e.highest_high_since_entry),
            position_quantity=d(e.position_quantity),
            cash=str(e.cash),
            equity=str(e.equity),
            frozen_config_hash=e.frozen_config_hash,
        )


class PositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    id: int
    symbol: str
    status: str
    entry_price: str
    quantity: str
    stop_loss: str
    trailing_stop_price: str | None
    opened_at: str

    @classmethod
    def from_position(cls, p: Position) -> PositionResponse:
        return cls(
            id=p.id,
            symbol=p.symbol,
            status=p.status,
            entry_price=str(p.entry_price),
            quantity=str(p.quantity),
            stop_loss=str(p.stop_loss),
            trailing_stop_price=(
                str(p.trailing_stop_price) if p.trailing_stop_price is not None else None
            ),
            opened_at=p.opened_at.isoformat(),
        )


class TradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    id: int
    entry_price: str
    exit_price: str
    quantity: str
    side: str
    gross_pnl: str
    commission: str
    net_pnl: str
    exit_reason: str
    opened_at: str
    closed_at: str

    @classmethod
    def from_trade(cls, t: Trade) -> TradeResponse:
        return cls(
            id=t.id,
            entry_price=str(t.entry_price),
            exit_price=str(t.exit_price),
            quantity=str(t.quantity),
            side=t.side,
            gross_pnl=str(t.gross_pnl),
            commission=str(t.commission),
            net_pnl=str(t.net_pnl),
            exit_reason=t.exit_reason,
            opened_at=t.opened_at.isoformat(),
            closed_at=t.closed_at.isoformat(),
        )


class EquityPointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    candle_close_time: str
    cash: str
    equity: str
    position_quantity: str | None
    signal: str

    @classmethod
    def from_evaluation(cls, e: ForwardSignalEvaluation) -> EquityPointResponse:
        return cls(
            candle_close_time=e.candle_close_time.isoformat(),
            cash=str(e.cash),
            equity=str(e.equity),
            position_quantity=(
                str(e.position_quantity) if e.position_quantity is not None else None
            ),
            signal=e.signal,
        )


class StartRequest(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    initial_capital: Decimal | None = None


class TradeStatsSummary(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: str | None
    total_realized_pnl: str
    total_realized_pnl_pct: str | None

    @classmethod
    def from_trades(cls, trades: list[Trade], initial_capital: Decimal) -> TradeStatsSummary:
        if not trades:
            return cls(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=None,
                total_realized_pnl="0",
                total_realized_pnl_pct=None,
            )
        winning = sum(1 for t in trades if t.net_pnl > Decimal("0"))
        losing = sum(1 for t in trades if t.net_pnl <= Decimal("0"))
        total_pnl = sum((t.net_pnl for t in trades), Decimal("0"))
        win_rate = Decimal(winning) * 100 / len(trades)
        pnl_pct = (total_pnl / initial_capital * 100) if initial_capital else None
        return cls(
            total_trades=len(trades),
            winning_trades=winning,
            losing_trades=losing,
            win_rate_pct=f"{float(win_rate):.1f}",
            total_realized_pnl=str(total_pnl),
            total_realized_pnl_pct=f"{float(pnl_pct):.2f}" if pnl_pct is not None else None,
        )


class DashboardPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    id: int
    symbol: str
    status: str
    entry_price: str
    quantity: str
    stop_loss: str
    trailing_stop_price: str | None
    opened_at: str
    entry_cost: str
    duration_hours: str
    highest_high_since_entry: str | None
    current_trailing_stop: str | None

    @classmethod
    def from_position(
        cls,
        p: Position,
        now: datetime,
        latest_eval: ForwardSignalEvaluation | None = None,
    ) -> DashboardPositionResponse:
        entry_cost = p.entry_price * p.quantity
        duration = (now - p.opened_at).total_seconds() / 3600
        return cls(
            id=p.id,
            symbol=p.symbol,
            status=p.status,
            entry_price=str(p.entry_price),
            quantity=str(p.quantity),
            stop_loss=str(p.stop_loss),
            trailing_stop_price=(
                str(p.trailing_stop_price) if p.trailing_stop_price is not None else None
            ),
            opened_at=p.opened_at.isoformat(),
            entry_cost=str(entry_cost),
            duration_hours=f"{duration:.1f}",
            highest_high_since_entry=(
                str(latest_eval.highest_high_since_entry)
                if latest_eval is not None and latest_eval.highest_high_since_entry is not None
                else None
            ),
            current_trailing_stop=(
                str(latest_eval.current_trailing_stop)
                if latest_eval is not None and latest_eval.current_trailing_stop is not None
                else None
            ),
        )


class DashboardTradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    id: int
    entry_price: str
    exit_price: str
    quantity: str
    side: str
    gross_pnl: str
    commission: str
    net_pnl: str
    exit_reason: str
    opened_at: str
    closed_at: str
    return_pct: str
    duration_hours: str

    @classmethod
    def from_trade(cls, t: Trade) -> DashboardTradeResponse:
        entry_cost = t.entry_price * t.quantity
        ret_pct = (
            (t.net_pnl / entry_cost * Decimal("100")) if entry_cost != 0 else Decimal("0")
        )
        duration = (t.closed_at - t.opened_at).total_seconds() / 3600
        return cls(
            id=t.id,
            entry_price=str(t.entry_price),
            exit_price=str(t.exit_price),
            quantity=str(t.quantity),
            side=t.side,
            gross_pnl=str(t.gross_pnl),
            commission=str(t.commission),
            net_pnl=str(t.net_pnl),
            exit_reason=t.exit_reason,
            opened_at=t.opened_at.isoformat(),
            closed_at=t.closed_at.isoformat(),
            return_pct=f"{float(ret_pct):.2f}",
            duration_hours=f"{duration:.1f}",
        )


class SystemStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    launch_exists: bool
    api_available: bool
    total_evaluations: int
    data_gap_errors: int
    last_evaluated_candle_close: str | None


class DashboardSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=False)

    generated_at: str
    launch_exists: bool
    launch: StatusResponse | None
    account: AccountResponse | None
    signals: list[SignalEvaluationResponse]
    position: DashboardPositionResponse | None
    recent_trades: list[DashboardTradeResponse]
    equity_curve: list[EquityPointResponse]
    frozen_config: dict[str, Any]
    next_eligible_close_utc: str | None
    trade_stats: TradeStatsSummary
    warnings: list[str]
    system: SystemStatusResponse


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", response_model=StatusResponse)
async def get_status(db: Session = Depends(get_db)) -> StatusResponse:
    launch = _get_launch_or_404(db)
    account = ForwardRepository(db).get_account(launch)
    return StatusResponse.from_launch(launch, account)


@router.get("/signals", response_model=list[SignalEvaluationResponse])
async def get_signals(
    limit: Annotated[int | None, Query(ge=1, le=_MAX_SIGNALS_LIMIT)] = None,
    db: Session = Depends(get_db),
) -> list[SignalEvaluationResponse]:
    launch = _get_launch_or_404(db)
    evaluations = ForwardRepository(db).list_evaluations(launch, limit=limit)
    return [SignalEvaluationResponse.from_evaluation(e) for e in evaluations]


@router.get("/position", response_model=PositionResponse | None)
async def get_position(db: Session = Depends(get_db)) -> PositionResponse | None:
    launch = _get_launch_or_404(db)
    position = ForwardRepository(db).get_open_position(launch)
    return PositionResponse.from_position(position) if position is not None else None


@router.get("/trades", response_model=list[TradeResponse])
async def get_trades(db: Session = Depends(get_db)) -> list[TradeResponse]:
    launch = _get_launch_or_404(db)
    trades = ForwardRepository(db).list_trades(launch)
    return [TradeResponse.from_trade(t) for t in trades]


@router.get("/equity", response_model=list[EquityPointResponse])
async def get_equity(db: Session = Depends(get_db)) -> list[EquityPointResponse]:
    launch = _get_launch_or_404(db)
    evaluations = ForwardRepository(db).list_evaluations(launch)
    return [EquityPointResponse.from_evaluation(e) for e in evaluations]


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """The frozen, immutable B_4h manifest. Never depends on launch state."""
    return {
        "frozen_configuration": build_frozen_manifest(),
        "historical_backtest_documentation": HISTORICAL_DOCUMENTATION,
        "future_evaluation_criteria": FUTURE_EVALUATION_CRITERIA,
        "paper_test_disclaimer": _DISCLAIMER,
    }


@router.post("/start", response_model=StatusResponse)
async def start_launch(
    body: StartRequest = StartRequest(), db: Session = Depends(get_db)
) -> StatusResponse:
    """Create or resume the single permanent launch. Never places any order.

    If a launch already exists, `initial_capital` is ignored — the spec
    forbids changing capital after launch. Refuses (409) if the frozen
    config hash recorded at first launch no longer matches the current code.
    """
    settings = get_settings()
    requested_capital = (
        body.initial_capital
        if body.initial_capital is not None
        else settings.forward_initial_capital
    )
    try:
        launch = start_or_resume_launch(db, now=_now_naive_utc(), initial_capital=requested_capital)
    except ForwardConfigMismatchError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    account = ForwardRepository(db).get_account(launch)
    return StatusResponse.from_launch(launch, account)


@router.post("/stop", response_model=StatusResponse)
async def stop_launch(db: Session = Depends(get_db)) -> StatusResponse:
    """Stop the engine from evaluating further candles. Never closes any position."""
    launch = _get_launch_or_404(db)
    repo = ForwardRepository(db)
    repo.mark_launch_status(launch, ForwardLaunchStatus.STOPPED)
    db.commit()
    db.refresh(launch)
    account = repo.get_account(launch)
    return StatusResponse.from_launch(launch, account)


@router.get("/dashboard-summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(
    response: Response, db: Session = Depends(get_db)
) -> DashboardSummaryResponse:
    """Aggregate read-only snapshot for the dashboard UI.

    Returns all display data in one round-trip: launch, account, last 50
    signals, open position, recent trades, full equity curve, frozen config,
    trade statistics, and system status.  Adds Cache-Control: no-store and
    X-Content-Type-Options headers.

    Never places orders, never mutates state, never calls Binance.
    """
    response.headers.update(_SECURITY_HEADERS)

    now = _now_naive_utc()
    repo = ForwardRepository(db)
    launch = repo.get_launch(STRATEGY_NAME, FORWARD_SYMBOL)

    if launch is None:
        return DashboardSummaryResponse(
            generated_at=now.isoformat(),
            launch_exists=False,
            launch=None,
            account=None,
            signals=[],
            position=None,
            recent_trades=[],
            equity_curve=[],
            frozen_config=build_frozen_manifest(),
            next_eligible_close_utc=None,
            trade_stats=TradeStatsSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=None,
                total_realized_pnl="0",
                total_realized_pnl_pct=None,
            ),
            warnings=[
                "No forward paper-trading launch exists yet. "
                "Call POST /api/v1/paper-breakout/start to create one."
            ],
            system=SystemStatusResponse(
                launch_exists=False,
                api_available=True,
                total_evaluations=0,
                data_gap_errors=0,
                last_evaluated_candle_close=None,
            ),
        )

    account = repo.get_account(launch)
    position = repo.get_open_position(launch)
    all_evaluations = repo.list_evaluations(launch)  # ASC by candle_close_time
    trades = repo.list_trades(launch)  # ASC by closed_at

    recent_signals = list(reversed(all_evaluations[-_DASHBOARD_SIGNALS_LIMIT:]))
    recent_trades = list(reversed(trades[-_DASHBOARD_TRADES_LIMIT:]))
    latest_eval = all_evaluations[-1] if all_evaluations else None

    data_gap_errors = sum(1 for e in all_evaluations if e.signal == "ERROR_DATA_GAP")

    warnings: list[str] = []
    if launch.status == "STOPPED":
        warnings.append(
            "El launch está en estado STOPPED. El motor no evaluará nuevas velas "
            "hasta que sea reiniciado."
        )
    elif launch.status == "ERROR":
        warnings.append("El launch está en estado ERROR.")
    if data_gap_errors > 0:
        warnings.append(
            f"{data_gap_errors} evaluación(es) encontraron datos faltantes (ERROR_DATA_GAP)."
        )

    pos_response: DashboardPositionResponse | None = None
    if position is not None:
        pos_response = DashboardPositionResponse.from_position(position, now, latest_eval)

    return DashboardSummaryResponse(
        generated_at=now.isoformat(),
        launch_exists=True,
        launch=StatusResponse.from_launch(launch, account),
        account=AccountResponse.from_account(account) if account is not None else None,
        signals=[SignalEvaluationResponse.from_evaluation(e) for e in recent_signals],
        position=pos_response,
        recent_trades=[DashboardTradeResponse.from_trade(t) for t in recent_trades],
        equity_curve=[EquityPointResponse.from_evaluation(e) for e in all_evaluations],
        frozen_config=build_frozen_manifest(),
        next_eligible_close_utc=_next_eligible_4h_close(launch),
        trade_stats=TradeStatsSummary.from_trades(
            trades, launch.initial_capital if launch else Decimal("10000")
        ),
        warnings=warnings,
        system=SystemStatusResponse(
            launch_exists=True,
            api_available=True,
            total_evaluations=len(all_evaluations),
            data_gap_errors=data_gap_errors,
            last_evaluated_candle_close=(
                launch.last_evaluated_candle_close.isoformat()
                if launch.last_evaluated_candle_close
                else None
            ),
        ),
    )


@router.get("/signals/export")
async def export_signals_csv(db: Session = Depends(get_db)) -> FileResponse:
    """Download all signal evaluations as CSV. Reuses the forward exporter."""
    launch = _get_launch_or_404(db)
    evaluations = ForwardRepository(db).list_evaluations(launch)
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="forward_signals_")
    os.close(fd)
    export_forward_signals_csv(evaluations, Path(path))
    return FileResponse(
        path,
        media_type="text/csv",
        filename="forward_signals.csv",
        background=BackgroundTask(os.unlink, path),
    )


@router.get("/trades/export")
async def export_trades_csv(db: Session = Depends(get_db)) -> FileResponse:
    """Download all closed trades as CSV. Reuses the forward exporter."""
    launch = _get_launch_or_404(db)
    trades = ForwardRepository(db).list_trades(launch)
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="forward_trades_")
    os.close(fd)
    export_forward_trades_csv(trades, Path(path))
    return FileResponse(
        path,
        media_type="text/csv",
        filename="forward_trades.csv",
        background=BackgroundTask(os.unlink, path),
    )


@router.get("/equity/export")
async def export_equity_csv(db: Session = Depends(get_db)) -> FileResponse:
    """Download the equity curve as CSV. Reuses the forward exporter."""
    launch = _get_launch_or_404(db)
    evaluations = ForwardRepository(db).list_evaluations(launch)
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="forward_equity_")
    os.close(fd)
    export_forward_equity_curve_csv(evaluations, Path(path))
    return FileResponse(
        path,
        media_type="text/csv",
        filename="forward_equity_curve.csv",
        background=BackgroundTask(os.unlink, path),
    )


@router.get("/config/export")
async def export_config_json(db: Session = Depends(get_db)) -> FileResponse:
    """Download the frozen manifest as JSON. Reuses the forward exporter."""
    launch = _get_launch_or_404(db)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="forward_manifest_")
    os.close(fd)
    export_forward_manifest_json(launch, Path(path))
    return FileResponse(
        path,
        media_type="application/json",
        filename="forward_manifest.json",
        background=BackgroundTask(os.unlink, path),
    )
