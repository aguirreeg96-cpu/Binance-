"""
Forward paper-trading REST API for the frozen B_4h Donchian breakout config.

GET  /api/v1/paper-breakout/status   — launch identity, status, timestamps
GET  /api/v1/paper-breakout/signals  — per-closed-candle signal evaluations
GET  /api/v1/paper-breakout/position — current open position, if any
GET  /api/v1/paper-breakout/trades   — closed trades
GET  /api/v1/paper-breakout/equity   — equity-curve points derived from evaluations
GET  /api/v1/paper-breakout/config   — the frozen, immutable B_4h manifest
POST /api/v1/paper-breakout/start    — create or resume the single permanent launch
POST /api/v1/paper-breakout/stop     — stop the engine from evaluating further candles

No Binance calls are made by this router. No orders are ever sent to
Binance. POST /start and POST /stop only ever create or update local,
database-only paper-trading state — they never place, modify, or cancel
any exchange order. PAPER/TEST only. Past results do NOT predict future
performance.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.forward.engine import start_or_resume_launch
from app.forward.exceptions import ForwardConfigMismatchError
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

_DISCLAIMER = (
    "PAPER/TEST only. No real money was used or is at risk. "
    "Past results do NOT predict future performance and are not investment advice."
)


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
