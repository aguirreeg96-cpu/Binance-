import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import run_migrations, verify_db_connection
from app.market_data.client import BinanceMarketDataClient

logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          ⚠  PAPER / TEST ENVIRONMENT — NO REAL MONEY  ⚠     ║
║                                                              ║
║  This application is for educational purposes only.          ║
║  No real funds are at risk. Past signals do NOT predict      ║
║  future results. Trading involves significant risk of loss.  ║
╚══════════════════════════════════════════════════════════════╝
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(BANNER)
    logger.warning("=" * 60)
    logger.warning("STARTING IN MODE: %s", settings.trading_mode.value.upper())
    logger.warning(
        "Symbol: %s | Interval: %s",
        settings.trading_symbol,
        settings.trading_interval,
    )
    logger.warning("API Key: %s", settings.masked_api_key())
    logger.warning("Market data URL: %s", settings.binance_market_data_url)
    logger.warning("=" * 60)

    # Apply DB migrations (alembic upgrade head)
    run_migrations()
    verify_db_connection()

    # Create shared market data client (no API key needed)
    market_client = BinanceMarketDataClient(
        base_url=settings.binance_market_data_url,
        timeout=settings.market_data_timeout,
        max_retries=settings.market_data_max_retries,
        max_retry_after=settings.market_data_max_retry_after,
    )
    app.state.market_data_client = market_client

    yield

    await market_client.close()
    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Binance Semi-Auto Trader",
        description=("⚠ PAPER/TEST environment only. No real money. Educational purposes."),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Routers
    from app.api.backtesting import router as backtesting_router
    from app.api.diagnostic import router as diagnostic_router
    from app.api.events import router as events_router
    from app.api.health import router as health_router
    from app.api.indicators import router as indicators_router
    from app.api.market_data import router as market_data_router
    from app.api.paper_breakout import router as paper_breakout_router
    from app.api.strategy import router as strategy_router
    from app.dashboard.router import router as dashboard_router

    app.include_router(health_router)
    app.include_router(market_data_router)
    app.include_router(indicators_router)
    app.include_router(strategy_router)
    app.include_router(backtesting_router)
    app.include_router(paper_breakout_router)
    app.include_router(events_router)
    app.include_router(diagnostic_router)
    app.include_router(dashboard_router)

    # Static files for the dashboard (CSS, JS)
    _static_dir = Path(__file__).parent / "dashboard" / "static"
    app.mount("/dashboard/static", StaticFiles(directory=str(_static_dir)), name="dashboard-static")

    @app.get("/config", tags=["system"])
    async def config_summary() -> dict[str, str | int]:
        return {
            "trading_mode": settings.trading_mode.value,
            "symbol": settings.trading_symbol,
            "interval": settings.trading_interval,
            "risk_per_trade": str(settings.risk_per_trade),
            "daily_loss_limit": str(settings.daily_loss_limit),
            "max_open_positions": settings.max_open_positions,
            "max_trades_per_day": settings.max_trades_per_day,
            "paper_initial_balance": str(settings.paper_initial_balance),
            "ema_fast": settings.ema_fast,
            "ema_slow": settings.ema_slow,
            "ema_trend": settings.ema_trend,
            "rsi_period": settings.rsi_period,
            "atr_period": settings.atr_period,
            "atr_sl_multiplier": str(settings.atr_sl_multiplier),
            "rr_ratio": str(settings.rr_ratio),
            "market_data_url": settings.binance_market_data_url,
        }

    return app


app = create_app()
