import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import init_db
from app.schemas.common import TradingMode

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
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Print safety banner — always visible at startup
    print(BANNER)
    logger.warning("=" * 60)
    logger.warning("STARTING IN MODE: %s", settings.trading_mode.upper())
    logger.warning("Symbol: %s | Interval: %s", settings.trading_symbol, settings.trading_interval)
    logger.warning("API Key: %s", settings.masked_api_key())
    logger.warning("=" * 60)

    # Initialize database
    init_db()
    logger.info("Database initialized.")

    yield

    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Binance Semi-Auto Trader",
        description=(
            "⚠ PAPER/TEST environment only. "
            "No real money. Educational purposes."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["system"])
    async def health():
        return {
            "status": "ok",
            "mode": settings.trading_mode,
            "symbol": settings.trading_symbol,
            "interval": settings.trading_interval,
            "warning": "PAPER/TEST environment — no real money",
        }

    @app.get("/config", tags=["system"])
    async def config_summary():
        """Return non-sensitive configuration for inspection."""
        return {
            "trading_mode": settings.trading_mode,
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
        }

    return app


app = create_app()
