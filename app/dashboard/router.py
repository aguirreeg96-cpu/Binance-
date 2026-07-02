"""Dashboard router — serves the paper trading web dashboard.

GET /dashboard  — returns the full single-page HTML dashboard.

The page fetches live data from GET /api/v1/paper-breakout/dashboard-summary
and renders it client-side using vanilla JavaScript.  No real orders, no
private keys, no Binance connection from this module.
PAPER/TEST only.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_CSP = (
    "default-src 'self'; "
    "style-src 'self'; "
    "script-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self';"
)

_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": _CSP,
}


@router.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> HTMLResponse:
    """Serve the paper trading dashboard HTML page."""
    html = (_TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html, headers=_HEADERS)
