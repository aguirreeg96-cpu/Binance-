"""Frozen B_4h forward paper-trading manifest — Etapa 6.1.

Forward paper trading activates the single configuration selected by the
Etapa 6.0 pre-registered evaluation (config "B" on the 4h timeframe). Every
parameter below is imported directly from app.backtesting.breakout_family
(the Stage 6.0 module) rather than redeclared, so there is no possibility
of drift between the backtested config and the forward-trading config.

DO NOT MODIFY any of these values. The forward engine computes a SHA-256
hash over this manifest and refuses to start if it does not match the hash
recorded in the database at first launch (see app/forward/engine.py).

PAPER/TEST only. No real money, no real orders. Past results do NOT
predict future performance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from decimal import Decimal
from typing import Any

from app.backtesting.breakout_family import (
    BREAKOUT_CONFIGS,
    BREAKOUT_SOURCE_INTERVAL,
    BREAKOUT_SYMBOL,
)
from app.backtesting.donchian_engine import DonchianConfig

STRATEGY_NAME = "donchian_breakout_B_4h"
STRATEGY_VERSION = "6.1.0"

FORWARD_SYMBOL = BREAKOUT_SYMBOL
FORWARD_SOURCE_INTERVAL = BREAKOUT_SOURCE_INTERVAL  # "15m"
FORWARD_TRADING_INTERVAL = "4h"

FORWARD_DONCHIAN_CONFIG: DonchianConfig = BREAKOUT_CONFIGS["B"]

FORWARD_FEE_PERCENTAGE = Decimal("0.1")
FORWARD_SLIPPAGE_PERCENTAGE = Decimal("0.05")

# Documentation only — recorded in the manifest/reports, never consumed by
# engine logic, and never used to alter forward behavior.
HISTORICAL_DOCUMENTATION: dict[str, Any] = {
    "evaluation_period": "2021-01-01 to 2025-12-31",
    "base_costs": {
        "compounded_return_pct": "13.5723",
        "profit_factor": "1.3047",
        "max_drawdown_pct": "6.5739",
        "total_trades": 109,
        "years_positive": 4,
        "years_total": 5,
        "yearly_returns_pct": {
            "2021": "1.62",
            "2022": "-5.34",
            "2023": "11.32",
            "2024": "4.41",
            "2025": "1.57",
        },
    },
    "conservative_costs": {
        "compounded_return_pct": "10.5411",
        "profit_factor": "1.2305",
        "max_drawdown_pct": "6.9392",
    },
}

FUTURE_EVALUATION_CRITERIA: dict[str, Any] = {
    "first_checkpoint_days": 90,
    "min_closed_trades_before_real_money": 20,
    "rules": [
        "No parameter changes during the evaluation period.",
        "No deleting losing trades.",
        "No resetting capital.",
        "No partial-result-driven strategy changes.",
        "No automatic transition to real money under any circumstance.",
    ],
}


def build_frozen_manifest() -> dict[str, Any]:
    """Canonical, JSON-serialisable dict of every frozen forward-trading parameter."""
    cfg = FORWARD_DONCHIAN_CONFIG
    return {
        "strategy_name": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "symbol": FORWARD_SYMBOL,
        "source_interval": FORWARD_SOURCE_INTERVAL,
        "trading_interval": FORWARD_TRADING_INTERVAL,
        "entry_lookback": cfg.entry_lookback,
        "exit_lookback": cfg.exit_lookback,
        "atr_period": cfg.atr_period,
        "atr_multiplier": str(cfg.atr_multiplier),
        "ema_period": cfg.ema_period,
        "ema_slope_lookback": cfg.ema_slope_lookback,
        "allocation_pct": str(cfg.allocation_pct),
        "long_only": True,
        "single_position": True,
        "pyramiding": False,
        "leverage": False,
        "fee_percentage": str(FORWARD_FEE_PERCENTAGE),
        "slippage_percentage": str(FORWARD_SLIPPAGE_PERCENTAGE),
        "stop_type": "ATR_INITIAL_THEN_NONDECREASING_TRAILING",
        "exit_rule": "ALTERNATIVE_DONCHIAN_EXIT",
        "execution_timing": "SIGNAL_AT_CLOSE_EXECUTE_AT_NEXT_OPEN",
    }


def compute_config_hash(manifest: dict[str, Any] | None = None) -> str:
    """SHA-256 hex digest over the canonical (sorted-key) JSON manifest."""
    m = manifest if manifest is not None else build_frozen_manifest()
    canonical = json.dumps(m, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_code_commit_hash() -> str | None:
    """Best-effort current git commit hash; None if unavailable."""
    try:
        result = subprocess.run(  # noqa: S603,S607 — fixed args, no shell, read-only
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


FROZEN_MANIFEST: dict[str, Any] = build_frozen_manifest()
FROZEN_CONFIG_HASH: str = compute_config_hash(FROZEN_MANIFEST)
