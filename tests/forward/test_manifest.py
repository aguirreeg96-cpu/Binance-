"""Stage 6.1 tests: frozen B_4h manifest content, immutability, and hashing."""

from __future__ import annotations

import json
from decimal import Decimal

from app.backtesting.breakout_family import BREAKOUT_CONFIGS, BREAKOUT_SOURCE_INTERVAL
from app.forward.manifest import (
    FORWARD_DONCHIAN_CONFIG,
    FORWARD_FEE_PERCENTAGE,
    FORWARD_SLIPPAGE_PERCENTAGE,
    FORWARD_SOURCE_INTERVAL,
    FORWARD_SYMBOL,
    FORWARD_TRADING_INTERVAL,
    FROZEN_CONFIG_HASH,
    FROZEN_MANIFEST,
    FUTURE_EVALUATION_CRITERIA,
    HISTORICAL_DOCUMENTATION,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    build_frozen_manifest,
    compute_config_hash,
)


class TestFrozenConfigMatchesStage60:
    def test_uses_config_b_directly(self):
        assert FORWARD_DONCHIAN_CONFIG is BREAKOUT_CONFIGS["B"]

    def test_immutable_parameter_values(self):
        cfg = FORWARD_DONCHIAN_CONFIG
        assert cfg.entry_lookback == 20
        assert cfg.exit_lookback == 10
        assert cfg.atr_period == 14
        assert cfg.atr_multiplier == Decimal("3.0")
        assert cfg.ema_period == 200
        assert cfg.ema_slope_lookback == 10
        assert cfg.allocation_pct == Decimal("25")

    def test_symbol_and_intervals(self):
        assert FORWARD_SYMBOL == "BTCUSDT"
        assert FORWARD_SOURCE_INTERVAL == BREAKOUT_SOURCE_INTERVAL == "15m"
        assert FORWARD_TRADING_INTERVAL == "4h"

    def test_fee_and_slippage(self):
        assert FORWARD_FEE_PERCENTAGE == Decimal("0.1")
        assert FORWARD_SLIPPAGE_PERCENTAGE == Decimal("0.05")


class TestBuildFrozenManifest:
    def test_contains_every_immutable_parameter(self):
        manifest = build_frozen_manifest()
        assert manifest["strategy_name"] == STRATEGY_NAME
        assert manifest["strategy_version"] == STRATEGY_VERSION
        assert manifest["symbol"] == "BTCUSDT"
        assert manifest["source_interval"] == "15m"
        assert manifest["trading_interval"] == "4h"
        assert manifest["entry_lookback"] == 20
        assert manifest["exit_lookback"] == 10
        assert manifest["atr_period"] == 14
        assert manifest["atr_multiplier"] == "3.0"
        assert manifest["ema_period"] == 200
        assert manifest["ema_slope_lookback"] == 10
        assert manifest["allocation_pct"] == "25"
        assert manifest["long_only"] is True
        assert manifest["single_position"] is True
        assert manifest["pyramiding"] is False
        assert manifest["leverage"] is False
        assert manifest["fee_percentage"] == "0.1"
        assert manifest["slippage_percentage"] == "0.05"
        assert manifest["execution_timing"] == "SIGNAL_AT_CLOSE_EXECUTE_AT_NEXT_OPEN"

    def test_manifest_is_json_serialisable(self):
        manifest = build_frozen_manifest()
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        assert json.loads(canonical) == manifest

    def test_module_level_manifest_matches_builder(self):
        assert FROZEN_MANIFEST == build_frozen_manifest()


class TestConfigHash:
    def test_deterministic(self):
        h1 = compute_config_hash(build_frozen_manifest())
        h2 = compute_config_hash(build_frozen_manifest())
        assert h1 == h2

    def test_module_level_hash_matches_builder(self):
        assert FROZEN_CONFIG_HASH == compute_config_hash(FROZEN_MANIFEST)

    def test_hash_is_sha256_hex(self):
        assert len(FROZEN_CONFIG_HASH) == 64
        int(FROZEN_CONFIG_HASH, 16)  # raises ValueError if not hex

    def test_changing_any_parameter_changes_hash(self):
        manifest = build_frozen_manifest()
        mutated = dict(manifest)
        mutated["atr_multiplier"] = "2.0"
        assert compute_config_hash(mutated) != compute_config_hash(manifest)

    def test_default_argument_uses_module_manifest(self):
        assert compute_config_hash() == FROZEN_CONFIG_HASH


class TestDocumentationIsRecordedButInert:
    def test_historical_documentation_values(self):
        base = HISTORICAL_DOCUMENTATION["base_costs"]
        assert base["compounded_return_pct"] == "13.5723"
        assert base["profit_factor"] == "1.3047"
        assert base["max_drawdown_pct"] == "6.5739"
        assert base["total_trades"] == 109
        assert base["years_positive"] == 4
        assert base["years_total"] == 5

        conservative = HISTORICAL_DOCUMENTATION["conservative_costs"]
        assert conservative["compounded_return_pct"] == "10.5411"
        assert conservative["profit_factor"] == "1.2305"
        assert conservative["max_drawdown_pct"] == "6.9392"

    def test_future_evaluation_criteria(self):
        assert FUTURE_EVALUATION_CRITERIA["first_checkpoint_days"] == 90
        assert FUTURE_EVALUATION_CRITERIA["min_closed_trades_before_real_money"] == 20
        rules = " ".join(FUTURE_EVALUATION_CRITERIA["rules"])
        assert "parameter changes" in rules.lower()
        assert "deleting losing trades" in rules.lower()
        assert "resetting capital" in rules.lower()
        assert "real money" in rules.lower()
