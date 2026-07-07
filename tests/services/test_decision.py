"""Unit tests for app.services.decision — signal explainability logic."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.services.decision import DecisionSnapshot, compute_decision


def _dec(signal, **kwargs):
    defaults = {
        "signal": signal,
        "reasons": [],
        "has_open_position": False,
        "health_status": "ACTIVE",
        "raw_market_price": None,
        "entry_donchian_level": None,
        "exit_donchian_level": None,
        "ema_200": None,
        "ema_slope": None,
        "atr": None,
        "equity": None,
    }
    defaults.update(kwargs)
    return compute_decision(**defaults)


# ---------------------------------------------------------------------------
# Signal → decision mapping
# ---------------------------------------------------------------------------


class TestSignalMapping:
    def test_wait_maps_to_no_comprar(self):
        snap = _dec("WAIT")
        assert snap.current_decision == "NO_COMPRAR"
        assert snap.action_label == "NO COMPRAR"
        assert snap.action_severity == "neutral"

    def test_buy_pending_maps_to_entrada_detectada(self):
        snap = _dec("BUY_PENDING")
        assert snap.current_decision == "ENTRADA_DETECTADA"
        assert snap.action_label == "ENTRADA DETECTADA"
        assert snap.action_severity == "success"

    def test_long_maps_to_posicion_abierta(self):
        snap = _dec("LONG")
        assert snap.current_decision == "POSICION_ABIERTA"
        assert snap.action_label == "POSICIÓN ABIERTA"
        assert snap.action_severity == "info"

    def test_sell_pending_maps_to_cerrar_posicion(self):
        snap = _dec("SELL_PENDING")
        assert snap.current_decision == "CERRAR_POSICION"
        assert snap.action_label == "CERRAR POSICIÓN"
        assert snap.action_severity == "warning"

    def test_exited_maps_to_operacion_cerrada(self):
        snap = _dec("EXITED")
        assert snap.current_decision == "OPERACION_CERRADA"
        assert snap.action_label == "OPERACIÓN CERRADA"
        assert snap.action_severity == "info"

    def test_error_data_gap_maps_to_revisar_sistema(self):
        snap = _dec("ERROR_DATA_GAP")
        assert snap.current_decision == "REVISAR_SISTEMA"
        assert snap.action_label == "REVISAR SISTEMA"
        assert snap.action_severity == "danger"

    def test_none_signal_maps_to_no_comprar(self):
        snap = _dec(None)
        assert snap.current_decision == "NO_COMPRAR"

    def test_unknown_signal_maps_to_no_comprar(self):
        snap = _dec("UNKNOWN_SIGNAL_XYZ")
        assert snap.current_decision == "NO_COMPRAR"


# ---------------------------------------------------------------------------
# Health-status override
# ---------------------------------------------------------------------------


class TestHealthStatusOverride:
    def test_error_health_overrides_wait_to_revisar(self):
        snap = _dec("WAIT", health_status="ERROR")
        assert snap.current_decision == "REVISAR_SISTEMA"

    def test_degraded_health_overrides_wait_to_revisar(self):
        snap = _dec("WAIT", health_status="DEGRADED")
        assert snap.current_decision == "REVISAR_SISTEMA"

    def test_error_health_overrides_none_signal_to_revisar(self):
        snap = _dec(None, health_status="ERROR")
        assert snap.current_decision == "REVISAR_SISTEMA"

    def test_error_health_overrides_error_data_gap_to_revisar(self):
        snap = _dec("ERROR_DATA_GAP", health_status="ERROR")
        assert snap.current_decision == "REVISAR_SISTEMA"

    def test_error_health_does_not_override_buy_pending(self):
        snap = _dec("BUY_PENDING", health_status="ERROR")
        assert snap.current_decision == "ENTRADA_DETECTADA"

    def test_stopped_health_does_not_override_wait(self):
        snap = _dec("WAIT", health_status="STOPPED")
        assert snap.current_decision == "NO_COMPRAR"


# ---------------------------------------------------------------------------
# Entry / exit signal flags
# ---------------------------------------------------------------------------


class TestSignalFlags:
    def test_buy_pending_is_entry_signal(self):
        snap = _dec("BUY_PENDING")
        assert snap.is_entry_signal is True
        assert snap.is_exit_signal is False

    def test_sell_pending_is_exit_signal(self):
        snap = _dec("SELL_PENDING")
        assert snap.is_exit_signal is True
        assert snap.is_entry_signal is False

    def test_wait_is_neither(self):
        snap = _dec("WAIT")
        assert snap.is_entry_signal is False
        assert snap.is_exit_signal is False

    def test_long_is_neither(self):
        snap = _dec("LONG", has_open_position=True)
        assert snap.is_entry_signal is False
        assert snap.is_exit_signal is False


# ---------------------------------------------------------------------------
# Failed conditions — WAIT with numeric data
# ---------------------------------------------------------------------------


class TestFailedConditions:
    def test_no_failed_conditions_for_buy_pending(self):
        snap = _dec("BUY_PENDING")
        assert snap.failed_conditions == []

    def test_no_failed_conditions_for_long(self):
        snap = _dec("LONG", has_open_position=True)
        assert snap.failed_conditions == []

    def test_insufficient_history_when_no_price(self):
        snap = _dec("WAIT")
        assert len(snap.failed_conditions) == 1
        assert "insuficiente" in snap.failed_conditions[0].lower()

    def test_price_below_donchian_reported(self):
        snap = _dec(
            "WAIT",
            raw_market_price=Decimal("95000"),
            entry_donchian_level=Decimal("100000"),
        )
        conds = snap.failed_conditions
        assert any("Donchian entrada" in c for c in conds)
        assert any("95000" in c for c in conds)

    def test_price_above_donchian_no_price_condition(self):
        snap = _dec(
            "WAIT",
            raw_market_price=Decimal("105000"),
            entry_donchian_level=Decimal("100000"),
        )
        conds = snap.failed_conditions
        assert not any("Donchian entrada" in c for c in conds)

    def test_price_below_ema_reported(self):
        snap = _dec(
            "WAIT",
            raw_market_price=Decimal("90000"),
            entry_donchian_level=Decimal("100000"),
            ema_200=Decimal("95000"),
        )
        conds = snap.failed_conditions
        assert any("EMA 200" in c for c in conds)

    def test_negative_ema_slope_reported(self):
        snap = _dec(
            "WAIT",
            raw_market_price=Decimal("105000"),
            entry_donchian_level=Decimal("100000"),
            ema_200=Decimal("90000"),
            ema_slope=Decimal("-1.5"),
        )
        conds = snap.failed_conditions
        assert any("Pendiente" in c for c in conds)

    def test_all_conditions_pass_but_wait_gets_fallback_message(self):
        # price above Donchian, above EMA, slope positive — still WAIT
        snap = _dec(
            "WAIT",
            raw_market_price=Decimal("105000"),
            entry_donchian_level=Decimal("100000"),
            ema_200=Decimal("90000"),
            ema_slope=Decimal("2.0"),
        )
        conds = snap.failed_conditions
        assert len(conds) == 1
        assert "insuficiente" in conds[0].lower() or "no cumplidas" in conds[0].lower()


# ---------------------------------------------------------------------------
# Price vs entry diff fields
# ---------------------------------------------------------------------------


class TestPriceVsDiff:
    def test_diff_computed_for_wait(self):
        snap = _dec(
            "WAIT",
            raw_market_price=Decimal("95000"),
            entry_donchian_level=Decimal("100000"),
        )
        assert snap.price_vs_entry_diff_usd is not None
        assert snap.price_vs_entry_diff_pct is not None
        assert float(snap.price_vs_entry_diff_usd) == pytest.approx(-5000.0, rel=1e-3)
        assert float(snap.price_vs_entry_diff_pct) == pytest.approx(-5.0, rel=1e-3)

    def test_diff_positive_when_price_above_entry(self):
        snap = _dec(
            "WAIT",
            raw_market_price=Decimal("105000"),
            entry_donchian_level=Decimal("100000"),
        )
        assert float(snap.price_vs_entry_diff_usd) == pytest.approx(5000.0, rel=1e-3)
        assert float(snap.price_vs_entry_diff_pct) == pytest.approx(5.0, rel=1e-3)

    def test_diff_none_when_no_price(self):
        snap = _dec("WAIT")
        assert snap.price_vs_entry_diff_usd is None
        assert snap.price_vs_entry_diff_pct is None

    def test_diff_none_for_non_wait_signal(self):
        snap = _dec(
            "BUY_PENDING",
            raw_market_price=Decimal("105000"),
            entry_donchian_level=Decimal("100000"),
        )
        assert snap.price_vs_entry_diff_usd is None
        assert snap.price_vs_entry_diff_pct is None


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_decision_snapshot(self):
        snap = _dec("WAIT")
        assert isinstance(snap, DecisionSnapshot)

    def test_snapshot_is_frozen(self):
        snap = _dec("WAIT")
        with pytest.raises((AttributeError, TypeError)):
            snap.current_decision = "other"  # type: ignore[misc]

    def test_explanations_are_non_empty(self):
        for signal in ("WAIT", "BUY_PENDING", "LONG", "SELL_PENDING", "EXITED", "ERROR_DATA_GAP"):
            snap = _dec(signal)
            assert len(snap.plain_language_explanation) > 10, f"No explanation for {signal}"
