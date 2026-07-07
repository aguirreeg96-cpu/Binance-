"""Decision and signal explainability service for the B_4h forward paper trader.

Maps engine signal states to human-readable decision labels, severities, and
plain-language Spanish explanations.  All failed_conditions are derived from
real stored indicator values — no conditions are ever invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_SIGNAL_TO_DECISION: dict[str, str] = {
    "WAIT": "NO_COMPRAR",
    "BUY_PENDING": "ENTRADA_DETECTADA",
    "LONG": "POSICION_ABIERTA",
    "SELL_PENDING": "CERRAR_POSICION",
    "EXITED": "OPERACION_CERRADA",
    "ERROR_DATA_GAP": "REVISAR_SISTEMA",
}

_DECISION_LABEL: dict[str, str] = {
    "NO_COMPRAR": "NO COMPRAR",
    "ENTRADA_DETECTADA": "ENTRADA DETECTADA",
    "POSICION_ABIERTA": "POSICIÓN ABIERTA",
    "CERRAR_POSICION": "CERRAR POSICIÓN",
    "OPERACION_CERRADA": "OPERACIÓN CERRADA",
    "REVISAR_SISTEMA": "REVISAR SISTEMA",
}

_DECISION_SEVERITY: dict[str, str] = {
    "NO_COMPRAR": "neutral",
    "ENTRADA_DETECTADA": "success",
    "POSICION_ABIERTA": "info",
    "CERRAR_POSICION": "warning",
    "OPERACION_CERRADA": "info",
    "REVISAR_SISTEMA": "danger",
}

_EXPLANATIONS: dict[str, str] = {
    "NO_COMPRAR": (
        "La estrategia analizó la última vela cerrada de 4h y no encontró una ruptura "
        "válida del canal Donchian. No se generó ninguna señal de entrada."
    ),
    "ENTRADA_DETECTADA": (
        "Se detectó una ruptura del canal Donchian al alza con la EMA 200 ascendente. "
        "Se programó una orden de compra para ejecutarse en la próxima apertura de vela."
    ),
    "POSICION_ABIERTA": (
        "Hay una posición virtual LONG abierta. La estrategia monitorea el precio "
        "frente al canal Donchian de salida y el trailing stop."
    ),
    "CERRAR_POSICION": (
        "El precio cerró por debajo del canal Donchian de salida. Se programó una "
        "orden de venta para ejecutarse en la próxima apertura de vela."
    ),
    "OPERACION_CERRADA": (
        "La operación anterior fue cerrada (por señal de salida Donchian o stop loss). "
        "El sistema espera la próxima señal de entrada."
    ),
    "REVISAR_SISTEMA": (
        "Se detectó un error en los datos de mercado o un gap en la información. "
        "Verificar la conectividad y la disponibilidad de datos históricos."
    ),
}

_HEALTH_DEGRADED_EXPLANATION = (
    "El sistema de salud reporta un estado degradado o de error. "
    "Verificar la conectividad, la base de datos y el proceso del paper trader."
)


@dataclass(frozen=True)
class DecisionSnapshot:
    """Immutable snapshot of the current trading decision for display."""

    current_decision: str
    action_label: str
    action_severity: str
    plain_language_explanation: str
    failed_conditions: list[str]
    is_entry_signal: bool
    is_exit_signal: bool
    has_open_position: bool
    raw_market_price: str | None
    entry_donchian_level: str | None
    exit_donchian_level: str | None
    atr: str | None
    equity: str | None
    price_vs_entry_diff_usd: str | None
    price_vs_entry_diff_pct: str | None


def _fmt(v: Decimal, dp: int = 2) -> str:
    return f"{v:.{dp}f}"


def _compute_failed_conditions(
    signal: str,
    raw_market_price: Decimal | None,
    entry_donchian_level: Decimal | None,
    ema_200: Decimal | None,
    ema_slope: Decimal | None,
) -> list[str]:
    """Derive unmet entry conditions for WAIT signals from stored indicator data.

    Returns an empty list for non-WAIT signals or when data is absent.
    """
    if signal != "WAIT":
        return []

    if raw_market_price is None or entry_donchian_level is None:
        return ["Historial insuficiente para calcular el nivel Donchian de entrada."]

    conditions: list[str] = []

    if raw_market_price <= entry_donchian_level:
        diff = raw_market_price - entry_donchian_level
        pct = diff / entry_donchian_level * Decimal("100")
        conditions.append(
            f"Precio ({_fmt(raw_market_price)}) ≤ Donchian entrada "
            f"({_fmt(entry_donchian_level)}): "
            f"diferencia {_fmt(diff)} USDT ({_fmt(pct)}%)"
        )

    if ema_200 is not None and raw_market_price <= ema_200:
        conditions.append(f"Precio ({_fmt(raw_market_price)}) ≤ EMA 200 ({_fmt(ema_200)})")

    if ema_slope is not None and ema_slope <= Decimal("0"):
        conditions.append(f"Pendiente EMA 200 no alcêsta ({_fmt(ema_slope, 6)})")

    if not conditions:
        conditions.append("Condiciones no cumplidas o historial insuficiente para generar señal.")

    return conditions


def compute_decision(
    signal: str | None,
    reasons: list[str],
    has_open_position: bool,
    health_status: str | None,
    raw_market_price: Decimal | None,
    entry_donchian_level: Decimal | None,
    exit_donchian_level: Decimal | None,
    ema_200: Decimal | None,
    ema_slope: Decimal | None,
    atr: Decimal | None,
    equity: Decimal | None,
) -> DecisionSnapshot:
    """Map engine signal state to a displayable DecisionSnapshot.

    health_status values: ACTIVE | STOPPED | ERROR | DEGRADED | None.
    ERROR or DEGRADED overrides WAIT/None/ERROR_DATA_GAP signals to REVISAR_SISTEMA.
    """
    effective_signal = signal or "WAIT"

    if health_status in ("ERROR", "DEGRADED") and effective_signal in (
        "WAIT",
        "ERROR_DATA_GAP",
    ):
        decision = "REVISAR_SISTEMA"
    else:
        decision = _SIGNAL_TO_DECISION.get(effective_signal, "NO_COMPRAR")

    label = _DECISION_LABEL[decision]
    severity = _DECISION_SEVERITY[decision]

    if decision == "REVISAR_SISTEMA" and health_status in ("ERROR", "DEGRADED"):
        explanation = _HEALTH_DEGRADED_EXPLANATION
    else:
        explanation = _EXPLANATIONS.get(decision, "")

    failed_conditions = _compute_failed_conditions(
        effective_signal,
        raw_market_price,
        entry_donchian_level,
        ema_200,
        ema_slope,
    )

    price_vs_entry_diff_usd: str | None = None
    price_vs_entry_diff_pct: str | None = None
    if (
        effective_signal == "WAIT"
        and raw_market_price is not None
        and entry_donchian_level is not None
    ):
        diff_usd = raw_market_price - entry_donchian_level
        diff_pct = diff_usd / entry_donchian_level * Decimal("100")
        price_vs_entry_diff_usd = _fmt(diff_usd)
        price_vs_entry_diff_pct = _fmt(diff_pct)

    def _d(v: Decimal | None, dp: int = 2) -> str | None:
        return _fmt(v, dp) if v is not None else None

    return DecisionSnapshot(
        current_decision=decision,
        action_label=label,
        action_severity=severity,
        plain_language_explanation=explanation,
        failed_conditions=failed_conditions,
        is_entry_signal=effective_signal == "BUY_PENDING",
        is_exit_signal=effective_signal == "SELL_PENDING",
        has_open_position=has_open_position,
        raw_market_price=_d(raw_market_price),
        entry_donchian_level=_d(entry_donchian_level),
        exit_donchian_level=_d(exit_donchian_level),
        atr=_d(atr),
        equity=_d(equity),
        price_vs_entry_diff_usd=price_vs_entry_diff_usd,
        price_vs_entry_diff_pct=price_vs_entry_diff_pct,
    )
