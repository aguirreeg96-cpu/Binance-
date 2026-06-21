"""Volume SMA and volume ratio.

volume_ratio = current_volume / volume_sma

Special cases:
  volume_sma == 0 and volume == 0  → ratio = 0      (no activity)
  volume_sma == 0 and volume  > 0  → ratio = None   (avoid division by zero / infinity)
  During warm-up                   → both sma and ratio are None
"""

from decimal import Decimal

from app.indicators.sma import compute_sma

_ZERO = Decimal("0")


def compute_volume_indicators(
    volumes: list[Decimal],
    period: int,
) -> tuple[list[Decimal | None], list[Decimal | None]]:
    """Return (volume_sma, volume_ratio) aligned with the input volumes."""
    sma_values = compute_sma(volumes, period)
    ratios: list[Decimal | None] = []

    for vol, sma in zip(volumes, sma_values, strict=True):
        if sma is None:
            ratios.append(None)
        elif sma == _ZERO:
            ratios.append(_ZERO if vol == _ZERO else None)
        else:
            ratios.append(vol / sma)

    return sma_values, ratios
