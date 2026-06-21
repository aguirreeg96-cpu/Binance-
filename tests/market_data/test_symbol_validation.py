"""Tests for _validate_symbol in HistoricalDataService.

Covers the Binance exchangeInfo format evolution:
  - Modern: isSpotTradingAllowed (bool) is authoritative; permissions may be [].
  - Transition: permissionSets contains SPOT groups; permissions still absent.
  - Legacy: permissions list contains "SPOT"; isSpotTradingAllowed absent.
"""

from unittest.mock import AsyncMock

import pytest

from app.market_data.exceptions import InvalidSymbolError, MarketDataError
from app.market_data.historical_service import HistoricalDataService

_SYMBOL = "BTCUSDT"


def _make_service(exchange_info_return=None, exchange_info_side_effect=None):
    client = AsyncMock()
    if exchange_info_side_effect is not None:
        client.get_exchange_info.side_effect = exchange_info_side_effect
    else:
        client.get_exchange_info.return_value = exchange_info_return
    return HistoricalDataService(client=client, max_requests=10)


def _sym(**kwargs):
    """Build a minimal symbol dict; caller can override any field."""
    base = {"symbol": _SYMBOL, "status": "TRADING"}
    base.update(kwargs)
    return {"symbols": [base]}


# ---------------------------------------------------------------------------
# Modern format: isSpotTradingAllowed present
# ---------------------------------------------------------------------------


class TestIsSpotTradingAllowed:
    @pytest.mark.asyncio
    async def test_true_with_empty_permissions_accepted(self):
        """Real-world Binance response: permissions=[], isSpotTradingAllowed=True."""
        info = _sym(isSpotTradingAllowed=True, permissions=[], permissionSets=[["SPOT", "MARGIN"]])
        service = _make_service(exchange_info_return=info)
        await service._validate_symbol(_SYMBOL)  # must not raise

    @pytest.mark.asyncio
    async def test_false_rejected_even_if_permissionsets_has_spot(self):
        """isSpotTradingAllowed=False is authoritative — reject regardless of other fields."""
        info = _sym(
            isSpotTradingAllowed=False,
            permissions=["SPOT"],
            permissionSets=[["SPOT"]],
        )
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="SPOT permission"):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_true_without_permissions_or_sets_accepted(self):
        """isSpotTradingAllowed=True alone is enough."""
        info = _sym(isSpotTradingAllowed=True)
        service = _make_service(exchange_info_return=info)
        await service._validate_symbol(_SYMBOL)  # must not raise


# ---------------------------------------------------------------------------
# Transition format: permissionSets present, isSpotTradingAllowed absent
# ---------------------------------------------------------------------------


class TestPermissionSets:
    @pytest.mark.asyncio
    async def test_spot_in_permission_sets_accepted(self):
        """permissionSets contains a group with SPOT — should be accepted."""
        info = _sym(permissions=[], permissionSets=[["SPOT", "MARGIN"]])
        service = _make_service(exchange_info_return=info)
        await service._validate_symbol(_SYMBOL)  # must not raise

    @pytest.mark.asyncio
    async def test_no_spot_in_any_permission_set_rejected(self):
        info = _sym(permissions=[], permissionSets=[["MARGIN", "LEVERAGED"]])
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="SPOT permission"):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_empty_permission_sets_rejected(self):
        info = _sym(permissions=[], permissionSets=[])
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="SPOT permission"):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_permission_sets_absent_and_permissions_empty_rejected(self):
        """Neither field present and permissions=[]: no evidence of SPOT."""
        info = _sym(permissions=[])
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="SPOT permission"):
            await service._validate_symbol(_SYMBOL)


# ---------------------------------------------------------------------------
# Legacy format: only permissions list, no isSpotTradingAllowed
# ---------------------------------------------------------------------------


class TestLegacyPermissions:
    @pytest.mark.asyncio
    async def test_spot_in_legacy_permissions_accepted(self):
        info = _sym(permissions=["SPOT"])
        service = _make_service(exchange_info_return=info)
        await service._validate_symbol(_SYMBOL)  # must not raise

    @pytest.mark.asyncio
    async def test_spot_and_margin_in_legacy_permissions_accepted(self):
        info = _sym(permissions=["SPOT", "MARGIN"])
        service = _make_service(exchange_info_return=info)
        await service._validate_symbol(_SYMBOL)  # must not raise

    @pytest.mark.asyncio
    async def test_only_margin_in_legacy_permissions_rejected(self):
        info = _sym(permissions=["MARGIN"])
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="SPOT permission"):
            await service._validate_symbol(_SYMBOL)


# ---------------------------------------------------------------------------
# Status checks
# ---------------------------------------------------------------------------


class TestSymbolStatus:
    @pytest.mark.asyncio
    async def test_halted_status_rejected(self):
        info = _sym(status="HALT", permissions=["SPOT"])
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="status"):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_pre_trading_status_rejected(self):
        info = _sym(status="PRE_TRADING", permissions=["SPOT"])
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="status"):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_break_status_rejected(self):
        info = _sym(status="BREAK", isSpotTradingAllowed=True)
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="status"):
            await service._validate_symbol(_SYMBOL)


# ---------------------------------------------------------------------------
# exchangeInfo error paths
# ---------------------------------------------------------------------------


class TestExchangeInfoErrors:
    @pytest.mark.asyncio
    async def test_market_data_error_raises_invalid_symbol(self):
        service = _make_service(
            exchange_info_side_effect=MarketDataError("HTTP 400: Invalid symbol")
        )
        with pytest.raises(InvalidSymbolError):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_empty_symbols_list_rejected(self):
        service = _make_service(exchange_info_return={"symbols": []})
        with pytest.raises(InvalidSymbolError, match="not found"):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_symbol_mismatch_rejected(self):
        info = {"symbols": [{"symbol": "ETHUSDT", "status": "TRADING", "permissions": ["SPOT"]}]}
        service = _make_service(exchange_info_return=info)
        with pytest.raises(InvalidSymbolError, match="mismatch"):
            await service._validate_symbol(_SYMBOL)

    @pytest.mark.asyncio
    async def test_unexpected_extra_fields_do_not_raise_type_error(self):
        """Extra unknown fields in the response must not cause TypeError."""
        info = _sym(
            isSpotTradingAllowed=True,
            permissions=[],
            permissionSets=[["SPOT"]],
            quoteAsset="USDT",
            baseAsset="BTC",
            filters=[{"filterType": "PRICE_FILTER"}],
            orderTypes=["LIMIT", "MARKET"],
        )
        service = _make_service(exchange_info_return=info)
        await service._validate_symbol(_SYMBOL)  # must not raise
