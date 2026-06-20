"""Tests for KlineData parsing and OHLCV validation."""

from decimal import Decimal

import pytest

from app.market_data.kline_parser import KlineData, _to_decimal
from app.market_data.exceptions import KlineParseError, KlineValidationError


def _raw(
    open_time=1700000000000,
    o="35000.00", h="35500.00", l="34800.00", c="35200.00",
    vol="100.5", close_time=1700003600000 - 1,
    quote_vol="3538100.00", trades=1500,
    tb_base="50.25", tb_quote="1769050.00",
) -> list:
    return [open_time, o, h, l, c, vol, close_time, quote_vol, trades, tb_base, tb_quote, "0"]


class TestKlineParser:
    def test_parse_valid_kline(self):
        kd = KlineData.from_raw(_raw(), "BTCUSDT", "1h")
        assert kd.symbol == "BTCUSDT"
        assert kd.interval == "1h"
        assert kd.open_time == 1700000000000
        assert kd.number_of_trades == 1500

    def test_symbol_normalized_to_uppercase(self):
        kd = KlineData.from_raw(_raw(), "btcusdt", "1h")
        assert kd.symbol == "BTCUSDT"

    def test_all_price_fields_are_decimal(self):
        kd = KlineData.from_raw(_raw(), "BTCUSDT", "1h")
        for field_name in ("open", "high", "low", "close", "volume",
                           "quote_asset_volume", "taker_buy_base_volume",
                           "taker_buy_quote_volume"):
            assert isinstance(getattr(kd, field_name), Decimal), (
                f"{field_name} should be Decimal"
            )

    def test_no_float_contamination(self):
        # Values that are lossy as float must survive as exact Decimal
        raw = _raw(o="0.00000001", h="0.00000002", l="0.00000001", c="0.00000001")
        kd = KlineData.from_raw(raw, "SHIBUSDT", "1m")
        assert kd.open == Decimal("0.00000001")
        assert kd.high == Decimal("0.00000002")

    def test_large_volume_precision(self):
        raw = _raw(vol="123456789.123456789")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        assert kd.volume == Decimal("123456789.123456789")

    def test_too_few_fields_raises(self):
        with pytest.raises(KlineParseError):
            KlineData.from_raw([1700000000000, "100", "101"], "X", "1m")

    def test_non_numeric_price_raises(self):
        raw = _raw(o="not_a_number")
        with pytest.raises(KlineParseError):
            KlineData.from_raw(raw, "BTCUSDT", "1h")

    def test_all_fields_stored(self):
        kd = KlineData.from_raw(_raw(), "BTCUSDT", "1h")
        assert kd.quote_asset_volume == Decimal("3538100.00")
        assert kd.taker_buy_base_volume == Decimal("50.25")
        assert kd.taker_buy_quote_volume == Decimal("1769050.00")
        assert kd.close_time == 1700003599999

    def test_open_time_and_close_time_are_int(self):
        kd = KlineData.from_raw(_raw(), "BTCUSDT", "1h")
        assert isinstance(kd.open_time, int)
        assert isinstance(kd.close_time, int)


class TestKlineValidation:
    def test_valid_kline_passes(self):
        KlineData.from_raw(_raw(), "BTCUSDT", "1h").validate()  # no exception

    def test_high_less_than_open_rejected(self):
        raw = _raw(o="35500.00", h="35000.00", l="34800.00", c="35200.00")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="high.*<.*open"):
            kd.validate()

    def test_high_less_than_close_rejected(self):
        raw = _raw(o="35000.00", h="35100.00", l="34800.00", c="35200.00")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="high.*<.*close"):
            kd.validate()

    def test_high_less_than_low_rejected(self):
        raw = _raw(o="35000.00", h="34000.00", l="34800.00", c="35000.00")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="high.*<.*low"):
            kd.validate()

    def test_low_greater_than_open_rejected(self):
        raw = _raw(o="34000.00", h="35500.00", l="34800.00", c="35000.00")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="low.*>.*open"):
            kd.validate()

    def test_low_greater_than_close_rejected(self):
        raw = _raw(o="35000.00", h="35500.00", l="35100.00", c="34900.00")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="low.*>.*close"):
            kd.validate()

    def test_negative_price_rejected(self):
        raw = _raw(o="-1.00")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="negative"):
            kd.validate()

    def test_negative_volume_rejected(self):
        raw = _raw(vol="-100")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="negative"):
            kd.validate()

    def test_open_time_gte_close_time_rejected(self):
        raw = _raw(open_time=1700003600000, close_time=1700000000000)
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        with pytest.raises(KlineValidationError, match="open_time.*>=.*close_time"):
            kd.validate()

    def test_zero_volume_is_valid(self):
        raw = _raw(vol="0", tb_base="0", tb_quote="0", quote_vol="0")
        kd = KlineData.from_raw(raw, "BTCUSDT", "1h")
        kd.validate()  # zero volume is valid (no trades in period)


class TestToDecimal:
    def test_string_input(self):
        assert _to_decimal("3.14159", "f") == Decimal("3.14159")

    def test_int_input(self):
        assert _to_decimal(42, "f") == Decimal("42")

    def test_invalid_raises(self):
        with pytest.raises(KlineParseError):
            _to_decimal("not_a_number", "price")
