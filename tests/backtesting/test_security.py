"""Security tests for the backtesting module.

Verify that the backtesting engine:
  1. Does NOT import BinanceMarketDataClient
  2. Does NOT use API keys or secrets
  3. Does NOT create real orders
  4. Does NOT modify Order, Trade, Position, or PaperAccount models
  5. Does NOT use leverage, margin, Futures, or short selling
  6. Never transitions automatically to real money
"""

from pathlib import Path

_BACKTESTING_FILES = [
    "app/backtesting/__init__.py",
    "app/backtesting/config.py",
    "app/backtesting/engine.py",
    "app/backtesting/equity_curve.py",
    "app/backtesting/exceptions.py",
    "app/backtesting/execution.py",
    "app/backtesting/exporters.py",
    "app/backtesting/metrics.py",
    "app/backtesting/portfolio.py",
    "app/backtesting/schemas.py",
    "app/backtesting/service.py",
    "app/api/backtesting.py",
    "app/cli/run_backtest.py",
]

_REPO_ROOT = Path(__file__).parent.parent.parent


def _source(rel_path: str) -> str:
    return (_REPO_ROOT / rel_path).read_text(encoding="utf-8")


class TestNoRealMarketClient:
    def test_engine_does_not_import_binance_client(self):
        src = _source("app/backtesting/engine.py")
        assert "BinanceMarketDataClient" not in src

    def test_service_does_not_import_binance_client(self):
        src = _source("app/backtesting/service.py")
        assert "BinanceMarketDataClient" not in src

    def test_no_module_imports_binance_client(self):
        for path in _BACKTESTING_FILES:
            src = _source(path)
            assert (
                "BinanceMarketDataClient" not in src
            ), f"BinanceMarketDataClient imported in {path}"


class TestNoApiKeys:
    def test_no_api_key_usage_in_backtesting(self):
        forbidden = ["api_key", "api_secret", "secret_key", "binance_api"]
        for path in _BACKTESTING_FILES:
            src = _source(path).lower()
            for term in forbidden:
                assert term not in src, f"API key reference {term!r} found in {path}"

    def test_config_has_no_api_key_field(self):
        src = _source("app/backtesting/config.py")
        assert "api_key" not in src.lower()
        assert "api_secret" not in src.lower()


class TestNoRealOrders:
    def test_no_order_creation_in_engine(self):
        src = _source("app/backtesting/engine.py")
        forbidden = ["place_order", "create_order", "submit_order", "Order("]
        for term in forbidden:
            assert term not in src, f"Order creation {term!r} found in engine"

    def test_no_paper_account_modification(self):
        for path in _BACKTESTING_FILES:
            src = _source(path)
            assert "PaperAccount" not in src, f"PaperAccount found in {path}"

    def test_no_position_model_modification(self):
        # The backtesting module may import PositionContext (strategy schema)
        # but must NOT import or modify the Position ORM model
        engine_src = _source("app/backtesting/engine.py")
        # PositionContext import is OK; Position model import is not
        assert "from app.models.position" not in engine_src
        assert "from app.models import position" not in engine_src

    def test_no_trade_model_modification(self):
        for path in _BACKTESTING_FILES:
            src = _source(path)
            assert "from app.models.trade" not in src, f"Trade ORM model imported in {path}"

    def test_no_order_model_modification(self):
        for path in _BACKTESTING_FILES:
            src = _source(path)
            assert "from app.models.order" not in src, f"Order ORM model imported in {path}"


class TestNoLeverageOrShorts:
    def test_no_leverage_in_engine(self):
        src = _source("app/backtesting/engine.py").lower()
        assert "leverage" not in src
        assert "margin" not in src
        assert "futures" not in src

    def test_no_short_selling(self):
        for path in _BACKTESTING_FILES:
            src = _source(path).lower()
            # "short" appears as "short_period" in indicator names; check more specifically
            assert "short_position" not in src, f"short_position found in {path}"
            assert "open_short" not in src, f"open_short found in {path}"
            assert "sell_short" not in src, f"sell_short found in {path}"

    def test_portfolio_has_no_leverage_field(self):
        src = _source("app/backtesting/portfolio.py").lower()
        assert "leverage" not in src
        assert "margin" not in src

    def test_config_has_no_leverage_field(self):
        src = _source("app/backtesting/config.py").lower()
        assert "leverage" not in src


class TestNoRealMoneyTransition:
    def test_no_real_mode_flag_in_backtesting(self):
        for path in _BACKTESTING_FILES:
            src = _source(path).lower()
            forbidden = ["trading_mode", "live_mode", "real_money", "live_trading"]
            for term in forbidden:
                assert term not in src, f"Real money term {term!r} found in {path}"

    def test_paper_warning_present_in_cli(self):
        src = _source("app/cli/run_backtest.py")
        # CLI must display a paper/test warning
        assert "PAPER" in src or "TEST" in src

    def test_paper_warning_present_in_api(self):
        src = _source("app/api/backtesting.py")
        assert "PAPER" in src or "TEST" in src

    def test_no_http_requests_in_backtesting(self):
        for path in _BACKTESTING_FILES:
            src = _source(path)
            assert "httpx" not in src
            assert "requests.get" not in src
            assert "aiohttp" not in src


class TestPaperModeDefault:
    def test_cli_warning_mentions_no_real_money(self):
        src = _source("app/cli/run_backtest.py")
        assert "NO REAL MONEY" in src or "no real money" in src.lower()

    def test_result_schema_has_paper_warning_in_docstring(self):
        src = _source("app/backtesting/schemas.py")
        assert "PAPER" in src or "TEST" in src
