"""Security tests: verify the strategy module has no trading execution capability."""

import ast
from pathlib import Path


def _imports_in_module(module_path: Path) -> set[str]:
    """Return all top-level names imported in a Python source file."""
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _all_strategy_source_files() -> list[Path]:
    root = Path("app/strategy")
    return list(root.glob("*.py"))


def _all_strategy_source_text() -> str:
    return "\n".join(f.read_text(encoding="utf-8") for f in _all_strategy_source_files())


class TestNoTradingClientImports:
    def test_strategy_does_not_import_binance_client(self):
        text = _all_strategy_source_text()
        assert "BinanceMarketDataClient" not in text
        assert "from app.market_data.client" not in text

    def test_strategy_does_not_import_trading_client(self):
        text = _all_strategy_source_text()
        forbidden = ["create_order", "cancel_order", "withdraw", "TradingClient"]
        for term in forbidden:
            assert term not in text, f"Found forbidden term {term!r} in strategy source"

    def test_strategy_does_not_use_api_keys(self):
        text = _all_strategy_source_text()
        assert "api_key" not in text.lower().replace("_", "")
        assert "API_KEY" not in text
        assert "BINANCE_API_KEY" not in text


class TestNoOrderExecution:
    def test_no_create_order_in_strategy(self):
        text = _all_strategy_source_text()
        assert "create_order" not in text

    def test_no_cancel_order_in_strategy(self):
        text = _all_strategy_source_text()
        assert "cancel_order" not in text

    def test_no_withdraw_in_strategy(self):
        text = _all_strategy_source_text()
        assert "withdraw" not in text


class TestNoDatabaseWrites:
    def test_strategy_does_not_import_order_model(self):
        text = _all_strategy_source_text()
        assert "from app.models.order" not in text
        assert "import Order" not in text

    def test_strategy_does_not_import_trade_model(self):
        text = _all_strategy_source_text()
        assert "from app.models.trade" not in text
        assert "import Trade" not in text

    def test_strategy_engine_does_not_write_positions(self):
        text = (Path("app/strategy/engine.py")).read_text()
        assert "session.add" not in text
        assert "session.commit" not in text
        assert "session.flush" not in text

    def test_strategy_rules_does_not_write_positions(self):
        text = (Path("app/strategy/rules.py")).read_text()
        assert "session.add" not in text
        assert "session.commit" not in text


class TestNoShortSelling:
    def test_no_short_in_strategy_source(self):
        import re

        text = _all_strategy_source_text()
        # "short" must not appear as a trading concept.
        # Allowed: indicator period names and comments that explicitly document
        # the absence of short-selling.
        _ALLOWED = (
            "ema_short",
            "sma_short",
            "EMA-short",
            "SMA-short",
            "never",   # "never opening a short", "never generates short-sell"
            "Spot-only",
            "short-sell",  # appears in "no short-sell signals" documentation
        )
        for line in text.splitlines():
            if not re.search(r"\bshort\b", line, re.IGNORECASE):
                continue
            if any(allowed in line for allowed in _ALLOWED):
                continue
            raise AssertionError(
                f"Found potential short-selling reference: {line.strip()!r}"
            )

    def test_strategy_action_has_no_short_enum(self):
        from app.strategy.schemas import StrategyAction

        actions = {a.value for a in StrategyAction}
        assert "SHORT" not in actions
        assert "short" not in actions


class TestNoFuturesOrMargin:
    def test_no_futures_in_strategy(self):
        text = _all_strategy_source_text()
        assert "futures" not in text.lower()

    def test_no_margin_in_strategy(self):
        text = _all_strategy_source_text()
        assert "margin" not in text.lower()

    def test_no_leverage_in_strategy(self):
        text = _all_strategy_source_text()
        assert "leverage" not in text.lower()


class TestStrategyCoreIsPure:
    def test_engine_module_does_not_import_session(self):
        text = (Path("app/strategy/engine.py")).read_text()
        assert "Session" not in text
        assert "get_db" not in text

    def test_rules_module_does_not_import_session(self):
        text = (Path("app/strategy/rules.py")).read_text()
        assert "Session" not in text
        assert "sqlalchemy" not in text

    def test_schemas_module_does_not_import_session(self):
        text = (Path("app/strategy/schemas.py")).read_text()
        assert "Session" not in text
        assert "sqlalchemy" not in text
