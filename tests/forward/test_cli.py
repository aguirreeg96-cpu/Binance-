"""Stage 6.1 tests: run_paper_breakout CLI — argument parsing and safety guards."""

from __future__ import annotations

import argparse

import pytest

from app.cli.run_paper_breakout import _build_parser


class TestArgumentParser:
    def test_requires_mode_flag(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_once_flag_accepted(self):
        parser = _build_parser()
        args = parser.parse_args(["--once"])
        assert args.once is True
        assert args.continuous is False

    def test_continuous_flag_accepted(self):
        parser = _build_parser()
        args = parser.parse_args(["--continuous"])
        assert args.continuous is True
        assert args.once is False

    def test_once_and_continuous_are_mutually_exclusive(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--once", "--continuous"])

    def test_poll_interval_seconds_default_is_none(self):
        parser = _build_parser()
        args = parser.parse_args(["--continuous"])
        assert args.poll_interval_seconds is None

    def test_poll_interval_seconds_accepted(self):
        parser = _build_parser()
        args = parser.parse_args(["--continuous", "--poll-interval-seconds", "60"])
        assert args.poll_interval_seconds == 60


class TestOneShotMode:
    @pytest.mark.asyncio
    async def test_run_one_cycle_uses_mocked_client(self):
        """One-shot cycle completes when DB, sync, and engine are all mocked.

        `_run_one_cycle` imports everything locally, so we patch the source
        modules rather than the CLI module itself.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_launch = MagicMock()
        mock_launch.id = 1
        mock_launch.status = "ACTIVE"

        with (
            patch("app.database.run_migrations"),
            patch("app.database.SessionLocal"),
            patch("app.market_data.client.BinanceMarketDataClient", return_value=AsyncMock()),
            patch("app.forward.engine.start_or_resume_launch", return_value=mock_launch),
            patch(
                "app.forward.sync.sync_forward_candles", new_callable=AsyncMock, return_value=None
            ),
            patch("app.forward.engine.ForwardPaperEngine") as MockEngine,
        ):
            MockEngine.return_value.run_cycle.return_value = []

            from app.cli.run_paper_breakout import _run_one_cycle

            args = argparse.Namespace(once=True, continuous=False, poll_interval_seconds=None)
            rc = await _run_one_cycle(args)

        assert rc == 0


class TestArchitectureGuard:
    def test_cli_does_not_place_real_orders(self):
        import app.cli.run_paper_breakout as cli_mod

        with open(cli_mod.__file__, encoding="utf-8") as f:
            src = f.read()
        for forbidden in ("place_order", "cancel_order", "api_key", "api_secret"):
            assert forbidden not in src

    def test_cli_warns_about_paper_mode(self):
        import app.cli.run_paper_breakout as cli_mod

        with open(cli_mod.__file__, encoding="utf-8") as f:
            src = f.read()
        assert "PAPER" in src
        assert "No real" in src or "no real" in src.lower()

    def test_cli_has_no_real_money_transition(self):
        """The CLI must never contain any mention of real order placement."""
        import app.cli.run_paper_breakout as cli_mod

        with open(cli_mod.__file__, encoding="utf-8") as f:
            src = f.read()
        assert "real_order" not in src
        assert "place_order" not in src
