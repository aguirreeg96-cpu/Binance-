"""Stage 6.1 tests: run_paper_breakout CLI — argument parsing and safety guards."""

from __future__ import annotations

import argparse
import inspect

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
            patch("app.database.SessionLocal"),
            patch("app.market_data.client.BinanceMarketDataClient", return_value=AsyncMock()),
            patch("app.forward.engine.start_or_resume_launch", return_value=mock_launch),
            patch(
                "app.forward.sync.sync_forward_candles", new_callable=AsyncMock, return_value=None
            ),
            patch("app.forward.engine.ForwardPaperEngine") as MockEngine,
        ):
            MockEngine.return_value.run_cycle.return_value = []
            # last_report=None suppresses the diagnostic print block.
            MockEngine.return_value.last_report = None

            from app.cli.run_paper_breakout import _run_one_cycle

            args = argparse.Namespace(once=True, continuous=False, poll_interval_seconds=None)
            rc = await _run_one_cycle(args)

        assert rc == 0

    @pytest.mark.asyncio
    async def test_continuous_runs_multiple_cycles_without_stopping(self):
        """_run_continuous() calls _run_one_cycle() repeatedly until interrupted."""
        from unittest.mock import patch

        from app.cli.run_paper_breakout import _run_continuous

        call_count = {"n": 0}

        async def fake_cycle(args):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise KeyboardInterrupt
            return 0

        with patch("app.cli.run_paper_breakout._run_one_cycle", side_effect=fake_cycle):
            args = argparse.Namespace(once=False, continuous=True, poll_interval_seconds=0)
            rc = await _run_continuous(args)

        assert rc == 0
        assert call_count["n"] == 2


class TestMigrationPolicy:
    """run_migrations() must run at startup (once), not on every poll cycle."""

    def test_run_one_cycle_does_not_call_run_migrations(self):
        """Static analysis: run_migrations must not appear inside _run_one_cycle."""
        import app.cli.run_paper_breakout as cli_mod

        src = inspect.getsource(cli_mod._run_one_cycle)
        assert "run_migrations" not in src, (
            "_run_one_cycle must not call run_migrations(); "
            "migrations belong in main() to avoid Alembic noise on every poll."
        )

    def test_main_calls_run_migrations_before_event_loop(self):
        """Static analysis: run_migrations must appear in main()."""
        import app.cli.run_paper_breakout as cli_mod

        src = inspect.getsource(cli_mod.main)
        assert "run_migrations" in src

    @pytest.mark.asyncio
    async def test_repeated_cycles_do_not_trigger_migrations(self):
        """Two consecutive _run_one_cycle calls never call run_migrations()."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_launch = MagicMock()
        mock_launch.id = 1
        mock_launch.status = "ACTIVE"

        with (
            patch("app.database.run_migrations") as mock_mig,
            patch("app.database.SessionLocal"),
            patch("app.market_data.client.BinanceMarketDataClient", return_value=AsyncMock()),
            patch("app.forward.engine.start_or_resume_launch", return_value=mock_launch),
            patch(
                "app.forward.sync.sync_forward_candles", new_callable=AsyncMock, return_value=None
            ),
            patch("app.forward.engine.ForwardPaperEngine") as MockEngine,
        ):
            MockEngine.return_value.run_cycle.return_value = []
            MockEngine.return_value.last_report = None

            from app.cli.run_paper_breakout import _run_one_cycle

            args = argparse.Namespace(once=True, continuous=False, poll_interval_seconds=None)
            await _run_one_cycle(args)
            await _run_one_cycle(args)

        mock_mig.assert_not_called()


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
