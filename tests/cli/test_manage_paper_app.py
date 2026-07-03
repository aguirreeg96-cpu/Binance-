"""Tests for the process manager CLI."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import app.cli.manage_paper_app as mgr


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_RUNTIME_DIR", str(tmp_path / "runtime"))
    # Reset module-level cached paths by monkey-patching _runtime_dir
    yield tmp_path / "runtime"


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------


class TestPidHelpers:
    def test_read_pid_missing_returns_none(self):
        assert mgr._read_pid("no_such_service") is None

    def test_write_and_read_pid(self):
        mgr._ensure_dirs()
        mgr._write_pid("test_service", 12345)
        assert mgr._read_pid("test_service") == 12345

    def test_remove_pid(self):
        mgr._ensure_dirs()
        mgr._write_pid("test_service2", 999)
        mgr._remove_pid("test_service2")
        assert mgr._read_pid("test_service2") is None

    def test_remove_pid_nonexistent(self):
        mgr._remove_pid("ghost")  # Should not raise

    def test_is_running_own_process(self):
        assert mgr._is_running(os.getpid()) is True

    def test_is_running_invalid_pid(self):
        assert mgr._is_running(99999999) is False

    def test_service_running_stale_pid(self):
        mgr._ensure_dirs()
        mgr._write_pid("stale_svc", 99999999)
        running, pid = mgr._service_running("stale_svc")
        assert running is False
        assert pid is None
        assert not mgr._pid_file("stale_svc").exists()

    def test_service_running_live_pid(self):
        mgr._ensure_dirs()
        mgr._write_pid("live_svc", os.getpid())
        running, pid = mgr._service_running("live_svc")
        assert running is True
        assert pid == os.getpid()


# ---------------------------------------------------------------------------
# SHA256 and backup helpers
# ---------------------------------------------------------------------------


class TestSha256:
    def test_known_checksum(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        import hashlib

        expected = hashlib.sha256(b"hello").hexdigest()
        assert mgr._sha256(f) == expected


# ---------------------------------------------------------------------------
# cmd_backup / cmd_list_backups / cmd_validate_backup
# ---------------------------------------------------------------------------


class TestBackupCommands:
    def _make_db(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

    def test_backup_creates_file(self, tmp_path):
        db_path = tmp_path / "trading.db"
        self._make_db(db_path)

        with patch.object(mgr, "_db_path", return_value=str(db_path)):
            mgr._ensure_dirs()
            args = MagicMock()
            rc = mgr.cmd_backup(args)

        assert rc == 0
        backups = list(mgr._backup_dir().glob("trading_*.db"))
        assert len(backups) == 1

    def test_backup_creates_metadata(self, tmp_path):
        db_path = tmp_path / "trading.db"
        self._make_db(db_path)

        with patch.object(mgr, "_db_path", return_value=str(db_path)):
            mgr._ensure_dirs()
            args = MagicMock()
            mgr.cmd_backup(args)

        metas = list(mgr._backup_dir().glob("trading_*.json"))
        assert len(metas) == 1
        meta = json.loads(metas[0].read_text())
        assert "sha256" in meta
        assert "size_bytes" in meta
        assert "source" in meta

    def test_list_backups_empty(self, capsys):
        mgr._ensure_dirs()
        args = MagicMock()
        rc = mgr.cmd_list_backups(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No backups" in out

    def test_validate_backup_valid(self, tmp_path):
        db_path = tmp_path / "backup.db"
        self._make_db(db_path)

        # Create metadata
        sha = mgr._sha256(db_path)
        meta = {"sha256": sha, "size_bytes": db_path.stat().st_size}
        meta_path = db_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta))

        args = MagicMock()
        args.file = str(db_path)
        rc = mgr.cmd_validate_backup(args)
        assert rc == 0

    def test_validate_backup_corrupted(self, tmp_path):
        db_path = tmp_path / "corrupt.db"
        db_path.write_bytes(b"not a valid sqlite db")

        sha = "wrong_hash"
        meta = {"sha256": sha}
        meta_path = db_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta))

        args = MagicMock()
        args.file = str(db_path)
        rc = mgr.cmd_validate_backup(args)
        assert rc == 1

    def test_validate_backup_missing_file(self, tmp_path):
        args = MagicMock()
        args.file = str(tmp_path / "nonexistent.db")
        rc = mgr.cmd_validate_backup(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# cmd_status — smoke test
# ---------------------------------------------------------------------------


class TestCmdStatus:
    def test_status_runs_without_db(self, capsys):
        with patch("app.database.SessionLocal") as mock_sl:
            mock_sl.side_effect = Exception("no db")
            args = MagicMock()
            rc = mgr.cmd_status(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "paper_trader" in out
        assert "api_server" in out


# ---------------------------------------------------------------------------
# cmd_diagnose — smoke test
# ---------------------------------------------------------------------------


class TestCmdDiagnose:
    def test_diagnose_runs_without_db(self, capsys):
        with patch("app.database.SessionLocal") as mock_sl:
            mock_sl.side_effect = Exception("no db")
            args = MagicMock()
            rc = mgr.cmd_diagnose(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# macOS autostart
# ---------------------------------------------------------------------------


class TestAutostart:
    def test_install_autostart_non_macos(self, capsys):
        with patch("app.cli.manage_paper_app.platform") as mock_plat:
            mock_plat.system.return_value = "Linux"
            args = MagicMock()
            rc = mgr.cmd_install_autostart(args)
        assert rc == 1

    def test_uninstall_autostart_non_macos(self, capsys):
        with patch("app.cli.manage_paper_app.platform") as mock_plat:
            mock_plat.system.return_value = "Linux"
            args = MagicMock()
            rc = mgr.cmd_uninstall_autostart(args)
        assert rc == 1
