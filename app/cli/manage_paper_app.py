"""Process manager for the paper trading system.

Usage:
    python -m app.cli.manage_paper_app start
    python -m app.cli.manage_paper_app stop
    python -m app.cli.manage_paper_app restart
    python -m app.cli.manage_paper_app status
    python -m app.cli.manage_paper_app logs [--lines N] [--service paper_trader|api_server]
    python -m app.cli.manage_paper_app backup
    python -m app.cli.manage_paper_app list-backups
    python -m app.cli.manage_paper_app validate-backup <file>
    python -m app.cli.manage_paper_app diagnose
    python -m app.cli.manage_paper_app install-autostart   (macOS only)
    python -m app.cli.manage_paper_app uninstall-autostart (macOS only)

Starts both the paper trader (run_paper_breakout --continuous) and the API
server (uvicorn) as separate background processes.  PID files and rotating
log files are stored under the runtime directory (default: ./runtime/).

No real orders are ever placed.  No private keys are used.  PAPER/TEST only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Runtime directory layout
# ---------------------------------------------------------------------------

_DEFAULT_RUNTIME = Path("runtime")

_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_LOG_BACKUP_COUNT = 5
_BACKUP_RETENTION = 30

_SERVICE_PAPER_TRADER = "paper_trader"
_SERVICE_API_SERVER = "api_server"
_SERVICES = (_SERVICE_PAPER_TRADER, _SERVICE_API_SERVER)

_WARNING = (
    "\n"
    "╔══════════════════════════════════════════════════════════════╗\n"
    "║   ⚠  PAPER / TEST — NO REAL MONEY — NO REAL ORDERS  ⚠       ║\n"
    "╚══════════════════════════════════════════════════════════════╝\n"
)


# ---------------------------------------------------------------------------
# Runtime paths
# ---------------------------------------------------------------------------


def _runtime_dir() -> Path:
    env = os.environ.get("PAPER_RUNTIME_DIR")
    return Path(env) if env else _DEFAULT_RUNTIME


def _pid_dir() -> Path:
    return _runtime_dir() / "pids"


def _log_dir() -> Path:
    return _runtime_dir() / "logs"


def _backup_dir() -> Path:
    return _runtime_dir() / "backups"


def _pid_file(service: str) -> Path:
    return _pid_dir() / f"{service}.pid"


def _log_file(service: str) -> Path:
    return _log_dir() / f"{service}.log"


def _manager_log_file() -> Path:
    return _log_dir() / "app_manager.log"


def _ensure_dirs() -> None:
    for d in (_pid_dir(), _log_dir(), _backup_dir()):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Manager logger (writes to app_manager.log + stderr)
# ---------------------------------------------------------------------------


def _setup_manager_logger() -> logging.Logger:
    _ensure_dirs()
    log = logging.getLogger("app_manager")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.handlers.RotatingFileHandler(
        _manager_log_file(), maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT
    )
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------


def _read_pid(service: str) -> Optional[int]:
    pf = _pid_file(service)
    try:
        return int(pf.read_text().strip())
    except Exception:
        return None


def _write_pid(service: str, pid: int) -> None:
    _pid_file(service).write_text(str(pid))


def _remove_pid(service: str) -> None:
    try:
        _pid_file(service).unlink()
    except FileNotFoundError:
        pass


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _service_running(service: str) -> tuple[bool, Optional[int]]:
    pid = _read_pid(service)
    if pid is None:
        return False, None
    if _is_running(pid):
        return True, pid
    # Stale PID file
    _remove_pid(service)
    return False, None


# ---------------------------------------------------------------------------
# Start / stop individual services
# ---------------------------------------------------------------------------


def _start_service(service: str, cmd: list[str], log: logging.Logger) -> int:
    logfile = open(_log_file(service), "a")  # noqa: WPS515
    proc = subprocess.Popen(
        cmd,
        stdout=logfile,
        stderr=logfile,
        start_new_session=True,
    )
    _write_pid(service, proc.pid)
    log.info("Started %s (pid=%d) → %s", service, proc.pid, _log_file(service))
    return proc.pid


def _stop_service(service: str, log: logging.Logger, timeout: int = 10) -> bool:
    running, pid = _service_running(service)
    if not running or pid is None:
        log.info("%s is not running.", service)
        return True

    log.info("Stopping %s (pid=%d)…", service, pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_pid(service)
        return True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_running(pid):
            _remove_pid(service)
            log.info("%s stopped.", service)
            return True
        time.sleep(0.5)

    # Force-kill if still running
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_pid(service)
    log.warning("%s did not stop gracefully; sent SIGKILL.", service)
    return True


# ---------------------------------------------------------------------------
# DB path helper
# ---------------------------------------------------------------------------


def _db_path() -> str:
    try:
        from app.config import get_settings

        return get_settings().database_url.replace("sqlite:///", "")
    except Exception:
        return "./trading.db"


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> int:
    log = _setup_manager_logger()
    _ensure_dirs()

    try:
        from app.config import get_settings

        settings = get_settings()
        host = settings.paper_app_host
        port = settings.paper_app_port
    except Exception:
        host = "127.0.0.1"
        port = 8000

    errors = 0
    for service in _SERVICES:
        running, pid = _service_running(service)
        if running:
            log.info("%s already running (pid=%d). Skipping.", service, pid)
            continue

        if service == _SERVICE_PAPER_TRADER:
            cmd = [sys.executable, "-m", "app.cli.run_paper_breakout", "--continuous"]
        else:
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                host,
                "--port",
                str(port),
            ]

        try:
            _start_service(service, cmd, log)
        except Exception as exc:
            log.error("Failed to start %s: %s", service, exc)
            errors += 1

    if errors:
        return 1
    print("Both services started. Use 'status' to verify.")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    log = _setup_manager_logger()
    ok = True
    for service in reversed(_SERVICES):
        if not _stop_service(service, log):
            ok = False
    return 0 if ok else 1


def cmd_restart(args: argparse.Namespace) -> int:
    rc = cmd_stop(args)
    if rc != 0:
        return rc
    time.sleep(1)
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    print(_WARNING)
    for service in _SERVICES:
        running, pid = _service_running(service)
        state = f"RUNNING (pid={pid})" if running else "STOPPED"
        print(f"  {service:<20} {state}")

    # Show latest heartbeat if available
    try:
        from app.database import SessionLocal
        from app.services.heartbeat import get_latest_heartbeat

        with SessionLocal() as session:
            hb = get_latest_heartbeat(session)
        if hb is None:
            print("\n  Heartbeat: no record yet")
        else:
            now = datetime.now(UTC).replace(tzinfo=None)
            age = int((now - hb.timestamp_utc).total_seconds())
            print(
                f"\n  Heartbeat: {hb.timestamp_utc.isoformat()}Z"
                f"  result={hb.cycle_result}"
                f"  age={age}s"
            )
    except Exception:
        pass

    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    lines = getattr(args, "lines", 50) or 50
    service = getattr(args, "service", None)
    targets = [service] if service else list(_SERVICES) + ["app_manager"]

    for svc in targets:
        lf = _log_file(svc) if svc != "app_manager" else _manager_log_file()
        print(f"\n=== {svc} ({lf}) ===")
        try:
            content = lf.read_text(errors="replace").splitlines()
            for line in content[-lines:]:
                print(line)
        except FileNotFoundError:
            print("  (no log file yet)")
    return 0


# ---------------------------------------------------------------------------
# Backup helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_backup(args: argparse.Namespace) -> int:
    log = _setup_manager_logger()
    _ensure_dirs()

    src = _db_path()
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dest = _backup_dir() / f"trading_{ts}.db"
    meta_path = _backup_dir() / f"trading_{ts}.json"

    log.info("Backing up %s → %s", src, dest)
    try:
        src_conn = sqlite3.connect(src)
        dst_conn = sqlite3.connect(str(dest))
        src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
    except Exception as exc:
        log.error("Backup failed: %s", exc)
        return 1

    checksum = _sha256(dest)
    size = dest.stat().st_size
    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": src,
        "backup_file": dest.name,
        "size_bytes": size,
        "sha256": checksum,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Backup complete: %s (%d bytes, sha256=%s…)", dest.name, size, checksum[:12])

    # Enforce retention limit
    _prune_backups(log)
    print(f"Backup: {dest}  sha256={checksum[:16]}…")
    return 0


def _prune_backups(log: logging.Logger) -> None:
    backups = sorted(_backup_dir().glob("trading_*.db"))
    excess = len(backups) - _BACKUP_RETENTION
    if excess > 0:
        for old in backups[:excess]:
            old.unlink(missing_ok=True)
            meta = old.with_suffix(".json")
            meta.unlink(missing_ok=True)
            log.info("Pruned old backup: %s", old.name)


def cmd_list_backups(args: argparse.Namespace) -> int:
    backups = sorted(_backup_dir().glob("trading_*.db"), reverse=True)
    if not backups:
        print("No backups found.")
        return 0
    print(f"{'File':<40} {'Size':>12}  SHA256 (first 16)")
    print("-" * 72)
    for b in backups:
        meta_path = b.with_suffix(".json")
        try:
            meta = json.loads(meta_path.read_text())
            sha = meta.get("sha256", "?")[:16]
            size = meta.get("size_bytes", b.stat().st_size)
        except Exception:
            sha = "?"
            size = b.stat().st_size
        print(f"{b.name:<40} {size:>12,}  {sha}")
    return 0


def cmd_validate_backup(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    # Check metadata if present
    meta_path = path.with_suffix(".json")
    expected_sha: str | None = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            expected_sha = meta.get("sha256")
        except Exception:
            pass

    actual_sha = _sha256(path)
    if expected_sha and actual_sha != expected_sha:
        print("CHECKSUM MISMATCH")
        print(f"  Expected: {expected_sha}")
        print(f"  Actual:   {actual_sha}")
        return 1

    # Try opening as SQLite
    try:
        conn = sqlite3.connect(str(path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
    except Exception as exc:
        print(f"SQLite open failed: {exc}")
        return 1

    table_names = [t[0] for t in tables]
    print(f"Backup valid: {path.name}")
    print(f"  SHA256:  {actual_sha[:32]}…")
    print(f"  Tables:  {', '.join(sorted(table_names))}")
    return 0


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


def cmd_diagnose(args: argparse.Namespace) -> int:
    print(_WARNING)
    print("=== Paper Trading Diagnostic ===\n")

    for service in _SERVICES:
        running, pid = _service_running(service)
        state = f"RUNNING (pid={pid})" if running else "STOPPED"
        print(f"  Process {service:<20} {state}")

    db_path = _db_path()
    print(f"\n  DB path:  {db_path}")
    try:
        size = Path(db_path).stat().st_size
        print(f"  DB size:  {size:,} bytes")
    except Exception:
        print("  DB size:  (unavailable)")

    try:
        usage = shutil.disk_usage(".")
        gb_free = usage.free / (1024**3)
        print(f"  Disk free: {gb_free:.2f} GB")
    except Exception:
        pass

    try:
        from app.database import SessionLocal
        from app.forward.manifest import FORWARD_SYMBOL, STRATEGY_NAME
        from app.repositories.forward_repository import ForwardRepository
        from app.services.heartbeat import get_latest_heartbeat

        with SessionLocal() as session:
            repo = ForwardRepository(session)
            launch = repo.get_launch(STRATEGY_NAME, FORWARD_SYMBOL)
            if launch is None:
                print("\n  Launch:   not started")
            else:
                print(f"\n  Launch id:     {launch.id}")
                print(f"  Launch status: {launch.status}")
                print(f"  Launch time:   {launch.launch_timestamp.isoformat()}Z")
                acct = repo.get_account(launch)
                if acct:
                    print(f"  Balance:       {acct.balance} USDT")
                    print(f"  Equity:        {acct.equity} USDT")
                pos = repo.get_open_position(launch)
                print(f"  Open position: {'Yes' if pos else 'No'}")

            hb = get_latest_heartbeat(session)
            if hb:
                now = datetime.now(UTC).replace(tzinfo=None)
                age = int((now - hb.timestamp_utc).total_seconds())
                print(f"\n  Heartbeat:     {hb.timestamp_utc.isoformat()}Z (age: {age}s)")
                print(f"  Last result:   {hb.cycle_result}")
                if hb.error_message:
                    print(f"  Error:         {hb.error_message}")
            else:
                print("\n  Heartbeat:     no record yet")

    except Exception as exc:
        print(f"\n  [Error querying DB: {exc}]")

    backups = sorted(_backup_dir().glob("trading_*.db"))
    print(f"\n  Backups:  {len(backups)} file(s) in {_backup_dir()}")
    return 0


# ---------------------------------------------------------------------------
# macOS autostart
# ---------------------------------------------------------------------------

_PLIST_NAME = "com.paper_trading.manager"
_LAUNCHAGENTS = Path.home() / "Library" / "LaunchAgents"


def _plist_path() -> Path:
    return _LAUNCHAGENTS / f"{_PLIST_NAME}.plist"


def cmd_install_autostart(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        print("install-autostart is only supported on macOS.")
        return 1

    _LAUNCHAGENTS.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    cwd = str(Path.cwd())

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{_PLIST_NAME}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>-m</string>
    <string>app.cli.manage_paper_app</string>
    <string>start</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{cwd}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>StandardOutPath</key>
  <string>{_manager_log_file()}</string>
  <key>StandardErrorPath</key>
  <string>{_manager_log_file()}</string>
</dict>
</plist>
"""
    _plist_path().write_text(plist)
    print(f"Installed: {_plist_path()}")
    print(f"Enable with: launchctl load {_plist_path()}")
    return 0


def cmd_uninstall_autostart(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        print("uninstall-autostart is only supported on macOS.")
        return 1
    p = _plist_path()
    if p.exists():
        p.unlink()
        print(f"Removed: {p}")
    else:
        print("No autostart plist found.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.manage_paper_app",
        description="Process manager for the paper trading system. PAPER/TEST only.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="Start paper trader and API server.")
    sub.add_parser("stop", help="Stop both services.")
    sub.add_parser("restart", help="Stop then start both services.")
    sub.add_parser("status", help="Show running status and latest heartbeat.")

    logs_p = sub.add_parser("logs", help="Tail log files.")
    logs_p.add_argument("--lines", type=int, default=50, help="Number of lines to show.")
    logs_p.add_argument(
        "--service",
        choices=[*_SERVICES, "app_manager"],
        default=None,
        help="Service to show logs for (default: all).",
    )

    sub.add_parser("backup", help="Create a SQLite backup with SHA-256 checksum.")
    sub.add_parser("list-backups", help="List existing backups.")

    vb = sub.add_parser("validate-backup", help="Validate a backup file.")
    vb.add_argument("file", help="Path to the .db backup file.")

    sub.add_parser("diagnose", help="Print full diagnostic report.")
    sub.add_parser("install-autostart", help="Install macOS LaunchAgent (macOS only).")
    sub.add_parser("uninstall-autostart", help="Remove macOS LaunchAgent (macOS only).")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_COMMANDS = {
    "start": cmd_start,
    "stop": cmd_stop,
    "restart": cmd_restart,
    "status": cmd_status,
    "logs": cmd_logs,
    "backup": cmd_backup,
    "list-backups": cmd_list_backups,
    "validate-backup": cmd_validate_backup,
    "diagnose": cmd_diagnose,
    "install-autostart": cmd_install_autostart,
    "uninstall-autostart": cmd_uninstall_autostart,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cmd = _COMMANDS.get(args.command)
    if cmd is None:
        parser.print_help()
        sys.exit(1)
    sys.exit(cmd(args))


if __name__ == "__main__":
    main()
