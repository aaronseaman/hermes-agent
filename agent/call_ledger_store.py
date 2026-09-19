"""Append-only JSONL storage for the call ledger (``agent/call_ledger.py``).

Records go to ``<HERMES_HOME>/call_ledger/calls-YYYY-MM-DD-<pid>.jsonl``: one file per UTC day
per process, so concurrent batch workers never interleave writes and retention can drop whole
files. A daemon thread does the disk I/O, so a slow or broken filesystem never stalls a turn;
the queue is bounded and drops (with a DEBUG count) rather than growing without limit.

Retention runs when a process first writes to a directory on a given day: files older than
``retention_days`` go first, then the oldest files until the directory is under ``max_mb``.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent.call_ledger")

LEDGER_DIRNAME = "call_ledger"
_FILE_PREFIX = "calls-"
_MAX_PENDING = 20_000
_IDLE_EXIT_SECONDS = 30.0


def ledger_dir() -> Path:
    """The active profile's ledger directory (resolved at call time, never cached)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / LEDGER_DIRNAME


def _utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _file_day(path: Path) -> str:
    """``calls-2026-09-17-1234.jsonl`` -> ``2026-09-17`` ("" for foreign files)."""
    stem = path.name[len(_FILE_PREFIX):]
    return stem[:10] if path.name.startswith(_FILE_PREFIX) and len(stem) >= 10 else ""


def prune(directory: Path, *, retention_days: int, max_mb: int, now: float | None = None) -> None:
    """Apply age then size retention to ledger files in *directory*."""
    files = sorted((p for p in directory.glob(f"{_FILE_PREFIX}*.jsonl") if _file_day(p)), key=lambda p: (_file_day(p), p.name))
    cutoff = _utc_day((now or time.time()) - timedelta(days=retention_days).total_seconds())
    kept = []
    for path in files:
        if _file_day(path) < cutoff:
            path.unlink(missing_ok=True)
        else:
            kept.append(path)
    budget = max_mb * 1024 * 1024
    sizes = {p: p.stat().st_size for p in kept if p.exists()}
    total = sum(sizes.values())
    for path in kept[:-1]:  # never delete the file being written today
        if total <= budget:
            break
        total -= sizes.get(path, 0)
        path.unlink(missing_ok=True)


class _Writer:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: deque[tuple[Path, Any, str]] = deque()
        self._busy = False
        self._thread: threading.Thread | None = None
        self._pruned: set[tuple[str, str]] = set()
        self.dropped = 0

    def submit(self, directory: Path, settings: Any, line: str) -> None:
        with self._cond:
            if len(self._pending) >= _MAX_PENDING:
                self.dropped += 1
                logger.debug("call ledger: queue full, dropped %d record(s)", self.dropped)
                return
            self._pending.append((directory, settings, line))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="call-ledger-writer", daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def flush(self, timeout: float) -> bool:
        """Wait up to *timeout* for queued records to reach disk; False on timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._pending and not self._busy, timeout)

    def _loop(self) -> None:
        while True:
            with self._cond:
                if not self._cond.wait_for(lambda: self._pending, _IDLE_EXIT_SECONDS):
                    self._thread = None
                    return
                batch = list(self._pending)
                self._pending.clear()
                self._busy = True
            try:
                self._write(batch)
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _write(self, batch: list[tuple[Path, Any, str]]) -> None:
        by_file: dict[Path, list[str]] = {}
        day = _utc_day(time.time())
        for directory, settings, line in batch:
            try:
                key = (str(directory), day)
                if key not in self._pruned:
                    directory.mkdir(parents=True, exist_ok=True)
                    prune(directory, retention_days=settings.retention_days, max_mb=settings.max_mb)
                    self._pruned.add(key)
                by_file.setdefault(directory / f"{_FILE_PREFIX}{day}-{os.getpid()}.jsonl", []).append(line)
            except Exception:
                logger.debug("call ledger: could not prepare %s", directory, exc_info=True)
        for path, lines in by_file.items():
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            except Exception:
                logger.debug("call ledger: write to %s failed (%d record(s) lost)", path, len(lines), exc_info=True)


WRITER = _Writer()
atexit.register(lambda: WRITER.flush(timeout=2.0))
