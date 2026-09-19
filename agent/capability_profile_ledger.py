"""The ledger-backed :class:`~agent.capability_resolver.ObservedProfile` (stage 4).

One :class:`LedgerProfile` per ledger directory, so a process serving several profiles keeps each
profile's measurements apart (the directory is ``<HERMES_HOME>/call_ledger`` resolved at call time).

Refresh is incremental, never a scan per call:

* this process's own attempts are folded in **live** as they are recorded
  (``call_ledger.record_capability_attempt`` → :meth:`LedgerProfile.ingest`), so the next selection
  sees them without waiting for the async writer;
* other processes' files (``calls-<day>-<pid>.jsonl``, one writer each) are tailed from the byte
  offset last read, at most once per ``REFRESH_INTERVAL_S``. Only complete lines are consumed, and
  only ``"kind":"capability"`` lines are parsed. This process's own files are skipped: everything in
  them was ingested live.

Measurements outlive retention inside a running process (a pruned file's records stay counted until
restart); a new process reads what retention kept.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent.capability_profile import ProfileStats

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_S = 2.0
_MARKER = '"kind":"capability"'


class LedgerProfile:
    def __init__(self, directory: Path, *, clock=time.monotonic):
        self.directory = Path(directory)
        self._clock = clock
        self._stats = ProfileStats()
        self._offsets: Dict[str, int] = {}
        self._next_refresh = 0.0
        self._lock = threading.Lock()
        self._own_suffix = f"-{os.getpid()}.jsonl"

    def ingest(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._stats.add(record)

    def observe(self, capability: str, identity: str):
        with self._lock:
            now = self._clock()
            if now >= self._next_refresh:
                self._next_refresh = now + REFRESH_INTERVAL_S
                self._tail_other_processes()
            return self._stats.observe(capability, identity)

    def _tail_other_processes(self) -> None:
        from agent.call_ledger_store import _FILE_PREFIX
        try:
            paths = sorted(self.directory.glob(f"{_FILE_PREFIX}*.jsonl"))
        except OSError:
            return
        for path in paths:
            if path.name.endswith(self._own_suffix):
                continue
            try:
                self._tail(path)
            except OSError:
                logger.debug("capability profile: could not read %s", path, exc_info=True)

    def _tail(self, path: Path) -> None:
        key = str(path)
        offset = self._offsets.get(key, 0)
        size = path.stat().st_size
        if size < offset:  # truncated or replaced: start over
            offset = 0
        if size == offset:
            return
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read(size - offset)
        end = chunk.rfind(b"\n") + 1  # a line the writer has not finished is read next time
        self._offsets[key] = offset + end
        for raw in chunk[:end].splitlines():
            line = raw.decode("utf-8", "replace")
            if _MARKER not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                self._stats.add(record)


_PROFILES: Dict[str, LedgerProfile] = {}
_PROFILES_LOCK = threading.Lock()


def profile_for(directory: Optional[Path] = None) -> LedgerProfile:
    """The profile for ``directory`` (default: the active profile's ledger directory)."""
    if directory is None:
        from agent.call_ledger_store import ledger_dir
        directory = ledger_dir()
    key = str(directory)
    with _PROFILES_LOCK:
        profile = _PROFILES.get(key)
        if profile is None:
            profile = _PROFILES[key] = LedgerProfile(Path(directory))
        return profile


def reset_profiles() -> None:
    """Forget every in-process profile (tests, and after a ledger directory is wiped)."""
    with _PROFILES_LOCK:
        _PROFILES.clear()
