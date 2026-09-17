"""Declared execution effects of a registered tool — the single source the runtime reads
instead of per-consumer tool-name lists (guardrails, batch planner, ...).

Dependency-free (``tools/registry.py`` imports it). A tool that declares nothing gets
``UNKNOWN_EFFECTS``: not idempotent, not parallel-safe — every consumer's conservative path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class ToolEffects:
    # Same args against unchanged state yield the same result, so an identical repeat is no
    # progress (feeds the guardrail's no-progress block).
    idempotent: bool = False
    # No shared mutable session state: may run concurrently with other calls in a batch.
    parallel_safe: bool = False
    # Waits on the user; always a batch barrier, whatever else is declared.
    interactive: bool = False
    # Filesystem reader/writer admitted to a parallel run by path overlap (scope comes from args).
    path_scope: Optional[Literal["read", "write"]] = None


UNKNOWN_EFFECTS = ToolEffects()
READ_ONLY_PARALLEL = ToolEffects(idempotent=True, parallel_safe=True)
