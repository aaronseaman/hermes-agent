"""Trace source for compiled-skill mining — the ONE place that reads execution history.

Today the source is the persisted SessionDB transcript: each user turn's assistant ``tool_calls``
paired with their ``tool`` results. Everything downstream (mining, compiling, replay) reads only
:class:`ProcedureTrace`, so the stage 0 local call ledger can feed mining later by replacing the
body of :func:`load_procedure_traces` — no consumer changes.

Saved trajectories (``agent/trajectory.py``) are not read: they are an opt-in training export
written to a cwd-relative file (not profile-scoped), carry no session id or working directory, and
flatten tool calls into ShareGPT text. SessionDB is the always-on, profile-scoped record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sessions whose tool calls are maintenance, not user work: mining them would compile the
# curator's own review loop.
_EXCLUDED_SOURCES = ("curator", "tool")


@dataclass(frozen=True)
class TraceStep:
    tool: str
    args: Dict[str, Any]
    result: Optional[Dict[str, Any]]  # parsed JSON object result; None when the result was not one


@dataclass(frozen=True)
class ProcedureTrace:
    """The tool calls of one user turn, in call order."""
    session_id: str
    cwd: Optional[str]
    steps: Tuple[TraceStep, ...]
    succeeded: bool  # every result is ok (see result_ok) AND the turn ended in a final answer


def result_ok(result: Any) -> bool:
    """A tool result counts as success when it is a JSON object with no error value, no
    ``success: false``, no non-zero ``exit_code`` and no read-dedup stub (whose content lives
    in an earlier result, not in this one)."""
    return (isinstance(result, dict) and not result.get("error") and result.get("success") is not False
            and result.get("exit_code") in (None, 0) and not result.get("dedup"))


def _parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _split_turns(messages: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    """Messages after each user message up to the next one (the model's work on that request)."""
    turn: Optional[List[Dict[str, Any]]] = None
    for msg in messages:
        if msg.get("role") == "user":
            if turn:
                yield turn
            turn = []
        elif turn is not None:
            turn.append(msg)
    if turn:
        yield turn


def turn_to_trace(session_id: str, cwd: Optional[str], turn: List[Dict[str, Any]]) -> Optional[ProcedureTrace]:
    """One turn -> trace; None when it made no tool calls or a call's arguments are unreadable."""
    results = {m.get("tool_call_id"): m.get("content") for m in turn if m.get("role") == "tool"}
    steps: List[TraceStep] = []
    for msg in turn:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            args = _parse_json_object(fn.get("arguments")) if isinstance(fn, dict) else None
            if not name or args is None:
                return None
            steps.append(TraceStep(name, args, _parse_json_object(results.get(call.get("id")))))
    if not steps:
        return None
    last = turn[-1]
    answered = last.get("role") == "assistant" and not last.get("tool_calls") and bool(str(last.get("content") or "").strip())
    return ProcedureTrace(session_id, cwd, tuple(steps), answered and all(result_ok(s.result) for s in steps))


def load_procedure_traces(*, limit_sessions: int = 200, db: Any = None) -> List[ProcedureTrace]:
    """Per-turn tool-call traces from the most recent ``limit_sessions`` sessions of the active
    profile, newest session first.

    Ledger seam: the stage 0 call ledger replaces this body once it records per-turn call sequences
    with arguments, results and outcomes."""
    own_db = db is None
    if own_db:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB
        # The bound profile's DB, resolved now: this runs outside a turn (the curator pass).
        path = get_hermes_home() / "state.db"
        if not path.exists():
            return []
        db = SessionDB(db_path=path, read_only=True)
    traces: List[ProcedureTrace] = []
    try:
        sessions = db.list_sessions_rich(exclude_sources=list(_EXCLUDED_SOURCES), limit=max(1, int(limit_sessions)))
        for row in sessions:
            session_id = row.get("id")
            if not session_id or not row.get("tool_call_count"):
                continue
            try:
                messages = db.get_messages_as_conversation(session_id)
            except Exception as exc:  # one unreadable session must not end the pass
                logger.debug("compiled skills: could not load session %s: %s", session_id, exc)
                continue
            for turn in _split_turns(messages):
                trace = turn_to_trace(session_id, row.get("cwd"), turn)
                if trace is not None:
                    traces.append(trace)
    finally:
        if own_db:
            db.close()
    return traces
