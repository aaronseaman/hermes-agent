"""Local call ledger: one JSONL record per model call, tool call and finished turn.

It measures where calls, tokens and money go, so later work (result reuse, model routing,
compiled skills) starts from a baseline instead of a guess. ``hermes insights --ledger``
(``agent/call_ledger_report.py``) is the consumer.

Local-only: records are appended under ``<HERMES_HOME>/call_ledger/`` and never sent anywhere.
Tool arguments are never stored, only a non-reversible signature hash. Off by default
(``agent.call_ledger.enabled``). Every entry point is best-effort: a failure logs at DEBUG
and the turn carries on.

A turn is bound through a ContextVar (``begin_turn`` / ``end_turn`` in ``agent/turn_facade.py``),
so the model-call, auxiliary-call and tool-call seams need no agent handle, worker threads
inherit it through ``tools.thread_context``, and a delegated child's turn nests inside the
parent's and restores it on exit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
import weakref
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from agent.call_ledger_store import WRITER, ledger_dir

logger = logging.getLogger(__name__)

# How the ``signature`` field was derived, so a state-aware call fingerprint can replace it later
# without changing the record shape. Repeats are decided when the record is written, so one turn
# never mixes kinds; the report carries the kinds it read so a mixed corpus is visible.
SIGNATURE_KIND = "tool+canonical_args"
_SESSION_SIGNATURE_CAP = 4096
_GUIDANCE_CODE_RE = re.compile(r"\[Tool loop (?:warning|hard stop): ([a-z_]+);")


@dataclass(frozen=True)
class LedgerSettings:
    enabled: bool = False
    retention_days: int = 14
    max_mb: int = 64


def resolve_ledger_settings(raw: Any) -> LedgerSettings:
    """``agent.call_ledger`` from config.yaml; a malformed section means the defaults (off)."""
    if not isinstance(raw, Mapping):
        return LedgerSettings()
    defaults = LedgerSettings()

    def _positive_int(key: str, default: int) -> int:
        try:
            value = int(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return LedgerSettings(
        enabled=raw.get("enabled") is True,
        retention_days=_positive_int("retention_days", defaults.retention_days),
        max_mb=_positive_int("max_mb", defaults.max_mb),
    )


@dataclass
class _SessionTally:
    """Per-agent state that outlives a turn: the turn counter and recently seen signatures."""

    turns: int = 0
    signatures: dict = field(default_factory=dict)  # insertion-ordered set, capped


@dataclass
class _TurnTally:
    path_dir: Path
    settings: LedgerSettings
    session: _SessionTally
    base: dict
    started_mono: float
    turn_signatures: set = field(default_factory=set)
    model_calls: int = 0
    model_errors: int = 0
    errors_since_ok: int = 0
    tool_calls: int = 0
    repeated_in_turn: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_active_turn: ContextVar[Optional[_TurnTally]] = ContextVar("call_ledger_turn", default=None)
_sessions: "weakref.WeakKeyDictionary[Any, _SessionTally]" = weakref.WeakKeyDictionary()
_sessions_lock = threading.Lock()


def active_turn_bound() -> bool:
    return _active_turn.get() is not None


def _role(agent: Any) -> str:
    return "subagent" if (getattr(agent, "platform", None) or "") == "subagent" else "main"


def _emit(turn: _TurnTally, record: dict) -> None:
    record = {"ts": round(time.time(), 3), **turn.base, **record}
    WRITER.submit(turn.path_dir, turn.settings, json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def begin_turn(agent: Any, task_id: str):
    """Bind a ledger turn for *agent* when the ledger is enabled; returns the reset token or None."""
    try:
        settings = getattr(agent, "_call_ledger_settings", None)
        if settings is None or not settings.enabled:
            return None
        with _sessions_lock:
            session = _sessions.get(agent)
            if session is None:
                session = _sessions[agent] = _SessionTally()
            session.turns += 1
            turn_index = session.turns
        base = {
            "session_id": str(getattr(agent, "session_id", None) or ""),
            "task_id": str(task_id or ""),
            "turn_id": uuid.uuid4().hex,
            "turn_index": turn_index,
            "role": _role(agent),
            "platform": str(getattr(agent, "platform", None) or ""),
        }
        tally = _TurnTally(
            path_dir=ledger_dir(), settings=settings, session=session, base=base,
            started_mono=time.monotonic(),
        )
        return _active_turn.set(tally)
    except Exception:
        logger.debug("call ledger: begin_turn failed", exc_info=True)
        return None


def end_turn(token, *, outcome: str) -> None:
    """Write the turn summary and release the binding.

    The summary is queued like every other record and never waited on: a turn must not pay for
    the ledger's disk I/O. The writer's ``atexit`` flush is what keeps a short-lived process
    (a batch worker) from exiting with an unwritten tail.
    """
    if token is None:
        return
    try:
        turn = _active_turn.get()
        _active_turn.reset(token)
        if turn is None:
            return
        with turn.lock:
            prompt = turn.input_tokens + turn.cache_read_tokens + turn.cache_write_tokens
            summary = {
                "kind": "turn",
                "outcome": outcome,
                "wall_s": round(time.monotonic() - turn.started_mono, 3),
                "model_calls": turn.model_calls,
                "model_errors": turn.model_errors,
                "tool_calls": turn.tool_calls,
                "repeated_signatures": turn.repeated_in_turn,
                "input_tokens": turn.input_tokens,
                "cache_read_tokens": turn.cache_read_tokens,
                "cache_write_tokens": turn.cache_write_tokens,
                "output_tokens": turn.output_tokens,
                "reasoning_tokens": turn.reasoning_tokens,
                "cache_read_share": round(turn.cache_read_tokens / prompt, 4) if prompt else None,
                "cost_usd": round(turn.cost_usd, 8),
                "unpriced_calls": turn.unpriced_calls,
            }
        _emit(turn, summary)
    except Exception:
        logger.debug("call ledger: end_turn failed", exc_info=True)


def _usage_fields(turn: _TurnTally, usage: Any, cost_usd: Optional[float]) -> dict:
    """Token buckets + cost for one call, folded into the turn totals (caller holds the lock)."""
    if usage is None:
        return {"usage_available": False, "cost_usd": None}
    counts = {
        name: int(getattr(usage, name, 0) or 0)
        for name in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens")
    }
    for name, value in counts.items():
        setattr(turn, name, getattr(turn, name) + value)
    if cost_usd is None:
        turn.unpriced_calls += 1
    else:
        turn.cost_usd += float(cost_usd)
    return {"usage_available": True, **counts, "cost_usd": None if cost_usd is None else round(float(cost_usd), 8)}


def record_model_call(
    agent: Any, *, usage: Any, cost_usd: Optional[float], latency_s: Optional[float],
    outcome: str = "ok", error_class: Optional[str] = None, retryable: Optional[bool] = None,
) -> None:
    """One main-loop (or subagent) provider attempt. ``outcome`` is ``ok`` or ``error``;
    ``retry_index`` counts the failed attempts that preceded this one in the turn."""
    turn = _active_turn.get()
    if turn is None:
        return
    try:
        with turn.lock:
            turn.model_calls += 1
            retry_index = turn.errors_since_ok
            if outcome == "ok":
                turn.errors_since_ok = 0
            else:
                turn.model_errors += 1
                turn.errors_since_ok += 1
            record = {
                "kind": "model",
                "call_role": turn.base["role"],
                "provider": str(getattr(agent, "provider", None) or ""),
                "model": str(getattr(agent, "model", None) or ""),
                "api_mode": str(getattr(agent, "api_mode", None) or ""),
                **_usage_fields(turn, usage, cost_usd),
                "latency_s": None if latency_s is None else round(float(latency_s), 3),
                "outcome": outcome,
                "error_class": error_class,
                "retryable": retryable,
                "retry_index": retry_index,
            }
        _emit(turn, record)
    except Exception:
        logger.debug("call ledger: record_model_call failed", exc_info=True)


def record_aux_call(task: str, *, model: str, provider: Optional[str], usage: Any, cost_usd: Optional[float]) -> None:
    """One successful auxiliary LLM call (compression, vision, web_extract, ...) inside a turn.
    The aux client has no timing at its accounting chokepoint, so latency is not recorded."""
    turn = _active_turn.get()
    if turn is None:
        return
    try:
        with turn.lock:
            turn.model_calls += 1
            record = {
                "kind": "model",
                "call_role": "auxiliary",
                "aux_task": task,
                "provider": provider or "",
                "model": model,
                "api_mode": "",
                **_usage_fields(turn, usage, cost_usd),
                "latency_s": None,
                "outcome": "ok",
                "error_class": None,
                "retryable": None,
                "retry_index": 0,
            }
        _emit(turn, record)
    except Exception:
        logger.debug("call ledger: record_aux_call failed", exc_info=True)


def record_capability_attempt(record: Mapping[str, Any]) -> None:
    """One ``semantic_call`` attempt at one implementation (``agent/semantic_call_cascade.py``).

    The model call inside it is already a ``model`` record (via ``aux_accounting``); this record is
    what the resolver learns from: capability, implementation identity, outcome after verification,
    latency, cache-adjusted cost. It does not touch the turn totals, so nothing is counted twice.
    Folded into the profile's in-process estimator at once, so the next selection sees it.
    """
    turn = _active_turn.get()
    if turn is None:
        return
    try:
        from agent.capability_profile_ledger import profile_for

        full = {"kind": "capability", **record}
        profile_for(turn.path_dir).ingest(full)
        _emit(turn, full)
    except Exception:
        logger.debug("call ledger: record_capability_attempt failed", exc_info=True)


def tool_signature(name: str, args: Any) -> str:
    """Non-reversible identity of (tool, canonical args), from the guardrail's own signature."""
    from agent.tool_guardrails import ToolCallSignature

    sig = ToolCallSignature.from_call(name, args if isinstance(args, Mapping) else {})
    return hashlib.sha256(f"{sig.tool_name}\0{sig.args_hash}".encode("utf-8", "surrogatepass")).hexdigest()[:24]


def guardrail_code(result: Any, *, blocked: bool) -> Optional[str]:
    """The guardrail decision that shaped this result, read from the result the guardrail
    produced: a blocked call's synthetic JSON, or the warning/hard-stop guidance appended to
    an executed call's result. Blocks by scope or plugin hooks report ``blocked``."""
    if not isinstance(result, str):
        return "blocked" if blocked else None
    if blocked:
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            return "blocked"
        guard = parsed.get("guardrail") if isinstance(parsed, dict) else None
        return str(guard.get("code")) if isinstance(guard, dict) and guard.get("code") else "blocked"
    match = _GUIDANCE_CODE_RE.search(result[-2048:])
    return match.group(1) if match else None


def record_tool_call(
    name: str, args: Any, *, segment: str, duration_s: float, raw_result: Any, final_result: Any,
    persisted_result: Any, failed: bool, blocked: bool,
) -> None:
    """One committed tool call. ``raw_result`` is the handler's output (sizes it);
    ``final_result`` is the post-guardrail text (carries the guardrail decision);
    ``persisted_result`` is what enters context (a spill-to-disk preview when oversized)."""
    turn = _active_turn.get()
    if turn is None:
        return
    try:
        from tools.registry import registry
        from tools.tool_result_storage import extract_persisted_path

        signature = tool_signature(name, args)
        effects = asdict(registry.get_effects(name))
        with turn.lock:
            turn.tool_calls += 1
            repeated_in_turn = signature in turn.turn_signatures
            turn.turn_signatures.add(signature)
            if repeated_in_turn:
                turn.repeated_in_turn += 1
            sigs = turn.session.signatures
            repeated_in_session = signature in sigs
            sigs[signature] = None
            if len(sigs) > _SESSION_SIGNATURE_CAP:
                sigs.pop(next(iter(sigs)))
        size = len(raw_result) if isinstance(raw_result, str) else len(str(raw_result))
        _emit(turn, {
            "kind": "tool",
            "tool": name,
            "signature": signature,
            "signature_kind": SIGNATURE_KIND,
            "effects": effects,
            "segment": segment,
            "duration_s": round(float(duration_s), 3),
            "result_chars": size,
            "spilled": extract_persisted_path(persisted_result) is not None,
            "failed": bool(failed),
            "blocked": bool(blocked),
            "guardrail": guardrail_code(final_result, blocked=blocked),
            "repeated_in_turn": repeated_in_turn,
            "repeated_in_session": repeated_in_session,
        })
    except Exception:
        logger.debug("call ledger: record_tool_call failed", exc_info=True)
