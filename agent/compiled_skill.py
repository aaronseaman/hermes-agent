"""Compiled skills: a recurring read-only procedure becomes a validated, deterministic skill.

The loop (roadmap stage 5, the first reasoning-into-infrastructure mechanism):

1. **traces** — per-turn tool-call sequences from SessionDB (``compiled_skill_traces``, the ledger seam);
2. **mine** — a sequence that succeeded in >= ``min_sessions`` sessions with varying arguments, every
   call idempotent and not destructive (``compiled_skill_mining``);
3. **compile** — an ordinary skill written through ``skill_manage`` (same storage, write-approval gate,
   audit ledger, provenance and curator lifecycle as any agent-created skill) plus a contract and a
   generated implementation (``compiled_skill_contract``);
4. **validate** — the implementation replays the contract's corpus in an isolated interpreter and must
   make the recorded calls and reproduce the recorded outputs (``compiled_skill_runtime``); only then is
   it activated, in the skill's usage record;
5. **run** — ``/<skill> k=v ...`` runs the implementation without a model call when the preconditions
   hold, validates the result, and otherwise falls back to the normal skill path, recording drift;
6. **retire** — the curator retires the compiled path after ``max_consecutive_failures`` failed runs;
   the skill stays as a plain procedure under the normal lifecycle.

Steps 1-4 and 6 run inside the curator pass (``agent/curator.py``): no daemon, no LLM. Activation writes
only the usage record and never touches a live conversation's system prompt or toolset; the new skill
enters the skills index of the next session, the deferred default of ``/skills install``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Iterator, List, Optional, Tuple, Union

from agent.compiled_skill_contract import (
    CONTRACT_PATH, IMPLEMENTATION_PATH, ContractError, environment_fingerprint, load, sha256_text, validate,
)

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {"enabled": False, "min_sessions": 3, "scan_sessions": 200, "max_consecutive_failures": 3}
STATE_ACTIVE, STATE_REJECTED, STATE_RETIRED = "active", "rejected", "retired"
_DRIFT_KEEP = 5
_REPLY_STEP_CHARS = 2000


def settings() -> Dict[str, Any]:
    """``curator.compiled_skills`` over :data:`DEFAULTS` (config.yaml, never env)."""
    try:
        from hermes_cli.config import load_config_readonly
        section = (load_config_readonly().get("curator") or {}).get("compiled_skills") or {}
    except Exception as exc:
        logger.debug("compiled skills: config unreadable: %s", exc)
        section = {}
    merged = {**DEFAULTS, **(section if isinstance(section, dict) else {})}
    for key in ("min_sessions", "scan_sessions", "max_consecutive_failures"):
        try:
            merged[key] = max(1, int(merged[key]))
        except (TypeError, ValueError):
            merged[key] = DEFAULTS[key]
    merged["enabled"] = bool(merged["enabled"])
    return merged


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(name: str) -> Dict[str, Any]:
    from tools import skill_usage
    rec = skill_usage.get_record(name).get("compiled")
    return rec if isinstance(rec, dict) else {}


def _update(name: str, **fields: Any) -> None:
    from tools import skill_usage
    skill_usage.update_compiled(name, lambda rec: rec.update(fields))


def _fingerprint(skill_dir: Path) -> Optional[str]:
    """Hash of the validated files. Any edit changes it, which switches the compiled path off (no
    drift: nothing ran) until the curator re-validates the edited files."""
    try:
        texts = [(skill_dir / rel).read_text(encoding="utf-8") for rel in (CONTRACT_PATH, IMPLEMENTATION_PATH)]
    except OSError:
        return None
    return sha256_text("\0".join(texts))


def is_compiled_skill_dir(skill_dir: Path) -> bool:
    """A skill whose SKILL.md declares ``metadata.hermes.compiled`` and whose contract exists."""
    if not (skill_dir / CONTRACT_PATH).is_file():
        return False
    try:
        from agent.skill_utils import parse_frontmatter
        frontmatter, _ = parse_frontmatter((skill_dir / "SKILL.md").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    hermes = (frontmatter.get("metadata") or {}).get("hermes") if isinstance(frontmatter, dict) else None
    return isinstance(hermes, dict) and isinstance(hermes.get("compiled"), dict)


def _local_skill_roots() -> List[Path]:
    from agent.skill_utils import get_skill_create_dir, get_skills_dir
    return [d for d in (get_skills_dir(), get_skill_create_dir()) if d is not None and d.is_dir()]


def compiled_skill_dirs() -> Dict[str, Path]:
    """``{name: skill_dir}`` for compiled skills in the profile's own skill roots (never external
    or project dirs: compiled skills are only ever written by the curator into these)."""
    from agent.skill_utils import iter_skill_index_files
    found: Dict[str, Path] = {}
    for root in _local_skill_roots():
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            if is_compiled_skill_dir(skill_md.parent):
                found.setdefault(skill_md.parent.name, skill_md.parent)
    return found


# --- validation / activation / retirement ---

def validate_skill(skill_dir: Path) -> Optional[str]:
    """Why *skill_dir* may not be activated, or None: contract integrity, unchanged environment,
    still read-only, and equivalent to every corpus entry under hermetic replay."""
    from agent.compiled_skill_runtime import replay_entry
    from tools.registry import registry
    try:
        contract, source = load(skill_dir)
    except ContractError as exc:
        return str(exc)
    env = contract["environment"]
    if env.get("platform") != sys.platform:
        return f"compiled on {env.get('platform')}, running on {sys.platform}"
    current = environment_fingerprint(list(env.get("tools") or {}))["tools"]
    if changed := sorted(t for t in env.get("tools") or {} if current.get(t) != env["tools"][t]):
        return f"tool schema changed: {', '.join(changed)}"
    corpus = contract.get("corpus") or []
    if not corpus:
        return "empty replay corpus"
    for entry in corpus:
        for call in entry["calls"]:
            effects = registry.resolve_effects(call["tool"], call["args"])
            if not effects.idempotent or effects.destructive:
                return f"{call['tool']} no longer resolves to read-only"
    for i, entry in enumerate(corpus, 1):
        if (error := replay_entry(contract, source, entry)) is not None:
            return f"corpus entry {i}: {error}"
    return None


def _review(name: str, skill_dir: Path, cfg: Dict[str, Any], counts: Counter) -> None:
    """Retire after repeated failed runs; otherwise (re-)validate and set active/rejected."""
    rec = _record(name)
    state = rec.get("state")
    if state == STATE_RETIRED:
        return
    if state == STATE_ACTIVE and int(rec.get("consecutive_failures") or 0) >= cfg["max_consecutive_failures"]:
        last = (rec.get("drift") or [{}])[-1].get("reason", "unknown")
        _update(name, state=STATE_RETIRED, retired_at=_now(),
                retired_reason=f"{rec.get('consecutive_failures')} consecutive failed runs; last: {last}")
        counts["retired"] += 1
        return
    error = validate_skill(skill_dir)
    fields: Dict[str, Any] = {"fingerprint": _fingerprint(skill_dir), "validated_at": _now(), "last_error": error}
    if error is None:
        fields["state"] = STATE_ACTIVE
        if state != STATE_ACTIVE:
            fields.update(activated_at=_now(), consecutive_failures=0)
            counts["activated"] += 1
    else:
        fields["state"] = STATE_REJECTED
        counts["deactivated" if state == STATE_ACTIVE else "validation_failed"] += 1
        logger.info("compiled skill %s not activated: %s", name, error)
    _update(name, **fields)


@contextlib.contextmanager
def _curator_write_origin() -> Iterator[None]:
    """Write as the background curator: curator-managed provenance, ``curator`` ledger actor."""
    from tools.skill_ledger import reset_ledger_actor, set_ledger_actor
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin
    origin, actor = set_current_write_origin(BACKGROUND_REVIEW), set_ledger_actor("curator")
    try:
        yield
    finally:
        reset_ledger_actor(actor)
        reset_current_write_origin(origin)


def _pending_skill_names() -> set:
    """Skills with a write staged for approval (``skills.write_approval``): never re-stage them."""
    try:
        from tools import write_approval as wa
        records = wa.list_pending(wa.SKILLS)
    except Exception:
        return set()
    names = set()
    for rec in records:
        payload = rec.get("payload") or {}
        names.update(op.get("name") for op in payload.get("operations") or [] if isinstance(op, dict))
        names.add(payload.get("name"))
    return names


def compile_candidate(candidate) -> Tuple[str, str]:
    """Write *candidate* as a skill via ``skill_manage`` -> ``("created"|"staged"|"failed", detail)``."""
    from agent.compiled_skill_contract import build
    from tools.skill_manager_tool import skill_manage
    contract, implementation, skill_md = build(candidate, now=_now())
    operations = [
        {"action": "create", "name": candidate.name, "content": skill_md},
        {"action": "write_file", "name": candidate.name, "file_path": IMPLEMENTATION_PATH, "file_content": implementation},
        {"action": "write_file", "name": candidate.name, "file_path": CONTRACT_PATH,
         "file_content": json.dumps(contract, indent=2, ensure_ascii=False) + "\n"},
    ]
    with _curator_write_origin():
        result = json.loads(skill_manage(action="create", name=candidate.name, operations=operations))
    if result.get("staged"):
        return "staged", str(result.get("pending_id") or "")
    return ("created", "") if result.get("success") else ("failed", str(result.get("error") or "unknown error"))


def curator_pass() -> Dict[str, int]:
    """One deterministic pass (the curator calls this after its auto-transitions): review existing
    compiled skills, then mine, compile and validate new ones. ``{}`` when disabled. Never raises."""
    cfg = settings()
    if not cfg["enabled"]:
        return {}
    counts: Counter = Counter()
    try:
        for name, skill_dir in compiled_skill_dirs().items():
            _review(name, skill_dir, cfg, counts)
        from agent.compiled_skill_mining import mine_candidates
        from agent.compiled_skill_traces import load_procedure_traces
        from tools.skill_manager_tool import _find_skill
        candidates, rejected = mine_candidates(load_procedure_traces(limit_sessions=cfg["scan_sessions"]),
                                               min_sessions=cfg["min_sessions"])
        counts["mined"] += len(candidates)
        counts["not_compilable"] += sum(rejected.values())
        pending = _pending_skill_names()
        for candidate in candidates:
            if candidate.name in pending or _find_skill(candidate.name):
                continue
            outcome, detail = compile_candidate(candidate)
            counts[outcome] += 1
            if outcome == "created" and (found := _find_skill(candidate.name)):
                _review(candidate.name, Path(found["path"]), cfg, counts)
            elif outcome == "failed":
                logger.info("compiled skill %s not written: %s", candidate.name, detail)
    except Exception as exc:
        logger.warning("compiled skills curator pass failed: %s", exc, exc_info=True)
        counts["errors"] += 1
    return dict(counts)


# --- invocation ---

@dataclass(frozen=True)
class CompiledInvocation:
    """What a ``/<skill>`` surface does next: show ``reply`` (the compiled path answered, no model
    call), or send the normal skill message carrying ``runtime_note`` (possibly empty)."""
    reply: Optional[str] = None
    runtime_note: str = ""


NOT_COMPILED = CompiledInvocation()


def session_tool_names(enabled_toolsets: Optional[List[str]], disabled_toolsets: Optional[List[str]]) -> set:
    """The tool names a session's toolset selection grants (what a compiled run may not exceed)."""
    from model_tools import _select_tool_names
    return set(_select_tool_names(enabled_toolsets, disabled_toolsets, quiet_mode=True))


def parse_invocation(contract: Dict[str, Any], instruction: str) -> Optional[Dict[str, Any]]:
    """``k=v`` tokens (shell quoting) -> typed params valid under the input schema, else None.
    Only explicit ``k=v`` matches: free text is a request for the agent, never a parameter value."""
    schema = contract["input_schema"]
    try:
        tokens = shlex.split(instruction or "")
    except ValueError:
        return None
    if not tokens or not all("=" in t for t in tokens):
        return None
    raw = dict(t.split("=", 1) for t in tokens)
    params: Dict[str, Any] = {}
    for key, text in raw.items():
        kind = (schema["properties"].get(key) or {}).get("type")
        try:
            params[key] = {"integer": int, "number": float, "boolean": lambda s: {"true": True, "false": False}[s.lower()]
                           }.get(kind, str)(text)
        except (KeyError, ValueError):
            return None
    return params if validate(schema, params) is None else None


def _workspace_root(task_id: Optional[str]) -> str:
    from tools.file_tools_paths import _authoritative_workspace_root
    return os.path.realpath(_authoritative_workspace_root(task_id or "default") or os.getcwd())


def _unmet_precondition(contract: Dict[str, Any], task_id: Optional[str], allowed_tools: Optional[Collection[str]]) -> Optional[str]:
    """A request the skill does not cover (not drift): the normal skill path handles it."""
    tools = {c["tool"] for c in contract["calls"]}
    if allowed_tools is None or not tools <= set(allowed_tools):
        return "a declared tool is not enabled for this session"
    if _workspace_root(task_id) != os.path.realpath(contract["environment"]["cwd"]):
        return "different working directory"
    return None


def _render_reply(name: str, contract: Dict[str, Any], output: Dict[str, Any]) -> str:
    lines = [f"⚡ {name} — compiled run, no model call"]
    for i, (call, result) in enumerate(zip(contract["calls"], output["steps"]), 1):
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if len(text) > _REPLY_STEP_CHARS:
            text = text[:_REPLY_STEP_CHARS] + "\n… (truncated)"
        lines += ["", f"{i}. {call['tool']}", text]
    return "\n".join(lines)


def _record_drift(name: str, reason: str) -> None:
    def _apply(rec: Dict[str, Any]) -> None:
        rec["failures"] = int(rec.get("failures") or 0) + 1
        rec["consecutive_failures"] = int(rec.get("consecutive_failures") or 0) + 1
        rec["drift"] = [*(rec.get("drift") or []), {"at": _now(), "reason": reason}][-_DRIFT_KEEP:]
    from tools import skill_usage
    skill_usage.update_compiled(name, _apply)
    logger.info("compiled skill %s drifted (%s); fell back to the agent path", name, reason)


def _record_success(name: str) -> None:
    def _apply(rec: Dict[str, Any]) -> None:
        rec.update(runs=int(rec.get("runs") or 0) + 1, consecutive_failures=0, last_run_at=_now())
    from tools import skill_usage
    skill_usage.update_compiled(name, _apply)


def run_skill_command(cmd_key: str, instruction: str, *, task_id: Optional[str],
                      allowed_tools: Union[None, Collection[str], Callable[[], Collection[str]]]) -> CompiledInvocation:
    """The seam every ``/<skill>`` surface calls before building the normal skill message.

    :data:`NOT_COMPILED` unless *cmd_key* is an active compiled skill whose preconditions hold for
    this request; then the implementation runs under the contract guard. A passing result is the
    reply; a failure records drift and returns a runtime note for the normal (agent) path.
    *allowed_tools* is the session's tool grant (None = unknown, which never runs), or a zero-arg
    callable resolved only once an active compiled skill is found."""
    try:
        from agent.skill_commands import get_skill_commands
        info = get_skill_commands().get(cmd_key)
        skill_dir = Path(info["skill_dir"]) if info else None
        if skill_dir is None or not is_compiled_skill_dir(skill_dir):
            return NOT_COMPILED
        cfg = settings()
        name = info["name"]
        rec = _record(name)
        if (not cfg["enabled"] or rec.get("state") != STATE_ACTIVE
                or int(rec.get("consecutive_failures") or 0) >= cfg["max_consecutive_failures"]
                or rec.get("fingerprint") != _fingerprint(skill_dir)):  # changed since validation
            return NOT_COMPILED
    except Exception as exc:
        logger.debug("compiled skill lookup failed for %s: %s", cmd_key, exc, exc_info=True)
        return NOT_COMPILED
    try:
        contract, source = load(skill_dir)
    except ContractError as exc:
        return _fall_back(name, str(exc))
    params = parse_invocation(contract, instruction)
    if params is None:
        return NOT_COMPILED
    try:
        granted = allowed_tools() if callable(allowed_tools) else allowed_tools
    except Exception as exc:
        logger.debug("compiled skill %s: session tool grant unresolved: %s", name, exc)
        granted = None
    if _unmet_precondition(contract, task_id, granted) is not None:
        return NOT_COMPILED
    env = contract["environment"]
    current = environment_fingerprint(list(env["tools"]))["tools"]
    if env["platform"] != sys.platform or current != env["tools"]:
        return _fall_back(name, "environment changed (platform or tool schema)")
    from agent.compiled_skill_runtime import run_live
    try:
        output, error = run_live(contract, source, params, session_task_id=task_id)
    except Exception as exc:
        output, error = None, f"{type(exc).__name__}: {exc}"
    if error is not None:
        return _fall_back(name, error)
    _record_success(name)
    from tools.skill_usage import bump_use
    bump_use(name, task_id=task_id)
    return CompiledInvocation(reply=_render_reply(name, contract, output))


def _fall_back(name: str, reason: str) -> CompiledInvocation:
    _record_drift(name, reason)
    return CompiledInvocation(runtime_note=(
        f"The compiled fast path for this skill did not complete ({reason[:300]}), so nothing from it was "
        "shown to the user. Follow the procedure in the skill below."))
