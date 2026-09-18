"""Execute a compiled skill's implementation: hermetic corpus replay and guarded live runs.

The implementation never runs inside the Hermes process. It runs in a separate, isolated Python
interpreter (``-I -S``, empty environment apart from ``PATH``, a throwaway temp dir as cwd) whose
only channel is a line protocol on stdin/stdout: each ``call(tool, args)`` is a request the parent
answers. That gives one executor with two answerers:

- **replay** answers from a corpus entry's recorded results — no tool runs, so the equivalence
  check (same parameters -> same calls -> same outputs as the source trace) is hermetic;
- **live** answers through :class:`ContractGuard`: the call must be the contract's next declared
  call with exactly the rendered arguments, must still resolve to idempotent and not destructive,
  and is dispatched through ``model_tools.handle_function_call`` — the same hooks, middleware and
  approval path as a model-issued call — under a task id that shares the session's terminal
  sandbox but keeps its own read-tracking state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.compiled_skill_contract import canonical, check_result, render_call, validate

REPLAY_TIMEOUT_S = 30
LIVE_TIMEOUT_S = 300

# Runs in the child interpreter. The implementation's own prints go to stderr so they can never
# forge a protocol message.
_DRIVER = r"""
import json, sys
_in, _out = sys.stdin, sys.stdout
sys.stdout = sys.stderr
def _send(msg):
    _out.write(json.dumps(msg) + "\n")
    _out.flush()
def call(tool, args):
    _send({"op": "call", "tool": tool, "args": args})
    reply = json.loads(_in.readline() or "null")
    if not isinstance(reply, dict) or "result" not in reply:
        raise RuntimeError(reply.get("error") if isinstance(reply, dict) else "call channel closed")
    return reply["result"]
try:
    job = json.loads(_in.readline())
    ns = {"__name__": "compiled_skill"}
    exec(compile(job["source"], "compiled_skill.py", "exec"), ns)
    _send({"op": "done", "output": ns["run"](job["params"], call)})
except BaseException as exc:
    _send({"op": "error", "error": type(exc).__name__ + ": " + str(exc)})
"""

# answer(index, tool, args) -> (result, refusal); exactly one of them is None.
Answer = Callable[[int, str, Dict[str, Any]], Tuple[Optional[Dict[str, Any]], Optional[str]]]


def _child_env() -> Dict[str, str]:
    # Nothing profile-scoped or secret crosses: the child needs an interpreter, not credentials.
    return {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}


def drive(source: str, params: Dict[str, Any], answer: Answer, *, timeout: float) -> Tuple[Optional[Any], Optional[str], List[Tuple[str, Dict[str, Any]]]]:
    """Run ``run(params, call)`` from *source* in an isolated child. ``(output, error, calls)``."""
    from hermes_cli._subprocess_compat import windows_hide_flags
    calls: List[Tuple[str, Dict[str, Any]]] = []
    output, error = None, None
    with tempfile.TemporaryDirectory(prefix="compiled_skill_") as tmp:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _DRIVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", cwd=tmp, env=_child_env(),
            creationflags=windows_hide_flags())
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        try:
            proc.stdin.write(json.dumps({"source": source, "params": params}) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                msg = json.loads(line)
                if msg.get("op") == "call":
                    tool, args = str(msg.get("tool")), msg.get("args")
                    args = args if isinstance(args, dict) else {}
                    calls.append((tool, args))
                    result, refusal = answer(len(calls) - 1, tool, args)
                    if refusal is not None:
                        error = error or refusal
                    proc.stdin.write(json.dumps({"result": result} if refusal is None else {"error": refusal}) + "\n")
                    proc.stdin.flush()
                    continue
                if msg.get("op") == "done":
                    output = msg.get("output")
                else:
                    error = error or str(msg.get("error") or "implementation failed")
                break
            else:
                error = error or f"implementation exited without a result (timeout {timeout:.0f}s?)"
        except (OSError, ValueError) as exc:
            error = error or f"implementation channel failed: {exc}"
        finally:
            timer.cancel()
            proc.kill()
            proc.wait()
    return (None if error else output), error, calls


def replay_entry(contract: Dict[str, Any], source: str, entry: Dict[str, Any]) -> Optional[str]:
    """Equivalence check against one corpus entry; None when the implementation is equivalent."""
    if (error := validate(contract["input_schema"], entry["params"])) is not None:
        return f"corpus params: {error}"
    recorded = entry["results"]

    def answer(i: int, tool: str, args: Dict[str, Any]):
        return (recorded[i], None) if i < len(recorded) else (None, "call beyond the recorded trace")

    output, error, calls = drive(source, entry["params"], answer, timeout=REPLAY_TIMEOUT_S)
    if error is not None:
        return f"replay: {error}"
    expected = [[c["tool"], c["args"]] for c in entry["calls"]]
    if canonical([[t, a] for t, a in calls]) != canonical(expected):
        return "replay: the implementation's calls differ from the recorded trace"
    if canonical(output) != canonical({"steps": recorded}):
        return "replay: the implementation's output differs from the recorded trace"
    return check_result(contract, output)


class ContractGuard:
    """Answers live calls, enforcing the contract: declared call, exact args, still read-only."""

    def __init__(self, contract: Dict[str, Any], params: Dict[str, Any], run_task_id: str):
        self.contract, self.params, self.run_task_id = contract, params, run_task_id

    def __call__(self, i: int, tool: str, args: Dict[str, Any]):
        declared = self.contract["calls"]
        if i >= len(declared):
            return None, f"call {i + 1} is beyond the {len(declared)} declared calls"
        step = declared[i]
        expected = render_call(step, self.params)
        if tool != step["tool"] or canonical(args) != canonical(expected):
            return None, f"call {i + 1} ({tool}) is not the declared call"
        from tools.registry import registry
        effects = registry.resolve_effects(tool, expected)
        if not effects.idempotent or effects.destructive:
            return None, f"call {i + 1} ({tool}) no longer resolves to read-only"
        from model_tools import handle_function_call
        raw = handle_function_call(tool, dict(expected), task_id=self.run_task_id)
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        return (parsed if isinstance(parsed, dict) else {"error": "tool returned a non-JSON-object result"}), None


def run_live(contract: Dict[str, Any], source: str, params: Dict[str, Any], *, session_task_id: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Execute for real; ``(output, None)`` only when the contract's result check passes."""
    from tools.file_tools import clear_file_ops_cache
    from tools.terminal_tool import clear_task_env_overrides, record_session_cwd, register_container_alias
    run_task_id = f"{session_task_id or 'default'}::compiled::{uuid.uuid4().hex[:8]}"
    # Share the session's terminal sandbox (as delegate_task children do) and pin the contract's
    # working directory; read tracking stays per run, so a compiled read never turns a later
    # model read_file of the same file into an "unchanged" stub for content the model never saw.
    register_container_alias(run_task_id, session_task_id)
    record_session_cwd(run_task_id, contract["environment"]["cwd"])
    try:
        output, error, calls = drive(source, params, ContractGuard(contract, params, run_task_id), timeout=LIVE_TIMEOUT_S)
    finally:
        clear_task_env_overrides(run_task_id)
        clear_file_ops_cache(run_task_id)
    if error is not None:
        return None, error
    if len(calls) != len(contract["calls"]):
        return None, f"the implementation made {len(calls)} of {len(contract['calls'])} declared calls"
    if (error := check_result(contract, output)) is not None:
        return None, error
    return output, None
