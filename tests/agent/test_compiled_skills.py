"""Compiled skills (roadmap stage 5): the whole conservative loop, end to end.

Real imports against the temp ``HERMES_HOME`` from ``tests/conftest.py``: synthetic SessionDB
sessions whose tool results are real outputs (``model_tools.handle_function_call`` on real files,
a real ``terminal``), the real curator pass, real ``skill_manage`` writes, the real isolated replay
interpreter, and the CLI's ``/<skill>`` handler. No API keys and no model: a model client
constructed anywhere fails the test.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import compiled_skill
from agent.compiled_skill_contract import CONTRACT_PATH, IMPLEMENTATION_PATH, sha256_text
from hermes_state import SessionDB
from model_tools import handle_function_call
from tools.terminal_tool import record_session_cwd

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "notes").mkdir(parents=True)
    (ws / "docs").mkdir()
    (ws / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    for name in ("alpha", "beta", "gamma", "delta"):
        (ws / "notes" / f"{name}.md").write_text(f"# {name}\nstatus: ok\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(ws))
    monkeypatch.chdir(ws)
    return ws


@pytest.fixture
def home():
    hermes_home = Path(os.environ["HERMES_HOME"])
    (hermes_home / "config.yaml").write_text(
        "curator:\n  compiled_skills:\n    enabled: true\n    min_sessions: 3\n    max_consecutive_failures: 2\n",
        encoding="utf-8")
    return hermes_home


@pytest.fixture
def no_model(monkeypatch):
    """Any main-loop or auxiliary model client construction fails the test."""
    import openai
    import run_agent

    def _boom(*_a, **_k):
        raise AssertionError("a model client was constructed")
    monkeypatch.setattr(run_agent, "AIAgent", _boom)
    monkeypatch.setattr(openai, "OpenAI", _boom)


def _record_session(db, sid, ws, calls):
    """One user turn that ran *calls* for real and ended in a final answer."""
    db.create_session(sid, source="cli", cwd=str(ws))
    db.append_message(sid, "user", content="check the note and the docs")
    tool_calls = [{"id": f"{sid}-{i}", "type": "function",
                   "function": {"name": tool, "arguments": json.dumps(args)}} for i, (tool, args) in enumerate(calls)]
    db.append_message(sid, "assistant", content="", tool_calls=tool_calls)
    record_session_cwd(f"source-{sid}", str(ws))  # the source session's own working directory
    for call, (tool, args) in zip(tool_calls, calls):
        result = handle_function_call(tool, args, task_id=f"source-{sid}")
        db.append_message(sid, "tool", content=result, tool_call_id=call["id"], tool_name=tool)
    db.append_message(sid, "assistant", content="Here is the note and the docs listing.")


def _note_then_ls(note):
    return [("read_file", {"path": f"notes/{note}.md"}), ("terminal", {"command": "ls docs"})]


def _run_cli_slash(cmd_key, rest):
    """The CLI's real ``/<skill>`` handler on a minimal CLI object; returns (printed, queued)."""
    import cli
    printed, queued = [], []
    fake = SimpleNamespace(session_id="cli-session", enabled_toolsets=None, disabled_toolsets=None,
                           _queue_loaded_skills=lambda msg, label, missing: queued.append(msg))
    original = cli._cprint
    cli._cprint = lambda text, *a, **k: printed.append(str(text))
    try:
        cli.HermesCLI._run_skill_slash_command(fake, cmd_key, {"name": cmd_key.lstrip("/")}, rest)
    finally:
        cli._cprint = original
    return printed, queued


def _compiled_record(name):
    from tools import skill_usage
    return skill_usage.get_record(name).get("compiled") or {}


def _record_three_sessions(ws):
    db = SessionDB()
    for i, note in enumerate(("alpha", "beta", "gamma")):
        _record_session(db, f"s{i}", ws, _note_then_ls(note))
    db.close()


def test_repeated_read_only_procedure_is_compiled_run_without_a_model_and_falls_back_on_drift(
        workspace, home, no_model):
    _record_three_sessions(workspace)

    # Mined, compiled through skill_manage, replay-validated and activated by the curator pass.
    counts = compiled_skill.curator_pass()
    assert counts.get("created") == 1 and counts.get("activated") == 1, counts
    (name, skill_dir), = compiled_skill.compiled_skill_dirs().items()
    contract = json.loads((skill_dir / CONTRACT_PATH).read_text(encoding="utf-8"))
    assert list(contract["input_schema"]["properties"]) == ["path"]
    assert [c["tool"] for c in contract["calls"]] == ["read_file", "terminal"]
    assert contract["calls"][1]["params"] == {}, "terminal args must stay constant"
    assert len(contract["corpus"]) == 3
    from tools import skill_usage
    assert skill_usage.is_curator_managed(name), "a compiled skill is a curator-managed skill"
    assert _compiled_record(name)["state"] == "active"

    # A new matching request runs deterministically: printed, nothing queued for the model.
    cmd_key = f"/{name}"
    from agent.skill_commands import scan_skill_commands
    scan_skill_commands()
    printed, queued = _run_cli_slash(cmd_key, "path=notes/delta.md")
    assert queued == [] and printed, (printed, queued)
    assert "# delta" in printed[0] and "guide.md" in printed[0]
    assert _compiled_record(name)["runs"] == 1
    assert skill_usage.get_record(name)["use_count"] == 1, "usage is tracked like any skill load"

    # A request outside the preconditions (free text) takes the normal skill path, no drift.
    printed, queued = _run_cli_slash(cmd_key, "summarise every note please")
    assert printed == [] and len(queued) == 1
    assert "compiled fast path" not in queued[0]
    assert not _compiled_record(name).get("drift")

    # The environment changes: the result check fails, the agent path takes over, drift is recorded.
    (workspace / "docs" / "guide.md").unlink()
    (workspace / "docs").rmdir()
    printed, queued = _run_cli_slash(cmd_key, "path=notes/delta.md")
    assert printed == [] and len(queued) == 1
    assert "compiled fast path for this skill did not complete" in queued[0]
    rec = _compiled_record(name)
    assert rec["consecutive_failures"] == 1 and "terminal" in rec["drift"][-1]["reason"]

    # Repeated failures: the curator retires the compiled path; the skill itself stays.
    _run_cli_slash(cmd_key, "path=notes/alpha.md")
    assert compiled_skill.curator_pass().get("retired") == 1
    assert _compiled_record(name)["state"] == "retired"
    assert (skill_dir / "SKILL.md").is_file()
    printed, queued = _run_cli_slash(cmd_key, "path=notes/delta.md")
    assert printed == [] and len(queued) == 1 and "compiled fast path" not in queued[0]


def _trace(sid, calls, *, cwd="/work"):
    from agent.compiled_skill_traces import ProcedureTrace, TraceStep
    ok = {"content": "1|x", "total_lines": 1, "output": "x", "exit_code": 0, "error": None}
    return ProcedureTrace(sid, cwd, tuple(TraceStep(t, a, dict(ok)) for t, a in calls), True)


@pytest.mark.parametrize(("second_call", "compilable"), [
    (lambda note: ("terminal", {"command": "ls docs"}), True),  # control: read-only, constant
    (lambda note: ("write_file", {"path": "out.md", "content": note}), False),  # writes
    (lambda note: ("terminal", {"command": "touch stamp"}), False),  # not provably read-only
    (lambda note: ("terminal", {"command": f"cat notes/{note}.md"}), False),  # read-only, but parameterised
    (lambda note: ("mcp_unknown_server_lookup", {"q": note}), False),  # effects unknown
])
def test_only_procedures_whose_every_call_is_read_only_with_fixed_effects_are_compiled(second_call, compilable):
    import model_tools  # noqa: F401  (registers the built-in tools)
    from agent.compiled_skill_mining import mine_candidates
    traces = [_trace(f"s{i}", [("read_file", {"path": f"notes/{note}.md"}), second_call(note)])
              for i, note in enumerate(("alpha", "beta", "gamma", "delta"))]
    candidates, _rejected = mine_candidates(traces, min_sessions=3)
    assert bool(candidates) is compilable
    if compilable:
        assert list(candidates[0].params) == ["path"]


def test_an_implementation_can_never_make_a_call_its_contract_does_not_declare(workspace, home, no_model):
    from agent.compiled_skill_contract import load
    from agent.compiled_skill_runtime import run_live
    _record_three_sessions(workspace)
    counts = compiled_skill.curator_pass()
    assert counts.get("activated") == 1, counts
    (name, skill_dir), = compiled_skill.compiled_skill_dirs().items()
    contract, source = load(skill_dir)
    target = workspace / "written.txt"
    tampered = source.replace(
        "    return {", f"    call('write_file', {{'path': {str(target)!r}, 'content': 'x'}})\n    return {{", 1)

    # The guard refuses the undeclared call: nothing is written, the run fails.
    output, error = run_live(contract, tampered, {"path": "notes/delta.md"}, session_task_id="t")
    assert output is None and "beyond the 2 declared calls" in error
    assert not target.exists()

    # Installed on disk with a matching pinned hash, it is not run until re-validated, and the
    # curator's replay rejects it (its calls differ from every recorded trace).
    (skill_dir / IMPLEMENTATION_PATH).write_text(tampered, encoding="utf-8")
    contract["implementation"]["sha256"] = sha256_text(tampered)
    (skill_dir / CONTRACT_PATH).write_text(json.dumps(contract), encoding="utf-8")
    from agent.skill_commands import scan_skill_commands
    scan_skill_commands()
    printed, queued = _run_cli_slash(f"/{name}", "path=notes/delta.md")
    assert printed == [] and len(queued) == 1 and not target.exists()
    compiled_skill.curator_pass()
    assert _compiled_record(name)["state"] == "rejected"
    assert not target.exists()


def test_curator_pass_and_invocation_stay_in_the_bound_profile(workspace, home, tmp_path, no_model):
    """A -> B -> A: mining reads only the bound profile's sessions, and a compiled skill exists and
    runs only in the profile that compiled it."""
    from agent.skill_commands import get_skill_commands
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    _record_three_sessions(workspace)
    assert compiled_skill.curator_pass().get("activated") == 1
    (name, _), = compiled_skill.compiled_skill_dirs().items()
    allowed = compiled_skill.session_tool_names(None, None)

    other = tmp_path / "profile-b"
    other.mkdir()
    (other / "config.yaml").write_text((home / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    token = set_hermes_home_override(other)
    try:
        assert compiled_skill.curator_pass().get("mined") == 0
        assert compiled_skill.compiled_skill_dirs() == {}
        assert f"/{name}" not in get_skill_commands()
        assert compiled_skill.run_skill_command(f"/{name}", "path=notes/delta.md", task_id="b",
                                                allowed_tools=allowed).reply is None
    finally:
        reset_hermes_home_override(token)

    assert f"/{name}" in get_skill_commands()
    reply = compiled_skill.run_skill_command(f"/{name}", "path=notes/delta.md", task_id="a", allowed_tools=allowed).reply
    assert reply is not None and "# delta" in reply
