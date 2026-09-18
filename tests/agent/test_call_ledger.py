"""Behaviour contracts for the local call ledger (agent/call_ledger.py).

A real AIAgent turn (scripted provider responses, real tool dispatch, real config loader) runs
against a temp HERMES_HOME, and the records on disk must account for exactly the calls the turn
made, never carry raw tool arguments, and never change the turn when the ledger cannot write.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_constants import get_hermes_home

SECRET = "sk-live-9f8e7d6c5b4a"


@pytest.fixture(autouse=True)
def _no_background_titles(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)


def _enable_ledger(home: Path, enabled: bool = True) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"agent": {"call_ledger": {"enabled": enabled}}}), encoding="utf-8")


def _usage(prompt, cached, out):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached), completion_tokens_details=None,
    )


def _response(*, content="", tool_calls=None, usage=None):
    finish = "tool_calls" if tool_calls else "stop"
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], model="test/model", usage=usage)


def _read_call(call_id, path):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name="read_file", arguments=json.dumps({"path": str(path)})))


def _agent():
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            enabled_toolsets=["file"], session_id="ledger-session", platform="cli",
        )
    agent.client = MagicMock()
    return agent


def _script_turn(agent, target: Path):
    """3 model calls, 3 tool calls: a batch of two identical reads, then the same read again."""
    agent.client.chat.completions.create.side_effect = [
        _response(tool_calls=[_read_call("c1", target), _read_call("c2", target)], usage=_usage(1000, 0, 40)),
        _response(tool_calls=[_read_call("c3", target)], usage=_usage(1400, 900, 30)),
        _response(content="done", usage=_usage(1600, 1300, 20)),
    ]


def _records(home: Path) -> list[dict]:
    from agent.call_ledger_report import load_records
    from agent.call_ledger_store import WRITER

    assert WRITER.flush(timeout=5.0)
    return load_records([home / "call_ledger"])


def test_turn_records_match_calls_and_never_store_raw_args(tmp_path):
    home = get_hermes_home()
    _enable_ledger(home)
    target = tmp_path / f"notes-{SECRET}.txt"
    target.write_text("hello\n", encoding="utf-8")
    agent = _agent()
    _script_turn(agent, target)
    try:
        result = agent.run_conversation("read the notes")
    finally:
        agent.close()

    assert result["final_response"] == "done"
    records = _records(home)
    model = [r for r in records if r["kind"] == "model"]
    tools = [r for r in records if r["kind"] == "tool"]
    turns = [r for r in records if r["kind"] == "turn"]
    assert len(model) == agent.client.chat.completions.create.call_count
    assert len(tools) == sum(1 for m in result["messages"] if m.get("role") == "tool")
    assert len(turns) == 1
    turn = turns[0]
    # The summary is the sum of its events, and every event belongs to that turn.
    assert {r["turn_id"] for r in records} == {turn["turn_id"]}
    assert (turn["model_calls"], turn["tool_calls"]) == (len(model), len(tools))
    for key in ("input_tokens", "cache_read_tokens", "output_tokens"):
        assert turn[key] == sum(m[key] for m in model)
    assert turn["repeated_signatures"] == sum(t["repeated_in_turn"] for t in tools)
    # Identical (tool, args) share one signature; only later calls count as repeats.
    assert len({t["signature"] for t in tools}) == 1
    assert [t["repeated_in_turn"] for t in tools].count(False) == 1
    assert all(t["effects"]["idempotent"] for t in tools)
    # Raw arguments (which may carry secrets) never reach disk.
    on_disk = "".join(p.read_text(encoding="utf-8") for p in (home / "call_ledger").glob("*.jsonl"))
    assert SECRET not in on_disk and str(tmp_path) not in on_disk


@pytest.mark.parametrize("breakage", ["writer_raises", "unwritable_dir"])
def test_ledger_failure_does_not_change_the_turn(tmp_path, monkeypatch, breakage):
    home = get_hermes_home()
    target = tmp_path / "notes.txt"
    target.write_text("hello\n", encoding="utf-8")

    def _run():
        agent = _agent()
        _script_turn(agent, target)
        try:
            result = agent.run_conversation("read the notes")
        finally:
            agent.close()
        return result["final_response"], [m.get("content") for m in result["messages"] if m.get("role") == "tool"]

    _enable_ledger(home, enabled=False)
    baseline = _run()

    _enable_ledger(home, enabled=True)
    if breakage == "writer_raises":
        from agent.call_ledger_store import WRITER

        def _boom(*a, **k):
            raise OSError("disk gone")
        monkeypatch.setattr(WRITER, "submit", _boom)
    else:
        (home / "call_ledger").write_text("not a directory", encoding="utf-8")

    assert _run() == baseline


def test_records_follow_the_profile_home_bound_at_turn_start(tmp_path, monkeypatch):
    """A→B→A: each turn's records land in the home that was active when the turn began."""
    from agent import call_ledger
    from agent.call_ledger_report import load_records
    from agent.call_ledger_store import WRITER

    homes = {"a": tmp_path / "a", "b": tmp_path / "b"}
    class _Agent:
        _call_ledger_settings = call_ledger.LedgerSettings(enabled=True)
        session_id, platform, provider, model, api_mode = "s", "cli", "p", "m", "chat_completions"

    agent = _Agent()
    for name in ("a", "b", "a"):
        monkeypatch.setenv("HERMES_HOME", str(homes[name]))
        token = call_ledger.begin_turn(agent, f"task-{name}")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "elsewhere"))  # a later switch must not move the turn
        call_ledger.record_model_call(agent, usage=None, cost_usd=None, latency_s=0.1)
        call_ledger.end_turn(token, outcome="success")
    assert WRITER.flush(timeout=5.0)

    assert [r["task_id"] for r in load_records([homes["a"] / "call_ledger"]) if r["kind"] == "turn"] == ["task-a", "task-a"]
    assert [r["task_id"] for r in load_records([homes["b"] / "call_ledger"]) if r["kind"] == "turn"] == ["task-b"]
    assert not (tmp_path / "elsewhere").exists()


def test_guardrail_code_is_read_from_the_guardrails_own_output():
    from agent.call_ledger import guardrail_code
    from agent.tool_guardrails import ToolGuardrailDecision, append_toolguard_guidance, toolguard_synthetic_result

    warn = ToolGuardrailDecision("warn", "idempotent_no_progress_warning", "no progress", "read_file", 2)
    block = ToolGuardrailDecision("block", "repeated_exact_failure_block", "stop", "terminal", 3)
    assert guardrail_code(append_toolguard_guidance("body", warn), blocked=False) == warn.code
    assert guardrail_code(toolguard_synthetic_result(block), blocked=True) == block.code
    assert guardrail_code(json.dumps({"error": "blocked by plugin"}), blocked=True) == "blocked"
    assert guardrail_code("plain result", blocked=False) is None


def test_retention_drops_expired_then_oldest_files(tmp_path):
    from agent.call_ledger_store import prune

    now = 1_790_000_000.0  # 2026-09-21 UTC
    names = ["calls-2026-08-01-1.jsonl", "calls-2026-09-19-1.jsonl", "calls-2026-09-20-1.jsonl", "calls-2026-09-21-1.jsonl"]
    for name in names:
        (tmp_path / name).write_bytes(b"x" * 600_000)
    (tmp_path / "unrelated.txt").write_text("keep", encoding="utf-8")

    prune(tmp_path, retention_days=14, max_mb=1, now=now)

    # Expired file gone; over the 1 MB cap the oldest in-window files go; the newest file stays.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calls-2026-09-21-1.jsonl", "unrelated.txt"]
