"""semantic_call from an execute_code sandbox: the real kernel, RPC pipeline (token → allow-list →
budget → dispatch), resolver and auxiliary client against a local stub provider."""

import json
import threading
import time

import pytest
import yaml

from tests.fakes.fake_chat_server import fake_chat_server

SECRET = "sk-sandbox-secret-not-real"
SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}

@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from agent.auxiliary_client import _reset_aux_unhealthy_cache, shutdown_cached_clients
    from tools.code_kernel import shutdown_all_kernels
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    shutdown_all_kernels()
    shutdown_cached_clients()
    _reset_aux_unhealthy_cache()
    yield
    shutdown_all_kernels()
    shutdown_cached_clients()


def _configure(tmp_path, monkeypatch, url, *, max_tool_calls=50):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SUMMARIZE_API_KEY", SECRET)
    config = {
        "code_execution": {"max_tool_calls": max_tool_calls},
        "auxiliary": {
            "summarize": {"sandbox": True, "provider": "custom", "base_url": url, "key_env": "SUMMARIZE_API_KEY",
                          "model": "stub-model"},
            "not_exposed": {"provider": "custom", "base_url": url, "key_env": "SUMMARIZE_API_KEY", "model": "m"},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _run(code):
    from tools.code_execution_tool import execute_code
    return json.loads(execute_code(code=code, task_id="semantic-sandbox", enabled_tools=["terminal"]))


def _reply(payload):
    return 200, json.dumps({"summary": "short"})


def test_sandbox_semantic_call_passes_allow_list_and_never_sees_credentials(tmp_path, monkeypatch):
    code = f"""
import json, os
from hermes_tools import semantic_call
ok = semantic_call("summarize", ["some text"], output_schema={json.dumps(SCHEMA)}, instructions="Summarize.")
refused = semantic_call("not_exposed", ["x"])
print(json.dumps({{"ok": ok, "refused": refused,
                  "secret_in_env": any("{SECRET}" in v for v in os.environ.values())}}))
"""
    with fake_chat_server(_reply) as (url, requests):
        _configure(tmp_path, monkeypatch, url)
        result = _run(code)
    assert result["status"] == "success", result
    out = json.loads(result["output"].strip().splitlines()[-1])
    assert out["ok"]["output"] == {"summary": "short"}
    assert "summarize" in out["ok"]["explanation"]
    assert "not available in execute_code" in out["refused"]["error"]
    assert out["secret_in_env"] is False
    assert SECRET not in result["output"]
    # The credential was used, host-side only: exactly one request, carrying it.
    assert len(requests) == 1
    assert requests[0][2].get("Authorization") == f"Bearer {SECRET}"


def test_sandbox_semantic_calls_count_against_the_call_budget(tmp_path, monkeypatch):
    code = """
import json
from hermes_tools import semantic_call
print(json.dumps([semantic_call("summarize", ["x"]) for _ in range(2)]))
"""
    with fake_chat_server(_reply) as (url, requests):
        _configure(tmp_path, monkeypatch, url, max_tool_calls=1)
        result = _run(code)
    first, second = json.loads(result["output"].strip().splitlines()[-1])
    assert "text" in first
    assert "Tool call limit reached" in second["error"]
    assert len(requests) == 1


def test_semantic_call_is_absent_from_the_sandbox_unless_configured(tmp_path, monkeypatch):
    from tools.code_execution_tool import build_execute_code_schema, _sandbox_tools_for
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"auxiliary": {"summarize": {"model": "m"}}}))
    tools = _sandbox_tools_for(["terminal"])
    assert "semantic_call" not in tools
    assert "semantic_call(" not in build_execute_code_schema(set(tools))["description"]


def test_ending_cell_cancels_an_in_flight_sandbox_semantic_call(tmp_path, monkeypatch):
    from tools.code_execution_semantic import handle_rpc
    release = threading.Event()

    def slow(payload):
        release.wait(20)
        return _reply(payload)

    with fake_chat_server(slow) as (url, requests):
        _configure(tmp_path, monkeypatch, url)
        ended_at = time.monotonic() + 0.5
        started = time.monotonic()
        try:
            result = json.loads(handle_rpc({"capability": "summarize", "inputs": ["x"]}, task_id="t",
                                           cancelled=lambda: time.monotonic() > ended_at))
        finally:
            release.set()
    assert "cancelled" in result["error"]
    assert time.monotonic() - started < 10


def test_cell_authority_reports_cancelled_on_stop_or_settle():
    from tools.code_kernel import CellAuthority
    from tools.interrupt import set_interrupt
    authority = CellAuthority("t")
    assert authority.cancelled() is False
    set_interrupt(True, authority.owner_tid)
    try:
        assert authority.cancelled() is True
    finally:
        set_interrupt(False, authority.owner_tid)
    authority.retire()
    assert authority.cancelled() is True
