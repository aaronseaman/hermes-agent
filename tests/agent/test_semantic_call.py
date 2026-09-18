"""semantic_call E2E through the real resolution chain: temp HERMES_HOME config.yaml → resolver →
auxiliary client → a local OpenAI-compatible stub (no real keys, no network)."""

import json
import socket

import pytest
import yaml

from tests.fakes.fake_chat_server import fake_chat_server

SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
REMOTE_HOST = "api.remote-frontier.test"


@pytest.fixture(autouse=True)
def _fresh_aux_state(monkeypatch):
    from agent.auxiliary_client import _reset_aux_unhealthy_cache, shutdown_cached_clients
    monkeypatch.setenv("NO_PROXY", f"127.0.0.1,localhost,{REMOTE_HOST}")
    resolve_address = socket.getaddrinfo

    def _resolve(host, *args, **kwargs):  # a non-local hostname that still lands on the stub
        return resolve_address("127.0.0.1" if host in (REMOTE_HOST, REMOTE_HOST.encode()) else host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _resolve)
    shutdown_cached_clients()
    _reset_aux_unhealthy_cache()
    yield
    shutdown_cached_clients()
    _reset_aux_unhealthy_cache()


def _write_config(tmp_path, monkeypatch, auxiliary, **extra):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"auxiliary": auxiliary, **extra}), encoding="utf-8")


def _route(url, model, **priors):
    return {"provider": "custom", "base_url": url, "api_key": "sk-test-not-real", "model": model, **priors}


def _models(requests):
    return [payload["model"] for _path, payload, _headers in requests]


def _summary(payload):
    return 200, json.dumps({"summary": f"by {payload['model']}"})


def test_policy_selects_the_implementation_and_validates_output(tmp_path, monkeypatch):
    from agent.semantic_call import semantic_call
    with fake_chat_server(_summary) as (url, requests):
        _write_config(tmp_path, monkeypatch, {"summarize": {"candidates": [
            _route(url, "pricey", cost={"input": 15, "output": 75}, quality="high"),
            _route(url, "cheap", cost={"input": 0.1, "output": 0.4}, quality="medium"),
        ]}})
        result = semantic_call("summarize", ["a long diff"], SCHEMA, {"optimize": "cost"},
                               instructions="Summarize.")
    assert _models(requests) == ["cheap"]
    assert result.output == {"summary": "by cheap"}
    assert "candidates[1]" in result.explanation and "optimize=cost" in result.explanation
    # Explicit inputs only: one system + one user message, nothing from any transcript.
    assert [m["role"] for m in requests[0][1]["messages"]] == ["system", "user"]
    assert requests[0][1]["response_format"]["json_schema"]["schema"] == SCHEMA


def test_schema_invalid_output_is_rejected(tmp_path, monkeypatch):
    from agent.semantic_call import SemanticOutputError, semantic_call
    with fake_chat_server(lambda payload: (200, '{"summary": 42}')) as (url, _requests):
        _write_config(tmp_path, monkeypatch, {"summarize": _route(url, "m")})
        with pytest.raises(SemanticOutputError) as err:
            semantic_call("summarize", ["x"], SCHEMA)
    assert err.value.text == '{"summary": 42}'
    assert err.value.errors


def test_unsatisfiable_policy_sends_nothing(tmp_path, monkeypatch):
    from agent.capability_resolver import UnsatisfiablePolicyError
    from agent.semantic_call import semantic_call
    with fake_chat_server(_summary) as (url, requests):
        remote = url.replace("127.0.0.1", REMOTE_HOST)
        _write_config(tmp_path, monkeypatch, {"summarize": _route(remote, "frontier", quality="high")})
        with pytest.raises(UnsatisfiablePolicyError, match="local_only"):
            semantic_call("summarize", ["x"], SCHEMA, {"local_only": True})
    assert requests == []


@pytest.mark.parametrize("policy,expected", [
    ({"local_only": True}, ["local-down", "local-ok"]),   # fenced: the resolver moves to the next local candidate
    (None, ["local-down", "remote-frontier"]),            # unfenced: auxiliary fallback behaves as for any task
])
def test_constrained_policy_never_falls_back_outside_admitted_candidates(tmp_path, monkeypatch, policy, expected):
    from agent.semantic_call import semantic_call

    def reply(payload):
        if payload["model"] == "local-down":
            return 402, "Insufficient credits"
        return _summary(payload)

    with fake_chat_server(reply) as (url, requests):
        remote = url.replace("127.0.0.1", REMOTE_HOST)
        _write_config(tmp_path, monkeypatch, {"summarize": {
            "fallback_chain": [_route(remote, "remote-frontier")],
            "candidates": [_route(url, "local-down"), _route(url, "local-ok")],
        }})
        result = semantic_call("summarize", ["x"], SCHEMA, policy)
    assert _models(requests) == expected
    assert result.output == {"summary": f"by {expected[-1]}"}


def test_calls_are_recorded_by_auxiliary_accounting_under_the_capability(tmp_path, monkeypatch):
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.semantic_call import semantic_call

    recorded = []

    class SessionDB:
        def record_auxiliary_usage(self, session_id, task, **usage):
            recorded.append((session_id, task, usage))

    with fake_chat_server(_summary) as (url, _requests):
        _write_config(tmp_path, monkeypatch, {"summarize": _route(url, "m")})
        token = set_accounting_context(SessionDB(), "session-1")
        try:
            semantic_call("summarize", ["x"], SCHEMA)
        finally:
            reset_accounting_context(token)
    assert [(sid, task) for sid, task, _ in recorded] == [("session-1", "summarize")]
    assert recorded[0][2]["input_tokens"] == 11 and recorded[0][2]["output_tokens"] == 7


def test_migrated_title_generation_sends_the_same_request_as_the_direct_auxiliary_call(tmp_path, monkeypatch):
    """Default config: generate_title (now via semantic_call) and the auxiliary call it replaced put
    the same request on the wire."""
    from agent import title_generator
    from agent.auxiliary_client import call_llm

    with fake_chat_server(lambda payload: (200, '{"title": "Fix the login button"}')) as (url, requests):
        _write_config(tmp_path, monkeypatch, {}, model={"provider": "custom", "default": "main-model",
                                                          "base_url": url, "api_key": "sk-test-not-real"})
        assert title_generator.generate_title("the login button is broken on desktop") == "Fix the login button"
        prompt = title_generator._TITLE_PROMPT_TEMPLATE.replace(
            "__LANGUAGE_RULE__", title_generator._LANGUAGE_RULE_MATCH_USER)
        call_llm(task="title_generation",
                 messages=[{"role": "system", "content": prompt},
                           {"role": "user", "content": "the login button is broken on desktop"}],
                 max_tokens=64, temperature=0.3,
                 extra_body={"response_format": {"type": "json_schema", "json_schema": {
                     "name": "session_title", "strict": True, "schema": title_generator._TITLE_OUTPUT.schema}}},
                 reasoning_config={"enabled": False})
    assert len(requests) == 2
    assert requests[0][1] == requests[1][1]
