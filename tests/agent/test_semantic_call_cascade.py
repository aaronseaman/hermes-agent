"""Verified cascades, admission and learning through the real semantic_call chain: temp HERMES_HOME
config.yaml → resolver → admission → auxiliary client → a local OpenAI-compatible stub (no keys)."""

import json
import time

import pytest
import yaml

from tests.fakes.fake_chat_server import fake_chat_server

SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    from agent.admission import ADMISSION
    from agent.auxiliary_client import _reset_aux_unhealthy_cache, shutdown_cached_clients
    from agent.capability_profile_ledger import reset_profiles
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    for reset in (shutdown_cached_clients, _reset_aux_unhealthy_cache, reset_profiles, ADMISSION.reset):
        reset()
    yield
    for reset in (shutdown_cached_clients, _reset_aux_unhealthy_cache, reset_profiles, ADMISSION.reset):
        reset()


def _config(tmp_path, monkeypatch, candidates, agent=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"auxiliary": {"extract": {"candidates": candidates}}, "agent": agent or {}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _route(url, model, usd, **extra):
    return {"provider": "custom", "base_url": url, "api_key": "sk-test-not-real", "model": model,
            "cost": {"input": usd, "output": usd}, **extra}


def _models(requests):
    return [payload["model"] for _path, payload, _headers in requests]


def _reply(bad=("cheap",)):
    def reply(payload):
        if payload["model"] in bad:
            return 200, "Sure! Here is a summary: it changed things."   # not JSON: fails the schema
        return 200, json.dumps({"summary": f"by {payload['model']}"})
    return reply


def _call(**kwargs):
    from agent.semantic_call import semantic_call
    kwargs.setdefault("output_schema", SCHEMA)
    kwargs.setdefault("policy", {"optimize": "cost"})
    return semantic_call("extract", ["a diff"], kwargs.pop("output_schema"), kwargs.pop("policy"),
                         max_tokens=50, **kwargs)


def _three(url):
    return [_route(url, "pricey", 30.0), _route(url, "cheap", 1.0), _route(url, "mid", 3.0)]


def test_a_cascade_escalates_exactly_on_schema_failure_and_stops_at_the_first_verified_success(tmp_path, monkeypatch):
    with fake_chat_server(_reply()) as (url, requests):
        _config(tmp_path, monkeypatch, _three(url))
        result = _call()
    assert _models(requests) == ["cheap", "mid"]           # cheapest first, one escalation, pricey never
    assert result.output == {"summary": "by mid"}
    assert "failed verification (malformed" in result.explanation and "escalating to" in result.explanation


def test_a_cascade_whose_cheapest_rung_verifies_sends_one_request(tmp_path, monkeypatch):
    with fake_chat_server(_reply(bad=())) as (url, requests):
        _config(tmp_path, monkeypatch, _three(url))
        assert _call().output == {"summary": "by cheap"}
    assert _models(requests) == ["cheap"]


def test_a_cascade_respects_max_attempts(tmp_path, monkeypatch):
    from agent.semantic_call import SemanticOutputError
    with fake_chat_server(_reply()) as (url, requests):
        _config(tmp_path, monkeypatch, _three(url), agent={"learned_routing": {"cascade": {"max_attempts": 1}}})
        with pytest.raises(SemanticOutputError) as err:
            _call()
    assert _models(requests) == ["cheap"]
    assert "cascade stopped before" in err.value.explanation and "max_attempts 1" in err.value.explanation


def test_a_cascade_never_starts_a_rung_the_cost_budget_cannot_cover(tmp_path, monkeypatch):
    """Each rung alone fits max_cost_usd, but the cheap attempt already spent plus mid's expected
    cost does not: the cascade stops instead of overspending."""
    from agent import semantic_call_io as io
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.semantic_call import SemanticOutputError
    n_in = estimate_messages_tokens_rough(io.build_messages("", ["a diff"])[0])
    mid_expected = (n_in + 50) * 3.0 / 1e6
    cheap_spent = min((n_in + 50) * 1.0, 11 + 7) / 1e6   # expected, or what the stub's usage bills
    with fake_chat_server(_reply()) as (url, requests):
        _config(tmp_path, monkeypatch, [_route(url, "cheap", 1.0), _route(url, "mid", 3.0)])
        with pytest.raises(SemanticOutputError) as err:
            _call(policy={"optimize": "cost", "max_cost_usd": mid_expected + cheap_spent / 2})
    assert _models(requests) == ["cheap"]
    assert "cascade stopped before" in err.value.explanation and "+ next ~$" in err.value.explanation


def test_without_a_verifier_there_is_no_cascade(tmp_path, monkeypatch):
    with fake_chat_server(_reply()) as (url, requests):
        _config(tmp_path, monkeypatch, _three(url))
        result = _call(output_schema=None)
    assert _models(requests) == ["cheap"]
    assert result.text.startswith("Sure!") and result.output is None


def test_a_caller_predicate_is_a_verifier(tmp_path, monkeypatch):
    with fake_chat_server(_reply(bad=())) as (url, requests):
        _config(tmp_path, monkeypatch, _three(url))
        result = _call(verify=lambda out: (out["summary"] != "by cheap", ["cheap summaries are not trusted"]))
    assert _models(requests) == ["cheap", "mid"]
    assert "rejected: cheap summaries are not trusted" in result.explanation


def test_other_objectives_do_not_cascade(tmp_path, monkeypatch):
    from agent.semantic_call import SemanticOutputError
    with fake_chat_server(_reply()) as (url, requests):
        _config(tmp_path, monkeypatch, [_route(url, "cheap", 1.0, latency_ms=100), _route(url, "mid", 3.0)])
        with pytest.raises(SemanticOutputError):
            _call(policy={"optimize": "latency"})
    assert _models(requests) == ["cheap"]


class _Agent:
    """What call_ledger.begin_turn reads from an agent."""

    def __init__(self):
        from agent.call_ledger import LedgerSettings
        self._call_ledger_settings = LedgerSettings(enabled=True)
        self.session_id, self.platform = "session-e2e", "cli"


def _ledger_records(home, kind):
    from agent.call_ledger_store import WRITER
    assert WRITER.flush(timeout=10)
    rows = []
    for path in sorted((home / "call_ledger").glob("calls-*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r.get("kind") == kind]


def test_e2e_escalation_is_recorded_and_later_selections_follow_the_outcomes(tmp_path, monkeypatch):
    """The cheap model keeps returning prose instead of JSON. Every escalation lands in the ledger,
    and once enough outcomes are recorded the cascade starts at the model that verifies."""
    from agent import call_ledger
    agent_cfg = {"call_ledger": {"enabled": True},
                 "learned_routing": {"exploration_rate": 0, "prior_strength": 1, "min_samples": 2}}
    per_call = []
    with fake_chat_server(_reply()) as (url, requests):
        _config(tmp_path, monkeypatch, [_route(url, "cheap", 1.0), _route(url, "mid", 3.0)], agent=agent_cfg)
        agent = _Agent()
        for i in range(4):
            token = call_ledger.begin_turn(agent, f"task-{i}")
            try:
                before = len(requests)
                assert _call().output == {"summary": "by mid"}
                per_call.append(_models(requests[before:]))
            finally:
                call_ledger.end_turn(token, outcome="success")
    assert per_call[0] == ["cheap", "mid"]      # no measurements: the prior says cheap first
    assert per_call[-1] == ["mid"]              # measured: cheap never verifies, so start at mid
    shift = per_call.index(["mid"])
    assert all(calls == ["cheap", "mid"] for calls in per_call[:shift])

    attempts = _ledger_records(tmp_path, "capability")
    outcomes = [(a["model"], a["outcome"], a["step"]) for a in attempts]
    assert outcomes[:2] == [("cheap", "malformed", 1), ("mid", "ok", 2)]
    assert len(attempts) == sum(len(calls) for calls in per_call)
    assert all(a["latency_ms"] >= 0 and a["cost_usd"] is not None and a["cascade"] for a in attempts)
    # The model calls themselves are still counted once, by the auxiliary seam.
    assert len(_ledger_records(tmp_path, "model")) == len(attempts)


def _patched_kind(monkeypatch, fail):
    """Replace the model kind's invoker: ``fail(candidate)`` may raise before the real call."""
    from agent import semantic_call, semantic_call_models
    real = semantic_call._KINDS["model"]

    def invoke(candidate, *args, **kwargs):
        fail(candidate)
        return semantic_call_models.invoke(candidate, *args, **kwargs)

    monkeypatch.setitem(semantic_call._KINDS, "model", real._replace(invoke=invoke))


def test_a_killed_attempt_releases_its_slot_and_a_retry_runs_cleanly(tmp_path, monkeypatch):
    """Restartability: an interrupt mid-attempt leaks no admission slot, makes nothing the incumbent
    and is recorded as cancelled, which never counts against the implementation."""
    from agent import call_ledger
    from agent.capability_profile_ledger import profile_for
    from agent.capability_resolver import _INCUMBENTS, _incumbent_key
    killed = []

    def kill_once(candidate):
        if not killed:
            killed.append(candidate)
            raise KeyboardInterrupt

    with fake_chat_server(_reply(bad=())) as (url, requests):
        _config(tmp_path, monkeypatch, [_route(url, "cheap", 1.0)],
                agent={"call_ledger": {"enabled": True},
                       "admission": {"max_wait_s": 0.5, "ceilings": {"endpoint:127.0.0.1": 1}}})
        _patched_kind(monkeypatch, kill_once)
        token = call_ledger.begin_turn(_Agent(), "task-kill")
        try:
            with pytest.raises(KeyboardInterrupt):
                _call()
            assert _INCUMBENTS.get(_incumbent_key("extract", "")) is None
            assert profile_for().observe("extract", killed[0].identity) is None
            assert _call().output == {"summary": "by cheap"}   # the slot came back: admitted at once
        finally:
            call_ledger.end_turn(token, outcome="success")
    assert _models(requests) == ["cheap"]
    assert [a["outcome"] for a in _ledger_records(tmp_path, "capability")] == ["cancelled", "ok"]


def test_a_rate_limited_provider_is_backed_off_before_the_next_attempt(tmp_path, monkeypatch):
    import httpx
    import openai

    def rate_limit_busy(candidate):
        if candidate.target.get("model") == "busy":
            response = httpx.Response(429, headers={"retry-after": "0.4"},
                                      request=httpx.Request("POST", "http://127.0.0.1/v1/chat/completions"))
            raise openai.RateLimitError("Rate limit reached", response=response, body=None)

    with fake_chat_server(_reply(bad=())) as (url, requests):
        _config(tmp_path, monkeypatch, [_route(url, "busy", 1.0), _route(url, "cheap", 2.0)],
                agent={"admission": {"backoff_base_s": 0.2}})
        _patched_kind(monkeypatch, rate_limit_busy)
        started = time.monotonic()
        result = _call(output_schema=None)
        elapsed = time.monotonic() - started
    # Both candidates sit on provider:custom @ 127.0.0.1, so the fallback waits out the backoff
    # (Retry-After 0.4s beats the 0.2s base) instead of hammering the same provider.
    assert result.model == "cheap" and _models(requests) == ["cheap"]
    assert elapsed >= 0.4
    assert "busy" not in _models(requests)
