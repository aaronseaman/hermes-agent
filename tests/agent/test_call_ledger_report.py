"""The baseline report's numbers are derived from, and consistent with, the ledger records."""

import json

from agent.call_ledger_report import build_report, load_records


def _model(turn, session, *, inp, read, write, out, cost, latency, outcome="ok", role="main"):
    return {"kind": "model", "turn_id": turn, "session_id": session, "call_role": role, "usage_available": outcome == "ok",
            "input_tokens": inp, "cache_read_tokens": read, "cache_write_tokens": write, "output_tokens": out,
            "cost_usd": cost, "latency_s": latency, "outcome": outcome, "error_class": None if outcome == "ok" else "timeout"}


def _tool(turn, session, *, sig, idempotent, repeated_turn, repeated_session, duration, blocked=False):
    return {"kind": "tool", "turn_id": turn, "session_id": session, "signature": sig, "effects": {"idempotent": idempotent},
            "repeated_in_turn": repeated_turn, "repeated_in_session": repeated_session, "duration_s": duration,
            "blocked": blocked, "failed": False, "spilled": False, "guardrail": None}


RECORDS = [
    _model("t1", "s1", inp=100, read=0, write=400, out=10, cost=0.01, latency=1.0),
    _model("t1", "s1", inp=50, read=450, write=0, out=20, cost=0.002, latency=3.0),
    _model("t1", "s1", inp=0, read=0, write=0, out=0, cost=None, latency=9.0, outcome="error"),
    _tool("t1", "s1", sig="a", idempotent=True, repeated_turn=False, repeated_session=False, duration=0.1),
    _tool("t1", "s1", sig="a", idempotent=True, repeated_turn=True, repeated_session=True, duration=0.3),
    _tool("t1", "s1", sig="b", idempotent=False, repeated_turn=False, repeated_session=False, duration=0.2, blocked=True),
    {"kind": "turn", "turn_id": "t1", "session_id": "s1", "outcome": "success", "wall_s": 12.0},
    _model("t2", "s2", inp=10, read=90, write=0, out=5, cost=0.001, latency=2.0, role="subagent"),
    {"kind": "turn", "turn_id": "t2", "session_id": "s2", "outcome": "failed", "wall_s": 2.0},
]


def test_aggregate_relates_to_records():
    agg = build_report(RECORDS)["aggregate"]
    model = [r for r in RECORDS if r["kind"] == "model"]
    tools = [r for r in RECORDS if r["kind"] == "tool"]
    turns = {r["turn_id"] for r in RECORDS}

    assert agg["model_calls_per_turn"] == len(model) / len(turns)
    assert agg["tool_calls_per_turn"] == len(tools) / len(turns)
    assert agg["repeated_signature_rate_turn"] == sum(t["repeated_in_turn"] for t in tools) / len(tools)
    assert agg["idempotent_share"] == sum(t["effects"]["idempotent"] for t in tools) / len(tools)
    prompt = sum(m["input_tokens"] + m["cache_read_tokens"] + m["cache_write_tokens"] for m in model)
    assert agg["cached_input_share"] == sum(m["cache_read_tokens"] for m in model) / prompt
    cost = sum(m["cost_usd"] for m in model if m["cost_usd"] is not None)
    assert abs(agg["cost_per_turn_usd"] - cost / len(turns)) < 1e-12
    # cost per successful turn only counts calls made in turns that succeeded
    assert abs(agg["cost_per_successful_turn_usd"] - sum(m["cost_usd"] or 0 for m in model if m["turn_id"] == "t1")) < 1e-12
    assert agg["success_rate"] == 0.5
    # blocked calls never ran: they are not deterministic ops and have no latency
    executed = [t for t in tools if not t["blocked"]]
    assert agg["deterministic_ops_per_model_call"] == len(executed) / len(model)
    assert agg["tool_latency_p95_s"] == max(t["duration_s"] for t in executed)
    latencies = sorted(m["latency_s"] for m in model)
    assert latencies[0] <= agg["model_latency_p50_s"] <= agg["model_latency_p95_s"] <= latencies[-1]


def test_sessions_partition_the_aggregate():
    report = build_report(RECORDS)
    for key in ("model_calls", "tool_calls", "turns", "prompt_tokens"):
        assert sum(s[key] for s in report["sessions"].values()) == report["aggregate"][key]


def test_loader_skips_malformed_lines_and_filters_by_session(tmp_path):
    lines = [json.dumps(r) for r in RECORDS] + ["{not json", json.dumps({"no": "kind"})]
    (tmp_path / "calls-2026-09-17-1.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(load_records([tmp_path])) == len(RECORDS)
    assert {r["session_id"] for r in load_records([tmp_path], session_id="s2")} == {"s2"}
