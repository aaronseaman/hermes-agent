"""Baseline report over call-ledger records (``agent/call_ledger.py``): ``hermes insights --ledger``.

Every number is derived from the per-call events grouped by ``turn_id``; turn summary records
contribute only what events cannot (final outcome, wall time). The report never re-prices:
cost is what the recording seam priced through ``agent.usage_pricing``.

"Deterministic ops per model call" (tool calls that executed, over model calls) is easy to
game by splitting work into trivial steps, so it is only ever shown next to the turn success
rate and the cost per successful turn.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_records(directories: Iterable[Path], *, days: Optional[int] = None, session_id: Optional[str] = None) -> List[dict]:
    """Records from every ``*.jsonl`` under *directories*; malformed lines are skipped."""
    cutoff = time.time() - days * 86400 if days else None
    records: List[dict] = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.jsonl")):
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(record, dict) or "kind" not in record:
                        continue
                    if cutoff is not None and float(record.get("ts") or 0) < cutoff:
                        continue
                    if session_id and record.get("session_id") != session_id:
                        continue
                    records.append(record)
    return records


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def summarize(records: List[dict]) -> Dict[str, Any]:
    """Aggregate metrics for one slice of records (a session or everything)."""
    model = [r for r in records if r.get("kind") == "model"]
    tools = [r for r in records if r.get("kind") == "tool"]
    turn_rows = {r["turn_id"]: r for r in records if r.get("kind") == "turn" and r.get("turn_id")}
    turn_ids = {r.get("turn_id") for r in records if r.get("turn_id")}
    n_turns = len(turn_ids)

    executed = [t for t in tools if not t.get("blocked")]
    tokens = {k: sum(int(m.get(k) or 0) for m in model)
              for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens")}
    prompt = tokens["input_tokens"] + tokens["cache_read_tokens"] + tokens["cache_write_tokens"]
    cost = sum(float(m["cost_usd"]) for m in model if m.get("cost_usd") is not None)
    outcomes = Counter(r.get("outcome") or "unknown" for r in turn_rows.values())
    successes = outcomes.get("success", 0)
    success_cost = sum(
        float(m["cost_usd"]) for m in model
        if m.get("cost_usd") is not None and turn_rows.get(m.get("turn_id"), {}).get("outcome") == "success"
    )
    return {
        "turns": n_turns,
        "turns_with_summary": len(turn_rows),
        "turn_outcomes": dict(outcomes),
        "success_rate": _ratio(successes, len(turn_rows)),
        "model_calls": len(model),
        "model_calls_by_role": dict(Counter(m.get("call_role") or "unknown" for m in model)),
        "model_errors": dict(Counter(m.get("error_class") or "unknown" for m in model if m.get("outcome") != "ok")),
        "model_calls_per_turn": _ratio(len(model), n_turns),
        "tool_calls": len(tools),
        "tool_calls_per_turn": _ratio(len(tools), n_turns),
        "tool_calls_blocked": len(tools) - len(executed),
        "tool_calls_failed": sum(1 for t in tools if t.get("failed")),
        "tool_calls_spilled": sum(1 for t in tools if t.get("spilled")),
        "repeated_signature_rate_turn": _ratio(sum(1 for t in tools if t.get("repeated_in_turn")), len(tools)),
        "repeated_signature_rate_session": _ratio(sum(1 for t in tools if t.get("repeated_in_session")), len(tools)),
        "idempotent_share": _ratio(sum(1 for t in tools if (t.get("effects") or {}).get("idempotent")), len(tools)),
        "guardrail_decisions": dict(Counter(t["guardrail"] for t in tools if t.get("guardrail"))),
        "tokens": tokens,
        "prompt_tokens": prompt,
        "cached_input_share": _ratio(tokens["cache_read_tokens"], prompt),
        "usage_missing_calls": sum(1 for m in model if m.get("outcome") == "ok" and not m.get("usage_available")),
        "cost_usd": cost,
        "unpriced_calls": sum(1 for m in model if m.get("usage_available") and m.get("cost_usd") is None),
        "cost_per_turn_usd": _ratio(cost, n_turns),
        "cost_per_successful_turn_usd": _ratio(success_cost, successes),
        "model_latency_p50_s": _percentile([float(m["latency_s"]) for m in model if m.get("latency_s") is not None], 50),
        "model_latency_p95_s": _percentile([float(m["latency_s"]) for m in model if m.get("latency_s") is not None], 95),
        "tool_latency_p50_s": _percentile([float(t["duration_s"]) for t in executed if t.get("duration_s") is not None], 50),
        "tool_latency_p95_s": _percentile([float(t["duration_s"]) for t in executed if t.get("duration_s") is not None], 95),
        "deterministic_ops_per_model_call": _ratio(len(executed), len(model)),
        "turn_wall_p50_s": _percentile([float(r["wall_s"]) for r in turn_rows.values() if r.get("wall_s") is not None], 50),
    }


def build_report(records: List[dict]) -> Dict[str, Any]:
    by_session: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        by_session[record.get("session_id") or ""].append(record)
    return {
        "aggregate": summarize(records),
        "sessions": {sid: summarize(rows) for sid, rows in sorted(by_session.items())},
    }


def _fmt(value: Any, kind: str = "num") -> str:
    if value is None:
        return "n/a"
    formatters = {
        "pct": lambda v: f"{v * 100:.1f}%",
        "usd": lambda v: f"${v:.4f}",
        "s": lambda v: f"{v:.2f}s",
        "num": lambda v: f"{v:,.2f}" if isinstance(v, float) else f"{v:,}",
    }
    return formatters[kind](value)


def _format_summary(s: Dict[str, Any], indent: str = "  ") -> List[str]:
    t = s["tokens"]
    rows = [
        ("Turns", f"{s['turns']} ({s['turns_with_summary']} with summary; outcomes {s['turn_outcomes'] or '{}'}; "
                  f"wall p50 {_fmt(s['turn_wall_p50_s'], 's')})"),
        ("LLM calls / turn", f"{_fmt(s['model_calls_per_turn'])} ({s['model_calls']} calls: {s['model_calls_by_role']})"),
        ("Tool calls / turn", f"{_fmt(s['tool_calls_per_turn'])} ({s['tool_calls']} calls; {s['tool_calls_blocked']} blocked, "
                              f"{s['tool_calls_failed']} failed, {s['tool_calls_spilled']} spilled)"),
        ("Repeated signatures", f"{_fmt(s['repeated_signature_rate_turn'], 'pct')} within turn, "
                                f"{_fmt(s['repeated_signature_rate_session'], 'pct')} within session"),
        ("Declared idempotent", _fmt(s["idempotent_share"], "pct")),
        ("Input tokens", f"{t['cache_read_tokens']:,} cached / {t['input_tokens'] + t['cache_write_tokens']:,} uncached "
                         f"({t['cache_write_tokens']:,} cache writes); cached share {_fmt(s['cached_input_share'], 'pct')}"),
        ("Output tokens", f"{t['output_tokens']:,} ({t['reasoning_tokens']:,} reasoning)"),
        ("Cost", f"{_fmt(s['cost_usd'], 'usd')} total, {_fmt(s['cost_per_turn_usd'], 'usd')} / turn "
                 f"({s['unpriced_calls']} unpriced, {s['usage_missing_calls']} without usage)"),
        ("Model latency", f"p50 {_fmt(s['model_latency_p50_s'], 's')}, p95 {_fmt(s['model_latency_p95_s'], 's')}"),
        ("Tool latency", f"p50 {_fmt(s['tool_latency_p50_s'], 's')}, p95 {_fmt(s['tool_latency_p95_s'], 's')}"),
        ("Deterministic ops / LLM call", f"{_fmt(s['deterministic_ops_per_model_call'])} "
                                         f"(success rate {_fmt(s['success_rate'], 'pct')}, "
                                         f"cost / successful turn {_fmt(s['cost_per_successful_turn_usd'], 'usd')})"),
    ]
    if s["model_errors"]:
        rows.append(("Model errors", str(s["model_errors"])))
    if s["guardrail_decisions"]:
        rows.append(("Guardrail decisions", str(s["guardrail_decisions"])))
    width = max(len(label) for label, _ in rows)
    return [f"{indent}{label:<{width}}  {value}" for label, value in rows]


def format_report(report: Dict[str, Any], *, sources: Iterable[Path], per_session: bool = True) -> str:
    agg = report["aggregate"]
    lines = ["Call ledger baseline", f"  Sources: {', '.join(str(p) for p in sources)}", ""]
    if not agg["model_calls"] and not agg["tool_calls"] and not agg["turns"]:
        lines.append("  No ledger records. Enable agent.call_ledger.enabled in config.yaml and run some turns.")
        return "\n".join(lines)
    lines.append("Aggregate")
    lines.extend(_format_summary(agg))
    if per_session:
        for sid, summary in report["sessions"].items():
            lines.extend(["", f"Session {sid or '(none)'}"])
            lines.extend(_format_summary(summary))
    return "\n".join(lines)
