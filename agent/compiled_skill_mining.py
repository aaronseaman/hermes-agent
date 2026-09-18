"""Mine compilable procedures from :class:`ProcedureTrace` records.

A candidate is one exact tool-call sequence (tool + argument keys, in order) that succeeded in at
least ``min_sessions`` distinct sessions with varying argument values. Scope is deliberately
narrow: whole-turn sequences only (no sub-sequence mining), and **every** call of **every** source
trace must resolve (``registry.resolve_effects``) to idempotent and not destructive. Anything that
writes, is irreversible, or whose effects are unknown (MCP tools without ``readOnlyHint``,
``execute_code``, a non-read-only ``terminal`` command, agent-loop tools) is out of scope: such a
procedure is never compiled, however often it recurs.

Generalisation is mechanical: an argument whose value is identical in every trace is a constant;
one that varies is a parameter, typed by its JSON scalar type. Varying slots with identical value
vectors are one parameter. A candidate is refused when a parameter's value first appears inside an
earlier step's result (the argument was derived from output, not supplied as input), when a
varying value is not a consistently-typed scalar, when it belongs to a tool whose effects are
resolved per call, or when the traces ran in different or unknown working directories.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.compiled_skill_traces import ProcedureTrace

MIN_STEPS, MAX_STEPS = 2, 8  # one call is just a tool call; long chains are not "a procedure"
MAX_CORPUS = 5
_NAME_MAX = 64


@dataclass(frozen=True)
class CorpusEntry:
    """One source trace, as replay input: the parameter values and the recorded calls/results."""
    session_id: str
    params: Dict[str, Any]
    calls: Tuple[Tuple[str, Dict[str, Any]], ...]
    results: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    shape_key: str
    steps: Tuple[Dict[str, Any], ...]      # {"tool", "const": {arg: value}, "params": {arg: param}}
    params: Dict[str, str]                 # param name -> JSON Schema type, in first-use order
    output_steps: Tuple[Dict[str, Any], ...]  # per step: {"required": [...], "types": {key: type}}
    corpus: Tuple[CorpusEntry, ...]
    cwd: str
    session_count: int


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _json_type(value: Any) -> Optional[str]:
    # bool before int: bool is an int subclass.
    for kind, label in ((bool, "boolean"), (int, "integer"), (float, "number"), (str, "string"),
                        (dict, "object"), (list, "array")):
        if isinstance(value, kind):
            return label
    return None


def _shape(trace: ProcedureTrace) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    return tuple((s.tool, tuple(sorted(s.args))) for s in trace.steps)


def _read_only(trace: ProcedureTrace) -> bool:
    from tools.registry import registry
    for step in trace.steps:
        effects = registry.resolve_effects(step.tool, step.args)
        if not effects.idempotent or effects.destructive:
            return False
    return True


def _effects_depend_on_args(tool: str) -> bool:
    """True for a tool whose effects are resolved per call (``terminal``). Its arguments may only be
    constants: a parameter would widen the permission from the replayed commands to any command
    the classifier happens to call read-only."""
    from tools.registry import registry
    return getattr(registry.get_entry(tool), "effects_fn", None) is not None


def _slug(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9-]", "-", text.lower())).strip("-")


def _skill_name(shape_key: str, tools: List[str]) -> str:
    digest = hashlib.sha256(shape_key.encode("utf-8")).hexdigest()[:8]
    body = _slug("-".join(tools))[: _NAME_MAX - len("compiled--") - len(digest)].strip("-")
    return f"compiled-{body}-{digest}"


def _param_name(arg: str, taken: Dict[str, str], vector_key: str, step: int) -> str:
    base = re.sub(r"\W", "_", arg).strip("_").lower() or "value"
    for candidate in (base, f"{base}_{step + 1}"):
        if taken.get(candidate, vector_key) == vector_key:
            return candidate
    n = 2
    while f"{base}_{step + 1}_{n}" in taken:
        n += 1
    return f"{base}_{step + 1}_{n}"


def _output_steps(sample: List[ProcedureTrace]) -> Tuple[Dict[str, Any], ...]:
    """Per step: keys present in every recorded result, and each key's JSON type when it is the
    same non-null type in every result. This is the output schema the replay and live runs check."""
    out = []
    for i in range(len(sample[0].steps)):
        results = [t.steps[i].result or {} for t in sample]
        required = sorted(set.intersection(*(set(r) for r in results)))
        types = {}
        for key in required:
            kinds = {_json_type(r[key]) if r[key] is not None else "null" for r in results}
            if len(kinds) == 1:
                types[key] = kinds.pop()
        out.append({"required": required, "types": types})
    return tuple(out)


def _generalise(shape, sample: List[ProcedureTrace]) -> Tuple[Optional[Candidate], str]:
    cwds = {t.cwd for t in sample}
    if len(cwds) != 1 or not next(iter(cwds)):
        return None, "cwd"
    steps: List[Dict[str, Any]] = []
    params: Dict[str, str] = {}
    by_vector: Dict[str, str] = {}   # value-vector key -> param name
    taken: Dict[str, str] = {}       # param name -> value-vector key
    first_use: Dict[str, int] = {}   # param name -> step index of first use
    for i, (tool, keys) in enumerate(shape):
        const, slots = {}, {}
        for key in keys:
            values = [t.steps[i].args[key] for t in sample]
            if len({_canonical(v) for v in values}) == 1:
                const[key] = values[0]
                continue
            if _effects_depend_on_args(tool):
                return None, "effects_param"
            kinds = {_json_type(v) for v in values}
            if len(kinds) != 1 or not kinds <= {"boolean", "integer", "number", "string"}:
                return None, "type"
            vector_key = _canonical(values)
            name = by_vector.get(vector_key) or _param_name(key, taken, vector_key, i)
            by_vector[vector_key], taken[name] = name, vector_key
            params.setdefault(name, kinds.pop())
            first_use.setdefault(name, i)
            slots[key] = name
        steps.append({"tool": tool, "const": const, "params": slots})
    if not params:
        return None, "constant"
    # A parameter first used at step i whose value already appeared in an earlier result was
    # derived from that output: compiling it as an input would change what the procedure means.
    for name, i in first_use.items():
        for trace in sample:
            value = next(trace.steps[i].args[k] for k, p in steps[i]["params"].items() if p == name)
            needle = _canonical(value)[1:-1] if isinstance(value, str) else _canonical(value)
            if any(needle in _canonical(trace.steps[j].result) for j in range(i)):
                return None, "dependency"
    corpus = []
    for trace in sample[:MAX_CORPUS]:
        values = {}
        for i, step in enumerate(steps):
            for key, name in step["params"].items():
                values.setdefault(name, trace.steps[i].args[key])
        corpus.append(CorpusEntry(
            session_id=trace.session_id, params=values,
            calls=tuple((s.tool, dict(s.args)) for s in trace.steps),
            results=tuple(dict(s.result or {}) for s in trace.steps)))
    shape_key = _canonical([[tool, list(keys)] for tool, keys in shape])
    return Candidate(
        name=_skill_name(shape_key, [tool for tool, _ in shape]), shape_key=shape_key, steps=tuple(steps),
        params=params, output_steps=_output_steps(sample), corpus=tuple(corpus), cwd=next(iter(cwds)),
        session_count=len(sample)), ""


def mine_candidates(traces: List[ProcedureTrace], *, min_sessions: int) -> Tuple[List[Candidate], Counter]:
    """Compilable candidates (newest trace per session wins) and a Counter of rejection reasons."""
    groups: Dict[Any, Dict[str, ProcedureTrace]] = {}
    for trace in traces:
        if trace.succeeded and MIN_STEPS <= len(trace.steps) <= MAX_STEPS:
            groups.setdefault(_shape(trace), {}).setdefault(trace.session_id, trace)
    candidates, rejected = [], Counter()
    for shape, by_session in groups.items():
        if len(by_session) < max(2, int(min_sessions)):
            continue
        sample = list(by_session.values())
        if not all(_read_only(t) for t in sample):
            rejected["effects"] += 1
            continue
        candidate, reason = _generalise(shape, sample)
        if candidate is None:
            rejected[reason] += 1
        else:
            candidates.append(candidate)
    return candidates, rejected
