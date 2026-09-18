"""The compiled-skill representation: contract, generated implementation and SKILL.md.

A compiled skill is an ordinary skill directory (same storage, discovery, curator lifecycle and
usage tracking as any other) with two extra files:

- ``references/compiled_contract.json`` — the contract: input/output JSON Schemas, the declared
  calls (the only tool calls the implementation may make, in order), their effects, the
  environment fingerprint, the preconditions, the implementation's sha256 and a small replay
  corpus taken from the source traces.
- ``scripts/compiled_skill.py`` — the implementation: ``run(params, call) -> {"steps": [...]}``.
  ``call`` is the only capability it is handed; the runtime refuses any call the contract does not
  declare. The file is generated, and its hash is pinned in the contract, so a replacement
  implementation must be re-validated against the same corpus before it runs.

``SKILL.md`` describes the same procedure in prose: it is what the agent follows whenever the
compiled path does not run (preconditions unmet, deactivated, or a failed result check).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.compiled_skill_mining import Candidate
from agent.compiled_skill_traces import result_ok

FORMAT = 1
CONTRACT_PATH = "references/compiled_contract.json"
IMPLEMENTATION_PATH = "scripts/compiled_skill.py"
_CLIP_STR, _CLIP_LIST, _CLIP_KEYS = 300, 10, 50  # stored corpus results keep shape, not bulk
MAX_CONTRACT_CHARS = 90_000  # skill_manage caps a supporting file at 100k chars
_JSON_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


class ContractError(Exception):
    """The skill's compiled files are missing, unreadable or inconsistent."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value[:_CLIP_STR]
    if isinstance(value, list):
        return [_clip(v) for v in value[:_CLIP_LIST]]
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in list(value.items())[:_CLIP_KEYS]}
    return value


def environment_fingerprint(tools: List[str]) -> Dict[str, Any]:
    """What the procedure depends on besides its parameters: host platform and the exact schema of
    every tool it calls (a changed schema means the recorded calls may no longer mean the same)."""
    from tools.registry import registry
    return {"platform": sys.platform,
            "tools": {t: sha256_text(canonical(registry.get_schema(t))) for t in sorted(set(tools))}}


def _input_schema(params: Dict[str, str]) -> Dict[str, Any]:
    return {"$schema": _JSON_SCHEMA, "type": "object", "additionalProperties": False,
            "required": list(params), "properties": {name: {"type": kind} for name, kind in params.items()}}


def _output_schema(output_steps) -> Dict[str, Any]:
    items = [{"type": "object", "required": s["required"],
              "properties": {k: {"type": t} for k, t in s["types"].items()}} for s in output_steps]
    return {"$schema": _JSON_SCHEMA, "type": "object", "required": ["steps"], "properties": {
        "steps": {"type": "array", "prefixItems": items, "minItems": len(items), "maxItems": len(items)}}}


def _implementation_source(name: str, steps, session_count: int) -> str:
    lines = [
        f'"""Compiled implementation of the `{name}` skill, generated from {session_count} session traces.',
        "",
        f"Contract: ../{CONTRACT_PATH}. Its implementation.sha256 pins this file: after any edit the",
        "compiled path stays off until the curator re-validates the file against the replay corpus.",
        '"""',
        "",
        "",
        "def run(params, call):",
        "    steps = []",
    ]
    for step in steps:
        pairs = [f"{k!r}: {v!r}" for k, v in step["const"].items()]
        pairs += [f"{k!r}: params[{p!r}]" for k, p in step["params"].items()]
        lines.append(f"    steps.append(call({step['tool']!r}, {{{', '.join(pairs)}}}))")
    lines += ['    return {"steps": steps}', ""]
    return "\n".join(lines)


def _description(tools: List[str]) -> str:
    unique = list(dict.fromkeys(tools))
    text = f"Compiled read-only procedure: {', '.join(unique)}."
    return text if len(text) <= 60 else f"Compiled {len(tools)}-step read-only procedure."


def build(candidate: Candidate, *, now: str) -> Tuple[Dict[str, Any], str, str]:
    """``(contract, implementation_source, skill_md)`` for a mined candidate."""
    tools = [s["tool"] for s in candidate.steps]
    implementation = _implementation_source(candidate.name, candidate.steps, candidate.session_count)
    contract: Dict[str, Any] = {
        "format": FORMAT,
        "name": candidate.name,
        "shape_key": candidate.shape_key,
        "implementation": {"path": IMPLEMENTATION_PATH, "sha256": sha256_text(implementation)},
        "input_schema": _input_schema(candidate.params),
        "output_schema": _output_schema(candidate.output_steps),
        "calls": [dict(s) for s in candidate.steps],
        # Every source call resolved to this (the mining gate); every live call is re-resolved.
        "effects": [{"tool": t, "idempotent": True, "destructive": False} for t in tools],
        "environment": {"cwd": candidate.cwd, **environment_fingerprint(tools)},
        "preconditions": [
            "the invocation's parameters validate against input_schema",
            f"the session's working directory is {candidate.cwd}",
            "every tool in calls is enabled for the session and its schema hash matches environment.tools",
            "every rendered call resolves (registry.resolve_effects) to idempotent and not destructive",
            f"{IMPLEMENTATION_PATH} matches implementation.sha256",
        ],
        "result_check": "output validates against output_schema and every step result is error-free "
                        "(no error value, success is not false, exit_code 0 when present)",
        "source": {"sessions": candidate.session_count, "mined_at": now},
        "corpus": [],
    }
    for entry in candidate.corpus:
        contract["corpus"].append({
            "session_id": entry.session_id, "params": entry.params,
            "calls": [{"tool": t, "args": a} for t, a in entry.calls],
            "results": [_clip(r) for r in entry.results]})
    while len(canonical(contract)) > MAX_CONTRACT_CHARS and len(contract["corpus"]) > 2:
        contract["corpus"].pop()
    return contract, implementation, _skill_md(contract, tools)


def _skill_md(contract: Dict[str, Any], tools: List[str]) -> str:
    import yaml
    frontmatter = {
        "name": contract["name"], "description": _description(tools), "version": "1.0.0",
        "author": "Hermes Agent (compiled from session traces)",
        "metadata": {"hermes": {"tags": ["compiled"], "compiled": {"contract": CONTRACT_PATH}}},
    }
    params = contract["input_schema"]["properties"]
    invocation = " ".join(f"{p}=<{spec['type']}>" for p, spec in params.items())
    procedure = []
    for i, step in enumerate(contract["calls"], 1):
        args = [f"`{k}` = `{json.dumps(v, ensure_ascii=False)}`" for k, v in step["const"].items()]
        args += [f"`{k}` = the `{p}` parameter" for k, p in step["params"].items()]
        procedure.append(f"{i}. Call `{step['tool']}` with {', '.join(args) or 'no arguments'}.")
    body = [
        f"# {contract['name']} Skill", "",
        f"A read-only procedure that succeeded in {contract['source']['sessions']} sessions, compiled so "
        f"`/{contract['name']} {invocation}` runs it without a model call when its preconditions hold. "
        "Otherwise, or when the compiled result fails its check, follow the procedure below.", "",
        "## When to Use", "", "When the user asks for this procedure with new values for its parameters.", "",
        "## Parameters", "", *[f"- `{p}` ({spec['type']})" for p, spec in params.items()], "",
        "## Procedure", "", *procedure, "",
        "## Verification", "",
        f"Every step returns an error-free result carrying the fields recorded in `{CONTRACT_PATH}`.", "",
    ]
    return "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n" + "\n".join(body)


def load(skill_dir: Path) -> Tuple[Dict[str, Any], str]:
    """``(contract, implementation_source)``; ContractError when missing, malformed or when the
    implementation no longer matches the hash its contract pins."""
    try:
        contract = json.loads((skill_dir / CONTRACT_PATH).read_text(encoding="utf-8"))
        impl_rel = contract["implementation"]["path"]
        implementation = (skill_dir / impl_rel).read_text(encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ContractError(f"unreadable compiled skill files: {exc}") from exc
    if contract.get("format") != FORMAT or impl_rel != IMPLEMENTATION_PATH:
        raise ContractError("unsupported contract format")
    if sha256_text(implementation) != contract["implementation"].get("sha256"):
        raise ContractError("implementation does not match the hash pinned in the contract")
    return contract, implementation


def validate(schema: Dict[str, Any], value: Any) -> Optional[str]:
    """First schema violation, or None. Fails closed: no validator means no pass."""
    try:
        from jsonschema.validators import validator_for
    except ImportError:
        return "jsonschema is not installed"
    errors = sorted(validator_for(schema)(schema).iter_errors(value), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    path = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in errors[0].absolute_path)
    return f"${path}: {errors[0].message}"


def check_result(contract: Dict[str, Any], output: Any) -> Optional[str]:
    """The contract's result check: output schema, then every step result error-free."""
    if (error := validate(contract["output_schema"], output)) is not None:
        return f"output schema: {error}"
    for i, result in enumerate(output["steps"]):
        if not result_ok(result):
            return f"step {i + 1} ({contract['calls'][i]['tool']}) returned an error result"
    return None


def render_call(step: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """The exact arguments the contract permits for *step* under *params*."""
    return {**step["const"], **{arg: params[name] for arg, name in step["params"].items()}}
