"""``semantic_call`` for execute_code scripts: a host-side RPC, not a model tool.

A script calls ``hermes_tools.semantic_call(...)``; the request rides the sandbox RPC through the
same pipeline as its tool calls (token check → allow-list → per-execution call budget →
dispatch) and runs HERE, in the host process, under the cell's context and profile scope.
Credentials, routes and the resolver stay host-side; the script gets back the validated output,
the answering provider/model and the resolver's explanation — never a key or an endpoint.

Exposure is config-driven and per capability: the stub exists only when some
``auxiliary.<capability>`` block sets ``sandbox: true``, and only those capabilities are callable.
It is never registered in the tool registry, so no model tool schema grows.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

RPC_NAME = "semantic_call"

# Upper bound on one sandbox semantic_call, inside the stub's 300s RPC read timeout.
_SANDBOX_CALL_CEILING_S = 240.0

# (signature, docstring, args_dict_expr) — the _TOOL_STUBS row for hermes_tools.py.
STUB = (
    "capability: str, inputs: list, output_schema: dict = None, policy: dict = None, "
    "instructions: str = \"\", max_tokens: int = None",
    '"""Model call by capability (the host picks the implementation). inputs: str / {"type": "image", "url": ...} '
    '/ any JSON value. output_schema: JSON Schema, validated host-side. policy: {optimize: cost|latency|quality|order, '
    'max_latency_ms, min_quality: low|medium|high, max_cost_usd, local_only} (can only tighten config). Returns dict '
    'with output (validated JSON, or None without a schema), text, provider, model, explanation — or error."""',
    '{"capability": capability, "inputs": inputs, "output_schema": output_schema, "policy": policy, '
    '"instructions": instructions, "max_tokens": max_tokens}',
)


def sandbox_capabilities() -> List[str]:
    """Capabilities whose ``auxiliary.<name>.sandbox`` is true for the active profile, sorted."""
    from hermes_cli.config import load_config_readonly
    aux = (load_config_readonly() or {}).get("auxiliary") or {}
    if not isinstance(aux, dict):
        return []
    return sorted(name for name, block in aux.items() if isinstance(block, dict) and block.get("sandbox") is True)


def host_rpc_tools() -> FrozenSet[str]:
    """Host-only RPC names execute_code exposes (empty unless a capability is configured)."""
    return frozenset({RPC_NAME}) if sandbox_capabilities() else frozenset()


def doc_line() -> str:
    names = ", ".join(sandbox_capabilities()) or "(none configured)"
    return ("  semantic_call(capability: str, inputs: list, output_schema: dict = None, policy: dict = None, "
            "instructions: str = \"\", max_tokens: int = None) -> dict\n"
            f"    Model call chosen by the host for a capability ({names}). Put everything the model needs in "
            "instructions + inputs. Returns {\"output\": <schema-valid JSON>, \"text\", \"provider\", \"model\", "
            "\"explanation\"} or {\"error\": ...}. Counts against the tool-call limit.")


def _bad_args(args: Dict[str, Any], allowed: List[str]) -> Optional[str]:
    capability = args.get("capability")
    if not isinstance(capability, str) or capability not in allowed:
        return (f"semantic_call capability {capability!r} is not available in execute_code. "
                f"Available: {', '.join(allowed) or 'none'} (auxiliary.<name>.sandbox: true).")
    if not isinstance(args.get("inputs"), (list, str)):
        return "semantic_call inputs must be a list of input blocks (or one string)."
    for key, kind in (("output_schema", dict), ("policy", dict), ("instructions", str), ("max_tokens", int)):
        value = args.get(key)
        if value is not None and (not isinstance(value, kind) or isinstance(value, bool)):
            return f"semantic_call {key} must be a {kind.__name__}."
    return None


def _error(message: str, **extra: Any) -> str:
    from agent.redact import redact_sensitive_text
    from tools.registry import tool_error
    return tool_error(redact_sensitive_text(message, force=True, redact_url_credentials=True), **extra)


def handle_rpc(args: Dict[str, Any], *, task_id: str, cancelled: Optional[Callable[[], bool]]) -> str:
    """Run one sandbox semantic_call; always returns a JSON string, never raises."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation, _get_task_timeout
    from agent.capability_resolver import ResolverConfigError, UnsatisfiablePolicyError
    from agent.semantic_call import SemanticOutputError, semantic_call
    args = args if isinstance(args, dict) else {}
    problem = _bad_args(args, sandbox_capabilities())
    if problem:
        return _error(problem)
    if cancelled is not None and cancelled():
        return _error("semantic_call cancelled: the execute_code call has ended.")
    capability, inputs = args["capability"], args["inputs"]
    try:
        result = semantic_call(
            capability, [inputs] if isinstance(inputs, str) else inputs, args.get("output_schema"), args.get("policy"),
            instructions=args.get("instructions") or "", max_tokens=args.get("max_tokens"),
            timeout=min(_get_task_timeout(capability), _SANDBOX_CALL_CEILING_S), cancel_check=cancelled,
            scope=f"execute_code:{task_id}",
        )
    except AuxiliaryExplicitCancellation:
        return _error("semantic_call cancelled: the execute_code call has ended.")
    except SemanticOutputError as exc:
        return _error(str(exc), schema_errors=exc.errors, text=exc.text)
    except (UnsatisfiablePolicyError, ResolverConfigError, ValueError) as exc:
        return _error(str(exc))
    except Exception as exc:
        logger.info("sandbox semantic_call %s failed (task %s): %s", capability, task_id, exc)
        return _error(f"semantic_call failed: {type(exc).__name__}: {exc}")
    return json.dumps({"output": result.output, "text": result.text, "provider": result.provider,
                       "model": result.model, "explanation": result.explanation}, ensure_ascii=False)
