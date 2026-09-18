"""semantic_call — a transformation addressed by capability, never by implementation.

    result = semantic_call(
        "summarize_diff",                         # an auxiliary.<name> block in config.yaml
        inputs=[diff_text, {"path": path}],       # explicit projection: no parent transcript
        output_schema={"type": "object", ...},    # validated host-side
        policy={"optimize": "cost", "max_latency_ms": 8000, "min_quality": "medium"},
        instructions="Summarize the change. Reply with JSON ...",
    )
    result.output, result.explanation

Each call is its own request: nothing from any conversation is added, the main conversation's
messages and prompt cache are untouched, and ``instructions`` are sent verbatim so a
capability's prompt stays cache-stable. :class:`agent.capability_resolver.Resolver` chooses
among the capability's implementations (``_KINDS``: models today); the chosen one runs, and if
it fails the next ADMITTED candidate runs — never one the policy rejected.

Raises, before anything is sent: ``UnsatisfiablePolicyError`` when no implementation satisfies
the policy, :class:`ResolverConfigError` for malformed config/policy, ``ValueError``
for malformed inputs/schema. After sending: :class:`SemanticOutputError` when the reply does
not validate against ``output_schema``, or the last implementation's own error when every
admitted implementation failed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence

from agent import semantic_call_io as io
from agent import semantic_call_models
from agent.capability_resolver import (
    Candidate, Policy, Request, Resolution, Resolver, ResolverConfigError, merge_policy,
)

logger = logging.getLogger(__name__)


class _Kind(NamedTuple):
    """How one kind of implementation becomes a candidate and how it runs."""

    build: Callable[..., Candidate]    # (entry, *, label, order, policy, own_route=False) -> Candidate
    invoke: Callable[..., Any]         # (candidate, capability, messages, **request) -> (response, route_info)


# A new kind of implementation (deterministic code, replay, API, CLI, composite) is a row here.
_KINDS: Dict[str, _Kind] = {
    semantic_call_models.KIND: _Kind(semantic_call_models.build_candidate, semantic_call_models.invoke),
}


class SemanticOutputError(RuntimeError):
    """The reply is not JSON valid under the requested schema. ``text`` keeps the raw reply."""

    def __init__(self, capability: str, text: str, errors: List[str], explanation: str):
        self.text, self.errors, self.explanation = text, list(errors), explanation
        super().__init__(f"semantic_call '{capability}': output failed schema validation: " + "; ".join(errors))


@dataclass(frozen=True)
class OutputSpec:
    """A JSON Schema plus its ``response_format`` name/strictness (a bare dict = defaults).

    ``enforce=False`` still asks the provider for the schema but skips host validation and leaves
    ``output`` None — for a consumer that parses the reply itself and must keep working when a
    provider ignores ``response_format`` (title generation's prose fallback).
    """

    schema: Mapping[str, Any]
    name: str = "semantic_output"
    strict: bool = False
    enforce: bool = True


@dataclass(frozen=True)
class SemanticResult:
    text: str
    output: Any                  # validated JSON when a schema is enforced, else None
    served_by: Candidate         # the implementation that answered
    provider: str                # the route that actually answered (after allowed fallback)
    model: str
    explanation: str
    resolution: Resolution
    latency_ms: int


def candidates_for(capability: str, task_config: Mapping[str, Any], policy: Policy) -> List[Candidate]:
    """Implementations declared for ``auxiliary.<capability>``: its ``candidates`` list, or the
    block's own route when it declares none."""
    raw = task_config.get("candidates")
    if raw is None:
        return [_KINDS[semantic_call_models.KIND].build(
            task_config, label=f"auxiliary.{capability}", order=0, policy=policy, own_route=True)]
    if not isinstance(raw, list) or not raw:
        raise ResolverConfigError(f"auxiliary.{capability}.candidates must be a non-empty list")
    out = []
    for order, entry in enumerate(raw):
        label = f"auxiliary.{capability}.candidates[{order}]"
        if not isinstance(entry, Mapping):
            raise ResolverConfigError(f"{label} must be a mapping")
        kind = _KINDS.get(str(entry.get("kind") or semantic_call_models.KIND))
        if kind is None:
            raise ResolverConfigError(f"{label}.kind must be one of {sorted(_KINDS)}, got {entry.get('kind')!r}")
        out.append(kind.build(entry, label=label, order=order, policy=policy))
    return out


def _effective_timeout(capability: str, timeout: Optional[float], max_latency_ms: Optional[int]) -> Optional[float]:
    """Caller timeout bounded by the latency budget; ``None`` keeps ``auxiliary.<cap>.timeout``."""
    if max_latency_ms is None:
        return timeout
    from agent.auxiliary_client import _get_task_timeout
    return min(max_latency_ms / 1000.0, timeout if timeout is not None else _get_task_timeout(capability))


def semantic_call(
    capability: str, inputs: Sequence[Any], output_schema: Any = None, policy: Any = None, *,
    instructions: str = "", max_tokens: Optional[int] = None, temperature: Optional[float] = None,
    reasoning: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None,
    main_runtime: Optional[Dict[str, Any]] = None, cancel_check: Optional[Callable[[], bool]] = None,
    scope: str = "", resolver: Optional[Resolver] = None,
) -> SemanticResult:
    """Run one capability call; see the module docstring.

    ``reasoning`` is the auxiliary ``reasoning_config``; ``timeout`` defaults to
    ``auxiliary.<capability>.timeout`` (bounded by ``policy.max_latency_ms``); ``cancel_check``
    hard-cancels the in-flight attempt when it returns True (raises
    ``AuxiliaryExplicitCancellation``); ``main_runtime`` is the session runtime snapshot for
    capabilities that inherit the main model; ``scope`` keys incumbent stickiness (a session or
    sandbox id; ``""`` = per profile).
    """
    if not isinstance(capability, str) or not capability.strip():
        raise ValueError("capability must be a non-empty string")
    from agent.auxiliary_client import AuxiliaryExplicitCancellation, _get_auxiliary_task_config
    from agent.model_metadata import estimate_messages_tokens_rough
    started = time.monotonic()
    spec = output_schema if isinstance(output_schema, OutputSpec) or output_schema is None \
        else OutputSpec(output_schema)
    schema = io.check_schema(spec.schema, require_validator=spec.enforce) if spec is not None else None
    messages, has_image = io.build_messages(instructions, inputs)
    task_config = _get_auxiliary_task_config(capability)
    merged = merge_policy(task_config.get("policy"), policy, capability=capability)
    request = Request(input_tokens=estimate_messages_tokens_rough(messages), max_output_tokens=max_tokens,
                      needs_image=has_image, wants_schema=schema is not None)
    resolver = resolver or Resolver()
    resolution = resolver.resolve(capability, candidates_for(capability, task_config, merged), merged, request,
                                  scope=scope)
    extra_body = {"response_format": io.response_format(schema, spec.name, spec.strict)} if schema is not None else None
    # Only a policy with constraints a cross-provider fallback could break fences the call; any
    # other policy leaves auxiliary fallback exactly as it is for every other task.
    fence = merged.describe() if merged.fences_fallback else None
    explanation = resolution.explanation
    for position, candidate in enumerate(resolution.ranked):
        try:
            response, route_info = _KINDS[candidate.kind].invoke(
                candidate, capability, messages, fence=fence, extra_body=extra_body, max_tokens=max_tokens,
                temperature=temperature, reasoning=reasoning,
                timeout=_effective_timeout(capability, timeout, merged.max_latency_ms),
                main_runtime=main_runtime, cancel_check=cancel_check)
            break
        except AuxiliaryExplicitCancellation:
            raise
        except Exception as exc:
            if position + 1 >= len(resolution.ranked):
                raise
            nxt = resolution.ranked[position + 1]
            logger.info("semantic_call %s: %s failed (%s); trying %s", capability, candidate.label,
                        type(exc).__name__, nxt.label)
            explanation += f"; {candidate.label} failed ({type(exc).__name__}), moved to {nxt.label}"
    resolver.served(capability, candidate, scope=scope)
    text = io.response_text(response)
    output = None
    if schema is not None and spec.enforce:
        output, errors = io.validate_output(text, schema)
        if errors:
            raise SemanticOutputError(capability, text, errors, explanation)
    return SemanticResult(
        text=text, output=output, served_by=candidate, provider=route_info.get("provider", ""),
        model=route_info.get("model", ""), explanation=explanation, resolution=resolution,
        latency_ms=max(0, int((time.monotonic() - started) * 1000)),
    )
