"""The attempt loop behind ``semantic_call``: admission, verified cascades, failover, and the ledger
record the resolver learns from.

Each implementation in ``resolution.ranked`` is tried in order:

1. **Admission** (``agent/admission.py``): wait, bounded, for the implementation's resources (its
   ceilings and any backoff). A refusal is not an attempt: nothing was sent, so the loop moves on.
2. **Invoke.** A provider error moves on to the next ADMITTED implementation, as in stage 3. An
   exhaustion error (429, overloaded) also backs its resources off for everyone.
3. **Verify**, only mechanically: host-side ``output_schema`` validation and/or the caller's
   ``verify`` predicate. What the model says about its own confidence is never consulted. A
   verification failure escalates only when the resolution is a **cascade** (a verifier exists,
   cascades are on and the objective is cost); otherwise it raises ``SemanticOutputError``.
4. **Record** one ledger ``capability`` record per attempt (outcome, latency, cache-adjusted cost).

A cascade is bounded: at most ``cascade.max_attempts`` attempts, and an escalation is not started
when the spend so far plus the next implementation's expected cost would exceed the policy's
``max_cost_usd``, or the elapsed time plus its expected latency would pass ``max_latency_ms``.

**Restartability.** An attempt has no side effects outside the ledger (the model kind is a pure
request), and admission state lives only in this process, so re-running a killed call from the
start is always safe: no slot or backoff outlives the process. What a kill loses: the in-flight
attempt's spend, and ledger records still in the writer's queue (``atexit`` flushes them, SIGKILL
does not). A cancellation mid-attempt (``/stop``, a settled cell, KeyboardInterrupt) releases its
admission slot, records ``cancelled`` (never counted against the implementation) and does not
make the implementation the incumbent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.capability_resolver import Candidate, Resolution

logger = logging.getLogger(__name__)

# (text) -> (output, errors, failed_outcome): errors empty = verified.
Verifier = Callable[[str], Tuple[Any, List[str], str]]


@dataclass(frozen=True)
class Served:
    candidate: Candidate
    response: Any
    route_info: Dict[str, str]
    text: str
    output: Any
    explanation: str


def _budget_stop(resolution: Resolution, position: int, attempts: int, spent: float, elapsed_ms: float,
                 max_attempts: int) -> Optional[str]:
    """Why the next cascade rung may not start (projected, never just spent), or None."""
    if attempts >= max_attempts:
        return f"cascade.max_attempts {max_attempts} reached"
    nxt = resolution.scored[position] if position < len(resolution.scored) else None
    policy = resolution.policy
    if policy.max_cost_usd is not None:
        expected = nxt.cost_usd if nxt is not None and nxt.cost_usd is not None else 0.0
        if spent + expected > policy.max_cost_usd:
            return f"spent ${spent:.6f} + next ~${expected:.6f} > max_cost_usd {policy.max_cost_usd:g}"
    if policy.max_latency_ms is not None:
        expected_ms = nxt.latency_ms if nxt is not None and nxt.latency_ms is not None else 0.0
        if elapsed_ms + expected_ms > policy.max_latency_ms:
            return f"elapsed {elapsed_ms:.0f}ms + next ~{expected_ms:.0f}ms > max_latency_ms {policy.max_latency_ms}"
    return None


def _usage_fields(response: Any, candidate: Candidate, route_info: Dict[str, str]) -> Dict[str, Any]:
    from agent.capability_profile import attempt_cost
    from agent.usage_pricing import normalize_usage
    raw = getattr(response, "usage", None)
    if raw is None:
        return {"cost_usd": None}
    provider = route_info.get("provider") or candidate.target.get("provider")
    usage = normalize_usage(raw, provider=provider)
    model = str(getattr(response, "model", "") or route_info.get("model") or "")
    cost, saving = attempt_cost(usage, model=model, provider=provider, base_url=candidate.target.get("base_url"),
                                contract=candidate.contract)
    return {
        "input_tokens": usage.input_tokens, "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens, "output_tokens": usage.output_tokens,
        "cost_usd": None if cost is None else round(cost, 8),
        "cache_saving_usd": None if saving is None else round(saving, 8),
    }


class _Recorder:
    """Ledger ``capability`` records for one call; a no-op unless a ledger turn is bound."""

    def __init__(self, capability: str, resolution: Resolution, scope: str):
        from agent.call_ledger import active_turn_bound
        self.enabled = active_turn_bound()
        self.capability, self.resolution, self.scope = capability, resolution, scope
        self.step = 0

    def __call__(self, candidate: Candidate, outcome: str, started: float, *, response: Any = None,
                 route_info: Optional[Dict[str, str]] = None, error: Optional[BaseException] = None,
                 errors: Optional[List[str]] = None) -> Optional[float]:
        """Write the record; returns the attempt's cost when known."""
        self.step += 1
        if not self.enabled:
            return None
        from agent.call_ledger import record_capability_attempt
        route_info = route_info or {}
        usage = _usage_fields(response, candidate, route_info) if response is not None else {"cost_usd": None}
        record_capability_attempt({
            "capability": self.capability, "identity": candidate.identity, "label": candidate.label,
            "display": candidate.display, "scope": self.scope, "step": self.step,
            "cascade": self.resolution.cascade, "outcome": outcome,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "provider": route_info.get("provider"), "model": route_info.get("model"),
            "error_class": type(error).__name__ if error is not None else None,
            "verify_errors": (errors or [])[:3], **usage,
        })
        return usage.get("cost_usd")


def _served_elsewhere(candidate: Candidate, route_info: Dict[str, str]) -> bool:
    """The candidate's own route failed and auxiliary fallback answered (possible when unfenced)."""
    declared, served = candidate.target.get("model"), route_info.get("model")
    return bool(declared and served and served != declared)


def run(capability: str, resolution: Resolution, *, invoke: Callable[[Candidate], Tuple[Any, Dict[str, str]]],
        verify: Optional[Verifier], output_error: Callable[..., Exception], scope: str,
        max_attempts: int, admission_settings: Any) -> Served:
    """Try ``resolution.ranked`` in order; return the first verified answer (see the module docstring)."""
    from agent.admission import ADMISSION, AdmissionTimeout, exhaustion_signal
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    record = _Recorder(capability, resolution, scope)
    explanation, spent, started_all = resolution.explanation, 0.0, time.monotonic()
    last_error: Optional[BaseException] = None
    ranked = resolution.ranked
    for position, candidate in enumerate(ranked):
        if position and resolution.cascade:
            stop = _budget_stop(resolution, position, record.step, spent,
                                (time.monotonic() - started_all) * 1000, max_attempts)
            if stop:
                explanation += f"; cascade stopped before {candidate.label}: {stop}"
                break
        resources = candidate.contract.resources
        started = time.monotonic()
        try:
            with ADMISSION.hold(resources, settings=admission_settings,
                                deadline=time.monotonic() + admission_settings.max_wait_s):
                started = time.monotonic()
                response, route_info = invoke(candidate)
        except AdmissionTimeout as exc:
            last_error = exc
            explanation += f"; {candidate.label} not admitted ({exc.resource} {exc.reason})"
            continue
        except AuxiliaryExplicitCancellation as exc:
            record(candidate, "cancelled", started, error=exc)
            raise
        except Exception as exc:
            record(candidate, "error", started, error=exc)
            signal = exhaustion_signal(exc, provider=candidate.target.get("provider", ""),
                                       model=candidate.target.get("model", ""))
            if signal is not None:
                ADMISSION.signal_exhausted(resources, retry_after_s=signal[1], settings=admission_settings)
            last_error = exc
            nxt = ranked[position + 1].label if position + 1 < len(ranked) else None
            logger.info("semantic_call %s: %s failed (%s)%s", capability, candidate.label, type(exc).__name__,
                        f"; trying {nxt}" if nxt else "")
            explanation += f"; {candidate.label} failed ({type(exc).__name__})" + (f", moved to {nxt}" if nxt else "")
            continue
        except BaseException as exc:  # KeyboardInterrupt / SystemExit mid-attempt: not the implementation's fault
            record(candidate, "cancelled", started, error=exc)
            raise
        ADMISSION.signal_ok(resources)
        from agent.semantic_call_io import response_text
        text = response_text(response)
        output, errors, failed = verify(text) if verify is not None else (None, [], "")
        outcome = failed if errors else ("fallback" if _served_elsewhere(candidate, route_info) else "ok")
        cost = record(candidate, outcome, started, response=response, route_info=route_info, errors=errors)
        spent += cost if cost is not None else (resolution.scored[position].cost_usd or 0.0
                                                if position < len(resolution.scored) else 0.0)
        if not errors:
            return Served(candidate, response, route_info, text, output, explanation)
        last_error = output_error(text, errors, explanation)
        if not resolution.cascade:
            raise last_error
        explanation += f"; {candidate.label} failed verification ({failed}: {errors[0]})"
        if position + 1 < len(ranked):
            explanation += f", escalating to {ranked[position + 1].label}"
    if isinstance(last_error, Exception) and hasattr(last_error, "explanation"):
        last_error.explanation = explanation
    raise last_error if last_error is not None else RuntimeError(f"semantic_call '{capability}': nothing ran")
