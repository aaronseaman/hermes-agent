"""Capability resolver scoring: hard constraints, objective ranking, stickiness, explanation.

Pure functions over :class:`agent.capability_resolver.Scored` — no config, network or clock — so
a stage-4 profile source changes the numbers, never the rules.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

from agent.capability_resolver import QUALITY_TIERS, Policy, Request, Scored

_Refusal = Callable[[Scored, Policy, Request], Optional[str]]
_INF = math.inf


def _local_only(s: Scored, policy: Policy, _req: Request) -> Optional[str]:
    local = s.candidate.contract.local
    if not policy.local_only or local is True:
        return None
    return "remote (local_only)" if local is False else "locality unknown (local_only; declare local: true)"


def _min_quality(s: Scored, policy: Policy, _req: Request) -> Optional[str]:
    quality = s.candidate.contract.quality
    if policy.min_quality is None:
        return None
    if quality is None:
        return f"quality unknown (min_quality {policy.min_quality}; declare quality:)"
    if QUALITY_TIERS.index(quality) < QUALITY_TIERS.index(policy.min_quality):
        return f"quality {quality} < min_quality {policy.min_quality}"
    return None


def _max_cost(s: Scored, policy: Policy, _req: Request) -> Optional[str]:
    if policy.max_cost_usd is None:
        return None
    if s.cost_usd is None:
        return f"cost unknown (max_cost_usd {policy.max_cost_usd:g}; declare cost: {{input, output}})"
    if s.cost_usd > policy.max_cost_usd:
        return f"est. ${s.cost_usd:.6f} > max_cost_usd {policy.max_cost_usd:g}"
    return None


def _max_latency(s: Scored, policy: Policy, _req: Request) -> Optional[str]:
    if policy.max_latency_ms is None or s.latency_ms is None or s.latency_ms <= policy.max_latency_ms:
        return None
    return f"latency {s.latency_ms:.0f}ms > max_latency_ms {policy.max_latency_ms}"


def _context_fit(s: Scored, _policy: Policy, req: Request) -> Optional[str]:
    window = s.candidate.contract.context_window
    needed = req.input_tokens + (req.max_output_tokens or 0)
    if window is None or needed <= window:
        return None
    return f"context window {window} < ~{needed} tokens needed"


def _image_input(s: Scored, _policy: Policy, req: Request) -> Optional[str]:
    return "no image input" if req.needs_image and s.candidate.contract.image_input is False else None


# The order reasons are reported in; the first refusal wins.
CONSTRAINTS: Tuple[_Refusal, ...] = (_local_only, _min_quality, _max_cost, _max_latency, _context_fit, _image_input)


def refusal(s: Scored, policy: Policy, request: Request) -> Optional[str]:
    return next((why for why in (check(s, policy, request) for check in CONSTRAINTS) if why), None)


def _or_inf(value: Optional[float]) -> float:
    return _INF if value is None else float(value)


def _quality_badness(s: Scored) -> float:
    quality = s.candidate.contract.quality
    return float(len(QUALITY_TIERS) - QUALITY_TIERS.index(quality)) if quality in QUALITY_TIERS else _INF


# objective -> (primary score, secondary score); lower is better, unknown is +inf.
_OBJECTIVES = {
    "order": lambda s: (float(s.candidate.order), 0.0),
    "cost": lambda s: (_or_inf(s.cost_usd), _or_inf(s.latency_ms)),
    "latency": lambda s: (_or_inf(s.latency_ms), _or_inf(s.cost_usd)),
    "quality": lambda s: (_quality_badness(s), _or_inf(s.cost_usd)),
}


def _soft_penalty(s: Scored, request: Request) -> Tuple[bool, bool]:
    """Preferences that outrank the objective without excluding: a route auxiliary routing marked
    unhealthy, and a model known to lack structured output when a schema is requested."""
    return (not s.candidate.healthy, request.wants_schema and s.candidate.contract.structured_output is False)


def rank(scored: Sequence[Scored], policy: Policy,
         request: Request) -> Tuple[List[Scored], List[Tuple[Scored, str]]]:
    admitted, rejected = [], []
    for s in scored:
        why = refusal(s, policy, request)
        if why:
            rejected.append((s, why))
        else:
            admitted.append(s)
    objective = _OBJECTIVES[policy.optimize]
    # Deterministic: config order, then label, break every remaining tie.
    admitted.sort(key=lambda s: (*_soft_penalty(s, request), *objective(s), s.candidate.order, s.candidate.label))
    return admitted, rejected


def sticky_choice(ranked: Sequence[Scored], incumbent: Optional[str], policy: Policy,
                  request: Request) -> Tuple[Scored, Optional[str]]:
    """``(chosen, note)``: the incumbent keeps the job unless the best challenger clears the margin.

    The margin is the switching penalty: ``policy.stickiness`` as a fraction of the incumbent's
    objective score. The ``order`` objective is the user's explicit ranking and is not damped.
    """
    best = ranked[0]
    if incumbent is None or best.candidate.identity == incumbent:
        return best, None
    current = next((s for s in ranked if s.candidate.identity == incumbent), None)
    if current is None:
        return best, "previous implementation no longer admitted"
    if _soft_penalty(current, request) > _soft_penalty(best, request):
        return best, f"switched from {current.label}: marked unhealthy or lacks structured output"
    if policy.optimize == "order":
        return best, f"switched from {current.label}: config order prefers {best.label}"
    objective = _OBJECTIVES[policy.optimize]
    challenger, held = objective(best)[0], objective(current)[0]
    if math.isinf(held) and not math.isinf(challenger):
        return best, f"switched from {current.label}: its {policy.optimize} is unknown"
    gain = (held - challenger) / held if held > 0 and not math.isinf(held) else 0.0
    if challenger < held * (1 - policy.stickiness):
        return best, (f"switched from {current.label}: {best.label} better on {policy.optimize} by "
                      f"{gain:.0%} > stickiness {policy.stickiness:.0%}")
    return current, (f"kept incumbent {current.label}: {best.label} better on {policy.optimize} by only "
                     f"{gain:.0%} <= stickiness {policy.stickiness:.0%}")


def _facts(s: Scored) -> str:
    contract = s.candidate.contract
    facts = [f"est. ${s.cost_usd:.6f}" if s.cost_usd is not None else "cost unknown",
             f"quality {contract.quality}" if contract.quality else "quality unknown"]
    if s.latency_ms is not None:
        facts.append(f"~{s.latency_ms:.0f}ms")
    if contract.local:
        facts.append("local")
    if s.observed:
        facts.append("observed")
    if not s.candidate.healthy:
        facts.append("marked unhealthy")
    return ", ".join(facts)


def explain(capability: str, policy: Policy, ordered: Sequence[Scored],
            rejected: Sequence[Tuple[Scored, str]], sticky_note: Optional[str]) -> str:
    chosen = ordered[0]
    text = (f"{capability}: chose {chosen.label} ({chosen.display}) [{_facts(chosen)}] by {policy.describe()}, "
            f"ties by config order")
    if sticky_note:
        text += f"; {sticky_note}"
    if len(ordered) > 1:
        text += "; also admitted: " + "; ".join(f"{s.label} ({s.display}) [{_facts(s)}]" for s in ordered[1:])
    if rejected:
        text += "; rejected: " + "; ".join(f"{s.label} ({s.display}): {why}" for s, why in rejected)
    return text
