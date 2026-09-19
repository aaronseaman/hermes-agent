"""Capability resolver scoring: hard constraints, the ``order`` objective, explanation.

Pure functions over :class:`agent.capability_resolver.Scored` (no config, network or clock). The
utility objectives (``cost``, ``latency``, ``quality``), the switching penalty and exploration live
in ``capability_resolver_utility.py``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from agent.capability_resolver import QUALITY_TIERS, Policy, Request, Scored

_Refusal = Callable[[Scored, Policy, Request], Optional[str]]


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
    # A measured implementation is held to its p95, not its median: the budget is a ceiling.
    latency = s.latency_p95_ms if s.latency_p95_ms is not None else s.latency_ms
    if policy.max_latency_ms is None or latency is None or latency <= policy.max_latency_ms:
        return None
    which = "p95 latency" if s.observed else "latency"
    return f"{which} {latency:.0f}ms > max_latency_ms {policy.max_latency_ms}"


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


def soft_penalty(s: Scored, request: Request) -> Tuple[bool, bool]:
    """Preferences that outrank the objective without excluding: a route auxiliary routing marked
    unhealthy, and a model known to lack structured output when a schema is requested."""
    return (not s.candidate.healthy, request.wants_schema and s.candidate.contract.structured_output is False)


def rank(scored: Sequence[Scored], policy: Policy,
         request: Request) -> Tuple[List[Scored], List[Tuple[Scored, str]]]:
    """Admit or reject every candidate, with a reason. Admitted rows come back in config order with
    soft penalties last; the objective orders them from there."""
    admitted, rejected = [], []
    for s in scored:
        why = refusal(s, policy, request)
        if why:
            rejected.append((s, why))
        else:
            admitted.append(s)
    admitted.sort(key=lambda s: (*soft_penalty(s, request), s.candidate.order, s.candidate.label))
    return admitted, rejected


def order_choice(ranked: Sequence[Scored], incumbent: Optional[str],
                 request: Request) -> Tuple[Scored, Optional[str]]:
    """The ``order`` objective: the first admitted candidate in config order. That is the user's
    explicit ranking, so it is never damped by stickiness and never learned over."""
    best = ranked[0]
    if incumbent is None or best.candidate.identity == incumbent:
        return best, None
    current = next((s for s in ranked if s.candidate.identity == incumbent), None)
    if current is None:
        return best, "previous implementation no longer admitted"
    if soft_penalty(current, request) > soft_penalty(best, request):
        return best, f"switched from {current.label}: marked unhealthy or lacks structured output"
    return best, f"switched from {current.label}: config order prefers {best.label}"


def _facts(s: Scored) -> str:
    contract, est = s.candidate.contract, s.estimate
    facts = [f"est. ${s.cost_usd:.6f}" if s.cost_usd is not None else "cost unknown",
             f"quality {contract.quality}" if contract.quality else "quality unknown"]
    if s.latency_ms is not None:
        facts.append(f"~{s.latency_ms:.0f}ms")
    if contract.local:
        facts.append("local")
    if est is not None:
        facts.append(f"P(ok) {est.p_success:.2f} n={est.samples}" + (" (prior)" if est.under_sampled else " observed"))
    if s.utility is not None:
        facts.append(f"U {s.utility:.3f} = " + " ".join(f"{name} {value:+.3f}" for name, value in s.terms))
    if not s.candidate.healthy:
        facts.append("marked unhealthy")
    return ", ".join(facts)


def explain(capability: str, policy: Policy, ordered: Sequence[Scored],
            rejected: Sequence[Tuple[Scored, str]], note: Optional[str]) -> str:
    chosen = ordered[0]
    how = "ties by config order" if policy.optimize == "order" else "by utility, ties by config order"
    text = f"{capability}: chose {chosen.label} ({chosen.display}) [{_facts(chosen)}] by {policy.describe()}, {how}"
    if note:
        text += f"; {note}"
    if len(ordered) > 1:
        text += "; also admitted: " + "; ".join(f"{s.label} ({s.display}) [{_facts(s)}]" for s in ordered[1:])
    if rejected:
        text += "; rejected: " + "; ".join(f"{s.label} ({s.display}): {why}" for s, why in rejected)
    return text
