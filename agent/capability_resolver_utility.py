"""Utility selection, switching penalty, exploration and cascade order (stage 4).

Pure functions over admitted :class:`agent.capability_resolver.Scored` rows (no config, network or
clock). They serve the ``cost``, ``latency`` and ``quality`` objectives; ``order`` stays the user's
explicit ranking (``capability_resolver_scoring.order_choice``).

    U = P(success) − λ·cost − μ·latency − ρ·risk        risk = malformed rate + posterior std of P

Cost and latency are normalized to the most expensive/slowest admitted candidate, so a success is
worth 1 and the weights (``WEIGHTS``, one row per objective) say how much of a success a full
step in cost or latency is worth. An unknown cost or latency counts as the worst admitted.

**Switching penalty (score the path, not the step).** The incumbent keeps the job unless a
challenger's utility beats it by more than what switching costs:

* ``floor``: differences this small are noise;
* ``uncertainty``: the two estimates' combined posterior std. Two unmeasured candidates are close to
  indistinguishable, and as the ledger fills this shrinks toward zero;
* ``cache``: the incumbent declares ``affinity: context`` and the ledger measured a cache saving
  per call. A switch throws that saving away, costed in the same normalized units as ``cost``;
* ``session``: the incumbent declares ``affinity: session``, a warm session or sandbox a switch
  discards;
* ``dependencies``: resources the challenger occupies that the incumbent does not (a new
  provider, credential or endpoint to warm up).

A caller's explicit ``policy.stickiness`` replaces the derived penalty with a fixed margin.

**Exploration** picks an under-sampled admitted candidate with probability ``exploration_rate``.
The draw comes from a ``random.Random`` seeded by (seed, capability, scope, samples so far), so the
same stats and seed always give the same decision. It never happens under a ``strict`` policy, never
on a candidate that declares ``reversible: false``, and never on an unhealthy one.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from agent.capability_resolver import Policy, Request, Scored

# objective -> (λ cost, μ latency, ρ risk), in units of one success.
WEIGHTS: Dict[str, Tuple[float, float, float]] = {
    "cost": (0.5, 0.05, 0.25),
    "latency": (0.05, 0.5, 0.25),
    "quality": (0.01, 0.01, 0.1),
}
SWITCH_FLOOR = 0.02
SESSION_SWITCH = 0.10
DEPENDENCY_SWITCH = 0.01


def _scale(values: Sequence[Optional[float]]) -> float:
    known = [v for v in values if v is not None]
    return max(known) if known else 0.0


def _norm(value: Optional[float], scale: float) -> float:
    if value is None:
        return 1.0
    return value / scale if scale > 0 else 0.0


def annotate(admitted: Sequence[Scored], policy: Policy) -> Tuple[List[Scored], float]:
    """Every admitted row with its utility and terms; also returns the cost scale (USD per unit)."""
    lam, mu, rho = WEIGHTS[policy.optimize]
    cost_scale = _scale([s.cost_usd for s in admitted])
    latency_scale = _scale([s.latency_ms for s in admitted])
    out = []
    for s in admitted:
        est = s.estimate
        terms = (
            ("value", est.p_success),
            ("cost", -lam * _norm(s.cost_usd, cost_scale)),
            ("latency", -mu * _norm(s.latency_ms, latency_scale)),
            ("risk", -rho * (est.malformed_rate + est.success_std)),
        )
        out.append(dataclasses.replace(s, utility=sum(v for _, v in terms), terms=terms))
    return out, cost_scale


def switching_penalty(current: Scored, challenger: Scored, policy: Policy,
                      cost_scale: float) -> Tuple[float, str]:
    """``(penalty, breakdown)`` in utility units for replacing ``current`` with ``challenger``."""
    if policy.stickiness is not None:
        return policy.stickiness, f"policy stickiness {policy.stickiness:g}"
    lam = WEIGHTS[policy.optimize][0]
    held, new = current.candidate.contract, challenger.candidate.contract
    parts = [("floor", SWITCH_FLOOR),
             ("uncertainty", math.hypot(current.estimate.success_std, challenger.estimate.success_std))]
    if held.affinity == "context" and current.estimate.cache_saving_usd > 0 and cost_scale > 0:
        parts.append(("cache", lam * current.estimate.cache_saving_usd / cost_scale))
    if held.affinity == "session":
        parts.append(("session", SESSION_SWITCH))
    added = sorted(set(new.resources) - set(held.resources))
    if added:
        parts.append(("dependencies", DEPENDENCY_SWITCH * len(added)))
    total = sum(v for _, v in parts)
    return total, " + ".join(f"{name} {value:.3f}" for name, value in parts)


def _sticky(ordered: Sequence[Scored], incumbent: Optional[str], policy: Policy, request: Request,
            cost_scale: float) -> Tuple[Scored, Optional[str]]:
    from agent.capability_resolver_scoring import soft_penalty
    best = ordered[0]
    if incumbent is None or best.candidate.identity == incumbent:
        return best, None
    current = next((s for s in ordered if s.candidate.identity == incumbent), None)
    if current is None:
        return best, "previous implementation no longer admitted"
    if soft_penalty(current, request) > soft_penalty(best, request):
        return best, f"switched from {current.label}: marked unhealthy or lacks structured output"
    gain = best.utility - current.utility
    penalty, breakdown = switching_penalty(current, best, policy, cost_scale)
    if gain > penalty:
        return best, (f"switched from {current.label}: {best.label} better by U {gain:+.3f} > "
                      f"switching penalty {penalty:.3f} ({breakdown})")
    return current, (f"kept incumbent {current.label}: {best.label} better by only U {gain:+.3f} <= "
                     f"switching penalty {penalty:.3f} ({breakdown})")


def _explore(ordered: Sequence[Scored], chosen: Scored, policy: Policy, *, rate: float, seed: int,
             capability: str, scope: str) -> Tuple[Optional[Scored], Optional[str]]:
    if rate <= 0 or policy.strict:
        return None, None
    pool = [s for s in ordered if s is not chosen and s.estimate.under_sampled and s.candidate.healthy
            and s.candidate.contract.reversible is not False]
    if not pool:
        return None, None
    samples = sum(s.estimate.samples for s in ordered)
    draw = random.Random(f"{seed}|{capability}|{scope}|{samples}").random()
    if draw >= rate:
        return None, None
    pick = min(pool, key=lambda s: (s.estimate.samples, s.candidate.order, s.candidate.label))
    return pick, (f"explored {pick.label}: n={pick.estimate.samples} under-sampled "
                  f"(draw {draw:.3f} < exploration_rate {rate:g})")


def select(admitted: Sequence[Scored], incumbent: Optional[str], policy: Policy, request: Request, *,
           exploration_rate: float, seed: int, capability: str, scope: str,
           cascade: bool) -> Tuple[List[Scored], Optional[str]]:
    """``(ordered, note)``: ``ordered[0]`` serves first, the rest are the escalation/failover order."""
    from agent.capability_resolver_scoring import soft_penalty
    rows, cost_scale = annotate(admitted, policy)
    rows.sort(key=lambda s: (*soft_penalty(s, request), -s.utility, s.candidate.order, s.candidate.label))
    if cascade:
        rows = cascade_order(rows, request)
        chosen, note = rows[0], "cascade: cheapest per verified success first, escalating on verification failure"
    else:
        chosen, note = _sticky(rows, incumbent, policy, request, cost_scale)
    explored, why = _explore(rows, chosen, policy, rate=exploration_rate, seed=seed, capability=capability,
                             scope=scope)
    if explored is not None:
        chosen, note = explored, why if note is None else f"{note}; {why}"
    return [chosen] + [s for s in rows if s is not chosen], note


def cascade_order(rows: Sequence[Scored], request: Request) -> List[Scored]:
    """Ascending expected cost per verified success (cost / P): the order that minimizes expected
    spend when every attempt is verified and a failure escalates to the next."""
    from agent.capability_resolver_scoring import soft_penalty

    def per_success(s: Scored) -> float:
        if s.cost_usd is None:
            return math.inf
        return s.cost_usd / max(s.estimate.p_success, 1e-9)

    return sorted(rows, key=lambda s: (*soft_penalty(s, request), per_success(s), -s.utility,
                                       s.candidate.order, s.candidate.label))
