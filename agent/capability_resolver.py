"""Capability resolver: choose which implementation serves a requested capability.

A capability ("summarize_diff", "title_generation") names a transformation, never an
implementation. Its implementations are :class:`Candidate` rows: a model route today, and
deterministic code, a replayed result, an API, a CLI or a composite later. Each carries a
declared :class:`Contract` (quality tier, locality, cost/latency priors, context window,
modalities) and an opaque ``target`` that only its kind's invoker reads. The resolver never
looks inside ``target``, so a new kind of implementation is a new candidate builder + invoker
(``agent/semantic_call.py::_KINDS``), not a change here.

``Resolver.resolve`` admits candidates that satisfy the :class:`Policy`'s hard constraints,
ranks the rest on the objective (config order breaks ties), then applies **incumbent
stickiness**: the implementation that last served this capability in this scope keeps the job
unless a challenger beats it by more than the switching margin, so near-equal candidates do not
thrash. Every :class:`Resolution` carries an ``explanation``. No admitted candidate raises
:class:`UnsatisfiablePolicyError` before anything is sent — an unmet policy never falls through
to some other (possibly expensive) implementation.

Stage-4 seam: :class:`ObservedProfile`. Scoring reads each candidate's *effective* latency and
cost — a measured :class:`Observation` when the profile source has one, else the declared
prior. Today the only source is :class:`NoObservations`; the ledger-backed source (stage 4)
implements ``observe`` over ``agent/call_ledger`` records and is passed to ``Resolver(observed=)``.
Nothing else changes.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

# Ordered low → high. ``min_quality`` admits its tier and every tier after it.
QUALITY_TIERS = ("low", "medium", "high")
# "order" = first admitted candidate in config order: the user's declared preference, no hidden one.
OBJECTIVES = ("order", "cost", "latency", "quality")
# A challenger must beat the incumbent's objective score by this fraction to take over.
DEFAULT_STICKINESS = 0.15
# Output-token assumption for a cost estimate when the caller sets no max_tokens.
DEFAULT_OUTPUT_TOKENS_FOR_COST = 1024
_INCUMBENT_CAP = 512


class ResolverConfigError(ValueError):
    """A capability's config (candidates / priors / policy) or a caller policy is malformed."""


class UnsatisfiablePolicyError(RuntimeError):
    """No candidate satisfies the policy. Raised BEFORE any request is sent."""

    def __init__(self, capability: str, policy: "Policy", rejected: Sequence[Tuple["Candidate", str]]):
        self.capability, self.policy, self.rejected = capability, policy, tuple(rejected)
        reasons = "; ".join(f"{c.label} ({c.display}): {why}" for c, why in self.rejected) or "no candidates"
        super().__init__(
            f"capability '{capability}': no implementation satisfies policy [{policy.describe()}] — "
            f"{reasons}. Nothing was sent. Loosen the policy or add a candidate under "
            f"auxiliary.{capability}.candidates."
        )


def non_negative(raw: Any, key: str, where: str) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
        raise ResolverConfigError(f"{where}.{key} must be a non-negative number, got {raw!r}")
    return float(raw)


def quality_tier(raw: Any, key: str, where: str) -> Optional[str]:
    if raw in (None, ""):
        return None
    tier = str(raw).strip().lower()
    if tier not in QUALITY_TIERS:
        raise ResolverConfigError(f"{where}.{key} must be one of {', '.join(QUALITY_TIERS)}, got {raw!r}")
    return tier


def optional_bool(raw: Any, key: str, where: str) -> Optional[bool]:
    if raw is None or isinstance(raw, bool):
        return raw
    raise ResolverConfigError(f"{where}.{key} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class Policy:
    """What the caller needs. ``optimize`` ranks and ``stickiness`` damps switching; the rest are
    hard constraints. An UNKNOWN latency passes ``max_latency_ms`` (the request timeout still
    bounds it); an unknown quality, cost or locality FAILS the matching constraint — the resolver
    never guesses in the expensive or leaky direction."""

    optimize: str = "order"
    max_latency_ms: Optional[int] = None
    min_quality: Optional[str] = None
    max_cost_usd: Optional[float] = None
    local_only: bool = False
    stickiness: float = DEFAULT_STICKINESS

    @property
    def fences_fallback(self) -> bool:
        """Constraints that a provider fallback outside the admitted candidates could silently break."""
        return self.local_only or self.min_quality is not None or self.max_cost_usd is not None

    def describe(self) -> str:
        parts = [f"optimize={self.optimize}"]
        parts += [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                  for k, v in (("max_latency_ms", self.max_latency_ms), ("min_quality", self.min_quality),
                               ("max_cost_usd", self.max_cost_usd)) if v is not None]
        if self.local_only:
            parts.append("local_only")
        return ", ".join(parts)


_POLICY_KEYS = frozenset({"optimize", "max_latency_ms", "min_quality", "max_cost_usd", "local_only", "stickiness"})


def _parse_policy(raw: Any, where: str) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, Policy):
        return {k: getattr(raw, k) for k in _POLICY_KEYS}
    if not isinstance(raw, Mapping):
        raise ResolverConfigError(f"{where} must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _POLICY_KEYS
    if unknown:
        raise ResolverConfigError(f"{where} has unknown keys {sorted(unknown)}; allowed: {sorted(_POLICY_KEYS)}")
    optimize = raw.get("optimize")
    if optimize is not None and optimize not in OBJECTIVES:
        raise ResolverConfigError(f"{where}.optimize must be one of {', '.join(OBJECTIVES)}, got {optimize!r}")
    stickiness = non_negative(raw.get("stickiness"), "stickiness", where)
    if stickiness is not None and stickiness >= 1:
        raise ResolverConfigError(f"{where}.stickiness must be in [0, 1), got {stickiness!r}")
    latency = non_negative(raw.get("max_latency_ms"), "max_latency_ms", where)
    return {
        "optimize": optimize,
        "max_latency_ms": int(latency) if latency is not None else None,
        "min_quality": quality_tier(raw.get("min_quality"), "min_quality", where),
        "max_cost_usd": non_negative(raw.get("max_cost_usd"), "max_cost_usd", where),
        "local_only": optional_bool(raw.get("local_only"), "local_only", where),
        "stickiness": stickiness,
    }


def _tighter(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return b if a is None else a if b is None else min(a, b)


def merge_policy(configured: Any, requested: Any, *, capability: str) -> Policy:
    """``auxiliary.<capability>.policy`` tightened by the caller's policy.

    The caller picks the objective and stickiness but can only TIGHTEN constraints: a sandbox
    script cannot lift a profile's ``local_only`` or lower its ``min_quality``.
    """
    cfg = _parse_policy(configured, f"auxiliary.{capability}.policy")
    req = _parse_policy(requested, "policy")
    tiers = [t for t in (cfg.get("min_quality"), req.get("min_quality")) if t]
    latency = _tighter(cfg.get("max_latency_ms"), req.get("max_latency_ms"))
    stickiness = req.get("stickiness")
    if stickiness is None:
        stickiness = cfg.get("stickiness")
    return Policy(
        optimize=req.get("optimize") or cfg.get("optimize") or "order",
        max_latency_ms=int(latency) if latency is not None else None,
        min_quality=max(tiers, key=QUALITY_TIERS.index) if tiers else None,
        max_cost_usd=_tighter(cfg.get("max_cost_usd"), req.get("max_cost_usd")),
        local_only=bool(cfg.get("local_only")) or bool(req.get("local_only")),
        stickiness=DEFAULT_STICKINESS if stickiness is None else stickiness,
    )


@dataclass(frozen=True)
class Request:
    """The shape of one request (never its content): what constraints and cost estimates read."""

    input_tokens: int
    max_output_tokens: Optional[int] = None
    needs_image: bool = False
    wants_schema: bool = False


@dataclass(frozen=True)
class Contract:
    """Declared properties of one implementation. ``None`` = unknown."""

    quality: Optional[str] = None
    local: Optional[bool] = None
    latency_ms: Optional[int] = None
    input_usd_per_mtok: Optional[float] = None
    output_usd_per_mtok: Optional[float] = None
    context_window: Optional[int] = None
    structured_output: Optional[bool] = None
    image_input: Optional[bool] = None

    def estimated_cost_usd(self, request: Request) -> Optional[float]:
        if self.input_usd_per_mtok is None or self.output_usd_per_mtok is None:
            return None
        out_tokens = request.max_output_tokens or DEFAULT_OUTPUT_TOKENS_FOR_COST
        return (request.input_tokens * self.input_usd_per_mtok + out_tokens * self.output_usd_per_mtok) / 1_000_000


@dataclass(frozen=True)
class Candidate:
    """One implementation of a capability.

    ``identity`` is stable across calls (stickiness and observations key on it); ``label`` is
    where it was declared (a config path) and ``display`` what it is, both for explanations.
    ``target`` is the kind's dispatch data — for a model, the route incl. credentials — and is
    excluded from repr/compare so it never lands in a log line or an explanation.
    """

    label: str
    kind: str
    identity: str
    display: str
    order: int
    contract: Contract
    healthy: bool = True
    target: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class Observation:
    """What the ledger measured for one implementation of one capability."""

    samples: int
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None


class ObservedProfile(Protocol):
    """Source of measured behaviour. Stage 4 implements this over the call ledger."""

    def observe(self, capability: str, identity: str) -> Optional[Observation]: ...


class NoObservations:
    """No measurements yet: every candidate is scored on its declared contract."""

    def observe(self, capability: str, identity: str) -> Optional[Observation]:
        return None


@dataclass(frozen=True)
class Scored:
    """A candidate with the numbers scoring reads: observation first, declared prior second."""

    candidate: Candidate
    latency_ms: Optional[float]
    cost_usd: Optional[float]
    observed: bool

    @property
    def label(self) -> str:
        return self.candidate.label

    @property
    def display(self) -> str:
        return self.candidate.display


@dataclass(frozen=True)
class Resolution:
    capability: str
    policy: Policy
    request: Request
    chosen: Candidate
    ranked: Tuple[Candidate, ...]          # admitted, best first; ``chosen`` is first
    rejected: Tuple[Tuple[Candidate, str], ...]
    incumbent: Optional[str]               # identity that served last time in this scope
    explanation: str


class IncumbentStore:
    """Which implementation last served (profile, scope, capability). Bounded LRU, process-local:
    stickiness is a within-process damping of switches, not persisted state."""

    def __init__(self, cap: int = _INCUMBENT_CAP):
        self._cap = cap
        self._rows: "OrderedDict[tuple, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[str]:
        with self._lock:
            return self._rows.get(key)

    def set(self, key: tuple, identity: str) -> None:
        with self._lock:
            self._rows[key] = identity
            self._rows.move_to_end(key)
            while len(self._rows) > self._cap:
                self._rows.popitem(last=False)


_INCUMBENTS = IncumbentStore()


def _incumbent_key(capability: str, scope: str) -> tuple:
    # Keyed by profile home: one process serves many profiles, and their implementations differ.
    from hermes_constants import hermes_home_key
    return hermes_home_key(), scope, capability


class Resolver:
    def __init__(self, observed: Optional[ObservedProfile] = None, incumbents: Optional[IncumbentStore] = None):
        self._observed = observed or NoObservations()
        self._incumbents = incumbents or _INCUMBENTS

    def _score(self, capability: str, candidate: Candidate, request: Request) -> Scored:
        obs = self._observed.observe(capability, candidate.identity)
        declared_cost = candidate.contract.estimated_cost_usd(request)
        if obs is not None and obs.samples > 0:
            latency = obs.latency_ms if obs.latency_ms is not None else candidate.contract.latency_ms
            cost = obs.cost_usd if obs.cost_usd is not None else declared_cost
            return Scored(candidate, latency, cost, observed=True)
        return Scored(candidate, candidate.contract.latency_ms, declared_cost, observed=False)

    def resolve(self, capability: str, candidates: Sequence[Candidate], policy: Policy, request: Request,
                *, scope: str = "") -> Resolution:
        """Rank ``candidates``; raise :class:`UnsatisfiablePolicyError` when none is admitted."""
        from agent.capability_resolver_scoring import explain, rank, sticky_choice
        scored = [self._score(capability, c, request) for c in candidates]
        ranked, rejected = rank(scored, policy, request)
        if not ranked:
            raise UnsatisfiablePolicyError(capability, policy, [(s.candidate, why) for s, why in rejected])
        incumbent = self._incumbents.get(_incumbent_key(capability, scope))
        chosen, sticky_note = sticky_choice(ranked, incumbent, policy, request)
        ordered = [chosen] + [s for s in ranked if s is not chosen]
        return Resolution(
            capability=capability, policy=policy, request=request, chosen=chosen.candidate,
            ranked=tuple(s.candidate for s in ordered),
            rejected=tuple((s.candidate, why) for s, why in rejected), incumbent=incumbent,
            explanation=explain(capability, policy, ordered, rejected, sticky_note),
        )

    def served(self, capability: str, candidate: Candidate, *, scope: str = "") -> None:
        """Record that ``candidate`` served the capability: it is the incumbent from now on."""
        self._incumbents.set(_incumbent_key(capability, scope), candidate.identity)
