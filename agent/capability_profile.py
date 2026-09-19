"""Observed profiles per (capability, implementation), shrunk toward the declared prior.

Pure estimator: no I/O, no clock. ``ProfileStats`` folds ledger ``capability`` records (one per
``semantic_call`` attempt, ``agent/call_ledger.py::record_capability_attempt``) into sufficient
statistics per key, so a refresh costs only the records written since the last one
(``agent/capability_profile_ledger.py`` does the tailing). ``estimate`` turns one key's
:class:`~agent.capability_resolver.Observation` plus the candidate's declared prior into what
scoring reads.

Shrinkage, chosen so two lucky calls cannot win:

* below ``min_samples`` the estimate IS the prior (percentiles of three calls are noise) and is
  flagged ``under_sampled``, which is what exploration offers;
* success and malformed-output rates are Beta posteriors whose prior mean is the declared prior
  and whose weight is ``prior_strength`` pseudo-calls;
* cost, latency and cache savings blend the sample statistic with the prior by
  ``n / (n + prior_strength)`` (an unknown prior yields to the measurement).

Cost is cache-adjusted at record time (:func:`attempt_cost`): cache reads and writes bill at their
own rates, never as fresh input, and the saving versus an uncached request is recorded next to it.
That saving is what a switch away from a warm implementation throws away.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Tuple

# Latency percentiles come from a bounded window of recent calls: memory is O(keys), not O(ledger).
LATENCY_WINDOW = 256

# The success prior a declared quality tier implies (the only stage-3 prior about reliability).
# Unknown quality sits between medium and low: no claim, no reward.
SUCCESS_PRIOR = {"low": 0.80, "medium": 0.90, "high": 0.95, None: 0.85}

# outcome -> (counts toward success rate, succeeded, malformed, measures cost/latency)
# ``fallback`` = the candidate's own route failed and auxiliary fallback answered: a failure of this
# implementation whose cost and latency belong to someone else. ``cancelled`` says nothing about it.
_OUTCOMES: Dict[str, Tuple[bool, bool, bool, bool]] = {
    "ok": (True, True, False, True),
    "malformed": (True, False, True, True),
    "rejected": (True, False, False, True),
    "error": (True, False, False, False),
    "fallback": (True, False, False, False),
    "cancelled": (False, False, False, False),
}
OUTCOMES = tuple(_OUTCOMES)


@dataclass(frozen=True)
class Prior:
    """What the declared contract says, before any measurement. ``None`` = unknown."""

    success: float
    cost_usd: Optional[float]
    latency_ms: Optional[float]
    malformed_rate: float = 0.0


@dataclass(frozen=True)
class Estimate:
    samples: int
    under_sampled: bool
    p_success: float
    success_std: float
    malformed_rate: float
    cost_usd: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    cache_saving_usd: float          # mean per call; what losing this implementation's warm cache costs


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]): deterministic, no interpolation."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def beta_posterior(prior_mean: float, strength: float, hits: int, n: int) -> Tuple[float, float]:
    """Mean and standard deviation of Beta(prior_mean·k + hits, (1 − prior_mean)·k + n − hits)."""
    p = min(max(prior_mean, 0.0), 1.0)
    alpha, beta = p * strength + hits, (1.0 - p) * strength + (n - hits)
    total = alpha + beta
    if total <= 0:
        return p, 0.5
    return alpha / total, math.sqrt(alpha * beta / (total * total * (total + 1.0)))


def _blend(prior: Optional[float], sample: float, n: int, strength: float) -> float:
    if prior is None:
        return sample
    w = n / (n + strength) if n + strength > 0 else 1.0
    return (1.0 - w) * prior + w * sample


@dataclass
class _KeyStats:
    n: int = 0
    successes: int = 0
    malformed: int = 0
    cost_n: int = 0
    cost_sum: float = 0.0
    saving_sum: float = 0.0
    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))


class ProfileStats:
    """Incremental per-(capability, identity) sufficient statistics over capability records."""

    def __init__(self) -> None:
        self._stats: Dict[Tuple[str, str], _KeyStats] = {}

    def add(self, record: Mapping[str, Any]) -> bool:
        """Fold one ledger ``capability`` record in; False when it is not one this can use."""
        capability, identity = record.get("capability"), record.get("identity")
        spec = _OUTCOMES.get(str(record.get("outcome")))
        if not capability or not identity or spec is None or not spec[0]:
            return False
        _counts, succeeded, malformed, measured = spec
        st = self._stats.setdefault((str(capability), str(identity)), _KeyStats())
        st.n += 1
        st.successes += succeeded
        st.malformed += malformed
        if measured:
            cost = record.get("cost_usd")
            if isinstance(cost, (int, float)) and cost >= 0:
                st.cost_n += 1
                st.cost_sum += float(cost)
                saving = record.get("cache_saving_usd")
                st.saving_sum += float(saving) if isinstance(saving, (int, float)) and saving > 0 else 0.0
            latency = record.get("latency_ms")
            if isinstance(latency, (int, float)) and latency >= 0:
                st.latencies.append(float(latency))
        return True

    def observation(self, capability: str, identity: str):
        from agent.capability_resolver import Observation
        st = self._stats.get((capability, identity))
        if st is None or st.n == 0:
            return None
        lat = list(st.latencies)
        return Observation(
            samples=st.n, successes=st.successes, malformed=st.malformed,
            latency_ms=percentile(lat, 50) if lat else None,
            latency_p95_ms=percentile(lat, 95) if lat else None,
            cost_usd=st.cost_sum / st.cost_n if st.cost_n else None,
            cache_saving_usd=st.saving_sum / st.cost_n if st.cost_n else None,
        )


def estimate(obs: Any, prior: Prior, *, prior_strength: float, min_samples: int) -> Estimate:
    """What scoring reads for one candidate: the prior, sharpened by measurement once there is enough."""
    k = max(float(prior_strength), 0.0)
    n = obs.samples if obs is not None else 0
    if n < max(int(min_samples), 1):
        _, std = beta_posterior(prior.success, k, 0, 0)
        return Estimate(samples=n, under_sampled=True, p_success=prior.success, success_std=std,
                        malformed_rate=prior.malformed_rate, cost_usd=prior.cost_usd,
                        latency_p50_ms=prior.latency_ms, latency_p95_ms=prior.latency_ms, cache_saving_usd=0.0)
    if obs.successes is None:  # a source that measured only cost/latency says nothing about reliability
        p, std = beta_posterior(prior.success, k, 0, 0)
        malformed = prior.malformed_rate
    else:
        p, std = beta_posterior(prior.success, k, obs.successes, n)
        malformed, _ = beta_posterior(prior.malformed_rate, k, obs.malformed or 0, n)
    cost = _blend(prior.cost_usd, obs.cost_usd, n, k) if obs.cost_usd is not None else prior.cost_usd
    p50 = _blend(prior.latency_ms, obs.latency_ms, n, k) if obs.latency_ms is not None else prior.latency_ms
    p95_obs = obs.latency_p95_ms if obs.latency_p95_ms is not None else obs.latency_ms
    p95 = _blend(prior.latency_ms, p95_obs, n, k) if p95_obs is not None else prior.latency_ms
    saving = _blend(0.0, obs.cache_saving_usd, n, k) if obs.cache_saving_usd else 0.0
    return Estimate(samples=n, under_sampled=False, p_success=p, success_std=std, malformed_rate=malformed,
                    cost_usd=cost, latency_p50_ms=p50, latency_p95_ms=p95, cache_saving_usd=saving)


def _declared_price(usage: Any, contract: Any) -> Tuple[Optional[float], Optional[float]]:
    """``(actual, uncached)`` USD from the contract's declared rates; cache rates default to input."""
    rate_in, rate_out = contract.input_usd_per_mtok, contract.output_usd_per_mtok
    if rate_in is None or rate_out is None:
        return None, None
    read = contract.cache_read_usd_per_mtok if contract.cache_read_usd_per_mtok is not None else rate_in
    write = contract.cache_write_usd_per_mtok if contract.cache_write_usd_per_mtok is not None else rate_in
    actual = (usage.input_tokens * rate_in + usage.cache_read_tokens * read + usage.cache_write_tokens * write
              + usage.output_tokens * rate_out) / 1_000_000
    return actual, (usage.prompt_tokens * rate_in + usage.output_tokens * rate_out) / 1_000_000


def attempt_cost(usage: Any, *, model: str, provider: Optional[str], base_url: Optional[str],
                 contract: Any) -> Tuple[Optional[float], Optional[float]]:
    """``(cost_usd, cache_saving_usd)`` of one attempt, cache-adjusted.

    The billing route's pricing (``agent.usage_pricing``, the same pricer the ledger's model records
    use) wins; the candidate's declared ``cost:`` rates price an unpriced route. The saving is the
    same usage priced as if every prompt token were fresh input, minus the actual cost.
    """
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
    uncached_usage = CanonicalUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.output_tokens)
    actual = estimate_usage_cost(model, usage, provider=provider, base_url=base_url).amount_usd
    if actual is not None:
        uncached = estimate_usage_cost(model, uncached_usage, provider=provider, base_url=base_url).amount_usd
        actual_f = float(actual)
        return actual_f, max(0.0, float(uncached) - actual_f) if uncached is not None else None
    declared, uncached_declared = _declared_price(usage, contract)
    if declared is None:
        return None, None
    return declared, max(0.0, uncached_declared - declared)
