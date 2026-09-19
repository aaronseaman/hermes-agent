"""Behaviour contracts for stage 4: the resolver learns from ledger capability records.

Selection, shrinkage toward the declared prior, the switching penalty, exploration and
cache-adjusted cost. The E2E through semantic_call lives in test_semantic_call_cascade.py.
"""

import json
import os

import pytest

from agent.capability_profile import ProfileStats, attempt_cost
from agent.capability_profile_ledger import LedgerProfile
from agent.capability_resolver import Candidate, Contract, IncumbentStore, Policy, Request, Resolver
from agent.capability_routing_config import LearnedRoutingSettings

REQUEST = Request(input_tokens=1000, max_output_tokens=500)


def _cand(name, order, usd, **contract):
    return Candidate(label=f"auxiliary.cap.candidates[{order}]", kind="model", identity=f"model:{name}",
                     display=name, order=order,
                     contract=Contract(input_usd_per_mtok=usd, output_usd_per_mtok=usd, **contract))


def _records(stats, name, outcomes, cost=None, saving=None, latency=100.0):
    for outcome in outcomes:
        stats.add({"kind": "capability", "capability": "cap", "identity": f"model:{name}", "outcome": outcome,
                   "cost_usd": cost, "cache_saving_usd": saving, "latency_ms": latency})


def _resolver(stats, **settings):
    base = dict(learning=True, exploration_rate=0.0, prior_strength=10, min_samples=5)
    return Resolver(observed=stats, incumbents=IncumbentStore(), settings=LearnedRoutingSettings(**{**base, **settings}))


# Under the declared priors alone the pricier, higher-quality model wins a cost policy: its cost
# is only 25% higher and "high" implies a better success prior than "low".
CHEAP, PRICEY = _cand("cheap", 0, 0.8, quality="low"), _cand("pricey", 1, 1.0, quality="high")


def test_prior_alone_prefers_the_higher_quality_model():
    assert _resolver(ProfileStats()).resolve("cap", [CHEAP, PRICEY], Policy(optimize="cost"), REQUEST) \
        .chosen.display == "pricey"


def test_a_cheap_model_that_reliably_succeeds_wins_a_cost_policy():
    stats = ProfileStats()
    _records(stats, "cheap", ["ok"] * 40)
    resolution = _resolver(stats).resolve("cap", [CHEAP, PRICEY], Policy(optimize="cost"), REQUEST)
    assert resolution.chosen.display == "cheap"
    assert "n=40 observed" in resolution.explanation


def test_a_cheap_model_that_keeps_failing_does_not():
    stats = ProfileStats()
    _records(stats, "cheap", ["malformed"] * 40)
    assert _resolver(stats).resolve("cap", [CHEAP, PRICEY], Policy(optimize="cost"), REQUEST) \
        .chosen.display == "pricey"


@pytest.mark.parametrize("min_samples", [5, 1])
def test_few_samples_leave_the_prior_in_charge(min_samples):
    """Three lucky calls: below min_samples the prior IS the estimate; above it the Beta prior
    (strength 10) still outweighs three successes."""
    stats = ProfileStats()
    _records(stats, "cheap", ["ok"] * 3)
    resolution = _resolver(stats, min_samples=min_samples).resolve("cap", [CHEAP, PRICEY], Policy(optimize="cost"),
                                                                   REQUEST)
    assert resolution.chosen.display == "pricey"


def _measured(stats, name, cost, n=400, saving=None):
    _records(stats, name, ["ok"] * n, cost=cost, saving=saving)


def _switch(incumbent_saving, challenger_cost):
    """Incumbent A at $0.010/call vs challenger B, both well measured, A serving in scope s."""
    stats = ProfileStats()
    a = _cand("a", 0, 1.0, affinity="context", resources=("provider:a",))
    b = _cand("b", 1, 1.0, affinity="context", resources=("provider:b",))
    _measured(stats, "a", 0.010, saving=incumbent_saving)
    _measured(stats, "b", challenger_cost)
    resolver = _resolver(stats)
    resolver.served("cap", a, scope="s")
    return resolver.resolve("cap", [a, b], Policy(optimize="cost"), REQUEST, scope="s")


def test_switching_penalty_keeps_the_incumbent_within_its_margin():
    kept = _switch(None, 0.0095)  # 5% cheaper: inside floor + uncertainty + the new provider to warm up
    assert kept.chosen.display == "a"
    assert "kept incumbent" in kept.explanation and "dependencies" in kept.explanation


def test_switching_penalty_allows_a_switch_beyond_it():
    switched = _switch(None, 0.007)
    assert switched.chosen.display == "b"
    assert "switched from" in switched.explanation


def test_a_measured_cache_saving_raises_the_cost_of_switching():
    """The same 30%-cheaper challenger is not worth losing an incumbent's warm cache that saves
    $0.004 per call."""
    held = _switch(0.004, 0.007)
    assert held.chosen.display == "a"
    assert "cache" in held.explanation


def test_an_explicit_stickiness_is_a_fixed_margin():
    stats = ProfileStats()
    a, b = _cand("a", 0, 1.0), _cand("b", 1, 1.0)
    _measured(stats, "a", 0.010)
    _measured(stats, "b", 0.007)
    resolver = _resolver(stats)
    resolver.served("cap", a, scope="s")
    assert resolver.resolve("cap", [a, b], Policy(optimize="cost", stickiness=0.5), REQUEST, scope="s") \
        .chosen.display == "a"


def _explore_setup(**policy):
    stats = ProfileStats()
    known = _cand("known", 0, 1.0, quality="high")
    fresh = _cand("fresh", 1, 1.0, quality="high")
    _measured(stats, "known", 0.001)
    return _resolver(stats, exploration_rate=1.0), [known, fresh], Policy(optimize="cost", **policy)


def test_exploration_tries_an_under_sampled_implementation_and_explains_why():
    resolver, candidates, policy = _explore_setup()
    resolution = resolver.resolve("cap", candidates, policy, REQUEST)
    assert resolution.chosen.display == "fresh"
    assert "explored" in resolution.explanation and "draw" in resolution.explanation


def test_a_strict_policy_never_explores():
    resolver, candidates, policy = _explore_setup(strict=True)
    assert resolver.resolve("cap", candidates, policy, REQUEST).chosen.display == "known"


def test_an_irreversible_implementation_is_never_explored():
    resolver, (known, fresh), policy = _explore_setup()
    irreversible = Candidate(label=fresh.label, kind="model", identity=fresh.identity, display="fresh", order=1,
                             contract=Contract(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0, quality="high",
                                               reversible=False))
    assert resolver.resolve("cap", [known, irreversible], policy, REQUEST).chosen.display == "known"


def test_same_stats_and_seed_give_the_same_decision():
    def decide(seed):
        stats = ProfileStats()
        _records(stats, "known", ["ok"] * 7)
        resolver = _resolver(stats, exploration_rate=0.5, seed=seed)
        candidates = [_cand("known", 0, 1.0), _cand("fresh", 1, 1.0)]
        return [resolver.resolve("cap", candidates, Policy(optimize="cost"), REQUEST, scope=str(i)).explanation
                for i in range(12)]
    assert decide(7) == decide(7)
    assert any("explored" in e for e in decide(7)) and any("explored" not in e for e in decide(7))


def test_learning_off_never_explores():
    stats = ProfileStats()
    resolver = Resolver(observed=stats, incumbents=IncumbentStore(),
                        settings=LearnedRoutingSettings(learning=False, exploration_rate=1.0))
    candidates = [_cand("a", 0, 1.0), _cand("b", 1, 2.0)]
    assert "explored" not in resolver.resolve("cap", candidates, Policy(optimize="cost"), REQUEST).explanation


class _Usage:
    def __init__(self, fresh, read, write, out):
        self.input_tokens, self.cache_read_tokens, self.cache_write_tokens, self.output_tokens = fresh, read, write, out

    @property
    def prompt_tokens(self):
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


def test_cost_is_cache_adjusted_and_the_saving_is_measured():
    contract = Contract(input_usd_per_mtok=10.0, output_usd_per_mtok=10.0, cache_read_usd_per_mtok=1.0)
    warm = _Usage(fresh=100, read=9000, write=0, out=100)
    cost, saving = attempt_cost(warm, model="stage4-unpriced-model", provider="custom", base_url=None,
                                contract=contract)
    uncached, _ = attempt_cost(_Usage(9100, 0, 0, 100), model="stage4-unpriced-model", provider="custom",
                               base_url=None, contract=contract)
    assert cost == pytest.approx(0.011)
    assert cost + saving == pytest.approx(uncached)


def test_a_cache_warm_implementation_wins_at_the_same_list_price():
    """Two routes with one list price: the one whose prompts hit the cache is cheaper in fact."""
    contract = Contract(input_usd_per_mtok=10.0, output_usd_per_mtok=10.0, cache_read_usd_per_mtok=1.0)
    stats = ProfileStats()
    for name, usage in (("warm", _Usage(100, 9000, 0, 100)), ("cold", _Usage(9100, 0, 0, 100))):
        cost, saving = attempt_cost(usage, model="stage4-unpriced-model", provider="custom", base_url=None,
                                    contract=contract)
        _measured(stats, name, cost, saving=saving)
    candidates = [_cand("cold", 0, 10.0), _cand("warm", 1, 10.0)]
    assert _resolver(stats).resolve("cap", candidates, Policy(optimize="cost"), REQUEST).chosen.display == "warm"


def _write(path, records, tail=""):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records) + tail)


def _rec(outcome):
    return {"kind": "capability", "capability": "cap", "identity": "model:x", "outcome": outcome,
            "cost_usd": 0.001, "latency_ms": 50}


def test_ledger_profile_tails_other_processes_incrementally(tmp_path):
    clock = [0.0]
    profile = LedgerProfile(tmp_path, clock=lambda: clock[0])
    other = tmp_path / "calls-2026-09-18-999999.jsonl"
    _write(other, [_rec("ok"), {"kind": "model", "outcome": "ok"}], tail='{"kind":"capabil')  # half-written line
    assert profile.observe("cap", "model:x").samples == 1

    with open(other, "a", encoding="utf-8") as fh:  # the writer finishes that line and adds another
        fh.write('ity","capability":"cap","identity":"model:x","outcome":"malformed"}\n')
    _write(other, [_rec("ok")])
    assert profile.observe("cap", "model:x").samples == 1  # refresh is throttled
    clock[0] += 10
    obs = profile.observe("cap", "model:x")
    assert (obs.samples, obs.successes, obs.malformed) == (3, 2, 1)


def test_ledger_profile_skips_its_own_files_because_they_were_ingested_live(tmp_path):
    profile = LedgerProfile(tmp_path, clock=lambda: 0.0)
    _write(tmp_path / f"calls-2026-09-18-{os.getpid()}.jsonl", [_rec("ok")])
    profile.ingest(_rec("ok"))
    assert profile.observe("cap", "model:x").samples == 1


def test_measurements_stay_with_their_profile(tmp_path, monkeypatch):
    """A→B→A: one process, two homes, two separate ledgers and profiles."""
    from agent.capability_profile_ledger import profile_for, reset_profiles
    reset_profiles()
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    profile_for().ingest(_rec("ok"))
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert profile_for().observe("cap", "model:x") is None
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert profile_for().observe("cap", "model:x").samples == 1
    reset_profiles()
