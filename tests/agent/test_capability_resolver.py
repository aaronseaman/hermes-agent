"""Behaviour contracts for the capability resolver (selection, stickiness, refusal)."""

import pytest

from agent.capability_resolver import (
    Candidate, Contract, IncumbentStore, Observation, Policy, Request, Resolver, ResolverConfigError,
    UnsatisfiablePolicyError, merge_policy,
)

REQUEST = Request(input_tokens=1000, max_output_tokens=500)


def _cand(name, order, **contract):
    return Candidate(label=f"auxiliary.cap.candidates[{order}]", kind="model", identity=f"model:{name}",
                     display=name, order=order, contract=Contract(**contract))


def _priced(name, order, usd_per_mtok, **extra):
    return _cand(name, order, input_usd_per_mtok=usd_per_mtok, output_usd_per_mtok=usd_per_mtok, **extra)


@pytest.fixture
def resolver():
    return Resolver(incumbents=IncumbentStore())


@pytest.mark.parametrize("optimize,expected", [("cost", "cheap"), ("latency", "fast"), ("quality", "smart")])
def test_selection_follows_objective_and_explains_itself(resolver, optimize, expected):
    candidates = [
        _cand("cheap", 0, input_usd_per_mtok=0.1, output_usd_per_mtok=0.1, latency_ms=4000, quality="low"),
        _cand("fast", 1, input_usd_per_mtok=5.0, output_usd_per_mtok=5.0, latency_ms=300, quality="medium"),
        _cand("smart", 2, input_usd_per_mtok=15.0, output_usd_per_mtok=15.0, latency_ms=9000, quality="high"),
    ]
    resolution = resolver.resolve("cap", candidates, Policy(optimize=optimize), REQUEST)
    assert resolution.chosen.display == expected
    assert f"chose {resolution.chosen.label} ({expected})" in resolution.explanation
    assert f"optimize={optimize}" in resolution.explanation


def test_hard_constraints_reject_with_reasons_and_unknowns_fail_closed(resolver):
    candidates = [
        _priced("remote", 0, 1.0, quality="high", local=False),
        _priced("unknown-locality", 1, 1.0, quality="high"),
        _priced("local-low", 2, 0.0, quality="low", local=True),
        _priced("local-high", 3, 0.0, quality="high", local=True),
    ]
    resolution = resolver.resolve("cap", candidates, Policy(local_only=True, min_quality="medium"), REQUEST)
    assert resolution.chosen.display == "local-high"
    reasons = {c.display: why for c, why in resolution.rejected}
    assert set(reasons) == {"remote", "unknown-locality", "local-low"}
    assert "locality unknown" in reasons["unknown-locality"]
    assert "min_quality" in reasons["local-low"]


def test_unsatisfiable_policy_fails_clearly_instead_of_picking_something(resolver):
    candidates = [_priced("frontier", 0, 15.0, quality="high"), _cand("unpriced", 1, quality="high")]
    with pytest.raises(UnsatisfiablePolicyError) as err:
        resolver.resolve("cap", candidates, Policy(max_cost_usd=0.001), REQUEST)
    message = str(err.value)
    assert "Nothing was sent" in message
    assert "frontier" in message and "unpriced" in message and "cost unknown" in message


def test_ties_break_deterministically_by_config_order(resolver):
    candidates = [_priced("b", 1, 1.0), _priced("a", 0, 1.0), _priced("c", 2, 1.0)]
    chosen = {resolver.resolve("cap", candidates, Policy(optimize="cost"), REQUEST).chosen.display for _ in range(5)}
    assert chosen == {"a"}


def test_stickiness_keeps_incumbent_within_margin_and_switches_beyond_it(resolver):
    incumbent, challenger = _priced("incumbent", 0, 1.00), _priced("challenger", 1, 0.95)
    policy = Policy(optimize="cost", stickiness=0.15)
    resolver.served("cap", incumbent, scope="s")

    kept = resolver.resolve("cap", [incumbent, challenger], policy, REQUEST, scope="s")
    assert kept.chosen.display == "incumbent"
    assert "kept incumbent" in kept.explanation

    clearly_cheaper = _priced("challenger", 1, 0.50)
    switched = resolver.resolve("cap", [incumbent, clearly_cheaper], policy, REQUEST, scope="s")
    assert switched.chosen.display == "challenger"
    assert "switched from" in switched.explanation


def test_without_an_incumbent_the_best_candidate_wins(resolver):
    policy = Policy(optimize="cost", stickiness=0.15)
    resolution = resolver.resolve("cap", [_priced("a", 0, 1.00), _priced("b", 1, 0.95)], policy, REQUEST, scope="s")
    assert resolution.chosen.display == "b"


def test_incumbency_is_scoped(resolver):
    incumbent, challenger = _priced("incumbent", 0, 1.00), _priced("challenger", 1, 0.95)
    resolver.served("cap", incumbent, scope="session-a")
    policy = Policy(optimize="cost")
    assert resolver.resolve("cap", [incumbent, challenger], policy, REQUEST, scope="session-a").chosen is incumbent
    assert resolver.resolve("cap", [incumbent, challenger], policy, REQUEST, scope="session-b").chosen is challenger


def test_unhealthy_incumbent_is_not_sticky(resolver):
    incumbent = Candidate(label="i", kind="model", identity="model:i", display="i", order=0,
                          contract=Contract(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0), healthy=False)
    resolver.served("cap", incumbent)
    resolution = resolver.resolve("cap", [incumbent, _priced("ok", 1, 1.05)], Policy(optimize="cost"), REQUEST)
    assert resolution.chosen.display == "ok"


def test_observed_profile_overrides_declared_prior():
    class Measured:
        def observe(self, capability, identity):
            return Observation(samples=20, latency_ms=5000.0) if identity == "model:claims-fast" else None

    candidates = [_cand("claims-fast", 0, latency_ms=100), _cand("steady", 1, latency_ms=900)]
    policy = Policy(optimize="latency")
    assert Resolver(incumbents=IncumbentStore()).resolve("cap", candidates, policy, REQUEST).chosen.display == \
        "claims-fast"
    measured = Resolver(observed=Measured(), incumbents=IncumbentStore()).resolve("cap", candidates, policy, REQUEST)
    assert measured.chosen.display == "steady"


def test_caller_policy_can_only_tighten_configured_constraints():
    configured = {"local_only": True, "min_quality": "high", "max_cost_usd": 0.01, "optimize": "quality"}
    merged = merge_policy(configured, {"local_only": False, "min_quality": "low", "max_cost_usd": 5,
                                       "optimize": "cost"}, capability="cap")
    assert merged.local_only is True
    assert merged.min_quality == "high"
    assert merged.max_cost_usd == 0.01
    assert merged.optimize == "cost"  # the objective is the caller's to pick


@pytest.mark.parametrize("bad", [{"optimize": "vibes"}, {"stickiness": 1.5}, {"max_cost_usd": -1}, {"surprise": 1}])
def test_malformed_policy_is_rejected(bad):
    with pytest.raises(ResolverConfigError):
        merge_policy(None, bad, capability="cap")
