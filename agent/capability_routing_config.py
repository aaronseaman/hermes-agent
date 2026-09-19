"""config.yaml knobs for learned routing (``agent.learned_routing``) and admission
(``agent.admission``), clamped to safe ranges. A malformed value means its default."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class LearnedRoutingSettings:
    # Learning needs measurements: False (the ledger is off) means every candidate is scored on its
    # declared contract and nothing is explored.
    learning: bool = False
    exploration_rate: float = 0.05
    prior_strength: float = 10.0
    min_samples: int = 5
    seed: int = 0
    cascade_enabled: bool = True
    cascade_max_attempts: int = 3

    @property
    def effective_exploration_rate(self) -> float:
        return self.exploration_rate if self.learning else 0.0


@dataclass(frozen=True)
class AdmissionSettings:
    max_wait_s: float = 60.0
    backoff_base_s: float = 2.0
    backoff_max_s: float = 300.0
    ceilings: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)

    def ceiling(self, resource: str):
        return dict(self.ceilings).get(resource)


def _num(raw: Any, default: float, lo: float, hi: float = math.inf) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        return default
    return min(max(float(raw), lo), hi)


def _flag(raw: Any, default: bool) -> bool:
    return raw if isinstance(raw, bool) else default


def _section(raw: Any, key: str) -> Mapping[str, Any]:
    value = raw.get(key) if isinstance(raw, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def parse_learned_routing(raw: Any, *, learning: bool) -> LearnedRoutingSettings:
    raw = raw if isinstance(raw, Mapping) else {}
    cascade = _section(raw, "cascade")
    d = LearnedRoutingSettings()
    return LearnedRoutingSettings(
        learning=learning,
        exploration_rate=_num(raw.get("exploration_rate"), d.exploration_rate, 0.0, 1.0),
        prior_strength=_num(raw.get("prior_strength"), d.prior_strength, 0.0),
        min_samples=int(_num(raw.get("min_samples"), d.min_samples, 1)),
        seed=int(_num(raw.get("seed"), d.seed, -2**62, 2**62)),
        cascade_enabled=_flag(cascade.get("enabled"), d.cascade_enabled),
        cascade_max_attempts=int(_num(cascade.get("max_attempts"), d.cascade_max_attempts, 1)),
    )


def parse_admission(raw: Any) -> AdmissionSettings:
    raw = raw if isinstance(raw, Mapping) else {}
    d = AdmissionSettings()
    ceilings = tuple(sorted(
        (str(k), int(v)) for k, v in _section(raw, "ceilings").items()
        if not isinstance(v, bool) and isinstance(v, int) and v > 0
    ))
    base = _num(raw.get("backoff_base_s"), d.backoff_base_s, 0.0)
    return AdmissionSettings(
        max_wait_s=_num(raw.get("max_wait_s"), d.max_wait_s, 0.0),
        backoff_base_s=base,
        backoff_max_s=_num(raw.get("backoff_max_s"), d.backoff_max_s, base),
        ceilings=ceilings,
    )


def _agent_section() -> Mapping[str, Any]:
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly() or {}
    agent = config.get("agent") if isinstance(config, Mapping) else None
    return agent if isinstance(agent, Mapping) else {}


def load_learned_routing() -> LearnedRoutingSettings:
    """The active profile's settings; learning is on exactly when its call ledger is."""
    agent = _agent_section()
    ledger = _section(agent, "call_ledger")
    return parse_learned_routing(agent.get("learned_routing"), learning=ledger.get("enabled") is True)


def load_admission() -> AdmissionSettings:
    return parse_admission(_agent_section().get("admission"))
