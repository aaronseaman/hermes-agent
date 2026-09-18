"""The ``model`` implementation kind: candidates from config + existing model metadata, dispatched
through ``agent.auxiliary_client.call_llm`` (no new client path).

A capability is an ``auxiliary.<name>`` block in config.yaml — the per-task shape auxiliary
routing already reads (provider/model/base_url/api_key/key_env/api_mode/timeout/extra_body/
reasoning_effort/fallback_chain). Without ``candidates`` the block's own route is the one
candidate and is dispatched with NO overrides, exactly as ``call_llm(task=<name>)`` resolves any
other auxiliary task. With ``candidates``, each entry is a route in the ``fallback_chain`` entry
shape. Either may declare contract priors: ``quality`` (low|medium|high), ``local`` (bool),
``latency_ms``, ``cost: {input, output}`` (USD per million tokens), ``context_window``.

Undeclared values come from what Hermes already keeps — ``agent.models_dev`` (offline cache:
context window, structured output, image input, list price), ``agent.usage_pricing`` (billing
route pricing; looked up only when the policy ranks or bounds cost), ``agent.model_metadata``
(local-endpoint detection) and auxiliary routing's unhealthy-provider cache. No model catalog
lives here.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from agent.capability_resolver import (
    Candidate, Contract, Policy, ResolverConfigError, non_negative, optional_bool, quality_tier,
)

logger = logging.getLogger(__name__)

KIND = "model"
_ROUTE_KEYS = ("provider", "model", "base_url", "api_key", "key_env", "api_key_env", "api_mode")


def _text(entry: Mapping[str, Any], key: str) -> str:
    return str(entry.get(key) or "").strip()


def _route_of(entry: Mapping[str, Any]) -> Dict[str, str]:
    route = {k: _text(entry, k) for k in _ROUTE_KEYS if _text(entry, k)}
    if route.get("provider", "").lower() == "auto":
        route.pop("provider")
    if route.get("model", "").lower() == "auto":
        route.pop("model")
    return route


def _declared_cost(entry: Mapping[str, Any], where: str) -> Optional[Tuple[float, float]]:
    raw = entry.get("cost")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or "input" not in raw or "output" not in raw:
        raise ResolverConfigError(f"{where}.cost must be a mapping with input and output (USD per million tokens)")
    return non_negative(raw["input"], "cost.input", where), non_negative(raw["output"], "cost.output", where)


def _metadata(provider: str, model: str) -> Any:
    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_info
        return get_model_info(provider, model, allow_network=False)
    except Exception:
        logger.debug("semantic_call: models.dev lookup failed for %s/%s", provider, model, exc_info=True)
        return None


def _endpoint(provider: str, base_url: str) -> str:
    if base_url or not provider:
        return base_url
    from agent.auxiliary_health import _custom_health_base_url
    return _custom_health_base_url(provider) or ""  # named custom providers resolve to an endpoint


def _locality(entry: Mapping[str, Any], url: str, where: str) -> Optional[bool]:
    declared = optional_bool(entry.get("local"), "local", where)
    if declared is not None or not url:
        return declared
    from agent.model_metadata import is_local_endpoint
    return is_local_endpoint(url)


def _pricing(entry: Mapping[str, Any], route: Mapping[str, str], url: str, local: Optional[bool],
             info: Any) -> Tuple[Optional[float], Optional[float]]:
    if local:
        return 0.0, 0.0
    provider, model = route.get("provider", ""), route.get("model", "")
    if not model:
        return None, None
    try:
        from agent.usage_pricing import get_pricing_entry
        from hermes_cli.fallback_config import resolve_entry_api_key
        priced = get_pricing_entry(model, provider=provider or None, base_url=url or None,
                                   api_key=resolve_entry_api_key(dict(entry)))
        if priced is not None and priced.input_cost_per_million is not None \
                and priced.output_cost_per_million is not None:
            return float(priced.input_cost_per_million), float(priced.output_cost_per_million)
    except Exception:
        logger.debug("semantic_call: pricing lookup failed for %s/%s", provider, model, exc_info=True)
    if info is not None and (info.cost_input or info.cost_output):
        return float(info.cost_input), float(info.cost_output)
    return None, None


def _healthy(provider: str, url: str) -> bool:
    if not provider:
        return True
    from agent.auxiliary_client import _is_provider_unhealthy, _normalize_chain_label
    return not _is_provider_unhealthy(_normalize_chain_label(provider), url or None)


def _host(url: str) -> str:
    from utils import base_url_hostname
    return base_url_hostname(url) or "" if url else ""


def build_candidate(entry: Mapping[str, Any], *, label: str, order: int, policy: Policy,
                    own_route: bool = False) -> Candidate:
    """One model candidate. ``own_route``: the capability block itself — dispatched with no
    route overrides, so it resolves exactly like any other auxiliary task."""
    if not own_route and not (_text(entry, "provider") or _text(entry, "base_url")):
        raise ResolverConfigError(f"{label} must declare provider or base_url")
    route = _route_of(entry)
    provider, model = route.get("provider", ""), route.get("model", "")
    url = _endpoint(provider, route.get("base_url", ""))
    info = _metadata(provider, model)
    local = _locality(entry, url, label)
    cost = _declared_cost(entry, label)
    if cost is None and (policy.optimize == "cost" or policy.max_cost_usd is not None):
        cost = _pricing(entry, route, url, local, info)
    latency = non_negative(entry.get("latency_ms"), "latency_ms", label)
    context = non_negative(entry.get("context_window"), "context_window", label)
    if context is None and info is not None and info.context_window:
        context = info.context_window
    return Candidate(
        label=label, kind=KIND, order=order, display=f"{provider or 'auto'}/{model or 'default'}",
        # Stable across calls; never carries a credential or a URL path/query.
        identity=f"{KIND}:{provider or 'auto'}/{model or 'default'}@{_host(url)}",
        contract=Contract(
            quality=quality_tier(entry.get("quality"), "quality", label), local=local,
            latency_ms=int(latency) if latency is not None else None,
            input_usd_per_mtok=cost[0] if cost else None, output_usd_per_mtok=cost[1] if cost else None,
            context_window=int(context) if context else None,
            structured_output=info.structured_output if info is not None else None,
            image_input=info.supports_vision() if info is not None else None,
        ),
        healthy=_healthy(provider, url),
        target={} if own_route else route,
    )


def _route_overrides(route: Mapping[str, str]) -> Dict[str, str]:
    from hermes_cli.fallback_config import resolve_entry_api_key
    overrides = {key: route[key] for key in ("provider", "model", "base_url", "api_mode") if route.get(key)}
    api_key = resolve_entry_api_key(dict(route))
    if api_key:
        overrides["api_key"] = api_key
    return overrides


def invoke(candidate: Candidate, capability: str, messages: List[Dict[str, Any]], *, fence: Optional[str],
           extra_body: Optional[Dict[str, Any]], max_tokens: Optional[int], temperature: Optional[float],
           reasoning: Optional[Dict[str, Any]], timeout: Optional[float], main_runtime: Optional[Dict[str, Any]],
           cancel_check: Optional[Callable[[], bool]]) -> Tuple[Any, Dict[str, str]]:
    """One request through ``call_llm(task=<capability>)``: credentials and pools, profile scope,
    per-task timeout/extra_body/reasoning/max_concurrency, transient retries, fallback (unless
    fenced), relay and usage accounting all come from the auxiliary client. Returns
    ``(response, route_info)``; ``route_info`` names the provider/model that answered."""
    from agent.auxiliary_client import aux_interrupt_protection, call_llm
    from agent.auxiliary_fallback_fence import fenced_provider_fallback
    route_info: Dict[str, str] = {}
    interruptible = (aux_interrupt_protection(True, cancel_check=cancel_check)
                     if cancel_check is not None else contextlib.nullcontext())
    with fenced_provider_fallback(fence), interruptible:
        response = call_llm(
            task=capability, messages=messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout,
            main_runtime=main_runtime, extra_body=extra_body, reasoning_config=reasoning, route_info=route_info,
            **_route_overrides(candidate.target),
        )
    return response, route_info
