"""Fallback fence: keep one auxiliary call on the route its caller chose.

``call_llm`` recovers from a provider that is missing or failing by moving to another one (the
task's ``fallback_chain``, the main agent model, the auto-detection chain). That is right for
ordinary side tasks and wrong for a ``semantic_call`` whose policy says ``local_only``,
``min_quality`` or ``max_cost_usd``: a local model that is down must not quietly become a remote
frontier model. Such a call installs the fence; the auxiliary client then treats every
cross-provider fallback as unavailable, the original error surfaces, and the capability
resolver moves on to its next *admitted* candidate instead. Same-provider recovery (transient
retries, parameter strips, credential refresh and pool rotation) is untouched.

No fence installed — every other caller — means no change.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Optional

_fence: ContextVar[Optional[str]] = ContextVar("aux_fallback_fence", default=None)


@contextlib.contextmanager
def fenced_provider_fallback(reason: Optional[str]) -> Iterator[None]:
    """Refuse cross-provider fallback for auxiliary calls made in this context (``None`` = no fence)."""
    token = _fence.set(reason)
    try:
        yield
    finally:
        _fence.reset(token)


def fallback_fence_reason() -> Optional[str]:
    """Why cross-provider fallback is refused right now, or ``None`` when it is allowed."""
    return _fence.get()
