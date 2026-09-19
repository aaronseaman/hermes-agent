"""Admission controller: work is admitted against configured concurrency ceilings per scarce
resource, backs off when a provider signals exhaustion, and queues (bounded) rather than fails.

It never claims to know remaining quota. Hermes sees rate-limit *signals* (429s, ``Retry-After``,
the cross-session Nous guard), not budgets, so admission reacts to signals and enforces ceilings
the user configured.

**Resources** are strings, one namespace each: ``provider:<name>`` and ``endpoint:<host>`` (a
``semantic_call`` model candidate occupies both; a local endpoint is a local process),
``aux_task:<task>`` (``auxiliary.<task>.max_concurrency``), and anything a candidate declares under
``resources:`` (for example ``sandbox:<name>``).

**Gates** are one registry of semaphores keyed by ``(profile home, resource)``, rebuilt when the
ceiling changes; async gates are also keyed per event loop. The auxiliary client's per-task limit
(#23324) takes its gates from here, with its old semantics: unbounded wait, no backoff.
``semantic_call`` takes gates through :meth:`AdmissionController.hold`, which adds:

* **backoff**: :meth:`signal_exhausted` blocks a resource for ``max(Retry-After, base·2^(k−1))``
  (capped) after the k-th consecutive exhaustion signal; :meth:`signal_ok` resets k. The Nous
  guard's shared reset time is read as a signal for ``provider:nous``;
* **bounded queueing**: a caller waits for a slot or a backoff up to a deadline, then gets
  :class:`AdmissionTimeout` naming the resource and why. A backoff longer than the remaining wait
  fails at once instead of sleeping uselessly.

State is in-process and nothing is persisted: a killed process leaks no slot and no backoff (see
"Restartability" in ``agent/semantic_call_cascade.py``). The Nous guard is the one cross-process
signal. Waiters are not served FIFO; the wait bound is what prevents starvation.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Failover reasons that mean "this provider is exhausted right now": back off, don't hammer it.
_EXHAUSTION_REASONS = frozenset({"rate_limit", "upstream_rate_limit", "overloaded"})


def _nous_remaining() -> Optional[float]:
    from agent.nous_rate_guard import nous_rate_limit_remaining
    return nous_rate_limit_remaining()


# resource -> seconds until an existing cross-session guard says it may be used again (None = clear).
_EXTERNAL_SIGNALS: Dict[str, Callable[[], Optional[float]]] = {"provider:nous": _nous_remaining}


class AdmissionTimeout(RuntimeError):
    """Admission could not be granted within the wait bound. Nothing was sent."""

    def __init__(self, resource: str, reason: str, waited_s: float):
        self.resource, self.reason, self.waited_s = resource, reason, waited_s
        super().__init__(f"admission: {resource} {reason}; gave up after {waited_s:.1f}s of the allowed wait. "
                         f"Nothing was sent. Raise agent.admission.max_wait_s or the ceiling under "
                         f"agent.admission.ceilings if this is expected load.")


def _home() -> str:
    from hermes_constants import hermes_home_key
    return hermes_home_key()


class AdmissionController:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._gates: Dict[Tuple[str, str], Tuple[int, threading.BoundedSemaphore]] = {}
        self._async_gates: Dict[Tuple[str, str, int], Tuple[int, Any]] = {}
        self._blocked_until: Dict[Tuple[str, str], float] = {}
        self._strikes: Dict[Tuple[str, str], int] = {}

    # ---- gates (the one registry of per-resource ceilings) ----

    def gate(self, resource: str, limit: int) -> threading.BoundedSemaphore:
        """The sync semaphore for ``resource`` at ``limit``; rebuilt when the limit changes."""
        return self._cached(self._gates, (_home(), resource), limit, threading.BoundedSemaphore)

    def async_gate(self, resource: str, limit: int):
        """The asyncio semaphore for ``resource`` on the running loop, or None outside a loop."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return self._cached(self._async_gates, (_home(), resource, id(loop)), limit, asyncio.Semaphore)

    def _cached(self, store: dict, key: Any, limit: int, factory: Callable[[int], Any]) -> Any:
        with self._lock:
            entry = store.get(key)
            if entry is None or entry[0] != limit:
                store[key] = entry = (limit, factory(limit))
            return entry[1]

    # ---- backoff ----

    def backoff_remaining(self, resource: str) -> float:
        own = self._blocked_until.get((_home(), resource), 0.0) - self._clock()
        external = _EXTERNAL_SIGNALS.get(resource)
        other = (external() or 0.0) if external is not None else 0.0
        return max(own, other, 0.0)

    def signal_exhausted(self, resources: Sequence[str], *, retry_after_s: Optional[float],
                         settings: Any) -> float:
        """A provider said "not now" for these resources; returns the backoff applied (seconds)."""
        applied = 0.0
        home, now = _home(), self._clock()
        with self._lock:
            for resource in resources:
                key = (home, resource)
                strikes = self._strikes.get(key, 0) + 1
                self._strikes[key] = strikes
                delay = min(settings.backoff_base_s * 2 ** (strikes - 1), settings.backoff_max_s)
                delay = max(delay, retry_after_s or 0.0)
                self._blocked_until[key] = max(self._blocked_until.get(key, 0.0), now + delay)
                applied = max(applied, delay)
        logger.info("admission: backing off %s for %.1fs after an exhaustion signal", ", ".join(resources), applied)
        return applied

    def signal_ok(self, resources: Sequence[str]) -> None:
        home = _home()
        with self._lock:
            for resource in resources:
                self._strikes.pop((home, resource), None)

    # ---- admission ----

    @contextlib.contextmanager
    def hold(self, resources: Sequence[str], *, settings: Any, deadline: float) -> Iterator[None]:
        """Occupy one slot of every resource with a configured ceiling, waiting (up to ``deadline``,
        a ``clock()`` value) for backoffs to end and slots to free; all-or-nothing."""
        started = self._clock()
        ordered = sorted(set(resources))  # one global order: two holders can never deadlock
        for resource in ordered:
            wait = self.backoff_remaining(resource)
            if wait <= 0:
                continue
            if self._clock() + wait > deadline:
                raise AdmissionTimeout(resource, f"is backing off for {wait:.1f}s after an exhaustion signal",
                                       self._clock() - started)
            self._sleep(wait)
        held = []
        try:
            for resource in ordered:
                limit = settings.ceiling(resource)
                if limit is None:
                    continue
                sem = self.gate(resource, limit)
                if not sem.acquire(timeout=max(0.0, deadline - self._clock())):
                    raise AdmissionTimeout(resource, f"is at its ceiling of {limit} concurrent calls",
                                           self._clock() - started)
                held.append(sem)
            yield
        finally:
            for sem in reversed(held):
                sem.release()

    def reset(self) -> None:
        """Forget every gate and backoff (tests)."""
        with self._lock:
            self._gates.clear()
            self._async_gates.clear()
            self._blocked_until.clear()
            self._strikes.clear()


def exhaustion_signal(error: BaseException, *, provider: str = "", model: str = "") -> Optional[Tuple[bool, Optional[float]]]:
    """``(True, retry_after_s)`` when ``error`` says the provider is exhausted, else None.
    Classified by the same classifier the turn loop's failover uses."""
    if not isinstance(error, Exception):
        return None
    from agent.error_classifier import classify_api_error
    from agent.retry_utils import parse_retry_after_seconds
    reason = classify_api_error(error, provider=provider, model=model).reason.value
    if reason not in _EXHAUSTION_REASONS:
        return None
    headers = getattr(getattr(error, "response", None), "headers", None)
    return True, parse_retry_after_seconds(headers) if headers is not None else None


ADMISSION = AdmissionController()
