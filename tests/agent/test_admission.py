"""Behaviour contracts for the admission controller (agent/admission.py): ceilings queue instead of
failing, waits are bounded with a clear error, exhaustion signals back a resource off, and the
auxiliary per-task limit it absorbed keeps its semantics."""

import threading
import time

import httpx
import openai
import pytest

from agent.admission import AdmissionController, AdmissionTimeout, exhaustion_signal
from agent.capability_routing_config import AdmissionSettings, parse_admission


def _settings(**ceilings):
    return AdmissionSettings(max_wait_s=10, backoff_base_s=2, backoff_max_s=60,
                             ceilings=tuple(sorted(ceilings.items())))


def test_work_past_the_ceiling_queues_until_a_slot_frees():
    admission = AdmissionController()
    settings = _settings(**{"provider:p": 1})
    first_in, release_first, second_in = threading.Event(), threading.Event(), threading.Event()

    def first():
        with admission.hold(["provider:p"], settings=settings, deadline=time.monotonic() + 10):
            first_in.set()
            release_first.wait(10)

    def second():
        first_in.wait(10)
        with admission.hold(["provider:p"], settings=settings, deadline=time.monotonic() + 10):
            second_in.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    assert first_in.wait(10)
    assert not second_in.wait(0.3)  # queued behind the ceiling, not failed
    release_first.set()
    assert second_in.wait(10)
    for t in threads:
        t.join(10)


def test_a_bounded_wait_fails_clearly_and_sends_nothing():
    admission = AdmissionController()
    settings = _settings(**{"provider:p": 1})
    with admission.hold(["provider:p"], settings=settings, deadline=time.monotonic() + 1):
        with pytest.raises(AdmissionTimeout, match="ceiling of 1") as err:
            with admission.hold(["provider:p"], settings=settings, deadline=time.monotonic() + 0.1):
                pytest.fail("admitted past the ceiling")
    assert "Nothing was sent" in str(err.value)


def test_resources_without_a_ceiling_are_not_limited():
    admission = AdmissionController()
    with admission.hold(["provider:p"], settings=_settings(), deadline=time.monotonic()):
        with admission.hold(["provider:p"], settings=_settings(), deadline=time.monotonic()):
            pass


class _FakeTime:
    def __init__(self):
        self.now, self.slept = 1000.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_a_rate_limit_signal_backs_off_exponentially_and_success_resets_it():
    fake = _FakeTime()
    admission = AdmissionController(clock=fake.clock, sleep=fake.sleep)
    settings = _settings()

    def admit():
        with admission.hold(["provider:p"], settings=settings, deadline=fake.now + 100):
            pass

    assert admission.signal_exhausted(["provider:p"], retry_after_s=None, settings=settings) == 2
    admit()
    assert admission.signal_exhausted(["provider:p"], retry_after_s=None, settings=settings) == 4
    admit()
    assert admission.signal_exhausted(["provider:p"], retry_after_s=30, settings=settings) == 30  # Retry-After wins
    admit()
    assert fake.slept == [2, 4, 30]
    admission.signal_ok(["provider:p"])
    assert admission.signal_exhausted(["provider:p"], retry_after_s=None, settings=settings) == 2
    admission.signal_ok(["provider:p"])
    admission.signal_exhausted(["provider:other"], retry_after_s=None, settings=settings)
    assert admission.backoff_remaining("provider:p") == 2 and admission.backoff_remaining("provider:q") == 0


def test_a_backoff_longer_than_the_wait_bound_fails_at_once():
    fake = _FakeTime()
    admission = AdmissionController(clock=fake.clock, sleep=fake.sleep)
    admission.signal_exhausted(["provider:p"], retry_after_s=120, settings=_settings())
    with pytest.raises(AdmissionTimeout, match="backing off"):
        with admission.hold(["provider:p"], settings=_settings(), deadline=fake.now + 5):
            pass
    assert fake.slept == []


def test_the_nous_guard_is_read_as_a_backoff_signal(monkeypatch):
    from agent import admission as admission_mod
    monkeypatch.setitem(admission_mod._EXTERNAL_SIGNALS, "provider:nous", lambda: 42.0)
    assert AdmissionController().backoff_remaining("provider:nous") == pytest.approx(42.0)


def _rate_limit_error(headers):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError("Rate limit reached", response=response, body=None)


def test_a_429_is_an_exhaustion_signal_carrying_retry_after():
    assert exhaustion_signal(_rate_limit_error({"retry-after": "7"})) == (True, 7.0)
    assert exhaustion_signal(ValueError("bad schema")) is None


def test_malformed_ceilings_are_ignored():
    settings = parse_admission({"ceilings": {"provider:a": 2, "provider:b": 0, "provider:c": "x", "provider:d": True},
                                "max_wait_s": -5})
    assert settings.ceilings == (("provider:a", 2),) and settings.max_wait_s == 0


def test_the_auxiliary_task_limit_is_an_admission_gate(monkeypatch):
    """The folded-in limit: auxiliary.<task>.max_concurrency is the gate for aux_task:<task> in the one
    registry, and without the key there is no gate at all (the default, unchanged)."""
    from agent import auxiliary_client as ac
    from agent.admission import ADMISSION
    ac._reset_aux_semaphores()
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {"max_concurrency": 2})
    assert ac._acquire_sync_aux_semaphore("compression") is ADMISSION.gate("aux_task:compression", 2)
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {})
    assert ac._acquire_sync_aux_semaphore("compression") is None
    ac._reset_aux_semaphores()
