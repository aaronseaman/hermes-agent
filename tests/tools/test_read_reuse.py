"""read_file result reuse: a repeat read is answered from context (the "unchanged" stub,
no execution) only while the read's ``CallFingerprint`` — canonical path/offset/limit plus
the file's stat state — still matches, and never after a call that may have written.

Real dispatch through ``model_tools.handle_function_call`` (real terminal, real files)
against the temp ``HERMES_HOME`` from ``tests/conftest.py``. Every time reuse is claimed,
the content already in context must be byte-identical to what a fresh execution returns.
"""

import itertools
import json
import os
from unittest.mock import MagicMock

import pytest

from model_tools import handle_function_call

_fresh_ids = itertools.count()


def _call(name: str, args: dict, task_id: str) -> dict:
    return json.loads(handle_function_call(name, args, task_id=task_id))


def _fresh(path: str) -> dict:
    """What a fresh execution returns now: a task that has never read anything."""
    return _call("read_file", {"path": path}, f"fresh-{next(_fresh_ids)}")


@pytest.fixture
def workspace(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\n", encoding="utf-8")
    return tmp_path, str(target)


def _assert_reuse_is_exact(task_id: str, path: str, reused: bool, last_full: dict) -> dict:
    """Read once more; check it was (not) reused and that reuse was never stale."""
    result = _call("read_file", {"path": path}, task_id)
    assert bool(result.get("dedup")) is reused, result
    if reused:
        assert last_full == _fresh(path), "reused a result a fresh execution would not return"
        return last_full
    assert result == _fresh(path)
    return result


def test_reuse_ends_when_any_state_dep_changes_even_with_size_and_mtime_restored(workspace):
    _, path = workspace
    full = _call("read_file", {"path": path}, "t-state")
    full = _assert_reuse_is_exact("t-state", path, True, full)

    # Same-size rewrite that puts the old mtime back (``cp -p`` / ``touch -r`` shape).
    before = os.stat(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("omega\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (os.stat(path).st_size, os.stat(path).st_mtime_ns) == (before.st_size, before.st_mtime_ns)

    full = _assert_reuse_is_exact("t-state", path, False, full)
    assert "omega" in full["content"]
    _assert_reuse_is_exact("t-state", path, True, full)


@pytest.mark.parametrize(("tool", "make_args", "reused"), [
    # Provably read-only terminal command: nothing can have changed.
    ("terminal", lambda d, p: {"command": f"ls {d}"}, True),
    # A terminal command that writes (anything, anywhere) ends every reuse of the task.
    ("terminal", lambda d, p: {"command": f"touch {d}/other.txt"}, False),
    # A declared writer only ends reuse of the path it wrote.
    ("write_file", lambda d, p: {"path": str(d / "other.txt"), "content": "x\n"}, True),
    ("write_file", lambda d, p: {"path": p, "content": "beta\n"}, False),
])
def test_reuse_never_survives_a_call_that_may_have_written_the_file(workspace, tool, make_args, reused):
    root, path = workspace
    task = f"t-{tool}-{reused}"
    full = _call("read_file", {"path": path}, task)
    full = _assert_reuse_is_exact(task, path, True, full)

    out = _call(tool, make_args(root, path), task)
    assert not out.get("error"), out

    _assert_reuse_is_exact(task, path, reused, full)


def test_a_sandbox_filesystem_read_is_never_reused(workspace, monkeypatch):
    """Remote backends read inside the sandbox; a host ``stat`` says nothing about that file,
    so it yields no fingerprint and every read executes."""
    import tools.file_tools as ft

    _, path = workspace
    sandbox_ops = MagicMock()
    sandbox_ops.env = MagicMock()  # not a LocalEnvironment
    sandbox_ops.read_file = lambda p, offset=1, limit=500: MagicMock(
        content="1|sandbox copy", to_dict=lambda: {"content": "1|sandbox copy", "total_lines": 1})
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": sandbox_ops)

    for _ in range(2):
        result = _call("read_file", {"path": path}, "t-sandbox")
        assert not result.get("dedup") and result["content"] == "1|sandbox copy"
