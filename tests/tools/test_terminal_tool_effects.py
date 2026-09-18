"""Per-call effects of ``terminal``, read through the registry seam consumers use.

Read-only recognition is an allowlist: a command is idempotent only when every simple command in
it is a known reader. Destructive recognition is the pre-command checkpoint trigger.
"""

import pytest

import tools.terminal_tool  # noqa: F401  (registers terminal with its per-call effects resolver)
from tools.registry import registry
from tools.tool_effects import UNKNOWN_EFFECTS


def _effects(command, **extra):
    return registry.resolve_effects("terminal", {"command": command, **extra})


@pytest.mark.parametrize("command, read_only", [
    ("ls -la", True),
    ("git status", True),
    ("git log --oneline -5 | head -3", True),
    ("cat a.txt | grep -n 'x|y;z' | wc -l", True),
    ("rg -n TODO src && git diff --stat", True),
    ("find . -name '*.py'", True),
    ('echo "$HOME"', True),
    # Every segment must be a known reader with no writing/executing option.
    ("ls && make", False),
    ("cat a | tee b", False),
    ("cd src; ls", False),
    ("FOO=1 ls", False),
    ("sudo ls", False),
    ("python -c 'print(1)'", False),
    ("find . -delete", False),
    ("find . -exec cat {} +", False),
    ("tail -f app.log", False),
    ("git checkout main", False),
    ("git -C repo status", False),
    ("git diff --output=out.patch", False),
    ("rg --pre ./decode foo", False),
    # Redirects, substitution, subshells, backgrounding, comments, line breaks, bad quoting.
    ("ls > files.txt", False),
    ("cat a 2>/dev/null", False),
    ("echo $(whoami)", False),
    ('echo "`whoami`"', False),
    ("(ls)", False),
    ("ls &", False),
    ("ls # comment", False),
    ("ls\nrm x", False),
    ("ls 'unterminated", False),
    ("", False),
])
def test_read_only_terminal_commands_are_allowlisted(command, read_only):
    effects = _effects(command)
    assert effects.idempotent is read_only
    assert not effects.parallel_safe  # terminal calls share the session's cwd/env


@pytest.mark.parametrize("command", [
    "cp .env.local .env", "rm -rf build", "echo hi > out.txt", "sed -i s/a/b/ f", "git reset --hard",
])
def test_destructive_commands_trigger_checkpoint_and_are_never_read_only(command):
    effects = _effects(command)
    assert effects.destructive and not effects.idempotent


def test_background_and_malformed_calls_are_conservative():
    assert _effects("ls").idempotent and not _effects("ls", background=True).idempotent
    assert registry.resolve_effects("terminal", {"command": None}) == UNKNOWN_EFFECTS
    assert registry.resolve_effects("terminal", "ls") == UNKNOWN_EFFECTS
