"""Per-call effects of a ``terminal`` call — the one classifier for what a shell command does.

``terminal_call_effects`` is registered as the terminal tool's ``effects_fn``; consumers read it
through ``registry.resolve_effects("terminal", args)`` (the pre-command filesystem checkpoint in
``agent/tool_executor.py`` calls it directly, so a terminal call is classified even before the
registry is consulted):

- ``destructive``: the command matches a known file-modifying shape (``rm``, ``mv``, ``sed -i``,
  a ``>`` overwrite, ...). A best-effort trigger for the pre-command filesystem checkpoint, never
  a safety verdict — ``False`` means "no known destructive shape", not "safe".
- ``idempotent``: the command is read-only by ALLOWLIST. Every simple command in it must be a
  known reader whose arguments carry no writing or program-running option, and the simple
  commands may be joined only by ``|``, ``&&``, ``||``, ``;`` or a newline. Redirects, command
  substitution, parameter expansion, subshells, backgrounding, comments, env assignments, unknown
  binaries and background calls all make the call not read-only. Quoting and escaping are read,
  not rejected: ``find . -name \\*.py`` is a read and ``"r"m x`` is not.

Terminal calls are never ``parallel_safe``: every call shares the session's cwd and env.

The allowlist is deliberately small. A false negative costs nothing (the call is treated as it
was before this classifier existed); a false positive tells the loop guardrail a mutating command
is a repeatable read.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Mapping, Optional

from tools.approval_detection import (
    _READ_TOOL_EXEC_FLAGS, _deobfuscate_shell_word_for_detection, _scan_shell)
from tools.tool_effects import UNKNOWN_EFFECTS, ToolEffects

# Terminal commands that may modify/delete files.
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        cp\s|install\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
# Output redirects that overwrite files (> but not >>)
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def is_destructive_command(cmd: str) -> bool:
    """Heuristic: does this terminal command look like it modifies/deletes files?"""
    return bool(cmd) and bool(_DESTRUCTIVE_PATTERNS.search(cmd) or _REDIRECT_OVERWRITE.search(cmd))


# Unquoted single characters that end one simple command and start the next.
_SEPARATOR_CHARS = frozenset(";|\n")
# Unquoted characters that make a command not provably read-only: redirects, subshells, process
# substitution, and a lone ``&`` (backgrounding). ``&&`` is recognised as a separator first.
_REJECT_UNQUOTED = frozenset("<>()&")
# Expansions that can run a command or assign a variable. ``_scan_shell`` reports ``$(``/backticks
# as substitutions, but not ``${X:=v}`` inside double quotes or ``$[X=1]`` at all; a plain
# ``$NAME`` is fine and stays a word.
_EXPANSIONS = ("$(", "${", "$[")


def _split_simple_commands(command: str) -> Optional[List[List[str]]]:
    """Split *command* into the word lists of its simple commands, or ``None`` when it uses any
    shell construct beyond plain words, quotes, escapes and ``| || && ; <newline>``.

    Built on ``approval_detection._scan_shell``, the repo's one non-expanding shell state machine,
    so quoting and escaping are read the same way the approval detector reads them. The detector
    scans *through* a redirect or a substitution to find a dangerous command inside it; here the
    mere presence of one is the verdict, so those step kinds end the parse instead.
    """
    segments: List[List[str]] = [[]]
    start: Optional[int] = None  # start offset of the word being read, None = between words
    end = 0
    open_quote = False
    skip = -1  # second character of a two-character operator, already consumed

    def close_word() -> None:
        nonlocal start
        if start is not None:
            segments[-1].append(_deobfuscate_shell_word_for_detection(command[start:end]))
            start = None

    for kind, i, j, quote in _scan_shell(command, subst="uq", brace=True, comments=True):
        if i == skip:
            continue
        if kind in {"subst", "comment"}:
            return None  # command substitution / ${...} / a trailing comment
        if kind == "quote":
            open_quote = not open_quote
        elif kind == "char":
            char = command[i]
            # Single quotes make every character literal; anywhere else an expansion can run.
            if quote != "'" and command.startswith(_EXPANSIONS, i):
                return None
            if quote is None:
                if char in "&|" and command.startswith(char * 2, i):  # && / ||
                    skip = i + 1
                    close_word()
                    segments.append([])
                    continue
                if char in _REJECT_UNQUOTED:
                    return None
                if char.isspace() or char in _SEPARATOR_CHARS:
                    close_word()
                    if char in _SEPARATOR_CHARS:
                        segments.append([])
                    continue
        # A quote, an escaped char, or an ordinary word char (quoted or not).
        start, end = (i if start is None else start), j
    close_word()
    # Unbalanced quoting, or a leading, trailing or doubled separator: not a plain chain.
    return None if open_quote else (segments if all(segments) else None)


def _any_args(args: List[str]) -> bool:
    return True


def _no_args_starting(*prefixes: str) -> Callable[[List[str]], bool]:
    return lambda args: not any(a.startswith(prefixes) for a in args)


def _no_exec_flags(binary: str, *also: str) -> Callable[[List[str]], bool]:
    """Reject the options by which this reader runs another program. The option names come from
    the approval detector's ``_READ_TOOL_EXEC_FLAGS``, so both classifiers learn a new one once."""
    return _no_args_starting(*sorted(_READ_TOOL_EXEC_FLAGS.get(binary, frozenset())), *also)


def _tail_args_ok(args: List[str]) -> bool:
    # -f/-F (also inside a short-option cluster) and --follow/--retry never return.
    return not any(
        a.startswith(("--follow", "--retry")) or (a.startswith("-") and not a.startswith("--") and ("f" in a or "F" in a))
        for a in args
    )


# ``git`` subcommands that only read. Anything that can create a ref, a file or a worktree
# (``checkout``, ``branch``, ``tag``, ``config``, ``stash``) stays off the list.
_GIT_READ_SUBCOMMANDS = frozenset({
    "blame", "cat-file", "describe", "diff", "grep", "log", "ls-files", "ls-tree",
    "rev-parse", "shortlog", "show", "status",
})


def _git_args_ok(args: List[str]) -> bool:
    # The subcommand must come first: a leading global option (-C, -c, --git-dir) changes which
    # repo or config applies. ``--output``/``-O`` write a file or hand output to a pager program.
    return bool(args) and args[0] in _GIT_READ_SUBCOMMANDS and _no_args_starting("--output", "-O")(args[1:])


# Read-only binaries -> predicate over their arguments (False = an option that writes or execs).
_READ_ONLY_COMMANDS: dict[str, Callable[[List[str]], bool]] = {
    "ag": _no_exec_flags("ag"),
    "basename": _any_args,
    "cat": _any_args,
    "cut": _any_args,
    "date": _no_args_starting("-s", "--set"),  # -s sets the system clock
    "df": _any_args,
    "diff": _any_args,
    "dirname": _any_args,
    "du": _any_args,
    "echo": _any_args,
    "egrep": _any_args,
    "fgrep": _any_args,
    "file": _any_args,
    "find": _no_args_starting("-delete", "-exec", "-ok", "-fprint", "-fls"),
    "git": _git_args_ok,
    "grep": _any_args,
    "head": _any_args,
    "hostname": _any_args,
    "id": _any_args,
    "jq": _any_args,
    "ls": _any_args,
    "nl": _any_args,
    "printf": _any_args,
    "ps": _any_args,
    "pwd": _any_args,
    "readlink": _any_args,
    "realpath": _any_args,
    "rg": _no_exec_flags("rg"),
    "stat": _any_args,
    "tail": _tail_args_ok,
    "tr": _any_args,
    "uname": _any_args,
    "wc": _any_args,
    "which": _any_args,
    "whoami": _any_args,
}


def is_read_only_command(command: str) -> bool:
    """Allowlist check: True only when every simple command is a known reader (see module doc)."""
    segments = _split_simple_commands(command)
    if not segments:
        return False
    return all(
        (check := _READ_ONLY_COMMANDS.get(words[0])) is not None and check(words[1:])
        for words in segments
    )


def terminal_call_effects(args: Mapping[str, Any]) -> ToolEffects:
    """Effects of one ``terminal`` call; a non-string command gets the conservative default.

    A ``background=True`` call returns a process handle rather than the command's output, so it
    is never the repeatable read that ``idempotent`` claims.
    """
    command = args.get("command")
    if not isinstance(command, str):
        return UNKNOWN_EFFECTS
    destructive = is_destructive_command(command)
    read_only = not destructive and not args.get("background") and is_read_only_command(command)
    return ToolEffects(idempotent=read_only, destructive=destructive)
