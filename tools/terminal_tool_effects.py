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
  commands may be joined only by ``|``, ``&&``, ``||`` or ``;``. Redirects, command substitution,
  subshells, backgrounding, escapes, comments, line breaks, env assignments, unknown binaries and
  background calls all make the call not read-only.

Terminal calls are never ``parallel_safe``: every call shares the session's cwd and env.

The allowlist is deliberately small. A false negative costs nothing (the call is treated as it
was before this classifier existed); a false positive tells the loop guardrail a mutating command
is a repeatable read.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Mapping, Optional

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


# Separators that join simple commands without changing what each one does -> chars consumed.
_SEPARATORS = {"|": 1, "||": 2, "&&": 2, ";": 1}
# Unquoted characters that make a command not provably read-only: redirects, subshells, command
# substitution, backgrounding, escapes, line breaks. ``&&`` is matched as a separator first.
_REJECT_UNQUOTED = frozenset("<>()&`\\\n\r")
# Expansions that can run a command or assign a variable (``${X:=v}``, ``$[X=1]``); ``$NAME`` is fine.
_EXPANSIONS = ("$(", "${", "$[")


def _split_simple_commands(command: str) -> Optional[List[List[str]]]:
    """Split *command* into the word lists of its simple commands, or ``None`` when it uses any
    shell construct beyond plain words, quotes and the ``_SEPARATORS``.

    Neither ``shlex`` nor the existing splitters answer this question: they tokenize a command
    that is already assumed to be ordinary, so a redirect or a ``$(...)`` comes back as just
    another word. Here the presence of such a construct is the whole verdict, which needs the
    quoting state at the character that carries it.
    """
    segments: List[List[str]] = [[]]
    word: Optional[str] = None  # None = between words ('' is a started, empty quoted word)
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = ""
            else:
                word += ch
        elif quote == '"':
            if ch in '`\\' or command.startswith(_EXPANSIONS, i):
                return None
            if ch == '"':
                quote = ""
            else:
                word += ch
        elif ch in "'\"":
            quote, word = ch, word or ""
        elif ch in " \t":
            if word is not None:
                segments[-1].append(word)
                word = None
        elif (op := command[i:i + 2] if command[i:i + 2] in _SEPARATORS else ch) in _SEPARATORS:
            if word is not None:
                segments[-1].append(word)
                word = None
            segments.append([])
            i += _SEPARATORS[op]
            continue
        elif ch in _REJECT_UNQUOTED or command.startswith(_EXPANSIONS, i) or (ch == "#" and word is None):
            return None
        else:
            word = (word or "") + ch
        i += 1
    if quote:
        return None
    if word is not None:
        segments[-1].append(word)
    # An empty segment means a leading, trailing or doubled separator: not a plain chain.
    return segments if all(segments) else None


def _any_args(args: List[str]) -> bool:
    return True


def _no_args_starting(*prefixes: str) -> Callable[[List[str]], bool]:
    return lambda args: not any(a.startswith(prefixes) for a in args)


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
    "rg": _no_args_starting("--pre"),  # --pre runs a preprocessor program
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
