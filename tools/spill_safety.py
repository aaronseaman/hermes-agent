"""Symlink-safe creation helpers for spill/cache files under ``~/.hermes``, where a plain
``open(path, "w")`` would follow a pre-planted symlink onto ``~/.bashrc`` etc. New files use
``O_CREAT | O_EXCL`` (fails on ANY existing path, even a dangling link); overwrites ``lstat`` +
``unlink`` first (removes the link, never its target) then create exclusively, so the pair
can't be raced. ``private=True`` (default) = ``0o700`` dirs / ``0o600`` files for spills that
may hold pre-redaction secrets; ``private=False`` keeps umask perms for cache dirs bind-mounted
into remote backends (``credential_files._CACHE_DIRS``). Disk failures raise ``OSError``.

``store_content_addressed`` names a spill by the sha256 of its bytes, so identical outputs are
stored once and a handle can be checked against the file it names. The name is predictable from
the content, so a reuse is only trusted after re-hashing a regular (non-symlink) file, and a new
file lands via an exclusive temp file + ``os.replace`` (never partial, never through a link)."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from pathlib import Path
from typing import IO

__all__ = ["content_address", "ensure_spill_dir", "open_exclusive", "store_content_addressed",
           "write_text_exclusive"]

# O_NOFOLLOW is POSIX-only; on Windows O_EXCL alone already refuses every pre-existing path.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def ensure_spill_dir(path: Path, *, private: bool = True) -> Path:
    """Create ``path`` (and parents) as a directory, refusing symlinks. ``private=True``
    creates the leaf ``0o700`` and tightens an existing leaf. Raises ``OSError`` if the leaf
    is not a real directory."""
    path = Path(path)
    path.mkdir(mode=0o700 if private else 0o777, parents=True, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"spill dir is not a directory (symlink?): {path}")
    if private and stat.S_IMODE(st.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return path


def open_exclusive(path: Path, *, private: bool = True, overwrite: bool = False,
                   encoding: str = "utf-8", errors: str = "strict") -> IO[str]:
    """Open ``path`` for writing via exclusive create; never follows a link. ``overwrite=True``
    first unlinks an existing path (``lstat``-checked, so only the link itself is removed and
    directories are refused), then creates exclusively — the overwrite path cannot be
    redirected through a symlink either."""
    path = Path(path)
    if overwrite:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(st.st_mode):
                raise OSError(f"refusing to overwrite a directory: {path}")
            os.unlink(path)
    mode = 0o600 if private else 0o666  # non-private honors umask
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, mode)
    try:
        return os.fdopen(fd, "w", encoding=encoding, errors=errors)
    except Exception:
        os.close(fd)
        raise


def write_text_exclusive(path: Path, text: str, *, private: bool = True, overwrite: bool = False,
                         encoding: str = "utf-8", errors: str = "strict") -> None:
    """``Path.write_text`` equivalent that refuses to follow symlinks."""
    with open_exclusive(path, private=private, overwrite=overwrite, encoding=encoding,
                        errors=errors) as fh:
        fh.write(text)


def content_address(data: bytes) -> str:
    """The handle of a content-addressed spill: sha256 hex of the stored bytes."""
    return hashlib.sha256(data).hexdigest()


def _reuse_verified(path: Path, digest: str) -> bool:
    """True when ``path`` is a regular file whose bytes hash to ``digest``; its mtime is then
    refreshed so age-based retention counts from this reference, not the first write. Opened
    ``O_NOFOLLOW`` and checked via ``fstat``, so a symlink planted at the predictable name is
    never trusted; a tampered or truncated file (e.g. edited through a sandbox bind mount) fails
    the hash and gets rewritten. ``O_NONBLOCK`` keeps a FIFO planted at the name from hanging
    the open."""
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return False
        h = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            return False
        os.utime(fd if os.utime in os.supports_fd else path)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def store_content_addressed(directory: Path, data: bytes, *, private: bool = True,
                            suffix: str = ".txt") -> tuple[Path, bool]:
    """Store ``data`` as ``<sha256><suffix>`` in ``directory``; returns ``(path, reused)``.

    ``reused`` is True when a verified identical file was already there. A write is size-checked
    before it is published, so a short write (quota, ENOSPC) raises ``OSError`` instead of leaving
    a file whose name promises bytes it does not hold."""
    directory = ensure_spill_dir(Path(directory), private=private)
    digest = content_address(data)
    path = directory / f"{digest}{suffix}"
    if _reuse_verified(path, digest):
        return path, True
    tmp = directory / f".{digest}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600 if private else 0o666)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if os.stat(tmp).st_size != len(data):
            raise OSError(f"spill write is not lossless: {path}")
        os.replace(tmp, path)  # replaces a planted link itself, never its target
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path, False
