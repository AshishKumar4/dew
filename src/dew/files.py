"""Writing a file so that a reader finds the old one or the new one, never part of one.

A run's record, a sweep's ledger, a rung, exported weights and their index
are all read by other processes or a later run, some while the writer is
still running. `replacing` hands out a temporary path beside the target,
unique to this write, and renames it over the target once the block ends with
its bytes flushed to disk, so a crash at any point leaves either file whole.
`write_atomically` is that for data in memory, and `staged` is it for a
directory, which a cache publishes once. A location with a scheme (`gs://`,
`s3://`) is written in place: its store makes an object visible whole or not
at all, and has no rename to stage one with.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import stat
from collections.abc import Iterator
from pathlib import Path

from etils import epath


@contextlib.contextmanager
def replacing(path: str | os.PathLike) -> Iterator[epath.Path]:
    """A path to write the new contents of `path` at, published over it when
    the block ends and removed if the block raises.

    The temporary sits in `path`'s directory, so the rename never crosses a
    filesystem, and is created for this write alone, so concurrent writers of
    one target (threads, processes) never share it. The file published has
    the mode a plain write gives: the target's own, or a new file's under the
    umask. A location with a scheme yields `path` itself.
    """
    if "://" in str(path):
        yield epath.Path(path)
        return
    target = Path(path)
    temporary = epath.Path(_beside(target, ".tmp"))
    os.close(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
    try:
        yield temporary
        with open(temporary, "rb+") as written:
            os.fsync(written.fileno())
        with contextlib.suppress(FileNotFoundError):
            os.chmod(temporary, stat.S_IMODE(os.stat(target).st_mode))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_atomically(path: str | os.PathLike, contents: str | bytes) -> None:
    """Write `contents`, text or bytes, as the whole of `path` (`replacing`)."""
    with replacing(path) as temporary:
        if isinstance(contents, str):
            temporary.write_text(contents)
        else:
            temporary.write_bytes(contents)


@contextlib.contextmanager
def staged(directory: Path) -> Iterator[Path]:
    """A directory to write the contents of `directory` into, renamed to it
    when the block ends and removed if the block raises.

    A `directory` another writer published first is kept, whole, and this
    write dropped: every writer publishes by the one rename. The directory
    has the mode `mkdir` gives under the umask.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = _beside(directory, "")
    staging.mkdir()
    try:
        yield staging
        try:
            os.replace(staging, directory)
        except OSError:
            if not directory.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _beside(target: Path, suffix: str) -> Path:
    """A name in `target`'s directory that no other write uses."""
    return target.with_name(f".{target.name}.{secrets.token_hex(8)}{suffix}")
