"""Writing a file so that a reader finds the old one or the new one, never part of one.

A run's record, a sweep's ledger, a rung, exported weights and their index
are all read by other processes or a later run, some while the writer is
still running. `replacing` hands out a temporary path beside the target,
unique to this write, and renames it over the target once the block ends with
its bytes flushed to disk, so a crash at any point leaves either file whole.
`write_atomically` is that for data in memory, and `staged` is it for a
directory, which a cache publishes once. A location with a scheme (`gs://`,
`s3://`) is written in place: its store makes an object visible whole or not
at all, and has no rename to stage one with. `stat_identity` is what such a
cache keys a source file by.
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


def stat_identity(path: str | os.PathLike) -> tuple[str, int, int]:
    """A file's resolved path, size and modification time in ns, which a
    conversion cache keys it by in place of a hash of its bytes.

    Hashing the contents would read the whole file on every cache hit, the
    cost the cache exists to avoid. A Hub file resolves to its
    content-addressed blob, so for a published file the key is a content
    key. A local file relies on its mtime: rewriting it in place at the
    same size within one mtime tick, or `cp -p` of another file of the
    same size over it, reuses the stale conversion; deleting the cache's
    directory under `dew.cache.dew_cache_dir()` clears it."""
    status = os.stat(path)
    return str(Path(path).resolve()), status.st_size, status.st_mtime_ns


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
        published(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def published(built: Path, directory: Path) -> bool:
    """Publish the directory `built`, written on `directory`'s filesystem, as `directory` by one
    rename, whether this call published it. A `directory` another writer published first is kept,
    whole, and `built` removed: every writer of it publishes by the one rename (`staged`)."""
    try:
        os.replace(built, directory)
    except OSError:
        if not directory.is_dir():
            raise
        shutil.rmtree(built, ignore_errors=True)
        return False
    return True


def _beside(target: Path, suffix: str) -> Path:
    """A name in `target`'s directory that no other write uses, named apart from `target`'s own, so a
    target whose name is as long as the filesystem allows has a temporary too."""
    return target.with_name(f".dew-{secrets.token_hex(8)}{suffix}")
