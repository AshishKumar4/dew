"""Writing a file so that a reader finds the old one or the new one, never part of one.

A run's record, a sweep's ledger, a rung, exported weights and their index
are all read by other processes or a later run, some while the writer is
still running. `replacing` hands out a temporary path beside the target,
unique to this write, and renames it over the target once the block ends with
its bytes flushed to disk, so a crash at any point leaves either file whole.
`write_atomically` is that for data in memory. A location with a scheme
(`gs://`, `s3://`) is written in place: its store makes an object visible
whole or not at all, and has no rename to stage one with.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from etils import epath


@contextlib.contextmanager
def replacing(path: str | os.PathLike) -> Iterator[epath.Path]:
    """A path to write the new contents of `path` at, published over it when
    the block ends and removed if the block raises.

    The temporary sits in `path`'s directory, so the rename never crosses a
    filesystem, and is created for this write alone, so concurrent writers of
    one target (threads, processes) never share it. A location with a scheme
    yields `path` itself.
    """
    if "://" in os.fspath(path):
        yield epath.Path(path)
        return
    target = Path(path)
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = epath.Path(name)
    try:
        yield temporary
        with open(temporary, "rb+") as written:
            os.fsync(written.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_atomically(path: str | os.PathLike, contents: str | bytes) -> None:
    """Write `contents`, text as UTF-8, as the whole of `path` (`replacing`)."""
    with replacing(path) as temporary:
        temporary.write_bytes(contents.encode() if isinstance(contents, str) else contents)
