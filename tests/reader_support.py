"""What an exact reader owes: a resumed reader reads on where the uninterrupted one does, and a write
that keeps a file's size changes its identity, so a position saved before the write is refused after it."""

from collections.abc import Callable
from pathlib import Path

import pytest


def assert_a_rewrite_moves_the_identity(reader: Callable, path: Path, patch: bytes,
                                        read: Callable[[object], None]) -> None:
    """Open `reader()`, `read` its first batch and save its position; overwrite the middle of `path`
    with `patch`, keeping its size; a fresh reader then identifies other bytes and refuses that
    position."""
    before = reader()
    try:
        identity = before.identity()
        read(next(before))
        position = before.get_state()
    finally:
        before.close()
    size = path.stat().st_size
    with path.open("r+b") as file:
        file.seek(size // 2)
        file.write(patch)
    assert path.stat().st_size == size
    after = reader()
    try:
        assert after.identity() != identity
        with pytest.raises(ValueError):
            after.set_state(position)
    finally:
        after.close()


def assert_resumes_where_it_stopped(opened: Callable, read: Callable, stopped: int,
                                    rest: int) -> tuple[list, bytes]:
    """Read `stopped + rest` steps of a reader `opened()` makes, uninterrupted; read `stopped` of a
    second, save its position, and restore it in a third, whose next `rest` are the uninterrupted
    reader's, each read as `read(reader, count)` reads `count` steps. Returns the uninterrupted
    reads and the saved position."""
    whole = opened()
    try:
        expected = read(whole, stopped + rest)
    finally:
        whole.close()
    before = opened()
    try:
        head = read(before, stopped)
        saved = before.get_state()
    finally:
        before.close()
    resumed = opened()
    try:
        resumed.set_state(saved)
        assert head + read(resumed, rest) == expected
    finally:
        resumed.close()
    return expected, saved
