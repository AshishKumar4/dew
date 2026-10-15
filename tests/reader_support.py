"""What an exact reader owes the bytes it reads: a write that keeps a file's size changes its identity,
and a position saved before the write is refused after it."""

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
