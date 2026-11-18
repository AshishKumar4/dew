"""A saved position resumed as the read that saved it would have read on: the one check of a position some
other Dew saved (tests/fixtures/positions)."""

import itertools
from collections.abc import Callable

from dew.data.dataset import DataPartition

Batches = list[list[int]]


def resumed_as_read(open_loader: Callable[[], object], steps: int, saved: bytes,
                    ids: Callable[[dict], list[int]], batches: int) -> tuple[Batches, Batches]:
    """The `batches` (each its `ids`) a read of `open_loader`'s loader stopped after `steps` reads next, and
    those a read of it restored to the position `saved` reads next."""
    streams = [open_loader().train(DataPartition()) for _ in range(2)]
    try:
        list(itertools.islice(streams[0], steps))
        streams[1].set_state(saved)
        stopped, restored = ([ids(batch) for batch in itertools.islice(stream, batches)]
                             for stream in streams)
    finally:
        for stream in streams:
            stream.close()
    return stopped, restored
