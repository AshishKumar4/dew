"""Grain's sliceable root over physical shards, shared by every streamed source."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Mapping

import grain.experimental as experimental
import grain.python as grain

from dew._records import boolean, record

from .dataset import LONG_RUN, DataPartition, Loading, logged_run
from .sources.base import Shard, ShardManifest

_log = logging.getLogger(__name__)

def streamed(
    manifest: ShardManifest,
    *,
    partition: DataPartition,
    seed: int | None = None,
    cycle_length: int = 16,
    loading: Loading,
    epochs: int | None = 1,
    name: str | None = None,
) -> grain.IterDataset:
    """The shards of `manifest` read `epochs` times over (endlessly where None), this share's part
    of them interleaved `cycle_length` at a time. An endless read of a store whose shards all state
    they hold no row is refused, as it would read nothing forever; any other run of empty shards is
    read through, logged at each doubling from `LONG_RUN` (`logged_run`)."""
    if cycle_length < 1:
        raise ValueError("interleave cycle_length must be positive")
    described = name or (manifest.shards[0].uri if manifest.shards else "a stream")
    if epochs is None and all(shard.rows == 0 for shard in manifest.shards):
        raise ValueError(f"{described} holds no rows, so reading it endlessly reads nothing forever")
    if not manifest.shards:
        return grain.MapDataset.source(()).to_iter_dataset()
    workers = loading.workers if isinstance(loading.workers, int) else 1
    required = partition.count * max(workers, 1)
    shards = manifest.shards
    if len(shards) < required and manifest.subdivide is not None:
        shards = manifest.subdivide(required)
    if len(shards) < required:
        raise ValueError(
            f"stream has {len(shards)} physical shards for {required} readers × workers; "
            "produce more shards or reduce readers/workers"
        )
    root = grain.MapDataset.source(shards)
    if seed is not None:
        root = root.shuffle(seed=seed)
    if epochs != 1:
        root = root.repeat(epochs)
    root = root[partition.cut()]
    return _ShardPass(root, manifest.reader, cycle_length, loading, described)


class _ShardPass(grain.IterDataset):
    def __init__(self, root: grain.MapDataset, reader: Callable[[Shard], grain.IterDataset], cycle: int,
                 loading: Loading, name: str):
        super().__init__(root)
        self.root, self.reader = root, reader
        self.cycle, self.loading, self.name = cycle, loading, name

    def set_slice(self, selection: slice, sequential_slice: bool = False) -> None:
        self.root = self.root[selection]

    def __iter__(self) -> grain.DatasetIterator:
        empty = _EmptyRun(self.name)
        return experimental.InterleaveIterDataset(
            self.root.map(_Reader(self.reader, empty)),
            cycle_length=self.cycle,
            num_make_iter_threads=max(self.loading.threads, 1),
            make_iter_buffer_size=self.loading.read_buffer,
        ).__iter__()


class _EmptyRun:
    """How many shards in a row one read found empty, which it logs at each doubling. Grain reads
    each shard on a prefetch thread of its own, so the count is kept under a lock."""

    def __init__(self, name: str):
        self.name, self.count = name, 0
        self.lock = threading.Lock()

    def ended(self, yielded: bool) -> None:
        with self.lock:
            self.count = 0 if yielded else self.count + 1
            count = self.count
        if logged_run(count, LONG_RUN):
            _log.warning("%s: its last %d shards held no row; the read goes on to the next", self.name, count)


@dataclasses.dataclass(frozen=True)
class _Reader:
    """A shard's reader, its rows watched for a run of empty shards (`_EmptyRun`)."""

    reader: Callable[[Shard], grain.IterDataset]
    empty: _EmptyRun

    def __call__(self, shard: Shard) -> grain.IterDataset:
        return _Watched(self.reader(shard), self.empty)


class _Watched(grain.IterDataset):
    """One shard's rows, telling `empty` whether it held any as it ends."""

    def __init__(self, parent: grain.IterDataset, empty: _EmptyRun):
        super().__init__(parent)
        self.input_parent, self.empty = parent, empty

    def __iter__(self) -> grain.DatasetIterator:
        return _WatchedIterator(self.input_parent.__iter__(), self.empty)


class _WatchedIterator(grain.DatasetIterator):
    def __init__(self, parent: grain.DatasetIterator, empty: _EmptyRun):
        super().__init__(parent)
        self.input_parent, self.empty = parent, empty
        self.yielded = False

    def __next__(self):
        try:
            row = next(self.input_parent)
        except StopIteration:
            self.empty.ended(self.yielded)
            raise
        self.yielded = True
        return row

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self.input_parent.get_state(), "yielded": self.yielded}

    def set_state(self, state: Mapping[str, object]) -> None:
        self.input_parent.set_state(record(state["parent"], "shard position"))
        self.yielded = boolean(state["yielded"], "shard yielded")

    def close(self) -> None:
        self.input_parent.close()
