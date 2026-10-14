"""Grain's sliceable root over physical shards, shared by every streamed source."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import grain.experimental as experimental
import grain.python as grain

from .dataset import DataPartition, Loading
from .sources.base import Shard, ShardManifest


def streamed(
    manifest: ShardManifest,
    *,
    partition: DataPartition,
    seed: int | None = None,
    cycle_length: int = 16,
    loading: Loading,
    epochs: int | None = 1,
    empty_shard_budget: int = 4096,
    name: str | None = None,
) -> grain.IterDataset:
    if cycle_length < 1:
        raise ValueError("interleave cycle_length must be positive")
    if not manifest.shards:
        return grain.MapDataset.source(()).to_iter_dataset()
    if empty_shard_budget < 1:
        raise ValueError("empty_shard_budget must be positive")
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
    root = root[partition.index :: partition.count]
    return _ShardPass(
        root, manifest.reader, cycle_length, loading, empty_shard_budget, name or shards[0].uri
    )


@dataclasses.dataclass(frozen=True)
class _EmptyShard:
    pass


class _ShardPass(grain.IterDataset):
    def __init__(
        self,
        root: grain.MapDataset,
        reader: Callable[[Shard], grain.IterDataset],
        cycle: int,
        loading: Loading,
        budget: int,
        name: str,
    ):
        super().__init__(root)
        self.root, self.reader = root, reader
        self.cycle, self.loading, self.budget, self.name = cycle, loading, budget, name

    def set_slice(self, selection: slice, sequential_slice: bool = False) -> None:
        self.root = self.root[selection]

    def __iter__(self) -> grain.DatasetIterator:
        parent = experimental.InterleaveIterDataset(
            self.root.map(_GuardedReader(self.reader)),
            cycle_length=self.cycle,
            num_make_iter_threads=max(self.loading.threads, 1),
            make_iter_buffer_size=self.loading.read_buffer,
        )
        return _BudgetIterator(parent.__iter__(), self.name, self.budget)


@dataclasses.dataclass(frozen=True)
class _GuardedReader:
    reader: Callable[[Shard], grain.IterDataset]

    def __call__(self, shard: Shard) -> grain.IterDataset:
        return _GuardedShard(self.reader(shard))


class _GuardedShard(grain.IterDataset):
    def __init__(self, parent: grain.IterDataset):
        super().__init__(parent)
        self.input_parent = parent

    def __iter__(self) -> grain.DatasetIterator:
        return _ShardIterator(self.input_parent.__iter__())


class _ShardIterator(grain.DatasetIterator):
    def __init__(self, parent: grain.DatasetIterator):
        super().__init__(parent)
        self.input_parent = parent
        self.yielded = False
        self.reported = False

    def __next__(self):
        try:
            row = next(self.input_parent)
        except StopIteration:
            if not self.yielded and not self.reported:
                self.reported = True
                return _EmptyShard()
            raise
        self.yielded = True
        return row

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self.input_parent.get_state(), "yielded": self.yielded, "reported": self.reported}

    def set_state(self, state: Mapping[str, object]) -> None:
        parent, yielded, reported = state["parent"], state["yielded"], state["reported"]
        if not isinstance(parent, Mapping) or not isinstance(yielded, bool) or not isinstance(reported, bool):
            raise ValueError("invalid physical shard reader state")
        self.input_parent.set_state(parent)
        self.yielded, self.reported = yielded, reported

    def close(self) -> None:
        self.input_parent.close()


class _BudgetIterator(grain.DatasetIterator):
    def __init__(self, parent: grain.DatasetIterator, name: str, limit: int):
        super().__init__(parent)
        self.input_parent, self.name, self.limit = parent, name, limit
        self.empty = 0

    def __next__(self):
        while isinstance(row := next(self.input_parent), _EmptyShard):
            self.empty += 1
            if self.empty >= self.limit:
                raise ValueError(
                    f"source {self.name} exhausted its search budget of {self.limit} empty shards"
                )
        self.empty = 0
        return row

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self.input_parent.get_state(), "empty_shards": self.empty}

    def set_state(self, state: Mapping[str, object]) -> None:
        parent, empty = state["parent"], state["empty_shards"]
        if not isinstance(parent, Mapping) or not isinstance(empty, int) or empty < 0:
            raise ValueError("invalid empty-shard search state")
        self.input_parent.set_state(parent)
        self.empty = empty

    def close(self) -> None:
        self.input_parent.close()
