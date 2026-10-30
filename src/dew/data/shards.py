"""Grain's sliceable root over physical shards, shared by every streamed source."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Iterator, Mapping

import grain.experimental as experimental
import grain.python as grain

from dew._records import boolean

from .dataset import LONG_RUN, DataPartition, Loading, logged_run, restore_parent
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
    examples: int | None = None,
) -> grain.IterDataset:
    """The shards of `manifest` read `epochs` times over (endlessly where None), this share's part
    of them interleaved `cycle_length` at a time; an endless read bounded to the stream's first `examples`
    rows reads this share's places among them alone, the last in part (`bounded_places`), so its
    interleave reads no later place's rows in their stead. An endless read of a store whose shards all
    state they hold no row is refused, as it would read nothing forever; any other run of empty shards is
    read through, logged at each doubling from `LONG_RUN` (`logged_run`)."""
    if cycle_length < 1:
        raise ValueError("interleave cycle_length must be positive")
    described = name or (manifest.shards[0].uri if manifest.shards else "a stream")
    if epochs is None and all(shard.rows == 0 for shard in manifest.shards):
        raise ValueError(f"{described} holds no rows, so reading it endlessly reads nothing forever")
    if not manifest.shards:
        return grain.MapDataset.source(()).to_iter_dataset()
    places = None if examples is None else bounded_places(
        manifest, partition=partition, seed=seed, loading=loading, epochs=epochs, examples=examples)
    if places is not None:
        reader = _Taken(manifest.reader)
        return _ShardPass(grain.MapDataset.source(places), reader, cycle_length, loading, described)
    root = _dealt(_shards(manifest, partition=partition, loading=loading), seed=seed, epochs=epochs)
    root = root[partition.cut()]
    return _ShardPass(root, manifest.reader, cycle_length, loading, described)


def bounded(manifest: ShardManifest, *, partition: DataPartition, seed: int | None, loading: Loading,
            epochs: int | None, examples: int) -> int | None:
    """This share's rows of the stream's first `examples`, its shards dealt to the shares place by place as
    `streamed` deals them (every count-th from the share's index): of an endless read, each pass its own
    order of places, the rows of each of its places the bound reaches (`_reached`); of one pass that a
    later repeat replays, each share its own places again, its rows in every whole pass the bound spans
    and of its places the last pass reaches. So each share reads its own shards' part of the bound and
    none repeats one while another's go unread. None where a shard states no rows."""
    shards = _shards(manifest, partition=partition, loading=loading)
    if any(shard.rows is None for shard in shards):
        return None
    if not any(shard.rows for shard in shards):
        return 0
    root = _dealt(shards, seed=seed, epochs=epochs)
    if epochs is None:
        return sum(taken for place, _, taken in _reached(root, examples) if place % partition.count
                   == partition.index)
    mine = sum(root[place].rows for place in range(partition.index, len(root), partition.count))
    passes, rest = divmod(examples, sum(root[place].rows for place in range(len(root))))
    return passes * mine + sum(taken for place, _, taken in _reached(root, rest)
                               if place % partition.count == partition.index)


def bounded_places(manifest: ShardManifest, *, partition: DataPartition, seed: int | None, loading: Loading,
                   epochs: int | None, examples: int) -> list[tuple[Shard, int]] | None:
    """This share's places among an endless stream's first `examples` rows, each with the rows it reads of
    it (`_reached`), in order: every row the bound deals this share and no other. None where the read is
    no endless one, or a shard states no rows."""
    shards = _shards(manifest, partition=partition, loading=loading)
    if epochs is not None or any(shard.rows is None for shard in shards):
        return None
    if not any(shard.rows for shard in shards):
        return []
    root = _dealt(shards, seed=seed, epochs=epochs)
    return [(shard, taken) for place, shard, taken in _reached(root, examples)
            if place % partition.count == partition.index and taken]


def _reached(root: grain.MapDataset, examples: int) -> Iterator[tuple[int, Shard, int]]:
    """Each place of `root` the first `examples` of its rows reach, read in order, its shard and the rows
    of it they hold. Some shard of `root` holds a row."""
    place = 0
    while examples > 0:
        shard = root[place]
        assert shard.rows is not None, "every shard a bound walks states its rows"
        yield place, shard, min(shard.rows, examples)
        examples -= shard.rows
        place += 1


def _shards(manifest: ShardManifest, *, partition: DataPartition, loading: Loading) -> tuple[Shard, ...]:
    """The shards a stream deals its shares and their workers: the manifest's, subdivided where too few
    fill them; refused where even so too few do."""
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
    return tuple(shards)


def _dealt(shards: tuple[Shard, ...], *, seed: int | None, epochs: int | None) -> grain.MapDataset:
    """The `shards` a stream reads (`_shards`), place by place over every pass, before the shares are cut
    from them: shuffled where `seed` is given (each pass its own order) and repeated `epochs` times
    (endlessly where None)."""
    root = grain.MapDataset.source(shards)
    if seed is not None:
        root = root.shuffle(seed=seed)
    if epochs != 1:
        root = root.repeat(epochs)
    return root


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
class _Taken:
    """A shard reader of `(shard, rows)` places (`bounded_places`): the shard's first `rows` rows."""

    reader: Callable[[Shard], grain.IterDataset]

    def __call__(self, place: tuple[Shard, int]) -> grain.IterDataset:
        shard, rows = place
        read = self.reader(shard)
        return read if rows == shard.rows else experimental.LimitIterDataset(read, rows)


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
        restore_parent(self.input_parent, state, "parent", "shard position")
        self.yielded = boolean(state["yielded"], "shard yielded")

    def close(self) -> None:
        self.input_parent.close()
