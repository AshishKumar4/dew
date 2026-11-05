"""Grain's sliceable root over physical shards, shared by every streamed source."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence

import grain.experimental as experimental
import grain.python as grain

from dew._records import boolean, record

from .dataset import LONG_RUN, DataPartition, Loading, logged_run, restore_parent
from .sources.base import Shard, ShardManifest, ShardRows

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
    places: Sequence[tuple[Shard, int]] | None = None,
    quota: int | None = None,
) -> grain.IterDataset:
    """The shards of `manifest` read `epochs` times over (endlessly where None), this share's part
    of them interleaved `cycle_length` at a time; or, given an endless read's bound, this share's
    `places` among the stream's first rows alone, each with the rows it reads of it (`bounded_places`),
    so its interleave reads no later pass's rows in their stead; or, given the share's `quota` of a
    bound over shards that state no rows, its first `quota` rows, each loading worker its own part of
    them. An endless read of a store whose shards
    all state they hold no row is refused, as it would read nothing forever; any other run of empty
    shards is read through, logged at each doubling from `LONG_RUN` (`logged_run`)."""
    if cycle_length < 1:
        raise ValueError("interleave cycle_length must be positive")
    described = name or (manifest.shards[0].uri if manifest.shards else "a stream")
    if epochs is None and all(shard.rows == 0 for shard in manifest.shards):
        raise ValueError(f"{described} holds no rows, so reading it endlessly reads nothing forever")
    if not manifest.shards:
        return grain.MapDataset.source(()).to_iter_dataset()
    if places is not None:
        reader = _Taken(manifest.reader)
        return _ShardPass(grain.MapDataset.source(places), reader, cycle_length, loading, described)
    root = _dealt(_shards(manifest, partition=partition, loading=loading), seed=seed, epochs=epochs)
    root = root[partition.cut()]
    return _ShardPass(root, manifest.reader, cycle_length, loading, described, quota)


def bounded(manifest: ShardManifest, *, partition: DataPartition, seed: int | None, loading: Loading,
            epochs: int | None, examples: int) -> int | None:
    """This share's rows of the stream's first `examples`, its shards dealt to the shares place by place as
    `streamed` deals them (every count-th from the share's index): of an endless read, each pass its own
    order of places, the rows of each of its places the bound reaches (`_reached`); of one pass that a
    later repeat replays, each share its own places again, its rows in every whole pass the bound spans
    and of its places the last pass reaches, in its own order: the repeat reads each pass whole before
    the next, so its interleave crosses no pass. So each share reads its own shards' part of the bound
    and none repeats one while another's go unread. Within that last pass, a share that interleaves its
    places reads its count's first rows in its own interleaved order, not the walk's, which an interleave
    defines on no process: the pool reads the bound's examples, each share's count exact, and no record
    a pass more than another of the bound's. None where a shard states no rows."""
    shards = _shards(manifest, partition=partition, loading=loading)
    if any(shard.rows is None for shard in shards):
        return None
    if not any(shard.rows for shard in shards):
        return 0
    root = _dealt(shards, seed=seed, epochs=epochs)
    if epochs is None:
        return sum(taken for place, _, taken in _reached(root, examples) if place % partition.count
                   == partition.index)
    rows = [_rows(root[place]) for place in range(len(root))]
    mine = sum(rows[partition.index::partition.count])
    passes, rest = divmod(examples, sum(rows))
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
        assert shard is not None, "a stream's dealt places are its shards"
        rows = _rows(shard)
        yield place, shard, min(rows, examples)
        examples -= rows
        place += 1


def _rows(shard: Shard | None) -> int:
    """The rows a dealt place's `shard` states, which every shard a bound walks does (`bounded`)."""
    assert shard is not None and shard.rows is not None, "every shard a bound walks states its rows"
    return shard.rows


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


class _ShardPass[Place](grain.IterDataset):
    """A share's places of a stream (its shards, or `(shard, rows)` of a bound's, `bounded_places`), each
    read by `reader`, interleaved `cycle` at a time, its first `quota` rows alone where the share's bound
    is held in examples over shards that state no rows. A loading worker reads its slice of the places
    (`set_slice`) and its part of the quota, every worker's count-th."""

    def __init__(self, root: grain.MapDataset, reader: Callable[[Place], ShardRows], cycle: int,
                 loading: Loading, name: str, quota: int | None = None):
        super().__init__(root)
        self.root, self.reader = root, reader
        self.cycle, self.loading, self.name, self.quota = cycle, loading, name, quota

    def set_slice(self, selection: slice, sequential_slice: bool = False) -> None:
        self.root = self.root[selection]
        if self.quota is not None:
            self.quota = len(range(*selection.indices(self.quota)))

    def __iter__(self) -> grain.DatasetIterator:
        if not len(self.root) or self.quota == 0:
            # A worker dealt no place or no row of the bound reads nothing (Grain's interleave of no dataset
            # raises).
            return _Passing(None)
        empty = _EmptyRun(self.name)
        rows = experimental.InterleaveIterDataset(
            self.root.map(_Reader(self.reader, empty, self.loading)),
            cycle_length=self.cycle,
            num_make_iter_threads=max(self.loading.threads, 1),
            make_iter_buffer_size=self.loading.read_buffer,
        )
        limited = rows if self.quota is None else experimental.LimitIterDataset(rows, self.quota)
        return _Passing(limited.__iter__())


NOTHING: Mapping[str, object] = {"rows": None}
"""The position of a share's pass dealt nothing (`_Passing`), which no read of rows holds."""


class _Passing(grain.DatasetIterator):
    """A share's rows of its places (`_ShardPass`), or none where it was dealt none, its position its rows'
    as they hold it, or `NOTHING` where it holds none. Grain saves an unstarted loading worker's position
    in the reading process, from the share's whole read before the worker takes its slice, so a worker
    dealt nothing restores a position of rows it never holds: it reads nothing, as it did unstarted."""

    def __init__(self, rows: grain.DatasetIterator | None):
        super().__init__(() if rows is None else rows)
        self._rows = rows

    def __next__(self):
        if self._rows is None:
            raise StopIteration
        return next(self._rows)

    def get_state(self) -> Mapping[str, object]:
        return NOTHING if self._rows is None else self._rows.get_state()

    def set_state(self, state: Mapping[str, object]) -> None:
        if self._rows is not None and state != NOTHING:
            self._rows.set_state(record(state, "a share's rows"))

    def close(self) -> None:
        if self._rows is not None:
            self._rows.close()


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


def iterated(rows: ShardRows, loading: Loading) -> grain.IterDataset:
    """A shard's `rows` streamed: read by index, they are read under the reading's `loading`, its threads
    and its read-ahead, a mix's share of them for a member of one (none of a thousand members' share of
    one thread, which reads on the thread that draws from it), where Grain's own default opened sixteen
    threads and five hundred rows ahead for every shard (row 4079: a thousand-member mix held 9,249
    threads after 190 packed steps, each opened member's own); streamed, as they are."""
    if isinstance(rows, grain.MapDataset):
        return rows.to_iter_dataset(grain.ReadOptions(loading.threads, loading.read_buffer))
    return rows


@dataclasses.dataclass(frozen=True)
class _Taken:
    """A shard reader of `(shard, rows)` places (`bounded_places`): the shard's first `rows` rows."""

    reader: Callable[[Shard], ShardRows]

    def __call__(self, place: tuple[Shard, int]) -> ShardRows:
        shard, rows = place
        read = self.reader(shard)
        if rows == shard.rows:
            return read
        if isinstance(read, grain.MapDataset):
            return read[:rows]
        return experimental.LimitIterDataset(read, rows)


@dataclasses.dataclass(frozen=True)
class _Reader[Place]:
    """A place's reader (a shard's), its rows watched for a run of empty shards (`_EmptyRun`): indexed, as
    the reading's own iterator of them opens them (`_Indexed`), streamed, as they are."""

    reader: Callable[[Place], ShardRows]
    empty: _EmptyRun
    loading: Loading

    def __call__(self, place: Place) -> grain.MapDataset | grain.IterDataset:
        rows = self.reader(place)
        if isinstance(rows, grain.MapDataset):
            return _Indexed(rows, self.empty, self.loading)
        return _Watched(rows, self.empty)


class _Indexed(grain.MapDataset):
    """A shard's indexed rows, which an interleave opens by their own iterator (`MapDataset.__iter__`,
    where a streamed shard is read through a thread of the interleave's that reads it ahead): the
    reading's, its rows streamed under its loading (`iterated`) and watched for a run of empty shards, so
    a mix member's share of no thread reads on the thread that draws from it (row 4079: each member's
    shard of a thousand-member mix read on a thread of its own)."""

    def __init__(self, rows: grain.MapDataset, empty: _EmptyRun, loading: Loading):
        super().__init__(rows)
        self.rows, self.empty, self.loading = rows, empty, loading

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self) -> grain.DatasetIterator:
        return _WatchedIterator(iterated(self.rows, self.loading).__iter__(), self.empty)


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
