"""Grain's sliceable root over physical shards, shared by every streamed source."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence

import grain.experimental as experimental
import grain.python as grain

from dew import _position
from dew._records import boolean, integer, integers, listed, record

from .dataset import LONG_RUN, DataPartition, Loading, logged_run, restore_parent
from .sources.base import EVEN_CUT, Shard, ShardManifest, ShardOpener

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
    shards = _shards(manifest, partition=partition, loading=loading)
    # The cut its places were dealt from, which its position holds (`_Passing`): the manifest's own, or
    # the one that cut it finer here.
    dealt = manifest.cut or (EVEN_CUT if shards != tuple(manifest.shards) else None)
    # An endless read whose shards are found empty, every place this reader can read, with no row read
    # between, reads nothing forever, which data.md §10 refuses (`_EmptyRun`), whether or not its store counts
    # its rows, as a store whose shards all state none is refused above.
    root = _dealt(shards, seed=seed, epochs=epochs)[partition.cut()]
    return _ShardPass(root, manifest.reader, cycle_length, loading, described, quota,
                      (seed is not None, len(shards)) if epochs is None else None, dealt)


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
    a pass more than another of the bound's. Its shards' rows are the ones they state or their store counts,
    `Loading.threads` at once (`ShardManifest.sized`); None where a shard states none."""
    shards = _shards(manifest.sized(loading.threads), partition=partition, loading=loading)
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
    no endless one, or a shard states no rows and its store counts none (`ShardManifest.sized`)."""
    if epochs is not None:
        return None
    shards = _shards(manifest.sized(loading.threads), partition=partition, loading=loading)
    if any(shard.rows is None for shard in shards):
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


def fills(stored: ShardManifest, read: ShardManifest) -> bool:
    """Whether a cached stream read over `read`'s shards fills the cache: where each is one of `stored`'s,
    the store's own or its parts, which every topology's fill names alike; a cut finer than the store for
    its readers keys its rows by cuts no other fill names, whose completions would block every later
    fill's publication, so it reads a published generation and fills none (review of row 11948)."""
    cuts = {shard.id for shard in read.shards}
    return cuts <= {shard.id for shard in stored.shards} or cuts <= {shard.id for shard in stored.parts()}


def cut(manifest: ShardManifest, *, partition: DataPartition, loading: Loading) -> ShardManifest:
    """`manifest` with the shards a stream deals its shares and their workers (`_shards`): its own, or its
    parts where too few fill them, then no further part (`subdivide`) and their rows their own, so a read
    that places its records (`dew.data.lowering._positioned`) places the shards it reads."""
    if not manifest.shards:
        return manifest
    shards = _shards(manifest, partition=partition, loading=loading)
    if shards == manifest.shards:
        return manifest
    return dataclasses.replace(manifest, shards=shards, subdivide=None, counted=None, cut=EVEN_CUT)


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

    def __init__(self, root: grain.MapDataset, reader: Callable[..., grain.IterDataset], cycle: int,
                 loading: Loading, name: str, quota: int | None = None,
                 endless: tuple[bool, int] | None = None, dealt: str | None = None):
        super().__init__(root)
        self.root, self.reader = root, reader
        self.cycle, self.loading, self.name, self.quota, self.endless = cycle, loading, name, quota, endless
        self.dealt = dealt

    def reachable(self) -> int | None:
        """The distinct places an endless read can read on this reader (and this loading worker's slice),
        which a run of empty ones must cover to refuse it (`_EmptyRun`): every shard of the store where a
        seed deals each pass its own order, so a share's slots reach each in turn; else the shards of one
        period of its fixed slots, which repeat within a pass's length of places. None for a read that
        ends."""
        if self.endless is None:
            return None
        seeded, shards = self.endless
        if seeded:
            return shards
        return len({self.root[index] for index in range(min(len(self.root), shards))})

    def set_slice(self, selection: slice, sequential_slice: bool = False) -> None:
        self.root = self.root[selection]
        if self.quota is not None:
            self.quota = len(range(*selection.indices(self.quota)))

    def __iter__(self) -> grain.DatasetIterator:
        if not len(self.root) or self.quota == 0:
            # A worker dealt no place or no row of the bound reads nothing (Grain's interleave of no dataset
            # raises).
            return _Passing(None, self.dealt)
        reader = _Reader(self.reader, _EmptyRun(self.name, self.reachable()), self.loading)
        rows = (_Sequential(self.root, reader, self.loading) if self.cycle == 1
                else interleaved(self.root, reader, self.cycle, self.loading))
        limited = rows if self.quota is None else experimental.LimitIterDataset(rows, self.quota)
        return _Passing(limited.__iter__(), self.dealt)


def interleaved(places: grain.MapDataset, reader: Callable[..., grain.IterDataset], cycle: int,
                loading: Loading) -> grain.IterDataset:
    """`places` read by `reader`, `cycle` of them open at once, by Grain's interleave: each open place read a
    row ahead on a thread of the interleave's own, the next ones opened ahead on the loading's threads."""
    return experimental.InterleaveIterDataset(places.map(reader), cycle_length=cycle,
                                              num_make_iter_threads=max(loading.threads, 1),
                                              make_iter_buffer_size=loading.read_buffer)


class _Sequential(grain.IterDataset):
    """A share's places read one after another, each by `reader` as the one before it ends, on the thread
    that reads them: the rows, in their order, of Grain's interleave of one place at a time (`interleaved`),
    which read each open place a row ahead on a thread of its own, so a thousand members of a mix, each its
    place open, held a thousand threads (row 4079); a place's own reads run under the reading's loading."""

    def __init__(self, places: grain.MapDataset, reader: Callable[..., grain.IterDataset], loading: Loading):
        super().__init__()
        self.places, self.reader, self.loading = places, reader, loading

    def __iter__(self) -> grain.DatasetIterator:
        return _SequentialIterator(self.places, self.reader, self.loading)


_ENDED = object()
"""A place's rows read to their end (`_SequentialIterator`)."""


class _SequentialIterator(grain.DatasetIterator):
    """`_Sequential`'s reads: the place it reads (`place`) and its rows' position there, None before it
    opens; a position an earlier Dew saved, its interleave's of one place at a time, read as the place it
    read and its rows' position there (`_sequential_position`)."""

    def __init__(self, places: grain.MapDataset, reader: Callable[..., grain.IterDataset], loading: Loading):
        super().__init__()
        self.places, self.reader, self.loading = places, reader, loading
        self.place, self.rows = 0, None

    def __next__(self):
        # Each place read to its end, then the next opened, until the last ends.
        while self.place < len(self.places):
            if self.rows is None:
                self.rows = iter(self.reader(self.places[self.place]))
            row = next(self.rows, _ENDED)
            if row is not _ENDED:
                return row
            self.rows.close()
            self.rows, self.place = None, self.place + 1
        raise StopIteration

    def get_state(self) -> Mapping[str, object]:
        return {"place": self.place, "rows": None if self.rows is None else self.rows.get_state()}

    def set_state(self, state: Mapping[str, object]) -> None:
        self.close()
        position = _sequential_position(state)
        self.place = integer(position["place"], "sequential shard pass's place")
        rows = position["rows"]
        if rows is not None:
            self.rows = iter(self.reader(self.places[self.place]))
            self.rows.set_state(record(rows, "sequential shard pass's rows"))

    def close(self) -> None:
        if self.rows is not None:
            self.rows.close()
        self.rows = None


_INTERLEAVED = frozenset({"iterators_in_use_indices", "exhausted", "iterators_in_use_states",
                          "next_index_in_datasets"})
"""The fields of the position an earlier Dew's interleave of one place at a time saved
(`_sequential_position`)."""


def _sequential_position(state: Mapping[str, object]) -> Mapping[str, object]:
    """`state`, a sequential pass's position (`_SequentialIterator`), or one an earlier Dew saved of the
    same rows in the same order, Grain's interleave of one place at a time, read as the sequential one: the
    place its iterator in use read and that iterator's position there (each place read by the same reader,
    so its rows' position is the one the sequential reader keeps), or, its iterator exhausted or none open,
    the next place it opens from its start. The one translation of it, at the restore, the run reading on
    by the sequential reader (row 4079); deleted at 1.0. Either recognized by its own fields; any other
    is a corrupt cursor (`MalformedPosition`), never taken for the other form."""
    if {"place", "rows"} <= state.keys():
        return state
    if not state.keys() >= _INTERLEAVED:
        raise _position.MalformedPosition(f"a sequential shard pass's saved position holds neither a place "
                                          f"and its rows nor an interleave's places: {sorted(state)}")
    (index,) = integers(state["iterators_in_use_indices"], "interleave's places in use")
    (exhausted,) = integers(state["exhausted"], "interleave's exhausted places")
    (rows,) = listed(state["iterators_in_use_states"], "interleave's places' positions",
                     read=lambda value, key: value, what="a list of positions")
    if rows is None or exhausted:
        return {"place": state["next_index_in_datasets"], "rows": None}
    return {"place": index, "rows": rows}


NOTHING: Mapping[str, object] = {"rows": None}
"""The position of a share's pass dealt nothing (`_Passing`), which no read of rows holds."""


class _Passing(grain.DatasetIterator):
    """A share's rows of its places (`_ShardPass`), or none where it was dealt none, its position its rows'
    as they hold it, or `NOTHING` where it holds none. Grain saves an unstarted loading worker's position
    in the reading process, from the share's whole read before the worker takes its slice, so a worker
    dealt nothing restores a position of rows it never holds: it reads nothing, as it did unstarted. A share
    whose places were dealt from its store cut finer (`dealt`, `ShardManifest.cut`) holds that cut beside
    its rows' position, and a position saved under another cut, or none, is refused (`PlanChanged`), its
    places other rows; one of a share no cut dealt holds none, as before any cut was named."""

    def __init__(self, rows: grain.DatasetIterator | None, dealt: str | None = None):
        super().__init__(() if rows is None else rows)
        self._rows, self.dealt = rows, dealt

    def __next__(self):
        if self._rows is None:
            raise StopIteration
        return next(self._rows)

    def get_state(self) -> Mapping[str, object]:
        state = NOTHING if self._rows is None else self._rows.get_state()
        return state if self.dealt is None else {"rows": state, "cut": self.dealt}

    def set_state(self, state: Mapping[str, object]) -> None:
        saved = state.get("cut") if "rows" in state and "cut" in state else None
        if saved != self.dealt:
            raise _position.PlanChanged(
                ("shards.cut",),
                f"the share's position was saved under its store's {saved or 'first'} cut and it reads "
                f"under the {self.dealt or 'first'} cut now: its places are other rows; a relaxed resume "
                "reads its data afresh")
        rows = record(state["rows"], "a share's rows") if saved is not None else state
        if self._rows is not None and rows != NOTHING:
            self._rows.set_state(record(rows, "a share's rows"))

    def close(self) -> None:
        if self._rows is not None:
            self._rows.close()


class _EmptyRun:
    """How many shards in a row one read found empty, no row read anywhere between them (`rows`, which every
    row read moves, an interleave's open shards' among them), which it logs at each doubling. Grain reads
    each shard on a prefetch thread of its own, so the run is kept under a lock. Given `places`, the
    distinct places an endless read can read on this reader (`_ShardPass.reachable`), it is refused once the
    run has found every one of them empty, by its place, where it would read nothing forever: one rule for
    a store that counts its rows and one that counts none, asking no store."""

    def __init__(self, name: str, places: int | None = None):
        self.name, self.count, self.places = name, 0, places
        self.rows = self.seen = 0
        self.empty: set[Hashable] = set()
        self.lock = threading.Lock()

    def ended(self, yielded: bool, place: Hashable) -> None:
        with self.lock:
            if yielded or self.rows != self.seen:
                self.count, self.empty = 0, set()
            self.seen = self.rows
            if not yielded:
                self.count += 1
                self.empty.add(place)
            count, found = self.count, len(self.empty)
        if self.places is not None and found >= self.places:
            raise ValueError(f"{self.name} holds no rows, so reading it endlessly reads nothing forever")
        if logged_run(count, LONG_RUN):
            _log.warning("%s: its last %d shards held no row; the read goes on to the next", self.name, count)


@dataclasses.dataclass(frozen=True)
class _Taken:
    """A shard reader of `(shard, rows)` places (`bounded_places`): the shard's first `rows` rows."""

    reader: ShardOpener

    def __call__(self, place: tuple[Shard, int],
                 options: grain.ReadOptions | None = None) -> grain.IterDataset:
        shard, rows = place
        read = self.reader(shard, options)
        return read if rows == shard.rows else experimental.LimitIterDataset(read, rows)


@dataclasses.dataclass(frozen=True)
class _Reader[Place]:
    """A place's reader (a shard's), its rows read under the reading's loading, its threads and its
    read-ahead, and watched for a run of empty shards (`_EmptyRun`). A mix member reads under its share of
    the run's (none of a thousand members' share of one thread, which read on the thread that draws
    from them), where Grain's own read each shard on sixteen threads of its own and five hundred rows
    ahead (row 4079: a thousand-member packed mix held 9,249 threads after 190 steps)."""

    reader: Callable[..., grain.IterDataset]
    empty: _EmptyRun
    loading: Loading

    def __call__(self, place: Place) -> grain.IterDataset:
        options = grain.ReadOptions(self.loading.threads, self.loading.read_buffer)
        return _Watched(self.reader(place, options), self.empty, place)


class _Watched(grain.IterDataset):
    """One shard's rows, telling `empty` whether it held any as it ends."""

    def __init__(self, parent: grain.IterDataset, empty: _EmptyRun, place: Hashable):
        super().__init__(parent)
        self.input_parent, self.empty, self.place = parent, empty, place

    def __iter__(self) -> grain.DatasetIterator:
        return _WatchedIterator(self.input_parent.__iter__(), self.empty, self.place)


class _WatchedIterator(grain.DatasetIterator):
    def __init__(self, parent: grain.DatasetIterator, empty: _EmptyRun, place: Hashable):
        super().__init__(parent)
        self.input_parent, self.empty, self.place = parent, empty, place
        self.yielded = False

    def __next__(self):
        try:
            row = next(self.input_parent)
        except StopIteration:
            self.empty.ended(self.yielded, self.place)
            raise
        self.yielded = True
        # Any row read ends a run of empty shards, an open shard's under an interleave among them.
        self.empty.rows += 1
        return row

    def get_state(self) -> Mapping[str, object]:
        return {"parent": self.input_parent.get_state(), "yielded": self.yielded}

    def set_state(self, state: Mapping[str, object]) -> None:
        restore_parent(self.input_parent, state, "parent", "shard position")
        self.yielded = boolean(state["yielded"], "shard yielded")

    def close(self) -> None:
        self.input_parent.close()
