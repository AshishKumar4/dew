"""Grain's sliceable root over physical shards, shared by every streamed source."""

from __future__ import annotations

import bisect
import collections
import dataclasses
import functools
import itertools
import logging
import threading
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import grain.experimental as experimental
import grain.python as grain

from dew import _position
from dew._records import boolean, integer, integers, listed, record

from .dataset import LONG_RUN, DataPartition, Loading, logged_run, restore_parent
from .sources.base import IndexedShards, Shard, ShardManifest, ShardOpener, ShardRows

if TYPE_CHECKING:
    from dew.batch import Batch

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
    shards = _shards(manifest, partition=partition, loading=loading)
    # The cut its places were dealt from, which its position holds (`_Passing`): the manifest's own, or
    # the changed rule that cut it finer here (`_changed_cut`).
    dealt = manifest.cut or _changed_cut(manifest, shards)
    alike = manifest.alike or _cut_alike(manifest, shards)
    if places is not None:
        reader = _Taken(manifest.reader)
        return _ShardPass(grain.MapDataset.source(places), reader, cycle_length, loading, described,
                          dealt=dealt, alike=alike)
    # An endless read whose shards are found empty, every place this reader can read, with no row read
    # between, reads nothing forever, which data.md §10 refuses (`_EmptyRun`), whether or not its store counts
    # its rows, as a store whose shards all state none is refused above.
    root = _dealt(shards, seed=seed, epochs=epochs)[partition.cut()]
    return _ShardPass(root, manifest.reader, cycle_length, loading, described, quota,
                      (seed is not None, len(shards)) if epochs is None else None, dealt, alike)


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
    return dataclasses.replace(manifest, shards=shards, subdivide=None, counted=None,
                               cut=_changed_cut(manifest, shards), alike=_cut_alike(manifest, shards))


def _changed_cut(manifest: ShardManifest, shards: tuple[Shard, ...]) -> str | None:
    """The changed rule `manifest`'s shards were cut finer by into `shards` (`ShardManifest.subdivided_by`),
    where its cut differs from the first rule's: ranges of as many records each, which the even cut's are
    where its readers' count divides its records (`dew.data.sources.base.EVEN_CUT`), so a position the
    first rule saved resumes there (Train 1248's review). None where it was not cut, or cut alike."""
    if shards == tuple(manifest.shards) or len({shard.rows for shard in shards}) <= 1:
        return None
    return manifest.subdivided_by


def _cut_alike(manifest: ShardManifest, shards: tuple[Shard, ...]) -> str | None:
    """The changed rule `manifest`'s shards were cut finer by into `shards` where its cut is the first
    rule's (`_changed_cut` names none): an earlier Dew named it in its positions, which read the same rows
    here (`_Passing.set_state`)."""
    if shards == tuple(manifest.shards) or _changed_cut(manifest, shards) is not None:
        return None
    return manifest.subdivided_by


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
                 endless: tuple[bool, int] | None = None, dealt: str | None = None, alike: str | None = None):
        super().__init__(root)
        self.root, self.reader = root, reader
        self.cycle, self.loading, self.name, self.quota, self.endless = cycle, loading, name, quota, endless
        self.dealt, self.alike = dealt, alike

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
            return _Passing(None, self.dealt, self.alike)
        reader = _Reader(self.reader, _EmptyRun(self.name, self.reachable()), self.loading)
        rows = (_Sequential(self.root, reader, self.loading) if self.cycle == 1
                else interleaved(self.root, reader, self.cycle, self.loading))
        limited = rows if self.quota is None else experimental.LimitIterDataset(rows, self.quota)
        return _Passing(limited.__iter__(), self.dealt, self.alike)


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
        # Each place's records by index, read as a window asks for them (`window`), by its place, the
        # latest asked last: at most `MAPPED` of them hold their file open at once (`ShardRows.release`).
        self._indexed: collections.OrderedDict[Hashable, ShardRows | None] = collections.OrderedDict()

    def _by_index(self, place: int) -> ShardRows | None:
        """Place `place`'s records by index (`_Reader.indexed`), opened once, None where its reader serves
        none so; the places asked longest ago let their files go past `MAPPED`, their records' places
        kept."""
        held = self.places[place]
        if held not in self._indexed:
            self._indexed[held] = self.reader.indexed(held) if isinstance(self.reader, _Reader) else None
        self._indexed.move_to_end(held)
        for rows in itertools.islice(reversed(self._indexed.values()), MAPPED, None):
            if rows is not None:
                rows.release()
        return self._indexed[held]

    def window(self, limit: int) -> Window | None:
        """The next `limit` records at most from where it stands, by index, none read (`Window`), every place
        on the way read by index; None where one is not, holds no record (a run of empty places is read as
        a stream, which watches it, `_EmptyRun`), or its position is unknown to its index."""
        place = self.place
        rows = None if place >= len(self.places) else self._by_index(place)
        if place < len(self.places) and rows is None:
            return None
        row = 0 if self.rows is None or rows is None else rows.index_of(
            record(self.rows.get_state()["parent"], "a place's rows"))
        parts: list[tuple[int, int, int, ShardRows]] = []
        held = 0
        while held < limit and place < len(self.places):
            rows = self._by_index(place)
            if rows is None:
                return None
            count = rows.rows(row + limit - held) - row
            if not count and not row:
                return None
            if count:
                parts.append((place, row, count, rows))
                held += count
            if held < limit:
                place, row = place + 1, 0
        return Window(tuple(parts), held)

    def skip(self, window: Window) -> None:
        """Stand past `window`'s records, as reading them would have left it (its rows' position the
        stream's there, `ShardRows.state_at`), none read."""
        if not window.parts:
            return
        place, row, count, rows = window.parts[-1]
        assert isinstance(self.reader, _Reader)
        for passed, _, _, _ in window.parts[:-1]:
            # Each place read through holds rows: no run of empty ones goes on past it (`_EmptyRun`).
            self.reader.empty.ended(yielded=True, place=self.places[passed])
        self.reader.empty.rows += window.held
        for passed_place, _, _, rows_passed in window.parts[:-1]:
            # A place read through holds its file no longer, nor its records' places.
            self._indexed.pop(self.places[passed_place], None)
            rows_passed.close()
        self._close_rows()
        self.place = place
        self.rows = iter(self.reader(self.places[place]))
        self.rows.set_state(self.reader.watched(rows.state_at(row + count)))

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
        self._close_rows()
        position = _sequential_position(state)
        self.place = integer(position["place"], "sequential shard pass's place")
        rows = position["rows"]
        if rows is not None:
            self.rows = iter(self.reader(self.places[self.place]))
            self.rows.set_state(record(rows, "sequential shard pass's rows"))

    def _close_rows(self) -> None:
        if self.rows is not None:
            self.rows.close()
        self.rows = None

    def close(self) -> None:
        self._close_rows()
        for rows in self._indexed.values():
            if rows is not None:
                rows.close()
        self._indexed = collections.OrderedDict()


def made_one(step: Callable, record: Batch) -> Batch:
    """`record` through `step`, maps alone, each making one record of one (`dew.data.analysis.indexable`)."""
    made = step(record)
    assert isinstance(made, Mapping), type(made)
    return made


class IndexedMap(grain.IterDataset):
    """`parent` through `step`, a run of maps making one record of each, as Grain's map runs it, its
    position its parent's; its next records served by index through `made` (`step`'s, each record a
    batch) where its parent serves its own so (`served`): a window shuffle after it reads the records it
    gives alone (`dew.data.ordering.WindowShuffle`)."""

    def __init__(self, parent: grain.IterDataset, step: Callable, made: Callable[[Batch], Batch]):
        super().__init__(parent)
        self.step, self.made = step, made

    def __iter__(self) -> grain.DatasetIterator:
        return _IndexedMapIterator(self._parent.__iter__(), self.step, self.made)


class _IndexedMapIterator(grain.DatasetIterator):
    def __init__(self, parent: grain.DatasetIterator, step: Callable, made: Callable[[Batch], Batch]):
        super().__init__(parent)
        self.step, self.made = step, made

    def __next__(self):
        return self.step(next(self._parent))

    def get_state(self):
        return self._parent.get_state()

    def set_state(self, state) -> None:
        self._parent.set_state(state)

    def window(self, limit: int) -> Window | None:
        window = served(self._parent, limit)
        return None if window is None else window.mapped(self.made)

    def skip(self, window: Window) -> None:
        passed(self._parent, window)


@runtime_checkable
class ServesWindows(Protocol):
    """An iterator whose next records can be served by index (`Window`) and stood past, none read: a shard
    pass's (`_Passing`), or one of maps after it (`IndexedMap`)."""

    def window(self, limit: int) -> Window | None: ...

    def skip(self, window: Window) -> None: ...


def served(rows: grain.DatasetIterator, limit: int) -> Window | None:
    """The next `limit` records of `rows` at most, by index, none read, where `rows` serves them so
    (`ServesWindows`); None for any other, read as a stream."""
    return rows.window(limit) if isinstance(rows, ServesWindows) else None


def passed(rows: grain.DatasetIterator, window: Window) -> None:
    """Stand `rows`, the iterator that served `window` (`served`), past its records."""
    assert isinstance(rows, ServesWindows)
    rows.skip(window)


MAPPED = 8
"""The places a sequential pass holds open for reads by index at once (`_SequentialIterator._by_index`): a
window over more one-record places than a process may open files reads each again where it is asked."""


@dataclasses.dataclass(frozen=True)
class Window:
    """Records ahead of a stream's position, by index (`_SequentialIterator.window`): `held` of them, from
    each of `parts` in order: a place, the index its records start at there, how many and its records by
    index."""

    parts: tuple[tuple[int, int, int, ShardRows], ...]
    held: int
    steps: tuple[Callable[[Batch], Batch], ...] = ()
    """What each record is made through after it is read, in order (`mapped`)."""

    @functools.cached_property
    def starts(self) -> list[int]:
        """Where each part's records start among the window's."""
        return list(itertools.accumulate((count for _, _, count, _ in self.parts[:-1]), initial=0))

    def mapped(self, step: Callable[[Batch], Batch]) -> Window:
        """This window, its records each made through `step` too (`IndexedMap`)."""
        return dataclasses.replace(self, steps=(*self.steps, step))

    def __getitem__(self, index: int) -> Batch:
        part = bisect.bisect_right(self.starts, index) - 1
        _, row, _, rows = self.parts[part]
        made = rows[row + index - self.starts[part]]
        for step in self.steps:
            made = step(made)
        return made


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

    def __init__(self, rows: grain.DatasetIterator | None, dealt: str | None = None,
                 alike: str | None = None):
        super().__init__(() if rows is None else rows)
        self._rows, self.dealt, self.alike = rows, dealt, alike

    def __next__(self):
        if self._rows is None:
            raise StopIteration
        return next(self._rows)

    def get_state(self) -> Mapping[str, object]:
        state = NOTHING if self._rows is None else self._rows.get_state()
        return state if self.dealt is None else {"rows": state, "cut": self.dealt}

    def set_state(self, state: Mapping[str, object]) -> None:
        saved = state.get("cut") if "rows" in state and "cut" in state else None
        # A position an earlier Dew saved under a cut it named that cuts as the first rule does (`alike`)
        # reads the rows it named.
        if saved != self.dealt and not (self.dealt is None and saved is not None and saved == self.alike):
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

    def window(self, limit: int) -> Window | None:
        """Its rows' next `limit` records by index (`_SequentialIterator.window`), None where they are read as
        a stream alone (an interleave, a bound in examples) or it was dealt none."""
        rows = self._rows
        return rows.window(limit) if isinstance(rows, _SequentialIterator) else None

    def skip(self, window: Window) -> None:
        """Stand past `window`'s records (`_SequentialIterator.skip`)."""
        assert isinstance(self._rows, _SequentialIterator)
        self._rows.skip(window)


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

    def indexed(self, place: Place) -> ShardRows | None:
        """`place`'s records by index as `__call__` reads them, a shard's whose reader serves them so
        (`IndexedShards`); None for any other, a bound's `(shard, rows)` place among them."""
        if not isinstance(place, Shard) or not isinstance(self.reader, IndexedShards):
            return None
        return self.reader.indexed(place)

    @staticmethod
    def watched(state: Mapping[str, object]) -> Mapping[str, object]:
        """The position of a place's watched rows (`_WatchedIterator`) whose reader stands at `state`,
        having read a row: one placed past rows read by index (`_SequentialIterator.skip`)."""
        return {"parent": state, "yielded": True}


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
