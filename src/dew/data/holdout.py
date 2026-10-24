"""Held-out records: a dataset split into the part a run trains on and the parts it scores."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from dew.batch import Batch
from dew.data.dataset import Records
from dew.data.generation import Generation, RowId, _GenerationSource, _row_key, _shard_inputs, _source_ids
from dew.data.pipeline import Dataset, Pipeline, with_stores
from dew.data.sources.base import (
    Read,
    Rows,
    Shard,
    ShardManifest,
    Source,
    View,
    _indexed_manifest,
    _uncounted,
    stated,
)
from dew.data.sources.codec import content_fingerprint

_INDEXED_READS: tuple[Read, ...] = ("memory", "indexed")
_BY_INDEX = ("reads by index, under read='memory' or 'indexed'; a stream holds out a hash fraction "
             "(holdout(fraction)) or nothing")
"""Why a holdout that picks records by their index refuses a streamed read, and what streams."""


@runtime_checkable
class _SplitSource(Protocol):
    def with_split(self, name: str) -> Source: ...


def _named_split(source: Source, name: str) -> Source:
    """`source`'s own split `name`, through its `with_split`; a source with no native splits
    refuses it by name."""
    if not isinstance(source, _SplitSource):
        raise ValueError(f"source {source.describe()} has no native split {name!r}")
    return source.with_split(name)


@dataclasses.dataclass(frozen=True)
class RecordSlice(View[Source]):
    """The records [`start`, `stop`) of `source`, by index (`holdout(records=)`): the first `start`
    held out of the part a run trains on, or those a run scores. A run's record names it."""

    rows: ClassVar[Rows] = "selected"
    reads: ClassVar[tuple[Read, ...]] = _INDEXED_READS
    refusal: ClassVar[str] = f"a records= holdout {_BY_INDEX}"
    source: Source
    start: int
    stop: int | None = None

    def describe(self) -> str:
        return f"{self.source.describe()}[{self.start}:{self.stop}]"

    def fingerprint(self) -> Mapping[str, object]:
        return _selected(self.source, "slice", {"slice": [self.start, self.stop]})

    def records(self, read: Read):
        from dew.data.dataset import SourceSlice

        records = self.source.records(read)
        return SourceSlice(records, *self.bounds(len(records)))

    def admits(self, position: int, row: Batch | bytes | None) -> bool:
        """Whether the record at `position` among its source's is in this slice."""
        return self.start <= position and (self.stop is None or position < self.stop)

    def admitted(self, records: int) -> None:
        """Refuse a slice past the `records` it was offered (`bounds`)."""
        self.bounds(records)

    def bounds(self, records: int) -> tuple[int, int]:
        """The first and the one past the last of `records` it keeps, refusing a slice past them."""
        stop = records if self.stop is None else self.stop
        if not 0 <= self.start <= stop <= records:
            raise ValueError(f"{self.describe()} exceeds the source's {records} records")
        return self.start, stop

    def metadata(self) -> Mapping[str, object]:
        return {**super().metadata(), **({"held_out": self.start} if self.start else {})}

    def lengths(self) -> np.ndarray | None:
        lengths = self.source.lengths()
        return None if lengths is None else lengths[self.start : self.stop]

    def manifest(self) -> ShardManifest:
        raise ValueError(self.refusal)


def _inside(row: Batch | bytes, low: float, high: float, key: str | None) -> bool:
    """Whether `row`'s identity falls in [`low`, `high`) of the unit interval: the hash of its
    `key` field, or of its whole content, the same wherever and however the row is read."""
    if key is None:
        identity = content_fingerprint(row)
    elif isinstance(row, Mapping) and key in row:
        identity = content_fingerprint(row[key])
    else:
        raise ValueError(f"a holdout by {key!r} reads that field, which this record lacks")
    hashed = int.from_bytes(identity[:8], "big")
    return int(low * 2**64) <= hashed < int(high * 2**64)


@dataclasses.dataclass(frozen=True)
class _SelectedRow:
    rows: Records

    def __call__(self, index: int) -> Batch | bytes | None:
        return self.rows[index]


@dataclasses.dataclass(frozen=True)
class HashPart(View[Source]):
    """The rows of `source` whose identity (its `key` field, or its content)
    hashes into [`low`, `high`) of the unit interval (`held`), or the rows
    that do not: a row is in exactly one part, the same in every run, read,
    machine and resume (`holdout(fraction)`). A run's record names it."""

    rows: ClassVar[Rows] = "selected"
    reads: ClassVar[tuple[Read, ...]] = ("memory", "streamed")
    refusal: ClassVar[str] = ("a hash holdout reads every row to place it, under read='memory' or "
                              "'streamed'; an indexed read holds out records=")
    source: Source
    low: float
    high: float
    held: bool
    key: str | None = None

    def describe(self) -> str:
        part = f"[{self.low},{self.high}),held={self.held},key={self.key}"
        return f"{self.source.describe()} hash holdout={part}"

    def fingerprint(self) -> Mapping[str, object]:
        part = {"holdout": [self.low, self.high], "held": self.held, "key": self.key,
                "content_hash": "sha256:sorted-mappings-v1"}
        return _selected(self.source, "hash", part)

    def records(self, read: Read):
        import grain.python as grain

        rows = self.source.records(read)
        keys = tuple(index for index in range(len(rows)) if self.keeps(rows[index]))
        return grain.MapDataset.source(keys).map(_SelectedRow(rows))

    def keeps(self, row: Batch | bytes | None) -> bool:
        """Whether this part holds `row`: one that is there, its identity in the part."""
        return row is not None and _inside(row, self.low, self.high, self.key) == self.held

    def admits(self, position: int, row: Batch | bytes | None) -> bool:
        """Whether this part holds `row`, wherever it is (`keeps`)."""
        return self.keeps(row)

    def admitted(self, records: int) -> None:
        """Any number of rows holds a hash part."""

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        """The source's shards, each read through the hash. Which of a shard's rows the hash keeps
        is known only once they are read, so no shard states its rows; the rows it keeps stream in
        the order its source's do, which is its index order where theirs is (`records`)."""
        manifest = self.source.manifest()
        subdivide = manifest.subdivide
        return ShardManifest(_unsized(manifest.shards), functools.partial(self._reader, manifest),
                             None if subdivide is None else functools.partial(_unsized_parts, subdivide),
                             index_ordered=manifest.index_ordered)

    def _reader(self, manifest: ShardManifest, shard: Shard):
        # The other part's rows are no rejection, as a memory read holds none of them.
        return manifest.reader(shard).filter(self.keeps)


def _unsized(shards: Sequence[Shard]) -> tuple[Shard, ...]:
    return tuple(dataclasses.replace(shard, rows=None) for shard in shards)


def _unsized_parts(subdivide: Callable[[int], tuple[Shard, ...]], count: int) -> tuple[Shard, ...]:
    return _unsized(subdivide(count))


def holdout(dataset: Dataset, fraction: float | None = None, *, key: str | None = None,
            records: int | None = None, split: str | tuple[str, ...] | None = None,
            **named: float) -> tuple[Pipeline, ...]:
    """`Dataset.holdout`."""
    if sum(form is not None for form in (fraction, records, split)) + bool(named) != 1:
        raise ValueError("holdout takes one of a fraction, records=, split= or named fractions")
    if split is not None:
        names = (split,) if isinstance(split, str) else split
        if not names or len(set(names)) != len(names):
            raise ValueError(f"holdout split= names one or more distinct splits, not {split!r}")
        held = (with_stores(dataset._chain(), functools.partial(_named_split, name=name)) for name in names)
        return with_stores(dataset._chain(), lambda source: source), *held
    if records is not None:
        if records < 1:
            raise ValueError(f"holdout records= is at least 1, not {records}")
        return (with_stores(dataset._chain(), lambda source: RecordSlice(source, records)),
                with_stores(dataset._chain(), lambda source: RecordSlice(source, 0, records)))
    shares = (fraction,) if fraction is not None else tuple(named.values())
    if any(not 0 < share < 1 for share in shares) or sum(shares) >= 1:
        raise ValueError(f"held-out fractions are each in (0, 1) and sum under 1, not {shares}")
    bounds = np.concatenate([[0.0], np.cumsum(shares)]).tolist()
    chain = dataset._chain()
    trained = with_stores(chain, lambda source: HashPart(source, 0.0, bounds[-1], held=False, key=key))
    held = (functools.partial(HashPart, low=low, high=high, held=True, key=key)
            for low, high in itertools.pairwise(bounds))
    return (trained, *(with_stores(chain, part) for part in held))


_SELECTIONS = (HashPart, RecordSlice)


def selecting(source: Source) -> Iterator[tuple[tuple[RecordSlice | HashPart, ...], Source]]:
    """`source`, then each source its holdout selections select from, down to their store, each with
    the selections above it, outermost first."""
    above: tuple[RecordSlice | HashPart, ...] = ()
    while True:
        yield above, source
        if not isinstance(source, (RecordSlice, HashPart)):
            return
        above, source = (*above, source), source.source


def _selected(source: Source, kind: str, selection: Mapping[str, object]) -> Mapping[str, object]:
    """A selection's identity: what it selects, of its source's identity, so selections stacked on
    one another each keep their own."""
    return {"selection": kind, "of": source.fingerprint(), **selection}


@dataclasses.dataclass(frozen=True)
class SelectedGeneration(View[Source]):
    """The rows `generation` holds for the inputs of `source`, the source it was published from, that
    the holdout `selections` keep (outermost first, `source` below the last), in their order: the
    generation standing in for its steps under selections written before it, each picking the
    inputs it picks uncached (`dew.data.analysis._in_place`). The selections name the part, as the
    run wrote them; `source` is read as the fill read it (`_consumed`), numbering its inputs by
    their place in a memory or indexed read (`placing`), or by its shards' rows in a stream, as the
    fill did (`_source_ids`). Ids that name other inputs than the fill's are refused, never read as none."""

    rows: ClassVar[Rows] = "derived"
    generation: Generation
    selections: tuple[RecordSlice | HashPart, ...]
    source: Source
    placing: Read

    def label(self) -> str:
        return self.selections[0].label()

    def describe(self) -> str:
        return f"generation:{self.generation.url} for {self.selections[0].describe()}"

    def fingerprint(self) -> Mapping[str, object]:
        generation = _GenerationSource(self.generation).fingerprint()
        return {**generation, "selected": self.selections[0].fingerprint()}

    # Every read reads the generation's rows; placing them reads `source` by `placing` whatever the run reads.
    def supports(self, read: Read) -> bool:
        return self.source.supports(self.placing)

    def can_provision(self, read: Read) -> bool:
        from dew.data.analysis import _provisioner

        return _provisioner(self.source, self.placing) is not None

    def provision(self) -> SelectedGeneration:
        from dew.data.analysis import _provisioner

        preparer = _provisioner(self.source, self.placing)
        if preparer is None:
            raise ValueError(f"{self.source.describe()} can't prepare the {self.placing} read that places "
                             "its inputs")
        preparer.provision()
        return self

    def records(self, read: Read) -> Records:
        return _Placed(self.generation, self._dense).rows()

    def manifest(self) -> ShardManifest:
        # A worker reads the placed rows alone (`_Placed`), not the store that placed them.
        return _indexed_manifest(self, self.generation.url, 4096, _Placed(self.generation, self._dense))

    def metadata(self) -> Mapping[str, object]:
        """The selections' facts, as uncached (what they hold out included), with this part's rows and
        the generation's count rules."""
        rules = _GenerationSource(self.generation).metadata()
        return {**_uncounted(self.selections[0].metadata()), "rows": len(self._dense),
                "accepted_rows": len(self._dense),
                **({"count_rules": rules["count_rules"]} if "count_rules" in rules else {})}

    def stated(self) -> Mapping[str, object]:
        return _uncounted(stated(self.source))

    @functools.cached_property
    def whole(self) -> Source:
        """The generation's rows for every input: the whole store this part's measured facts are
        measured over (`dew.data.analysis._stated`), as uncached they are over the store below its
        selections."""
        return self.generation.source()

    def order_fingerprints(self) -> Mapping[str, str]:
        """Which of the generation's rows this part reads, in their order."""
        return {"selected": hashlib.sha256(np.ascontiguousarray(self._dense, np.int64).tobytes()).hexdigest()}

    def lengths(self) -> np.ndarray | None:
        """None: a part packs as it does uncached, where the steps the generation stands for come
        between it and the pack."""
        return None

    @functools.cached_property
    def _dense(self) -> np.ndarray:
        """The indices of the generation's rows of the inputs the selections keep. The generation's
        rows are its inputs' outputs in input order (`_publish`), so they are walked in step with
        the inputs (`_admitted`), each row the next input's that names it."""
        admitted = self._admitted()
        dense = []
        current, kept = None, False
        index = inputs = 0
        for page in range(len(self.generation.manifest["row_pages"])):
            for entry in self.generation._rows(page):
                while entry["id"] != current:
                    current, kept = next(admitted, (None, False))
                    inputs += current is not None
                    if current is None:
                        raise ValueError(
                            f"generation {self.generation.url} holds input {entry['id']}, which "
                            f"{self.source.describe()} doesn't name by its {self.placing} read here (a "
                            "stream names its inputs by where it read their shards, a memory or indexed read "
                            "by their place), so the holdout written before it can't pick its inputs "
                            "from it; read it where it was filled, or publish it again from a "
                            f"{self.placing} read")
                if kept:
                    dense.append(index)
                index += 1
        inputs += sum(1 for _ in admitted)
        filled = self.generation.manifest["inputs"]
        if inputs != filled:
            raise ValueError(f"generation {self.generation.url} was filled from {filled} inputs, and "
                             f"{self.source.describe()} holds {inputs} here, so the holdout written before "
                             "it can't pick its inputs from it")
        return np.asarray(dense, np.int64)

    def _admitted(self) -> Iterator[tuple[str, bool]]:
        """Each input of `source`, by the key the fill gave it (`_row_key`), and whether the selections
        keep it: they apply innermost first as the inputs stream past, each at its own position among
        the inputs the ones within it admitted (`admits`); only a hash holdout reads the rows."""
        inner = self.selections[::-1]
        hashed = any(isinstance(selection, HashPart) for selection in inner)
        if self.placing != "streamed":
            records = self.source.records(self.placing)
            inputs: Iterator[tuple[RowId, Batch | bytes | None]] = (
                (index, records[index] if hashed else None) for index in range(len(records)))
        elif hashed:
            manifest = self.source.manifest()
            inputs = (pair for shard in manifest.shards for pair in _shard_inputs(manifest, shard))
        else:
            inputs = ((row_id, None) for row_id in _source_ids(self.source, streamed=True))
        reached = [0] * len(inner)
        for row_id, row in inputs:
            admitted = True
            for level, selection in enumerate(inner):
                position, reached[level] = reached[level], reached[level] + 1
                if not selection.admits(position, row):
                    admitted = False
                    break
            yield _row_key(row_id), admitted
        for level, selection in enumerate(inner):
            selection.admitted(reached[level])


@dataclasses.dataclass(frozen=True)
class _Placed:
    """The rows of `generation` at `indices`: a selected generation's records (`rows`), and the reader
    of a shard of them a worker takes, holding nothing of the store that placed them."""

    generation: Generation
    indices: np.ndarray

    def rows(self):
        import grain.python as grain

        return grain.MapDataset.source(self.indices).map(_SelectedRow(self.generation))

    def __call__(self, shard: Shard):
        return self.rows()[shard.start : shard.stop].to_iter_dataset()
