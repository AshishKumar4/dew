"""Held-out records: a dataset split into the part a run trains on and the parts it scores."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset, Mixture, Phases, Pipeline
from dew.data.sources.base import Read, Shard, ShardManifest, Source, _consumed, _uncounted, stated
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
class RecordSlice:
    """The records [`start`, `stop`) of `source`, by index (`holdout(records=)`): the first `start`
    held out of the part a run trains on, or those a run scores. A run's record names it."""

    reads: ClassVar[tuple[Read, ...]] = _INDEXED_READS
    refusal: ClassVar[str] = f"a records= holdout {_BY_INDEX}"
    source: Source
    start: int
    stop: int | None = None

    def consumed(self, read: Read) -> RecordSlice:
        return dataclasses.replace(self, source=_consumed(self.source, read))

    def label(self) -> str:
        return self.source.label()

    def describe(self) -> str:
        return f"{self.source.describe()}[{self.start}:{self.stop}]"

    def fingerprint(self) -> Mapping[str, object]:
        return _selected(self.source, "slice", {"slice": [self.start, self.stop]})

    def supports(self, read: Read) -> bool:
        return read in self.reads and self.source.supports(read)

    def records(self, read: Read):
        from dew.data.dataset import SourceSlice

        records = self.source.records(read)
        return SourceSlice(records, *self.bounds(len(records)))

    def bounds(self, records: int) -> tuple[int, int]:
        """The first and the one past the last of `records` it keeps, refusing a slice past them."""
        stop = records if self.stop is None else self.stop
        if not 0 <= self.start <= stop <= records:
            raise ValueError(f"{self.describe()} exceeds the source's {records} records")
        return self.start, stop

    def metadata(self) -> Mapping[str, object]:
        return {**_uncounted(self.source.metadata()), **({"held_out": self.start} if self.start else {})}

    def stated(self) -> Mapping[str, object]:
        return _uncounted(stated(self.source))

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
class _HashRows:
    low: float
    high: float
    held: bool
    key: str | None = None

    def __call__(self, row: Batch) -> bool:
        """Whether `row` is in this part: the other part's rows are no rejection, as a memory read
        holds none of them."""
        return _inside(row, self.low, self.high, self.key) == self.held


@dataclasses.dataclass(frozen=True)
class _SelectedRow:
    rows: Records

    def __call__(self, index: int) -> Batch | bytes | None:
        return self.rows[index]


@dataclasses.dataclass(frozen=True)
class HashPart:
    """The rows of `source` whose identity (its `key` field, or its content)
    hashes into [`low`, `high`) of the unit interval (`held`), or the rows
    that do not: a row is in exactly one part, the same in every run, read,
    machine and resume (`holdout(fraction)`). A run's record names it."""

    reads: ClassVar[tuple[Read, ...]] = ("memory", "streamed")
    refusal: ClassVar[str] = ("a hash holdout reads every row to place it, under read='memory' or "
                              "'streamed'; an indexed read holds out records=")
    source: Source
    low: float
    high: float
    held: bool
    key: str | None = None

    def consumed(self, read: Read) -> HashPart:
        return dataclasses.replace(self, source=_consumed(self.source, read))

    def label(self) -> str:
        return self.source.label()

    def describe(self) -> str:
        part = f"[{self.low},{self.high}),held={self.held},key={self.key}"
        return f"{self.source.describe()} hash holdout={part}"

    def fingerprint(self) -> Mapping[str, object]:
        part = {"holdout": [self.low, self.high], "held": self.held, "key": self.key,
                "content_hash": "sha256:sorted-mappings-v1"}
        return _selected(self.source, "hash", part)

    def supports(self, read: Read) -> bool:
        return read in self.reads and self.source.supports(read)

    def records(self, read: Read):
        import grain.python as grain

        rows = self.source.records(read)
        keys = tuple(index for index in range(len(rows)) if self.keeps(rows[index]))
        return grain.MapDataset.source(keys).map(_SelectedRow(rows))

    def keeps(self, row: Batch | bytes | None) -> bool:
        """Whether this part holds `row`: one that is there, its identity in the part."""
        return row is not None and _inside(row, self.low, self.high, self.key) == self.held

    def metadata(self) -> Mapping[str, object]:
        return _uncounted(self.source.metadata())

    def stated(self) -> Mapping[str, object]:
        return _uncounted(stated(self.source))

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        """The source's shards, each read through the hash. Which of a shard's rows the hash keeps
        is known only once they are read, so no shard states its rows."""
        manifest = self.source.manifest()
        subdivide = manifest.subdivide
        return ShardManifest(_unsized(manifest.shards), functools.partial(self._reader, manifest),
                             None if subdivide is None else functools.partial(_unsized_parts, subdivide))

    def _reader(self, manifest: ShardManifest, shard: Shard):
        return manifest.reader(shard).filter(_HashRows(self.low, self.high, self.held, self.key))


def _unsized(shards: Sequence[Shard]) -> tuple[Shard, ...]:
    return tuple(dataclasses.replace(shard, rows=None) for shard in shards)


def _unsized_parts(subdivide: Callable[[int], tuple[Shard, ...]], count: int) -> tuple[Shard, ...]:
    return _unsized(subdivide(count))


def _rebased(dataset: Dataset, store: Callable[[Source], Source]) -> Pipeline:
    """`dataset` with each of its stores read through `store`, its steps unchanged."""
    chain = dataset._chain()
    source = chain.source
    if isinstance(source, (Mixture, Phases)):
        children = tuple(_rebased(child, store) for child in source.pipelines)
        source = dataclasses.replace(source, pipelines=children)
    else:
        source = store(source)
    return dataclasses.replace(chain, source=source)


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
        held = (_rebased(dataset, functools.partial(_named_split, name=name)) for name in names)
        return _rebased(dataset, lambda source: source), *held
    if records is not None:
        if records < 1:
            raise ValueError(f"holdout records= is at least 1, not {records}")
        return (_rebased(dataset, lambda source: RecordSlice(source, records)),
                _rebased(dataset, lambda source: RecordSlice(source, 0, records)))
    shares = (fraction,) if fraction is not None else tuple(named.values())
    if any(not 0 < share < 1 for share in shares) or sum(shares) >= 1:
        raise ValueError(f"held-out fractions are each in (0, 1) and sum under 1, not {shares}")
    bounds = np.concatenate([[0.0], np.cumsum(shares)]).tolist()
    trained = _rebased(dataset, lambda source: HashPart(source, 0.0, bounds[-1], held=False, key=key))
    held = (functools.partial(HashPart, low=low, high=high, held=True, key=key)
            for low, high in itertools.pairwise(bounds))
    return (trained, *(_rebased(dataset, part) for part in held))


_SELECTIONS = (HashPart, RecordSlice)


def selecting(source: Source | Mixture | Phases) -> Iterator[Source | Mixture | Phases]:
    """`source`, then each source its holdout selections select from, down to their store."""
    yield source
    if isinstance(source, _SELECTIONS):
        yield from selecting(source.source)


def _selected(source: Source, kind: str, selection: Mapping[str, object]) -> Mapping[str, object]:
    """A selection's identity: what it selects, of its source's identity, so selections stacked on
    one another each keep their own."""
    return {"selection": kind, "of": source.fingerprint(), **selection}
