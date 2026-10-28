"""What a pipeline reads from: a store, its identity, and how it can be read.

A `Source` names a store (records in memory, files at a URL, a Hub dataset,
a prepared TFDS build, a torch dataset, a catalogue of URLs) and says which of
the three read modes it supports (`Read`). Memory and indexed reads hand over
records by index (`dew.data.dataset.Records`); a streamed read hands over a
`ShardManifest`, the store's physical shards with one reader each, which the
pipeline slices over its readers and workers and interleaves.

A source states facts about itself (`metadata`, `lengths`) and never scans to
find them: a step that needs a global fact reads it here or from config, and
a streamed source without it is refused before the run opens. `describe` and
`fingerprint` are its identity: the first is what a saved position names, the
second what a cache's key and a run's plan record (the URI, the resolved
revision, each file's size and ETag). `label` is the short name a person reads
in a run's metrics, never its identity.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, Self, TypeGuard, runtime_checkable

from dew._records import JSON, json_value

if TYPE_CHECKING:
    import grain.python as pygrain
    import numpy as np

    from dew.data.dataset import Records
    from dew.data.pipeline import Stage

type Read = Literal["memory", "indexed", "streamed"]
"""How a run reads every source it names: `memory` reads each whole into RAM
(and may materialize its deterministic stages, design §10.3), `indexed` reads
records by index through an index the store holds, `streamed` reads shards in
order with no index."""

READS: tuple[Read, ...] = ("memory", "indexed", "streamed")

TOKENIZER_CONTRACT = ("tokenizer", "vocab_size", "eos_id", "tokenizer_revision")
"""The facts of a store of token ids that say what its ids mean, which every corpus a run reads
together, and every store a concatenation joins, must state alike where they state them."""


def tokenizer_contract(stated: Iterable[tuple[str, Mapping[str, object]]]) -> dict[str, JSON]:
    """The tokenizer contract (`TOKENIZER_CONTRACT`) of stores read together, each `(name,
    metadata)`: every fact any of them states, which every other that states it states alike.
    Raises ValueError naming the store whose fact differs from the first's."""
    declared: dict[str, JSON] = {}
    for name, metadata in stated:
        for key in TOKENIZER_CONTRACT:
            if key not in metadata:
                continue
            value = json_value(metadata[key], key)
            if key in declared and value != declared[key]:
                raise ValueError(f"{name} has {key}={value!r}, expected {declared[key]!r}; stores read "
                                 "together must share the tokenizer contract, whose ids they hold")
            declared[key] = value
    return declared


@dataclasses.dataclass(frozen=True)
class Shard:
    """One piece of a streamed source: a file, a byte or row range of one,
    a Parquet row group, or a disjoint row share. `id` is immutable across runs
    (path and range, or a Hub revision and path), so a cache keys rows by `(id, row)`."""

    id: str
    uri: str
    start: int = 0
    stop: int | None = None
    rows: int | None = None


def preparing(source: Source, size: int | None, into: str) -> None:
    """Log the one line a source logs as it prepares a read: what, how large, and where."""
    stated = "size not stated" if size is None else f"{size / 2**20:.1f} MiB"
    logging.getLogger("dew.data").warning("preparing %s (%s) into %s", source.describe(), stated, into)


_COUNTS = ("rows", "accepted", "accepted_rows", "windows", "valid_targets", "zero_target_windows")
"""The `metadata` facts that count a source's rows and their target tokens."""


def stated(source: Source) -> Mapping[str, object]:
    """What `source` states of its records without reading them (`dew.data.pipeline.Stated`), none
    where it states nothing so."""
    from dew.data.pipeline import Stated

    return source.stated() if isinstance(source, Stated) else {}


def _uncounted(metadata: Mapping[str, object]) -> Mapping[str, object]:
    """`metadata` without its counts (`_COUNTS`): the facts a source that reads part of another's
    rows, or rows made from them, keeps of the other's. Its own counts, where it knows them, it
    states itself; a weight or a check that reads another's would describe rows it never reads."""
    return {name: value for name, value in metadata.items() if name not in _COUNTS}


@dataclasses.dataclass(frozen=True)
class ShardManifest:
    """A streamed source's shards, in canonical order, and the reader that
    opens one: a picklable function returning a checkpointable grain
    `IterDataset` over the shard's records in order.

    When physical shards cannot fill the readers and workers, `subdivide`
    receives the needed count and returns exactly that many `Shard`s, each
    a disjoint part of these records with an exact per-part position read
    by the same `reader`. None keeps the physical-shard requirement.

    `index_ordered` states that the shards, one after another, read the
    records in their index order, as a store read by index streams them
    (`_indexed_manifest`)."""

    shards: tuple[Shard, ...]
    reader: Callable[[Shard], pygrain.IterDataset]
    subdivide: Callable[[int], tuple[Shard, ...]] | None = None
    index_ordered: bool = False


@runtime_checkable
class Source(Protocol):
    """A store a pipeline reads, and the read modes it supports."""

    def describe(self) -> str:
        """A stable description: store, URI, revision and split, never an address."""
        ...

    def label(self) -> str:
        """The store's short name in a run's metrics: its dataset, builder or file name."""
        ...

    def fingerprint(self) -> Mapping[str, object]:
        """The store's identity for a cache key and a run's plan: the URI, the
        resolved revision, and each file's size and ETag or generation."""
        ...

    def supports(self, read: Read) -> bool:
        """Whether the source can be read this way without provisioning."""
        ...

    def records(self, read: Read) -> Records:
        """The records by index, for a memory or an indexed read."""
        ...

    def manifest(self) -> ShardManifest:
        """The shards and their reader, for a streamed read."""
        ...

    def metadata(self) -> Mapping[str, object]:
        """Facts the store states about itself (class names, a mixture's
        shares, slot maxima, per-corpus token counts), never found by a scan."""
        ...

    def lengths(self) -> np.ndarray | None:
        """Each record's length when the store states it (a producer's index),
        which length grouping ranks by; None when it does not."""
        ...


_UNNAMED = itertools.count()
"""The places the stores a program makes without a name take in the order it makes them, which tell
apart two such stores read in one run (`unnamed`)."""


def unnamed() -> int:
    """The next place among the stores a program makes without a name: in its run, the key its
    counters and phases keep apart, the same in every run of the program, and no content
    version, which a store without a name has none of (`dew.data.plan.identified`)."""
    return next(_UNNAMED)


def url_label(url: str) -> str:
    """The name a store at `url` goes by: its last path part that is no pattern, without extensions
    (`s3://corpus/forum/*.jsonl` is `forum`, `data/train.jsonl.gz` is `train`)."""
    parts = [part for part in url.split("://", 1)[-1].split("/") if part and not set(part) & set("*?[{")]
    return parts[-1].split(".", 1)[0] if parts else url


@runtime_checkable
class Consumed(Protocol):
    """A source whose read of a given kind goes through another source.

    Wrappers rebuild over their consumed children, retaining their own policy.
    """

    def consumed(self, read: Read) -> Source:
        """The source a read of this one reads its records from."""
        ...


def _consumed(source: Source, read: Read) -> Source:
    return source.consumed(read) if isinstance(source, Consumed) else source


type Rows = Literal["read", "selected", "described", "materialized", "derived"]
"""How a view's rows stand to its source's (`View.rows`): its source's rows as a read took them, its
facts pinned (`read`); some of them, unchanged (`selected`); all of them with facts of its own
added (`described`); rows a memory read made by running a pipeline's first stages over them
(`materialized`); or rows another layout makes of them, windows or a cache's outputs
(`derived`)."""

SAME_ROWS: tuple[Rows, ...] = ("read", "selected", "described")
"""The views whose rows are their source's, or some of them: a walk to the store a run's records
come from (`dew.data.analysis._store`) passes through them."""


class View[S: Source]:
    """A source made of another's rows, `source`: a holdout's selection of them, a compile's read of
    them pinned, a memory read's materialized rows, windows packed of them. Every view names the
    source it reads `source`, so one walk reaches the sources beneath it (`beneath`), and says how
    its rows stand to that source's (`rows`), which decides what it keeps of the source unless it
    states its own: its name, how it is read (of its source's reads, those in `reads`, fewer where
    a selection takes fewer, `refusal` saying why and what reads instead), its records, shards and
    lengths; its facts (`metadata`, `stated`) whole
    where its rows are the source's, without their counts (`_uncounted`) where they are not; and a
    consumed read (`consumed`) taken through the source where its rows are the source's as read,
    or held as they are where it already holds what it read. Its identity (`fingerprint`) is its
    own. A store of several sources (a concatenation) or one that stands in for another when read
    (a Hub dataset's files) is no view."""

    source: S
    rows: ClassVar[Rows]
    reads: ClassVar[tuple[Read, ...]] = READS
    refusal: ClassVar[str] = ""
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]
    """Every view is a dataclass, which `consumed` rebuilds over its consumed source."""

    def label(self) -> str:
        return self.source.label()

    def describe(self) -> str:
        return self.source.describe()

    def supports(self, read: Read) -> bool:
        return read in self.reads and self.source.supports(read)

    def records(self, read: Read) -> Records:
        return self.source.records(read)

    def manifest(self) -> ShardManifest:
        return self.source.manifest()

    def lengths(self) -> np.ndarray | None:
        return self.source.lengths()

    def metadata(self) -> Mapping[str, object]:
        facts = self.source.metadata()
        return facts if self.rows in ("read", "described") else _uncounted(facts)

    def stated(self) -> Mapping[str, object]:
        return _uncounted(stated(self.source)) if self.rows == "selected" else {}

    def consumed(self, read: Read) -> Self:
        if self.rows in ("selected", "described"):
            return dataclasses.replace(self, source=_consumed(self.source, read))
        return self


def beneath(source: Source, through: Collection[Rows]) -> Iterator[Source]:
    """`source`, then the source each view reads, down through every view whose rows are of a kind
    in `through`, ending at the first source that is not one."""
    yield source
    while isinstance(source, View) and source.rows in through:
        source = source.source
        yield source


class Materialized(Source, Protocol):
    """A memory read's rows (`dew.data.materialize`): the source it read, the stages it ran over its
    rows and the source position each row came from."""

    source: Source
    ran: tuple[Stage, ...]
    identifiers: tuple[tuple[int, ...], ...]


def materialized(source: Source) -> TypeGuard[Materialized]:
    """Whether `source` holds rows a memory read made (`View.rows`)."""
    return isinstance(source, View) and source.rows == "materialized"


def _indexed_manifest(source: Source, uri: str, rows: int,
                      reader: Callable[[Shard], pygrain.IterDataset] | None = None) -> ShardManifest:
    """The records `source` reads by index as a stream: consecutive ranges of
    at most `rows` records, each `Shard` named by the record it starts at,
    which no copy of the source names otherwise, wherever it is, and read at
    `uri` as that slice of its indexed records (`_IndexedSlice`), or by
    `reader`, which reads the same slice with less than `source` to pickle."""
    count = len(source.records("indexed"))
    return ShardManifest(
        tuple(
            Shard(f"records:{start}", uri, start, min(start + rows, count), min(rows, count - start))
            for start in range(0, count, rows)
        ),
        _IndexedSlice(source) if reader is None else reader,
        index_ordered=True,
    )


@dataclasses.dataclass(frozen=True)
class _IndexedSlice:
    """Reads a shard of `_indexed_manifest` as the slice of `source`'s indexed
    records it spans. It pickles with its source, as Grain's worker processes
    need."""

    source: Source

    def __call__(self, shard: Shard) -> pygrain.IterDataset:
        import grain.python as pygrain

        records = pygrain.MapDataset.source(self.source.records("indexed"))
        return records[shard.start : shard.stop].to_iter_dataset()


__all__ = ["READS", "Consumed", "Read", "Shard", "ShardManifest", "Source"]
