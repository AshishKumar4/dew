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
revision, each file's size and ETag).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import grain.python as pygrain
    import numpy as np

    from dew.data.dataset import Records

type Read = Literal["memory", "indexed", "streamed"]
"""How a run reads every source it names: `memory` reads each whole into RAM
(and may materialize its deterministic stages, design §10.3), `indexed` reads
records by index through an index the store holds, `streamed` reads shards in
order with no index."""

READS: tuple[Read, ...] = ("memory", "indexed", "streamed")


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
    by the same `reader`. None keeps the physical-shard requirement."""

    shards: tuple[Shard, ...]
    reader: Callable[[Shard], pygrain.IterDataset]
    subdivide: Callable[[int], tuple[Shard, ...]] | None = None


@runtime_checkable
class Source(Protocol):
    """A store a pipeline reads, and the read modes it supports."""

    def describe(self) -> str:
        """A stable description: store, URI, revision and split, never an address."""
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


def _indexed_manifest(source: Source, prefix: str, uri: str, rows: int) -> ShardManifest:
    """The records `source` reads by index as a stream: consecutive ranges of
    at most `rows` records, each `Shard` named `<prefix>:<start>` at `uri`,
    read as that slice of its indexed records (`_IndexedSlice`)."""
    count = len(source.records("indexed"))
    return ShardManifest(
        tuple(
            Shard(f"{prefix}:{start}", uri, start, min(start + rows, count), min(rows, count - start))
            for start in range(0, count, rows)
        ),
        _IndexedSlice(source),
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
