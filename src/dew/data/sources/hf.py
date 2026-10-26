"""A Hugging Face `datasets` dataset read as a Dew store."""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grain.python as grain
import numpy as np

from dew._records import record
from dew.batch import Batch
from dew.data.dataset import Closeable, Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest
from .codec import Value
from .iterator import _StreamIterator
from .memory import _record

if TYPE_CHECKING:
    import datasets
    from datasets.iterable_dataset import _BaseExamplesIterable


@dataclass(frozen=True)
class _HFShard(Shard):
    """One library shard, or a row-wise share when readers outnumber shards."""

    count: int = 1
    index: int = 0


@dataclass(frozen=True, eq=False)
class HF(Dataset):
    """A `datasets.Dataset`, read by index from its Arrow table or as one
    stream of it, or an `IterableDataset`, read over its shards whose
    positions its own `state_dict` checkpoints; the `.map()`s it holds are already applied. An
    image column reads as its encoded bytes, which `images.DecodeImage` decodes.
    `name` is its identity in a run's plan, a cache's key and its logs, its
    fingerprint by default; a stream has no fingerprint of its content, so
    it is named, with its version, or refused.

    A stream resumes exactly from its `state_dict` only unshuffled, since
    Hugging Face's shuffle buffer is not part of that state: one that
    shuffles through it is refused; shuffle with the chain's
    `shuffle(window=)` instead.

    A pool larger than the physical shards uses a row-wise node split.
    These parts resume exactly at the same topology, but their cache fill
    cannot publish: publication needs the store's own shards."""

    dataset: datasets.Dataset | datasets.IterableDataset
    name: str | None = None

    def __post_init__(self) -> None:
        from datasets import IterableDataset

        if not isinstance(self.dataset, IterableDataset):
            return
        if self.name is None:
            raise ValueError("an IterableDataset has no fingerprint of its content; name it, with its "
                             "version, so a resume can tell it from another stream: from_hf(dataset, name=)")
        if _buffer_shuffled(self.dataset._ex_iterable):
            raise ValueError("an IterableDataset's own shuffle loses its buffer on resume; pass it "
                             "unshuffled and shuffle with the chain's shuffle(window=)")

    @property
    def streamed(self) -> bool:
        from datasets import IterableDataset

        return isinstance(self.dataset, IterableDataset)

    def label(self) -> str:
        return self.name or self.dataset.info.dataset_name or "hf"

    def describe(self) -> str:
        from datasets import Dataset as Arrow

        if self.name:
            return f"hf:{self.name}"
        if isinstance(self.dataset, Arrow):
            return f"hf:{self.dataset._fingerprint}"
        return f"hf:{type(self.dataset).__name__}"

    def fingerprint(self) -> Mapping[str, object]:
        return {"store": "hf-datasets", "name": self.describe(), **self.metadata()}

    def supports(self, read: Read) -> bool:
        return read == "streamed" or not self.streamed

    def records(self, read: Read) -> Records:
        from datasets import Dataset as Arrow

        if not isinstance(self.dataset, Arrow):
            raise ValueError(f"{self.describe()} is an IterableDataset, read only as a stream")
        return arrow_rows(self.dataset)

    def manifest(self) -> ShardManifest:
        from datasets import Dataset as Arrow

        if isinstance(self.dataset, Arrow):
            rows = len(self.dataset)
            # One shard, the Arrow dataset's records in their index order.
            return ShardManifest((Shard(self.describe(), self.describe(), 0, rows, rows),), self._read_shard,
                                 index_ordered=True)
        return ShardManifest(self._shards(self.dataset.n_shards), self._read_shard, self._shards)

    def _shards(self, count: int) -> tuple[Shard, ...]:
        name = self.describe()
        return tuple(_HFShard(f"{name}/shards-{count}/{index}", name, count=count, index=index)
                     for index in range(count))

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        from datasets import Dataset as Arrow

        if isinstance(self.dataset, Arrow):
            return arrow_rows(self.dataset).to_iter_dataset()
        if not isinstance(shard, _HFShard):
            raise ValueError("a Hugging Face stream needs its own shard descriptor")
        return _Streamed(_undecoded(self.dataset), shard)

    def metadata(self) -> Mapping[str, object]:
        """Its class names (`class_names`), as Hub states a prepared dataset's, and
        its rows where the dataset is indexed."""
        from datasets import Dataset as Arrow

        rows = {"rows": len(self.dataset)} if isinstance(self.dataset, Arrow) else {}
        return {"class_names": class_names(self.dataset.features), **rows}

    def lengths(self) -> np.ndarray | None:
        return None


def arrow_rows(dataset: datasets.Dataset) -> grain.MapDataset:
    """An Arrow dataset's rows by index, as Dew reads them: its image columns undecoded, which Dew's
    own steps decode, and each row a record (`_record`). A Hugging Face dataset and a prepared Hub
    split read their rows through it."""
    return grain.MapDataset.source(_undecoded(dataset)).map(_record)


def _undecoded[D: (datasets.Dataset, datasets.IterableDataset)](dataset: D) -> D:
    """`dataset` with its decoding image columns left as bytes, which Dew's own steps decode."""
    from datasets import Image

    for name, feature in (dataset.features or {}).items():
        if isinstance(feature, Image) and feature.decode:
            dataset = dataset.cast_column(name, Image(decode=False))
    return dataset


def class_names(features: Mapping[str, object] | None) -> dict[str, tuple[str, ...]]:
    """The names of each `ClassLabel` feature's classes, by feature, which a
    configured caption reads in place of the label's number."""
    from datasets import ClassLabel

    return {name: tuple(feature.names) for name, feature in (features or {}).items()
            if isinstance(feature, ClassLabel)}


class _Streamed(grain.IterDataset):
    """An `IterableDataset` as a Grain stream, its position its `state_dict`.
    Each opening shards its own dataset, also before a row-wise node split:
    an IterableDataset keeps the position of the iterator it made last, and
    the position `load_state_dict` gave it for every later one."""

    def __init__(self, dataset: datasets.IterableDataset, shard: _HFShard):
        super().__init__()
        self._dataset = dataset
        self._shard = shard

    def __iter__(self) -> grain.DatasetIterator:
        shard = self._shard
        if shard.count <= self._dataset.n_shards:
            dataset = self._dataset.shard(num_shards=shard.count, index=shard.index, contiguous=True)
        else:
            from datasets.distributed import split_dataset_by_node

            dataset = self._dataset.shard(num_shards=1, index=0, contiguous=True)
            dataset = split_dataset_by_node(dataset, rank=shard.index, world_size=shard.count)
        # The iterator bridge reads it (`iterator._StreamIterator`), its position the dataset's own.
        return _StreamIterator(functools.partial(_Rows, dataset), shard.id, state=_Rows.state,
                               load_state=_Rows.resume)


class _Rows(Iterator[Batch]):
    """An `IterableDataset`'s rows, read from its iterator, made on the first read so that a position
    `resume` loads first is where it starts."""

    def __init__(self, dataset: datasets.IterableDataset):
        self._dataset = dataset
        self._rows: Iterator[Batch] | None = None
        # The position resumed from, which the dataset reports as its own only once it reads.
        self._resumed: Value | None = None

    def __next__(self) -> Batch:
        if self._rows is None:
            self._rows = iter(self._dataset)
        return next(self._rows)

    def state(self) -> Value:
        """Where the rows are: the position they resumed from until they read, which an
        `IterableDataset` reports as its own only once its iterator starts."""
        if self._rows is None and self._resumed is not None:
            return self._resumed
        return self._dataset.state_dict()

    def resume(self, position: Value) -> None:
        self._dataset.load_state_dict(dict(record(position, "a Hugging Face stream's state_dict")))
        self._resumed = position

    def close(self) -> None:
        if isinstance(self._rows, Closeable):
            self._rows.close()


def from_hf(dataset: datasets.Dataset | datasets.IterableDataset, name: str | None = None) -> HF:
    """A Hugging Face `datasets.Dataset` or `IterableDataset` as a Dew
    dataset (`HF`): an `IterableDataset` named, unshuffled."""
    return HF(dataset, name)


def _buffer_shuffled(layer: _BaseExamplesIterable) -> bool:
    """At the Hugging Face boundary: whether `layer`, an IterableDataset's
    examples, or any layer it reads (a map's one, an interleave's several) is
    the shuffle buffer, whose rows and draws HF's `state_dict` leaves out."""
    from datasets.iterable_dataset import BufferShuffledExamplesIterable

    if isinstance(layer, BufferShuffledExamplesIterable):
        return True
    read = [getattr(layer, "ex_iterable", None), *getattr(layer, "ex_iterables", ())]
    return any(_buffer_shuffled(inner) for inner in read if inner is not None)
