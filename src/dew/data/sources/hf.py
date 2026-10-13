"""A Hugging Face `datasets` dataset read as a Dew store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest
from .memory import _record

if TYPE_CHECKING:
    import datasets
    from datasets.iterable_dataset import _BaseExamplesIterable


@dataclass(frozen=True, eq=False)
class HF(Dataset):
    """A `datasets.Dataset`, read by index from its Arrow table, or an
    `IterableDataset`, read as one stream whose position its own
    `state_dict` checkpoints; the `.map()`s it holds are already applied. An
    image column reads as its encoded bytes, which `images.decode` decodes.
    `name` is its identity in a run's plan, a cache's key and its logs, its
    fingerprint by default; a stream has no fingerprint of its content, so
    it is named, with its version, or refused.

    A stream resumes exactly from its `state_dict` only unshuffled, since
    Hugging Face's shuffle buffer is not part of that state: one that
    shuffles through it is refused; shuffle with the chain's
    `shuffle(window=)` instead."""

    dataset: Any
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.streamed:
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
        return (read == "streamed") == self.streamed

    def _undecoded(self):
        from datasets import Image

        dataset = self.dataset
        for name, feature in (dataset.features or {}).items():
            if isinstance(feature, Image) and feature.decode:
                dataset = dataset.cast_column(name, Image(decode=False))
        return dataset

    def records(self, read: Read) -> Records:
        if self.streamed:
            raise ValueError(f"{self.describe()} is an IterableDataset, read only as a stream")
        return grain.MapDataset.source(self._undecoded()).map(_record)

    def manifest(self) -> ShardManifest:
        if not self.streamed:
            rows = len(self.dataset)
            return ShardManifest((Shard(self.describe(), self.describe(), 0, rows, rows),), self._read_shard)
        return ShardManifest((Shard(self.describe(), self.describe()),), self._read_shard)

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        if not self.streamed:
            return grain.MapDataset.source(self._undecoded()).map(_record).to_iter_dataset()
        return _Streamed(self._undecoded())

    def metadata(self) -> Mapping[str, object]:
        return {} if self.streamed else {"rows": len(self.dataset)}

    def lengths(self) -> np.ndarray | None:
        return None


class _Streamed(grain.IterDataset):
    """An `IterableDataset` as a Grain stream, its position its `state_dict`.
    Each opening reads its own dataset over the same examples (`shard(1, 0)`):
    an IterableDataset keeps the position of the iterator it made last, and
    the position `load_state_dict` gave it for every later one."""

    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    def __iter__(self) -> grain.DatasetIterator:
        return _StreamedIterator(self._dataset.shard(num_shards=1, index=0))


class _StreamedIterator(grain.DatasetIterator):
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset
        self._rows = iter(dataset)

    def __next__(self) -> Batch | bytes:
        row = None
        while row is None:
            row = _record(next(self._rows))
        return row

    def get_state(self) -> Mapping[str, object]:
        return self._dataset.state_dict()

    def set_state(self, state: Mapping[str, object]) -> None:
        self._dataset.load_state_dict(dict(state))
        self._rows = iter(self._dataset)


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
