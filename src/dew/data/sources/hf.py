"""A Hugging Face `datasets` dataset read as a Dew store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch, Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest
from .memory import _record


@dataclass(frozen=True, eq=False)
class HF(Dataset):
    """A `datasets.Dataset`, read by index from its Arrow table, or an
    `IterableDataset`, read as one stream whose position its own
    `state_dict` checkpoints; the `.map()`s it holds are already applied. An
    image column reads as its encoded bytes, which `images.decode` decodes.
    `name` is its identity in a run's plan, a cache's key and its logs, its
    fingerprint by default."""

    dataset: object
    name: str | None = None

    @property
    def streamed(self) -> bool:
        from datasets import IterableDataset

        return isinstance(self.dataset, IterableDataset)

    def describe(self) -> str:
        return f"hf:{self.name or getattr(self.dataset, '_fingerprint', None) or type(self.dataset).__name__}"

    def snapshot(self) -> Mapping[str, object]:
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
    """An `IterableDataset` as a Grain stream, its position its `state_dict`."""

    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    def __iter__(self) -> grain.DatasetIterator:
        return _StreamedIterator(self._dataset)


class _StreamedIterator(grain.DatasetIterator):
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset
        self._rows = iter(dataset)

    def __next__(self) -> Batch:
        return _record(next(self._rows))

    def get_state(self) -> dict:
        return self._dataset.state_dict()

    def set_state(self, state: dict) -> None:
        self._dataset.load_state_dict(state)
        self._rows = iter(self._dataset)


def from_hf(dataset: object, name: str | None = None) -> HF:
    """A Hugging Face `datasets.Dataset` or `IterableDataset`, its `.map()`s
    already applied, as a Dew dataset (`HF`)."""
    return HF(dataset, name)
