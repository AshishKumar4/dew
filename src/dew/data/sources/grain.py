"""A Grain dataset read as a Dew store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import grain.python as grain
import numpy as np

from dew.data.dataset import Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest


@dataclass(frozen=True, eq=False)
class Grain(Dataset):
    """A Grain `MapDataset`, read by index, or `IterDataset`, read as one
    stream whose position Grain checkpoints. Grain states no store of its own,
    so `name` is its identity in a run's plan, a cache's key and its logs."""

    dataset: grain.MapDataset | grain.IterDataset
    name: str

    def describe(self) -> str:
        return f"grain:{self.name}"

    def snapshot(self) -> Mapping[str, object]:
        return {"store": "grain", "name": self.name, **self.metadata()}

    def supports(self, read: Read) -> bool:
        return (read == "streamed") == isinstance(self.dataset, grain.IterDataset)

    def records(self, read: Read) -> Records:
        if not isinstance(self.dataset, grain.MapDataset):
            raise ValueError(f"{self.describe()} is a Grain IterDataset, read only as a stream")
        return self.dataset

    def manifest(self) -> ShardManifest:
        if isinstance(self.dataset, grain.MapDataset):
            rows = len(self.dataset)
            return ShardManifest((Shard(self.describe(), self.describe(), 0, rows, rows),), self._read_shard)
        return ShardManifest((Shard(self.describe(), self.describe()),), self._read_shard)

    def _read_shard(self, shard: Shard) -> grain.IterDataset:
        if isinstance(self.dataset, grain.MapDataset):
            return self.dataset.to_iter_dataset()
        return self.dataset

    def metadata(self) -> Mapping[str, object]:
        return {"rows": len(self.dataset)} if isinstance(self.dataset, grain.MapDataset) else {}

    def lengths(self) -> np.ndarray | None:
        return None


def from_grain(dataset: grain.MapDataset | grain.IterDataset, name: str) -> Grain:
    """`dataset` as a Dew dataset under `name` (`Grain`)."""
    return Grain(dataset, name)
