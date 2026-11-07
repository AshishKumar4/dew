"""A Grain dataset read as a Dew store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import grain.python as grain
import numpy as np

from dew.data.dataset import Records
from dew.data.pipeline import Dataset

from .base import Read, Shard, ShardManifest, unnamed


@dataclass(frozen=True, eq=False)
class Grain(Dataset):
    """A Grain `MapDataset`, read by index or as one stream, or an
    `IterDataset`, read as one stream whose position Grain checkpoints. Grain
    states no store of its own, so `identity` declares its content version in a
    run's plan, a cache's key and its logs; with none, a run reads it and a
    resume needs `resume="non-exact"`, and no cache keys it (`Manifest.exact`)."""

    dataset: grain.MapDataset | grain.IterDataset
    identity: str | None = None
    place: int = field(default_factory=unnamed, repr=False)

    def __post_init__(self) -> None:
        if self.identity == "":
            raise ValueError("a Grain dataset's identity declares its content version; give one, or none")

    def label(self) -> str:
        return self.identity or "grain"

    def describe(self) -> str:
        return f"grain:{self.identity}" if self.identity is not None else f"grain:unnamed-{self.place}"

    def fingerprint(self) -> Mapping[str, object]:
        return {"store": "grain", "identity_required": True, "identity": self.identity, **self.metadata()}

    def supports(self, read: Read) -> bool:
        return read == "streamed" or isinstance(self.dataset, grain.MapDataset)

    def records(self, read: Read) -> Records:
        if not isinstance(self.dataset, grain.MapDataset):
            raise ValueError(f"{self.describe()} is a Grain IterDataset, read only as a stream")
        return self.dataset

    def manifest(self, version: int) -> ShardManifest:
        if isinstance(self.dataset, grain.MapDataset):
            rows = len(self.dataset)
            # One shard, the MapDataset's records in their index order.
            return ShardManifest((Shard(self.describe(), self.describe(), 0, rows, rows),), self._read_shard,
                                 index_ordered=True)
        return ShardManifest((Shard(self.describe(), self.describe()),), self._read_shard)

    def _read_shard(self, shard: Shard, options: grain.ReadOptions | None = None) -> grain.IterDataset:
        if isinstance(self.dataset, grain.MapDataset):
            return self.dataset.to_iter_dataset(options)
        return self.dataset

    def metadata(self) -> Mapping[str, object]:
        return {"rows": len(self.dataset)} if isinstance(self.dataset, grain.MapDataset) else {}

    def lengths(self) -> np.ndarray | None:
        return None


def from_grain(dataset: grain.MapDataset | grain.IterDataset, identity: str | None = None) -> Grain:
    """`dataset` as a Dew dataset, its content version `identity` where given (`Grain`)."""
    return Grain(dataset, identity)
