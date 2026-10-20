"""Several stores read one after another as one store."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import cached_property

import grain.python as grain
import numpy as np

from dew.data.dataset import Records
from dew.data.pipeline import Dataset
from dew.data.sources.base import Read, Shard, ShardManifest, Source, _consumed


@dataclasses.dataclass(frozen=True)
class Concatenate(Dataset):
    """The records of `sources` one store after another, as one store: two splits of one Hub
    dataset read as one (`Concatenate((hub.with_split("train"), hub.with_split("validation")))`),
    or stores of any kinds. A record's position is its place in that order, every source's shards
    are its shards, and its identity is its sources' in order, so a run resumes on it as on any
    store and a changed source refuses the resume as it would alone. It is a store wherever one is
    read, a recipe's source (`ImageData(source=...)`) among them; `concat` joins stores as a
    dataset."""

    sources: tuple[Source, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("concatenation needs at least one source")

    def label(self) -> str:
        return "+".join(dict.fromkeys(source.label() for source in self.sources))

    def describe(self) -> str:
        return "concatenate[" + ",".join(source.describe() for source in self.sources) + "]"

    def consumed(self, read: Read) -> Concatenate:
        return dataclasses.replace(self, sources=tuple(_consumed(source, read) for source in self.sources))

    def fingerprint(self) -> Mapping[str, object]:
        return {"concatenate": [source.fingerprint() for source in self.sources]}

    def supports(self, read: Read) -> bool:
        return all(source.supports(read) for source in self.sources)

    def records(self, read: Read) -> Records:
        return grain.MapDataset.concatenate([grain.MapDataset.source(source.records(read))
                                             for source in self.sources])

    def metadata(self) -> Mapping[str, object]:
        stated = [source.metadata() for source in self.sources]
        rows = [facts.get("rows") for facts in stated]
        return {"rows": sum(count for count in rows if isinstance(count, int))} if all(
            isinstance(count, int) for count in rows) else {}

    def lengths(self) -> np.ndarray | None:
        lengths = [source.lengths() for source in self.sources]
        measured = [length for length in lengths if length is not None]
        return np.concatenate(measured) if len(measured) == len(lengths) else None

    @cached_property
    def _shards(self) -> Mapping[str, tuple[ShardManifest, Shard]]:
        return {f"{index}/{shard.id}": (manifest, shard)
                for index, source in enumerate(self.sources)
                for manifest in (source.manifest(),) for shard in manifest.shards}

    def manifest(self) -> ShardManifest:
        return ShardManifest(tuple(dataclasses.replace(shard, id=identifier)
                                   for identifier, (_, shard) in self._shards.items()), self._reader)

    def _reader(self, shard: Shard) -> grain.IterDataset:
        manifest, original = self._shards[shard.id]
        return manifest.reader(original)
