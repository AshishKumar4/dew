"""Several stores read one after another as one store."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence

import grain.python as grain
import numpy as np

from dew._records import integer
from dew.data.dataset import Records
from dew.data.pipeline import Dataset
from dew.data.sources.base import (
    _COUNTS,
    Read,
    Shard,
    ShardManifest,
    Source,
    _consumed,
    stated,
    tokenizer_contract,
)
from dew.data.steps.text import stated_rules

COUNTED = (*_COUNTS, "held_out")
"""The facts that count a store's records, which a concatenation's sources add."""


@dataclasses.dataclass(frozen=True)
class Concatenate(Dataset):
    """The records of `sources` one store after another, as one store: two splits of one Hub
    dataset read as one (`Concatenate((hub.with_split("train"), hub.with_split("validation")))`),
    or stores of any kinds. A record's position is its place in that order, every source's shards
    are its shards, and its identity is its sources' in order, so a run resumes on it as on any
    store and a changed source refuses the resume as it would alone. It is a store wherever one is
    read; `concat` joins stores as a dataset."""

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
        """What every source states of its records, as one store's: the counts (`COUNTED`) added
        where each source states one, the tokenizer contract where the sources that state it agree
        (`tokenizer_contract`, which refuses two), the rules any part's rows were counted by
        where every part states its own (`count_rules`), and any other fact, class names among
        them, where every source states the one value."""
        return self._merged([source.metadata() for source in self.sources])

    def stated(self) -> Mapping[str, object]:
        """What every part states of its records without reading them, merged as `metadata` is."""
        return self._merged([stated(source) for source in self.sources])

    def _merged(self, stated: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
        contract = tokenizer_contract((source.describe(), facts)
                                      for source, facts in zip(self.sources, stated, strict=True))
        facts: dict[str, object] = dict(contract)
        for key in dict.fromkeys(key for given in stated for key in given):
            values = [given[key] for given in stated if key in given]
            if key == "count_rules":
                # The rules any part's rows were counted by, where every part states its own.
                if len(values) == len(stated):
                    rules = {rule for value in values for rule in stated_rules(value)}
                    facts[key] = [list(rule) for rule in sorted(rules, key=repr)]
            elif key in COUNTED:
                if len(values) == len(stated) and all(isinstance(value, int) for value in values):
                    facts[key] = sum(integer(value, key) for value in values)
            elif key not in contract and len(values) == len(stated) and all(
                    value == values[0] for value in values):
                facts[key] = values[0]
        return facts

    def lengths(self) -> np.ndarray | None:
        lengths = [source.lengths() for source in self.sources]
        measured = [length for length in lengths if length is not None]
        return np.concatenate(measured) if len(measured) == len(lengths) else None

    def manifest(self, version: int) -> ShardManifest:
        """Each store's shards at `version`, one store after another: their index order where each
        store's is, as `records` joins them; their rows counted (`ShardManifest.sized`) where every
        store's are stated or counted."""
        manifests = tuple(source.manifest(version) for source in self.sources)
        shards = {f"{index}/{shard.id}": (manifest, shard)
                  for index, manifest in enumerate(manifests) for shard in manifest.shards}
        ordered = all(manifest.index_ordered for manifest in manifests)
        countable = any(manifest.counted is not None for manifest in manifests) and all(
            manifest.counted is not None or all(shard.rows is not None for shard in manifest.shards)
            for manifest in manifests)
        return ShardManifest(tuple(dataclasses.replace(shard, id=identifier)
                                   for identifier, (_, shard) in shards.items()),
                             functools.partial(_read_joined, shards), index_ordered=ordered,
                             counted=functools.partial(_counted, manifests) if countable else None)


def _read_joined(shards: Mapping[str, tuple[ShardManifest, Shard]], shard: Shard,
                 options: grain.ReadOptions | None = None) -> grain.IterDataset:
    """Shard `shard` of a concatenation, its store's own (`shards`, by the concatenation's ids) read by
    its store's reader."""
    manifest, original = shards[shard.id]
    return manifest.reader(original, options)


def _counted(manifests: Sequence[ShardManifest], reads: int) -> tuple[int, ...]:
    """The rows of each shard of the stores `manifests` name, one store after another, each store's as it
    counts them, `reads` at once (`ShardManifest.sized`)."""
    return tuple(int(shard.rows or 0) for manifest in manifests for shard in manifest.sized(reads).shards)
