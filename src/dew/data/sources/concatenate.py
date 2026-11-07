"""Several stores read one after another as one store."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable, Mapping, Sequence

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
        ordered = all(manifest.index_ordered for manifest in manifests)
        countable = any(manifest.counted is not None for manifest in manifests) and all(
            manifest.counted is not None or all(shard.rows is not None for shard in manifest.shards)
            for manifest in manifests)
        divisible = [manifest for manifest in manifests if manifest.subdivide is not None]
        # Its parts are fixed where every store it cuts cuts its own so (a Parquet store's row groups): a read
        # by more readers than its shards cuts every store, past each one's shards.
        return ShardManifest(_joined(manifest.shards for manifest in manifests),
                             functools.partial(_read_joined, manifests), index_ordered=ordered,
                             subdivide=functools.partial(_subdivided, manifests) if divisible else None,
                             counted=functools.partial(_counted, manifests) if countable else None,
                             fixed_parts=bool(divisible) and all(manifest.fixed_parts
                                                                 for manifest in divisible))


def _joined(shards: Iterable[Iterable[Shard]]) -> tuple[Shard, ...]:
    """The shards of a concatenation's stores, one store's after another, each named by its store's
    place among them and its own name in it, which `_read_joined` reads it by."""
    return tuple(dataclasses.replace(shard, id=f"{index}/{shard.id}")
                 for index, held in enumerate(shards) for shard in held)


def _read_joined(manifests: Sequence[ShardManifest], shard: Shard,
                 options: grain.ReadOptions | None = None) -> grain.IterDataset:
    """Shard `shard` of a concatenation, its store's own or a part of one (`_subdivided`), read by its
    store's reader as the store names it."""
    index, _, own = shard.id.partition("/")
    return manifests[int(index)].reader(dataclasses.replace(shard, id=own), options)


def _subdivided(manifests: Sequence[ShardManifest], count: int) -> tuple[Shard, ...]:
    """A concatenation's shards for `count` readers and workers: each store's parts for them where it
    cuts its own (`ShardManifest.subdivide`), its shards where it cuts none, one store's after another."""
    return _joined(manifest.shards if manifest.subdivide is None or len(manifest.shards) >= count
                   else manifest.subdivide(count) for manifest in manifests)


def _counted(manifests: Sequence[ShardManifest], reads: int) -> tuple[int, ...]:
    """The rows of each shard of the stores `manifests` name, one store after another, each store's as it
    counts them, `reads` at once (`ShardManifest.sized`)."""
    return tuple(int(shard.rows or 0) for manifest in manifests for shard in manifest.sized(reads).shards)
