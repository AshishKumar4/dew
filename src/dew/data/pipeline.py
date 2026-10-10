"""Immutable record pipelines, lowered to Grain by :func:`compile`."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import grain.python as pygrain
import numpy as np

from dew.data.dataset import Batch, LengthGroups
from dew.data.sources.base import Source

if TYPE_CHECKING:
    from dew.data.cache import Cache, Generation


@dataclasses.dataclass(frozen=True)
class Rejected:
    reason: str
    permanent: bool = True


type RecordStep = pygrain.MapTransform | Callable[[Batch], Batch | Rejected]
type RandomStep = pygrain.RandomMapTransform | Callable[[Batch, np.random.Generator], Batch | Rejected]
type HostStep = Callable[[Batch], Batch] | pygrain.MapTransform


@runtime_checkable
class Rejecting(Protocol):
    rejects: bool


@runtime_checkable
class Volatile(Protocol):
    volatile: bool


@runtime_checkable
class Expansion(Protocol):
    def flat_map(self, element: Batch) -> Sequence[Batch | Rejected]: ...


@dataclasses.dataclass(frozen=True)
class Map:
    step: RecordStep
    identity: str | None = None
    rejects: bool = False


@dataclasses.dataclass(frozen=True)
class RandomMap:
    step: RandomStep
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class MapWithIndex:
    step: Callable[[int, Batch], Batch | Rejected]
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class Filter:
    predicate: Callable[[Batch], bool]
    reason: str = "filtered"
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class FlatMap:
    step: Expansion
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class Shuffle:
    seed: int
    window: int | None = None


@dataclasses.dataclass(frozen=True)
class Repeat:
    epochs: int | None = None


@dataclasses.dataclass(frozen=True)
class Slice:
    selection: slice


@dataclasses.dataclass(frozen=True)
class ShardByHost:
    pass


@dataclasses.dataclass(frozen=True)
class Interleave:
    cycle_length: int = 16


@dataclasses.dataclass(frozen=True)
class BatchStage:
    size: int
    drop_remainder: bool = True
    batch_fn: Callable[[Sequence[Batch]], Batch] | None = None


@dataclasses.dataclass(frozen=True)
class Pack:
    seq_len: int
    fields: tuple[str, ...] = ("text",)
    bins: int = 32
    policy: Literal["first-fit", "best-fit", "concat"] = "first-fit"


@dataclasses.dataclass(frozen=True)
class Windows:
    seq_len: int
    stride: int | None = None
    fields: tuple[str, ...] = ("text",)


@dataclasses.dataclass(frozen=True)
class Rebatch:
    size: int


@dataclasses.dataclass(frozen=True)
class Prefetch:
    buffer: int


@dataclasses.dataclass(frozen=True)
class Workers:
    count: int
    buffer: int = 2


@dataclasses.dataclass(frozen=True)
class Limit:
    count: int


@dataclasses.dataclass(frozen=True)
class Group:
    groups: LengthGroups


@dataclasses.dataclass(frozen=True)
class MapBatch:
    step: HostStep
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class Artifact:
    cache: Cache | Generation


type Stage = (Map | RandomMap | MapWithIndex | Filter | FlatMap | Shuffle | Repeat | Slice
              | ShardByHost | Interleave | BatchStage | Pack | Rebatch | Prefetch | Workers
              | Limit | Group | MapBatch | Artifact | Windows)


@dataclasses.dataclass(frozen=True)
class Mixture:
    pipelines: tuple[Pipeline, ...]
    weights: tuple[float, ...]
    tokens: Callable[[Batch], int] | None = None

    def __post_init__(self) -> None:
        if len(self.pipelines) != len(self.weights) or len(self.pipelines) < 2:
            raise ValueError("mix needs at least two pipelines and one weight per pipeline")
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights):
            raise ValueError("mix weights must be finite and positive")


@dataclasses.dataclass(frozen=True)
class Pipeline:
    source: Source | Mixture
    stages: tuple[Stage, ...] = ()
    name: str | None = None

    def then(self, *stages: Stage) -> Pipeline:
        return dataclasses.replace(self, stages=(*self.stages, *stages))

    def map(self, *steps: RecordStep, identity: str | None = None, rejects: bool = False) -> Pipeline:
        return self.then(*(Map(step, identity, rejects or isinstance(step, Rejecting) and step.rejects)
                           for step in steps))

    def random_map(self, *steps: RandomStep, identity: str | None = None) -> Pipeline:
        return self.then(*(RandomMap(step, identity) for step in steps))

    def map_with_index(self, step: Callable[[int, Batch], Batch | Rejected], *,
                       identity: str | None = None) -> Pipeline:
        return self.then(MapWithIndex(step, identity))

    def filter(self, predicate: Callable[[Batch], bool], *, reason: str = "filtered",
               identity: str | None = None) -> Pipeline:
        return self.then(Filter(predicate, reason, identity))

    def flat_map(self, step: Expansion, *, identity: str | None = None) -> Pipeline:
        return self.then(FlatMap(step, identity))

    def shuffle(self, seed: int = 0, *, window: int | None = None) -> Pipeline:
        return self.then(Shuffle(seed, window))

    def repeat(self, epochs: int | None = None) -> Pipeline:
        return self.then(Repeat(epochs))

    def slice(self, selection: slice) -> Pipeline:
        return self.then(Slice(selection))

    def shard_by_host(self) -> Pipeline:
        return self.then(ShardByHost())

    def mix(self, *others: Pipeline, weights: Sequence[float] | None = None,
            tokens: Callable[[Batch], int] | None = None) -> Pipeline:
        pipelines = (self, *others)
        return Pipeline(Mixture(pipelines, tuple(weights) if weights is not None else (1.,) * len(pipelines),
                                tokens))

    def interleave(self, cycle_length: int = 16) -> Pipeline:
        return self.then(Interleave(cycle_length))

    def batch(self, size: int, *, drop_remainder: bool = True,
              batch_fn: Callable[[Sequence[Batch]], Batch] | None = None) -> Pipeline:
        if size < 1:
            raise ValueError("batch size must be positive")
        return self.then(BatchStage(size, drop_remainder, batch_fn))

    def pack(self, seq_len: int, *, fields: tuple[str, ...] = ("text",), bins: int = 32,
             policy: Literal["first-fit", "best-fit", "concat"] = "first-fit") -> Pipeline:
        return self.then(Pack(seq_len, fields, bins, policy))

    def windows(self, seq_len: int, *, stride: int | None = None,
                fields: tuple[str, ...] = ("text",)) -> Pipeline:
        return self.then(Windows(seq_len, stride, fields))

    def rebatch(self, size: int) -> Pipeline:
        return self.then(Rebatch(size))

    def prefetch(self, buffer: int = 64) -> Pipeline:
        return self.then(Prefetch(buffer))

    def workers(self, count: int, *, buffer: int = 2) -> Pipeline:
        return self.then(Workers(count, buffer))

    def take(self, count: int) -> Pipeline:
        return self.then(Limit(count))

    def limit(self, count: int) -> Pipeline:
        return self.take(count)

    def cache(self, cache: str | Cache | Generation) -> Pipeline:
        from dew.data.cache import Cache

        return self.then(Artifact(Cache(cache) if isinstance(cache, str) else cache))

    def group(self, length: Callable[[Batch], int], *, window: int = 64) -> Pipeline:
        return self.then(Group(LengthGroups(length, window)))

    def map_batch(self, *steps: HostStep, identity: str | None = None) -> Pipeline:
        return self.then(*(MapBatch(step, identity) for step in steps))


@dataclasses.dataclass(frozen=True)
class ObjectiveData:
    train: tuple[RecordStep | RandomStep | Expansion, ...] = ()
    evaluate: tuple[RecordStep | Expansion, ...] = ()
    group: LengthGroups | None = None
    train_batch: tuple[HostStep, ...] = ()
    evaluate_batch: tuple[HostStep, ...] = ()
