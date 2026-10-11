"""Immutable record pipelines, lowered to Grain by :func:`compile`."""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
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


type Record = Batch | bytes | Rejected | None


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
class SingleReader(Protocol):
    single_reader: bool


@dataclasses.dataclass(frozen=True)
class TokenField:
    name: str
    extra: int = 0
    side: Literal["left", "right"] = "right"
    pad: int | bool = 0


@runtime_checkable
class TokenGeometry(Protocol):
    @property
    def max_length(self) -> int: ...
    @property
    def token_fields(self) -> tuple[TokenField, ...]: ...
    @property
    def length_field(self) -> str: ...


@dataclasses.dataclass(frozen=True)
class Geometry:
    max_length: int
    token_fields: tuple[TokenField, ...]
    length_field: str


@dataclasses.dataclass(frozen=True)
class TokenBudget:
    tokens: int
    lengths: tuple[int, ...] = ()
    window: int = 8
    geometry: Geometry | None = None
    drop_remainder: bool = True

    def __post_init__(self) -> None:
        if self.tokens <= 0 or self.window <= 0:
            raise ValueError("token budget and grouping window must be positive")
        if self.lengths and (
            len(self.lengths) > 6
            or tuple(sorted(set(self.lengths))) != self.lengths
            or any(length <= 0 or self.tokens % length for length in self.lengths)
        ):
            raise ValueError(
                "token budget lengths need at most six increasing positive divisors of the budget"
            )


@runtime_checkable
class Expansion(Protocol):
    def flat_map(self, element: Batch, /) -> Sequence[Batch | Rejected]: ...


@dataclasses.dataclass(frozen=True)
class Configured:
    """Resolve one record step from source facts before any record reader opens."""

    factory: Callable[[Mapping[str, object]], RecordStep | pygrain.RandomMapTransform | Expansion]
    needs: tuple[str, ...]
    scope: Literal["source", "run"] = "source"
    random: bool = False


@dataclasses.dataclass(frozen=True)
class Map:
    step: RecordStep
    identity: str | None = None
    rejects: bool = False
    configured: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RandomMap:
    step: RandomStep
    identity: str | None = None
    rejects: bool = False
    configured: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MapWithIndex:
    step: Callable[[int, Batch], Batch | Rejected]
    identity: str | None = None
    rejects: bool = False


@dataclasses.dataclass(frozen=True)
class Filter:
    predicate: Callable[[Batch], bool]
    reason: str = "filtered"
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class FlatMap:
    step: Expansion
    identity: str | None = None
    configured: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Shuffle:
    seed: int
    window: int | None = None
    block_shards: int | None = None


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
class TokenSplit:
    fraction: float
    held: bool
    fields: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        if not 0 <= self.fraction < 1:
            raise ValueError("token holdout fraction must be in [0, 1)")


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


type Stage = (
    Map
    | RandomMap
    | MapWithIndex
    | Filter
    | FlatMap
    | Shuffle
    | Repeat
    | Slice
    | ShardByHost
    | Interleave
    | BatchStage
    | Pack
    | Rebatch
    | Prefetch
    | Workers
    | Limit
    | Group
    | MapBatch
    | Artifact
    | Windows
    | Configured
    | TokenSplit
    | TokenBudget
)


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
class Phases:
    pipelines: tuple[Pipeline, ...]
    until_steps: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if (
            not self.pipelines
            or len(self.pipelines) != len(self.until_steps)
            or self.until_steps[-1] is not None
        ):
            raise ValueError("phases need one boundary per pipeline; the last boundary is None")
        previous = 0
        for boundary in self.until_steps[:-1]:
            if boundary is None or boundary <= previous:
                raise ValueError("phase step boundaries must increase from above zero")
            previous = boundary


@dataclasses.dataclass(frozen=True)
class Pipeline:
    source: Source | Mixture | Phases
    stages: tuple[Stage, ...] = ()
    name: str | None = None

    def then(self, *stages: Stage) -> Pipeline:
        return dataclasses.replace(self, stages=(*self.stages, *stages))

    def map(
        self, *steps: RecordStep | Configured, identity: str | None = None, rejects: bool | None = None
    ) -> Pipeline:
        for step in steps:
            if isinstance(step, pygrain.RandomMapTransform):
                raise ValueError(
                    f"stochastic step {type(step).__name__} needs random_map after the cache prefix"
                )
        return self.then(
            *(
                step
                if isinstance(step, Configured)
                else Map(step, identity, _rejects(step) if rejects is None else rejects)
                for step in steps
            )
        )

    def random_map(self, *steps: RandomStep, identity: str | None = None) -> Pipeline:
        return self.then(*(RandomMap(step, identity, _rejects(step)) for step in steps))

    def map_with_index(
        self, step: Callable[[int, Batch], Batch | Rejected], *, identity: str | None = None
    ) -> Pipeline:
        return self.then(MapWithIndex(step, identity, _rejects(step)))

    def filter(
        self, predicate: Callable[[Batch], bool], *, reason: str = "filtered", identity: str | None = None
    ) -> Pipeline:
        return self.then(Filter(predicate, reason, identity))

    def flat_map(self, step: Expansion, *, identity: str | None = None) -> Pipeline:
        return self.then(FlatMap(step, identity))

    def shuffle(
        self, seed: int = 0, *, window: int | None = None, block_shards: int | None = None
    ) -> Pipeline:
        if window is not None and block_shards is not None:
            raise ValueError("shuffle uses a window or physical shard blocks, not both")
        if block_shards is not None and block_shards < 1:
            raise ValueError("block_shards must be positive")
        return self.then(Shuffle(seed, window, block_shards))

    def repeat(self, epochs: int | None = None) -> Pipeline:
        return self.then(Repeat(epochs))

    def slice(self, selection: slice) -> Pipeline:
        return self.then(Slice(selection))

    def shard_by_host(self) -> Pipeline:
        return self.then(ShardByHost())

    def mix(
        self,
        *others: Pipeline,
        weights: Sequence[float] | None = None,
        tokens: Callable[[Batch], int] | None = None,
    ) -> Pipeline:
        pipelines = (self, *others)
        return Pipeline(
            Mixture(pipelines, tuple(weights) if weights is not None else (1.0,) * len(pipelines), tokens)
        )

    def interleave(self, cycle_length: int = 16) -> Pipeline:
        return self.then(Interleave(cycle_length))

    def batch(
        self,
        size: int | None = None,
        *,
        tokens: int | None = None,
        lengths: tuple[int, ...] = (),
        drop_remainder: bool = True,
        batch_fn: Callable[[Sequence[Batch]], Batch] | None = None,
    ) -> Pipeline:
        if (size is None) == (tokens is None):
            raise ValueError("batch needs exactly one of row size or tokens")
        if tokens is not None:
            if batch_fn is not None:
                raise ValueError("token batching owns its variable-size stack; batch_fn needs row batching")
            return self.then(TokenBudget(tokens, lengths, drop_remainder=drop_remainder))
        if size is None or size < 1:
            raise ValueError("batch size must be positive")
        return self.then(BatchStage(size, drop_remainder, batch_fn))

    def pack(
        self,
        seq_len: int,
        *,
        fields: tuple[str, ...] = ("text",),
        bins: int = 32,
        policy: Literal["first-fit", "best-fit", "concat"] = "first-fit",
    ) -> Pipeline:
        return self.then(Pack(seq_len, fields, bins, policy))

    def windows(
        self, seq_len: int, *, stride: int | None = None, fields: tuple[str, ...] = ("text",)
    ) -> Pipeline:
        return self.then(Windows(seq_len, stride, fields))

    def token_split(self, fraction: float, *, held: bool, fields: tuple[str, ...] = ("text",)) -> Pipeline:
        return self.then(TokenSplit(fraction, held, fields))

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
    train: tuple[RecordStep | pygrain.RandomMapTransform | Expansion | Configured, ...] = ()
    evaluate: tuple[RecordStep | Expansion | Configured, ...] = ()
    group: LengthGroups | None = None
    train_batch: tuple[HostStep, ...] = ()
    evaluate_batch: tuple[HostStep, ...] = ()


def _rejects(step: RecordStep | RandomStep | Callable[[int, Batch], Batch | Rejected]) -> bool:
    if isinstance(step, Rejecting):
        return step.rejects
    if isinstance(step, pygrain.MapTransform):
        method = step.map
    elif isinstance(step, pygrain.RandomMapTransform):
        method = step.random_map
    else:
        method = step
    returned = inspect.signature(method).return_annotation
    return "Rejected" in str(returned) or (
        returned is inspect.Signature.empty and not isinstance(step, pygrain.RandomMapTransform)
    )
