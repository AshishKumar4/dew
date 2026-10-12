"""Immutable record pipelines, lowered to Grain by :func:`compile`."""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Concatenate, Literal, Protocol, runtime_checkable

import grain.python as pygrain
import numpy as np

from dew.data.dataset import Batch, LengthGroups

if TYPE_CHECKING:
    from dew.data.cache import Cache, Generation
    from dew.data.dataset import Loading
    from dew.data.sources.base import Read, Source


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
class Bound:
    """The configured step a record step was resolved from, with the
    metadata values its factory read."""

    step: Configured
    values: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class Batched:
    """A function of `size` records at a time, which returns any number of
    records, as Hugging Face's `map(batched=True)`; the last group of a
    finite pass may be smaller."""

    step: Callable[[list[Batch]], Sequence[Batch]]
    size: int
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class Native:
    """A function of the Grain dataset the stages before it make, for a Grain
    operation Dew does not name; the rows after it are read per share."""

    function: Callable[[pygrain.MapDataset | pygrain.IterDataset], pygrain.MapDataset | pygrain.IterDataset]
    identity: str | None = None


@dataclasses.dataclass(frozen=True)
class Map:
    step: RecordStep
    identity: str | None = None
    rejects: bool = False
    configured: Bound | None = None


@dataclasses.dataclass(frozen=True)
class RandomMap:
    step: RandomStep
    identity: str | None = None
    rejects: bool = False
    configured: Bound | None = None


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
    configured: Bound | None = None


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
    | Native
    | Batched
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


class Dataset:
    """Records a run reads: a store (`Files`, `Hub`, `TFDS`, `Memory`,
    `Torch`), or a store with the steps the verbs added, in the order they
    are written. A dataset never changes; every verb returns a new one.
    `Trainer.fit` and `iterate` lower it to Grain.

    The verb shows the cardinality: `map` is one record to one, `random_map`
    one to one with a generator, `filter` one to none or one, `flat_map`
    one to many; dataset-level stages, such as `text.pack`, are functions
    of a dataset applied with `pipe`. The compiler takes each stage where it
    stands or refuses it before a reader opens.
    """

    def then(self, *stages: Stage) -> Pipeline:
        """This dataset with `stages` after its own."""
        return Pipeline(self, stages)

    def holdout(self, fraction: float | None = None, *, key: str | None = None, records: int | None = None,
                split: str | None = None, **named: float) -> tuple[Dataset, ...]:
        """This dataset split into the records a run trains on and the ones it
        scores, with the steps it has replayed on each, so hold out before
        preparing.

        `holdout(0.01)` returns (train, held), and `holdout(val=0.0005,
        test=0.0005)` the remainder and one dataset per name, in order: a
        record goes to the part its identity hashes into, the same in every
        run, on every machine and after every resume. `key` names the field
        that identifies a document (an id or a url), so its versions and
        chunks land together; without one, a record's identity is its whole
        content. `records=n` holds out the first n records, and
        `split="validation"` the store's own split.
        """
        from dew.data.holdout import holdout

        return holdout(self, fraction, key=key, records=records, split=split, **named)

    def named(self, name: str) -> Pipeline:
        """This dataset under `name`, which its counts log under in a mix."""
        chain = self if isinstance(self, Pipeline) else Pipeline(self)
        return dataclasses.replace(chain, name=name)

    def pipe[**Options](self, stages: Callable[Concatenate[Dataset, Options], Dataset],
                        *args: Options.args, **kwargs: Options.kwargs) -> Dataset:
        """`stages(self, *args, **kwargs)`: a dataset-level stage, such as `text.pack`."""
        return stages(self, *args, **kwargs)

    def map(self, *steps: RecordStep | Configured, identity: str | None = None) -> Pipeline:
        """One record to one: each step returns its record's replacement. A
        step that drops records (returns `Rejected`) goes through `filter_map`."""
        for step in steps:
            if isinstance(step, pygrain.RandomMapTransform):
                raise TypeError(f"{_named_step(step)} draws from a generator; apply it with random_map")
            if not isinstance(step, Configured) and _declares_rejects(step):
                raise TypeError(f"{_named_step(step)} drops records; apply it with filter_map")
        return self.then(*(step if isinstance(step, Configured) else Map(step, identity) for step in steps))

    def random_map(self, *steps: RandomStep, identity: str | None = None) -> Pipeline:
        """One record to one, each step drawing from the record's generator
        (`fn(record, rng)` or a `grain.RandomMapTransform`), the same draws on
        every read and resume."""
        for step in steps:
            if _declares_rejects(step):
                raise TypeError(f"{_named_step(step)} drops records; apply it with random_filter_map")
        return self.then(*(RandomMap(step, identity) for step in steps))

    def filter_map(self, *steps: RecordStep, identity: str | None = None) -> Pipeline:
        """One record to none or one: each step returns its record's
        replacement or `Rejected(reason)`, which drops it, counted by reason
        in the reader's checkpointed counts, as `images.fetch` and
        `images.decode` do."""
        for step in steps:
            if isinstance(step, pygrain.RandomMapTransform):
                raise TypeError(f"{_named_step(step)} draws from a generator; use random_filter_map")
        return self.then(*(Map(step, identity, rejects=True) for step in steps))

    def random_filter_map(self, *steps: RandomStep, identity: str | None = None) -> Pipeline:
        """`filter_map` for steps that draw from the record's generator
        (`fn(record, rng)` or a `grain.RandomMapTransform`), as
        `video.decode_clip` picks a random start and drops a clip that fails."""
        return self.then(*(RandomMap(step, identity, rejects=True) for step in steps))

    def map_with_index(
        self, step: Callable[[int, Batch], Batch | Rejected], *, identity: str | None = None
    ) -> Pipeline:
        """Map each row with its position in its source, as the read numbers it.

        That is the index of an indexed source, the rows of the earlier shards
        plus the row's place in its own for a streamed source, and a cache's
        input index. A filter keeps it, so a rejected row leaves a gap, a
        flat_map's outputs share their input's, and shuffles, repeats and
        slices carry it with the row, so it is the same in every read, under
        a cache, on resume and at any share or worker count. A row built from
        several (after pack, windows, group or batch) or a mixture of corpora
        has no single position, so the step is refused there.
        """
        return self.then(MapWithIndex(step, identity, _rejects(step)))

    def filter(
        self, predicate: Callable[[Batch], bool], *, reason: str | None = None, identity: str | None = None
    ) -> Pipeline:
        """Keep the records `predicate` accepts; the rest are counted under
        `reason`, by default the predicate's own (`images.min_side`'s "too_small")."""
        return self.then(Filter(predicate, reason or getattr(predicate, "reason", "filtered"), identity))

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
        """Keep the first `fraction` of the order's tokens (`held=True`) or the
        rest, cutting at most one document in two, so a held branch and a
        trained branch over one order are disjoint and together the corpus.
        It reads a finite indexed order, before any repeat."""
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
        """Fill `cache` with the deterministic record steps before it, or read a
        published `Generation` as the source in their place, with its published
        packing in place of a `pack` that follows."""
        from dew.data.cache import Cache

        return self.then(Artifact(Cache(cache) if isinstance(cache, str) else cache))

    def group(self, length: Callable[[Batch], int], *, window: int = 64) -> Pipeline:
        return self.then(Group(LengthGroups(length, window)))

    def map_batches(self, step: HostStep | Callable[[list[Batch]], Sequence[Batch]], *,
                    size: int | None = None, identity: str | None = None) -> Pipeline:
        """With `size`, `step` reads a list of `size` records and returns a list
        of records, any number, as Hugging Face's `map(batched=True)` (a batched
        tokenizer, a filter that looks at neighbours), and the chain goes on
        record by record: `pack`, `mix` and `batch` still follow. Without
        `size`, after `batch`, `step` runs on each training batch on the host."""
        if size is None:
            return self.then(MapBatch(step, identity))
        if size < 1:
            raise ValueError(f"map_batches reads groups of at least one record, not {size}")
        return self.then(Batched(step, size, identity))

    def snapshot(self, url: str, *, format: Literal["array_record", "parquet"] = "array_record",
                 rows_per_shard: int = 100_000) -> Mapping[str, object]:
        """Run this dataset once over every record and seal what it makes at
        `url`: immutable shards and their committed manifest, which
        `Files(url)` reads back. A fetch's outcomes are sealed with it, so a
        run that reads the snapshot resumes exactly. Returns the manifest."""
        from dew.data.sinks import export

        chain = self if isinstance(self, Pipeline) else Pipeline(self)
        return export(chain, url, format=format, rows_per_shard=rows_per_shard)

    def rename(self, **names: str) -> Pipeline:
        """Each record with its field `old` under the name `new`, as `rename(caption="txt")`."""
        return self.map(_Renamed(tuple(names.items())), identity=f"dew.rename:{sorted(names.items())}")

    def keep(self, *fields: str) -> Pipeline:
        """Each record with `fields` alone."""
        return self.map(_Kept(fields), identity=f"dew.keep:{list(fields)}")

    def pipe_grain(self, function: Callable[[pygrain.MapDataset | pygrain.IterDataset],
                                            pygrain.MapDataset | pygrain.IterDataset],
                   *, identity: str | None = None) -> Pipeline:
        """`function` of the Grain dataset this chain makes, for a Grain
        operation Dew does not name (`zip`, Grain's own interleave,
        `FlatMapMapDataset`). The chain after it is read per share and
        checkpoints through Grain's own state; `identity` names it in a run's
        plan, its code and closure by default."""
        return self.then(Native(function, identity))

    def stats(self, records: int = 1000) -> dict[str, float]:
        """The counts a run logs, from a fresh reader's first `records`
        records: attempted, accepted, and dropped by reason, per source."""
        from dew.data.compile import compile
        from dew.data.dataset import DataPartition

        chain = self if isinstance(self, Pipeline) else Pipeline(self)
        if any(isinstance(stage, (BatchStage, TokenBudget)) for stage in chain.stages):
            raise ValueError("stats counts records; take them before the batch")
        loader = compile(chain.repeat().batch(1), read=chain.mode, **_loading(chain))
        reader = loader.train(DataPartition())
        try:
            for _ in range(records):
                next(reader)
            return dict(loader.statistics())
        finally:
            reader.close()

    def read(self, mode: Read) -> Pipeline:
        """Read the stores this way, rather than as they prefer: "memory" holds
        every record and runs the deterministic steps once, "indexed" reads by
        index, "streamed" reads shards in order."""
        chain = self if isinstance(self, Pipeline) else Pipeline(self)
        return dataclasses.replace(chain, mode=mode)

    def threads(self, count: int) -> Pipeline:
        """Read with `count` threads a worker, rather than the measured default."""
        from dew.data.dataset import Loading

        chain = self if isinstance(self, Pipeline) else Pipeline(self)
        loading = chain.loading or Loading()
        return dataclasses.replace(chain, loading=dataclasses.replace(
            loading, threads=count, read_buffer=max(loading.read_buffer, count)))

    def iterate(self) -> Iterator[Batch]:
        """This dataset's records, or its batches where it batches, read as a
        run reads them: one pass, or endlessly after a `repeat`."""
        from dew.data.compile import validation
        from dew.data.dataset import DataPartition
        from dew.objectives.base import VALID_ROWS

        chain = self if isinstance(self, Pipeline) else Pipeline(self)
        batched = any(isinstance(stage, (BatchStage, TokenBudget)) for stage in chain.stages)
        reader = validation(chain if batched else chain.batch(1, drop_remainder=False), batch=1,
                            read=chain.mode, provision=True, **_loading(chain))(DataPartition())
        try:
            for batch in reader:
                yield batch if batched else {
                    name: value[0] for name, value in batch.items() if name != VALID_ROWS}
        finally:
            reader.close()

    def show(self, count: int = 8) -> None:
        """Print the first `count` records."""
        from rich.pretty import pprint

        for record in itertools.islice(self.iterate(), count):
            pprint({name: _shown(value) for name, value in record.items()} if isinstance(record, Mapping)
                   else record)


@dataclasses.dataclass(frozen=True)
class Pipeline(Dataset):
    """A dataset's store and the stages its verbs added; what every verb returns."""

    source: Source | Mixture | Phases
    stages: tuple[Stage, ...] = ()
    name: str | None = None
    mode: Read | None = None
    loading: Loading | None = None

    def then(self, *stages: Stage) -> Pipeline:
        return dataclasses.replace(self, stages=(*self.stages, *stages))


@dataclasses.dataclass(frozen=True)
class _Renamed:
    names: tuple[tuple[str, str], ...]

    def __call__(self, record: Batch) -> Batch:
        moved = {old for _, old in self.names}
        return {**{key: value for key, value in record.items() if key not in moved},
                **{new: record[old] for new, old in self.names}}


@dataclasses.dataclass(frozen=True)
class _Kept:
    fields: tuple[str, ...]

    def __call__(self, record: Batch) -> Batch:
        return {name: record[name] for name in self.fields}


def concat(datasets: Sequence[Dataset]) -> Pipeline:
    """The records of `datasets`, stores read one after another, as one dataset."""
    from dew.data.sources.concatenate import Concatenate

    if any(isinstance(dataset, Pipeline) for dataset in datasets):
        raise ValueError("concat joins stores; add the steps after it, or mix datasets that have steps")
    return Pipeline(Concatenate(tuple(datasets)))


def _loading(chain: Pipeline) -> dict[str, Loading]:
    return {} if chain.loading is None else {"loading": chain.loading}


def _shown(value: object) -> object:
    """A field as `show` prints it: an array as its shape and dtype, bytes as their length."""
    if isinstance(value, np.ndarray) and value.size > 16:
        return f"{value.dtype}{list(value.shape)}"
    if isinstance(value, bytes) and len(value) > 64:
        return f"<{len(value)} bytes>"
    return value


def mix(datasets: Sequence[Dataset], weights: Sequence[float] | None = None, *,
        by: Literal["records", "tokens"] = "records", field: str = "text") -> Pipeline:
    """`datasets` read together, each filling its share of a step, as Grain's
    `mix`: `weights` are the shares, equal by default, counted in records,
    or with `by="tokens"` in the valid target tokens of each member's `field`.
    Each member logs under its name, its store's by default (`Dataset.named`)."""
    tokens = None
    if by == "tokens":
        from dew.data.steps.text import ValidTargets

        tokens = ValidTargets(field)
    elif by != "records":
        raise ValueError(f"mix weighs records or tokens, not {by!r}")
    if not datasets:
        raise ValueError("mix needs at least one dataset")
    shares = (1.0,) * len(datasets) if weights is None else tuple(float(weight) for weight in weights)
    if len(shares) != len(datasets):
        raise ValueError(f"mix has {len(datasets)} datasets and {len(shares)} weights")
    chains = tuple(dataset if isinstance(dataset, Pipeline) else Pipeline(dataset) for dataset in datasets)
    return Pipeline(Mixture(chains, shares, tokens))


@dataclasses.dataclass(frozen=True)
class ObjectiveData:
    train: tuple[RecordStep | pygrain.RandomMapTransform | Expansion | Configured, ...] = ()
    evaluate: tuple[RecordStep | Expansion | Configured, ...] = ()
    group: LengthGroups | None = None
    train_batch: tuple[HostStep, ...] = ()
    evaluate_batch: tuple[HostStep, ...] = ()


def _named_step(step: object) -> str:
    return getattr(step, "__qualname__", None) or type(step).__name__


def _declares_rejects(step: object) -> bool:
    """Whether `step` says it drops records, with a `rejects` flag (`images.fetch`, `images.decode`)."""
    return isinstance(step, Rejecting) and bool(step.rejects)


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
