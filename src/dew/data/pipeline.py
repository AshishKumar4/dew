"""Immutable record pipelines, lowered to Grain by :func:`compile`."""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import math
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Concatenate, Literal, Protocol, TypeGuard, runtime_checkable

import grain.python as pygrain
import numpy as np

from dew.data.dataset import Batch

if TYPE_CHECKING:
    import jax

    from dew.data.cache import Cache
    from dew.data.dataset import Loading
    from dew.data.generation import Generation
    from dew.data.sources.base import Read, Source


@dataclasses.dataclass(frozen=True)
class Rejected:
    reason: str
    permanent: bool = True


type Record = Batch | bytes | Rejected | None


type RecordStep = pygrain.MapTransform | Callable[[Batch], Batch | Rejected]
type RandomStep = pygrain.RandomMapTransform | Callable[[Batch, np.random.Generator], Batch | Rejected]
type HostStep = Callable[[Batch], Batch] | pygrain.MapTransform
type ColumnStep = Callable[[dict[str, list]], Mapping[str, Sequence]]
"""A function of records read as a dict of columns, each a list, as Hugging
Face's `map(batched=True)` reads them."""


@runtime_checkable
class Rejecting(Protocol):
    rejects: bool


@runtime_checkable
class Reasoned(Protocol):
    """A predicate that names what it drops (`images.min_side`'s "too_small")."""

    reason: str


@runtime_checkable
class Volatile(Protocol):
    volatile: bool


@runtime_checkable
class SingleReader(Protocol):
    single_reader: bool


@runtime_checkable
class TokenGeometry(Protocol):
    """A record step whose rows each hold `max_length` tokens, which a token
    budget divides into rows (`TokenBudget`)."""

    @property
    def max_length(self) -> int: ...


@dataclasses.dataclass(frozen=True)
class TokenBudget:
    """A batch of `tokens` tokens: as many rows as the pipeline's fixed row
    length (`TokenGeometry`) divides it into, resolved when it compiles."""

    tokens: int
    drop_remainder: bool = True
    batch_fn: Callable[[Sequence[Batch]], Batch] | None = None

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("a token budget must be positive")


@runtime_checkable
class Expansion(Protocol):
    def flat_map(self, element: Batch, /) -> Sequence[Batch | Rejected]: ...


type ExpandStep = Expansion | Callable[[Batch], Iterable[Batch | Rejected]]


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
    """A function of `size` records at a time, read as a dict of columns, each
    a list, as Hugging Face's `map(batched=True)`: with `same_rows`, its
    columns of as many rows join the records' own; otherwise they are the
    records that replace the group. The last group of a finite pass may be
    smaller."""

    step: ColumnStep
    size: int
    identity: str | None = None
    same_rows: bool = True

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"a batched step reads at least one record at a time, not {self.size}")


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
    step: ExpandStep
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
class Interleave:
    cycle_length: int = 16


@dataclasses.dataclass(frozen=True)
class BatchStage:
    size: int
    drop_remainder: bool = True
    batch_fn: Callable[[Sequence[Batch]], Batch] | None = None


@dataclasses.dataclass(frozen=True)
class Pack:
    """Documents packed into windows of `seq_len` tokens along `fields`.

    `along` packs other fields on axes of their own, each to its own length,
    and `segments` bounds the documents a window holds. A first-fit or
    best-fit window closes when any of them is full, as Grain's packer does
    with its `length_struct` and `max_sequences_per_bin`.
    """

    seq_len: int
    fields: tuple[str, ...] = ("text",)
    bins: int = 32
    policy: Literal["first-fit", "best-fit", "concat"] = "first-fit"
    segments: int | None = None
    along: tuple[tuple[str, int], ...] = ()

    @property
    def packed(self) -> tuple[str, ...]:
        """Every field a window holds: `fields` and those `along` axes of their own."""
        return (*self.fields, *(name for name, _ in self.along))

    def __post_init__(self) -> None:
        if self.policy == "concat" and (self.segments is not None or self.along):
            raise ValueError("concatenated packing cuts documents across windows; it takes no "
                             "segment bound or fields along other axes")
        if self.segments is not None and self.segments < 1:
            raise ValueError(f"a window holds at least one document, not {self.segments}")


@dataclasses.dataclass(frozen=True)
class Windows:
    seq_len: int
    stride: int | None = None
    fields: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        if self.seq_len < 1 or (self.stride is not None and self.stride < 1) or not self.fields:
            raise ValueError("causal windows need fields and positive lengths and strides")


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


class DeviceStep:
    """A step over the training batch that runs inside the compiled training
    step, on the accelerator, rather than in the reader: pure and traceable,
    a batch of arrays in and out, its randomness only from per-record key
    fields (`images.augmentation_key`), so its draws are the same at any
    split, microbatch or pool size and on resume. The reader leaves the fields
    it reads in the batch it delivers (`Loader.device`)."""

    def __call__(self, batch: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class MapBatch:
    step: HostStep | DeviceStep
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
    | Interleave
    | BatchStage
    | Pack
    | Rebatch
    | Prefetch
    | Workers
    | Limit
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
        return dataclasses.replace(self.chain(), stages=(*self.chain().stages, *stages))

    def chain(self) -> Pipeline:
        """This dataset as a store and its stages, a store's being none."""
        from dew.data.sources.base import Source

        if not isinstance(self, Source):
            raise TypeError(f"{type(self).__name__} is a Dataset that is neither a store nor a chain")
        return Pipeline(self)

    def holdout(self, fraction: float | None = None, *, key: str | None = None, records: int | None = None,
                split: str | tuple[str, ...] | None = None, **named: float) -> tuple[Dataset, ...]:
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
        `split="validation"` the store's own split, or
        `split=("validation", "test")` each of them, in order, after the
        records the store reads by default.
        """
        from dew.data.holdout import holdout

        return holdout(self, fraction, key=key, records=records, split=split, **named)

    def named(self, name: str) -> Pipeline:
        """This dataset under `name`, which its counts log under in a mix."""
        chain = self.chain()
        return dataclasses.replace(chain, name=name)

    def pipe[**Options](self, stages: Callable[Concatenate[Dataset, Options], Dataset],
                        *args: Options.args, **kwargs: Options.kwargs) -> Dataset:
        """`stages(self, *args, **kwargs)`: a dataset-level stage, such as `text.pack`."""
        return stages(self, *args, **kwargs)

    def map(self, *steps: RecordStep | Configured | HostStep, batched: bool = False, batch_size: int = 1000,
            identity: str | None = None) -> Pipeline:
        """One record to one: each step returns its record's replacement. A
        step that drops records (returns `Rejected`) goes through `filter_map`.

        With `batched`, a step reads `batch_size` records at a time as a dict
        of columns, each a list, and returns columns of as many rows, which
        replace or join the record's own (Hugging Face's `map(batched=True)`).
        After `batch`, a step reads each training batch on the host.
        """
        if self.batched:
            if batched:
                raise ValueError("after batch, map already reads the training batch")
            hosts = [step for step in steps if _host(step)]
            if len(hosts) < len(steps):
                raise TypeError("after batch, map takes functions of the training batch that keep every row")
            # A device step runs in the compiled training step, after every host step the reader runs:
            # a host step written after one would read what it had not yet made. The objective's own
            # batch steps read the fields it leaves alone, and attach after it (`compile`).
            written = [stage.step for stage in self.chain().stages if isinstance(stage, MapBatch)] + hosts
            first = next((index for index, step in enumerate(written) if isinstance(step, DeviceStep)), None)
            if first is not None and not all(isinstance(step, DeviceStep) for step in written[first:]):
                raise ValueError(f"{type(written[first]).__name__} runs in the compiled training step, after "
                                 "every host step of the batch; map the host steps before it")
            return self.then(*(MapBatch(step, identity) for step in hosts))
        if batched:
            return self.then(*(Batched(step, batch_size, identity) for step in _columns(steps)))
        for step in steps:
            if isinstance(step, pygrain.RandomMapTransform):
                raise TypeError(f"{_named_step(step)} draws from a generator; apply it with random_map")
            if not isinstance(step, Configured) and _declares_rejects(step):
                raise TypeError(f"{_named_step(step)} drops records; apply it with filter_map")
        return self.then(*(step if isinstance(step, Configured) else Map(step, identity) for step in steps))

    def random_map(self, *steps: RandomStep, identity: str | None = None) -> Pipeline:
        """One record to one, each step drawing from the record's generator
        (`fn(record, rng)` or a `grain.RandomMapTransform`). A draw is keyed by
        the record's place in the run's order, its epoch included: the same in
        every read mode and reader for one order, and on resume; a different
        shuffle draws differently, and a read cut per share (one that drops
        records, or a streamed one) keys it by the record's place in its share."""
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
                raise TypeError(f"{_named_step(step)} draws from a generator; "
                                "apply it with random_filter_map")
        return self.then(*(Map(step, identity, rejects=True) for step in steps))

    def random_filter_map(self, *steps: RandomStep, identity: str | None = None) -> Pipeline:
        """One record to none or one, each step drawing from the record's
        generator (`fn(record, rng)` or a `grain.RandomMapTransform`) and
        returning its record's replacement or `Rejected(reason)`, which drops
        it, counted by reason, as `video.pick_clip` drops a clip shorter than
        its window. Its draws are keyed as `random_map`'s are, by the record's
        place in the run's order."""
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
        return self.then(Filter(predicate, reason or _reason(predicate), identity))

    def flat_map(self, step: ExpandStep | ColumnStep, *, batched: bool = False,
                 batch_size: int = 1000, identity: str | None = None) -> Pipeline:
        """One record to any number: the step returns a record's records. With
        `batched`, it reads `batch_size` records at a time as a dict of columns
        and returns columns of any length, the records that replace them."""
        if batched:
            (columns,) = _columns((step,))
            return self.then(Batched(columns, batch_size, identity, same_rows=False))
        if not _expands(step):
            raise TypeError(f"{_named_step(step)} is not a record expansion function or transform")
        return self.then(FlatMap(step, identity))

    @property
    def batched(self) -> bool:
        """Whether this chain has made its training batch, after which `map` reads batches."""
        return isinstance(self, Pipeline) and any(isinstance(stage, (BatchStage, TokenBudget))
                                                  for stage in self.stages)

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

    def interleave(self, cycle_length: int = 16) -> Pipeline:
        return self.then(Interleave(cycle_length))

    def batch(
        self,
        size: int | None = None,
        *,
        tokens: int | None = None,
        drop_remainder: bool = True,
        batch_fn: Callable[[Sequence[Batch]], Batch] | None = None,
    ) -> Pipeline:
        if (size is None) == (tokens is None):
            raise ValueError("batch needs exactly one of row size or tokens")
        if tokens is not None:
            return self.then(TokenBudget(tokens, drop_remainder, batch_fn))
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
        segments: int | None = None,
        along: tuple[tuple[str, int], ...] = (),
    ) -> Pipeline:
        """Records laid end to end in rows of `seq_len` tokens: each row holds
        the per-token `fields` of the records it packs, each `along` field on
        an axis of its own, and at most `segments` records; a field that is
        one value a record has no place in a row of several and is left out."""
        return self.then(Pack(seq_len, fields, bins, policy, segments, along))

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

    def snapshot(self, url: str, *, format: Literal["array_record", "parquet"] = "array_record",
                 rows_per_shard: int = 100_000) -> Mapping[str, object]:
        """Run this dataset once over every record and seal what it makes at
        `url`: immutable shards and their committed manifest, which
        `Files(url)` reads back. A fetch's outcomes are sealed with it, so a
        run that reads the snapshot resumes exactly. Returns the manifest."""
        from dew.data.sinks import export

        chain = self.chain()
        return export(chain, url, format=format, rows_per_shard=rows_per_shard)

    def rename_columns(self, names: Mapping[str, str]) -> Pipeline:
        """Each record with its fields renamed, as `rename_columns({"txt": "caption"})`."""
        from dew.data.steps.generic import Fields

        moves = tuple(names.items())
        return self.map(Fields(rename=moves), identity=f"dew.rename_columns:{sorted(moves)}")

    def select_columns(self, *names: str) -> Pipeline:
        """Each record with the fields `names` alone."""
        from dew.data.steps.generic import Fields

        return self.map(Fields(names), identity=f"dew.select_columns:{list(names)}")

    def remove_columns(self, *names: str) -> Pipeline:
        """Each record without the fields `names`."""
        from dew.data.steps.generic import Fields

        return self.map(Fields(remove=names), identity=f"dew.remove_columns:{list(names)}")

    def grain(self, function: Callable[[pygrain.MapDataset | pygrain.IterDataset],
                                            pygrain.MapDataset | pygrain.IterDataset],
                   *, identity: str | None = None) -> Pipeline:
        """`function` of the Grain dataset this chain makes, for a Grain
        operation Dew does not name (`zip`, Grain's own interleave,
        `FlatMapMapDataset`). The chain after it is read per share and
        checkpoints through Grain's own state; `identity` names it in a run's
        plan, its code and closure by default."""
        return self.then(Native(function, identity))

    def stats(self, records: int = 1000) -> dict[str, float]:
        """The counts a run logs from a fresh, lazy pass over at most
        `records` accepted records: attempted, accepted and dropped by reason."""
        from dew.data.compile import _diagnostic
        from dew.data.mixing import state_counts
        from dew.records import record

        chain = self.chain()
        if any(isinstance(stage, (BatchStage, TokenBudget)) for stage in chain.stages):
            raise ValueError("stats counts records; take them before the batch")
        if records < 0:
            raise ValueError("stats records must be nonnegative")
        if records == 0:
            return {}
        reader = _diagnostic(chain, records=records)
        try:
            for _ in reader:
                pass
            counted = state_counts(record(reader.get_state(), "diagnostic state"))
            return {name: value for counts in counted.values() for name, value in counts.metrics().items()}
        finally:
            reader.close()

    def read(self, mode: Read) -> Pipeline:
        """Read the stores this way, rather than as they prefer: "memory" holds
        every record and runs the deterministic steps once, "indexed" reads by
        index, "streamed" reads shards in order."""
        chain = self.chain()
        return dataclasses.replace(chain, mode=mode)

    def threads(self, count: int) -> Pipeline:
        """Read with `count` threads a worker, rather than the measured default."""
        from dew.data.dataset import Loading

        chain = self.chain()
        loading = chain.loading or Loading()
        return dataclasses.replace(chain, loading=dataclasses.replace(
            loading, threads=count, read_buffer=max(loading.read_buffer, count)))

    def iterate(self) -> Iterator[Batch]:
        """This dataset's records, or its batches where it batches: one pass,
        or the passes its `repeat` asks for, with training's random draws."""
        from dew.data.compile import _batch_steps, _diagnostic
        from dew.data.dataset import take_batch
        from dew.objectives.base import VALID_ROWS

        chain = self.chain()
        batched = any(isinstance(stage, (BatchStage, TokenBudget)) for stage in chain.stages)
        # Device steps run here, on the batches read, as the compiled training step runs them.
        _, device = _batch_steps(chain)
        chain = dataclasses.replace(chain, stages=tuple(
            stage for stage in chain.stages
            if not (isinstance(stage, MapBatch) and isinstance(stage.step, DeviceStep))))
        reader = _diagnostic(chain)
        try:
            for batch in reader:
                for step in device:
                    batch = step(batch)
                yield batch if batched else take_batch(
                    {name: value for name, value in batch.items() if name != VALID_ROWS}, 0)
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

    def chain(self) -> Pipeline:
        return self


def concat(datasets: Sequence[Dataset]) -> Pipeline:
    """The records of `datasets`, stores read one after another, as one dataset."""
    from dew.data.sources.base import Source
    from dew.data.sources.concatenate import Concatenate

    stores = tuple(dataset for dataset in datasets if isinstance(dataset, Source))
    if len(stores) < len(datasets):
        raise ValueError("concat joins stores; add the steps after it, or mix datasets that have steps")
    return Pipeline(Concatenate(stores))


def _shown[Field](value: np.ndarray | bytes | Field) -> np.ndarray | bytes | Field | str:
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
    chains = tuple(dataset.chain() for dataset in datasets)
    return Pipeline(Mixture(chains, shares, tokens))


@dataclasses.dataclass(frozen=True)
class ObjectiveData:
    train: tuple[RecordStep | pygrain.RandomMapTransform | Expansion | Configured, ...] = ()
    evaluate: tuple[RecordStep | Expansion | Configured, ...] = ()
    pack: Pack | None = None
    packed: tuple[RecordStep, ...] = ()
    train_batch: tuple[HostStep, ...] = ()
    evaluate_batch: tuple[HostStep, ...] = ()


type _Step = RecordStep | RandomStep | HostStep | Configured | ExpandStep | ColumnStep


def _named_step(step: _Step) -> str:
    if isinstance(step, types.FunctionType | types.MethodType | types.BuiltinFunctionType):
        return step.__qualname__
    return type(step).__name__


def _reason(predicate: Callable[[Batch], bool]) -> str:
    return predicate.reason if isinstance(predicate, Reasoned) else "filtered"


def _host(step: RecordStep | Configured | HostStep) -> TypeGuard[HostStep]:
    """Whether `step` can read a training batch: a function or a Grain map that
    neither drops rows nor waits on a source's facts."""
    return not isinstance(step, Configured | pygrain.RandomMapTransform) and not _declares_rejects(step)


def _columns(steps: Sequence[_Step]) -> list[ColumnStep]:
    """`steps` as functions of columns, refusing a record transform."""
    columns = [step for step in steps if _reads_columns(step)]
    if len(columns) < len(steps):
        raise TypeError("a batched step is a function of columns, not a record transform")
    return columns


def _reads_columns(step: _Step) -> TypeGuard[ColumnStep]:
    return callable(step) and not isinstance(
        step, pygrain.MapTransform | pygrain.RandomMapTransform | Configured | Expansion)


def _expands(step: ExpandStep | ColumnStep) -> TypeGuard[ExpandStep]:
    return isinstance(step, Expansion) or _reads_columns(step)


def _declares_rejects(step: _Step) -> bool:
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
