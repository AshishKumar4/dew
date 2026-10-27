"""Immutable record pipelines, lowered to Grain by :func:`compile`."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import time
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Concatenate, Literal, Protocol, TypeGuard, runtime_checkable

import grain.python as pygrain
import numpy as np

from dew.batch import Batch
from dew.data.dataset import WORKER_BUFFER, Records, phase_ends

if TYPE_CHECKING:
    import jax

    from dew.data.dataset import Loading
    from dew.data.generation import DependencyManifest
    from dew.data.sinks import Writer
    from dew.data.sources.base import Read, Source
    from dew.data.text import NearDuplicates


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
    """A predicate that names what it drops (`images.MinSide`'s "too_small")."""

    reason: str


PACK_POLICY: Literal["first-fit", "best-fit", "concat"] = "first-fit"
"""How a pack fills its rows where none is asked for."""
DROP_REMAINDER = True
"""Whether a batch a training pass fills only in part is dropped, where none is asked for."""
TOKEN_FIELDS = ("text",)
"""The per-token fields a pack or a window lays out where none are named."""
PACK_BINS = 32
"""The rows a pack holds open at once where none is asked for."""
INTERLEAVE = 16
"""The sources an interleave reads at once where none is asked for."""
PREFETCH = 64
"""The records a prefetch reads ahead where none is asked for."""


@runtime_checkable
class Volatile(Protocol):
    volatile: bool


@runtime_checkable
class External(Protocol):
    """A step that reads the storage a record names by path, as a media decoder does when its field
    holds a path rather than bytes: what it makes is keyed by the path, not by what the path holds, so
    no automatic cache's prefix runs past it (`dew.data.compile._automatic`), a file replaced at the same
    path reading anew; a chain's own `cache=` keeps what it made, as it is told to."""

    external: bool


@runtime_checkable
class Costly(Protocol):
    """A step that costs more to run on a record than reading what it made back from a cache costs,
    as measured (`tools/benchmark_data.py automatic-cache`), and makes it from the record alone: so a
    chain that runs it first caches what it makes with no `cache=` (`dew.data.compile._automatic`).
    An image decode is not one: an image a record names by path is read from storage the record's
    content does not key."""

    costly: bool


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
    drop_remainder: bool = DROP_REMAINDER
    batch_fn: Callable[[Sequence[Batch]], Batch] | None = None

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """Rows of the pipeline's row length, as many a batch as `tokens` holds."""
        width = facts.get("seq_len")
        return {**facts, "batch": self.tokens // (width + 1)} if isinstance(width, int) else dict(facts)

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("a token budget must be positive")


@runtime_checkable
class Expansion(Protocol):
    def flat_map(self, element: Batch, /) -> Sequence[Batch | Rejected]: ...


type ExpandStep = Expansion | Callable[[Batch], Iterable[Batch | Rejected]]


@runtime_checkable
class Declares(Protocol):
    """A step, or a stage, that says what it makes of the facts of the records it reads
    (`Dataset.facts`): the fields it writes and their shapes, the tokenizer it encodes with."""

    def declared(self, facts: Mapping[str, object], /) -> Mapping[str, object]: ...


@runtime_checkable
class Stated(Protocol):
    """A store that states facts of its records without reading them (`Dataset.facts`): a token
    store's tokenizer and vocabulary, say."""

    def stated(self) -> Mapping[str, object]: ...


@runtime_checkable
class Measure(Protocol):
    """A fact a configured step needs that a memory read measures over a store's records where the
    store states none (`Configured.measures`): `fact` names it, and calling it on the records gives
    its value."""

    fact: str

    def __call__(self, records: Records, /) -> int: ...


@dataclasses.dataclass(frozen=True)
class Configured:
    """Resolve one record step from source facts before any record reader opens. `one_to_one`
    states that the step keeps one record for each input whatever the facts, which a cache after it
    reads before it binds; binding it to a step that drops or expands records is refused.
    `measures` measures a fact it needs over each store it reads, in a memory read: the whole store,
    below any holdout, so the parts a run trains and scores on agree; a store's own statement of it
    gives way to the measure, and a mixture's step binds the largest of its stores'. `examples` marks
    an objective's step that may make several rows of each source example, as one not `one_to_one`
    may: it runs after the chain's record order, a run's `examples=` bound stands before it, and the
    step it binds to, an expansion, carries the mark (`FlatMap.examples`)."""

    factory: Callable[[Mapping[str, object]], RecordStep | pygrain.RandomMapTransform | Expansion]
    needs: tuple[str, ...]
    scope: Literal["source", "run"] = "source"
    random: bool = False
    one_to_one: bool = False
    measures: Measure | None = None
    examples: bool = False


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
    examples: bool = False
    """Whether this is an objective's step that makes several rows of each source example (a decision
    layout's row a question): a corpus counts its source examples before it (`Accepted`), and a run's
    `examples=` bound stands before it (`dew.data.compile._bounded`)."""


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
    cycle_length: int = INTERLEAVE


@dataclasses.dataclass(frozen=True)
class BatchStage:
    size: int
    drop_remainder: bool = DROP_REMAINDER
    batch_fn: Callable[[Sequence[Batch]], Batch] | None = None

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """The same records, `size` of them a batch."""
        return {**facts, "batch": self.size}


@dataclasses.dataclass(frozen=True)
class Pack:
    """Documents packed into windows of `seq_len` tokens along `fields`.

    `along` packs other fields on axes of their own, each to its own length,
    and `segments` bounds the documents a window holds. A first-fit or
    best-fit window closes when any of them is full, as Grain's packer does
    with its `length_struct` and `max_sequences_per_bin`.

    A concatenating pack cuts documents where its windows end; `overlap` is
    how many tokens before a cut through a document the next window holds
    again, so 1 keeps the next-token target at every cut. A fitting pack cuts
    no document: one longer than a window is cut before it, into pieces that
    overlap as `dew.data.text.Chunks` says.
    """

    seq_len: int
    fields: tuple[str, ...] = TOKEN_FIELDS
    bins: int = PACK_BINS
    policy: Literal["first-fit", "best-fit", "concat"] = PACK_POLICY
    segments: int | None = None
    along: tuple[tuple[str, int], ...] = ()
    overlap: int = 0

    @property
    def packed(self) -> tuple[str, ...]:
        """Every field a window holds: `fields` and those `along` axes of their own."""
        return (*self.fields, *(name for name, _ in self.along))

    def document(self, element: Batch) -> Batch:
        """`element` as a document a window packs: its `packed` fields alone, its token `fields` of one
        length (`dew.data.steps.generic.aligned`), its `along` fields beside them."""
        from dew.data.steps.generic import aligned

        return aligned(element, self.fields, tuple(name for name, _ in self.along))

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """Rows of `seq_len` tokens along `fields`, the last of them a target: `seq_len - 1` inputs."""
        from dew.data.facts import windowed

        return windowed(facts, self.seq_len - 1, self.fields, packed=True, along=self.along)

    def __post_init__(self) -> None:
        if self.policy == "concat" and (self.segments is not None or self.along):
            raise ValueError("concatenated packing cuts documents across windows; it takes no "
                             "segment bound or fields along other axes")
        if self.policy != "concat" and self.overlap:
            raise ValueError(f"a {self.policy} pack cuts no document, so no cut overlaps; cut a long "
                             "document before it with Chunks(overlap=...)")
        if not 0 <= self.overlap < self.seq_len:
            raise ValueError(f"a cut overlaps by 0 to {self.seq_len - 1} tokens of a {self.seq_len}-token "
                             f"window, not {self.overlap}")
        if self.segments is not None and self.segments < 1:
            raise ValueError(f"a window holds at least one document, not {self.segments}")


@dataclasses.dataclass(frozen=True)
class Windows:
    seq_len: int
    stride: int | None = None
    fields: tuple[str, ...] = TOKEN_FIELDS

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """Windows of `seq_len` inputs and their targets along `fields`."""
        from dew.data.facts import windowed

        return windowed(facts, self.seq_len, self.fields, packed=False)

    def __post_init__(self) -> None:
        if self.seq_len < 1 or (self.stride is not None and self.stride < 1) or not self.fields:
            raise ValueError("causal windows need fields and positive lengths and strides")


@dataclasses.dataclass(frozen=True)
class TokenSplit:
    fraction: float
    held: bool
    fields: tuple[str, ...] = TOKEN_FIELDS

    def __post_init__(self) -> None:
        if not 0 <= self.fraction < 1:
            raise ValueError("token holdout fraction must be in [0, 1)")


@dataclasses.dataclass(frozen=True)
class Prefetch:
    buffer: int = PREFETCH


@dataclasses.dataclass(frozen=True)
class Workers:
    count: int
    buffer: int = WORKER_BUFFER


@dataclasses.dataclass(frozen=True)
class Limit:
    count: int


class DeviceStep:
    """A step over the training batch that runs inside the compiled training
    step, on the accelerator, rather than in the reader: pure and traceable,
    a batch of arrays in and out, its randomness only from per-record key
    fields (`images.AugmentationKey`), so its draws are the same at any
    split, microbatch or pool size and on resume. The reader leaves the fields
    it reads in the batch it delivers (`Loader.device`)."""

    def __call__(self, batch: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class MapBatch:
    step: HostStep | DeviceStep
    identity: str | None = None


@runtime_checkable
class Cached(Protocol):
    """What a cache stage fills and reads, a `dew.data.cache.Cache`, or the published generation
    it stands in for, a `dew.data.generation.Generation`: each says where it lives and which rows
    a plan reads from it."""

    @property
    def url(self) -> str: ...

    def reference(self, manifest: DependencyManifest, /) -> str: ...


@dataclasses.dataclass(frozen=True)
class Artifact:
    cache: Cached
    automatic: bool = False
    """Whether the compiler placed it (`dew.data.compile._automatic`) rather than the chain: it holds
    the records its prefix makes, so a run reads the same records with or without it and its plan
    names none (`dew.data.cache.automatic_root`)."""


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

type Step = (
    RecordStep
    | RandomStep
    | ExpandStep
    | HostStep
    | DeviceStep
    | ColumnStep
    | Callable[[int, Batch], Batch | Rejected]
    | Callable[[Batch], bool]
    | Callable[[pygrain.MapDataset | pygrain.IterDataset], pygrain.MapDataset | pygrain.IterDataset]
    | Callable[[Mapping[str, object]], RecordStep | pygrain.RandomMapTransform | Expansion]
)
"""What a stage runs and its identity names (`dew.data.plan.StepIdentity`): a record, random,
indexed, expanding, batch or column step, a predicate, a Grain function or a configured step's
factory."""


@dataclasses.dataclass(frozen=True)
class Runs:
    """What a stage runs (`runs`): its callable, the identity written for it, and the configured step
    it was bound from, the one description its plan identity, its cache keyability and `show` read."""

    step: Step
    identity: str | None = None
    configured: Bound | None = None


def runs(stage: Stage) -> Runs | None:
    """What `stage` runs, None for a stage that runs no step of its own (an order, a batch, a cache)."""
    if isinstance(stage, Configured):
        return Runs(stage.factory)
    if isinstance(stage, (Map, RandomMap, FlatMap)):
        return Runs(stage.step, stage.identity, stage.configured)
    if isinstance(stage, (MapWithIndex, MapBatch, Batched)):
        return Runs(stage.step, stage.identity)
    if isinstance(stage, Filter):
        return Runs(stage.predicate, stage.identity)
    if isinstance(stage, Native):
        return Runs(stage.function, stage.identity)
    return None


@dataclasses.dataclass(frozen=True)
class Mixture:
    pipelines: tuple[Pipeline, ...]
    weights: tuple[float, ...]
    tokens: Callable[[Batch], int] | None = None

    def __post_init__(self) -> None:
        if not self.pipelines:
            raise ValueError("mix needs at least one dataset")
        if len(self.weights) != len(self.pipelines):
            raise ValueError(f"mix has {len(self.pipelines)} datasets and {len(self.weights)} weights")
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights):
            raise ValueError("mix weights must be finite and positive")


@dataclasses.dataclass(frozen=True)
class Phases:
    pipelines: tuple[Pipeline, ...]
    ends: tuple[int | None, ...]
    """Where each phase ends, as `clock` counts them: at the training step it names (`phases`), or at
    the run's committed tokens (`schedule`). The last runs on (None); on the token clock it may end
    instead, where the data ends."""
    clock: Literal["steps", "tokens"] = "steps"

    def __post_init__(self) -> None:
        if len(self.pipelines) != len(self.ends):
            raise ValueError(f"phases need one end per pipeline: {len(self.pipelines)} pipelines and "
                             f"{len(self.ends)} ends")
        if self.clock == "steps":
            phase_ends(self.ends, "step")
        else:
            phase_ends(self.ends if self.ends[-1] is None else (*self.ends, None), "token")


def children(pipeline: Pipeline) -> tuple[Pipeline, ...]:
    """The chains `pipeline`'s mixture or phases read, none where it reads a store."""
    source = pipeline.source
    return source.pipelines if isinstance(source, (Mixture, Phases)) else ()


def chains(pipeline: Pipeline) -> Iterator[Pipeline]:
    """Every chain of `pipeline`'s tree, children first: each corpus under its mixtures and phases,
    then the chain that mixes or phases them, and `pipeline` last."""
    for child in children(pipeline):
        yield from chains(child)
    yield pipeline


def with_children(pipeline: Pipeline, rewrite: Callable[[Pipeline], Pipeline]) -> Pipeline:
    """`pipeline` with each chain its mixture or phases read rewritten by `rewrite`; `pipeline` itself
    where it reads a store."""
    source = pipeline.source
    if not isinstance(source, (Mixture, Phases)):
        return pipeline
    rewritten = tuple(rewrite(child) for child in source.pipelines)
    return dataclasses.replace(pipeline, source=dataclasses.replace(source, pipelines=rewritten))


def with_stores(pipeline: Pipeline, store: Callable[[Source], Source]) -> Pipeline:
    """`pipeline` with each store under its mixtures and phases read through `store`, every step as it
    was."""
    if isinstance(pipeline.source, (Mixture, Phases)):
        return with_children(pipeline, functools.partial(with_stores, store=store))
    return dataclasses.replace(pipeline, source=store(pipeline.source))


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

    def _then(self, *stages: Stage) -> Pipeline:
        """This dataset with `stages` after its own."""
        return dataclasses.replace(self._chain(), stages=(*self._chain().stages, *stages))

    def _chain(self) -> Pipeline:
        """This dataset as a store and its stages, a store's being none."""
        from dew.data.sources.base import Source

        if not isinstance(self, Source):
            raise TypeError(f"{type(self).__name__} is a Dataset that is neither a store nor a chain")
        return Pipeline(self)

    def holdout(self, fraction: float | None = None, *, key: str | None = None, records: int | None = None,
                split: str | tuple[str, ...] | None = None, tokens: float | None = None,
                fields: tuple[str, ...] = TOKEN_FIELDS, **named: float) -> tuple[Dataset, ...]:
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
        records the store reads by default. `tokens=0.01` holds out the first
        hundredth of the tokens of the order the steps so far read, cutting at
        most one document in two, so the parts are disjoint and together the
        corpus; it reads a finite indexed order, before any repeat, and cuts the
        per-token `fields` together, counting the first.
        """
        if tokens is not None:
            if fraction is not None or key is not None or records is not None or split is not None or named:
                raise ValueError("holdout(tokens=) holds out a share of the tokens alone")
            chain = self._chain()
            return (chain._then(TokenSplit(tokens, held=False, fields=fields)),
                    chain._then(TokenSplit(tokens, held=True, fields=fields)))
        from dew.data.holdout import holdout

        return holdout(self, fraction, key=key, records=records, split=split, **named)

    def facts(self) -> Mapping[str, object]:
        """What this dataset's records are, read without opening a reader: the facts its stores
        state (a token store's tokenizer and vocabulary), then what each step makes of them in
        order: `fields`, each field's shape a record holds; `seq_len` and `packed` for token rows;
        `tokenizer`, `vocab_size`; `batch` once batched. A step that changes records and doesn't
        say how leaves their fields unknown."""
        from dew.data.facts import folded

        return folded(self._chain())

    def named(self, name: str) -> Pipeline:
        """This dataset under `name`, which its counts log under in a mix."""
        chain = self._chain()
        return dataclasses.replace(chain, name=name)

    def pipe[**Options](self, stages: Callable[Concatenate[Dataset, Options], Dataset],
                        *args: Options.args, **kwargs: Options.kwargs) -> Dataset:
        """`stages(self, *args, **kwargs)`: a dataset-level stage, such as `text.pack`. The chain
        it returns keeps the call beside the stages it made (`Piped`), which a run's record writes
        in their place (`dew.data.records`)."""
        before = self._chain()
        after = stages(self, *args, **kwargs)
        if not isinstance(after, Pipeline):
            return after
        called = Piped(stages, args, tuple(kwargs.items()), len(before.stages), len(after.stages))
        kept = tuple(entry for entry in after.piped if entry.end <= len(after.stages))
        if after.source is before.source and len(after.stages) >= len(before.stages) and all(
                made is had for made, had in zip(after.stages, before.stages, strict=False)):
            # The call extends its input: what the input kept still stands, and the call covers the rest.
            inherited = tuple(entry for entry in before.piped if all(entry is not other for other in kept))
            return dataclasses.replace(after, piped=(*inherited, *kept, called))
        # The call made another store of its input: it stands for its input and all it made.
        return dataclasses.replace(after, piped=(*kept, dataclasses.replace(called, start=0, input=before)))

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
            written = [stage.step for stage in self._chain().stages if isinstance(stage, MapBatch)] + hosts
            first = next((index for index, step in enumerate(written) if isinstance(step, DeviceStep)), None)
            if first is not None and not all(isinstance(step, DeviceStep) for step in written[first:]):
                raise ValueError(f"{type(written[first]).__name__} runs in the compiled training step, after "
                                 "every host step of the batch; map the host steps before it")
            return self._then(*(MapBatch(step, identity) for step in hosts))
        if batched:
            return self._then(*(Batched(step, batch_size, identity) for step in _columns(steps)))
        for step in steps:
            if isinstance(step, pygrain.RandomMapTransform):
                raise TypeError(f"{_named_step(step)} draws from a generator; apply it with random_map")
            if not isinstance(step, Configured) and _declares_rejects(step):
                raise TypeError(f"{_named_step(step)} drops records; apply it with filter_map")
        return self._then(*(step if isinstance(step, Configured) else Map(step, identity) for step in steps))

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
        return self._then(*(RandomMap(step, identity) for step in steps))

    def filter_map(self, *steps: RecordStep, identity: str | None = None) -> Pipeline:
        """One record to none or one: each step returns its record's
        replacement or `Rejected(reason)`, which drops it, counted by reason
        in the reader's checkpointed counts, as `images.fetch` and
        `images.DecodeImage` do."""
        for step in steps:
            if isinstance(step, pygrain.RandomMapTransform):
                raise TypeError(f"{_named_step(step)} draws from a generator; "
                                "apply it with random_filter_map")
        return self._then(*(Map(step, identity, rejects=True) for step in steps))

    def random_filter_map(self, *steps: RandomStep, identity: str | None = None) -> Pipeline:
        """One record to none or one, each step drawing from the record's
        generator (`fn(record, rng)` or a `grain.RandomMapTransform`) and
        returning its record's replacement or `Rejected(reason)`, which drops
        it, counted by reason, as `video.PickClip` drops a clip shorter than
        its window. Its draws are keyed as `random_map`'s are, by the record's
        place in the run's order."""
        return self._then(*(RandomMap(step, identity, rejects=True) for step in steps))

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
        has no single position, so the step is refused there. A step that
        drops rows says so with `rejects = True` and returns `Rejected(reason)`.
        """
        return self._then(MapWithIndex(step, identity, _declares_rejects(step)))

    def filter(
        self, predicate: Callable[[Batch], bool], *, reason: str | None = None, identity: str | None = None
    ) -> Pipeline:
        """Keep the records `predicate` accepts; the rest are counted under
        `reason`, by default the predicate's own (`images.MinSide`'s "too_small")."""
        return self._then(Filter(predicate, reason or _reason(predicate), identity))

    def flat_map(self, step: ExpandStep | ColumnStep, *, batched: bool = False,
                 batch_size: int = 1000, identity: str | None = None) -> Pipeline:
        """One record to any number: the step returns a record's records. With
        `batched`, it reads `batch_size` records at a time as a dict of columns
        and returns columns of any length, the records that replace them."""
        if batched:
            (columns,) = _columns((step,))
            return self._then(Batched(columns, batch_size, identity, same_rows=False))
        if not _expands(step):
            raise TypeError(f"{_named_step(step)} is not a record expansion function or transform")
        return self._then(FlatMap(step, identity))

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
        return self._then(Shuffle(seed, window, block_shards))

    def repeat(self, epochs: int | None = None) -> Pipeline:
        return self._then(Repeat(epochs))

    def interleave(self, cycle_length: int = INTERLEAVE) -> Pipeline:
        return self._then(Interleave(cycle_length))

    def batch(
        self,
        size: int | None = None,
        *,
        tokens: int | None = None,
        drop_remainder: bool = DROP_REMAINDER,
        batch_fn: Callable[[Sequence[Batch]], Batch] | None = None,
    ) -> Pipeline:
        if (size is None) == (tokens is None):
            raise ValueError("batch needs exactly one of row size or tokens")
        if tokens is not None:
            return self._then(TokenBudget(tokens, drop_remainder, batch_fn))
        if size is None or size < 1:
            raise ValueError("batch size must be positive")
        return self._then(BatchStage(size, drop_remainder, batch_fn))

    def pack(
        self,
        seq_len: int,
        *,
        fields: tuple[str, ...] = TOKEN_FIELDS,
        bins: int = PACK_BINS,
        policy: Literal["first-fit", "best-fit", "concat"] = PACK_POLICY,
        segments: int | None = None,
        along: tuple[tuple[str, int], ...] = (),
        overlap: int = 0,
    ) -> Pipeline:
        """Records laid end to end in rows of `seq_len` tokens: each row holds
        the per-token `fields` of the records it packs, each `along` field on
        an axis of its own, and at most `segments` records; a field that is
        one value a record has no place in a row of several and is left out.
        A concatenating pack's row starts again `overlap` tokens before a cut
        through a record (`Pack`)."""
        return self._then(Pack(seq_len, fields, bins, policy, segments, along, overlap))

    def windows(
        self, seq_len: int, *, stride: int | None = None, fields: tuple[str, ...] = TOKEN_FIELDS
    ) -> Pipeline:
        return self._then(Windows(seq_len, stride, fields))

    def prefetch(self, buffer: int = PREFETCH) -> Pipeline:
        return self._then(Prefetch(buffer))

    def workers(self, count: int, *, buffer: int = WORKER_BUFFER) -> Pipeline:
        return self._then(Workers(count, buffer))

    def take(self, count: int) -> Pipeline:
        return self._then(Limit(count))

    def cache(self, cache: str | Cached) -> Pipeline:
        """Fill `cache` with the deterministic record steps before it, or read a
        published `Generation` as the source in their place, with its published
        packing in place of a `pack` that follows."""
        from dew.data.cache import Cache

        return self._then(Artifact(Cache(cache) if isinstance(cache, str) else cache))

    def snapshot(self, url: str, *, format: Literal["array_record", "parquet"] | Writer = "array_record",
                 rows_per_shard: int = 100_000,
                 deduplicate: NearDuplicates | None = None) -> Mapping[str, object]:
        """Run this dataset once over every record and seal what it makes at
        `url`: immutable shards and their committed manifest, which
        `Files(url)` reads back, written as ArrayRecord, as Parquet or by a
        native writer (`dew.data.sinks.Writer`). A fetch's outcomes are sealed with it, so a
        run that reads the snapshot resumes exactly. With `deduplicate`
        (`dew.data.text.NearDuplicates`, the `curation` extra), each group of
        near-duplicates in that one pass is sealed as one record, and the
        manifest's metadata records the rule, DataTrove's release and the
        records removed. Returns the manifest."""
        from dew.data.sinks import export, passed

        chain = self._chain()
        if deduplicate is None:
            return export(chain, url, format=format, rows_per_shard=rows_per_shard)
        import tempfile
        from importlib.metadata import version

        from dew.data.sources import Files
        from dew.data.text import near_duplicates_removed

        # The one pass sealed first, so every later read of it reads the same records in one order,
        # whatever a fetch makes of them the next time.
        with tempfile.TemporaryDirectory(prefix="dew-deduplicate-") as sealed:
            # A pass that kept no record seals none, and has none to deduplicate.
            if export(chain, sealed, format="array_record", rows_per_shard=rows_per_shard)["rows"]:
                kept, removed = near_duplicates_removed(lambda: passed(Files(sealed)), deduplicate)
            else:
                kept, removed = iter(()), 0
            rule = {**dataclasses.asdict(deduplicate), "datatrove": version("datatrove"), "removed": removed}
            return export(kept, url, format=format, rows_per_shard=rows_per_shard,
                          metadata={"near_duplicates": rule})

    def rename_columns(self, names: Mapping[str, str]) -> Pipeline:
        """Each record with its fields renamed, as `rename_columns({"txt": "caption"})`."""
        from dew.data.steps.generic import Fields

        moves = tuple(names.items())
        return self.map(Fields(rename=moves, bookkeeping=True),
                        identity=f"dew.rename_columns:{sorted(moves)}")

    def select_columns(self, *names: str) -> Pipeline:
        """Each record with the fields `names` alone."""
        from dew.data.steps.generic import Fields

        return self.map(Fields(names, bookkeeping=True), identity=f"dew.select_columns:{list(names)}")

    def remove_columns(self, *names: str) -> Pipeline:
        """Each record without the fields `names`."""
        from dew.data.steps.generic import Fields

        return self.map(Fields(remove=names, bookkeeping=True), identity=f"dew.remove_columns:{list(names)}")

    def grain(self, function: Callable[[pygrain.MapDataset | pygrain.IterDataset],
                                            pygrain.MapDataset | pygrain.IterDataset],
                   *, identity: str | None = None) -> Pipeline:
        """`function` of the Grain dataset this chain makes, for a Grain
        operation Dew does not name (`zip`, Grain's own interleave,
        `FlatMapMapDataset`). The chain after it is read per share and
        checkpoints through Grain's own state; `identity` names it in a run's
        plan, its code and closure by default."""
        return self._then(Native(function, identity))

    def stats(self, records: int = 1000) -> dict[str, float]:
        """The counts a run logs from a fresh, lazy pass over at most `records` accepted records:
        the rows delivered and the drops by reason (`dew.data.mixing.CorpusCounts`), the rows the
        pass read a second once its reader was open (`data/records_per_second`: the loading
        settings default to none, while workers or prefetching the chain writes run as written),
        and for token rows, under "text", the tokens they hold (`data/tokens`) and the share of
        their places that pad a row (`data/padding`)."""
        from dew._records import record
        from dew.data.compile import _diagnostic
        from dew.data.mixing import data_metrics, state_counts

        chain = self._chain()
        if any(isinstance(stage, (BatchStage, TokenBudget)) for stage in chain.stages):
            raise ValueError("stats counts records; take them before the batch")
        if records < 0:
            raise ValueError("stats records must be nonnegative")
        if records == 0:
            return {}
        reader = _diagnostic(chain, records=records)
        rows = places = padding = 0
        started = time.perf_counter()
        try:
            for batch in reader:
                rows += 1
                held, padded = _token_places(batch)
                places, padding = places + held, padding + padded
            seconds = time.perf_counter() - started
            metrics = data_metrics(state_counts(record(reader.get_state(), "diagnostic state")).values())
        finally:
            reader.close()
        metrics["data/records_per_second"] = rows / seconds if seconds > 0 else math.inf
        if places:
            metrics["data/tokens"] = places - padding
            metrics["data/padding"] = padding / places
        return metrics

    def read(self, mode: Read) -> Pipeline:
        """Read the stores this way, rather than as they prefer: "memory" holds
        every record and runs the deterministic steps once, "indexed" reads by
        index, "streamed" reads shards in order."""
        chain = self._chain()
        return dataclasses.replace(chain, mode=mode)

    def threads(self, count: int) -> Pipeline:
        """Read with `count` threads a worker, rather than the measured default."""
        from dew.data.dataset import Loading

        loading = self._chain()._loading or Loading()
        return self.loading(dataclasses.replace(loading, threads=count,
                                                read_buffer=max(loading.read_buffer, count)))

    def loading(self, loading: Loading) -> Pipeline:
        """Read with these workers, threads and buffers (`Loading`), rather than the measured
        defaults, recorded with the chain; `threads` sets its read threads alone."""
        return dataclasses.replace(self._chain(), _loading=loading)

    def __iter__(self) -> Iterator[Batch]:
        """This dataset's records, or its batches where it batches: one pass,
        or the passes its `repeat` asks for, with training's random draws
        (`for record in dataset`)."""
        from dew.batch import VALID_ROWS
        from dew.data.compile import _batch_steps, _diagnostic
        from dew.data.dataset import take_batch

        chain = self._chain()
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
        """Print the chain, each verb and step a preset or function expanded to in order, then
        its first `count` records."""
        from dew.data.show import shown

        shown(self._chain(), itertools.islice(self, count))


@dataclasses.dataclass(frozen=True)
class Piped:
    """A call of `Dataset.pipe` and what it made: `function(input, *args, **settings)` made the
    stages `start` to `end` of the chain that keeps it, or, where it made another store of its
    input (a token store's windows), the store and the stages to `end` from `input`."""

    function: Callable[..., Dataset]
    args: tuple[object, ...]
    settings: tuple[tuple[str, object], ...]
    start: int
    end: int
    input: Pipeline | None = None


@dataclasses.dataclass(frozen=True)
class Pipeline(Dataset):
    """A dataset's store and the stages its verbs added; what every verb returns."""

    source: Source | Mixture | Phases
    stages: tuple[Stage, ...] = ()
    name: str | None = None
    mode: Read | None = None
    _loading: Loading | None = None
    piped: tuple[Piped, ...] = dataclasses.field(default=(), compare=False, repr=False)
    """The calls of `pipe` that made its stages (`Piped`), outermost last; no plan reads them."""

    def _then(self, *stages: Stage) -> Pipeline:
        return dataclasses.replace(self, stages=(*self.stages, *stages))

    def _chain(self) -> Pipeline:
        return self


def concat(datasets: Sequence[Dataset]) -> Pipeline:
    """The records of `datasets`, stores read one after another, as one dataset."""
    from dew.data.sources.base import Source
    from dew.data.sources.concatenate import Concatenate

    stores = tuple(dataset for dataset in datasets if isinstance(dataset, Source))
    if len(stores) < len(datasets):
        raise ValueError("concat joins stores; add the steps after it, or mix datasets that have steps")
    return Pipeline(Concatenate(stores))


def _token_places(batch: Batch) -> tuple[int, int]:
    """The places a batch's token rows under "text" hold, and those of them that pad a row, as a
    model reads them (`dew.batch.real_slots`): segment 0 of a packed row and the places its
    validity marks invalid, in its `text_coordinates` or in stored `text_segment_ids` and
    `text_attention_mask`; none where it holds no token ids."""
    tokens = batch.get("text")
    if not isinstance(tokens, np.ndarray) or tokens.dtype.kind not in "iu":
        return 0, 0
    from dew.batch import COORDINATES_FIELD, VALIDITY_FIELD, real_slots

    fields = {name: batch[f"text_{name}"] for name in (COORDINATES_FIELD, VALIDITY_FIELD, "segment_ids")
              if f"text_{name}" in batch}
    real = real_slots(fields, tokens.shape, xp=np)
    return int(tokens.size), int(tokens.size - np.count_nonzero(real))


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
    shares = (1.0,) * len(datasets) if weights is None else tuple(float(weight) for weight in weights)
    chains = tuple(dataset._chain() for dataset in datasets)
    return Pipeline(Mixture(chains, shares, tokens))


def phases(*entries: tuple[Dataset, int | None]) -> Pipeline:
    """Datasets read one after another: each `(dataset, until_step)` until the training step it
    names, the last until the run ends (`None`). Each phase is a whole chain, laid out and batched
    as it reads, each of the same batch."""
    if not entries:
        raise ValueError("phases need at least one (dataset, until_step)")
    return Pipeline(Phases(tuple(dataset._chain() for dataset, _ in entries),
                           tuple(until for _, until in entries)))


def schedule(stages: Sequence[tuple[Dataset, float | None]]) -> Pipeline:
    """Datasets read one after another on the run's committed-token clock, the clock `fit(tokens=)`
    counts: each `(dataset, tokens)` for the valid target tokens its steps commit, the last for the
    rest of the run (`None`), or for its tokens, where the data ends (the run's last step). A stage
    ends at the step whose commit reaches its tokens, on every process alike, so its last step may
    commit past them; one that a single step's commit passes reads nothing. A corpus that an earlier
    stage read continues from where it left it, and what the run had read ahead of the end is not
    trained on. Each stage is a whole chain, laid out and batched as it reads."""
    if not stages:
        raise ValueError("a schedule needs at least one (dataset, tokens)")
    budgets = [tokens for _, tokens in stages]
    if any(tokens is None for tokens in budgets[:-1]) or any(
            tokens is not None and not (math.isfinite(tokens) and tokens >= 1) for tokens in budgets):
        raise ValueError(f"every stage but the last commits a count of tokens, of at least one; got "
                         f"{budgets}")
    ends = list(itertools.accumulate(int(tokens) for tokens in budgets if tokens is not None))
    return Pipeline(Phases(tuple(dataset._chain() for dataset, _ in stages),
                           (*ends[:len(stages) - 1], None if budgets[-1] is None else ends[-1]), "tokens"))


@dataclasses.dataclass(frozen=True)
class ObjectiveData:
    train: tuple[RecordStep | pygrain.RandomMapTransform | Expansion | Configured, ...] = ()
    evaluate: tuple[RecordStep | Expansion | Configured, ...] = ()
    pack: Pack | None = None
    packed: tuple[RecordStep, ...] = ()
    train_batch: tuple[HostStep, ...] = ()
    evaluate_batch: tuple[HostStep, ...] = ()


type _Step = RecordStep | RandomStep | HostStep | Configured | ExpandStep | ColumnStep


def record_stage(step: RecordStep | pygrain.RandomMapTransform | Expansion, *, identity: str | None = None,
                 configured: Bound | None = None) -> Map | RandomMap | FlatMap:
    """The stage a record step is applied by where no verb names it, from what the step says it is:
    an expansion's records (`FlatMap`), a generator's draws (`RandomMap`), or one record's
    replacement (`Map`), dropping records where it says it does (`rejects`). An objective's steps
    and a configured step's binding are read through it; a verb (`map`, `random_map`, `filter_map`,
    `random_filter_map`, `flat_map`) builds the stage its name states, which a step of another
    kind is refused."""
    if isinstance(step, Expansion):
        return FlatMap(step, identity, configured)
    if isinstance(step, pygrain.RandomMapTransform):
        return RandomMap(step, identity, _declares_rejects(step), configured)
    return Map(step, identity, _declares_rejects(step), configured)


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


def _declares_rejects(step: _Step | Callable[[int, Batch], Batch | Rejected]) -> bool:
    """Whether `step` says it drops records, with a `rejects` flag (`images.fetch`, `images.DecodeImage`).
    A step that says nothing keeps every record, whatever its signature reads."""
    return isinstance(step, Rejecting) and bool(step.rejects)
