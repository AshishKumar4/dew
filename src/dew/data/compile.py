"""Compile the same pipeline to an elastic order or an exact per-share reader.

The public entry points, `compile`, `iterate`, `validation`, `publish` and
`objective_steps`, share one preparation (`_prepared`): the checks,
generation promotion, the sources' capabilities and pinned facts, the
tokenizer contract, objective steps and configured steps bound to metadata.
A memory read then runs its eager stages once (`dew.data.materialize`), the
analysis (`dew.data.analysis`) settles how the order is cut, and the Grain
lowering (`dew.data.lowering`) builds the readers. `compile` saves with a
training read the plan a resume checks: the dependency manifest, the
canonical order and the derived order facts.
"""

from __future__ import annotations

import bisect
import dataclasses
import functools
import itertools
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Literal, Protocol, runtime_checkable

import grain.python as pygrain
import numpy as np

from dew._position import unplaced
from dew._records import JSON, integer, json_value, record
from dew.batch import Batch
from dew.data.analysis import (
    _VIEWS,
    Measuring,
    _analyzed,
    _at,
    _cache_key,
    _cache_manifest,
    _cacheable,
    _capabilities,
    _configured,
    _cut,
    _drops,
    _external,
    _identity,
    _in_memory,
    _leaves,
    _Lowering,
    _manifest,
    _measures,
    _member_keys,
    _members,
    _name,
    _once,
    _preferred_read,
    _promoted,
    _stated,
    _steps,
    _until,
    _volatile,
    _written,
    draws,
    reach,
    written,
)
from dew.data.bounds import (
    _bounded,
    _countable,
    _counted,
    _expands,
    _lays_out,
    _native_after_bound,
    _one_example_until,
)
from dew.data.compilation import compiling, within
from dew.data.dataset import (
    Closeable,
    DataPartition,
    Ends,
    FilledBatch,
    Forwarding,
    GlobalReader,
    GlobalStream,
    Loader,
    Loading,
    PhasedStream,
    Reader,
    Records,
    StepPhases,
    phase_stages,
    rows_of,
    take_batch,
)
from dew.data.epochs import epoch_records, order_records, tail
from dew.data.facts import laid_out, merge_facts, row_lengths
from dew.data.handoff import record_stage_of
from dew.data.lowering import (
    _as_iter,
    _Build,
    _Locality,
    _member_key,
    _Present,
    _repeated,
    _split_spans,
    _workers,
    cache_prefix,
)
from dew.data.materialize import _materialize, _Memory
from dew.data.mixing import CorpusCounts, Proportions, _weights, mixed_indexed, warn_rejections
from dew.data.ordering import place_of
from dew.data.phase_corpora import _alone_mixed, _corpora, _shared
from dew.data.pipeline import (
    UNCOUNTED,
    Artifact,
    Batched,
    BatchStage,
    Combined,
    Concatenated,
    Configured,
    Costly,
    Dataset,
    DeviceStep,
    Expansion,
    Filter,
    FlatMap,
    HostStep,
    Interleave,
    LaysOutWindows,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    MarkConsumed,
    Mixture,
    Native,
    ObjectiveData,
    Pack,
    PackedLayout,
    Phases,
    Pipeline,
    Prefetch,
    RandomMap,
    RecordStep,
    Repeat,
    Shortest,
    Shuffle,
    SingleReader,
    Slice,
    Stage,
    TokenBudget,
    Windows,
    Workers,
    children,
    examples_bound,
    record_stage,
    runs,
    with_children,
    with_stores,
)
from dew.data.plan import (
    ASKED,
    DataResume,
    Plan,
    StepIdentity,
    described,
    earlier_readers,
    fingerprint,
    identified,
    rule_in_force,
)
from dew.data.readers import GRAIN, Replayed, _CanonicalPass, _Host, _PerShare, _Statistics, statistics_of
from dew.data.sources.base import (
    TOKENIZER_CONTRACT,
    Read,
    Rows,
    ShardManifest,
    Source,
    View,
    _consumed,
    asked,
    beneath,
    materialized,
    stream_manifest,
    tokenizer_contract,
)
from dew.data.steps.generic import Coordinated
from dew.data.storage import Mirrored

_DEFAULT_LOADING = Loading()
_TRAINING_LOADING = Loading(workers="auto")
"""A training read's loading where its chain names none (`compile`): automatic workers (`Loading.resolved`),
which read its records' Python off the training process where it has some (`_Build.tuned`)."""


@runtime_checkable
class _OrderFacts(Protocol):
    def order_fingerprints(self) -> Mapping[str, str]: ...


def _writable(root: str) -> bool:
    """Whether this process may make and write `root`: the nearest directory of it that exists may
    be written (a read-only home, say, may not)."""
    existing = os.path.abspath(root)
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    return os.path.isdir(existing) and os.access(existing, os.W_OK | os.X_OK)


def _automatic(pipeline: Pipeline, read: Read, root: str | None) -> Pipeline:
    """`pipeline` with a cache at `root` after its leading deterministic record maps, where one of them
    costs more to run than its records cost to read back (`Costly`: a learned tokenizer's encoding)
    and the chain writes no cache: each corpus of a mixture or phases its own, in a root this process
    may write. The maps are the ones
    a cache keys (`_cacheable`) and reads again unchanged (none `_volatile`), one that rejects among
    them too: a read by index reads its cache back as each input's outputs, faster than it reruns them
    (row 803: 1,500 1024x768 JPEGs decoded and resized in 2.4 s on 16 threads, read back in 0.6 s).
    A stream reads it as the steps it stands for are read (`dew.data.lowering._transparent`, row 4217).
    It stands for them only by each input's place in a store
    whose content is named, in a clean checkout whose steps are exact (`Manifest`), and nowhere
    else, so a chain no cache could serve reads as it is written, never refused. It holds the
    records the maps make, and fills, publishes, replays and evicts as `cache(url)` does (`Cache`),
    though local storage that fails it stops its fill and the run reads on (`Cache.optional`),
    so a run reads the same records with or without it (`Artifact.automatic`), and a run of the
    same maps over the same records reads them back once one has read them whole. Its root is the
    run's to name or refuse (`automatic_root`)."""
    from dew.data.cache import Cache

    if root is None:
        return pipeline
    if read == "streamed" and any(isinstance(stage, MapWithIndex) for stage in pipeline.stages):
        # A stream positions its records once, under the steps that read their places, which a cache
        # standing for some of them would split.
        return pipeline
    if isinstance(pipeline.source, Combined):
        return with_children(pipeline, lambda child: _automatic(child, read, root))
    if any(isinstance(stage, Artifact) for stage in _steps(pipeline)):
        return pipeline

    def kept(stage: Stage) -> bool:
        return (isinstance(stage, (Map, MapWithIndex)) and _cacheable(stage) and not _volatile(stage)
                and not _external(stage))

    at = next((index for index, stage in enumerate(pipeline.stages) if not kept(stage)), len(pipeline.stages))
    prefix = pipeline.stages[:at]
    steps = [stage.step for stage in prefix if isinstance(stage, (Map, MapWithIndex))]
    if not any(isinstance(step, Costly) and step.costly for step in steps):
        return pipeline
    if not pipeline.source.supports(read) or not _writable(root):
        return pipeline
    # Whether a cache can key it, as its fill keys it at its lowering (`dew.data.lowering._fill`).
    key = _cache_key(pipeline.source, prefix, read)
    if key is None or not (key.clean and key.exact):
        return pipeline
    # A stream's corpus may be larger than the root: its fill takes only the room the root has free.
    cached = Artifact(Cache(root, optional=True, evicts=read != "streamed"), automatic=True)
    return dataclasses.replace(pipeline, stages=(*prefix, cached, *pipeline.stages[at:]))


def _cache_references(pipeline: Pipeline, read: Read) -> tuple[str, ...]:
    """The caches each corpus of `pipeline` reads through, among its stages as
    its pipeline wrote them (`_written`), the ones a memory read already ran included, each named against
    the source it came from."""
    references = []
    for leaf, source in _leaves(pipeline):
        stages, origin = _written(source, leaf.stages), source
        while materialized(origin):
            origin = origin.source
        references += [stage.cache.reference(_cache_manifest(origin, stages[:index], read))
                       for index, stage in enumerate(stages) if isinstance(stage, Artifact)]
    return tuple(references)


@dataclasses.dataclass(eq=False)
class _PinnedRead:
    """The memory read source pinning took (`_pin_sources`), which the compile's memory reads of
    the source take in place of another until it has materialized every branch and lets the read go
    (`_staged`), so a compile reads each source once and a loader holds only the rows it made."""

    records: Records | None


@dataclasses.dataclass(frozen=True)
class _Facts(View[Source]):
    """`source` as the compile's read of it took it (`_pin_sources`): its name, identity and facts
    as they were then, and the memory read pinning took, which its memory reads reuse."""

    rows: ClassVar[Rows] = "read"
    source: Source
    read: Read
    stated_snapshot: Mapping[str, object]
    stated_metadata: Mapping[str, object] | None
    """The facts the source states as pinning read them, or None where pinning read none (a stream's,
    `_stated`), which the first reader of them reads then (`metadata`): an undrawn member's files are not
    opened for facts the plan never asks (row 9009)."""
    stated_description: str
    """The source's description as the compile took it, which every reader the compile makes names its
    corpus by: a store in memory describes itself by a hash of every record it holds, which a reader
    opened outside the compile would take again (row 803)."""
    held: _PinnedRead | None = None
    """The memory read pinning took, which the compile's memory reads of the source reuse, so its
    facts and the rows a run materializes are one read's."""
    stated_label: str = ""
    """The source's name as it was written, before a read put another in its place (a Hub split's
    Parquet files, `dew.data.sources.base.Consumed`), which the run's metrics go by."""

    def label(self) -> str:
        return self.stated_label

    def describe(self) -> str:
        return self.stated_description

    def fingerprint(self) -> Mapping[str, object]:
        return self.stated_snapshot

    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    def asked(self, keys: frozenset[str]) -> Mapping[str, object]:
        """Those of its facts named by `keys`: as pinning read them, or, where it read none, asked of its
        store alone (`asked`), which opens no file for a fact its store cannot state (row 9009)."""
        if self.stated_metadata is None:
            return asked(self.source, keys)
        return {key: value for key, value in self.stated_metadata.items() if key in keys}

    @functools.cached_property
    def _metadata(self) -> Mapping[str, object]:
        return self.source.metadata() if self.stated_metadata is None else self.stated_metadata

    def supports(self, read: Read) -> bool:
        return read == self.read or self.source.supports(read)

    def records(self, read: Read) -> Records:
        held = None if self.held is None else self.held.records
        return held if read == "memory" and held is not None else self.source.records(read)

    def positioned_manifest(self) -> ShardManifest:
        """Its source's shards for a stream that numbers its rows (`stream_manifest`)."""
        return stream_manifest(self.source, positioned=True)

    @functools.cached_property
    def _lengths(self) -> np.ndarray | None:
        return self.source.lengths()

    def lengths(self) -> np.ndarray | None:
        return self._lengths

    def order_fingerprints(self) -> Mapping[str, str]:
        return self.source.order_fingerprints() if isinstance(self.source, _OrderFacts) else {}

    def locality(self) -> tuple[tuple[int, ...], bool]:
        if not isinstance(self.source, _Locality):
            raise ValueError(f"{self.describe()} does not state physical shard locality")
        return self.source.locality()


def _released(pipelines: Sequence[Pipeline]) -> None:
    """Let go of the memory reads pinning took for `pipelines` (`_PinnedRead`), once every branch
    of their preparation is materialized: under each source a memory read made (`materialized`),
    its preparation's included, down to the pinned one."""
    for _, leaf in (leaf for pipeline in pipelines for leaf in _leaves(pipeline)):
        for source in beneath(leaf, ("read", "materialized")):
            if isinstance(source, _Facts) and source.held is not None:
                source.held.records = None


def _pin_sources(pipelines: Sequence[Pipeline], read: Read) -> tuple[Pipeline, ...]:
    sources = {id(source): source for branch in pipelines for _, source in _leaves(branch)}
    measured: dict[int, dict[str, Measuring]] = {}
    for branch in pipelines:
        for key, measuring in _measured(branch).items():
            measured.setdefault(key, {}).update({each.measure.fact: each for each in measuring})
    # Each store held beside its records, so no store made later takes a freed one's address and records.
    stores: dict[int, tuple[Source, Records]] = {}
    reading = threading.Lock()

    def whole(store: Source) -> Records:
        # A store several holdout parts measure over is read once for them all.
        with reading:
            if id(store) not in stores:
                stores[id(store)] = (store, store.records(read))
            return stores[id(store)][1]

    def pin(source: Source) -> _Facts:
        if isinstance(source, _Facts):
            return source
        label = source.label()
        measuring = tuple(measured.get(id(source), {}).values())
        source = _consumed(source, read)
        records, metadata = _stated(source, read, measuring, whole)
        held = None if records is None else _PinnedRead(records)
        return _Facts(source, read, source.fingerprint(), metadata, source.describe(), held, label)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as readers:
        # Each read in this compilation, which fingerprints each source once (`compiling`).
        tasks = [within(functools.partial(pin, source)) for source in sources.values()]
        pinned = dict(zip(sources, readers.map(lambda task: task(), tasks), strict=True))

    return tuple(with_stores(branch, lambda source: pinned[id(source)]) for branch in pipelines)


def _measured(pipeline: Pipeline, above: tuple[Measuring, ...] = ()) -> dict[int, tuple[Measuring, ...]]:
    """The facts each store `pipeline` reads has measured for the configured steps over it, its own
    and its mixtures' (`Configured.measures`), each after the stages its store's records pass
    through before that step, by the store's identity."""
    passed = tuple(dataclasses.replace(each, before=(*pipeline.stages, *each.before)) for each in above)
    measuring = (*_measures(pipeline.stages), *passed)
    nested = children(pipeline)
    if not nested:
        return {id(pipeline.source): measuring}
    found: dict[int, tuple[Measuring, ...]] = {}
    for child in nested:
        for key, child_measuring in _measured(child, measuring).items():
            found[key] = (*found.get(key, ()), *child_measuring)
    return found


def _tokenizer_contract(pipelines: Sequence[Pipeline]) -> None:
    """Every corpus's tokenizer facts read alike (`tokenizer_contract`), each asked of its store alone
    (`asked`), in threads of their own as pinning reads them (`_pin_sources`; row 11638: one after
    another)."""
    corpora = [(f"corpus {_name(leaf)}", source)
               for pipeline in pipelines for leaf, source in _leaves(pipeline)]
    # One read a store, whatever corpora read it (as `_pin_sources` reads each once): a store's facts are
    # cached unlocked, which two threads reading one store at once would race on (row 11638's review).
    stores = {id(_store_of(source)): source for _, source in corpora}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(stores)))) as readers:
        facts = readers.map(lambda source: asked(source, TOKENIZER_CONTRACT), stores.values())
        read = dict(zip(stores, facts, strict=True))
    tokenizer_contract((name, read[id(_store_of(source))]) for name, source in corpora)


def _store_of(source: Source) -> Source:
    """The store a compile's pinned source reads (`_Facts`): itself where it is none."""
    return source.source if isinstance(source, _Facts) else source


def _row_tokens(pipeline: Pipeline) -> int | None:
    """The tokens each row of `pipeline` holds (`row_lengths`), its corpora alike; corpora whose rows hold
    several lengths are refused."""
    lengths = row_lengths(pipeline)
    if len(lengths) > 1:
        raise ValueError(f"a token budget needs rows of one length, and the corpora's are {set(lengths)}")
    (length,) = lengths
    return length


def _budgeted(pipeline: Pipeline, fallback: int | None = None) -> Pipeline:
    """`pipeline` with its token budget as the rows of fixed length it divides into."""
    budget = next((stage for stage in pipeline.stages if isinstance(stage, TokenBudget)), None)
    if budget is None:
        return pipeline
    length = _row_tokens(pipeline) or fallback
    if length is None:
        raise ValueError(
            f"token-budget pipeline {_name(pipeline)} needs a record step declaring TokenGeometry"
        )
    count = budget.rows(length)
    if count is None:
        raise ValueError(f"a token budget of {budget.tokens} is not whole rows of {length} tokens")
    rows = BatchStage(count, budget.drop_remainder, budget.batch_fn)
    return dataclasses.replace(
        pipeline, stages=tuple(rows if stage is budget else stage for stage in pipeline.stages)
    )


def _windowed(pipeline: Pipeline, objective: ObjectiveData) -> Pipeline:
    """`pipeline` with the objective's packing, and the steps over each window, before its batch."""
    if objective.pack is None:
        return pipeline
    at = _until(pipeline.stages, (BatchStage, TokenBudget))
    windows = dataclasses.replace(pipeline, stages=pipeline.stages[:at])._then(objective.pack)
    for step in objective.packed:
        windows = windows.map(step)
    return dataclasses.replace(windows, stages=(*windows.stages, *pipeline.stages[at:]))


_KEYED_ABOVE = (Map, RandomMap, Filter, BatchStage, Prefetch, Workers, Repeat, Limit, Slice, MapBatch)
"""The stages above a streamed mix that hold none of its rows across a saved position, so the mix's
members may be restored by key beneath them: a shuffle's window, a packer's remainder or an
expansion's tail holds rows of the mix that saved them."""


def _member_manifests(train: Pipeline, build: _Build, host: Sequence[HostStep]) -> dict[str, JSON]:
    """What each member of a streamed mix reads and makes, as one digest under its key
    (`_member_keys`): its order, sources, steps with their code and dependencies, the facts its
    order rests on and its caches; and under the empty key the same of the mixture's own stages
    and the compile's host steps. A saved position of the mix keeps each member by its key, so a
    resume takes a member joining or leaving and compares the others member by member
    (`dew.data.plan.Plan.check`). Nothing where a resume cannot: a pipeline that mixes nothing, a
    global order, or a mix beneath a stage that holds its rows (`_KEYED_ABOVE`).

    Each member is read in `train`, the tree the plan's lowering describes (`build`), which a
    memory read made of the written pipeline and which holds no member that accepted no row; its
    caches are those of each of its corpora's stages as its pipeline wrote them (`_written`), the
    ones a memory read already ran among them."""
    source = train.source
    if (not isinstance(source, Mixture) or build.global_order
            or not all(isinstance(stage, _KEYED_ABOVE) for stage in train.stages)):
        return {}

    def read_of(read: Source) -> Source:
        while materialized(read):
            read = read.source
        return read

    def steps(identities: Sequence[StepIdentity]) -> JSON:
        return [{**step.semantic(), "modules": list(step.modules)} for step in identities]

    corpora = build.lowering.order["corpora"]
    if not isinstance(corpora, list):
        raise TypeError("a mixture's order holds its corpora's orders as a list")
    members: dict[str, JSON] = {"": {"steps": steps([
        *(_identity(stage) for stage in written(train.stages) if not isinstance(stage, (Workers, Prefetch))),
        *(StepIdentity.of(step) for step in host)])}}
    for index, (key, child) in enumerate(zip(_member_keys(_members(source.pipelines)), source.pipelines,
                                             strict=True)):
        sources = [read for _, read in _leaves(child)]
        identities = [*(_identity(stage) for read in sources for stage in _written(read, ())),
                      *(_identity(stage) for stage in written(tuple(_steps(child)))
                        if not isinstance(stage, (Workers, Prefetch)))]
        members[key] = json_value({
            "order": unplaced(described(corpora[index])),
            "sources": [read_of(read).fingerprint() for read in sources],
            "steps": steps(identities),
            "derived": dict(_derived(child, build.child(index))),
            "cache": list(_cache_references(child, build.read)),
        }, "member manifest")
    return members


def _derived(pipeline: Pipeline, build: _Build) -> Mapping[str, str]:
    seed, read = build.seed, build.read
    arrays: dict[str, str] = {}
    for index, (leaf, source) in enumerate(_leaves(pipeline)):
        if isinstance(source, _Memory):
            arrays.update({f"{index}/{key}": value for key, value in source.derived.items()})
        # A store's lengths since `ASKED` where a step packs by them, its files opened for nothing else.
        packs = any(isinstance(stage, Pack) for stage in leaf.stages)
        lengths = source.lengths() if packs or rule_in_force() < ASKED else None
        if isinstance(source, _OrderFacts):
            arrays.update({f"{index}/{key}": value for key, value in source.order_fingerprints().items()})
        if lengths is not None and f"{index}/lengths" not in arrays:
            arrays[f"{index}/lengths"] = fingerprint(lengths)
        if any(isinstance(stage, Shuffle) and stage.block_shards is not None for stage in leaf.stages):
            if not isinstance(source, _Locality):
                raise ValueError(f"source {_name(leaf)} does not state physical shard locality")
            sizes, _ = source.locality()
            arrays[f"{index}/physical_shards"] = fingerprint(list(sizes))
        if read != "streamed":
            arrays[f"{index}/index"] = fingerprint([len(source.records(read)), seed])
        elif identified(json_value(source.fingerprint(), "source")):
            # The `shards` fact: the shards a stream reads in turn (`ShardManifest.digest`), of a store
            # that states its content (a store given no name names its shards by no content of its own).
            arrays[f"{index}/shards"] = stream_manifest(
                source, positioned=any(isinstance(stage, MapWithIndex) for stage in leaf.stages)).digest
        metadata = asked(source, ("valid_targets",))
        if "valid_targets" in metadata:
            arrays[f"{index}/valid_targets"] = fingerprint(
                integer(metadata["valid_targets"], "valid_targets")
            )
    splits = list(_split_spans(pipeline, build))
    if splits:
        arrays["token_spans"] = fingerprint(json_value(splits, "token spans"))
    return arrays


def _prepared(
    branches: Sequence[Pipeline],
    read: Read,
    *,
    provision: bool,
    training: bool = False,
    objective: ObjectiveData | None = None,
    widths: Mapping[str, object] | None = None,
    counts: Sequence[dict[str, CorpusCounts]] | None = None,
    automatic: str | None = None,
    examples: int | None = None,
    measured: _Measured | None = None,
    sealing: bool = False,
) -> tuple[Pipeline, ...]:
    """The one preparation compile, iterate, validation and publish lower from.

    With `training` the first branch trains; every other branch is scored.
    Each branch is checked, its generations are promoted to their sources and
    it is checked again; its sources are provisioned and pinned to the facts
    they state once, objective steps take their places, and configured steps
    bind to the sources' metadata. With read='memory' and `counts`, the
    source-scoped steps run and materialize first, so run-scoped steps bind
    to measured facts; publication passes no counts and reads nothing before
    its fill. The caller materializes what remains. A training read caches
    each corpus's leading deterministic steps under `automatic` (`_automatic`). A training read bounded
    to `examples` source examples takes its bound as the objective's steps are placed, before any step
    runs, so no eager pass expands an example the bound counts (`_bounded`). Branches a run's widths
    already measured this way (`measured`, `_measure`) are bound and laid out as they stand. A `sealing`
    pass (a snapshot's) is scored, but draws as its records stand in the one order it seals: no draw of its
    is held to a record's canonical index (row 10567).
    """
    if measured is None:
        measured = _measure(branches, read, provision=provision, training=training, objective=objective,
                            counts=counts, automatic=automatic, examples=examples, sealing=sealing)
    branches, metadata = measured.branches, measured.metadata
    held = [not training or index > 0 for index in range(len(branches))]
    if metadata is not None:
        # A run's static widths, measured over every branch it reads, bind this one's run-scoped steps
        # (`run_widths`): a validation split takes training's shapes, not its own.
        branches = tuple(_configured(branch, {**metadata, **(widths or {})}, read) for branch in branches)
    if objective is not None:
        branches = tuple(_windowed(branch, objective) for branch in branches)
    if any(isinstance(stage, TokenBudget) for branch in branches for stage in branch.stages):
        # A token budget's rows are the training rows' fixed length (`_budgeted`).
        trained = _row_tokens(branches[0])
        branches = tuple(_budgeted(branch, trained) for branch in branches)
    for branch, scored in zip(branches, held, strict=True):
        _analyzed(branch, read, validation=scored, sealing=sealing)
    return branches


@dataclasses.dataclass(frozen=True)
class _Measured:
    """Branches prepared through their source's own steps (`_measure`), materialized where a memory read
    runs them: the facts they state then (`metadata`, None where no step binds to facts), which a run's
    widths merge (`run_widths`) and its preparation binds (`_prepared`), and the counts that read filled."""

    branches: tuple[Pipeline, ...]
    metadata: Mapping[str, object] | None
    counts: tuple[dict[str, CorpusCounts], ...]
    ahead: str | None = None
    """Where its training read is bounded ahead of what its memory would hold (`_held_ahead`): its
    expansion, to a count set once it holds the examples (`_UNCOUNTED`, `_staged`), or its pack, the bound
    held with it, its passes counted as it holds them (`Limit.passes`); None for neither."""
    counted: int | None = None
    """The source examples one pass holds of a memory read a run counting its passes marks ahead of the
    expansion it holds (`_counted`), counted before it held them; None for any other read."""


@dataclasses.dataclass(frozen=True)
class _TrainingMeasured(_Measured):
    """A training read measured (`_training_measure`), with the mirror counters its statistics count from:
    taken before it read anything, as a memory read downloads its corpus as it is measured, before any
    reader is built (rows 5220, 5258). A phased run's statistics are its first phase's (`_compile_phases`),
    counting from that phase's measure: what this process's mirrors do from there on, its later phases'
    reads among them, and none an earlier reader made."""

    mirrored: Mirrored = dataclasses.field(kw_only=True)


def _measure(
    branches: Sequence[Pipeline],
    read: Read,
    *,
    provision: bool,
    training: bool = False,
    objective: ObjectiveData | None = None,
    counts: Sequence[dict[str, CorpusCounts]] | None = None,
    automatic: str | None = None,
    examples: int | None = None,
    epochs: int | None = None,
    counting: bool = False,
    sealing: bool = False,
) -> _Measured:
    """`_prepared`'s first part: each branch checked, its generations promoted and checked again, its
    sources provisioned and pinned, the objective's steps placed, a training read's bound taken, and with
    read='memory' and `counts` its source-scoped steps run and materialized, so the facts the branches
    state are measured ones; publication passes no counts and reads nothing before its fill."""
    if read not in ("memory", "indexed", "streamed"):
        raise ValueError(f"unknown read mode {read!r}")
    held = [not training or index > 0 for index in range(len(branches))]
    for branch, scored in zip(branches, held, strict=True):
        _analyzed(branch, read, validation=scored, promotion=True, sealing=sealing)
    branches = tuple(_promoted(branch) for branch in branches)
    for branch, scored in zip(branches, held, strict=True):
        _analyzed(branch, read, validation=scored, sealing=sealing)
    _capabilities(branches, read, provision,
                  [branch for branch, scored in zip(branches, held, strict=True) if scored])
    branches = _pin_sources(branches, read)
    _tokenizer_contract(branches)
    branches = tuple(_automatic(branch, read, automatic) for branch in branches)
    if objective is not None:
        branches = tuple(
            _objective_branch(branch, objective, train=not scored, read=read)
            for branch, scored in zip(branches, held, strict=True)
        )
    # A memory read whose expansion it would hold is bounded before that expansion as it holds the rest, to
    # a count set once it holds them (`_staged`).
    counting = counting and training and _countable(branches[0])
    ahead = None if (epochs is None and not counting) or not training else _held_ahead(branches[0], read)
    counted = None
    if counting:
        # A run that counts its passes marks a memory read's examples ahead of the expansion it holds
        # (`_counted`), its pass the source examples reaching the mark, counted before it holds them; it
        # holds no pack's passes ahead, reading on without end, and bounds nothing.
        counted = _pass_examples(branches[0], read, None, passes=True) if ahead == "expansion" else None
        ahead = "expansion" if counted is not None else None
        if counted is not None:
            branches = (_counted(branches[0]), *branches[1:])
    else:
        examples = _UNCOUNTED if ahead is not None else examples
        if examples is not None and training:
            branches = (_bounded(branches[0], examples), *branches[1:])
            if ahead == "layout":
                branches = (_rebound(branches[0], _UNCOUNTED, epochs), *branches[1:])
    if any(isinstance(stage, Configured) for branch in branches for stage in _steps(branch)):
        def stored(branch: Pipeline) -> bool:
            return any(len(source.records(read)) for _, source in _leaves(branch))

        if training and read == "memory" and not stored(branches[0]):
            raise ValueError(f"no accepted rows of {_name(branches[0])}; source is empty")
        metadata = _metadata(branches, read)
        if read == "memory" and counts is not None:
            branches = tuple(
                _materialize(_configured(branch, metadata, read, "source"), count, training=not scored,
                             sealing=sealing)
                for branch, count, scored in zip(branches, counts, held, strict=True)
            )
            if training and not stored(branches[0]):
                raise ValueError(f"no accepted rows of {_name(branches[0])}; raise the row budgets")
            metadata = _metadata(branches, read)
        return _Measured(tuple(branches), metadata, tuple(counts or ()), ahead, counted)
    return _Measured(tuple(branches), None, tuple(counts or ()), ahead, counted)


def _made(prepared: Sequence[Pipeline], read: Read, counts: Sequence[dict[str, CorpusCounts]], *,
          training: bool, sealing: bool = False) -> tuple[Pipeline, ...]:
    """Prepared branches as a memory read makes them, the first training where `training`: their
    first stages run and their rows held (`_materialize`), counted into `counts`, then the memory
    reads pinning took let go (`_released`); another read's as they are."""
    if read != "memory":
        return tuple(prepared)
    made = tuple(_materialize(branch, count, training=training and not index, sealing=sealing)
                 for index, (branch, count) in enumerate(zip(prepared, counts, strict=True)))
    _released(prepared)
    return made


def _own_key(branch: Pipeline) -> int | None:
    """A named chain's own key, read outside any mixture: the key a mixture keys it by as its first member
    of that name (`_member_key`), which its shuffles draw by from plan version 9 (`_Build.ordered`), so it
    reads one order mixed and alone; None for a chain given no name."""
    return None if branch.name is None else _member_key(_members([branch]), 0)


def _keys_shuffles(build: _Build) -> bool:
    """Whether `build` shuffles its chain by the chain's own name (`_own_key`): from plan version 9, a named
    chain whose order shuffles at its root; one that shuffles nothing reads alike under any name."""
    return build.named is not None and any(
        isinstance(stage, Shuffle) for stage in build.lowering.stages)


def _build(branch: Pipeline, read: Read, loading: Loading, *, batch: int, training: bool, seed: int = 0,
           lowering: _Lowering | None = None) -> _Build:
    """The build a reader of a prepared `branch` reads it by, in batches of `batch`: training's, or a
    scored or published pass's, by the lowering its analysis gives unless `lowering` says otherwise, and
    the branch's own key where it is named (`_own_key`)."""
    lowered = _analyzed(branch, read, validation=not training) if lowering is None else lowering
    pooled = None if lowered.cache is None else cache_prefix(branch, lowered.cache).pool
    if read == "indexed" and pooled is not None:
        # An indexed read fills its cache as its threads read the cached inputs, so a thread pool among the
        # cached steps (a process pool there is refused as the read is analyzed) reads with that many.
        threads = max(loading.threads, pooled.size)
        loading = dataclasses.replace(loading, threads=threads, read_buffer=max(loading.read_buffer, threads))
    return _Build(read, loading, seed, batch, lowered, training=training, elastic=branch._elastic,
                  named=_own_key(branch))


@compiling
def compile(
    train: Dataset,
    *,
    validation: Dataset | None = None,
    host: tuple[HostStep, ...] = (),
    objective: ObjectiveData | None = None,
    read: Read | None = None,
    provision: bool = False,
    loading: Loading | None = None,
    seed: int = 0,
    rejections: Literal["drop", "mask"] = "drop",
    resume: DataResume = "relaxed",
    examples: int | None = None,
    widths: Mapping[str, object] | None = None,
    measured: _TrainingMeasured | tuple[_TrainingMeasured, ...] | None = None,
    epochs: int | None = None,
    steps: int | None = None,
    tokens: int | None = None,
    count_passes: bool = False,
) -> Loader:
    """Check every leaf before opening readers, then build both branches; `widths` are the run's
    static widths its steps bind to (`run_widths`), and `measured` its training branch as their
    measurement prepared it (`_training_measure`), which it binds as it stands: of a run of phases, each
    phase's, which a phase that reads its corpora from their start binds (`_compile_phases`).
    With no `read`, the sources choose it (`_preferred_read`). `examples` bounds the training read to
    its first that many source examples, before they are packed and before the objective makes rows
    of them (`_bounded`), a run's `examples=` budget. `epochs` reads the chain endlessly, or, where its
    read has no record count, bounds it to that many passes over the source examples it states a pass
    holds once it is prepared (`_staged`, `Loader.pass_examples`). `steps` or `tokens` is the
    run's length where it is no bound of the read's own, which automatic loading sizes its workers for
    (`_run_batches`). `count_passes`, of a run bounded otherwise whose cadence reads its passes (an
    `epoch` or `passes` cadence), counts the source examples its read consumes where its read has no
    record count and states a pass's (`Loader.pass_examples`, `_staged`), its read marking them with no
    bound; a run whose cadence reads none is compiled as it was, its plan and batches unchanged."""
    # A dataset a stage returned as a dataset (`Dataset.pipe`) is read as its chain.
    train, validation = train._chain(), None if validation is None else validation._chain()
    branches = (train,) if validation is None else (train, validation)
    read = read or train.mode or _preferred_read(branches, provision=provision)
    loading = loading or train._loading or _TRAINING_LOADING
    if isinstance(train.source, Phases) and examples is None:
        phased = _compile_phases(
            train,
            validation=validation,
            host=host,
            objective=objective,
            read=read,
            provision=provision,
            loading=loading,
            seed=seed,
            rejections=rejections,
            resume=resume,
            steps=steps,
            budget=tokens,
            widths=widths,
            measured=measured if isinstance(measured, tuple) else None,
        )
        return dataclasses.replace(phased, resume=resume)
    if isinstance(measured, tuple):
        raise TypeError("a measure of each phase binds a run of phases alone")
    counting = count_passes and examples is None and epochs is None
    staged = _staged(train, validation, objective=objective, read=read, provision=provision, loading=loading,
                     seed=seed, rejections=rejections, examples=examples, widths=widths, measured=measured,
                     epochs=epochs, counting=counting)
    batches = _run_batches(staged, steps=steps, examples=examples, epochs=epochs, tokens=tokens)
    compiled = _compiled(staged, _coordinated(staged.train), host=host, resume=resume, batches=batches)
    loader = dataclasses.replace(compiled, resume=resume)
    if staged.passes is not None and epochs is not None:
        return dataclasses.replace(loader, examples=epochs * staged.passes, epochs=epochs,
                                   pass_examples=staged.passes, bound_ends=_bound_ends(staged.original))
    if staged.passes is not None and counting:
        return dataclasses.replace(loader, pass_examples=staged.passes)
    if examples is None:
        return loader
    return dataclasses.replace(loader, examples=examples, bound_ends=_bound_ends(staged.original))


def _bound_ends(pipeline: Pipeline) -> bool:
    """Whether `pipeline`'s read, bounded in examples (`_bounded`), ends where its bound does: no limit or
    slice after the bound ends it first (`Loader.bound_ends`)."""
    at = examples_bound(pipeline.stages)
    after = () if at is None else pipeline.stages[at + 1:]
    return not any(isinstance(stage, (Limit, Slice)) for stage in after)


def _training_measure(train: Pipeline, validation: Pipeline | None, *, objective: ObjectiveData | None,
                      read: Read, provision: bool, loading: Loading, examples: int | None,
                      epochs: int | None = None, counting: bool = False) -> _TrainingMeasured:
    """A compile's training branch, and its `validation`, prepared through their source's own steps as
    the compile reads them (`_measure`): bounded to `examples`, a chain's own layout as written, before
    preparation could plan it into its source, and the objective's with its preparation; each corpus's
    leading deterministic steps cached where the read is no stream; their counts filled, and its mirror
    counters taken before it reads (`_TrainingMeasured.mirrored`)."""
    from dew.data.cache import automatic_root

    since = Mirrored.now()
    written = None if examples is None or not _lays_out(train) else examples
    if written is not None:
        train = _bounded(train, written)
    counts: dict[str, CorpusCounts] = {}
    validation_counts: dict[str, CorpusCounts] = {}
    measured = _measure(
        (train,) if validation is None else (train, validation),
        read,
        provision=provision,
        training=True,
        objective=objective,
        counts=(counts,) if validation is None else (counts, validation_counts),
        automatic=automatic_root(loading),
        examples=examples if written is None else None,
        epochs=epochs,
        counting=counting,
    )
    return _TrainingMeasured(**{field.name: getattr(measured, field.name)
                                for field in dataclasses.fields(measured)}, mirrored=since)


@dataclasses.dataclass(frozen=True)
class _Staged:
    """A run's branches prepared, and materialized for a memory read, before
    their readers are built (`_compiled`): `original` is the training branch
    before a memory read ran it."""

    train: Pipeline
    validation: Pipeline | None
    original: Pipeline
    build: _Build
    counts: dict[str, CorpusCounts]
    validation_counts: dict[str, CorpusCounts]
    batch_tokens: int | None
    passes: int | None = None
    """The source examples one pass of a training read bounded to its epochs, or counting its passes,
    holds (`_staged`); None for one not bounded or counted so."""
    mirrored: Mirrored = dataclasses.field(kw_only=True)
    """The mirror counters its run counts from (`_TrainingMeasured.mirrored`)."""


def _staged(
    train: Pipeline,
    validation: Pipeline | None,
    *,
    objective: ObjectiveData | None,
    read: Read,
    provision: bool,
    loading: Loading,
    seed: int,
    rejections: Literal["drop", "mask"],
    examples: int | None = None,
    widths: Mapping[str, object] | None = None,
    measured: _TrainingMeasured | None = None,
    epochs: int | None = None,
    counting: bool = False,
) -> _Staged:
    if rejections == "mask":
        raise ValueError(
            "rejections='mask' needs an objective proven mask-safe; no objective currently declares it"
        )
    stated = next((stage for stage in train.stages if isinstance(stage, TokenBudget)), None)
    batch_tokens = None if stated is None else stated.tokens
    if measured is None:
        measured = _training_measure(train, validation, objective=objective, read=read, provision=provision,
                                     loading=loading, examples=examples, epochs=epochs, counting=counting)
    counts, validation_counts = (*measured.counts, {})[:2]
    branches = _prepared(measured.branches, read, provision=provision, training=True, objective=objective,
                         widths=widths, measured=measured)
    if not any(isinstance(stage, BatchStage) for stage in branches[0].stages):
        raise ValueError("compile needs one batch(global_size) stage on the training pipeline")
    original = branches[0]
    train, *scored = _made(branches, read, (counts, validation_counts)[: len(branches)], training=True)
    if read == "memory":
        warn_rejections(counts, loading.warn_over)
    validation = scored[0] if scored else None
    # A memory read that kept none of the scored records scores none.
    if validation_counts and not any(count.accepted for count in validation_counts.values()):
        validation = None
    build = _build(train, read, loading, batch=_output_batch(original), training=True, seed=seed)
    passes = None
    if epochs is not None and measured.ahead == "layout":
        # The bound it held with its pack counted its passes as it held the documents (`Limit.passes`). The
        # prepared chain states that count where it holds the bound unrun; one its measure held already (a
        # configured chain's, `_measure`) ran it to that count.
        count = _held_count(train)
        passes = count // epochs
        if examples_bound(original.stages) is not None:
            original = _rebound(original, count)
    elif epochs is not None and measured.ahead == "expansion":
        # Bounded before the expansion it would hold, it now holds the examples a pass reaches it with.
        passes = _pass_examples(train, read, build, passes=False)
        if passes is not None:
            original, train = _rebound(original, epochs * passes), _rebound(train, epochs * passes)
            build = _build(train, read, loading, batch=_output_batch(original), training=True, seed=seed)
    elif epochs is not None:
        # Epochs of a read with a record count count it, read endlessly; of one with none, the source
        # examples a pass holds, its prepared rows as held, bounded to that many passes (`_bounded`).
        passes = None if epoch_records(train, build) is not None else _pass_examples(train, read, build,
                                                                                     passes=True)
        original, train = ((_endless(original), _endless(train)) if passes is None else
                           (_bounded(original, epochs * passes, passes=True),
                            _bounded(train, epochs * passes, passes=True)))
        build = _build(train, read, loading, batch=_output_batch(original), training=True, seed=seed)
    elif counting and measured.ahead == "expansion":
        # Marked ahead of the expansion its read holds (`_measure`): a pass the source examples it held.
        passes = measured.counted
    elif counting:
        # A run bounded in steps whose cadence counts its passes (`compile`'s `count_passes`): where its read
        # has no record count but states a pass's source examples, each one it consumes is marked where a
        # pass is counted, with no bound (`_counted`), its read as it would be counting none.
        passes = (None if epoch_records(train, build) is not None or not _countable(train)
                  else _pass_examples(train, read, build, passes=True))
        if passes is not None:
            original, train = _counted(original), _counted(train)
            build = _build(train, read, loading, batch=_output_batch(original), training=True, seed=seed)
    return _Staged(train, validation, original, build, counts, validation_counts, batch_tokens, passes,
                   mirrored=measured.mirrored)


_UNCOUNTED = UNCOUNTED
"""The bound in examples a memory read whose expansion it would hold takes before it holds them, past any
count, set to its epochs' passes once it holds them (`_rebound`, `_staged`)."""


def _pass_examples(train: Pipeline, read: Read, build: _Build | None, *, passes: bool) -> int | None:
    """The source examples one pass of a training read with no record count holds (`Loader.pass_examples`,
    `dew.data.epochs.epoch_records`): those reaching where a bound in examples stands (`_bounded`:
    the first step that packs or windows them or makes several rows of each, else the batch), before any
    repeat. Its source's records, as a memory read holds them (its leading steps run, the present ones),
    or a streamed source's rows as its shards state or count them at the build's plan version
    (`ShardManifest.sized`), each slice and limit before that place applied;
    a mixture by weights, those in which every part it mixes comes round once (`Proportions.covering`); a
    concatenation, its members' added. None
    for a mixture by tokens, a memory read that laid its examples out as it held them, or where a
    step before that place drops or adds one, or keeps them by a rule not yet bound, or nothing states
    the count. With `passes`, that place is where a run's epochs bound its pass (`_bounded`), before a step
    that rejects a record it cannot make; a mixture's or concatenation's members are counted as their rows
    leave them, where the bound after the mix counts them, so a member that rejects counts none ahead."""
    source = train.source
    if isinstance(source, Phases):
        return None
    at = _one_example_until(train, passes=passes)
    # Every step before the bound's place reads the pass, a repeat's included; its slices and limits before
    # the first repeat bound it.
    if any(_drops(stage) or isinstance(stage, Native)
           or (isinstance(stage, Configured) and not stage.one_to_one) for stage in train.stages[:at]):
        return None
    # A memory read that laid its examples out as it held them (a pack before its shuffle) holds no source
    # example a bound before that layout could count, as a run bounded in examples refuses it too.
    held = () if isinstance(source, Combined) else _written(source, ())
    if any(isinstance(stage, (Pack, Windows)) or (isinstance(stage, FlatMap) and stage.examples)
           for stage in held):
        return None
    prefix = train.stages[:min(at, _until(train.stages, (Repeat,)))]
    if isinstance(source, Mixture):
        if source.tokens is not None:
            # A mix by tokens draws each part as its tokens fall short (`TokenDeficit`), which no count of its
            # examples states ahead.
            return None
        # A member's pass where the bound after the mix counts it: its rows as they leave the member.
        sizes = [_pass_examples(child, read, build, passes=False) for child in source.pipelines]
        known = [size for size in sizes if size is not None]
        if len(known) != len(sizes):
            return None
        count = Proportions.of(_weights(source), source.seed).covering(known)
    elif isinstance(source, Concatenated):
        # Its members' passes one after another, each read once; one a member repeats is no pass of its own.
        sizes = [None if any(isinstance(stage, Repeat) for stage in _steps(child))
                 else _pass_examples(child, read, build, passes=False) for child in source.pipelines]
        known = [size for size in sizes if size is not None]
        if len(known) != len(sizes):
            return None
        count = sum(known)
    elif read == "streamed" and not _in_memory(source):
        if build is None:
            return None
        stated = [shard.rows for shard in
                  source.manifest().sized(build.stream_loading.threads).shards]
        held = [rows for rows in stated if rows is not None]
        if len(held) != len(stated):
            return None
        count = sum(held)
    else:
        records = source.records(read)
        present = _Present.of(records) if _in_memory(source) else None
        count = len(records) if present is None else len(present.positions)
    for stage in prefix:
        if isinstance(stage, Limit):
            count = min(count, stage.count)
        elif isinstance(stage, Slice):
            count = len(range(count)[stage.selection])
    return count


def _held_ahead(train: Pipeline, read: Read) -> str | None:
    """Where a run of epochs bounds a memory read before it holds anything, as a run bounded in examples
    does (`_bounded`), its bound standing among the steps the read runs over its whole source (`_EAGER`):
    "expansion" where the read would make rows of its source examples as it holds them, the bound then
    ahead of that step, its count set once the read holds the rest (`_UNCOUNTED`, `_rebound`); "layout"
    where it would pack or window them as it holds them and reads the windows per share after (an
    objective's rows), the bound held before the layout, its passes counted as the read holds the documents
    (`Limit.passes`).
    None for any other read, whose pass is counted once it is prepared (`_staged`)."""
    if read != "memory" or isinstance(train.source, Combined):
        return None
    # The steps the read runs as it holds its rows, as the analysis places them once its configured steps
    # bind (after a cache it fills), a configured step among them where it binds to such a step.
    eager = _analyzed(train, read, bound=True).eager
    at = _one_example_until(train)
    if at >= eager:
        return None
    if _expands(train.stages[at]):
        return "expansion"
    after = dataclasses.replace(train, stages=train.stages[eager:])
    laid = isinstance(train.stages[at], (Pack, Windows))
    return "layout" if laid and _analyzed(after, read).per_share else None


def _rebound(pipeline: Pipeline, examples: int, passes: int | None = None) -> Pipeline:
    """`pipeline`, bounded (`_bounded`), its bound's limit `examples`, or `passes` over the rows reaching
    it, which a memory read holding them counts (`Limit.passes`)."""
    at = examples_bound(pipeline.stages)
    assert at is not None and isinstance(pipeline.stages[at - 1], Limit)
    return dataclasses.replace(pipeline, stages=(*pipeline.stages[:at - 1], Limit(examples, passes),
                                                 *pipeline.stages[at:]))


def _held_count(train: Pipeline) -> int:
    """The examples a memory read's held bound resolved to (`Limit.passes`), as the stages it ran state."""
    source = train.source
    ran = () if isinstance(source, Combined) else _written(source, ())
    at = examples_bound(ran)
    limit = None if at is None else ran[at - 1]
    assert isinstance(limit, Limit), "a held bound is among the stages it ran"
    return limit.count


def _output_batch(pipeline: Pipeline) -> int:
    """The rows each batch of `pipeline` holds as its reader yields them, which its loader declares
    and its share pads to."""
    return next(stage.size for stage in pipeline.stages if isinstance(stage, BatchStage))


def _validation_build(build: _Build, validation: Pipeline, read: Read) -> _Build:
    """The build a validation pass of `validation` reads by: training's loading as the chain states it,
    not as training resolved it, sized for its own pass (`_Build.tuned` over `pass_batches`): a reader opens
    each pass afresh and starts its workers again, so a pass whose Python pays for no start reads in the
    process, where training's count started up to a worker a batch of every pass (train 746's review)."""
    held = dataclasses.replace(build, lowering=_analyzed(validation, read, validation=True), training=False,
                               batch=_output_batch(validation), named=_own_key(validation))
    passed = held.pass_batches(validation) if held.loading.workers == "auto" else None
    return held.tuned(validation, batches=passed)


def _run_batches(staged: _Staged, *, steps: int | None, examples: int | None, epochs: int | None,
                 tokens: int | None) -> int | None:
    """The batches a run reads, which automatic loading sizes its workers for (`Loading.resolved`): its
    `steps`; an `examples` bound's batches; `epochs` passes over its records, or over the source examples
    a pass holds where its read counts no records (`_Staged.passes`); a `tokens` budget over the most a
    batch holds (its stated token batch, else its rows times the tokens a row holds, `_row_tokens`), the
    fewest batches it may take; None where nothing states its length, which reads in the process."""
    batch = staged.build.batch
    if steps is not None:
        return steps
    if examples is not None:
        return -(-examples // batch)
    if epochs is not None:
        records = staged.passes if staged.passes is not None else epoch_records(staged.train, staged.build)
        return None if records is None else epochs * -(-records // batch)
    if tokens is not None:
        row = _row_tokens(staged.train)
        held = staged.batch_tokens or (None if row is None else batch * row)
        return None if not held else -(-tokens // held)
    return None


def _compiled(
    staged: _Staged,
    packed: tuple[Coordinated, ...],
    *,
    host: tuple[HostStep, ...],
    resume: DataResume,
    batches: int | None = None,
) -> Loader:
    """The readers of `staged`, each training batch's packed windows laid out
    as `packed` declares them and each validation batch's as its own packs do
    (`_held_coordinated`)."""
    train, validation, original, build = staged.train, staged.validation, staged.original, staged.build
    counts, validation_counts = staged.counts, staged.validation_counts
    read, seed, batch, global_order = (
        build.read,
        build.seed,
        build.batch,
        build.global_order,
    )
    sources = [source for _, source in _leaves(original)]
    # The plan of this read, whole: its manifest, its facts, its members' digests, its caches and its rules.
    described = [stage for stage in written(tuple(_steps(original)))
                 if not isinstance(stage, (Workers, Prefetch))]
    def planned() -> Plan:
        """The plan, its steps described under the field rule in force (`dew.data.plan.described_under`),
        which a position saved under an earlier version is compared with (`Plan.under`)."""
        manifest = _manifest({"sources": [source.fingerprint() for source in sources]}, sources,
                             (*map(_identity, described), *map(StepIdentity.of, host)))
        members = _member_manifests(train, build, host)
        return Plan(
            "global" if global_order else "per-share",
            manifest,
            # A named chain's own key, which its shuffles draw by (`_own_key`), so a position saved under
            # another name is refused, as another permutation of its records.
            {"read": read, "seed": seed, **({"named": build.named} if _keys_shuffles(build) else {}),
             **build.lowering.order},
            _derived(train, build),
            cache=_cache_references(original, read),
            members={key: fingerprint(member) for key, member in members.items()},
            rules=build.rules(train),
            volatile=any(_volatile(stage) for stage in (*_steps(original), *(
                stage for source in sources for stage in _written(source, ())))),
            # The steps it describes read a saved plan's earlier descriptions of them (`ReadsEarlier`).
            earlier=earlier_readers(described, host),
            # Rebuilt under an earlier rule, it is that rule's plan and builds no other.
            rebuilt=planned if rule_in_force() >= ASKED else None,
        )

    plan = planned()
    tuned = build.tuned(train, batches=batches)
    # The training read's workers move before what drops records (`pooled`); validation reads its own.
    build = tuned.pooled(train)
    train_host, device = _batch_steps(train)
    train_host = packed + train_host + host
    statistics = _Statistics(counts, staged.mirrored)
    training = functools.partial(
        _training_reader,
        pipeline=train,
        build=build,
        plan=plan,
        host=train_host,
        resume=resume,
        statistics=statistics,
    )
    held_reader = (
        None
        if validation is None
        else functools.partial(
            _validation_reader,
            pipeline=validation,
            build=_validation_build(staged.build, validation, read),
            host=_held_coordinated(validation, packed) + _held_steps(validation) + host,
        )
    )
    records = epoch_records(train, build)
    if records == 0:
        raise ValueError(f"corpus {_name(train)} has no accepted rows")
    ordered = order_records(train, build)
    if ordered is None and _native_after_bound(train):
        # A native Grain step rebuilds or reorders rows as it likes, so the examples each row consumes
        # (`CONSUMED`), which a run no plan counts ahead reads its progress off, are kept by no contract.
        raise ValueError(f"{_name(train)} runs a native Grain step (`grain`) after its examples bound, which "
                         "keeps no count of the examples its rows consume: run it before the bound, as a "
                         "step Dew names, or read the data in a global order")
    held_out = sum(integer(asked(source, ("held_out",)).get("held_out", 0), "held_out")
                   for _, source in _leaves(train))
    return Loader(
        training,
        held_reader,
        records,
        batch,
        held_out=held_out,
        counts=counts,
        validation_counts=validation_counts,
        plan=plan,
        statistics=statistics,
        batch_tokens=staged.batch_tokens,
        device=device,
        fields=_fixed_fields(original, train_host),
        widths=_bound_widths(original),
        tail=tail(train, build),
        order_records=ordered,
    )


def _fixed_fields(pipeline: Pipeline, host: Sequence[HostStep]) -> Mapping[str, tuple[int | None, ...]]:
    """Each field's per-row shape the batches the training stream hands its step hold (`Loader.fields`), as
    the facts state them (`Dataset.facts`): `pipeline`'s record stages, then the batch steps its reader runs
    on the host (`host`: a pack's layout, the chain's own and the compile's), a step that does not say what
    it makes leaving no field fixed; never a device step's, which the step runs after the check (row 9533:
    a batch was held to a device crop it had not run, and without a host step's drop it had). The
    producer's shape, which every batch is held to (docs/design/training.md, the producer's contract)."""
    from dew.data.facts import _after, folded

    records = dataclasses.replace(pipeline, stages=tuple(stage for stage in pipeline.stages
                                                         if not isinstance(stage, MapBatch)))
    made = functools.reduce(lambda facts, step: _after(MapBatch(step), facts), host, folded(records))
    stated = made.get("fields", {})
    return {name: tuple(shape) for name, shape in record(stated, "fields").items()
            if isinstance(shape, list) and all(axis is None or isinstance(axis, int) for axis in shape)}


def _bound_widths(pipeline: Pipeline) -> Mapping[str, object]:
    """The run's static widths `pipeline`'s run-scoped configured steps bound (`Bound`, `run_widths`)."""
    bound = (ran.configured for ran in map(runs, _steps(pipeline))
             if ran is not None and ran.configured is not None and ran.configured.step.scope == "run")
    return {name: value for binding in bound for name, value in binding.values.items()}


def _batch_steps(pipeline: Pipeline) -> tuple[tuple[HostStep, ...], tuple[DeviceStep, ...]]:
    """The training batch's steps: those the reader runs on the host, then
    those the compiled step runs on the accelerator (`DeviceStep`)."""
    steps = [stage.step for stage in pipeline.stages if isinstance(stage, MapBatch)]
    return (tuple(step for step in steps if not isinstance(step, DeviceStep)),
            tuple(step for step in steps if isinstance(step, DeviceStep)))


def _coordinated(*pipelines: Pipeline) -> tuple[Coordinated, ...]:
    """The step that lays each packed window's documents out as the model
    reads them (`Coordinated`), one per packed ids field, declaring the most
    documents a window of `pipelines` holds (`Coordinates.max_segments`),
    which a sequence kernel's work follows. A plan's windows hold as many as
    it placed (`PackedSource`) and a memory read's as many as it packed; a
    pack that runs as rows arrive holds its windows to its cap
    (`Pack.segments`), or to its window's length where it has none: a window
    holds no more documents than tokens. A corpus that shows no packing
    while another packs into the same batches holds windows packed where
    none of this can see, so then each declares its window's length."""
    fields: dict[str, tuple[str, ...]] = {}
    bounds: dict[str, int] = {}
    laid: dict[str, bool] = {}
    unseen = False

    def declare(pack: Pack, most: int | None, laid_out: bool = True) -> None:
        ids = pack.fields[0]
        fields[ids] = tuple(dict.fromkeys((*fields.get(ids, ()), *pack.fields)))
        bound = pack.seq_len if most is None else max(1, min(pack.seq_len, most))
        bounds[ids] = max(bounds.get(ids, 0), bound)
        # A field's windows laid out by the read unless a step over a pack's windows lays them out itself.
        laid[ids] = laid.get(ids, True) and laid_out

    def walk(pipeline: Pipeline, *, packed: bool, itself: bool = False) -> None:
        nonlocal unseen
        # A window step that lays its windows out (`LaysOutWindows`), after any pack, a plan's or a source's.
        itself = itself or any(isinstance(stage, Map) and isinstance(stage.step, LaysOutWindows)
                               for stage in pipeline.stages)
        for at, stage in enumerate(pipeline.stages):
            if isinstance(stage, Pack):
                floor = _floor(pipeline, at, stage)
                declare(stage, stage.segments if floor is None else min(
                    stage.segments or stage.seq_len, max(1, stage.seq_len // floor)), not itself)
                packed = True
        source = pipeline.source
        if isinstance(source, Combined):
            for child in source.pipelines:
                walk(child, packed=packed, itself=itself)
            return
        documents = _packed_documents(source)
        written = _written(source, ())
        itself = itself or any(isinstance(stage, Map) and isinstance(stage.step, LaysOutWindows)
                               for stage in written)
        for stage in written:
            if isinstance(stage, Pack):
                declare(stage, stage.segments if documents is None else documents, not itself)
                packed = True
        *_, source = beneath(source, ("read", "materialized"))
        if isinstance(source, PackedLayout):
            declare(*source.packed_layout(), not itself)
            packed = True
        unseen = unseen or not packed

    for pipeline in pipelines:
        walk(pipeline, packed=False)
    return tuple(Coordinated.laid(fields[ids], None if unseen else bound, laid[ids])
                 for ids, bound in bounds.items())


def _held_coordinated(validation: Pipeline, training: tuple[Coordinated, ...]) -> tuple[Coordinated, ...]:
    """A validation pass's `Coordinated` steps: its own packs' declarations,
    and for a field only the training packs it, a window's length, since a
    validation source may hold packed windows whose packing it does not show."""
    held = _coordinated(validation)
    declared = {step.fields[0] for step in held}
    return held + tuple(Coordinated.laid(step.fields, None, step.laid_out) for step in training
                        if step.fields[0] not in declared)


def _floor(pipeline: Pipeline, at: int, pack: Pack) -> int | None:
    """The fewest tokens along `pack`'s ids field of a record the pack at `at` receives, where a memory read
    holds the records its stages before it read (`_Memory`) and each of those stages hands a record on as it
    is (a view of its inputs, `dew.data.analysis._VIEWS`, a bound's marks) but at most one, a step that states
    its floor (`Shortest`): each held record's length, or that step's floor of it. A pack's window then holds
    at most `seq_len` over it documents, the most a memory read that cannot run its pack (a random step before
    it) declares, a fitting pack cutting no document. None where it cannot be read so: a concatenating pack,
    which cuts documents where its windows end, a stream, a held record still encoded, or another step
    between."""
    source = pipeline.source
    if pack.policy == "concat" or not isinstance(source, _Memory) or not source.held:
        return None
    stating: Shortest | None = None
    for stage in pipeline.stages[:at]:
        if isinstance(stage, _VIEWS) or (isinstance(stage, Map)
                                                  and isinstance(stage.step, MarkConsumed)):
            continue
        ran = runs(stage)
        if stating is not None or ran is None or not isinstance(ran.step, Shortest):
            return None
        stating = ran.step
    field = pack.fields[0]
    floors = []
    for row in source.held:
        # A stating step makes the field from what it reads (an encoding, its tokens); else the row holds it.
        if not isinstance(row, Mapping) or (stating is None and field not in row):
            return None
        floors.append(len(row[field]) if stating is None else stating.shortest(row, field))
    return max(1, min(floors))


def _packed_documents(source: Source) -> int | None:
    """The most documents a window of a memory read's packed windows holds, None where it packed none."""
    return source.segments if isinstance(source, _Memory) else None


def _held_steps(pipeline: Pipeline) -> tuple[HostStep, ...]:
    """A validation batch's host steps; a device step is a training transform."""
    host, device = _batch_steps(pipeline)
    if device:
        raise ValueError(f"{type(device[0]).__name__} runs in the training step; a validation dataset "
                         "takes its batches as they are read")
    return host


def _phase_batches(source: Phases, index: int, phase: _Staged, *, steps: int | None, budget: int | None,
                   tokens: bool) -> int | None:
    """The batches phase `index` of `source` reads, which automatic loading sizes its workers for
    (`Loading.resolved`): of a run of `steps`, its own steps on the step clock, else the run's; of a run of
    `budget` tokens, its own span of them on the token clock (`tokens`), else the run's, over the most a
    batch of it holds (`_run_batches`)."""
    start = (source.ends[index - 1] if index else 0) or 0
    until = source.ends[index]

    def span(length: int) -> int:
        return max(0, (length if until is None else min(until, length)) - start)

    if steps is not None:
        return steps if tokens else span(steps)
    if budget is None:
        return None
    return _run_batches(phase, steps=None, examples=None, epochs=None,
                        tokens=span(budget) if tokens else budget)


@dataclasses.dataclass(frozen=True)
class _Trained:
    """What the phases before one trained of a corpus, by its places, where its chain's first repeat reads
    it (its rows where it repeats nothing): every place before `low`, and `past`, each past it."""

    low: int = 0
    past: frozenset[int] = frozenset()

    def drawn(self, count: int) -> _Trained:
        """These and the next `count` places not among them, as a phase that reads on in order draws them."""
        skipped = sorted(place - self.low for place in self.past)
        # The fewest places from `low` that hold `count` not trained already.
        reach = count
        while (moved := count + bisect.bisect_left(skipped, reach)) != reach:
            reach = moved
        return _Trained(self.low + reach).joined(place for place in self.past if place >= self.low + reach)

    def joined(self, places: Iterable[int]) -> _Trained:
        """These and `places` (none before `low`)."""
        held, low = set(self.past).union(places), self.low
        while low in held:
            held.discard(low)
            low += 1
        return _Trained(low, frozenset(held))

    def skipped(self) -> tuple[int, ...]:
        """The places past `low` a phase continuing the corpus draws past, counted from `low`
        (`_Build.skipped`)."""
        return tuple(sorted(place - self.low for place in self.past))


def _consumed_by(phases: Sequence[_Staged], ends: Sequence[int]) -> dict[str, _Trained]:
    """What `phases` trained of each corpus, each ending at its record end of `ends`: where its order moves
    rows past where a corpus is placed (`_reorders`: a shuffle after a mix), the places its order's twin of
    draws (`_Build.drawn_order`) holds in its records up to its end; else the counts the lowering's mix draws
    its members by in that many records, each read on in order from where the phases before left it."""
    consumed: dict[str, _Trained] = {}
    for phase, (start, end) in zip(phases, itertools.pairwise((0, *ends)), strict=True):
        if _reorders(phase.train):
            order = _drawn_order(phase.build, phase.train)
            places: dict[str, list[int]] = {}
            for index in range(end - start):
                match order(index):
                    case (str() as name, int() as place):
                        places.setdefault(name, []).append(place)
                    case drawn:
                        raise TypeError(f"a twin of draws holds a corpus and its place, not {drawn!r}")
            for name, held in places.items():
                consumed[name] = consumed.get(name, _Trained()).joined(held)
        else:
            _delivered(phase.train, end - start, consumed)
    return consumed


def _drawn_order(build: _Build, pipeline: Pipeline) -> Callable[[int], tuple[str, int]]:
    """The twin of draws of the global order a training read of `pipeline` by `build` counts its records in
    (`_Build.batch_order`), by index: the corpus and the place its `index`th record was drawn at,
    where the corpus's chain first repeats (`place_of`); batched and through the stages after its batch
    that hold a place in it as its batches are (`_Build.placed`), each record its batch's. The twin is no
    run's training read, so automatic loading reads it in the process (`Loading.unsized`), as a phase's
    build is handed it before a read sizes its workers."""
    build = dataclasses.replace(build, loading=build.loading.unsized())
    places = _drawn_records(build, pipeline)
    if build.lowering.rooted:
        places = _repeated(places, None, _name(pipeline), build.loading)
    stage = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage))
    batched = pipeline.stages.index(stage)
    after = tuple(range(batched + 1, len(pipeline.stages)))
    batches = build.placed(places.batch(stage.size, batch_fn=list), pipeline, DataPartition(), after)

    def record(index: int) -> tuple[str, int]:
        held = batches[index // stage.size]
        if held is None:
            raise IndexError(f"record {index} lies past the order's last batch")
        return held[index % stage.size]
    return record


def _drawn_records(build: _Build, pipeline: Pipeline) -> pygrain.MapDataset:
    """`pipeline`'s rows before its batch as their twin of draws: a corpus's rows read to where its places are
    (`place_of`), each its place from there, and a mix's draws of its corpora's twins as its read draws them
    (`mixed_indexed`), each through the steps after that move rows (`_drawn_after`)."""
    source = pipeline.source
    if isinstance(source, Mixture):
        members = [_drawn_records(build.child(index), child) for index, child in enumerate(source.pipelines)]
        return _drawn_after(build, mixed_indexed(members, source), pipeline, 0)
    repeat = place_of(pipeline)
    placed = len(pipeline.stages) if repeat is None else repeat + 1
    if build.lowering.cache is not None and build.lowering.cache >= placed:
        raise ValueError(f"{_name(pipeline)} is cached after its first repeat, where a phase that continues "
                         "it places its records; a phase of one global order continues a corpus cached "
                         "before it repeats, as a cache of a corpus's own rows is written")
    rows = build.records(dataclasses.replace(pipeline, stages=pipeline.stages[:placed]), DataPartition())
    if not isinstance(rows, pygrain.MapDataset):
        raise TypeError(f"{_name(pipeline)} is no global order, whose records a twin of draws follows")
    name = _name(pipeline)
    return _drawn_after(build, pygrain.MapDataset.range(len(rows)).map(lambda place: (name, place)), pipeline,
                        placed)


def _drawn_after(build: _Build, places: pygrain.MapDataset, pipeline: Pipeline, start: int
                 ) -> pygrain.MapDataset:
    """`places`, a global order's twin of draws where `pipeline`'s stage `start` reads it, through the stages
    from there to its batch that move rows as `build`'s read runs them (`_reordered`, `_repeated`, and a
    continued corpus's `_Build.skipping`); a step that makes rows moves none."""
    for index, stage in enumerate(pipeline.stages[start:], start=start):
        if isinstance(stage, (BatchStage, MapBatch)):
            break
        if record_stage_of(stage) or isinstance(stage, (Prefetch, Workers)):
            continue
        moved = build.stage(places, stage, pipeline, DataPartition(), index)
        if not isinstance(moved, pygrain.MapDataset):
            raise ValueError(f"{type(stage).__name__} in {_name(pipeline)} moves the rows of a global order "
                             "in no way its twin of draws follows")
        places = moved
    return places


def _reorders(pipeline: Pipeline) -> bool:
    """Whether `pipeline`'s order moves its rows past the places its corpora are drawn at (`_Trained`): a
    shuffle, slice, take or repeat after a mix, or a shuffle after a corpus's first repeat, where the records
    it read hold no prefix of each corpus's draws."""
    source = pipeline.source
    if isinstance(source, Mixture):
        return (any(isinstance(stage, (Shuffle, Slice, Limit, Repeat)) for stage in pipeline.stages)
                or any(_reorders(child) for child in source.pipelines))
    repeat = place_of(pipeline)
    return repeat is not None and any(isinstance(stage, Shuffle) for stage in pipeline.stages[repeat + 1:])


def _built_phases(source: Phases, staged: Sequence[tuple[_Staged, int | None]],
                  packed: tuple[Coordinated, ...], *,
                  host: tuple[HostStep, ...], resume: DataResume, steps: int | None,
                  budget: int | None, tokens: bool) -> list[tuple[Loader, int | None]]:
    """Each staged phase compiled (`_compiled`) at the batches it reads (`_phase_batches`), beside its end:
    every phase of a run runs the first's device steps, so a phase whose own differ is refused."""
    built: list[tuple[Loader, int | None]] = []
    for index, (phase, end) in enumerate(staged):
        batches = _phase_batches(source, index, phase, steps=steps, budget=budget, tokens=tokens)
        loaded = _compiled(phase, packed, host=host, resume=resume, batches=batches)
        if built and loaded.device != built[0][0].device:
            # The run applies one set of device steps to every batch (`Loader.device`), so a phase
            # whose own differ would run the first phase's.
            raise ValueError(f"phase {index} ends in other device steps than phase 0; every phase of "
                             f"a run runs the same ones, written after the phases")
        built.append((loaded, end))
    return built


def _measured_phase(measured: tuple[_TrainingMeasured, ...] | None, index: int, continued: Pipeline,
                    phases: Sequence[Pipeline], validation: Pipeline | None) -> _TrainingMeasured | None:
    """The measure among `measured` (each phase's, as `run_widths` read them) that phase `index` binds where
    it reads its corpora from their start (`continued`, the phase as written) with no validation beside it;
    None for one continued past an earlier phase, measured as it continues (row 6713: each phase was
    measured twice, both copies of a memory read held at the start's peak)."""
    fresh = measured is not None and continued is phases[index] and validation is None
    return measured[index] if fresh and measured is not None else None


def _compile_phases(
    train: Pipeline,
    *,
    validation: Pipeline | None,
    host: tuple[HostStep, ...],
    objective: ObjectiveData | None,
    read: Read,
    provision: bool,
    loading: Loading,
    seed: int,
    rejections: Literal["drop", "mask"],
    resume: DataResume,
    steps: int | None = None,
    budget: int | None = None,
    widths: Mapping[str, object] | None = None,
    measured: tuple[_TrainingMeasured, ...] | None = None,
) -> Loader:
    """`train`'s phases, each compiled on its own, every phase and the validation binding the run's
    `widths` (`run_widths`), measured over every phase and held chain, one program for them all, a phase
    binding its measure among `measured` where it may (`_measured_phase`)."""
    if not isinstance(train.source, Phases):
        raise TypeError("phased compilation needs a Phases source")
    source: Phases = train.source

    _tokenizer_contract((train,) if validation is None else (train, validation))

    def staged_at(index: int, consumed: Mapping[str, _Trained], members: Mapping[str, str] | None = None
                  ) -> _Staged:
        """Phase `index`, each corpus it reads continued past what the phases before it trained of it
        (`_continued`, `_skipping`): on its own stream, those it shares with other phases stamped (`members`,
        `_stamped`)."""
        continued, alone = _continued(source.pipelines[index], consumed), validation if index == 0 else None
        phase = _staged(continued._then(*train.stages), alone, objective=objective, read=read,
                        provision=provision, loading=loading, seed=seed, rejections=rejections, widths=widths,
                        measured=_measured_phase(measured, index, continued, source.pipelines, alone))
        return _skipping(_stamped(phase, members or {}), consumed, source.pipelines[index])

    tokens = source.clock == "tokens"
    first = staged_at(0, {})
    # Phases on the step clock whose first reads per share (its order cut where each share reads it: a
    # stream batched by its tokens after steps that drop rows) read each its own stream (`StepPhases`).
    per_share = not tokens and not first.build.global_order
    source = _alone_mixed(source) if per_share else source
    shared = _shared(source) if per_share else tuple({} for _ in source.pipelines)
    staged = (_own_streams(source, first, shared, staged_at) if per_share
              else _in_one_order(source, first, staged_at))
    # Every phase declares the most documents a window of any phase holds, so the run steps one program.
    packed = _coordinated(*(phase.train for phase, _ in staged))
    built = _built_phases(source, staged, packed, host=host, resume=resume, steps=steps, budget=budget,
                          tokens=tokens)

    # The loaders the run's reads opened, in the order they opened them, whose counts its statistics are
    # (`statistics_of`): the first phase's from the start, each later one's as the run reaches it.
    reading: dict[int, _Statistics] = {}

    def counted(loader: Loader) -> Loader:
        """`loader`, its counts among the run's from now on."""
        if not isinstance(loader.statistics, _Statistics):
            raise TypeError(f"a phase's loader counts by {type(loader.statistics).__name__}, not its reads")
        reading.setdefault(id(loader.statistics), loader.statistics)
        return loader

    counted(built[0][0])

    @functools.cache
    def opened(index: int, ends: tuple[int, ...]) -> Loader:
        """Phase `index` as it reads where the phases before it ended at `ends`, in records (`resolved`)."""
        phase = resolved(index, ends)
        if phase is staged[index][0]:
            return counted(built[index][0])
        batches = _phase_batches(source, index, phase, steps=steps, budget=budget, tokens=tokens)
        return counted(_compiled(phase, packed, host=host, resume=resume, batches=batches))

    @functools.cache
    def resolved(index: int, ends: tuple[int, ...]) -> _Staged:
        """Phase `index` staged where the phases before it ended at `ends`, in records: as staged, unless a
        corpus it reads continues from what those phases, each resolved alike, trained of it."""
        if not tokens or not index:
            return staged[index][0]
        consumed = _consumed_by([resolved(before, ends[:before]) for before in range(index)], ends)
        if all(consumed.get(_name(chain), _Trained()) == _Trained()
               for chain in _corpora(source.pipelines[index])):
            return staged[index][0]
        return staged_at(index, consumed)

    batches = tuple(phase.build.batch for phase, _ in staged)
    stages = () if tokens or per_share else phase_stages(
        batches, (0, *(end for _, end in staged[:-1] if end is not None)))
    # A read of phases each on its own stream states its own (`StepPhases.stages`).

    def training(partition: DataPartition) -> Iterator[Batch]:
        if not tokens:
            _updates_aligned(source.ends, batches, partition.window)
        if per_share:
            readers = [functools.partial(loaded.train, partition) for loaded, _ in built]
            return StepPhases(readers, source.ends, batches=batches,
                              opened=lambda index: counted(built[index][0]) and None, shared=shared)

        def global_reader(index: int, ends: tuple[int, ...]) -> GlobalReader:
            opened_reader = opened(index, ends).train(partition)
            if not isinstance(opened_reader, GlobalReader):
                if isinstance(opened_reader, Closeable):
                    opened_reader.close()
                raise ValueError("a phase must expose a global record-count position")
            return opened_reader

        phases = [(functools.partial(global_reader, index), end) for index, (_, end) in enumerate(built)]
        return PhasedStream(phases, loading.stop_seconds, batches=batches, stages=stages,
                            tokens=source.ends if tokens else None)

    # Only the last phase's order can end, so its batch decides what an epoch's short batch does.
    def statistics() -> Mapping[str, float]:
        return statistics_of(tuple(reading.values()))

    return dataclasses.replace(built[0][0], train=training, statistics=statistics, tail=built[-1][0].tail,
                               order_records=None,
                               token_ends=source.ends if tokens else None, batches=batches, stages=stages,
                               token_batches=tuple(loaded.batch_tokens for loaded, _ in built),
                               widths=merge_facts(loaded.widths for loaded, _ in built))


def _in_one_order(source: Phases, first: _Staged, staged_at: Callable[[int, Mapping[str, _Trained]], _Staged]
                  ) -> list[tuple[_Staged, int | None]]:
    """Each phase of `source` read in one global order, staged beside its end in records: `first` for phase
    0, the rest by `staged_at` past what the phases before them trained of each corpus (`_consumed_by`). A
    phase on the token clock ends where the run reaches it, so it is staged as it reads first, and again with
    the ends of the phases before it once the run reaches them (`_compile_phases`' `opened`)."""
    tokens = source.clock == "tokens"
    staged: list[tuple[_Staged, int | None]] = []
    for index, until in enumerate(source.ends):
        ends = [end for _, end in staged if end is not None]
        phase = _ordered(index, first if not index else staged_at(
            index, {} if tokens else _consumed_by([phase for phase, _ in staged], ends)))
        # A phase on the step clock ends at its step's record: the records of the phases before it, and its
        # own steps in its own rows (row 4812).
        before, started = (ends[-1] if ends else 0), (source.ends[index - 1] or 0) if index else 0
        staged.append((phase, None if tokens or until is None
                       else before + (until - started) * phase.build.batch))
    return staged


def _ordered(index: int, phase: _Staged) -> _Staged:
    """Phase `index` of phases read in one global order, refused where it reads in none: a phase after it
    continues each corpus past the records it read by its order's shares (`consumed_by`), which a read
    streamed, rejecting or packed online has none of; on the step clock, a run whose first phase reads per
    share reads each phase on its own stream (`StepPhases`), its phases' too."""
    if not phase.build.global_order:
        raise ValueError(f"phase {index} needs a global order, as the phases before it read in one; "
                         "streamed, rejecting and online-packed phases are read each on its own stream on "
                         "the step clock where the first phase is one too, and refused on the token clock")
    return phase


def _own_streams(source: Phases, first: _Staged, shared: Sequence[Mapping[str, str]],
                 staged_at: Callable[[int, Mapping[str, _Trained], Mapping[str, str]], _Staged]
                 ) -> list[tuple[_Staged, int | None]]:
    """Each phase of `source` read on its own stream (`StepPhases`), staged beside its end, its step's:
    `first` for phase 0 where it shares nothing, the rest by `staged_at` with the corpora each shares
    (`shared`); phases that share corpora hand their members' rows on in groups of the first one's batch
    alike (`_Build.grouped`)."""
    grouped = first.build.batch if any(shared) else None
    staged: list[tuple[_Staged, int | None]] = []
    for index, until in enumerate(source.ends):
        phase = first if not index and not shared[0] else staged_at(index, {}, shared[index])
        if shared[index]:
            phase = dataclasses.replace(phase, build=dataclasses.replace(phase.build, grouped=grouped))
        staged.append((phase, until))
    return staged


def _stamped(staged: _Staged, members: Mapping[str, str]) -> _Staged:
    """`staged`, a phase on its own stream, its mix stamping each draw of the members it shares with other
    phases (`members`, `_Build.stamped`)."""
    if not members:
        return staged
    return dataclasses.replace(staged, build=dataclasses.replace(staged.build, stamped=frozenset(members)))


def _updates_aligned(ends: Sequence[int | None], batches: Sequence[int], window: int = 1) -> None:
    """Refuse phases on the step clock where one whose batch differs from the one before it begins inside
    an update of `window` steps (`DataPartition.window`): an update pools one batch's microbatches, so the
    run would drop the part of the update read in the batch before it (`Trainer._staged`)."""
    for index, (until, batch, after) in enumerate(zip(ends, batches, batches[1:], strict=False)):
        if after != batch and until is not None and until % window:
            raise ValueError(f"phase {index + 1} reads batches of {after} rows after phase {index}'s "
                             f"{batch}, from step {until}, inside an update of {window} steps, which pools "
                             f"one batch's microbatches; end phase {index} at a multiple of {window} steps")


def _continued(pipeline: Pipeline, consumed: Mapping[str, _Trained]) -> Pipeline:
    """`pipeline` reading each corpus on from where the phases before left it (`consumed`): from the first
    place they did not train (a slice after its chain's first repeat), the places past it they did drawn past
    by its build (`_skipping`)."""
    if isinstance(pipeline.source, Phases):
        raise ValueError("nested phases are not an indexed record order")
    if isinstance(pipeline.source, Mixture):
        return with_children(pipeline, functools.partial(_continued, consumed=consumed))
    trained = consumed.get(_name(pipeline), _Trained())
    offset = trained.low
    if not offset and not trained.past:
        return pipeline
    at = None if (repeat := place_of(pipeline)) is None else repeat + 1
    if at is None:
        pipeline = pipeline.repeat()
        at = len(pipeline.stages)
    return dataclasses.replace(
        pipeline, stages=(*pipeline.stages[:at], Slice(slice(offset, None)), *pipeline.stages[at:])
    )


def _skipping(staged: _Staged, consumed: Mapping[str, _Trained], pipeline: Pipeline) -> _Staged:
    """`staged`, the phase `pipeline` continued past `consumed` (`_continued`), drawing past the places of its
    corpora the run trained already (`_Build.skipped`); a corpus it does not read it records nothing of."""
    reads = {_name(chain) for chain in _corpora(pipeline)}
    skipped = {name: held.skipped() for name, held in consumed.items() if held.past and name in reads}
    if not skipped:
        return staged
    return dataclasses.replace(staged, build=dataclasses.replace(staged.build, skipped=skipped))


def _delivered(pipeline: Pipeline, count: int, consumed: dict[str, _Trained]) -> None:
    source = pipeline.source
    if isinstance(source, Phases):
        raise ValueError("nested phases have no single corpus share")
    if isinstance(source, Mixture):
        # The counts the indexed mix reads its corpora by (`ProportionalSelection`).
        counts = Proportions.of(_weights(source), source.seed).counts(count)
        for child, taken in zip(source.pipelines, counts, strict=True):
            _delivered(child, taken, consumed)
    else:
        name = _name(pipeline)
        consumed[name] = consumed.get(name, _Trained()).drawn(count)


def _training_reader(
    partition: DataPartition,
    *,
    pipeline: Pipeline,
    build: _Build,
    plan: Plan,
    host: tuple[HostStep, ...],
    resume: DataResume,
    statistics: _Statistics,
) -> Iterator[Batch]:
    partition.rows(build.batch)
    if plan.volatile and partition.readers > 1:
        raise ValueError("volatile pipeline requires one reader per share until its bytes are persisted")
    if plan.volatile and build.elastic and partition.count > 1:
        # Every share reads an elastic order's prefix whole on its own (`_elastic_cut`), so a source
        # whose rows can differ between reads (a fetch that drops failed records) orders them apart.
        raise ValueError(f"an elastic read's {partition.count} shares each read its whole order, which this "
                         "pipeline's volatile source can read differently on each; persist its bytes first "
                         "(`Pipeline.cache`), or read it without `.elastic()`")
    workers = (_workers(pipeline, build.loading)
               or any(isinstance(stage, Workers) and stage.count for stage in _steps(pipeline)))
    if plan.volatile and build.elastic and build.read == "streamed" and workers:
        # Each loading worker reads an elastic stream whole and keeps its groups by their place in it
        # (`dew.data.elastic_order._StreamOrder`), so a volatile step that drops a record in one worker
        # alone moves every later group's rows there: groups overlap and leave rows out.
        raise ValueError(f"{pipeline.name or 'this chain'} is read elastically and streamed in loading "
                         "workers, each reading the whole stream, which this pipeline's volatile source can "
                         "read differently in each; persist its bytes first (`Pipeline.cache`), or read it "
                         "with Loading(workers=0)")
    _one_shape(host, partition)
    opened: Iterator[Batch]
    if build.global_order:
        opened = GlobalStream(
            functools.partial(build.batches, pipeline, partition),
            build.batch,
            plan,
            build.loading,
            resume=resume,
            endless=build.lowering.endless,
            published=statistics.mirrored(),
        )
    else:
        # The plan its positions name: its rules its build's, and Grain's, whose iterator state its position
        # holds (`GRAIN`).
        own_plan = dataclasses.replace(
            plan, rules={**build.rules(pipeline), "grain": GRAIN},
            # An elastic read's order is one reader's at any count, its position every reader's.
            topology={"elastic": True} if build.elastic else {
                "shares": partition.count,
                "share": partition.index,
                # A read in its inputs' order in any count of workers (`_Build.in_order`) names none.
                **({} if build.in_order else {"workers": _workers(pipeline, build.loading)}),
                "row_multiple": partition.row_multiple,
                "cycle_lengths": [
                    stage.cycle_length for stage in _steps(pipeline) if isinstance(stage, Interleave)
                ],
                # Where a read that drops records reads in workers of each corpus's own (`pooled`),
                # each worker's position is that corpus's, saved in that many.
                **({} if build.pool is None or build.in_order else {"pools": build.pools(pipeline)}),
            })
        opened = _PerShare(
            functools.partial(build.batches, pipeline, partition),
            own_plan,
            resume,
            stop_seconds=build.loading.stop_seconds,
            endless=build.lowering.endless,
            replicated=build.elastic,
        )
        statistics.observe(opened)
    return _Host(opened, host) if host else opened


def _validation_reader(
    partition: DataPartition, *, pipeline: Pipeline, build: _Build, host: tuple[HostStep, ...]
) -> Iterator[Batch]:
    _one_shape(host, partition)
    opened: Iterator[Batch]
    if not build.global_order:
        build = dataclasses.replace(build, loading=dataclasses.replace(build.loading, workers=0))
    # A token budget is fixed-length rows by now (`_budgeted`).
    if build.read == "streamed":
        opened = _CanonicalPass(
            functools.partial(build.batches, pipeline, fill=False), partition, build.batch
        )
    else:
        opened = build.batches(pipeline, partition)
    return _Host(opened, host) if host else opened


def _one_shape(host: tuple[HostStep, ...], partition: DataPartition) -> None:
    """Refuse more than one share to a host step that shapes each batch from
    its own rows (`SingleReader`), since each share's batch would take a
    shape of its own."""
    for step in host:
        if partition.count > 1 and isinstance(step, SingleReader) and step.single_reader:
            raise ValueError(f"host stage {type(step).__name__} requires one reader for its batch shape")


def objective_steps(pipeline: Pipeline, objective: ObjectiveData, *, train: bool, read: Read) -> Pipeline:
    """Put objective record steps on each corpus before its mixture, where the
    mixture is batched as it is; after a mixture's own steps (a preset's
    tokenize and pack over raw stores), on the mixture."""
    batched = _until(pipeline.stages, (BatchStage, TokenBudget))
    own = any(isinstance(stage, (Map, RandomMap, FlatMap, Filter, Pack, Windows, Native, Batched))
              for stage in pipeline.stages[:batched])
    if isinstance(pipeline.source, Combined) and not own:
        return with_children(pipeline, functools.partial(objective_steps, objective=objective, train=train,
                                                         read=read))
    boundary = (
        (Shuffle, Repeat, BatchStage, TokenBudget)
        if read == "memory"
        else (BatchStage, TokenBudget)
    )
    # The objective's steps read the rows a pack or window makes, never the documents it packs.
    layouts = [index + 1 for index, stage in enumerate(pipeline.stages) if isinstance(stage, (Pack, Windows))]
    laid = max(layouts, default=0)
    at = laid + _until(pipeline.stages[laid:], boundary)
    suffix = pipeline.stages[at:]
    # Rows the chain lays out hold the length it gives them, which the objective's geometry gives way to:
    # a pack's or windows', or a source's that reads them laid out (a planned pack, a store's windows).
    lays = layouts or laid_out(pipeline) is not None
    geometry = () if lays or objective.geometry is None else (objective.geometry,)
    pipeline = _objective_records(dataclasses.replace(pipeline, stages=pipeline.stages[:at]),
                                  (*(objective.train if train else objective.evaluate), *geometry))
    return dataclasses.replace(pipeline, stages=(*pipeline.stages, *suffix))


def _objective_records(
    pipeline: Pipeline, steps: Sequence[RecordStep | pygrain.RandomMapTransform | Expansion | Configured]
) -> Pipeline:
    """Append objective record steps at the caller's preparation boundary, each step that makes several
    rows of a source example marked as one (`FlatMap.examples`)."""
    def staged(step) -> Stage:
        if isinstance(step, Configured):
            # One stated to expand records binds to an expansion that carries the mark (`_bind`).
            return dataclasses.replace(step, examples=True) if step.expands else step
        stage = record_stage(step)
        return dataclasses.replace(stage, examples=True) if isinstance(stage, FlatMap) else stage
    return pipeline._then(*(staged(step) for step in steps))


def _objective_branch(pipeline: Pipeline, objective: ObjectiveData, *, train: bool, read: Read) -> Pipeline:
    """Prepare each corpus, then attach objective randomness, and the steps from the first that makes
    several rows of a record on, after the user's record order: the user's order (its shuffle and
    mixture) is of source examples, and a run's `examples=` bound counts them there (`_bounded`)."""
    steps = objective.train if train else objective.evaluate
    random = next((index for index, step in enumerate(steps)
                   if isinstance(step, (pygrain.RandomMapTransform, Expansion))
                   or (isinstance(step, Configured) and (step.random or step.expands))), len(steps))
    prefix = dataclasses.replace(objective, train=objective.train[:random],
                                 evaluate=objective.evaluate[:random])
    branch = objective_steps(pipeline, prefix, train=train, read=read)
    at = _until(branch.stages, (BatchStage, TokenBudget))
    suffix = branch.stages[at:]
    branch = dataclasses.replace(branch, stages=branch.stages[:at])
    branch = _objective_records(branch, steps[random:])
    branch = dataclasses.replace(branch, stages=(*branch.stages, *suffix))
    if not branch.batched:
        # A held chain measured before its run's batch is known takes the objective's batch steps where
        # it is batched (`_batch_steps_after`, `held`).
        return branch
    return _batch_steps_after(branch, objective, train=train)


def _batch_steps_after(branch: Pipeline, objective: ObjectiveData, *, train: bool) -> Pipeline:
    """`branch` with the objective's batch steps after its batch."""
    steps = objective.train_batch if train else objective.evaluate_batch
    # A string or object field stays in the batch for the host code and metrics that read it; the
    # prefetcher places the rest (`dew.distributed.batches.split_fields`).
    return branch._then(*(MapBatch(step) for step in steps))


def _metadata(pipelines: Sequence[Pipeline], read: Read) -> Mapping[str, object]:
    # A fact a configured step measures binds a mixture's step to the largest of its stores'.
    measured = frozenset(each.measure.fact for pipeline in pipelines
                         for each in _measures(tuple(_steps(pipeline))))
    return merge_facts((source.metadata() for pipeline in pipelines for _, source in _leaves(pipeline)
                        if not (isinstance(source, _Memory) and not source.held)), measured=measured)


@compiling
def _diagnostic(pipeline: Pipeline, *, records: int | None = None) -> _Host:
    """Inspect through training's lowering, finite unless a repeat asks otherwise.

    A bounded record pass runs lazily without loader read-ahead or
    eager memory transforms, and leaves its counters in the Grain state.
    """
    read = pipeline.mode or _preferred_read((pipeline,), provision=True)
    loading = (pipeline._loading or _DEFAULT_LOADING).unsized()
    if records is not None:
        pipeline = pipeline._then(Limit(records))
        # Serial where the chain states no loading; the loading it states reads as written (data.md, stats;
        # row 9058: a stated one was read serially too, so a pass measured another reader than it asked).
        if pipeline._loading is None:
            loading = dataclasses.replace(loading, workers=0, threads=0, read_buffer=0)
    if not pipeline.batched:
        pipeline = pipeline.batch(1, drop_remainder=False)
    counts: dict[str, CorpusCounts] = {}
    finite = not any(isinstance(stage, Repeat) for stage in _steps(pipeline))
    if finite:
        # A finite pass keeps a token budget's last rows; its budget becomes rows of fixed length.
        pipeline = dataclasses.replace(pipeline, stages=tuple(
            dataclasses.replace(stage, drop_remainder=False) if isinstance(stage, TokenBudget) else stage
            for stage in pipeline.stages))
    prepared = _prepared((pipeline,), read, provision=True, training=True,
                         counts=(counts,) if records is None else None)
    # A bounded pass reads lazily: nothing runs ahead of it in memory.
    (pipeline,) = prepared if records is not None else _made(prepared, read, (counts,), training=True)
    packed = _coordinated(pipeline)
    lowering = _analyzed(pipeline, read)
    if records is not None:
        lowering = _cut(lowering, per_share=True)
    if finite:
        # A non-repeated diagnostic pass, without training's root repeats.
        lowering = _once(lowering)
    # A token budget is fixed-length rows by now (`_budgeted`).
    build = _build(pipeline, read, loading, batch=_output_batch(pipeline), training=True,
                   lowering=lowering).tuned(pipeline).pooled(pipeline)
    return _Host(build.batches(pipeline, DataPartition()), packed + _batch_steps(pipeline)[0])


@compiling
def iterate(
    pipeline: Pipeline,
    *,
    read: Read = "streamed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
    seed: int | None = None,
) -> pygrain.IterDataset:
    """A finite record pass through the same lowering, for publication and export, in the process:
    automatic workers (`Loading.unsized`) and those the chain names (row 5735), whose turns would
    interleave the records out of the order the pass publishes. A pass that seals what it makes (a
    snapshot's, `seed` given) draws its random steps' values under `seed`, which the rows it seals keep
    (row 9056: a video's clips drawn once and sealed); publication, whose cache a later pass writes
    through, holds no draw."""
    loading = loading.unsized()
    pipeline = _in_process(pipeline)
    if (seed is None and draws(pipeline)) or any(isinstance(stage, (BatchStage, MapBatch, Repeat))
                                                 for stage in _steps(pipeline)):
        raise ValueError("publication needs deterministic, unbatched pipelines without repeat; a snapshot "
                         "seals a random step's draws (`snapshot(url, seed=...)`)")
    # A sealing pass draws as its records stand in the one order it seals them in, wherever its steps reorder
    # or drop them before a draw (a filter, a shuffle): validation's rule, a draw keyed by the record's
    # canonical index, is a scored split's, which a later pass must draw alike (row 10567).
    sealing = seed is not None
    (pipeline,) = _made(_prepared((pipeline,), read, provision=provision, counts=({},), sealing=sealing),
                        read, ({},), training=False, sealing=sealing)
    # Publication reads records, unbatched.
    build = _build(pipeline, read, loading, batch=1, training=False, seed=0 if seed is None else seed,
                   lowering=_analyzed(pipeline, read, validation=True, sealing=True) if sealing else None)
    rows = build.records(pipeline, DataPartition())
    return _as_iter(rows, loading)


def _in_process(pipeline: Pipeline) -> Pipeline:
    """`pipeline` and each chain it mixes without the loading workers they name (`Workers`)."""
    held = with_children(pipeline, _in_process)
    kept = tuple(stage for stage in held.stages if not isinstance(stage, Workers))
    return dataclasses.replace(held, stages=kept)


type Handed = pygrain.MapDataset | pygrain.IterDataset | Mapping[str, np.ndarray]
"""Data handed to `fit` as it is, which a run steps on as given (`native`): a Grain dataset's
elements, or arrays whole."""


class _Whole:
    """The same batch every step, the arrays a run was handed whole; its position the steps read. It
    never runs out (`Identified.ends`), and names what it reads as a stream Dew names no further does:
    by its type (`stream_identity`)."""

    ends: Ends = "never"

    def __init__(self, batch: Batch):
        self.batch, self.read = batch, 0

    def identity(self) -> str:
        return type(self).__name__

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        self.read += 1
        return self.batch

    def get_state(self) -> bytes:
        return str(self.read).encode()

    def set_state(self, state: bytes) -> None:
        self.read = int(state)


def _share(batch: Batch, partition: DataPartition, *, held: bool) -> Batch:
    """`partition`'s share of `batch` (`DataPartition`: rows `index :: count`); `held`, a validation
    share, filled out to every share's rows, a whole number of its row shards, its repeats marked
    (`DataPartition.held`, `FilledBatch`), so every share reads a batch its devices split where any
    does and every real row scores once."""
    rows = rows_of(batch)
    if not held:
        return batch if partition.count == 1 else take_batch(batch, np.arange(rows)[partition.cut()])
    mine, real = partition.held(rows) if partition.count > 1 else (slice(None), True)
    shards = partition.count * partition.row_multiple
    filled = -(-rows // shards) * partition.row_multiple
    return FilledBatch(filled, real=real).map(take_batch(batch, np.arange(rows)[mine]))


class _Shared(Forwarding):
    """The `partition`'s share of each batch a stream yields (`_share`), its position, stop and close
    the stream's; it never ends where it reads a map dataset repeated without end (`native`, `endless`),
    which says so of nothing it reads."""

    def __init__(self, batches: pygrain.DatasetIterator | _Whole | Replayed, partition: DataPartition, *,
                 held: bool, endless: bool = False):
        self._source, self.partition, self.held, self._endless = batches, partition, held, endless

    @property
    def ends(self) -> Ends:
        return "never" if self._endless else super().ends

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        return _share(next(self._source), self.partition, held=self.held)

    def get_state(self):
        return self._source.get_state()

    def set_state(self, state) -> None:
        self._source.set_state(state)


def native(handed: Handed, *, training: bool) -> Reader:
    """A reader of the data handed to `fit` as it is, each element a batch the run steps on, never
    batched again: a Grain dataset's elements (batched by Grain, `MapDataset.batch`), a map dataset's
    read endlessly where it trains, or a mapping of arrays read whole, every step the same batch (a
    full-batch run; a run of minibatches states them, `Memory(arrays).batch(size)`)."""
    if isinstance(handed, Mapping):
        batch = {name: np.asarray(column) for name, column in handed.items()}
        if training:
            return lambda partition: _Shared(_Whole(batch), partition, held=False)
        return lambda partition: iter([_share(batch, partition, held=True)])
    repeated = training and isinstance(handed, pygrain.MapDataset)
    stream: pygrain.IterDataset = (
        (handed.repeat() if training else handed).to_iter_dataset() if isinstance(handed, pygrain.MapDataset)
        else handed)
    return lambda partition: _Shared(iter(stream), partition, held=not training, endless=repeated)


def native_loader(handed: Handed) -> Loader:
    """The data handed to `fit` as it is (`native`) as the loader the run reads, its global batch
    its first element's rows. A Grain iterable is read once for that element: its run's first stream
    gives it again and reads on (`Replayed`), as the batch contract asks (training.md; row 6663)."""
    if isinstance(handed, pygrain.IterDataset):
        probe = iter(handed)
        try:
            before = probe.get_state()
            first = next(probe)
        except BaseException:
            if isinstance(probe, Closeable):
                probe.close()
            raise
        replayed = [Replayed(probe, first, before)]

        def train(partition: DataPartition) -> Iterator[Batch]:
            return _Shared(replayed.pop() if replayed else iter(handed), partition, held=False)

        return Loader(train=train, val=None, records=None, batch=rows_of(_present(first)))
    first = handed[0] if isinstance(handed, pygrain.MapDataset) else {
        name: np.asarray(column) for name, column in handed.items()}
    batch = rows_of(_present(first))
    records = len(handed) * batch if isinstance(handed, pygrain.MapDataset) else batch
    return Loader(train=native(handed, training=True), val=None, records=records, batch=batch)


def _present(first):
    """`first`, a dataset's first element, refused where it is missing."""
    if first is None:
        raise ValueError("a Grain dataset handed to fit gives its batches, and its first is missing (None)")
    return first


def run_widths(measured: Sequence[_Measured], objective: ObjectiveData) -> Mapping[str, object]:
    """A run's static widths: the facts its objective's run-scoped steps read (`Configured.scope`,
    such as a decision's `slots`), measured over its training chain and every held one, each prepared
    in its own role with the objective's steps in their places and its source's own steps run where a
    memory read runs them (`_measure`), and merged as a run reads them together (`merge_facts`: slots
    as their maxima). Training and every validation split bind them alike (`fitted`), so every split
    reads in training's shapes, one program for them all, where a split once took its own. None are
    measured where the objective's steps read none."""
    needs = {need for step in (*objective.train, *objective.evaluate)
             if isinstance(step, Configured) and step.scope == "run" for need in step.needs}
    if not needs:
        return {}
    merged = merge_facts([each.metadata for each in measured if each.metadata is not None])
    return {name: merged[name] for name in sorted(needs) if name in merged}


def fitted[Entry](
    dataset: Dataset | Loader | Handed,
    validation: Dataset | Mapping[str, Dataset | Entry] | None,
    *,
    objective: ObjectiveData,
    seed: int = 0,
    examples: int | None = None,
    epochs: int | None = None,
    steps: int | None = None,
    tokens: int | None = None,
    resume: DataResume = "relaxed",
    count_passes: bool = False,
) -> tuple[Loader, Mapping[str, Reader | Entry] | None]:
    """`dataset` and each validation dataset as `Trainer.fit` reads them,
    compiled with the objective's own stages and the run's data `seed`,
    their stores provisioned, or training data handed as it is read as
    handed (`native`; `dew.training._fit.handed_splits` reads validation so).
    A training dataset without a repeat repeats before its batch, so a run's
    length is its steps, epochs or tokens, or with `examples` reads that many
    records before they are packed and ends (`compile`'s `examples`); a
    validation dataset without a batch takes the
    training batch, and is read and loaded as it asks. A resume reads its position as `resume` says
    (`dew.data.plan.DataResume`), the run's. A Loader is already
    compiled, and an entry that is no Dataset (a reader, an evaluation suite)
    is passed on as it is. A run of `epochs` over a chain whose read has no
    record count, but states the source examples a pass holds
    (`Loader.pass_examples`), reads it bounded to that many passes, as
    `examples` bounds it (`Loader.epochs`), in the one compile (`_staged`). `count_passes` is `compile`'s:
    a run bounded otherwise whose cadence reads its passes counts them where its read states them."""
    if isinstance(dataset, Dataset):
        dataset = dataset._chain()
    if isinstance(validation, Dataset):
        validation = validation._chain()
    elif validation is not None:
        validation = {name: entry._chain() if isinstance(entry, Dataset) else entry
                      for name, entry in validation.items()}
    if not isinstance(dataset, (Dataset, Loader)):
        if examples is not None:
            raise ValueError("examples= counts a Dataset chain's source examples, and data handed as it "
                             "is has none: read it as a chain (dew.data.Grain, dew.data.Memory), or set "
                             "steps=")
        dataset = native_loader(dataset)
    named = {"val": validation} if isinstance(validation, Dataset) else validation
    # Every chain the run reads binds the widths measured over them all (`run_widths`), each measured
    # once, as its read prepares it, and read on from there.
    # Each held chain is measured in the batches it is read in: its own, else training's (`held`), where
    # training states its batch ahead; a token budget's rows are counted as it compiles.
    trained = (next((stage.size for stage in dataset._chain().stages if isinstance(stage, BatchStage)), None)
               if isinstance(dataset, Dataset) else dataset.batch)
    held_chains = {name: _held_batched(entry._chain(), trained, unbatched=trained is None)
                   for name, entry in (named or {}).items() if isinstance(entry, Dataset)}
    scored = {name: _held_measure(chain, objective) for name, chain in held_chains.items()}
    if isinstance(dataset, Dataset):
        chain = dataset._chain()
        # A run of epochs reads its chain as written, which staging reads endlessly or bounds to its
        # epochs' passes over its examples (`_staged`).
        # Phases count no pass: each ends at its step (`_compile_phases`).
        counting = (count_passes and examples is None and epochs is None
                    and not isinstance(chain.source, Phases))
        trained = chain if examples is not None or epochs is not None else _endless(chain)
        read = trained.mode or _preferred_read((trained,), provision=True)
        # A run of phases compiles each phase on its own (`_compile_phases`), each binding the widths
        # measured over every phase and held chain (data.md §10, `run_widths`).
        phased = isinstance(trained.source, Phases) and examples is None
        measured = (None if not scored or phased else
                    _training_measure(trained, None, objective=objective, read=read, provision=True,
                                      loading=trained._loading or _DEFAULT_LOADING, examples=examples,
                                      epochs=epochs if examples is None else None, counting=counting))
        phases = ([_training_measure(phase._then(*trained.stages), None, objective=objective, read=read,
                                     provision=True, loading=trained._loading or _DEFAULT_LOADING,
                                     examples=None)
                   for phase in trained.source.pipelines] if phased and isinstance(trained.source, Phases)
                  else [])
        widths = (run_widths([*([measured] if measured else []), *phases, *scored.values()], objective)
                  if scored or phases else {})
        loader = compile(trained, objective=objective, provision=True, seed=seed, examples=examples,
                         widths=widths, read=read, measured=tuple(phases) if phased else measured,
                         epochs=epochs if examples is None else None,
                         steps=steps, tokens=tokens, resume=resume, count_passes=counting)
    else:
        widths = run_widths(list(scored.values()), objective) if scored else {}
        loader = dataset
        # A loader compiled already resumes as it was compiled to (`Loader.resume`): the run's own mode,
        # one statement of how its resume treats its data.
        if loader.resume is not None and loader.resume != resume:
            raise ValueError(f"the run resumes its data {resume}, and the loader it is given was compiled to "
                             f"resume {loader.resume}; compile it with resume={resume!r}, or hand the run "
                             "its chain")
    if named is None:
        return loader, None
    return loader, {name: held(entry, objective=objective, batch=loader.batch, seed=seed, widths=widths,
                               measured=scored[name])
                    if isinstance(entry, Dataset) else entry
                    for name, entry in named.items()}


def _endless(chain: Pipeline) -> Pipeline:
    """`chain` read endlessly: a repeat before its batch where it has none and its read can end. A
    chain of phases reads each phase through the chain's own stages, and where one of them already
    reads on (`_reads_on`), as Grain refuses to repeat an order without end, each other phase repeats
    at the place the chain's own repeat takes, its share of the chain's stages before the batch moved
    into it; a mixture whose members all read on is read as it is. A phase that batches inside itself and
    states no repeat reads on with its repeat before its own batch (`_repeated_inside`)."""
    if any(isinstance(stage, Repeat) for stage in chain.stages) or _reads_on(chain):
        return chain
    source = chain.source
    if isinstance(source, Phases):
        chain = dataclasses.replace(chain, source=dataclasses.replace(
            source, pipelines=tuple(map(_repeated_inside, source.pipelines))))
        source = chain.source
    batching = [index for index, stage in enumerate(chain.stages)
                if isinstance(stage, (BatchStage, TokenBudget))]
    at = batching[0] if batching else len(chain.stages)
    if isinstance(source, Phases):
        read = [child._then(*chain.stages[:at]) for child in source.pipelines]
        if any(map(_reads_on, read)):
            each = tuple(phase if _reads_on(phase) else phase.repeat() for phase in read)
            return dataclasses.replace(chain, source=dataclasses.replace(source, pipelines=each),
                                       stages=chain.stages[at:])
    return dataclasses.replace(chain, stages=(*chain.stages[:at], Repeat(None), *chain.stages[at:]))


def _repeated_inside(phase: Pipeline) -> Pipeline:
    """`phase` read on with its repeat before its own batch, as a single stream's is, where it batches
    inside itself and states no repeat: a repeat after its batch, the chain's, dropped what each pass left
    of its last batch, and a pass holding less than a batch never yielded one (12 read it so, and a run
    saved under it reads so still). A phase that states a repeat, or batches outside itself, as it is."""
    batching = [index for index, stage in enumerate(phase.stages)
                if isinstance(stage, (BatchStage, TokenBudget))]
    if not batching or _reads_on(phase) or any(isinstance(stage, Repeat) for stage in phase.stages):
        return phase
    return dataclasses.replace(phase, stages=(*phase.stages[:batching[0]], Repeat(None),
                                              *phase.stages[batching[0]:]))


def _reads_on(chain: Pipeline) -> bool:
    """Whether `chain` reads without end as written (`analysis.reach`): its stages leave it endless, or
    it mixes members that all read on."""
    source = chain.source
    unending = isinstance(source, Mixture) and all(_reads_on(child) for child in source.pipelines)
    return reach(chain.stages, unending)[0]


def held(dataset: Dataset, *, objective: ObjectiveData | None = None, batch: int | None = None,
         seed: int = 0, widths: Mapping[str, object] | None = None,
         measured: _Measured | None = None) -> Reader:
    """`dataset` read once to validate or evaluate on, in its own batches, else in batches of `batch`,
    read and loaded as it asks, its random steps drawing from `seed`; one that states neither batch is
    refused. A pass reads every record: a batch it fills only in part keeps the records it holds,
    filled out with copies its `VALID_ROWS` marks, where a training read would drop them. `measured`
    is the chain as its run's widths measured it (`_held_measure`), which it reads as it stands."""
    chain = _held_batched(dataset._chain(), batch)
    if measured is None:
        measured = _held_measure(chain, objective)
    elif not measured.branches[0].batched:
        # Measured before its run's batch was known (`fitted`): batched now, the objective's batch steps
        # after its batch.
        def batched(branch: Pipeline) -> Pipeline:
            branch = _held_batched(branch, batch)
            return branch if objective is None else _batch_steps_after(branch, objective, train=False)

        measured = dataclasses.replace(measured, branches=tuple(map(batched, measured.branches)))
    # The chain's own batch is the one it is read in; validation's `batch` adds none to it.
    return validation(measured.branches[0], batch=1, objective=objective, read=_held_read(chain),
                      loading=chain._loading or _DEFAULT_LOADING, provision=True, seed=seed, widths=widths,
                      measured=measured)


def _held_batched(chain: Pipeline, batch: int | None, *, unbatched: bool = False) -> Pipeline:
    """A held `chain` in its own batches, else in batches of `batch`, each keeping a short last one; one
    that states neither is refused, or left as it is where it may be (`unbatched`, a batch not yet
    known). It is batched before it is measured (`_held_measure`), so an objective's batch steps follow
    its batch as they do in training."""
    if not chain.batched:
        if batch is None and unbatched:
            return chain
        if batch is None:
            raise ValueError(f"{_name(chain)} states no batch; batch it (`.batch(n)`) or give batch=")
        chain = chain.batch(batch, drop_remainder=False)
    return dataclasses.replace(chain, stages=tuple(
        dataclasses.replace(stage, drop_remainder=False) if isinstance(stage, (BatchStage, TokenBudget))
        else stage for stage in chain.stages))


def _held_read(chain: Pipeline) -> Read:
    """The read a held chain asks, else the one its sources choose (`_preferred_read`)."""
    return chain.mode or _preferred_read((chain,), provision=True)


def _held_measure(chain: Pipeline, objective: ObjectiveData | None) -> _Measured:
    """A held chain prepared through its source's own steps as its pass reads them (`_measure`), its
    counts filled where a memory read runs them."""
    return _measure((chain,), _held_read(chain), provision=True, objective=objective, counts=({},))


@compiling
def validation(
    pipeline: Pipeline,
    *,
    batch: int,
    objective: ObjectiveData | None = None,
    host: tuple[HostStep, ...] = (),
    read: Read | None = "indexed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
    seed: int = 0,
    widths: Mapping[str, object] | None = None,
    measured: _Measured | None = None,
) -> Reader:
    """Compile one finite evaluation branch without constructing a training
    stream. With no `read`, the sources choose it (`_preferred_read`); `widths` are the run's static
    widths its steps bind to (`run_widths`), and `measured` the branch as they measured it, batched."""
    read = read or _preferred_read((pipeline,), provision=provision)

    # One pass reads every record, in the chain's own batches or `batch`'s (`_held_batched`).
    pipeline = _held_batched(pipeline, batch)
    if measured is not None:
        measured = dataclasses.replace(measured, branches=tuple(
            _held_batched(branch, batch) for branch in measured.branches))
    prepared = _prepared((pipeline,), read, provision=provision, objective=objective, counts=({},),
                         widths=widths, measured=measured)
    (pipeline,) = _made(prepared, read, ({},), training=False)
    packed = _coordinated(pipeline)
    # A token budget is fixed-length rows by now (`_budgeted`).
    build = _build(pipeline, read, loading, batch=_output_batch(prepared[0]), training=False, seed=seed)
    return functools.partial(_validation_reader, pipeline=pipeline, build=build.tuned(pipeline),
                             host=packed + _held_steps(pipeline) + host)


@compiling
def publish(
    pipeline: Pipeline,
    cache,
    *,
    read: Read = "streamed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
):
    """Fill and seal a deterministic prefix, then publish its certified
    generation, unless one is already published under its key (with the plan
    of a terminal pack), which it returns, its stream's fill numbering its rows
    by the shards its source states (`dew.data.sources.base.Source.manifest`)."""
    from dew.data.cache import Cache
    from dew.data.generation import Generation

    if not isinstance(cache, Cache):
        raise TypeError("publish needs a Cache destination")
    if isinstance(pipeline.source, Combined):
        raise ValueError("publish each source prefix before mixing")
    planned = _at(pipeline.stages, Pack)
    builder, accepted = None, None
    if planned is not None:
        # The packer imported where a pack is published: a prefix that packs nothing imports none.
        from dew.data.packing import PackingPlan, plan_name

        packing = pipeline.stages[planned]
        if not isinstance(packing, Pack) or planned != len(pipeline.stages) - 1:
            raise ValueError("publication accepts one terminal packing stage after the cached prefix")
        builder, accepted = PackingPlan(packing), plan_name(packing, "accepted")
        pipeline = dataclasses.replace(pipeline, stages=pipeline.stages[:planned])
    (pipeline,) = _prepared((pipeline,), read, provision=provision)
    source = pipeline.source
    if isinstance(source, Combined):
        raise ValueError("configured publication needs one physical source")
    manifest = _cache_manifest(source, pipeline.stages, read)

    def published() -> Generation | None:
        generation = cache.generation(manifest)
        held = generation is not None and (accepted is None or accepted in generation.manifest["plan"])
        return generation if held else None

    found = published()
    if found is not None:
        return found
    iterator = iter(iterate(pipeline.cache(cache), read=read, loading=loading))
    try:
        for _ in iterator:
            pass
    finally:
        iterator.close()
    # A fill one reader completed publishes itself (`Cache.mapped`).
    found = published()
    return cache.publish(manifest, source, plan=builder) if found is None else found
