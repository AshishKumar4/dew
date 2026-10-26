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

import dataclasses
import functools
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Literal, Protocol, runtime_checkable

import grain.python as pygrain
import numpy as np

from dew._records import JSON, integer, json_value
from dew.batch import Batch
from dew.data.analysis import (
    Measuring,
    _analyzed,
    _at,
    _cache_key,
    _cache_manifest,
    _cacheable,
    _capabilities,
    _configured,
    _cut,
    _identity,
    _leaves,
    _Lowering,
    _manifest,
    _measures,
    _member_keys,
    _members,
    _name,
    _preferred_read,
    _promoted,
    _stated,
    _steps,
    _until,
    _volatile,
    _written,
)
from dew.data.dataset import (
    Closeable,
    DataPartition,
    GlobalReader,
    GlobalStream,
    Loader,
    Loading,
    PhasedStream,
    Reader,
    Records,
)
from dew.data.facts import merge_facts
from dew.data.lowering import (
    _as_iter,
    _Build,
    _epoch_records,
    _Locality,
    _order_records,
    _split_spans,
    _tail,
    _weights,
    _workers,
)
from dew.data.materialize import _materialize, _Memory
from dew.data.mixing import CorpusCounts, Proportions
from dew.data.pipeline import (
    Artifact,
    Batched,
    BatchStage,
    Configured,
    Costly,
    Dataset,
    DeviceStep,
    Expansion,
    Filter,
    FlatMap,
    HostStep,
    Interleave,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    Mixture,
    Native,
    ObjectiveData,
    Pack,
    Phases,
    Pipeline,
    Prefetch,
    RandomMap,
    RecordStep,
    Repeat,
    Shuffle,
    SingleReader,
    Slice,
    Stage,
    TokenBudget,
    TokenGeometry,
    Windows,
    Workers,
    children,
    record_stage,
    with_children,
    with_stores,
)
from dew.data.plan import Plan, StepIdentity, fingerprint, unplaced
from dew.data.readers import _CanonicalPass, _Host, _PerShare, _Statistics
from dew.data.sources.base import (
    Read,
    Rows,
    Source,
    View,
    _consumed,
    beneath,
    materialized,
    tokenizer_contract,
)
from dew.data.steps.generic import Coordinated

_DEFAULT_LOADING = Loading()


@runtime_checkable
class _OrderFacts(Protocol):
    def order_fingerprints(self) -> Mapping[str, str]: ...


def _automatically(stage: Stage) -> bool:
    """Whether the compiler placed `stage`, a cache the chain did not write (`_automatic`)."""
    return isinstance(stage, Artifact) and stage.automatic


def _automatic(pipeline: Pipeline, read: Read, root: str | None) -> Pipeline:
    """`pipeline` with a cache at `root` after its leading deterministic record maps, where one of them
    costs more to run than its records cost to read back (`Costly`: a decode, a learned tokenizer)
    and the chain writes no cache: each corpus of a mixture or phases its own. The maps are the ones
    a cache keys (`_cacheable`) and reads again unchanged (none `_volatile`), and in an indexed read
    none that rejects, whose cache a read by index reads back as a stream, slower than it reruns
    them. It stands for them only in a memory or indexed read, by each input's place in a store
    whose content is named, in a clean checkout whose steps are exact (`Manifest`), and nowhere
    else, so a chain no cache could serve reads as it is written, never refused. It holds the
    records the maps make, and fills, publishes, replays and evicts as `cache(url)` does (`Cache`),
    so a run reads the same records with or without it (`Artifact.automatic`), and a run of the
    same maps over the same records reads them back once one has read them whole. Its root is the
    run's to name or refuse (`automatic_root`)."""
    from dew.data.cache import Cache

    if root is None or read == "streamed":
        return pipeline
    if isinstance(pipeline.source, (Mixture, Phases)):
        return with_children(pipeline, lambda child: _automatic(child, read, root))
    if any(isinstance(stage, Artifact) for stage in _steps(pipeline)):
        return pipeline

    def kept(stage: Stage) -> bool:
        return (isinstance(stage, (Map, MapWithIndex)) and _cacheable(stage) and not _volatile(stage)
                and not (read == "indexed" and stage.rejects))

    at = next((index for index, stage in enumerate(pipeline.stages) if not kept(stage)), len(pipeline.stages))
    prefix = pipeline.stages[:at]
    steps = [stage.step for stage in prefix if isinstance(stage, (Map, MapWithIndex))]
    if not any(isinstance(step, Costly) and step.costly for step in steps):
        return pipeline
    if not pipeline.source.supports(read):
        return pipeline
    key = _cache_key(pipeline.source, prefix, read)
    if key is None or not (key.clean and key.exact):
        return pipeline
    cached = Artifact(Cache(root), automatic=True)
    return dataclasses.replace(pipeline, stages=(*prefix, cached, *pipeline.stages[at:]))


def _cache_references(pipeline: Pipeline, read: Read) -> tuple[str, ...]:
    """The caches each corpus of `pipeline` reads through, among its stages as its pipeline wrote
    them (`_written`), the ones a memory read already ran included, each named against the source
    it came from."""
    references = []
    for leaf, source in _leaves(pipeline):
        stages, written = _written(source, leaf.stages), source
        while materialized(written):
            written = written.source
        references += [stage.cache.reference(_cache_manifest(written, stages[:index], read))
                       for index, stage in enumerate(stages)
                       if isinstance(stage, Artifact) and not stage.automatic]
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
    stated_metadata: Mapping[str, object]
    held: _PinnedRead | None = None
    """The memory read pinning took, which the compile's memory reads of the source reuse, so its
    facts and the rows a run materializes are one read's."""
    stated_label: str = ""
    """The source's name as it was written, before a read put another in its place (a Hub split's
    Parquet files, `Consumed`), which the run's metrics go by."""

    def label(self) -> str:
        return self.stated_label

    def fingerprint(self) -> Mapping[str, object]:
        return self.stated_snapshot

    def metadata(self) -> Mapping[str, object]:
        return self.stated_metadata

    def supports(self, read: Read) -> bool:
        return read == self.read or self.source.supports(read)

    def records(self, read: Read) -> Records:
        held = None if self.held is None else self.held.records
        return held if read == "memory" and held is not None else self.source.records(read)

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
    stores: dict[int, Records] = {}
    reading = threading.Lock()

    def whole(store: Source) -> Records:
        # A store several holdout parts measure over is read once for them all.
        with reading:
            if id(store) not in stores:
                stores[id(store)] = store.records(read)
            return stores[id(store)]

    def pin(source: Source) -> _Facts:
        if isinstance(source, _Facts):
            return source
        label = source.label()
        measuring = tuple(measured.get(id(source), {}).values())
        source = _consumed(source, read)
        records, metadata = _stated(source, read, measuring, whole)
        held = None if records is None else _PinnedRead(records)
        return _Facts(source, read, source.fingerprint(), metadata, held, label)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as readers:
        pinned = dict(zip(sources, readers.map(pin, sources.values()), strict=True))

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
    tokenizer_contract((f"corpus {_name(leaf)}", source.metadata())
                       for pipeline in pipelines for leaf, source in _leaves(pipeline))


def _row_tokens(pipeline: Pipeline) -> int | None:
    """The tokens each row of `pipeline` holds, as its last record step
    declaring them states (`TokenGeometry`), or its corpora alike."""
    declared = [
        stage.step
        for stage in pipeline.stages
        if isinstance(stage, (Map, RandomMap)) and isinstance(stage.step, TokenGeometry)
    ]
    if declared:
        return declared[-1].max_length
    if nested := children(pipeline):
        lengths = {_row_tokens(child) for child in nested}
        if len(lengths) > 1:
            raise ValueError(f"a token budget needs rows of one length, and the corpora's are {lengths}")
        return lengths.pop()
    return None


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
    if budget.tokens % length:
        raise ValueError(f"a token budget of {budget.tokens} is not whole rows of {length} tokens")
    rows = BatchStage(budget.tokens // length, budget.drop_remainder, budget.batch_fn)
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


def _bounded(pipeline: Pipeline, examples: int) -> Pipeline:
    """`pipeline`'s training read bounded to its first `examples` records as they reach the first
    stage that packs or windows them, or its batch where none does: the records the objective's
    steps make, not the windows a pack lays them in. A repeat without end goes before the bound
    where none is written, so the records may span passes, and the batch keeps the short one the
    bound ends inside, filled and marked, so each record bounded is read once (`Loader.examples`).
    Refuses a bound it cannot place: a repeat with a count before it, or any repeat after it, the
    batch's included, a mixed corpus that packs on its own, and phases, which end by steps. A chain
    without a batch is left for `compile` to refuse."""
    if examples < 1:
        raise ValueError(f"examples must be at least 1, got {examples}")
    if isinstance(pipeline.source, Phases):
        raise ValueError("phases end each phase at a step, so their run is bounded in steps, not examples")
    if any(isinstance(stage, (Pack, Windows)) for child in children(pipeline) for stage in _steps(child)):
        raise ValueError(f"{_name(pipeline)} packs inside a corpus it mixes, so its records reach the mix as "
                         "windows; pack after the mix to bound the run in examples")
    batched = _until(pipeline.stages, (BatchStage, TokenBudget))
    if batched == len(pipeline.stages):
        return pipeline
    at = _until(pipeline.stages[:batched], (Pack, Windows))
    repeats = [stage for stage in pipeline.stages[:at] if isinstance(stage, Repeat)]
    if any(stage.epochs is not None for stage in repeats) or any(
            isinstance(stage, Repeat) for stage in pipeline.stages[at:]):
        raise ValueError(f"{_name(pipeline)} repeats with a count, or after the records it packs or batches, "
                         f"which a bound of {examples} examples would cut or repeat; leave the repeat out")
    bound = (Limit(examples),) if repeats else (Repeat(None), Limit(examples))
    batch = dataclasses.replace(pipeline.stages[batched], drop_remainder=False)
    return dataclasses.replace(pipeline, stages=(*pipeline.stages[:at], *bound, *pipeline.stages[at:batched],
                                                 batch, *pipeline.stages[batched + 1:]))


def _lays_out(pipeline: Pipeline) -> bool:
    """Whether `pipeline` packs or windows its records itself, before its batch, which its preparation
    may move into its source (`_promoted`), where a bound would come after the windows."""
    return any(isinstance(stage, (Pack, Windows))
               for stage in pipeline.stages[:_until(pipeline.stages, (BatchStage, TokenBudget))])


_KEYED_ABOVE = (Map, RandomMap, Filter, BatchStage, Prefetch, Workers, Repeat, Limit, Slice, MapBatch)
"""The stages above a streamed mix that hold none of its rows across a saved position, so the mix's
members may be restored by key beneath them: a shuffle's window, a packer's remainder or an
expansion's tail holds rows of the mix that saved them."""


def _member_manifests(train: Pipeline, build: _Build, host: Sequence[HostStep]) -> dict[str, str]:
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
    members = {"": fingerprint({"steps": steps([*(_identity(stage) for stage in train.stages
                                                  if not isinstance(stage, (Workers, Prefetch))),
                                                *(StepIdentity.of(step) for step in host)])})}
    for index, (key, child) in enumerate(zip(_member_keys(_members(source.pipelines)), source.pipelines,
                                             strict=True)):
        sources = [read for _, read in _leaves(child)]
        identities = [*(_identity(stage) for read in sources for stage in _written(read, ())),
                      *(_identity(stage) for stage in _steps(child)
                        if not isinstance(stage, (Workers, Prefetch)))]
        members[key] = fingerprint({
            "order": unplaced(json_value(corpora[index], "member order")),
            "sources": [read_of(read).fingerprint() for read in sources],
            "steps": steps(identities),
            "derived": dict(_derived(child, build.child(index))),
            "cache": list(_cache_references(child, build.read)),
        })
    return members


def _derived(pipeline: Pipeline, build: _Build) -> Mapping[str, str]:
    seed, read = build.seed, build.read
    arrays: dict[str, str] = {}
    for index, (leaf, source) in enumerate(_leaves(pipeline)):
        if isinstance(source, _Memory):
            arrays.update({f"{index}/{key}": value for key, value in source.derived.items()})
        lengths = source.lengths()
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
        metadata = source.metadata()
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
    counts: Sequence[dict[str, CorpusCounts]] | None = None,
    automatic: str | None = None,
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
    each corpus's leading deterministic steps under `automatic` (`_automatic`).
    """
    if read not in ("memory", "indexed", "streamed"):
        raise ValueError(f"unknown read mode {read!r}")
    held = [not training or index > 0 for index in range(len(branches))]
    for branch, scored in zip(branches, held, strict=True):
        _analyzed(branch, read, validation=scored, promotion=True)
    branches = tuple(_promoted(branch) for branch in branches)
    for branch, scored in zip(branches, held, strict=True):
        _analyzed(branch, read, validation=scored)
    _capabilities(branches, read, provision)
    branches = _pin_sources(branches, read)
    _tokenizer_contract(branches)
    branches = tuple(_automatic(branch, read, automatic) for branch in branches)
    if objective is not None:
        branches = tuple(
            _objective_branch(branch, objective, train=not scored, read=read)
            for branch, scored in zip(branches, held, strict=True)
        )
    if any(isinstance(stage, Configured) for branch in branches for stage in _steps(branch)):
        def stored(branch: Pipeline) -> bool:
            return any(len(source.records(read)) for _, source in _leaves(branch))

        if training and read == "memory" and not stored(branches[0]):
            raise ValueError(f"no accepted rows of {_name(branches[0])}; source is empty")
        metadata = _metadata(branches, read)
        if read == "memory" and counts is not None:
            branches = tuple(
                _materialize(_configured(branch, metadata, read, "source"), count, training=not scored)
                for branch, count, scored in zip(branches, counts, held, strict=True)
            )
            if training and not stored(branches[0]):
                raise ValueError(f"no accepted rows of {_name(branches[0])}; raise the row budgets")
            metadata = _metadata(branches, read)
        branches = tuple(_configured(branch, metadata, read) for branch in branches)
    if objective is not None:
        branches = tuple(_windowed(branch, objective) for branch in branches)
    if any(isinstance(stage, TokenBudget) for branch in branches for stage in branch.stages):
        # A token budget's rows are the training rows' fixed length (`_budgeted`).
        trained = _row_tokens(branches[0])
        branches = tuple(_budgeted(branch, trained) for branch in branches)
    for branch, scored in zip(branches, held, strict=True):
        _analyzed(branch, read, validation=scored)
    return branches


def _made(prepared: Sequence[Pipeline], read: Read, counts: Sequence[dict[str, CorpusCounts]], *,
          training: bool) -> tuple[Pipeline, ...]:
    """Prepared branches as a memory read makes them, the first training where `training`: their
    first stages run and their rows held (`_materialize`), counted into `counts`, then the memory
    reads pinning took let go (`_released`); another read's as they are."""
    if read != "memory":
        return tuple(prepared)
    made = tuple(_materialize(branch, count, training=training and not index)
                 for index, (branch, count) in enumerate(zip(prepared, counts, strict=True)))
    _released(prepared)
    return made


def _build(branch: Pipeline, read: Read, loading: Loading, *, batch: int, training: bool, seed: int = 0,
           lowering: _Lowering | None = None) -> _Build:
    """The build a reader of a prepared `branch` reads it by, in batches of `batch`: training's, or a
    scored or published pass's, by the lowering its analysis gives unless `lowering` says otherwise."""
    lowered = _analyzed(branch, read, validation=not training) if lowering is None else lowering
    return _Build(read, loading, seed, batch, lowered, training=training)


def compile(
    train: Pipeline,
    *,
    validation: Pipeline | None = None,
    host: tuple[HostStep, ...] = (),
    objective: ObjectiveData | None = None,
    read: Read | None = None,
    provision: bool = False,
    loading: Loading | None = None,
    seed: int = 0,
    rejections: Literal["drop", "mask"] = "drop",
    resume: Literal["exact", "non-exact"] = "exact",
    examples: int | None = None,
) -> Loader:
    """Check every leaf before opening readers, then build both branches.
    With no `read`, the sources choose it (`_preferred_read`). `examples` bounds the training read to
    its first that many records before they are packed (`_bounded`), a run's `examples=` budget."""
    branches = (train,) if validation is None else (train, validation)
    read = read or train.mode or _preferred_read(branches, provision=provision)
    loading = loading or train._loading or _DEFAULT_LOADING
    if isinstance(train.source, Phases) and examples is None:
        return _compile_phases(
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
        )
    staged = _staged(train, validation, objective=objective, read=read, provision=provision, loading=loading,
                     seed=seed, rejections=rejections, examples=examples)
    loader = _compiled(staged, _coordinated(staged.train), host=host, resume=resume)
    return loader if examples is None else dataclasses.replace(loader, examples=examples)


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
) -> _Staged:
    from dew.data.cache import automatic_root

    if rejections == "mask":
        raise ValueError(
            "rejections='mask' needs an objective proven mask-safe; no objective currently declares it"
        )
    stated = next((stage for stage in train.stages if isinstance(stage, TokenBudget)), None)
    batch_tokens = None if stated is None else stated.tokens
    counts: dict[str, CorpusCounts] = {}
    validation_counts: dict[str, CorpusCounts] = {}
    # A chain's own layout is bounded as written, before preparation could plan it into its source; the
    # objective's goes in with its preparation, so the bound follows that.
    written = None if examples is None or not _lays_out(train) else examples
    if written is not None:
        train = _bounded(train, written)
    branches = _prepared(
        (train,) if validation is None else (train, validation),
        read,
        provision=provision,
        training=True,
        objective=objective,
        counts=(counts,) if validation is None else (counts, validation_counts),
        automatic=automatic_root(loading) if read != "streamed" else None,
    )
    if not any(isinstance(stage, BatchStage) for stage in branches[0].stages):
        raise ValueError("compile needs one batch(global_size) stage on the training pipeline")
    if examples is not None and written is None:
        branches = (_bounded(branches[0], examples), *branches[1:])
    original = branches[0]
    train, *scored = _made(branches, read, (counts, validation_counts)[: len(branches)], training=True)
    validation = scored[0] if scored else None
    # A memory read that kept none of the scored records scores none.
    if validation_counts and not any(count.accepted for count in validation_counts.values()):
        validation = None
    build = _build(train, read, loading, batch=_output_batch(original), training=True, seed=seed)
    return _Staged(train, validation, original, build, counts, validation_counts, batch_tokens)


def _output_batch(pipeline: Pipeline) -> int:
    """The rows each batch of `pipeline` holds as its reader yields them, which its loader declares
    and its share pads to."""
    return next(stage.size for stage in pipeline.stages if isinstance(stage, BatchStage))


def _compiled(
    staged: _Staged,
    packed: tuple[Coordinated, ...],
    *,
    host: tuple[HostStep, ...],
    resume: Literal["exact", "non-exact"],
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
    manifest = _manifest(
        {"sources": [source.fingerprint() for source in sources]},
        sources,
        (
            *(_identity(stage) for stage in _steps(original)
              if not isinstance(stage, (Workers, Prefetch)) and not _automatically(stage)),
            *(StepIdentity.of(step) for step in host),
        ),
    )
    plan = Plan(
        "global" if global_order else "per-share",
        manifest,
        {"read": read, "seed": seed, **build.lowering.order},
        _derived(train, build),
        cache=_cache_references(original, read),
        members=_member_manifests(train, build, host),
        volatile=any(_volatile(stage) for stage in
                     (*_steps(original), *(stage for source in sources for stage in _written(source, ())))),
    )
    tuned = build.tuned(train)
    # The training read's workers move before what drops records (`pooled`); validation reads its own.
    build = tuned.pooled(train)
    train_host, device = _batch_steps(train)
    train_host = packed + train_host + host
    statistics = _Statistics(counts)
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
            build=dataclasses.replace(
                tuned, lowering=_analyzed(validation, read, validation=True), training=False,
                batch=_output_batch(validation),
            ).tuned(validation),
            host=_held_coordinated(validation, packed) + _held_steps(validation) + host,
        )
    )
    records = _epoch_records(train, build)
    if records == 0:
        raise ValueError(f"corpus {_name(train)} has no accepted rows")
    held_out = sum(integer(source.metadata().get("held_out", 0), "held_out") for _, source in _leaves(train))
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
        tail=_tail(train, build),
        order_records=_order_records(train, build),
    )


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
    from dew.data.packing import PackedSource

    fields: dict[str, tuple[str, ...]] = {}
    bounds: dict[str, int] = {}
    unseen = False

    def declare(pack: Pack, most: int | None) -> None:
        ids = pack.fields[0]
        fields[ids] = tuple(dict.fromkeys((*fields.get(ids, ()), *pack.fields)))
        bound = pack.seq_len if most is None else max(1, min(pack.seq_len, most))
        bounds[ids] = max(bounds.get(ids, 0), bound)

    def walk(pipeline: Pipeline, *, packed: bool) -> None:
        nonlocal unseen
        for stage in pipeline.stages:
            if isinstance(stage, Pack):
                declare(stage, stage.segments)
                packed = True
        source = pipeline.source
        if isinstance(source, (Mixture, Phases)):
            for child in source.pipelines:
                walk(child, packed=packed)
            return
        documents = _packed_documents(source)
        for stage in _written(source, ()):
            if isinstance(stage, Pack):
                declare(stage, stage.segments if documents is None else documents)
                packed = True
        *_, source = beneath(source, ("read", "materialized"))
        if isinstance(source, PackedSource):
            declare(source.plan.pack, source.plan.most_documents)
            packed = True
        unseen = unseen or not packed

    for pipeline in pipelines:
        walk(pipeline, packed=False)
    return tuple(Coordinated(fields[ids], None if unseen else bound) for ids, bound in bounds.items())


def _held_coordinated(validation: Pipeline, training: tuple[Coordinated, ...]) -> tuple[Coordinated, ...]:
    """A validation pass's `Coordinated` steps: its own packs' declarations,
    and for a field only the training packs it, a window's length, since a
    validation source may hold packed windows whose packing it does not show."""
    held = _coordinated(validation)
    declared = {step.fields[0] for step in held}
    return held + tuple(Coordinated(step.fields, None) for step in training if step.fields[0] not in declared)


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
    resume: Literal["exact", "non-exact"],
) -> Loader:
    source = train.source
    if not isinstance(source, Phases):
        raise TypeError("phased compilation needs a Phases source")
    _tokenizer_contract((train,) if validation is None else (train, validation))
    consumed: dict[str, int] = {}
    staged: list[tuple[_Staged, int | None]] = []
    previous = 0
    for index, (pipeline, until) in enumerate(zip(source.pipelines, source.until_steps, strict=True)):
        phase = _staged(_continued(pipeline, consumed)._then(*train.stages),
                        validation if index == 0 else None, objective=objective, read=read,
                        provision=provision, loading=loading, seed=seed, rejections=rejections)
        end = None if until is None else until * phase.build.batch
        staged.append((phase, end))
        if end is not None:
            # The phase as it reads: its members a memory read kept, at the shares the lowering mixes.
            _delivered(phase.train, end - previous, consumed)
            previous = end
    # Every phase declares the most documents a window of any phase holds, so the run steps one program.
    packed = _coordinated(*(phase.train for phase, _ in staged))
    built: list[tuple[Loader, int | None]] = []
    for index, (phase, end) in enumerate(staged):
        loaded = _compiled(phase, packed, host=host, resume=resume)
        if loaded.plan is None or loaded.plan.lowering != "global":
            raise ValueError(
                f"phase {index} needs a global order; "
                "streamed, rejecting and online-packed phases are refused"
            )
        if built and loaded.device != built[0][0].device:
            # The run applies one set of device steps to every batch (`Loader.device`), so a phase
            # whose own differ would run the first phase's.
            raise ValueError(f"phase {index} ends in other device steps than phase 0; every phase of "
                             f"a run runs the same ones, written after the phases")
        built.append((loaded, end))

    def training(partition: DataPartition) -> Iterator[Batch]:
        def global_reader(reader: Reader) -> GlobalReader:
            opened = reader(partition)
            if not isinstance(opened, GlobalReader):
                if isinstance(opened, Closeable):
                    opened.close()
                raise ValueError("a phase must expose a global record-count position")
            return opened

        phases = [(functools.partial(global_reader, loaded.train), end) for loaded, end in built]
        return PhasedStream(phases, loading.stop_seconds)

    # Only the last phase's order can end, so its batch decides what an epoch's short batch does.
    return dataclasses.replace(built[0][0], train=training, tail=built[-1][0].tail, order_records=None)


def _continued(pipeline: Pipeline, consumed: Mapping[str, int]) -> Pipeline:
    if isinstance(pipeline.source, Phases):
        raise ValueError("nested phases are not an indexed record order")
    if isinstance(pipeline.source, Mixture):
        return with_children(pipeline, functools.partial(_continued, consumed=consumed))
    offset = consumed.get(_name(pipeline), 0)
    if not offset:
        return pipeline
    at = next((index + 1 for index, stage in enumerate(pipeline.stages) if isinstance(stage, Repeat)), None)
    if at is None:
        pipeline = pipeline.repeat()
        at = len(pipeline.stages)
    return dataclasses.replace(
        pipeline, stages=(*pipeline.stages[:at], Slice(slice(offset, None)), *pipeline.stages[at:])
    )


def _delivered(pipeline: Pipeline, count: int, consumed: dict[str, int]) -> None:
    source = pipeline.source
    if isinstance(source, Phases):
        raise ValueError("nested phases have no single corpus share")
    if isinstance(source, Mixture):
        # The counts the indexed mix reads its corpora by (`ProportionalSelection`).
        counts = Proportions.of(_weights(source)).counts(count)
        for child, taken in zip(source.pipelines, counts, strict=True):
            _delivered(child, taken, consumed)
    else:
        name = _name(pipeline)
        consumed[name] = consumed.get(name, 0) + count


def _training_reader(
    partition: DataPartition,
    *,
    pipeline: Pipeline,
    build: _Build,
    plan: Plan,
    host: tuple[HostStep, ...],
    resume: Literal["exact", "non-exact"],
    statistics: _Statistics,
) -> Iterator[Batch]:
    partition.rows(build.batch)
    if plan.volatile and partition.readers > 1:
        raise ValueError("volatile pipeline requires one reader per share until its bytes are persisted")
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
        )
    else:
        own_plan = dataclasses.replace(
            plan,
            topology={
                "shares": partition.count,
                "share": partition.index,
                "workers": _workers(pipeline, build.loading),
                "row_multiple": partition.row_multiple,
                "cycle_lengths": [
                    stage.cycle_length for stage in _steps(pipeline) if isinstance(stage, Interleave)
                ],
                # Where a read that drops records reads in workers of each corpus's own (`pooled`),
                # each worker's position is that corpus's, saved in that many.
                **({} if build.pool is None else {"pools": build.pools(pipeline)}),
            },
        )
        opened = _PerShare(
            functools.partial(build.batches, pipeline, partition),
            own_plan,
            resume,
            stop_seconds=build.loading.stop_seconds,
            endless=build.lowering.endless,
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
    # A token budget has no fixed row shape; its lowering enforces one share.
    budgeted = any(isinstance(stage, TokenBudget) for stage in pipeline.stages)
    if build.read == "streamed" and not budgeted:
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
    if isinstance(pipeline.source, (Mixture, Phases)) and not own:
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
    pipeline = _objective_records(dataclasses.replace(pipeline, stages=pipeline.stages[:at]),
                                  objective.train if train else objective.evaluate)
    return dataclasses.replace(pipeline, stages=(*pipeline.stages, *suffix))


def _objective_records(
    pipeline: Pipeline, steps: Sequence[RecordStep | pygrain.RandomMapTransform | Expansion | Configured]
) -> Pipeline:
    """Append objective record steps at the caller's preparation boundary."""
    return pipeline._then(*(step if isinstance(step, Configured) else record_stage(step) for step in steps))


def _objective_branch(pipeline: Pipeline, objective: ObjectiveData, *, train: bool, read: Read) -> Pipeline:
    """Prepare each corpus, then attach objective randomness after the user's record order."""
    steps = objective.train if train else objective.evaluate
    random = next((index for index, step in enumerate(steps)
                   if isinstance(step, pygrain.RandomMapTransform)
                   or (isinstance(step, Configured) and step.random)), len(steps))
    prefix = dataclasses.replace(objective, train=objective.train[:random],
                                 evaluate=objective.evaluate[:random])
    branch = objective_steps(pipeline, prefix, train=train, read=read)
    at = _until(branch.stages, (BatchStage, TokenBudget))
    suffix = branch.stages[at:]
    branch = dataclasses.replace(branch, stages=branch.stages[:at])
    branch = _objective_records(branch, steps[random:])
    branch = dataclasses.replace(branch, stages=(*branch.stages, *suffix))
    steps = objective.train_batch if train else objective.evaluate_batch
    # A string or object field stays in the batch for the host code and metrics that read it; the
    # prefetcher places the rest (`dew.training.distributed.split_fields`).
    return branch._then(*(MapBatch(step) for step in steps))


def _metadata(pipelines: Sequence[Pipeline], read: Read) -> Mapping[str, object]:
    # A fact a configured step measures binds a mixture's step to the largest of its stores'.
    measured = frozenset(each.measure.fact for pipeline in pipelines
                         for each in _measures(tuple(_steps(pipeline))))
    return merge_facts((source.metadata() for pipeline in pipelines for _, source in _leaves(pipeline)
                        if not (isinstance(source, _Memory) and not source.held)), measured=measured)


def _diagnostic(pipeline: Pipeline, *, records: int | None = None) -> _Host:
    """Inspect through training's lowering, finite unless a repeat asks otherwise.

    A bounded record pass runs lazily without loader read-ahead or
    eager memory transforms, and leaves its counters in the Grain state.
    """
    read = pipeline.mode or _preferred_read((pipeline,), provision=True)
    loading = pipeline._loading or _DEFAULT_LOADING
    if records is not None:
        pipeline = pipeline._then(Limit(records))
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
        lowering = _finite(lowering)
    # A token budget is fixed-length rows by now (`_budgeted`).
    build = _build(pipeline, read, loading, batch=_output_batch(pipeline), training=True,
                   lowering=lowering).tuned(pipeline).pooled(pipeline)
    return _Host(build.batches(pipeline, DataPartition()), packed + _batch_steps(pipeline)[0])


def _finite(lowering: _Lowering) -> _Lowering:
    """A non-repeated diagnostic pass without training's implicit root repeats."""
    return dataclasses.replace(lowering, parts_unending=False, unending=False, records_unending=False,
                               onward=False, rooted=False, repeated=True, epochs=1,
                               children=tuple(_finite(child) for child in lowering.children))


def iterate(
    pipeline: Pipeline,
    *,
    read: Read = "streamed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
) -> pygrain.IterDataset:
    """A finite record pass through the same lowering, for publication and export."""
    if any(
        isinstance(stage, (RandomMap, BatchStage, MapBatch, Workers, Repeat)) for stage in _steps(pipeline)
    ):
        raise ValueError("publication needs deterministic, unbatched pipelines without repeat")
    (pipeline,) = _made(_prepared((pipeline,), read, provision=provision, counts=({},)), read, ({},),
                        training=False)
    # Publication reads records, unbatched.
    build = _build(pipeline, read, loading, batch=1, training=False)
    rows = build.records(pipeline, DataPartition())
    return _as_iter(rows, loading)


def fitted[Entry](
    dataset: Dataset | Loader,
    validation: Dataset | Mapping[str, Dataset | Entry] | None,
    *,
    objective: ObjectiveData,
    seed: int = 0,
    examples: int | None = None,
) -> tuple[Loader, Mapping[str, Reader | Entry] | None]:
    """`dataset` and each validation dataset as `Trainer.fit` reads them,
    compiled with the objective's own stages and the run's data `seed`,
    their stores provisioned. A training dataset without a repeat repeats
    before its batch, so a run's length is its steps, epochs or tokens, or
    with `examples` reads that many records before they are packed and ends
    (`compile`'s `examples`); a validation dataset without a batch takes the
    training batch, and is read and loaded as it asks. A Loader is already
    compiled, and an entry that is no Dataset (a reader, an evaluation suite)
    is passed on as it is."""
    if isinstance(dataset, Dataset):
        chain = dataset._chain()
        loader = compile(chain if examples is not None else _endless(chain), objective=objective,
                         provision=True, seed=seed, examples=examples)
    else:
        loader = dataset
    named = {"val": validation} if isinstance(validation, Dataset) else validation
    if named is None:
        return loader, None
    return loader, {name: held(entry, objective=objective, batch=loader.batch, seed=seed)
                    if isinstance(entry, Dataset) else entry for name, entry in named.items()}


def _endless(chain: Pipeline) -> Pipeline:
    """`chain` read endlessly: a repeat before its batch where it has none."""
    if any(isinstance(stage, Repeat) for stage in chain.stages):
        return chain
    batching = [index for index, stage in enumerate(chain.stages)
                if isinstance(stage, (BatchStage, TokenBudget))]
    at = batching[0] if batching else len(chain.stages)
    return dataclasses.replace(chain, stages=(*chain.stages[:at], Repeat(None), *chain.stages[at:]))


def held(dataset: Dataset, *, objective: ObjectiveData | None = None, batch: int | None = None,
         seed: int = 0) -> Reader:
    """`dataset` read once to validate or evaluate on, in its own batches, else in batches of `batch`,
    read and loaded as it asks, its random steps drawing from `seed`; one that states neither batch is
    refused. A pass reads every record: a batch it fills only in part keeps the records it holds,
    filled out with copies its `VALID_ROWS` marks, where a training read would drop them."""
    chain = dataset._chain()
    if not any(isinstance(stage, (BatchStage, TokenBudget)) for stage in chain.stages):
        if batch is None:
            raise ValueError(f"{_name(chain)} states no batch; batch it (`.batch(n)`) or give batch=")
        chain = chain.batch(batch, drop_remainder=False)
    batched = (BatchStage, TokenBudget)
    chain = dataclasses.replace(chain, stages=tuple(
        dataclasses.replace(stage, drop_remainder=False) if isinstance(stage, batched) else stage
        for stage in chain.stages))
    # The chain's own batch is the one it is read in; validation's `batch` adds none to it.
    return validation(chain, batch=1, objective=objective, read=chain.mode,
                      loading=chain._loading or _DEFAULT_LOADING, provision=True, seed=seed)


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
) -> Reader:
    """Compile one finite evaluation branch without constructing a training
    stream. With no `read`, the sources choose it (`_preferred_read`)."""
    read = read or _preferred_read((pipeline,), provision=provision)
    if not any(isinstance(stage, (BatchStage, TokenBudget)) for stage in pipeline.stages):
        pipeline = pipeline.batch(batch, drop_remainder=False)
    prepared = _prepared((pipeline,), read, provision=provision, objective=objective, counts=({},))
    (pipeline,) = _made(prepared, read, ({},), training=False)
    packed = _coordinated(pipeline)
    # A token budget is fixed-length rows by now (`_budgeted`).
    build = _build(pipeline, read, loading, batch=_output_batch(prepared[0]), training=False, seed=seed)
    return functools.partial(_validation_reader, pipeline=pipeline, build=build.tuned(pipeline),
                             host=packed + _held_steps(pipeline) + host)


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
    of a terminal pack), which it returns."""
    from dew.data.cache import Cache
    from dew.data.generation import Generation
    from dew.data.packing import PackingPlan, plan_name

    if not isinstance(cache, Cache):
        raise TypeError("publish needs a Cache destination")
    if isinstance(pipeline.source, (Mixture, Phases)):
        raise ValueError("publish each source prefix before mixing")
    planned = _at(pipeline.stages, Pack)
    builder = None
    if planned is not None:
        packing = pipeline.stages[planned]
        if not isinstance(packing, Pack) or planned != len(pipeline.stages) - 1:
            raise ValueError("publication accepts one terminal packing stage after the cached prefix")
        builder = PackingPlan(packing)
        pipeline = dataclasses.replace(pipeline, stages=pipeline.stages[:planned])
    (pipeline,) = _prepared((pipeline,), read, provision=provision)
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        raise ValueError("configured publication needs one physical source")
    manifest = _cache_manifest(source, pipeline.stages, read)

    def published() -> Generation | None:
        generation = cache.generation(manifest)
        held = generation is not None and (
            builder is None or plan_name(builder.pack, "accepted") in generation.manifest["plan"])
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
