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
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Protocol, runtime_checkable

import grain.python as pygrain
import numpy as np

from dew.data.analysis import (
    _analyzed,
    _at,
    _cache_manifest,
    _capabilities,
    _cut,
    _identity,
    _leaves,
    _Lowering,
    _manifest,
    _name,
    _preferred_read,
    _promoted,
    _Ran,
    _steps,
    _until,
    _written,
)
from dew.data.dataset import (
    Batch,
    Closeable,
    DataPartition,
    GlobalReader,
    GlobalStream,
    Loader,
    Loading,
    PhasedStream,
    Reader,
    Records,
    weight_counts,
)
from dew.data.lowering import _as_iter, _Build, _epoch_records, _Locality, _split_spans, _workers
from dew.data.materialize import _materialize, _Memory
from dew.data.mixing import CorpusCounts, SearchBudget
from dew.data.pipeline import (
    Artifact,
    Batched,
    BatchStage,
    Bound,
    Configured,
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
    Volatile,
    Windows,
    Workers,
    _rejects,
)
from dew.data.plan import Plan, StepIdentity, fingerprint
from dew.data.readers import _CanonicalPass, _Host, _PerShare, _Statistics
from dew.data.sources.base import Read, ShardManifest, Source, _consumed
from dew.data.steps.generic import Coordinated
from dew.records import JSON, integer, json_value, record

_DEFAULT_LOADING = Loading()
_DEFAULT_BUDGET = SearchBudget()


@runtime_checkable
class _OrderFacts(Protocol):
    def order_fingerprints(self) -> Mapping[str, str]: ...


def _cache_references(pipeline: Pipeline, read: Read) -> tuple[str, ...]:
    return tuple(
        stage.cache.reference(_cache_manifest(source, leaf.stages[:index], read))
        for leaf, source in _leaves(pipeline)
        for index, stage in enumerate(leaf.stages)
        if isinstance(stage, Artifact)
    )


@dataclasses.dataclass(frozen=True)
class _Facts:
    source: Source
    read: Read
    stated_snapshot: Mapping[str, object]
    stated_metadata: Mapping[str, object]

    def describe(self) -> str:
        return self.source.describe()

    def fingerprint(self) -> Mapping[str, object]:
        return self.stated_snapshot

    def metadata(self) -> Mapping[str, object]:
        return self.stated_metadata

    def supports(self, read: Read) -> bool:
        return read == self.read or self.source.supports(read)

    def records(self, read: Read) -> Records:
        return self.source.records(read)

    def manifest(self) -> ShardManifest:
        return self.source.manifest()

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


def _pin_sources(pipelines: Sequence[Pipeline], read: Read) -> tuple[Pipeline, ...]:
    sources = {id(source): source for branch in pipelines for _, source in _leaves(branch)}

    def pin(source: Source) -> _Facts:
        if isinstance(source, _Facts):
            return source
        source = _consumed(source, read)
        if read == "memory":
            source.records(read)
        return _Facts(source, read, source.fingerprint(), source.metadata())

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as readers:
        pinned = dict(zip(sources, readers.map(pin, sources.values()), strict=True))

    def replace(branch: Pipeline) -> Pipeline:
        source = branch.source
        source = (
            dataclasses.replace(source, pipelines=tuple(replace(child) for child in source.pipelines))
            if isinstance(source, (Mixture, Phases))
            else pinned[id(source)]
        )
        return dataclasses.replace(branch, source=source)

    return tuple(replace(branch) for branch in pipelines)


def _tokenizer_contract(pipelines: Sequence[Pipeline]) -> None:
    declared: dict[str, JSON] = {}
    for pipeline in pipelines:
        for leaf, source in _leaves(pipeline):
            metadata = source.metadata()
            for key in ("tokenizer", "vocab_size", "eos_id", "tokenizer_revision"):
                if key not in metadata:
                    continue
                value = json_value(metadata[key], key)
                if key in declared and value != declared[key]:
                    raise ValueError(
                        f"corpus {_name(leaf)} has {key}={value!r}, expected {declared[key]!r}; "
                        "mixed corpora must share the tokenizer contract"
                    )
                declared[key] = value


def _configured(
    pipeline: Pipeline,
    shared: Mapping[str, object],
    read: Read,
    scope: Literal["source", "run"] | None = None,
) -> Pipeline:
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        children = tuple(_configured(child, shared, read, scope) for child in source.pipelines)
        source = dataclasses.replace(source, pipelines=children)
        local = shared
    else:
        local = source.metadata()
    stages: list[Stage] = []
    for stage in pipeline.stages:
        if not isinstance(stage, Configured) or (scope is not None and stage.scope != scope):
            stages.append(stage)
            continue
        metadata = shared if stage.scope == "run" else local
        missing = sorted(set(stage.needs) - metadata.keys())
        if missing:
            raise ValueError(
                f"{_name(pipeline)} step {_identity(stage).name} needs metadata {missing}; "
                "provide producer metadata, recipe fields or read='memory'"
            )
        binding = Bound(stage, {key: metadata[key] for key in stage.needs})
        resolved = stage.factory(metadata)
        if isinstance(resolved, pygrain.RandomMapTransform) != stage.random:
            raise ValueError(
                f"Configured {_identity(stage).name} random={stage.random} disagrees with its step"
            )
        if isinstance(resolved, Expansion):
            stages.append(FlatMap(resolved, configured=binding))
        elif isinstance(resolved, pygrain.RandomMapTransform):
            stages.append(RandomMap(resolved, rejects=_rejects(resolved), configured=binding))
        else:
            stages.append(Map(resolved, rejects=_rejects(resolved), configured=binding))
    return dataclasses.replace(pipeline, source=source, stages=tuple(stages))


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
    if isinstance(pipeline.source, (Mixture, Phases)):
        lengths = {_row_tokens(child) for child in pipeline.source.pipelines}
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
    windows = dataclasses.replace(pipeline, stages=pipeline.stages[:at]).then(objective.pack)
    for step in objective.packed:
        windows = windows.map(step)
    return dataclasses.replace(windows, stages=(*windows.stages, *pipeline.stages[at:]))


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
) -> tuple[Pipeline, ...]:
    """The one preparation compile, iterate, validation and publish lower from.

    With `training` the first branch trains; every other branch is scored.
    Each branch is checked, its generations are promoted to their sources and
    it is checked again; its sources are provisioned and pinned to the facts
    they state once, objective steps take their places, and configured steps
    bind to the sources' metadata. With read='memory' and `counts`, the
    source-scoped steps run and materialize first, so run-scoped steps bind
    to measured facts; publication passes no counts and reads nothing before
    its fill. The caller materializes what remains.
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
    budget: SearchBudget = _DEFAULT_BUDGET,
    resume: Literal["exact", "non-exact"] = "exact",
) -> Loader:
    """Check every leaf before opening readers, then build both branches.
    With no `read`, the sources choose it (`_preferred_read`)."""
    branches = (train,) if validation is None else (train, validation)
    read = read or train.mode or _preferred_read(lambda _: branches, provision=provision)
    loading = loading or train.loading or _DEFAULT_LOADING
    if isinstance(train.source, Phases):
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
            budget=budget,
            resume=resume,
        )
    if rejections == "mask":
        raise ValueError(
            "rejections='mask' needs an objective proven mask-safe; no objective currently declares it"
        )
    stated = next((stage for stage in train.stages if isinstance(stage, TokenBudget)), None)
    batch_tokens = None if stated is None else stated.tokens
    counts: dict[str, CorpusCounts] = {}
    validation_counts: dict[str, CorpusCounts] = {}
    branches = _prepared(
        branches,
        read,
        provision=provision,
        training=True,
        objective=objective,
        counts=(counts,) if validation is None else (counts, validation_counts),
    )
    train, validation = branches[0], (None if validation is None else branches[1])
    batching = next((stage for stage in train.stages if isinstance(stage, BatchStage)), None)
    if batching is None:
        raise ValueError("compile needs one batch(global_size) stage on the training pipeline")
    batch = batching.size
    original = train
    if read == "memory":
        train = _materialize(train, counts)
        validation = (
            None if validation is None else _materialize(validation, validation_counts, training=False)
        )
        if validation_counts and not any(count.accepted for count in validation_counts.values()):
            validation = None
    build = _Build(read, loading, seed, batch, budget, _analyzed(train, read), training=True)
    return _compiled(
        train,
        validation,
        original,
        build,
        host=host,
        resume=resume,
        counts=counts,
        validation_counts=validation_counts,
        batch_tokens=batch_tokens,
    )


def _compiled(
    train: Pipeline,
    validation: Pipeline | None,
    original: Pipeline,
    build: _Build,
    *,
    host: tuple[HostStep, ...],
    resume: Literal["exact", "non-exact"],
    counts: Mapping[str, CorpusCounts],
    validation_counts: Mapping[str, CorpusCounts],
    batch_tokens: int | None,
) -> Loader:
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
            *(_identity(stage) for stage in _steps(original) if not isinstance(stage, (Workers, Prefetch))),
            *(StepIdentity.of(step) for step in host),
        ),
    )
    plan = Plan(
        "global" if global_order else "per-share",
        manifest,
        {"read": read, "seed": seed, **build.lowering.order},
        _derived(train, build),
        cache=_cache_references(original, read),
        volatile=any(
            isinstance(stage.step, Volatile) and stage.step.volatile
            for stage in (*_steps(original), *(stage for source in sources for stage in _written(source, ())))
            if isinstance(stage, Map)
        ),
    )
    build = build.tuned(train)
    train_host, device = _batch_steps(train)
    packed = _coordinated(original)
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
                build, lowering=_analyzed(validation, read, validation=True), training=False
            ).tuned(validation),
            host=packed + _held_steps(validation) + host,
        )
    )
    records = _epoch_records(train, build)
    if records == 0:
        raise ValueError(f"corpus {_name(train)} has no accepted rows")
    if records is not None and records < batch:
        raise ValueError(f"corpus {_name(train)} has {records} records, fewer than one batch of {batch}")
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
        batch_tokens=batch_tokens,
        device=device,
    )


def _batch_steps(pipeline: Pipeline) -> tuple[tuple[HostStep, ...], tuple[DeviceStep, ...]]:
    """The training batch's steps: those the reader runs on the host, then
    those the compiled step runs on the accelerator (`DeviceStep`)."""
    steps = [stage.step for stage in pipeline.stages if isinstance(stage, MapBatch)]
    return (tuple(step for step in steps if not isinstance(step, DeviceStep)),
            tuple(step for step in steps if isinstance(step, DeviceStep)))


def _coordinated(pipeline: Pipeline) -> tuple[Coordinated, ...]:
    """The step that lays each packed window's documents out as the model
    reads them (`Coordinated`), one per packed ids field, before the stages
    that pack are planned away: a window holds no more documents than tokens.
    A pack counts wherever it runs: as a stage, in the stages a memory read
    already ran, or as the plan a source reads its windows by (`PackedSource`)."""
    from dew.data.packing import PackedSource

    packs = [stage for stage in _steps(pipeline) if isinstance(stage, Pack)]
    for _, source in _leaves(pipeline):
        packs += [stage for stage in _written(source, ()) if isinstance(stage, Pack)]
        while isinstance(source, (_Facts, _Ran)):
            source = source.source if isinstance(source, _Facts) else source.original
        if isinstance(source, PackedSource):
            packs.append(source.plan.pack)
    caps: dict[tuple[str, ...], int] = {}
    for stage in packs:
        caps[stage.fields] = max(caps.get(stage.fields, 0), stage.seq_len)
    return tuple(Coordinated(fields, cap) for fields, cap in caps.items())


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
    budget: SearchBudget,
    resume: Literal["exact", "non-exact"],
) -> Loader:
    source = train.source
    if not isinstance(source, Phases):
        raise TypeError("phased compilation needs a Phases source")
    _tokenizer_contract((train,) if validation is None else (train, validation))
    consumed: dict[str, int] = {}
    built: list[tuple[Loader, int | None]] = []
    previous = 0
    for index, (pipeline, until) in enumerate(zip(source.pipelines, source.until_steps, strict=True)):
        continued = _continued(pipeline, consumed).then(*train.stages)
        loaded = compile(
            continued,
            validation=validation if index == 0 else None,
            host=host,
            objective=objective,
            read=read,
            provision=provision,
            loading=loading,
            seed=seed,
            rejections=rejections,
            budget=budget,
            resume=resume,
        )
        if loaded.plan is None or loaded.plan.lowering != "global":
            raise ValueError(
                f"phase {index} needs a global order; "
                "streamed, rejecting and online-packed phases are refused"
            )
        end = None if until is None else until * loaded.batch
        if built and loaded.device != built[0][0].device:
            # The run applies one set of device steps to every batch (`Loader.device`), so a phase
            # whose own differ would run the first phase's.
            raise ValueError(f"phase {index} ends in other device steps than phase 0; every phase of "
                             f"a run runs the same ones, written after the phases")
        built.append((loaded, end))
        if end is not None:
            _delivered(pipeline, end - previous, consumed)
            previous = end

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

    return dataclasses.replace(built[0][0], train=training)


def _continued(pipeline: Pipeline, consumed: Mapping[str, int]) -> Pipeline:
    if isinstance(pipeline.source, Phases):
        raise ValueError("nested phases are not an indexed record order")
    if isinstance(pipeline.source, Mixture):
        children = tuple(_continued(child, consumed) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
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
        weights = tuple(weight / sum(source.weights) for weight in source.weights)
        for child, taken in zip(source.pipelines, weight_counts(weights, count), strict=True):
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
        children = tuple(
            objective_steps(child, objective, train=train, read=read) for child in pipeline.source.pipelines
        )
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
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
    for step in steps:
        if isinstance(step, Expansion):
            pipeline = pipeline.flat_map(step)
        elif isinstance(step, Configured):
            pipeline = pipeline.then(step)
        elif isinstance(step, pygrain.RandomMapTransform):
            pipeline = pipeline.random_filter_map(step) if _rejects(step) else pipeline.random_map(step)
        elif _rejects(step):
            pipeline = pipeline.filter_map(step)
        else:
            pipeline = pipeline.map(step)
    return pipeline


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
    # prefetcher places the rest (`dew.training.distributed.device_fields`).
    return branch.then(*(MapBatch(step) for step in steps))


def _metadata(pipelines: Sequence[Pipeline], read: Read) -> Mapping[str, object]:
    combined: Batch = {}
    conflicted: set[str] = set()
    slots: dict[str, int] = {}
    for pipeline in pipelines:
        for _, source in _leaves(pipeline):
            if isinstance(source, _Memory) and not source.held:
                continue
            facts = source.metadata()
            for key, value in facts.items():
                if key == "slots":
                    for name, size in record(value, "slots").items():
                        slots[name] = max(slots.get(name, 0), integer(size, f"slots.{name}"))
                elif key not in combined:
                    combined[key] = value
                elif combined[key] != value:
                    # A fact two sources state differently is no fact of the run, whatever a third states.
                    conflicted.add(key)
    shared = {key: value for key, value in combined.items() if key not in conflicted}
    return {**shared, "slots": slots} if slots else shared


def _diagnostic(pipeline: Pipeline, *, records: int | None = None) -> _Host:
    """Inspect through training's lowering, finite unless a repeat asks otherwise.

    A bounded record pass runs lazily without loader read-ahead or
    eager memory transforms, and leaves its counters in the Grain state.
    """
    read = pipeline.mode or _preferred_read(lambda _: (pipeline,), provision=True)
    loading = pipeline.loading or _DEFAULT_LOADING
    if records is not None:
        pipeline = pipeline.then(Limit(records))
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
    (pipeline,) = _prepared((pipeline,), read, provision=True, training=True,
                            counts=(counts,) if records is None else None)
    packed = _coordinated(pipeline)
    if read == "memory" and records is None:
        pipeline = _materialize(pipeline, counts)
    lowering = _analyzed(pipeline, read)
    if records is not None:
        lowering = _cut(lowering, per_share=True)
    if finite:
        lowering = _finite(lowering)
    # A token budget is fixed-length rows by now (`_budgeted`).
    batch = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage)).size
    build = _Build(read, loading, 0, batch, _DEFAULT_BUDGET, lowering, training=True)
    return _Host(build.tuned(pipeline).batches(pipeline, DataPartition()), packed + _batch_steps(pipeline)[0])


def _finite(lowering: _Lowering) -> _Lowering:
    """A non-repeated diagnostic pass without training's implicit root repeats."""
    return dataclasses.replace(lowering, endless=False, repeated=True, epochs=1,
                               children=tuple(_finite(child) for child in lowering.children))


def iterate(
    pipeline: Pipeline,
    *,
    read: Read = "streamed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
    budget: SearchBudget = _DEFAULT_BUDGET,
) -> pygrain.IterDataset:
    """A finite record pass through the same lowering, for publication and export."""
    if any(
        isinstance(stage, (RandomMap, BatchStage, MapBatch, Workers, Repeat)) for stage in _steps(pipeline)
    ):
        raise ValueError("publication needs deterministic, unbatched pipelines without repeat")
    (pipeline,) = _prepared((pipeline,), read, provision=provision, counts=({},))
    if read == "memory":
        pipeline = _materialize(pipeline, {}, training=False)
    lowering = _analyzed(pipeline, read, validation=True)
    build = _Build(read, loading, 0, 1, budget, lowering, training=False)
    rows = build.records(pipeline, DataPartition())
    return _as_iter(rows, loading)


def fitted[Entry](
    dataset: Dataset | Loader,
    validation: Dataset | Mapping[str, Dataset | Entry] | None,
    *,
    objective: ObjectiveData,
) -> tuple[Loader, Mapping[str, Reader | Entry] | None]:
    """`dataset` and each validation dataset as `Trainer.fit` reads them,
    compiled with the objective's own stages, their stores provisioned. A
    training dataset without a repeat repeats before its batch, so a run's
    length is its steps, epochs or tokens; a validation dataset without a
    batch takes the training batch, and is read and loaded as it asks. A
    Loader is already compiled, and an entry that is no Dataset (a reader,
    an evaluation suite) is passed on as it is."""
    if isinstance(dataset, Dataset):
        train = _endless(dataset.chain())
        loader = compile(train, objective=objective, provision=True)
    else:
        loader = dataset
    named = {"val": validation} if isinstance(validation, Dataset) else validation
    if named is None:
        return loader, None
    return loader, {name: _held(entry.chain(), loader.batch, objective) if isinstance(entry, Dataset)
                    else entry for name, entry in named.items()}


def _endless(chain: Pipeline) -> Pipeline:
    """`chain` read endlessly: a repeat before its batch where it has none."""
    if any(isinstance(stage, Repeat) for stage in chain.stages):
        return chain
    batching = [index for index, stage in enumerate(chain.stages)
                if isinstance(stage, (BatchStage, TokenBudget))]
    at = batching[0] if batching else len(chain.stages)
    return dataclasses.replace(chain, stages=(*chain.stages[:at], Repeat(None), *chain.stages[at:]))


def _held(chain: Pipeline, batch: int, objective: ObjectiveData) -> Reader:
    """`chain` read once to validate on, batched by `batch` where it states no
    batch, read and loaded as it asks."""
    return validation(chain, batch=batch, objective=objective, read=chain.mode,
                      loading=chain.loading or _DEFAULT_LOADING, provision=True)


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
    budget: SearchBudget = _DEFAULT_BUDGET,
) -> Reader:
    """Compile one finite evaluation branch without constructing a training
    stream. With no `read`, the sources choose it (`_preferred_read`)."""
    read = read or _preferred_read(lambda _: (pipeline,), provision=provision)
    if not any(isinstance(stage, (BatchStage, TokenBudget)) for stage in pipeline.stages):
        pipeline = pipeline.batch(batch, drop_remainder=False)
    (pipeline,) = _prepared((pipeline,), read, provision=provision, objective=objective, counts=({},))
    packed = _coordinated(pipeline)
    # A token budget is fixed-length rows by now (`_budgeted`).
    batch = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage)).size
    if read == "memory":
        pipeline = _materialize(pipeline, {}, training=False)
    lowering = _analyzed(pipeline, read, validation=True)
    build = _Build(read, loading, seed, batch, budget, lowering, training=False)
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
    """Fill and seal a deterministic prefix, then publish its certified generation."""
    from dew.data.cache import Cache

    if not isinstance(cache, Cache):
        raise TypeError("publish needs a Cache destination")
    if isinstance(pipeline.source, (Mixture, Phases)):
        raise ValueError("publish each source prefix before mixing")
    planned = _at(pipeline.stages, Pack)
    builder = None
    if planned is not None:
        from dew.data.packing import PackingPlan

        packing = pipeline.stages[planned]
        if not isinstance(packing, Pack) or planned != len(pipeline.stages) - 1:
            raise ValueError("publication accepts one terminal packing stage after the cached prefix")
        builder = PackingPlan(packing)
        pipeline = dataclasses.replace(pipeline, stages=pipeline.stages[:planned])
    (pipeline,) = _prepared((pipeline,), read, provision=provision)
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        raise ValueError("configured publication needs one physical source")
    cached = pipeline.cache(cache)
    manifest = _cache_manifest(source, pipeline.stages, read)
    iterator = iter(iterate(cached, read=read, loading=loading))
    try:
        for _ in iterator:
            pass
    finally:
        iterator.close()
    return cache.publish(manifest, source, plan=builder)
