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
    _identity,
    _leaves,
    _manifest,
    _name,
    _order_at,
    _preferred_read,
    _promoted,
    _steps,
    _until,
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
    BatchStage,
    Bound,
    Configured,
    Expansion,
    FlatMap,
    Geometry,
    Group,
    HostStep,
    Interleave,
    Map,
    MapBatch,
    Mixture,
    ObjectiveData,
    Pack,
    Phases,
    Pipeline,
    Prefetch,
    RandomMap,
    Repeat,
    Shuffle,
    SingleReader,
    Slice,
    Stage,
    TokenBudget,
    TokenGeometry,
    Volatile,
    Workers,
    _rejects,
)
from dew.data.plan import Plan, StepIdentity, fingerprint
from dew.data.readers import _CanonicalPass, _Host, _PerShare, _Statistics
from dew.data.sources.base import Read, ShardManifest, Source, _consumed
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

    def snapshot(self) -> Mapping[str, object]:
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
        return _Facts(source, read, source.snapshot(), source.metadata())

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


def _geometry(pipeline: Pipeline) -> Geometry | None:
    declared = [
        stage.step
        for stage in pipeline.stages
        if isinstance(stage, (Map, RandomMap)) and isinstance(stage.step, TokenGeometry)
    ]
    if declared:
        last = declared[-1]
        return Geometry(last.max_length, last.token_fields, last.length_field)
    if isinstance(pipeline.source, (Mixture, Phases)):
        children = [_geometry(child) for child in pipeline.source.pipelines]
        if not children or any(child is None for child in children):
            return None
        known = [child for child in children if child is not None]
        if any(
            child.token_fields != known[0].token_fields or child.length_field != known[0].length_field
            for child in known
        ):
            raise ValueError("mixed token geometries need matching token fields and length columns")
        return dataclasses.replace(known[0], max_length=max(child.max_length for child in known))
    return None


def _token_geometry(pipeline: Pipeline, fallback: Geometry | None = None) -> Pipeline:
    if not any(isinstance(stage, TokenBudget) for stage in pipeline.stages):
        return pipeline
    geometry = _geometry(pipeline) or fallback
    if geometry is None:
        raise ValueError(
            f"token-budget pipeline {_name(pipeline)} needs a record step declaring TokenGeometry"
        )
    return dataclasses.replace(
        pipeline,
        stages=tuple(
            dataclasses.replace(stage, geometry=geometry) if isinstance(stage, TokenBudget) else stage
            for stage in pipeline.stages
        ),
    )


def _budget_rows(budget: TokenBudget) -> int:
    from dew.data.batching import bucket_lengths

    lengths = bucket_lengths(budget)
    return budget.tokens // min(lengths)


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
        if objective is not None and training:
            branches = (_objective_order(branches[0], objective), *branches[1:])
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
    provision: bool = True,
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
    train = _token_geometry(train)
    if validation is not None:
        validation = _token_geometry(validation, _geometry(train))
    batching = next((stage for stage in train.stages if isinstance(stage, (BatchStage, TokenBudget))), None)
    if batching is None:
        raise ValueError("compile needs one batch(global_size) stage on the training pipeline")
    batch = batching.size if isinstance(batching, BatchStage) else _budget_rows(batching)
    batch_tokens = None if isinstance(batching, BatchStage) else batching.tokens
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
    read, seed, loading, batch, global_order = (
        build.read,
        build.seed,
        build.loading,
        build.batch,
        build.global_order,
    )
    sources = [source for _, source in _leaves(original)]
    manifest = _manifest(
        {"sources": [source.snapshot() for source in sources]},
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
            for stage in _steps(train)
            if isinstance(stage, Map)
        ),
    )
    if loading.workers == "auto":
        tuning = dataclasses.replace(build, loading=dataclasses.replace(loading, workers=0))
        loading = loading.resolved(tuning.batched(train, DataPartition()))
        build = dataclasses.replace(build, loading=loading)
    train_host = tuple(stage.step for stage in train.stages if isinstance(stage, MapBatch)) + host
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
            ),
            host=tuple(stage.step for stage in validation.stages if isinstance(stage, MapBatch)) + host,
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
    )


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
    """Put objective record steps on each corpus before its mixture."""
    if isinstance(pipeline.source, (Mixture, Phases)):
        children = tuple(
            objective_steps(child, objective, train=train, read=read) for child in pipeline.source.pipelines
        )
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    boundary = (
        (Shuffle, Repeat, Group, BatchStage, TokenBudget)
        if read == "memory"
        else (Group, BatchStage, TokenBudget)
    )
    at = _until(pipeline.stages, boundary)
    suffix = pipeline.stages[at:]
    pipeline = dataclasses.replace(pipeline, stages=pipeline.stages[:at])
    for step in objective.train if train else objective.evaluate:
        if isinstance(step, Expansion):
            pipeline = pipeline.flat_map(step)
        elif isinstance(step, Configured):
            pipeline = pipeline.then(step)
        elif _rejects(step):
            pipeline = pipeline.filter_map(step)
        elif isinstance(step, pygrain.RandomMapTransform):
            pipeline = pipeline.random_map(step)
        else:
            pipeline = pipeline.map(step)
    return dataclasses.replace(pipeline, stages=(*pipeline.stages, *suffix))


def _objective_order(pipeline: Pipeline, objective: ObjectiveData) -> Pipeline:
    def shared(stage: Stage) -> bool:
        return isinstance(stage, RandomMap) and (
            (stage.configured is not None and stage.configured.step.scope == "run")
            or any(stage.step is step for step in objective.train)
        )

    selected: tuple[Stage, ...] = ()
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        choices = [tuple(stage for stage in child.stages if shared(stage)) for child in source.pipelines]
        if choices and all(stages == choices[0] for stages in choices):
            selected = choices[0]
            children = tuple(
                dataclasses.replace(child, stages=tuple(stage for stage in child.stages if not shared(stage)))
                for child in source.pipelines
            )
            source = dataclasses.replace(source, pipelines=children)
    stages = tuple(stage for stage in pipeline.stages if not shared(stage))
    selected = (*selected, *(stage for stage in pipeline.stages if shared(stage)))
    at = _until(stages, (BatchStage, TokenBudget))
    return dataclasses.replace(pipeline, source=source, stages=(*stages[:at], *selected, *stages[at:]))


def _objective_branch(pipeline: Pipeline, objective: ObjectiveData, *, train: bool, read: Read) -> Pipeline:
    branch = objective_steps(pipeline, objective, train=train, read=read)
    if isinstance(branch.source, Mixture):
        children = branch.source.pipelines
        suffixes = [
            tuple(stage for stage in child.stages if isinstance(stage, RandomMap)) for child in children
        ]
        if suffixes and all(suffix == suffixes[0] for suffix in suffixes):
            children = tuple(
                dataclasses.replace(
                    child, stages=tuple(stage for stage in child.stages if not isinstance(stage, RandomMap))
                )
                for child in children
            )
            branch = dataclasses.replace(
                branch,
                source=dataclasses.replace(branch.source, pipelines=children),
                stages=(*suffixes[0], *branch.stages),
            )
    ordering = tuple(stage for stage in branch.stages if isinstance(stage, (Shuffle, Repeat)))
    without = tuple(stage for stage in branch.stages if not isinstance(stage, (Shuffle, Repeat)))
    at, _ = _order_at(without, read)
    if ordering and read == "memory" and at < len(without):
        branch = dataclasses.replace(branch, stages=(*without[:at], *ordering, *without[at:]))
    if objective.group is not None and not any(isinstance(stage, TokenBudget) for stage in branch.stages):
        at = _until(branch.stages, (RandomMap, BatchStage, TokenBudget))
        branch = dataclasses.replace(
            branch, stages=(*branch.stages[:at], Group(objective.group), *branch.stages[at:])
        )
    return branch.map_batches(*(objective.train_batch if train else objective.evaluate_batch))


def _metadata(pipelines: Sequence[Pipeline], read: Read) -> Mapping[str, object]:
    combined: Batch = {}
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
                    combined.pop(key, None)
    return {**combined, "slots": slots} if slots else combined


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
    rows, _ = build.records(pipeline, DataPartition())
    return _as_iter(rows, loading)


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
    if not any(isinstance(stage, BatchStage) for stage in pipeline.stages):
        pipeline = pipeline.batch(batch, drop_remainder=False)
    (pipeline,) = _prepared((pipeline,), read, provision=provision, objective=objective, counts=({},))
    if read == "memory":
        pipeline = _materialize(pipeline, {}, training=False)
    lowering = _analyzed(pipeline, read, validation=True)
    build = _Build(read, loading, seed, batch, budget, lowering, training=False)
    if loading.workers == "auto":
        tuning = dataclasses.replace(build, loading=dataclasses.replace(loading, workers=0))
        loading = loading.resolved(tuning.batched(pipeline, DataPartition()))
        build = dataclasses.replace(build, loading=loading)
    stages = tuple(stage.step for stage in pipeline.stages if isinstance(stage, MapBatch)) + host
    return functools.partial(_validation_reader, pipeline=pipeline, build=build, host=stages)


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
