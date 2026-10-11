"""Compile the same pipeline to an elastic order or an exact per-share reader."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, Protocol, get_args, runtime_checkable

import grain.experimental as gx
import grain.python as pygrain
import numpy as np

from dew.data.dataset import (
    Batch,
    Closeable,
    DataPartition,
    Dataset,
    FilledBatch,
    GlobalReader,
    GlobalStream,
    LengthGroups,
    Loading,
    PhasedStream,
    Reader,
    Records,
    _Grouped,
    _WorkerBatches,
    weight_counts,
)
from dew.data.mix import (
    Accepted,
    CorpusCounts,
    PrimedMix,
    SearchBudget,
    TokenDeficit,
    primed_maps,
    token_weights,
)
from dew.data.pipeline import (
    Artifact,
    BatchStage,
    Configured,
    Expansion,
    Filter,
    FlatMap,
    Geometry,
    Group,
    HostStep,
    Interleave,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    Mixture,
    ObjectiveData,
    Pack,
    Phases,
    Pipeline,
    Prefetch,
    RandomMap,
    Rebatch,
    Record,
    Rejected,
    Repeat,
    ShardByHost,
    Shuffle,
    SingleReader,
    Slice,
    Stage,
    TokenBudget,
    TokenGeometry,
    TokenSplit,
    Volatile,
    Windows,
    Workers,
    _rejects,
)
from dew.data.plan import Manifest, Plan, StepIdentity, fingerprint
from dew.data.readers import _CanonicalPass, _FillShare, _GroupedRows, _Host, _PerShare, _Statistics
from dew.data.sources.base import READS, Read, Shard, ShardManifest, Source
from dew.data.sources.memory import Memory
from dew.data.tokens import _WrappingDataset
from dew.records import JSON, integer, json_value, record

if TYPE_CHECKING:
    from dew.data.cache import Cache

type Rows = pygrain.MapDataset | pygrain.IterDataset
_log = logging.getLogger(__name__)
_DEFAULT_LOADING = Loading()
_DEFAULT_BUDGET = SearchBudget()
_STAGES = get_args(Stage.__value__)


@runtime_checkable
class _Provisionable(Protocol):
    def provision(self) -> Source: ...


@runtime_checkable
class _Locality(Protocol):
    def locality(self) -> tuple[tuple[int, ...], bool]: ...


@runtime_checkable
class _OrderFacts(Protocol):
    def order_fingerprints(self) -> Mapping[str, str]: ...


@runtime_checkable
class _Wrapping(Protocol):
    source: Source


def _preferred_read(pipelines: Sequence[Pipeline]) -> Read:
    """The read a run's sources choose when it states none: memory for records
    held in memory (`Memory`, through any source that wraps one), indexed for
    a store that reads by index, and streamed otherwise. A mixture takes the
    least eager of its corpora's choices."""
    chosen: list[Read] = []
    for source in (source for pipeline in pipelines for _, source in _leaves(pipeline)):
        held = source
        while isinstance(held, _Wrapping):
            held = held.source
        if isinstance(held, Memory):
            chosen.append("memory")
        else:
            chosen.append("indexed" if source.supports("indexed") else "streamed")
    return max(chosen, key=READS.index)


def _at(stages: Sequence[Stage], kinds: type | tuple[type, ...]) -> int | None:
    """The index of the first of `stages` that is one of `kinds`, if any."""
    return next((index for index, stage in enumerate(stages) if isinstance(stage, kinds)), None)


def _until(stages: Sequence[Stage], kinds: type | tuple[type, ...]) -> int:
    """How many of `stages` come before the first one of `kinds`."""
    at = _at(stages, kinds)
    return len(stages) if at is None else at


def _leaves(pipeline: Pipeline) -> Iterator[tuple[Pipeline, Source]]:
    """Each corpus under a pipeline's mixtures and phases, with the source it reads."""
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        for child in source.pipelines:
            yield from _leaves(child)
    else:
        yield pipeline, source


def _steps(pipeline: Pipeline) -> Iterator[Stage]:
    if isinstance(pipeline.source, (Mixture, Phases)):
        for child in pipeline.source.pipelines:
            yield from _steps(child)
    yield from pipeline.stages


def _name(pipeline: Pipeline) -> str:
    if pipeline.name is not None:
        return pipeline.name
    return "mixture" if isinstance(pipeline.source, (Mixture, Phases)) else pipeline.source.describe()


def _identity(stage: Stage, *, cache: bool = False) -> StepIdentity:
    if isinstance(stage, Artifact):
        return StepIdentity(f"dew.data.cache.{type(stage.cache).__name__}", stage.cache.url, exact=True)
    if isinstance(stage, Configured):
        return StepIdentity.of(stage.factory, cache=cache)
    if isinstance(stage, (Map, RandomMap, MapWithIndex, FlatMap, MapBatch)):
        identity = StepIdentity.of(stage.step, stage.identity, cache=cache)
        if isinstance(stage, (Map, RandomMap, FlatMap)) and stage.configured:
            identity = dataclasses.replace(
                identity,
                config=json.dumps({"step": identity.config, "configured": stage.configured}, sort_keys=True),
            )
        return identity
    if isinstance(stage, Filter):
        return StepIdentity.of(stage.predicate, stage.identity, cache=cache)
    return StepIdentity.of(stage)


def _cache_manifest(source: Source, prefix: Sequence[Stage], read: Read) -> Manifest:
    return Manifest.of(
        source.snapshot(),
        tuple(_identity(stage, cache=True) for stage in prefix),
        fill={"read": read, "row_ids": "sharded" if read == "streamed" else "indexed"},
    )


def _cache_references(pipeline: Pipeline, read: Read) -> tuple[str, ...]:
    return tuple(
        stage.cache.reference(_cache_manifest(source, leaf.stages[:index], read))
        for leaf, source in _leaves(pipeline)
        for index, stage in enumerate(leaf.stages)
        if isinstance(stage, Artifact)
    )


def _promoted(pipeline: Pipeline) -> Pipeline:
    """The pipeline reading each published generation it caches as a source,
    in place of the steps the generation already ran, and its published
    packing in place of the pack stage that follows it."""
    from dew.data.cache import Generation

    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        source = dataclasses.replace(source, pipelines=tuple(_promoted(child) for child in source.pipelines))
    for index, stage in enumerate(pipeline.stages):
        if isinstance(stage, Artifact) and isinstance(stage.cache, Generation):
            from dew.data.packing import PackedSource, plan_name

            suffix = pipeline.stages[index + 1 :]
            promoted: Source = stage.cache.source()
            if suffix and isinstance(suffix[0], Pack):
                pack = suffix[0]
                plans = record(stage.cache.manifest.get("plan", {}), "published plans")
                if plan_name(pack, "accepted") in plans:
                    promoted = PackedSource(promoted, stage.cache, pack)
                    suffix = suffix[1:]
            return dataclasses.replace(pipeline, source=promoted, stages=suffix)
    return dataclasses.replace(pipeline, source=source)


def _filled(stage: Artifact) -> Cache:
    """The cache an artifact stage fills. A generation is read as a source,
    which `_promoted` puts in its place before anything opens."""
    from dew.data.cache import Cache

    if not isinstance(stage.cache, Cache):
        raise ValueError(f"generation {stage.cache.url} must be promoted to a source before it is read")
    return stage.cache


def _check_batch(stage: Stage, *, batched: bool, budgeted: bool, name: str) -> tuple[bool, bool]:
    if isinstance(stage, (BatchStage, TokenBudget)):
        if batched:
            raise ValueError(f"pipeline {name} is already batched; use rebatch")
        return True, isinstance(stage, TokenBudget)
    if isinstance(stage, Rebatch):
        if not batched:
            raise ValueError(f"rebatch in {name} must follow batch")
        if budgeted:
            raise ValueError(f"rebatch in {name} changes the fixed token budget; use batch(tokens=)")
    return batched, budgeted


_EAGER = (Map, MapWithIndex, Filter, FlatMap, Pack, Windows, TokenSplit)
"""The stages a memory read runs over the whole source before the first other stage."""


def _random(stage: Stage) -> bool:
    """Whether a stage draws from the record rng, so what it makes differs
    each epoch and no cache can hold it."""
    return isinstance(stage, RandomMap) or (isinstance(stage, Configured) and stage.random)


def _drops(stage: Stage) -> bool:
    """Whether a stage can drop or make rows, so the rows after it are not
    the rows before it, one for one."""
    return isinstance(stage, (Filter, FlatMap, Pack, Windows)) or (
        isinstance(stage, (Map, MapWithIndex, RandomMap)) and stage.rejects
    )


@dataclasses.dataclass(frozen=True)
class _Lowering:
    """What a pipeline's stages mean for its lowering, analyzed once in one
    walk (`_analyzed`), which also refuses what no lowering takes.

    `per_share` cuts the order into shares at the read, for an iterator order
    that cannot keep the global contract; otherwise one global order is cut
    just before batching, after every stage that defines it, so a prefetch
    never moves the cut and a limit counts the whole order. `cache` is the
    cache stage a memory read fills, and the stages before `eager` (after that
    cache) are the ones a memory read runs over the whole source. `order` is
    the canonical order a resume must find unchanged, and `endless` whether
    the training order never runs out.
    """

    per_share: bool
    cache: int | None
    eager: int
    order: Mapping[str, object]
    endless: bool


def _analyzed(
    pipeline: Pipeline, read: Read, *, validation: bool = False, promotion: bool = False
) -> _Lowering:
    source = pipeline.source
    nested = source.pipelines if isinstance(source, (Mixture, Phases)) else ()
    children = [_analyzed(child, read, validation=validation, promotion=promotion) for child in nested]
    ancestry = tuple(stage for child in nested for stage in _steps(child))
    stages = pipeline.stages
    cache = _at(stages, Artifact)
    if nested or (cache is not None and any(isinstance(stage, Configured) for stage in stages[:cache])):
        cache = None
    eager = 0 if cache is None else cache + 1
    while read == "memory" and not nested and eager < len(stages) and isinstance(stages[eager], _EAGER):
        eager += 1
    per_share = _checked(pipeline, read, ancestry, eager, validation=validation, promotion=promotion)
    if isinstance(source, Mixture) and source.tokens is not None:
        per_share = per_share or any(
            isinstance(child.source, (Mixture, Phases)) or "valid_targets" not in child.source.metadata()
            for child in nested
        )
    steps = (*ancestry, *stages)
    repeats = [stage.epochs for stage in steps if isinstance(stage, Repeat)]
    endless = not any(isinstance(stage, Limit) for stage in steps) and (
        not repeats or any(epochs is None for epochs in repeats)
    )
    if isinstance(source, Phases):
        order: Mapping[str, object] = {
            "phases": [child.order for child in children],
            "until_steps": list(source.until_steps),
        }
    else:
        order = {
            **(
                {
                    "corpora": [child.order for child in children],
                    "weights": list(source.weights),
                    "shares": "tokens" if source.tokens is not None else "accepted rows",
                }
                if isinstance(source, Mixture)
                else {"source": source.describe()}
            ),
            "stages": [
                repr(stage)
                for stage in stages
                if isinstance(stage, (Shuffle, Repeat, Slice, Limit, TokenSplit, Pack, BatchStage))
            ],
            "groups": [stage.groups.window for stage in stages if isinstance(stage, Group)],
            "token_budgets": [
                dataclasses.asdict(stage) for stage in stages if isinstance(stage, TokenBudget)
            ],
        }
    return _Lowering(per_share or any(child.per_share for child in children), cache, eager, order, endless)


def _checked(
    pipeline: Pipeline,
    read: Read,
    ancestry: tuple[Stage, ...],
    eager: int,
    *,
    validation: bool,
    promotion: bool,
) -> bool:
    """Walk the stages once, refusing what no lowering takes where it stands,
    and say whether the order must be cut per share."""
    random = next((_identity(stage).name for stage in ancestry if isinstance(stage, RandomMap)), None)
    indexed, batched, workers, host = read != "streamed", False, False, False
    budgeted = False
    canonical = True
    finite = not any(isinstance(stage, Repeat) for stage in ancestry)
    combined = "mix" if isinstance(pipeline.source, (Mixture, Phases)) else None
    per_share = read == "streamed"
    prefix = list(ancestry)
    for index, stage in enumerate(pipeline.stages):
        name = _name(pipeline)
        if not isinstance(stage, _STAGES):
            raise ValueError(f"{type(stage).__name__} in {name} is not a pipeline stage and has no lowering")
        if isinstance(stage, TokenSplit) and not (indexed and finite):
            raise ValueError(
                f"token split in {name} needs a finite indexed order: read='memory' or 'indexed', "
                "before repeat and any streaming stage"
            )
        if isinstance(stage, MapWithIndex) and combined is not None:
            raise ValueError(
                f"map_with_index in {name} follows {combined}, whose rows have no single source position"
            )
        if isinstance(stage, Interleave) and (
            read != "streamed" or any(not isinstance(step, (*_LOCAL, Artifact)) for step in prefix)
        ):
            raise ValueError(
                f"interleave in {name} needs a shard-manifest source and only record steps before it, "
                "which run inside each shard"
            )
        layout = isinstance(stage, (Pack, Windows, Group, BatchStage, TokenBudget))
        combined = combined or (type(stage).__name__ if layout else None)
        finite = finite and not isinstance(stage, Repeat)
        if _random(stage):
            random = _identity(stage).name
            if validation and not canonical:
                raise ValueError(f"validation random step {random} must use the canonical record index")
        reorders = isinstance(stage, (Shuffle, Repeat, Slice, Group, Limit)) or _drops(stage)
        canonical = canonical and not reorders
        if isinstance(stage, Artifact):
            if random is not None:
                raise ValueError(
                    f"cache prefix in {name} contains random step {random}; cache before stochastic steps"
                )
            for earlier in prefix:
                _identity(earlier, cache=True)
        if isinstance(stage, Shuffle) and stage.window is None and not indexed and not promotion:
            raise ValueError(f"global shuffle in {name} needs an indexed pipeline; give window=")
        streams = isinstance(stage, (Interleave, Rebatch, Workers, BatchStage, TokenBudget)) or (
            _drops(stage) and index >= eager
        )
        indexed = indexed and not (streams or (isinstance(stage, Artifact) and read != "memory"))
        shared = _drops(stage) or isinstance(stage, (Interleave, Artifact, Rebatch, TokenBudget))
        per_share = per_share or shared or (isinstance(stage, Workers) and not batched)
        batched, budgeted = _check_batch(stage, batched=batched, budgeted=budgeted, name=name)
        if isinstance(stage, Workers):
            if workers:
                raise ValueError(f"pipeline {name} contains workers more than once")
            if host:
                raise ValueError(f"workers in {name} must precede map_batch")
            workers = True
        if isinstance(stage, MapBatch):
            if not batched:
                raise ValueError(f"map_batch in {name} must follow batch")
            host = True
        elif host:
            raise ValueError(f"{type(stage).__name__} in {name} follows the host map_batch stage")
        prefix.append(stage)
    return per_share


def _capabilities(pipelines: Sequence[Pipeline], read: Read, provision: bool) -> None:
    sources = [source for pipeline in pipelines for _, source in _leaves(pipeline)]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as readers:
        supported = tuple(readers.map(lambda source: source.supports(read), sources))
    missing = [source for source, available in zip(sources, supported, strict=True) if not available]
    if provision:
        for source in missing:
            if isinstance(source, _Provisionable):
                source.provision()
    refused = [source.describe() for source in missing if not source.supports(read)]
    if refused:
        raise ValueError(
            f"read={read!r} is unavailable for {', '.join(refused)}; "
            "provision the named artifact with dew data provision or provision=True"
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
        binding = {
            "factory": dataclasses.asdict(_identity(stage)),
            "scope": stage.scope,
            "values": {key: metadata[key] for key in stage.needs},
        }
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


def _outputs(
    stage: Stage, position: int | None, value: Record, rng: np.random.Generator | None = None
) -> tuple[Record, ...]:
    """What a record stage makes of one row, the single definition the cache
    fill, memory materialization and Grain lowering all apply: its outputs,
    with a `Rejected` marker in place of each row it drops. A marker passes
    through and a missing row is rejected. `position` is the row's source
    position, as `Pipeline.map_with_index` defines it.
    """
    if isinstance(value, Rejected):
        return (value,)
    if value is None:
        return (Rejected("missing"),)
    if isinstance(stage, MapWithIndex):
        if position is None:
            raise ValueError(f"map_with_index step {_identity(stage).name} read a row without its position")
        return (stage.step(position, record(value, "indexed map input")),)
    if isinstance(stage, Filter):
        return (value if stage.predicate(record(value, "filter input")) else Rejected(stage.reason),)
    if isinstance(stage, FlatMap):
        return tuple(stage.step.flat_map(record(value, "flat_map input"))) or (Rejected("empty_expansion"),)
    if isinstance(stage, RandomMap):
        if rng is None:
            raise ValueError(f"random step {_identity(stage).name} needs a record rng")
        step = stage.step
        return (
            step.random_map(value, rng)
            if isinstance(step, pygrain.RandomMapTransform)
            else step(record(value, "random map input"), rng),
        )
    if isinstance(stage, Map):
        step = stage.step
        return (
            step.map(value) if isinstance(step, pygrain.MapTransform) else step(record(value, "map input")),
        )
    raise ValueError(f"{type(stage).__name__} is not a record stage")


_POSITION = "_dew_position"


def _placed(value: Record, position: int) -> Record:
    return value if value is None or isinstance(value, (bytes, Rejected)) else {**value, _POSITION: position}


def _position(value: Record) -> int | None:
    if value is None or isinstance(value, (bytes, Rejected)) or _POSITION not in value:
        return None
    return int(value[_POSITION])


def _unplaced(value: Record) -> Record:
    if value is None or isinstance(value, (bytes, Rejected)) or _POSITION not in value:
        return value
    return {name: column for name, column in value.items() if name != _POSITION}


@dataclasses.dataclass(frozen=True)
class _Placed:
    """The source position of each row a read numbers from `start`, or of each
    output a cache holds for it; a materialized source states its rows'
    positions (`identifiers`)."""

    start: int = 0
    identifiers: tuple[tuple[int, ...], ...] = ()

    def __call__(self, index: int, value: Record | tuple[Record, ...]) -> Record | tuple[Record, ...]:
        position = self.identifiers[index][0] if self.identifiers else self.start + index
        if isinstance(value, tuple):
            return tuple(_placed(output, position) for output in value)
        return _placed(value, position)


@dataclasses.dataclass(frozen=True)
class _PlacedShards:
    reader: Callable[[Shard], pygrain.IterDataset]
    starts: Mapping[str, int]

    def __call__(self, shard: Shard) -> pygrain.IterDataset:
        return self.reader(shard).map_with_index(_Placed(self.starts[shard.id]))


def _positioned(manifest: ShardManifest) -> ShardManifest:
    rows = [shard.rows for shard in manifest.shards if shard.rows is not None]
    if len(rows) != len(manifest.shards):
        raise ValueError("map_with_index over a streamed read needs every shard to state its rows")
    starts = np.cumsum([0, *rows[:-1]])
    placed = {shard.id: int(start) for shard, start in zip(manifest.shards, starts, strict=True)}
    return dataclasses.replace(manifest, reader=_PlacedShards(manifest.reader, placed))


@dataclasses.dataclass(frozen=True)
class _Rows:
    """Grain's form of a record stage: `_outputs` over each element, keeping
    the source position an element carries out of the step's sight."""

    stage: Map | RandomMap | MapWithIndex | Filter | FlatMap

    def __call__(self, element: Record, rng: np.random.Generator | None = None) -> Record:
        return self.flat_map(element, rng)[0]

    def flat_map(self, element: Record, rng: np.random.Generator | None = None) -> tuple[Record, ...]:
        position = _position(element)
        if position is None:
            return _outputs(self.stage, None, element, rng)
        outputs = _outputs(self.stage, position, _unplaced(element), rng)
        return tuple(_placed(output, position) for output in outputs)


@dataclasses.dataclass(frozen=True)
class _Prefix:
    stages: tuple[Stage, ...]

    @property
    def volatile(self) -> bool:
        return any(
            isinstance(stage, Map) and isinstance(stage.step, Volatile) and stage.step.volatile
            for stage in self.stages
        )

    def __call__(self, index: int, raw: Record) -> tuple[Record, ...]:
        position = _position(raw)
        if position is not None:
            index, raw = position, _unplaced(raw)
        held: tuple[Record, ...] = (raw,)
        for stage in self.stages:
            held = tuple(output for value in held for output in _outputs(stage, index, value))
        return held


_LOCAL = (Map, MapWithIndex, Filter, FlatMap)
"""The deterministic record steps that run before a cache or an interleave,
inside the reader of each input or shard."""


@dataclasses.dataclass(frozen=True)
class _ShardSteps:
    """A shard reader that runs the record steps written before an interleave
    inside each shard, after flattening a cache's outputs, so the interleave
    flattens the shards where it stands. Rejected rows stay marked for the
    accepting reader after it."""

    reader: Callable[[Shard], pygrain.IterDataset]
    stages: tuple[Map | MapWithIndex | Filter | FlatMap, ...]
    cached: bool

    def __call__(self, shard: Shard) -> pygrain.IterDataset:
        rows = self.reader(shard)
        rows = gx.FlatMapIterDataset(rows, _CachedOutputs()) if self.cached else rows
        for stage in self.stages:
            lowered = _Rows(stage)
            rows = gx.FlatMapIterDataset(rows, lowered) if isinstance(stage, FlatMap) else rows.map(lowered)
        return rows


@dataclasses.dataclass(frozen=True)
class _CachedOutputs:
    """What a cache holds for one input, as rows: its outputs, a missing one
    and an empty expansion each rejected, which every read of a cache uses."""

    def flat_map(self, outputs: Sequence[Record]) -> tuple[Record, ...]:
        held = tuple(Rejected("missing") if value is None else value for value in outputs)
        return held if held else (Rejected("empty_expansion"),)


def _fill(pipeline: Pipeline, source: Source, at: int, read: Read) -> tuple[Cache, Manifest, _Prefix]:
    """The cache stage at `at`, the manifest its fill is keyed by and the
    record steps before it that fill it, from which memory, indexed and
    streamed reads all open the cache."""
    artifact, prefix = pipeline.stages[at], pipeline.stages[:at]
    if not isinstance(artifact, Artifact) or any(not isinstance(stage, _LOCAL) for stage in prefix):
        raise ValueError(f"cache prefix in {_name(pipeline)} must hold only deterministic record steps")
    return _filled(artifact), _cache_manifest(source, prefix, read), _Prefix(prefix)


@dataclasses.dataclass(frozen=True)
class _Memory:
    original: Source
    held: tuple[Batch | bytes, ...]
    measured: np.ndarray | None
    derived: Mapping[str, str]
    facts: Mapping[str, object] = dataclasses.field(default_factory=dict)
    identifiers: tuple[tuple[int, ...], ...] = ()
    inputs: int = 0
    prefix: tuple[StepIdentity, ...] = ()

    def describe(self) -> str:
        return self.original.describe()

    def snapshot(self) -> Mapping[str, object]:
        return {
            "materialized": self.original.snapshot(),
            "derived": self.derived,
            "steps": [dataclasses.asdict(step) for step in self.prefix],
        }

    def supports(self, read: Read) -> bool:
        return read in ("memory", "indexed")

    def records(self, read: Read) -> tuple[Batch | bytes, ...]:
        return self.held

    def metadata(self) -> Mapping[str, object]:
        return {**self.original.metadata(), **self.facts, "accepted": len(self.held)}

    def lengths(self) -> np.ndarray | None:
        return self.measured

    def locality(self) -> tuple[tuple[int, ...], bool]:
        if not isinstance(self.original, _Locality) or any(
            len(identifier) != 1 for identifier in self.identifiers
        ):
            raise ValueError("block shuffle needs an unchanged physical shard index")
        sizes, remote = self.original.locality()
        ends = np.cumsum(sizes)
        selected = np.bincount(
            [int(np.searchsorted(ends, identifier[0], side="right")) for identifier in self.identifiers],
            minlength=len(sizes),
        )
        return tuple(int(size) for size in selected), remote

    def manifest(self) -> ShardManifest:
        raise ValueError("materialized records are read by index")


def _materialize(
    pipeline: Pipeline,
    counts: dict[str, CorpusCounts],
    groups: LengthGroups | None = None,
    *,
    training: bool = True,
) -> Pipeline:
    groups = next((stage.groups for stage in pipeline.stages if isinstance(stage, Group)), groups)
    if isinstance(pipeline.source, Phases):
        raise ValueError("materialize each phase before constructing its global order")
    if isinstance(pipeline.source, (Mixture, Phases)):
        return _memory_mixture(pipeline, pipeline.source, counts, groups, training=training)
    source = pipeline.source
    stored = source.records("memory")
    counter = counts.setdefault(_name(pipeline), CorpusCounts(_name(pipeline)))
    if not isinstance(source, _Memory):
        counter.attempted = len(stored)
    lowering = _analyzed(pipeline, "memory")
    rows, consumed = _memory_inputs(pipeline, source, stored, counter, lowering.cache)
    return _memory_stages(
        pipeline, source, stored, rows, consumed, lowering.eager, counter, groups, training=training
    )


def _memory_inputs(
    pipeline: Pipeline, source: Source, stored: Records, counter: CorpusCounts, artifact_at: int | None
) -> tuple[list[tuple[tuple[int, ...], Record]], int]:
    rows: list[tuple[tuple[int, ...], Record]] = []
    consumed = 0
    if artifact_at is not None:
        filled, manifest, prefix = _fill(pipeline, source, artifact_at, "memory")
        cached = filled.mapped(pygrain.MapDataset.source(stored), manifest, source, transform=prefix)
        try:
            for index in range(len(cached)):
                outputs = cached[index]
                if not isinstance(outputs, tuple):
                    raise TypeError("cached input must return its tuple of expansion outputs")
                for child, value in enumerate(_CachedOutputs().flat_map(outputs)):
                    if isinstance(value, Rejected):
                        counter.rejected[value.reason] += 1
                    else:
                        rows.append(((index, child), value))
            cached.finish()
        finally:
            cached.close()
        consumed = artifact_at + 1
    else:
        for index in range(len(stored)):
            value = stored[index]
            if value is None:
                counter.rejected["missing"] += 1
            else:
                identifier = source.identifiers[index] if isinstance(source, _Memory) else (index,)
                rows.append((identifier, value))
    return rows, consumed


def _memory_stages(
    pipeline: Pipeline,
    source: Source,
    stored: Records,
    rows: list[tuple[tuple[int, ...], Record]],
    consumed: int,
    eager: int,
    counter: CorpusCounts,
    groups: LengthGroups | None,
    *,
    training: bool,
) -> Pipeline:
    derived: dict[str, str] = {}
    targets: Mapping[str, int] | None = None
    for stage in pipeline.stages[consumed:eager]:
        if isinstance(stage, TokenSplit):
            split = _TokenSplit(pygrain.MapDataset.source([value for _, value in rows]), stage)
            spans = enumerate(split.spans)
            rows = [((*rows[row][0], first, last), split[index]) for index, (row, first, last) in spans]
        elif isinstance(stage, Windows):
            from dew.data.steps.windows import WindowRecords

            derived[f"accepted_before_windows_{consumed}"] = fingerprint(
                [list(identifier) for identifier, _ in rows]
            )
            windows = WindowRecords(tuple(record(value, "window document") for _, value in rows), stage)
            derived["windows"] = fingerprint([windows.size, stage.seq_len, stage.stride])
            rows, targets = _targeted([((index,), windows[index]) for index in range(len(windows))], counter)
        elif isinstance(stage, Pack):
            from dew.data.tokens import PackedWindows

            derived[f"accepted_before_pack_{consumed}"] = fingerprint(
                [list(identifier) for identifier, _ in rows]
            )
            documents = tuple(_Payload()(record(value, "packing document")) for _, value in rows)
            sizes = np.asarray([len(value[stage.fields[0]]) for value in documents], np.int64)
            if stage.policy == "first-fit":
                packed = PackedWindows(
                    pygrain.MapDataset.source(documents), sizes, stage.seq_len, stage.bins, source.describe()
                )
                derived["packing"] = fingerprint([packed.order.tolist(), packed.starts.tolist()])
            else:
                # Only first-fit has an indexed plan; the other policies are Grain's
                # own packer, run once over the corpus as an indexed read streams it.
                packed = list(_packed(pygrain.MapDataset.source(documents).to_iter_dataset(), stage))
                derived["packing"] = fingerprint(sizes)
            rows, targets = _targeted([((index,), packed[index]) for index in range(len(packed))], counter)
        else:
            expanded: list[tuple[tuple[int, ...], Record]] = []
            for identifier, value in rows:
                for child, output in enumerate(_outputs(stage, identifier[0], value)):
                    if isinstance(output, Rejected):
                        counter.rejected[output.reason] += 1
                    else:
                        expanded.append(
                            ((*identifier, child) if isinstance(stage, FlatMap) else identifier, output)
                        )
            rows = expanded
        consumed += 1
    if targets is not None and targets["valid_targets"] == 0 and training:
        raise ValueError(f"corpus {_name(pipeline)} has zero valid target tokens")
    inputs = source.inputs if isinstance(source, _Memory) else len(stored)
    return _memory_order(pipeline, rows, consumed, source, inputs, counter, derived, groups, targets)


def _targeted(
    rows: list[tuple[tuple[int, ...], Record]], counter: CorpusCounts
) -> tuple[list[tuple[tuple[int, ...], Record]], Mapping[str, int]]:
    """The windows a pack or windows stage made that hold a valid target, and
    their count and targets, taken where the windows are made so later steps
    cannot change them."""
    from dew.data.steps.text import ValidTargets

    kept, total = [], 0
    for identifier, value in rows:
        count = ValidTargets()(record(value, "packed window"))
        if count:
            kept.append((identifier, value))
            total += count
        else:
            counter.rejected["zero_targets"] += 1
    return kept, {"windows": len(kept), "valid_targets": total}


def _token_spans(documents: Iterable[Record], split: TokenSplit) -> list[tuple[int, int, int]]:
    """The documents a token split keeps, in the order it reads them, as
    (document, first, last) token spans. The held side is the first
    `fraction` of all their tokens and the trained side the rest, so the two
    are disjoint and together are the corpus."""
    sizes = []
    for document in documents:
        columns = [np.asarray(record(document, "token split document")[name]) for name in split.fields]
        if any(column.ndim != 1 for column in columns):
            raise ValueError("token split fields must be one-dimensional token arrays")
        sizes.append(len(columns[0]))
    total = sum(sizes)
    boundary = min(int(total * split.fraction), max(0, total - 1))
    start, stop = (0, boundary) if split.held else (boundary, total)
    spans = []
    offset = 0
    for index, size in enumerate(sizes):
        first, last = max(0, start - offset), min(size, stop - offset)
        if first < last:
            spans.append((index, first, last))
        offset += size
    return spans


class _TokenSplit(_WrappingDataset):
    """A token split of a finite indexed order, such as a shuffled corpus:
    the spans `_token_spans` keeps, computed once over the order as it reads."""

    def __init__(self, parent: pygrain.MapDataset, split: TokenSplit):
        super().__init__(parent)
        self._split = split
        self.spans = _token_spans((parent[index] for index in range(len(parent))), split)

    def __len__(self) -> int:
        return len(self.spans)

    def record(self, index: int) -> Batch:
        document, first, last = self.spans[index]
        return {
            key: np.asarray(column)[first:last] if key in self._split.fields else column
            for key, column in record(self._parent[document], "token split document").items()
        }


def _memory_mixture(
    pipeline: Pipeline,
    mixture: Mixture,
    counts: dict[str, CorpusCounts],
    groups: LengthGroups | None,
    *,
    training: bool,
) -> Pipeline:
    children = tuple(_materialize(child, counts, groups, training=training) for child in mixture.pipelines)
    kept = tuple(
        (child, weight)
        for child, weight in zip(children, mixture.weights, strict=True)
        if isinstance(child.source, (Mixture, Phases)) or len(child.source.records("memory"))
    )
    omitted = [_name(child) for child in children if all(child is not accepted for accepted, _ in kept)]
    if omitted:
        _log.warning("left out of the mixture: %s (zero accepted rows)", ", ".join(omitted))
    if not kept:
        if not training:
            return dataclasses.replace(children[0], stages=(*children[0].stages, *pipeline.stages))
        raise ValueError(
            f"no accepted row of {', '.join(_name(child) for child in children)}; raise the row budgets"
        )
    if len(kept) == 1:
        child = kept[0][0]
        return dataclasses.replace(child, stages=(*child.stages, *pipeline.stages))
    children, weights = zip(*kept, strict=True)
    return dataclasses.replace(
        pipeline, source=dataclasses.replace(mixture, pipelines=tuple(children), weights=tuple(weights))
    )


def _memory_order(
    pipeline: Pipeline,
    collected: Sequence[tuple[tuple[int, ...], Record]],
    consumed: int,
    source: Source,
    inputs: int,
    counter: CorpusCounts,
    derived: dict[str, str],
    groups: LengthGroups | None,
    packed: Mapping[str, int] | None,
) -> Pipeline:
    rows = [
        (identifier, value if isinstance(value, bytes) else record(value, "materialized record"))
        for identifier, value in collected
    ]
    remaining = pipeline.stages[consumed:]
    lengths = (
        np.asarray([groups.length(record(value, "grouped record")) for _, value in rows], np.int64)
        if groups is not None
        else source.lengths()
        if consumed == 0
        else None
    )
    fanouts = np.zeros(inputs, np.int64)
    for identifier, _ in rows:
        if identifier[0] < len(fanouts):
            fanouts[identifier[0]] += 1
    derived["fanouts"] = fingerprint(fanouts)
    if lengths is not None:
        derived["lengths"] = fingerprint(lengths)
    counter.accepted = len(rows)
    counter.truncated = sum(
        bool(value.get("_dew_truncated", False)) for _, value in rows if not isinstance(value, bytes)
    )
    facts: Batch = _measured_bounds(
        [record(value, "measured row") for _, value in rows if not isinstance(value, bytes)]
    )
    if packed is not None:
        facts = {**facts, **packed}
        derived["valid_targets"] = fingerprint(facts)
    derived["accepted_ids"] = fingerprint([list(identifier) for identifier, _ in rows])
    return dataclasses.replace(
        pipeline,
        source=_Memory(
            source,
            tuple(value for _, value in rows),
            lengths,
            derived,
            facts,
            tuple(identifier for identifier, _ in rows),
            inputs,
            tuple(_identity(stage) for stage in pipeline.stages[:consumed]),
        ),
        stages=remaining,
    )


def _measured_bounds(rows: Sequence[Batch]) -> Mapping[str, object]:
    bounds: dict[str, JSON] = {}

    def merge(target: dict[str, JSON], given: Mapping[str, object]) -> None:
        for key, value in given.items():
            if isinstance(value, Mapping):
                previous = {
                    name: json_value(datum, name) for name, datum in record(target.get(key, {}), key).items()
                }
                merge(previous, value)
                target[key] = previous
            else:
                target[key] = max(integer(value, key), integer(target.get(key, 0), key))

    for row in rows:
        merge(bounds, record(row.get("_dew_bounds", {}), "record bounds"))
    return bounds


def _as_iter(rows: Rows, loading: Loading) -> pygrain.IterDataset:
    return (
        rows.to_iter_dataset(pygrain.ReadOptions(loading.threads, loading.read_buffer))
        if isinstance(rows, pygrain.MapDataset)
        else rows
    )


def _packed(documents: pygrain.IterDataset, stage: Pack) -> pygrain.IterDataset:
    """`documents` packed by Grain's own packer for the stage's policy."""
    sizes = dict.fromkeys(stage.fields, stage.seq_len)
    if stage.policy == "concat":
        return gx.ConcatThenSplitIterDataset(documents, length_struct=sizes)
    packer = gx.FirstFitPackIterDataset if stage.policy == "first-fit" else gx.BestFitPackIterDataset
    return packer(documents, length_struct=sizes, num_packing_bins=stage.bins, shuffle_bins=False)


def _reordered(stage: Shuffle | Repeat | Slice | Limit, rows: pygrain.MapDataset) -> pygrain.MapDataset:
    """An order stage over an indexed dataset, applied alike to its rows and their lengths."""
    if isinstance(stage, Shuffle):
        if stage.window is None:
            return rows.shuffle(stage.seed)
        return gx.WindowShuffleMapDataset(rows, window_size=stage.window, seed=stage.seed)
    if isinstance(stage, Repeat):
        return rows.repeat(stage.epochs)
    if isinstance(stage, Slice):
        return rows.slice(stage.selection)
    return rows[: min(stage.count, len(rows))]


@dataclasses.dataclass(frozen=True)
class _Build:
    read: Read
    loading: Loading
    seed: int
    batch: int
    budget: SearchBudget
    lowering: _Lowering
    training: bool

    @property
    def global_order(self) -> bool:
        return not self.lowering.per_share

    def mixed(self, pipeline: Pipeline, partition: DataPartition) -> tuple[Rows, pygrain.MapDataset | None]:
        source = pipeline.source
        if not isinstance(source, Mixture):
            raise ValueError("phases need the global phased lowering")
        built = [self.records(child, partition) for child in source.pipelines]
        parents, lengths = zip(*built, strict=True)
        if all(isinstance(parent, pygrain.MapDataset) for parent in parents):
            mapped = [parent for parent in parents if isinstance(parent, pygrain.MapDataset)]
            weights = _weights(source)
            measured = [length for length in lengths if length is not None]
            checked = primed_maps(mapped, [_name(child) for child in source.pipelines])
            return pygrain.MapDataset.mix(checked, weights).seed(self.seed), (
                pygrain.MapDataset.mix(measured, weights) if len(measured) == len(parents) else None
            )
        streams = [
            Accepted(_as_iter(parent, self.loading), _name(child), budget=self.budget)
            for parent, child in zip(parents, source.pipelines, strict=True)
        ]
        if source.tokens is not None:
            return TokenDeficit(
                streams,
                source.weights,
                tokens=source.tokens,
                names=[_name(child) for child in source.pipelines],
                budget=self.budget,
            ), None
        return PrimedMix(streams, source.weights, [_name(child) for child in source.pipelines]).seed(
            self.seed
        ), None

    def indexed(
        self, pipeline: Pipeline, source: Source, partition: DataPartition, last: int
    ) -> tuple[Rows, pygrain.MapDataset | None, int]:
        """The root of an indexed read, cut to its share by index (`shared`),
        and through its cache when it has one, with the stages it consumed."""
        stored = pygrain.MapDataset.source(source.records(self.read)).seed(self.seed)
        cache = _at(pipeline.stages, Artifact)
        if cache is None:
            identifiers = source.identifiers if isinstance(source, _Memory) else ()
            rows = stored.map_with_index(_Placed(identifiers=identifiers)) if last >= 0 else stored
            lengths = source.lengths()
            measured = None if lengths is None else pygrain.MapDataset.source(lengths)
            if measured is not None:
                measured = self.shared(measured, pipeline, partition)
            return self.shared(rows, pipeline, partition), measured, 0
        filled, manifest, prefix = _fill(pipeline, source, cache, self.read)
        cached = filled.mapped(stored, manifest, source, transform=prefix)
        view = self.shared(cached.map_with_index(_Placed()) if last > cache else cached, pipeline, partition)
        rows = cached.iterated(view, pygrain.ReadOptions(self.loading.threads, self.loading.read_buffer))
        flat = gx.FlatMapIterDataset(rows, _CachedOutputs())
        return Accepted(flat, _name(pipeline), budget=self.budget), None, cache + 1

    def sharded(
        self, pipeline: Pipeline, source: Source, partition: DataPartition, last: int
    ) -> tuple[pygrain.IterDataset, int]:
        """The root of a streamed read and the stages it consumed: the cache
        prefix runs inside the cache's shard readers, and the record steps
        before an interleave run inside each shard, which the interleave
        flattens where it is written."""
        stages = pipeline.stages
        cache, interleave = _at(stages, Artifact), _at(stages, Interleave)
        start = 0 if cache is None else cache + 1
        shards = source.manifest()
        if cache is not None:
            filled, manifest, prefix = _fill(pipeline, source, cache, self.read)
            shards = _positioned(shards) if 0 <= last < cache else shards
            reader = filled.streamed(shards.reader, manifest, shards, transform=prefix)
            shards = dataclasses.replace(shards, reader=reader)
        shards = _positioned(shards) if last >= start else shards
        written = stages[start : start if interleave is None else interleave]
        local = tuple(stage for stage in written if isinstance(stage, _LOCAL))
        shards = dataclasses.replace(shards, reader=_ShardSteps(shards.reader, local, cache is not None))
        rows = self.streamed(shards, pipeline, partition)
        if cache is not None or any(_drops(stage) for stage in local):
            rows = Accepted(rows, _name(pipeline), budget=self.budget)
        return rows, start if interleave is None else interleave + 1

    def shared(
        self, rows: pygrain.MapDataset, pipeline: Pipeline, partition: DataPartition
    ) -> pygrain.MapDataset:
        """The share of an indexed root that a per-share training read takes,
        repeated unless the pipeline repeats it. Both are views by index, so a
        cache under them still keys each row by its place in the source."""
        if self.global_order or not self.training:
            return rows
        rows = rows[partition.index :: partition.count]
        return rows if any(isinstance(stage, Repeat) for stage in pipeline.stages) else rows.repeat(None)

    def streamed(
        self, shards: ShardManifest, pipeline: Pipeline, partition: DataPartition
    ) -> pygrain.IterDataset:
        from dew.data.shards import streamed

        cycle = next((stage.cycle_length for stage in pipeline.stages if isinstance(stage, Interleave)), 1)
        return streamed(
            shards,
            partition=partition,
            seed=self.seed if self.training else None,
            cycle_length=cycle,
            loading=self.loading,
            epochs=_epochs(pipeline) if self.training else 1,
        ).seed(self.seed)

    def records(self, pipeline: Pipeline, partition: DataPartition) -> tuple[Rows, pygrain.MapDataset | None]:
        """The rows before the batch, with their lengths where known. A global
        order comes back indexed, whole, for `batches` to cut into shares; a
        per-share order comes back as this share's stream of accepted rows."""
        source = pipeline.source
        placed = [index for index, stage in enumerate(pipeline.stages) if isinstance(stage, MapWithIndex)]
        last = max(placed, default=-1)
        lengths: pygrain.MapDataset | None = None
        if isinstance(source, (Mixture, Phases)):
            (rows, lengths), consumed = self.mixed(pipeline, partition), 0
        elif self.read == "streamed":
            rows, consumed = self.sharded(pipeline, source, partition, last)
        else:
            rows, lengths, consumed = self.indexed(pipeline, source, partition, last)
        for index, stage in enumerate(pipeline.stages[consumed:], start=consumed):
            if isinstance(stage, (BatchStage, TokenBudget, MapBatch, Rebatch)):
                break
            if self.read == "streamed" and isinstance(stage, Repeat) and _root_repeat(pipeline):
                continue
            if isinstance(stage, Prefetch) and self.global_order:
                continue
            rows, lengths = self.stage(rows, lengths, stage, pipeline, partition)
            if index == last:
                rows = rows.map(_unplaced)
        if not self.global_order:
            rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
        return rows, lengths

    def stage(
        self,
        rows: Rows,
        lengths: pygrain.MapDataset | None,
        stage: Stage,
        pipeline: Pipeline,
        partition: DataPartition,
    ) -> tuple[Rows, pygrain.MapDataset | None]:
        if isinstance(stage, (Map, RandomMap, MapWithIndex, Filter, FlatMap)):
            return self.record_stage(rows, lengths, stage, pipeline)
        if isinstance(stage, Shuffle) and stage.block_shards is not None:
            return self.block_shuffle(rows, lengths, stage, pipeline, partition)
        if isinstance(stage, (Shuffle, Repeat, Slice, Limit)) and isinstance(rows, pygrain.MapDataset):
            return _reordered(stage, rows), None if lengths is None else _reordered(stage, lengths)
        if isinstance(stage, Shuffle) and stage.window is not None:
            rows = gx.WindowShuffleIterDataset(
                _as_iter(rows, self.loading), window_size=stage.window, seed=stage.seed
            )
        elif isinstance(stage, (Shuffle, Slice, ShardByHost)):
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError(f"{type(stage).__name__} in {_name(pipeline)} needs an indexed pipeline")
        elif isinstance(stage, Repeat):
            rows = gx.RepeatIterDataset(_as_iter(rows, self.loading), stage.epochs)
        elif isinstance(stage, Limit):
            rows = gx.LimitIterDataset(_as_iter(rows, self.loading), stage.count)
        elif isinstance(stage, Prefetch):
            rows = gx.ThreadPrefetchIterDataset(
                _as_iter(rows, self.loading), prefetch_buffer_size=stage.buffer
            )
        elif isinstance(stage, Workers):
            rows = _as_iter(rows, self.loading).mp_prefetch(
                pygrain.MultiprocessingOptions(
                    num_workers=stage.count, per_worker_buffer_size=stage.buffer
                )
            )
            lengths = None
        elif isinstance(stage, Rebatch):
            rows = gx.RebatchIterDataset(
                _as_iter(rows, self.loading), partition.rows(stage.size), drop_remainder=self.training
            )
        else:
            return self.layout_stage(rows, lengths, stage, pipeline, partition)
        return rows, lengths

    def layout_stage(
        self,
        rows: Rows,
        lengths: pygrain.MapDataset | None,
        stage: Stage,
        pipeline: Pipeline,
        partition: DataPartition,
    ) -> tuple[Rows, pygrain.MapDataset | None]:
        if isinstance(stage, TokenSplit):
            if not isinstance(rows, pygrain.MapDataset):
                raise ValueError(f"token split in {_name(pipeline)} needs a finite indexed order")
            rows, lengths = _TokenSplit(rows, stage), None
        elif isinstance(stage, Pack):
            rows = _packed(_as_iter(rows, self.loading).map(_Payload()), stage)
            lengths = None
        elif isinstance(stage, Windows):
            from dew.data.steps.windows import CausalWindows

            rows = CausalWindows(_as_iter(rows, self.loading), stage)
            lengths = None
        elif isinstance(stage, Group):
            if isinstance(rows, pygrain.MapDataset):
                if lengths is None:
                    raise ValueError(
                        f"length grouping of {_name(pipeline)} needs producer lengths or read='memory'"
                    )
                rows = _Grouped(rows, stage.groups, self.batch, self.seed, length_at=_LengthAt(lengths))
            else:
                rows = _GroupedRows(rows, stage.groups, partition.rows(self.batch), self.seed)
            lengths = None
        elif isinstance(stage, Artifact):
            if not isinstance(rows, pygrain.MapDataset) or isinstance(pipeline.source, (Mixture, Phases)):
                raise ValueError("a cache belongs on each corpus's indexed rows, before mix and interleave")
            prefix = pipeline.stages[: pipeline.stages.index(stage)]
            manifest = _cache_manifest(pipeline.source, prefix, self.read)
            rows = _filled(stage).mapped(rows, manifest, pipeline.source)
        elif not isinstance(stage, Interleave):
            raise ValueError(f"{type(stage).__name__} in {_name(pipeline)} has no lowering here")
        return rows, lengths

    def block_shuffle(
        self,
        rows: Rows,
        lengths: pygrain.MapDataset | None,
        stage: Shuffle,
        pipeline: Pipeline,
        partition: DataPartition,
    ) -> tuple[Rows, pygrain.MapDataset | None]:
        from dew.data.ordering import BlockShuffle, PermutedLengths

        if not isinstance(rows, pygrain.MapDataset) or not isinstance(pipeline.source, _Locality):
            raise ValueError(
                f"block shuffle of {_name(pipeline)} needs an indexed source with physical shard counts"
            )
        if stage.block_shards is None:
            raise ValueError("block shuffle needs a block_shards size")
        sizes, _ = pipeline.source.locality()
        if not self.global_order and self.training:
            starts = np.cumsum((0, *sizes[:-1]))
            sizes = tuple(
                max(
                    0,
                    math.ceil((int(start) + size - partition.index) / partition.count)
                    - math.ceil((int(start) - partition.index) / partition.count),
                )
                for start, size in zip(starts, sizes, strict=True)
            )
        blocked = BlockShuffle(rows, sizes, stage.block_shards, stage.seed)
        lengths = None if lengths is None else PermutedLengths(lengths, blocked.index)
        return blocked, lengths

    def record_stage(
        self,
        rows: Rows,
        lengths: pygrain.MapDataset | None,
        stage: Map | RandomMap | MapWithIndex | Filter | FlatMap,
        pipeline: Pipeline,
    ) -> tuple[Rows, pygrain.MapDataset | None]:
        lowered = _Rows(stage)
        if isinstance(stage, FlatMap):
            rows = gx.FlatMapIterDataset(_as_iter(rows, self.loading), lowered)
        else:
            rows = rows.random_map(lowered) if isinstance(stage, RandomMap) else rows.map(lowered)
        if isinstance(stage, (Filter, FlatMap)) or stage.rejects:
            rows = Accepted(_as_iter(rows, self.loading), _name(pipeline), budget=self.budget)
            lengths = None
        return rows, lengths

    def batches(
        self, pipeline: Pipeline, partition: DataPartition, offset: int = 0, *, fill: bool = True
    ) -> pygrain.DatasetIterator[Batch]:
        budget = next((stage for stage in pipeline.stages if isinstance(stage, TokenBudget)), None)
        if budget is not None:
            return self.token_batches(pipeline, partition, budget)
        canonical = not self.training and not self.global_order
        rows, _ = self.records(pipeline, DataPartition() if canonical else partition)
        batch_stage = next(stage for stage in pipeline.stages if isinstance(stage, BatchStage))
        shared_size = partition.rows(batch_stage.size)
        size = batch_stage.size if canonical else shared_size
        workers = _workers(pipeline, self.loading)
        empty = False
        if isinstance(rows, pygrain.MapDataset):
            if self.training and not any(isinstance(stage, (Repeat, Limit)) for stage in _steps(pipeline)):
                rows = rows.repeat(None)
            elif self.training and not self.lowering.endless:
                rows = rows[: len(rows) // batch_stage.size * batch_stage.size]
            whole = rows
            rows = rows[offset + partition.index :: partition.count]
            empty = not self.training and not len(rows) and bool(len(whole))
            if empty:
                rows = whole[:1]
            if workers:
                rows = _WorkerBatches(rows, size, workers, remainder=not self.training)
        stream = _as_iter(rows, self.loading)
        if self.global_order:
            for stage in pipeline.stages[: pipeline.stages.index(batch_stage)]:
                if isinstance(stage, Prefetch):
                    advanced, _ = self.stage(stream, None, stage, pipeline, partition)
                    stream = _as_iter(advanced, self.loading)
        stream = (
            stream
            .map(_Payload())
            .batch(
                size,
                drop_remainder=batch_stage.drop_remainder if self.training else False,
                batch_fn=batch_stage.batch_fn,
            )
        )
        if not self.training and fill:
            stream = stream.map(
                _FillShare(partition, batch_stage.size) if canonical else FilledBatch(size, real=not empty)
            )
        return iter(self.after_batch(stream, pipeline, partition, batch_stage))

    def after_batch(
        self,
        stream: pygrain.IterDataset,
        pipeline: Pipeline,
        partition: DataPartition,
        batch_stage: BatchStage | TokenBudget,
    ) -> pygrain.IterDataset:
        """The stages after the batch, each as `stage` lowers it, with the
        loading's own workers last when the pipeline names none. Host steps
        run in the reader that wraps this stream."""
        stages = pipeline.stages[pipeline.stages.index(batch_stage) + 1 :]
        if self.loading.workers and _at(pipeline.stages, Workers) is None:
            stages = (*stages, Workers(self.loading.worker_count, self.loading.worker_buffer))
        for stage in stages:
            if not isinstance(stage, MapBatch):
                advanced, _ = self.stage(stream, None, stage, pipeline, partition)
                stream = _as_iter(advanced, self.loading)
        return stream

    def token_batches(
        self, pipeline: Pipeline, partition: DataPartition, budget: TokenBudget
    ) -> pygrain.DatasetIterator[Batch]:
        from dew.data.batching import TokenBatches

        if partition.count > 1:
            raise ValueError(
                "token-budget batching requires one share until a synchronized shape schedule is supplied"
            )
        rows, _ = self.records(pipeline, partition)
        stream = TokenBatches(_as_iter(rows, self.loading), budget, partition, self.seed)
        return iter(self.after_batch(stream, pipeline, partition, budget))


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


def _derived(pipeline: Pipeline, seed: int, read: Read) -> Mapping[str, str]:
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
    read: Read | None = "indexed",
    provision: bool = False,
    loading: Loading = _DEFAULT_LOADING,
    seed: int = 0,
    rejections: Literal["drop", "mask"] = "drop",
    budget: SearchBudget = _DEFAULT_BUDGET,
    resume: Literal["exact", "non-exact"] = "exact",
) -> Dataset:
    """Check every leaf before opening readers, then build both branches.
    With no `read`, the sources choose it (`_preferred_read`)."""
    read = read or _preferred_read((train,) if validation is None else (train, validation))
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
        (train,) if validation is None else (train, validation),
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
    if not build.global_order and any(isinstance(stage, TokenSplit) for stage in _steps(train)):
        raise ValueError(f"token split in {_name(train)} needs a global record order, not one per share")
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
) -> Dataset:
    read, seed, loading, batch, global_order = (
        build.read,
        build.seed,
        build.loading,
        build.batch,
        build.global_order,
    )
    manifest = Manifest.of(
        {"sources": [source.snapshot() for _, source in _leaves(original)]},
        (
            *(_identity(stage) for stage in _steps(original) if not isinstance(stage, (Workers, Prefetch))),
            *(StepIdentity.of(step) for step in host),
        ),
    )
    plan = Plan(
        "global" if global_order else "per-share",
        manifest,
        {"read": read, "seed": seed, **build.lowering.order},
        _derived(train, seed, read),
        cache=_cache_references(original, read),
        volatile=any(
            isinstance(stage.step, Volatile) and stage.step.volatile
            for stage in _steps(train)
            if isinstance(stage, Map)
        ),
    )
    if loading.workers == "auto":
        tuning, _ = build.records(train, DataPartition())
        loading = loading.resolved(_as_iter(tuning, dataclasses.replace(loading, workers=0)))
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
            build=dataclasses.replace(build, lowering=_analyzed(validation, read), training=False),
            host=tuple(stage.step for stage in validation.stages if isinstance(stage, MapBatch)) + host,
        )
    )
    records = _epoch_records(train, build)
    if records == 0:
        raise ValueError(f"corpus {_name(train)} has no accepted rows")
    if records is not None and records < batch:
        raise ValueError(f"corpus {_name(train)} has {records} records, fewer than one batch of {batch}")
    held_out = sum(integer(source.metadata().get("held_out", 0), "held_out") for _, source in _leaves(train))
    return Dataset(
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
) -> Dataset:
    source = train.source
    if not isinstance(source, Phases):
        raise TypeError("phased compilation needs a Phases source")
    _tokenizer_contract((train,) if validation is None else (train, validation))
    consumed: dict[str, int] = {}
    built: list[tuple[Dataset, int | None]] = []
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
    for step in host:
        if partition.count > 1 and isinstance(step, SingleReader) and step.single_reader:
            raise ValueError(f"host stage {type(step).__name__} requires one reader for its batch shape")
    opened: Iterator[Batch]
    if build.global_order:
        opened = GlobalStream(
            functools.partial(build.batches, pipeline, partition),
            build.batch,
            json.dumps(plan.order, sort_keys=True),
            build.loading,
            plan=plan,
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
            endless=build.lowering.endless,
        )
        statistics.observe(opened)
    return _Host(opened, host) if host else opened


def _validation_reader(
    partition: DataPartition, *, pipeline: Pipeline, build: _Build, host: tuple[HostStep, ...]
) -> Iterator[Batch]:
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


def _epoch_records(train: Pipeline, build: _Build) -> int | None:
    """The records in one epoch of a global order: each corpus's indexed order
    up to its first repeat, as the lowering builds it, so a slice, a limit and
    a token split count exactly; a mixture reads until its first corpus would
    come round."""
    if not build.global_order:
        return None

    def counted(leaf: Pipeline) -> int | None:
        at = _until(leaf.stages, (Repeat, Group))
        rows, _ = build.records(dataclasses.replace(leaf, stages=leaf.stages[:at]), DataPartition())
        return len(rows) if isinstance(rows, pygrain.MapDataset) else None

    source = train.source
    if not isinstance(source, Mixture):
        return counted(train)
    counts = [counted(child) for child in source.pipelines if not isinstance(child.source, (Mixture, Phases))]
    sizes = [count for count in counts if count is not None]
    if len(sizes) != len(counts):
        return None
    weights = _weights(source)
    return max(math.ceil(size * sum(weights) / weight) for size, weight in zip(sizes, weights, strict=True))


def _weights(mixture: Mixture) -> Sequence[float]:
    """A mixture's weights, as shares of its corpora's stated valid target
    tokens when it mixes by tokens."""
    if mixture.tokens is None:
        return mixture.weights
    leaves = [child.source for child in mixture.pipelines if not isinstance(child.source, (Mixture, Phases))]
    facts = [source.metadata() for source in leaves]
    return token_weights(
        [_name(child) for child in mixture.pipelines],
        mixture.weights,
        [integer(stated["windows"], "windows") for stated in facts],
        [integer(stated["valid_targets"], "valid_targets") for stated in facts],
    )


def _workers(pipeline: Pipeline, loading: Loading) -> int:
    return next(
        (stage.count for stage in pipeline.stages if isinstance(stage, Workers)), loading.worker_count
    )


def _root_repeat(pipeline: Pipeline) -> bool:
    for stage in pipeline.stages:
        if isinstance(stage, RandomMap):
            return False
        if isinstance(stage, Repeat):
            return True
    return True


def _epochs(pipeline: Pipeline) -> int | None:
    return (
        next((stage.epochs for stage in pipeline.stages if isinstance(stage, Repeat)), None)
        if _root_repeat(pipeline)
        else 1
    )


@dataclasses.dataclass(frozen=True)
class _Payload:
    def __call__(self, row: Batch) -> Batch:
        return {name: value for name, value in row.items() if not name.startswith("_dew_")}


@dataclasses.dataclass(frozen=True)
class _LengthAt:
    lengths: pygrain.MapDataset[int]

    def __call__(self, index: int) -> int:
        measured = self.lengths[index]
        if measured is None:
            raise ValueError(f"length metadata is missing index {index}")
        return int(measured)


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
        elif isinstance(step, pygrain.RandomMapTransform):
            pipeline = pipeline.random_map(step)
        else:
            pipeline = pipeline.map(step)
    return dataclasses.replace(pipeline, stages=(*pipeline.stages, *suffix))


def _objective_order(pipeline: Pipeline, objective: ObjectiveData) -> Pipeline:
    def shared(stage: Stage) -> bool:
        return isinstance(stage, RandomMap) and (
            stage.configured.get("scope") == "run" or any(stage.step is step for step in objective.train)
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
    if ordering and read == "memory" and _at(branch.stages, RandomMap) is not None:
        without = tuple(stage for stage in branch.stages if not isinstance(stage, (Shuffle, Repeat)))
        first_random = _until(without, RandomMap)
        stages = (*without[:first_random], *ordering, *without[first_random:])
        branch = dataclasses.replace(branch, stages=stages)
    if (
        train
        and objective.group is not None
        and not any(isinstance(stage, TokenBudget) for stage in branch.stages)
    ):
        at = _until(branch.stages, (RandomMap, BatchStage, TokenBudget))
        branch = dataclasses.replace(
            branch, stages=(*branch.stages[:at], Group(objective.group), *branch.stages[at:])
        )
    return branch.map_batch(*(objective.train_batch if train else objective.evaluate_batch))


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
    build = _Build(read, loading, 0, 1, budget, _analyzed(pipeline, read), training=False)
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
    read = read or _preferred_read((pipeline,))
    if not any(isinstance(stage, BatchStage) for stage in pipeline.stages):
        pipeline = pipeline.batch(batch, drop_remainder=False)
    (pipeline,) = _prepared((pipeline,), read, provision=provision, objective=objective, counts=({},))
    if read == "memory":
        pipeline = _materialize(pipeline, {}, training=False)
    build = _Build(read, loading, seed, batch, budget, _analyzed(pipeline, read), training=False)
    if loading.workers == "auto":
        rows, _ = build.records(pipeline, DataPartition())
        build = dataclasses.replace(build, loading=loading.resolved(_as_iter(rows, loading)))
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
    (pipeline,) = _prepared((pipeline,), read, provision=provision)
    planned = _at(pipeline.stages, Pack)
    builder = None
    if planned is not None:
        from dew.data.packing import PackingPlan

        packing = pipeline.stages[planned]
        if not isinstance(packing, Pack) or planned != len(pipeline.stages) - 1:
            raise ValueError("publication accepts one terminal packing stage after the cached prefix")
        builder = PackingPlan(packing)
        pipeline = dataclasses.replace(pipeline, stages=pipeline.stages[:planned])
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
