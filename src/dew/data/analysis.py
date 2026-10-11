"""What a pipeline's stages mean, analyzed once before anything reads them.

`_analyzed` walks a pipeline once and returns `_Lowering`: whether its order
is cut into shares at the read or kept global and cut just before batching,
the cache a memory read fills and the eager stages it runs over the whole
source, the canonical order a resume checks, and whether the order ever
ends. The same walk (`_checked`) refuses, where it stands, any stage no
lowering takes. The memory executor (`dew.data.materialize`), the Grain
lowering (`dew.data.lowering`) and the plan `compile` saves read this
analysis instead of deriving it again.

The stage facts they share are here too: a pipeline's corpora and steps, a
step's identity and the manifest a cache's fill is keyed by, which stages
draw from the record rng, which drop or make rows, and which run inside a
cache's or a shard's reader. `_capabilities` and `_preferred_read` ask a
run's sources which reads they support and which they prefer.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, get_args, runtime_checkable

from dew.data.pipeline import (
    Artifact,
    BatchStage,
    Configured,
    Filter,
    FlatMap,
    Group,
    Interleave,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    Mixture,
    Pack,
    Phases,
    Pipeline,
    RandomMap,
    Rebatch,
    Repeat,
    Shuffle,
    Slice,
    Stage,
    TokenBudget,
    TokenSplit,
    Windows,
    Workers,
)
from dew.data.plan import Manifest, StepIdentity
from dew.data.sources.base import READS, Read, Source
from dew.data.sources.memory import Memory

_STAGES = get_args(Stage.__value__)


@runtime_checkable
class _Provisionable(Protocol):
    def provision(self) -> Source: ...


@runtime_checkable
class _Wrapping(Protocol):
    source: Source


def _preferred_read(pipelines: Sequence[Pipeline]) -> Read:
    """The read a run's sources choose when it states none, which compile,
    validation and load all take from here.

    Each source prefers memory for records held in memory (`Memory`, through
    any source that wraps one), indexed for a store that reads by index, and
    streamed otherwise, and a mixture takes the least eager of its corpora's
    preferences. The run reads by the first read, from that preference
    toward the more eager ones, that every source takes or can be provisioned
    for. A CSV prefers a stream, a random holdout of it cannot be streamed,
    and the CSV has no index to read it by, so the run reads it into memory.
    When no read is left, the preference stands for `_capabilities` to refuse
    with its remedy.
    """
    chosen: list[Read] = []
    for source in (source for pipeline in pipelines for _, source in _leaves(pipeline)):
        held = source
        while isinstance(held, _Wrapping):
            held = held.source
        if isinstance(held, Memory):
            chosen.append("memory")
        else:
            chosen.append("indexed" if source.supports("indexed") else "streamed")
    preferred = READS.index(max(chosen, key=READS.index))
    return next(
        (
            read
            for read in reversed(READS[: preferred + 1])
            if all(isinstance(source, _Provisionable) for source in _unsupported(pipelines, read))
        ),
        READS[preferred],
    )


def _unsupported(pipelines: Sequence[Pipeline], read: Read) -> list[Source]:
    """The sources under `pipelines` that cannot take `read`, asked together."""
    sources = [source for pipeline in pipelines for _, source in _leaves(pipeline)]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as readers:
        supported = tuple(readers.map(lambda source: source.supports(read), sources))
    return [source for source, available in zip(sources, supported, strict=True) if not available]


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


_LOCAL = (Map, MapWithIndex, Filter, FlatMap)
"""The deterministic record steps that run before a cache or an interleave,
inside the reader of each input or shard."""


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
    missing = _unsupported(pipelines, read)
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
