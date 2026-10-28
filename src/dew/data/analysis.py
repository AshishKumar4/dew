"""What a pipeline's stages mean, analyzed once before anything reads them.

`_analyzed` walks a pipeline once and returns `_Lowering`: whether its order
is cut into shares at the read or kept global and cut just before batching,
where its cache and interleave stand, the eager stages a memory read runs
over the whole source, where its order repeats, the canonical order a
resume checks, and whether the order ever ends, with the same for each
corpus it mixes. The same walk (`_checked`) refuses, where it stands, any
stage no lowering takes. The memory executor (`dew.data.materialize`), the
Grain lowering (`dew.data.lowering`) and the plan `compile` saves read this
analysis instead of deriving it again.

The stage facts they share are here too: a pipeline's corpora and steps, a
step's identity and the manifest a plan or a cache's fill records
(`_manifest`), which stages draw from the record rng, which drop or make
rows, which a cache can hold (`_cacheable`), and which run inside a cache's or
a shard's reader. `_capabilities` and `_preferred_read` ask a run's sources
which reads they support and which they prefer.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
import itertools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, Protocol, get_args, runtime_checkable

import grain.python as pygrain

from dew._records import json_value, record
from dew.data.dataset import Records, SourceSlice
from dew.data.pipeline import (
    TOKEN_FIELDS,
    Artifact,
    Batched,
    BatchStage,
    Bound,
    Configured,
    Expansion,
    External,
    Filter,
    FlatMap,
    Interleave,
    Limit,
    Map,
    MapBatch,
    MapWithIndex,
    Measure,
    Mixture,
    Native,
    Pack,
    Phases,
    Pipeline,
    Prefetch,
    RandomMap,
    Repeat,
    Shuffle,
    Slice,
    Stage,
    TokenBudget,
    TokenSplit,
    Volatile,
    Windows,
    Workers,
    _declares_rejects,
    chains,
    children,
    record_stage,
    runs,
    with_children,
)
from dew.data.plan import Manifest, StepIdentity, _changed, _semantic, identified
from dew.data.sources.base import READS, SAME_ROWS, Read, Source, View, _consumed, beneath, materialized
from dew.data.sources.concatenate import Concatenate
from dew.data.sources.memory import Memory

if TYPE_CHECKING:
    from dew.data.generation import Generation

_STAGES = get_args(Stage.__value__)


@runtime_checkable
class _Provisionable(Protocol):
    def can_provision(self, read: Read) -> bool: ...
    def provision(self) -> Source: ...


@dataclasses.dataclass(frozen=True)
class _ConcatenationPreparers:
    """A concatenation's stores that cannot take a read, each prepared by its own preparer
    (`_provisioner`), so the concatenation takes it once they have."""

    concatenation: Concatenate
    preparers: tuple[_Provisionable, ...]

    def can_provision(self, read: Read) -> bool:
        return True

    def provision(self) -> Source:
        for preparer in self.preparers:
            preparer.provision()
        return self.concatenation


def _provisioner(source: Source, read: Read) -> _Provisionable | None:
    """The source that can prepare `read`, through the views whose rows are their source's
    (`SAME_ROWS`) over it, unless one takes no such read; for a concatenation, each of its stores
    that cannot take `read` yet, all of which must prepare it."""
    preparing = source
    for preparing in beneath(source, SAME_ROWS):
        if isinstance(preparing, _Provisionable):
            break
        if isinstance(preparing, View) and read not in preparing.reads:
            return None
    source = preparing
    if isinstance(source, Concatenate):
        preparers = [_provisioner(member, read) for member in source.sources if not member.supports(read)]
        if not preparers or any(preparer is None for preparer in preparers):
            return None
        ready = tuple(preparer for preparer in preparers if preparer is not None)
        return _ConcatenationPreparers(source, ready)
    return source if isinstance(source, _Provisionable) and source.can_provision(read) else None


def _store(source: Source) -> Source:
    """The store under the views whose rows are its own, or some of them (`SAME_ROWS`): holdouts,
    a compile's pinned read, a decision table."""
    *_, store = beneath(source, SAME_ROWS)
    return store


def _in_memory(source: Source) -> bool:
    """Whether a source holds its records in memory (`Memory`), itself or
    through the sources that wrap it."""
    return isinstance(_store(source), Memory)


@dataclasses.dataclass(frozen=True)
class Measuring:
    """A fact a configured step measures (`Configured.measures`) and the stages before it, which
    make the records it reads from the store's."""

    measure: Measure
    before: tuple[Stage, ...]


def _stated(source: Source, read: Read, measuring: Sequence[Measuring] = (),
            whole: Callable[[Source], Records] | None = None) -> tuple[Records | None, Mapping[str, object]]:
    """The records a memory read of `source` holds, then its metadata, with each fact `measuring`
    measures in place of any the store states, in a memory read: over the whole store's records,
    below any holdout (a generation's part's over the generation's rows for every input,
    `SelectedGeneration.whole`), so each part of it binds alike, as the stages before the measuring
    step make them (`dew.data.lowering.prepared_rows`). `whole` reads the store once for every part
    of it."""
    from dew.data.holdout import SelectedGeneration
    from dew.data.lowering import prepared_rows

    held = source.records(read) if read == "memory" else None
    metadata = source.metadata()
    if held is None or not measuring:
        return held, metadata
    store = _store(source)
    if isinstance(store, SelectedGeneration):
        store = store.whole
    # A records holdout reads its part of the whole store's records, which it keeps.
    rows = (held if store is source else held.source if isinstance(held, SourceSlice)
            else whole(store) if whole is not None else store.records(read))
    return held, {**metadata, **{each.measure.fact: each.measure(prepared_rows(rows, each.before))
                                 for each in measuring}}


def _measures(stages: Sequence[Stage]) -> tuple[Measuring, ...]:
    """The facts the configured steps among `stages` measure (`Configured.measures`), each once,
    with the stages before each."""
    found: dict[str, Measuring] = {}
    for index, stage in enumerate(stages):
        if isinstance(stage, Configured) and stage.measures is not None:
            found.setdefault(stage.measures.fact, Measuring(stage.measures, tuple(stages[:index])))
    return tuple(found.values())


def _prefix_manifest(source: Source, prefix: Sequence[Stage], read: Read) -> Manifest:
    """The manifest of what a cache after `prefix` over `source` holds, as a
    publication records it (`dew.data.compile.publish`): the source as `read`
    consumes it, and the configured steps bound to the facts it states."""
    consumed = _consumed(source, read)
    pipeline = Pipeline(consumed, tuple(prefix))
    if any(isinstance(stage, Configured) for stage in prefix):
        stated = _stated(consumed, read, _measures(prefix))[1]
        pipeline = _configured(pipeline, stated, read, stated=stated)
    return _cache_manifest(consumed, pipeline.stages, read)


def _in_place(generation: Generation, source: Source | Mixture | Phases,
              prefix: Sequence[Stage]) -> tuple[Source | None, tuple[str, ...]]:
    """The rows of `generation` in place of `prefix` over `source`, or None
    and the fields of its manifest that differ: in place of the source it
    was published from, `source` itself or one its holdout selections select
    from, read as it was filled; under selections above that source, the
    rows of the inputs they keep, each picked from that source as it is
    uncached (`SelectedGeneration`), where no step before the cache reads its
    record's index (which the selections number otherwise than that source)
    and a records holdout numbers that source's inputs as the fill did (by
    index, or by a stream that reads them in index order). Its rows are its
    source, steps, tokenizers and fill (`Manifest.semantic`); the code and
    machine that filled it (`Manifest.environment`) do not change them. A
    step bound to the whole run's facts (scope='run') binds to another run's
    facts in a generation than in this run, and a step no cache can key
    (`_keyed`) is in no generation's prefix, so no generation stands for
    either."""
    from dew.data.holdout import RecordSlice, SelectedGeneration, selecting

    dependency = record(generation.manifest["dependency"], "generation dependency")
    fill = record(dependency.get("fill", {}), "generation fill")
    if fill.get("read") not in READS:
        return None, ("fill",)
    read = READS[READS.index(fill["read"])]
    published = dependency.get("source")
    found = None if isinstance(source, (Mixture, Phases)) else next(
        ((above, candidate) for above, candidate in selecting(source)
         if json_value(_consumed(candidate, read).fingerprint(), "source") == published), None)
    if found is None:
        return None, ("source",)
    selections, stored = found
    unkeyed = [_identity(stage).name for stage in prefix if not _keyed(stage)
               or (isinstance(stage, Configured) and stage.scope == "run")]
    if unkeyed:
        return None, (f"steps ({', '.join(unkeyed)} bind to one run's facts or have no "
                      "identity='name@version', which no generation holds)",)
    changed = _changed(_semantic(dependency), _prefix_manifest(stored, prefix, read).semantic())
    if changed:
        return None, changed
    if not selections:
        return generation.source(), ()
    if any(isinstance(stage, MapWithIndex) for stage in prefix):
        raise ValueError(
            f"a step before generation {generation.url} reads its record's index, which the holdout written "
            "before the cache numbers otherwise than the source the generation was published from; cache "
            "the part with a Cache(url), which fills from the part's own records")
    placed = _consumed(stored, read)
    # The fill numbered its inputs by their place where it read them in memory or by index, else by
    # its stream (`_source_ids`).
    placing: Read = ("streamed" if read == "streamed"
                     else "indexed" if placed.supports("indexed") else "memory")
    if (placing == "streamed" and any(isinstance(selection, RecordSlice) for selection in selections)
            and not placed.manifest().index_ordered):
        raise ValueError(
            f"a records holdout numbers its inputs by index, and generation {generation.url} numbers them by "
            f"the shards a stream of {placed.describe()} read, which it doesn't state is the index order; "
            "publish it from a memory read, or an indexed one (provision=True)")
    return SelectedGeneration(generation, selections, placed, placing), ()


def _standing_in(generation: Generation, pipeline: Pipeline, source: Source | Mixture | Phases,
                 prefix: Sequence[Stage]) -> Source:
    """The rows of `generation` in place of `prefix` over `source`
    (`_in_place`). A generation holding other rows than the steps before it
    make is refused, naming what differs."""
    replacement, changed = _in_place(generation, source, prefix)
    if replacement is None:
        raise ValueError(
            f"generation {generation.url} does not hold the rows the steps before it in {_name(pipeline)} "
            f"make: their {', '.join(changed)} differ; cache it after the source and steps it was published "
            "from, or read it as a source of its own (`Pipeline(generation.source())`)"
        )
    return replacement


def _promoted(pipeline: Pipeline) -> Pipeline:
    """The pipeline as the run reads it. Each published generation it caches
    is read as a source, in place of the steps the generation already ran
    (`_standing_in`), with its published packing in place of the pack stage
    that follows it. A first-fit pack of tokens alone directly on a source that reads its
    documents by index and states their lengths reads that source's planned windows
    (`PackedSource.planned`), by the plan a published generation stores, so
    the windows are indexed and the order is global in every read; a chunk
    step that cuts documents as the plan does (`chunks_for`) may come between,
    and the plan then cuts them, which it otherwise refuses. A source read
    only as a stream, such as one filtered by a predicate, is packed as its
    rows arrive, and so are documents counted by a role or a pad id (`_ruled`),
    whose windows a plan of their lengths does not count."""
    from dew.data.generation import Generation, _GenerationSource
    from dew.data.packing import PackedSource, chunks_for, plan_name
    from dew.data.sources.tokens import TokenFiles
    from dew.data.steps.windows import WindowSource

    def cutting(stages: Sequence[Stage]) -> int | None:
        """The overlap of a chunk step that cuts documents as the pack after it plans them
        (`chunks_for`), or None where none comes first."""
        if len(stages) > 1 and isinstance(stages[0], FlatMap) and isinstance(stages[1], Pack):
            return chunks_for(stages[0].step, stages[1])
        return None

    pipeline = with_children(pipeline, _promoted)
    source, stages = pipeline.source, pipeline.stages
    for index, stage in enumerate(stages):
        if isinstance(stage, Artifact) and isinstance(stage.cache, Generation):
            # The views of the inputs written before the generation read its rows as they read the
            # inputs, where it holds one row for each input, in their order (`_one_to_one`).
            views = tuple(earlier for earlier in stages[:index] if isinstance(earlier, _VIEWS))
            if views and not _one_to_one(stages[:index]):
                raise ValueError(
                    f"generation {stage.cache.url} holds the rows its steps kept, not one for each input, "
                    f"so the views written before it in {_name(pipeline)} can't read it by input; give "
                    "the cache that published it, Cache(url), which reads each input's rows")
            source = _standing_in(stage.cache, pipeline, source, stages[:index])
            stages = (*views, *stages[index + 1 :])
            plans = record(stage.cache.manifest.get("plan", {}), "published plans")
            overlap = cutting(stages)
            at = int(overlap is not None)
            packing = stages[at] if len(stages) > at else None
            # A published plan cuts a document as a window-long chunk step without overlap does.
            if (isinstance(source, _GenerationSource) and isinstance(packing, Pack)
                    and plan_name(packing, "accepted") in plans and overlap in (None, 0)
                    and not _ruled(source)):
                source = PackedSource.published(source, stage.cache, packing, chunked=overlap is not None)
                stages = stages[at + 1 :]
            break
    overlap = cutting(stages)
    at = int(overlap is not None)
    pack = stages[at] if len(stages) > at else None
    if (
        isinstance(pack, Pack)
        and pack.policy == "first-fit"
        and not pack.along
        and not isinstance(source, (Mixture, Phases))
        and source.supports("indexed")
        and source.lengths() is not None
        and not _ruled(source)
    ):
        source, stages = PackedSource.planned(source, pack, overlap=overlap), stages[at + 1 :]
    first = stages[0] if stages else None
    if isinstance(first, Windows) and first.fields == TOKEN_FIELDS and isinstance(source, TokenFiles):
        # A token store's own stream, cut where the windows are written: read by index.
        stride = first.seq_len if first.stride is None else first.stride
        source, stages = WindowSource(source, source.tokens, first.seq_len, stride), stages[1:]
    return dataclasses.replace(pipeline, source=source, stages=stages)


def _ruled(source: Source) -> bool:
    """Whether `source`'s documents may have been counted by a role or a pad id (`CountRule`), whose
    windows that rule counts and a plan of document lengths, counting every target, does not. A
    store certifies the rules its rows were counted by (`count_rules`): a generation and an export
    write them, a store held in memory reads them off its rows, a token store's spans hold none,
    and a store made of others gathers theirs. A store that certifies none may hold any, and its
    rows are counted as they arrive."""
    stated = source.metadata()
    return "count_rules" not in stated or bool(stated["count_rules"])


def _preferred_read(pipelines: Sequence[Pipeline], *, provision: bool) -> Read:
    """The read a run of `pipelines` takes when it states none, decided here
    once for compile, a held pass and an export.

    Each store the run reads, after its published generations are promoted,
    prefers memory for records held in memory (`Memory`, through any source
    that wraps one), indexed for a store that reads by index or can provision
    only that read, and streamed otherwise, and a mixture takes the least eager
    of its corpora's preferences. The run reads by the first read, from that preference toward
    the more eager ones, that it can take (`_takes`).

    A source that cannot take the preferred read until it is provisioned
    stops the walk there, so the preparation refuses it and names
    provisioning: a token store without its length index prefers a stream it
    cannot read, and provision=True builds the index rather than reading the
    store whole into memory. Past the preference, where the stages need a
    more eager read than the sources prefer, a read a source takes only once
    provisioned is passed over: JSON lines with a token split, which needs a
    finite indexed order, and a CSV with a records= holdout, which cannot be
    streamed, read into memory when they have no index. When no read is
    left, the preference stands, and the preparation refuses it with its own
    remedy.
    """
    promoted = tuple(_promoted(pipeline) for pipeline in pipelines)
    chosen: list[Read] = []
    for source in (source for pipeline in promoted for _, source in _leaves(pipeline)):
        source = _store(source)
        if _in_memory(source):
            chosen.append("memory")
        else:
            indexed = source.supports("indexed") or (
                not source.supports("streamed") and _provisioner(source, "streamed") is None
                and _provisioner(source, "indexed") is not None
            )
            chosen.append("indexed" if indexed else "streamed")
    preferred = READS.index(max(chosen, key=READS.index))
    for read in READS[preferred::-1]:
        missing = _unsupported(promoted, read)
        provisionable = all(_provisioner(source, read) is not None for source in missing)
        if read == READS[preferred] and missing and provisionable and not provision:
            return read
        if (not missing or (provision and provisionable)) and _lowers(promoted, read):
            return read
    return READS[preferred]


def _lowers(pipelines: Sequence[Pipeline], read: Read) -> bool:
    """Whether the analysis lowers every stage of `pipelines` at `read`, the
    first as the run trains it and the rest as scored branches. Its refusals
    are the one statement of what a read takes, so this asks them and states
    none of its own."""
    try:
        for index, pipeline in enumerate(pipelines):
            _analyzed(pipeline, read, validation=index > 0)
    except ValueError:
        return False
    return True


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
    return ((chain, chain.source) for chain in chains(pipeline)
            if not isinstance(chain.source, (Mixture, Phases)))


def _steps(pipeline: Pipeline) -> Iterator[Stage]:
    """Every stage of a pipeline's tree, each corpus's before the chain that mixes them (`chains`)."""
    return (stage for chain in chains(pipeline) for stage in chain.stages)


def _name(pipeline: Pipeline) -> str:
    return _called(pipeline, lambda source: source.describe())


def _label(pipeline: Pipeline) -> str:
    """The short name a corpus's metrics go by: the one it was given (`named`), else its store's
    (`Source.label`); its identity (`_name`) stays the plan's and the saved position's."""
    return _called(pipeline, lambda source: source.label())


def _called(pipeline: Pipeline, store: Callable[[Source], str]) -> str:
    """The name `pipeline` was given (`named`), a mixture's, or what `store` calls its store."""
    if pipeline.name is not None:
        return pipeline.name
    return "mixture" if isinstance(pipeline.source, (Mixture, Phases)) else store(pipeline.source)


def _members(pipelines: Sequence[Pipeline]) -> list[str]:
    """The names a mixture's members go by, for their draws and in its plan: the one each was given
    (`Pipeline.named`, as `name:<name>`), else its place among the members given none (`place:0`,
    `place:1`, ...), which no mount, read or preparation of its source changes; the two kinds never
    meet, whatever a name holds. A member keeps its draws and its place across a member joining or
    leaving by its name; one without keeps them while no other unnamed member joins or leaves ahead
    of it."""
    unnamed = itertools.count()
    return [f"name:{pipeline.name}" if pipeline.name is not None else f"place:{next(unnamed)}"
            for pipeline in pipelines]


def _member_keys(members: Sequence[str]) -> list[str]:
    """Each mixture member's key: the name it goes by (`_members`), as the plan's order records it
    for every read alike, with its place among the members of that name, so two such members are
    apart. A member's draws and its saved position are keyed by it, so adding or removing another
    member leaves them as they were."""
    seen: collections.Counter[str] = collections.Counter()
    keys = []
    for name in members:
        keys.append(f"{seen[name]}:{name}")
        seen[name] += 1
    return keys


def _identity(stage: Stage, *, cache: bool = False) -> StepIdentity:
    """What a stage runs, as the plan's manifest and a cache's key record it,
    refused for a cache's key where none can key it (`StepIdentity.of`). A
    cache stage is its kind alone. The plan names the rows a cache serves by
    their key (`Cache.reference`), and where the cache lives is no part of
    the data, so a run resumes against the same rows cached anywhere. A step
    a `Configured` stage resolved names its factory by what it computes and
    carries the factory's modules after its own."""
    if isinstance(stage, Artifact):
        return StepIdentity(f"dew.data.cache.{type(stage.cache).__name__}", "", exact=True)
    ran = runs(stage)
    if ran is None:
        return StepIdentity.of(stage)
    identity = StepIdentity.of(ran.step, ran.identity, cache=cache)
    if ran.configured is not None:
        bound = ran.configured
        factory = _identity(bound.step)
        binding = {"factory": factory.semantic(), "scope": bound.step.scope, "values": bound.values}
        identity = dataclasses.replace(
            identity,
            config=json.dumps({"step": identity.config, "configured": binding}, sort_keys=True),
            modules=(*identity.modules, *factory.modules),
        )
    if isinstance(stage, Batched):
        # The group size and whether the rows keep their count decide what a resume reads.
        shape = f"[batched {stage.size}, {'same rows' if stage.same_rows else 'any rows'}]"
        identity = dataclasses.replace(identity, name=identity.name + shape, config=identity.config + shape)
    return identity


def _configured(
    pipeline: Pipeline,
    shared: Mapping[str, object],
    read: Read,
    scope: Literal["source", "run"] | None = None,
    *,
    stated: Mapping[str, object] | None = None,
) -> Pipeline:
    """`pipeline` with its configured steps bound: a source-scoped one to the facts its source
    states, or `stated` (what `_stated` reads of it, measured facts included), a run-scoped one to
    the run's (`shared`)."""
    pipeline = with_children(pipeline, functools.partial(_configured, shared=shared, read=read, scope=scope))
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        local = shared
    else:
        local = source.metadata() if stated is None else stated
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
                "provide producer metadata, the step's settings or read='memory'"
            )
        stages.append(_bind(stage, metadata))
    return dataclasses.replace(pipeline, source=source, stages=tuple(stages))


def _bind(stage: Configured, metadata: Mapping[str, object]) -> Map | RandomMap | FlatMap:
    """The record step `stage` resolves to from the facts `metadata` states."""
    binding = Bound(stage, {key: metadata[key] for key in stage.needs})
    resolved = stage.factory(metadata)
    if isinstance(resolved, pygrain.RandomMapTransform) != stage.random:
        raise ValueError(f"Configured {_identity(stage).name} random={stage.random} disagrees with its step")
    if isinstance(resolved, Expansion) != stage.expands:
        raise ValueError(f"Configured {_identity(stage).name} expands={stage.expands} disagrees with its "
                         "step")
    if stage.one_to_one and (isinstance(resolved, Expansion) or _declares_rejects(resolved)):
        raise ValueError(f"Configured {_identity(stage).name} states one_to_one=True and binds to a step "
                         "that drops or expands records")
    bound = record_stage(resolved, configured=binding)
    # An objective's expansion keeps the mark its configured step carried (`Configured.examples`).
    return dataclasses.replace(bound, examples=True) if stage.examples else bound


def _keyed(stage: Stage) -> bool:
    """Whether a cache can key what `stage` runs (`StepIdentity.keyable`):
    the step `runs` gives, and a configured one's factory beside it. A cache
    prefix is refused by it (`_in_place`, `_cacheable`), where a configured
    factory's own identity is no cache key of its own."""
    ran = runs(stage)
    if ran is None:
        return True
    function = inspect.isfunction(ran.step)
    external = not (ran.step if function else type(ran.step)).__module__.startswith("dew.")
    keyable = StepIdentity.of(ran.step, ran.identity).keyable(function=function, external=external)
    return keyable and (ran.configured is None or _keyed(ran.configured.step))


def _manifest(
    snapshot: Mapping[str, object],
    sources: Sequence[Source],
    steps: Sequence[StepIdentity],
    *,
    fill: Mapping[str, object] | None = None,
) -> Manifest:
    """The manifest of the rows `sources` hold, named by `snapshot`, through
    `steps`. A source a memory read made (`materialized`) names the steps it already
    ran by what they compute; their modules join the environment here."""
    ran = tuple(_identity(stage) for source in sources for stage in _written(source, ()))
    return Manifest.of(snapshot, steps, ran=ran, fill=fill)


def _cache_manifest(source: Source, prefix: Sequence[Stage], read: Read) -> Manifest:
    """The key of a cache of `prefix` over `source`: the source's content and the record steps,
    without the views of its inputs (`_VIEWS`), which order or select what it holds and make none of
    it. A store given no name states no content (`Manifest.exact`), and its cache would serve any
    other's rows."""
    manifest = _cache_key(source, prefix, read)
    if manifest is None:
        raise ValueError(f"a cache keys its store's content, which {source.describe()} doesn't "
                         "state: name it (from_grain(..., name=), from_iter(..., name=), "
                         "from_torch(..., identity=))")
    return manifest


def _cache_key(source: Source, prefix: Sequence[Stage], read: Read) -> Manifest | None:
    """`_cache_manifest`, or None where `source` states no content of its own to key."""
    stated = source.fingerprint()
    if not identified(json_value(stated, "source")):
        return None
    return _manifest(
        stated,
        (source,),
        tuple(_identity(stage, cache=True) for stage in _cache_steps(prefix)),
        fill={"read": read, "row_ids": "sharded" if read == "streamed" else "indexed"},
    )


def _check_batch(stage: Stage, *, batched: bool, name: str) -> bool:
    """Whether the pipeline is batched once `stage` is read, refusing a second batch."""
    if isinstance(stage, (BatchStage, TokenBudget)):
        if batched:
            raise ValueError(f"pipeline {name} is already batched")
        return True
    return batched


_EAGER = (Map, MapWithIndex, Filter, FlatMap, Pack, Windows, TokenSplit)
"""The stages a memory read runs over the whole source before the first other stage."""


_LOCAL = (Map, MapWithIndex, Filter, FlatMap)
"""The deterministic record steps that run before a cache or an interleave,
inside the reader of each input or shard."""

_ROOTED = (*_LOCAL, Configured, Artifact, Interleave, Shuffle, Pack, Windows, Prefetch, Workers)
"""The stages a streamed read's repeat can move before, to its root: none
counts the rows or draws from the record rng, so they read one continuous
stream whose shards reshuffle each epoch."""


_VIEWS = (Shuffle, Repeat, Slice, Limit)
"""The stages that order or select a source's inputs and change none: a cache keyed by each input's
place in its source holds what the record steps make of it whatever view reads it, so these may
stand before a cache (`_cache_view`) and are no part of its key (`_cache_manifest`)."""


def _cache_steps(prefix: Sequence[Stage]) -> tuple[Stage, ...]:
    """What a cache after `prefix` holds and is keyed by: the record steps written before it, without
    the views of its inputs (`_VIEWS`), which every read applies to the inputs (`_Build.viewed`)."""
    return tuple(stage for stage in prefix if not isinstance(stage, _VIEWS))


def _cache_view(stages: Sequence[Stage]) -> str | None:
    """Why a cache after `stages` cannot hold what they make of each input, or None where it can: its
    stages are deterministic record steps and views of the inputs (`_VIEWS`), and a step before a
    view keeps one record for each input, so a view of the inputs reads what the written chain
    does (a record dropped, kept twice or replaced by pieces before a view would change which
    inputs the view orders)."""
    last = max((index for index, stage in enumerate(stages) if isinstance(stage, _VIEWS)), default=-1)
    for index, stage in enumerate(stages):
        if isinstance(stage, _VIEWS):
            continue
        if not _cacheable(stage):
            return f"{type(stage).__name__} is no deterministic record step a cache can key"
        if index < last and (_drops(stage) or isinstance(stage, FlatMap)):
            view = type(stages[last]).__name__
            return (f"{type(stage).__name__} before {view} drops or expands records, so the view orders "
                    "other rows than the source's inputs (expansions and drops before a view: roadmap 803b)")
    return None


def _cacheable(stage: Stage) -> bool:
    """Whether a cache can hold what `stage` makes: a deterministic record
    step, or a configured one, which binds to such a step, that a cache can
    key (`_keyed`)."""
    return isinstance(stage, (*_LOCAL, Configured)) and not _random(stage) and _keyed(stage)


def _one_to_one(stages: Sequence[Stage]) -> bool:
    """Whether a cache after `stages` holds one record for each of its inputs: none of them drops
    or expands records. Such a cache reads as the source's own rows do, by index and in one global
    order (`dew.data.lowering._Inputs`); one that drops or expands is read as a stream, per share. A
    step a run's facts haven't bound yet (`Configured`) may drop records once bound, so it counts as
    one that does unless it states it keeps one for each (`Configured.one_to_one`)."""
    return not any(_drops(stage) or (isinstance(stage, Configured) and not stage.one_to_one)
                   for stage in stages)


def _indexed_cache(prefix: Sequence[Stage], read: Read) -> bool:
    """Whether a cache after `prefix` reads as the source's rows do, by index and in one global order:
    one that holds one record for each input (`_one_to_one`), in a memory or indexed read; a
    streamed read fills its cache shard by shard."""
    return read != "streamed" and _one_to_one(prefix)


def _random(stage: Stage) -> bool:
    """Whether a stage draws from the record rng, so what it makes differs
    each epoch and no cache can hold it."""
    return isinstance(stage, RandomMap) or (isinstance(stage, Configured) and stage.random)


def _external(stage: Stage) -> bool:
    """Whether `stage` reads storage its record names by path (a step that says it is `External`), which
    the record's content does not key, so no automatic cache stands for it."""
    return (isinstance(stage, (Map, MapWithIndex)) and isinstance(stage.step, External)
            and stage.step.external)


def _volatile(stage: Stage) -> bool:
    """Whether `stage` reads what may change between reads (a step that says it is `Volatile`, as a
    live fetch does), so neither a cache nor a second reader may stand for it: the one test the
    cache prefix and the plan read."""
    return (isinstance(stage, (Map, MapWithIndex)) and isinstance(stage.step, Volatile)
            and stage.step.volatile)


def _drops(stage: Stage) -> bool:
    """Whether a stage can drop or make rows, so the rows after it are not
    the rows before it, one for one."""
    return isinstance(stage, (Filter, FlatMap, Pack, Windows, Native, Batched)) or (
        isinstance(stage, (Map, MapWithIndex, RandomMap)) and stage.rejects
    )


def _written(source: Source, stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """A corpus's stages as its pipeline wrote them: the ones a memory read
    already ran (`materialized`), then the ones left to lower, without a cache
    the compiler placed (`written`)."""
    while materialized(source):
        stages = (*source.ran, *stages)
        source = source.source
    return written(stages)


def written(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """`stages` as a chain wrote them: without a cache the compiler placed (`Artifact.automatic`),
    which holds what the stages before it make and is no step of the chain's identity, plan or
    layout."""
    return tuple(stage for stage in stages if not (isinstance(stage, Artifact) and stage.automatic))


@dataclasses.dataclass(frozen=True)
class _Lowering:
    """What a pipeline's stages mean for its lowering, analyzed once in one
    walk (`_analyzed`), which also refuses what no lowering takes. The memory
    executor, the Grain lowering and the plan read these decisions and scan
    the stages for none of them.

    `per_share` cuts the order into shares at the read, for an iterator order
    that cannot keep the global contract; otherwise one global order is cut
    just before batching, after every stage that defines it, so a prefetch
    never moves the cut and a limit counts the whole order. `children` are the
    analyses of a mixture's or phases' pipelines, in order, each cut where
    the order that mixes them is (`_cut`).

    `cache` is the cache stage, once every step before it is bound, and the
    stages before `eager` (after that cache) are the ones a memory read runs
    over the whole source. `interleave` is the interleave a streamed read
    flattens its shards at, after running the record steps before it inside
    each shard.

    `repeated` says whether any of its steps repeats the order; a training
    read of an order that repeats nowhere repeats it for ever, at its root.
    A streamed read repeats at its root in place of the first written repeat
    (`absorbed`) when only stages it can move across (`_ROOTED`) come before
    it, and its root reads `epochs` epochs when it trains: the absorbed
    repeat's, one when a repeat stays where it is written, and endless when
    none is written.

    `splits` are the positions of the token splits that run in the lowered
    order rather than in a memory read's eager pass, and `split` says whether
    any does, here or in a corpus it mixes; the plan takes the spans each
    keeps through this tree (`dew.data.lowering._split_spans`). `order` is the
    canonical order a resume must find unchanged. `parts_unending` says whether
    its source never ends: a mix of parts whose orders never end (`unending`),
    or a source a training read cut per share repeats at its root where its
    order repeats nowhere (`_Build.shared`, `_Build.streamed`); `unending`
    whether its order, its stages read in turn from there, never ends
    (`reach`), so a part repeated at its root and then cut by a limit ends;
    `records_unending` whether its records before its batch never end, and
    `onward` whether they are read on without end anywhere on the way; and
    `rooted` whether a training read repeats it at its root, as it does where
    nothing in it repeats or ends it. A training stream is `endless` where its
    order or that root repeat never ends. `stages` are its own stages, from
    which a cut restates these (`_cut`).
    """

    per_share: bool
    children: tuple[_Lowering, ...]
    cache: int | None
    eager: int
    interleave: int | None
    repeated: bool
    absorbed: int | None
    epochs: int | None
    splits: tuple[int, ...]
    order: Mapping[str, object]
    parts_unending: bool
    unending: bool
    records_unending: bool
    onward: bool
    rooted: bool
    one_to_one: bool
    """Whether the cache holds one record for each input (`_indexed_cache`), so a memory or indexed
    read reads it by index, in the order the uncached stages would make (`dew.data.lowering._Inputs`)."""
    stages: tuple[Stage, ...]

    @property
    def split(self) -> bool:
        return bool(self.splits) or any(child.split for child in self.children)

    @property
    def endless(self) -> bool:
        """Whether a training order never runs out: its own, or the repeat a training read adds at its
        root (`rooted`)."""
        return self.unending or self.rooted


def ends(stage: Stage) -> bool:
    """Whether `stage` ends the order it reads: a limit, or a slice with an end."""
    return isinstance(stage, Limit) or (isinstance(stage, Slice) and stage.selection.stop is not None)


def reach(stages: Sequence[Stage], unending: bool) -> tuple[bool, bool]:
    """Whether an order read through `stages` in turn, from one that never ends where `unending`,
    never ends, and whether it reads on without end anywhere on the way: a repeat without a count
    makes it endless, and a stage that `ends` it ends it again, as a Grain transformation of its own
    (`Native`) may, which nothing says it doesn't. The analysis's one reading of an order's
    finiteness, which its lowering states (`_Lowering`)."""
    onward = unending
    for stage in stages:
        if isinstance(stage, Repeat) and stage.epochs is None:
            unending = True
        elif ends(stage) or isinstance(stage, Native):
            unending = False
        onward = onward or unending
    return unending, onward


def _analyzed(
    pipeline: Pipeline, read: Read, *, validation: bool = False, promotion: bool = False
) -> _Lowering:
    """The analysis of `pipeline` at `read`, a scored branch with
    `validation`. A token split in a training order cut per share is refused
    once the steps are bound, since an unbound configured step may still be
    one a memory read runs eagerly; a scored branch reads its whole order in
    one canonical pass, so its split is whole."""
    source = pipeline.source
    nested = children(pipeline)
    if isinstance(source, Mixture):
        # A mixture reads its members' records and makes one batch of its own, so a member's batch,
        # and every stage after it, would never run.
        batched = next((child for child in nested if child.batched), None)
        if batched is not None:
            raise ValueError(f"mixture member {_name(batched)} is batched; a mixture batches once, after "
                             f"mix: move the member's batch and the stages after it onto the mixture")
    lowered = tuple(_analyzed(child, read, validation=validation, promotion=promotion) for child in nested)
    ancestry = tuple(stage for child in nested for stage in _steps(child))
    stages = pipeline.stages
    cache = _at(stages, Artifact)
    if nested or (cache is not None and any(isinstance(stage, Configured) for stage in stages[:cache])):
        cache = None
    # A view of the inputs before a cache leaves the cache to the lowering, which reads it by input
    # as an indexed read does; a memory read runs the steps before the first view over the whole
    # source first, as it does without a cache, and the cache holds the rest.
    viewed = cache is not None and any(isinstance(stage, _VIEWS) for stage in stages[:cache])
    eager = 0 if cache is None or viewed else cache + 1
    while read == "memory" and not nested and eager < len(stages) and isinstance(stages[eager], _EAGER):
        eager += 1
    per_share = _checked(pipeline, read, ancestry, eager, validation=validation, promotion=promotion)
    if isinstance(source, Mixture) and source.tokens is not None:
        per_share = per_share or not all(_states_its_targets(child) for child in nested)
    per_share = per_share or any(child.per_share for child in lowered)
    lowered = tuple(_cut(child, per_share, validation) for child in lowered)
    steps = (*ancestry, *stages)
    repeats = [stage.epochs for stage in steps if isinstance(stage, Repeat)]
    parts_unending, unending, records_unending, onward = _finiteness(
        stages, lowered, per_share and not validation and not repeats)
    written = next(((index, stage) for index, stage in enumerate(stages) if isinstance(stage, Repeat)), None)
    absorbed = (
        written
        if read == "streamed"
        and written is not None
        and all(isinstance(stage, _ROOTED) and not _random(stage) for stage in stages[: written[0]])
        else None
    )
    epochs = None if written is None else 1 if absorbed is None else absorbed[1].epochs
    if isinstance(source, Phases):
        order: Mapping[str, object] = {
            "phases": [child.order for child in lowered],
            "members": _members(source.pipelines),
            # A run's phases on the step clock keep the key their plans were written with.
            "until_steps" if source.clock == "steps" else "until_tokens": list(source.ends),
        }
    else:
        written = stages if isinstance(source, Mixture) else _written(source, stages)
        order = {
            **(
                {
                    "corpora": [child.order for child in lowered],
                    "members": _members(source.pipelines),
                    "weights": list(source.weights),
                    "shares": "tokens" if source.tokens is not None else "accepted rows",
                }
                if isinstance(source, Mixture)
                else {"source": source.describe()}
            ),
            "stages": [
                repr(stage)
                for stage in written
                if isinstance(stage, (Shuffle, Repeat, Slice, Limit, TokenSplit, Pack, Windows, BatchStage))
            ],
        }
    lowering = _Lowering(
        per_share,
        lowered,
        cache,
        eager,
        _at(stages, Interleave),
        bool(repeats),
        None if absorbed is None else absorbed[0],
        epochs,
        tuple(at for at, stage in enumerate(stages) if at >= eager and isinstance(stage, TokenSplit)),
        order,
        parts_unending,
        unending,
        records_unending,
        onward,
        not repeats and not any(ends(stage) for stage in steps),
        cache is not None and _indexed_cache(stages[:cache], read),
        tuple(stages),
    )
    configured = any(isinstance(stage, Configured) for stage in steps)
    if per_share and lowering.split and not validation and not configured:
        raise ValueError(f"token split in {_name(pipeline)} needs a global record order, not one per share")
    return lowering


def _states_its_targets(pipeline: Pipeline) -> bool:
    """Whether a corpus a mixture weighs by tokens delivers the rows its source's stated rows and
    valid targets count, so a weight from them holds its share of the targets (`token_weights`):
    a source that states them, whose stages only order, repeat and place those rows
    (`_KEEPS_TARGETS`), and after its repeat may take a window of the repeated rows, as a phase
    does. A step still to run over them, a random crop or a filter, before its repeat or after
    it, or a slice, limit or token split that takes a part of one epoch, delivers other rows than
    the stated ones, so that corpus's targets are counted as its rows arrive (`TokenDeficit`)."""
    source, at = pipeline.source, _until(pipeline.stages, (Repeat,))
    return (not isinstance(source, (Mixture, Phases)) and "valid_targets" in source.metadata()
            and all(isinstance(stage, _KEEPS_TARGETS) for stage in pipeline.stages[:at])
            and all(isinstance(stage, (*_KEEPS_TARGETS, Slice, Limit)) for stage in pipeline.stages[at:]))


_KEEPS_TARGETS = (Shuffle, Repeat, Prefetch, Workers, Interleave, Artifact)
"""The stages that order, repeat or place a corpus's rows, each row and its targets as they
arrive; a mixture's members are not batched, so these are all a member's stages may be."""


def _finiteness(stages: Sequence[Stage], children: Sequence[_Lowering], repeated_at_root: bool
                ) -> tuple[bool, bool, bool, bool]:
    """An order's `parts_unending`, `unending`, `records_unending` and `onward` (`_Lowering`): read
    through `stages` from a source that never ends where it is a mix or a run of phases none of whose
    parts ends, or a source the read repeats at its root (`repeated_at_root`), as a training read cut
    per share repeats one whose order repeats nowhere (`_Build.shared`, `_Build.streamed`)."""
    parts_unending = all(child.unending for child in children) if children else repeated_at_root
    unending, _ = reach(stages, parts_unending)
    records_unending, onward = reach(stages[:_until(stages, (BatchStage, TokenBudget))], parts_unending)
    return parts_unending, unending, records_unending, onward


def _cut(lowering: _Lowering, per_share: bool, validation: bool = False) -> _Lowering:
    """A mixed corpus's analysis, cut where the order that mixes it is cut:
    per share when that order is, so each corpus reads only its share, and its
    finiteness restated for that read, which repeats at its root a corpus whose
    order repeats nowhere, and so every order that mixes it."""
    children = tuple(_cut(child, per_share, validation) for child in lowering.children)
    parts_unending, unending, records_unending, onward = _finiteness(
        lowering.stages, children, per_share and not validation and not lowering.repeated)
    return dataclasses.replace(lowering, per_share=per_share, children=children,
                               parts_unending=parts_unending, unending=unending,
                               records_unending=records_unending, onward=onward)


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
        layout = isinstance(stage, (Pack, Windows, BatchStage, TokenBudget))
        combined = combined or (type(stage).__name__ if layout else None)
        finite = finite and not isinstance(stage, Repeat)
        if _random(stage):
            random = _identity(stage).name
            if validation and not canonical:
                raise ValueError(f"validation random step {random} must use the canonical record index")
        reorders = isinstance(stage, (Shuffle, Repeat, Slice, Limit)) or _drops(stage)
        canonical = canonical and not reorders
        if isinstance(stage, Artifact):
            if random is not None:
                raise ValueError(
                    f"cache prefix in {name} contains random step {random}; cache before stochastic steps"
                )
            for earlier in _cache_steps(prefix):
                _identity(earlier, cache=True)
            mixed = isinstance(pipeline.source, (Mixture, Phases))
            refused = None if mixed else _cache_view(pipeline.stages[:index])
            viewed = any(isinstance(earlier, _VIEWS) for earlier in prefix)
            if refused is None and read == "streamed" and viewed:
                refused = ("a streamed read fills its cache shard by shard, and a view of the inputs "
                           "before it reads them by index: read='indexed' or 'memory', or write the view "
                           "after the cache (streamed views before a cache: roadmap 803b)")
            if refused is not None:
                raise ValueError(f"cache prefix in {name} must hold deterministic record steps and views of "
                                 f"the inputs: {refused}")
        if isinstance(stage, Shuffle) and stage.window is None and not indexed and not promotion:
            raise ValueError(f"global shuffle in {name} needs an indexed pipeline; give window=")
        if isinstance(stage, Slice) and not indexed and not promotion:
            raise ValueError(f"{type(stage).__name__} in {name} needs an indexed pipeline")
        streams = isinstance(stage, (Interleave, Workers, BatchStage, TokenBudget)) or (
            _drops(stage) and index >= eager
        )
        # A cache reads as the stages it holds would: by index and in one order where they keep one
        # record for each input, as a stream and per share where they drop or expand them.
        streamed_cache = isinstance(stage, Artifact) and not _indexed_cache(prefix, read)
        indexed = indexed and not (streams or (streamed_cache and read != "memory"))
        shared = _drops(stage) or isinstance(stage, (Interleave, TokenBudget)) or streamed_cache
        per_share = per_share or shared or (isinstance(stage, Workers) and not batched)
        batched = _check_batch(stage, batched=batched, name=name)
        if isinstance(stage, Workers):
            if workers:
                raise ValueError(f"pipeline {name} contains workers more than once")
            workers = True
        if isinstance(stage, MapBatch):
            if not batched:
                raise ValueError(f"the host batch step in {name} must follow batch")
            host = True
        elif host and not isinstance(stage, (Workers, Prefetch)):
            raise ValueError(f"{type(stage).__name__} in {name} follows a host batch step")
        prefix.append(stage)
    return per_share


def _capabilities(pipelines: Sequence[Pipeline], read: Read, provision: bool) -> None:
    missing = _unsupported(pipelines, read)
    if provision:
        for source in missing:
            preparer = _provisioner(source, read)
            if preparer is not None:
                preparer.provision()
    refused: dict[str, list[str]] = {}
    for source in missing:
        if not source.supports(read):
            refused.setdefault(_refusal(source, read, provision=provision), []).append(source.describe())
    if refused:
        raise ValueError(f"read={read!r} is unavailable: "
                         + "; ".join(f"{', '.join(names)}: {why}" for why, names in refused.items()))


def _refusal(source: Source, read: Read, *, provision: bool) -> str:
    """Why `source` cannot take `read`: a holdout or slice over its store that
    reads no such way, which says what does, else its store, which a run
    that did not provision may prepare."""
    for view in beneath(source, SAME_ROWS):
        if isinstance(view, View) and read not in view.reads:
            return view.refusal
    if provision:
        return "its store takes no such read, and provisioning prepared none"
    return "provision the named artifact with dew data provision or provision=True"
