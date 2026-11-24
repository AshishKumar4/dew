"""The trainer's side of placement: the step's batch arguments (`STEP_BATCH`, `step_reads`), a stream's
checks and microbatch staging (`check_stream`, `stage_microbatches`), the links a run measures between
processes, and a run's first state traced and placed (`traced_draw`, `placed_start`). Where arrays live
is `dew.distributed.placement`'s."""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import math
import statistics
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any, Concatenate, Protocol

import jax
import numpy as np
from flax import linen as nn
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P, Sharding

from dew import _facts
from dew._cache import holding, keep, kept_beside, read_kept
from dew._coordination import from_every_process
from dew._records import JSON, number, record
from dew.batch import EXAMPLE_ROWS, Batch, Variables
from dew.data.dataset import (
    Checkpointable,
    DataPartition,
    Loader,
    PhasedStream,
    RampedStream,
    StepPhases,
    rows_of,
    stream_identity,
)
from dew.distributed.batches import Signature
from dew.distributed.placement import Layout, MeshSpec, Placement, link_topology, placed_shapes
from dew.mesh import STAGE_AXIS, LayoutRefused, batch_axes
from dew.nn.kernels.options import Compiling, OuterProgram, Placements, compiling, owned_program
from dew.nn.sharding import microbatched
from dew.objectives.base import Held, Initializer
from dew.objectives.drawn import Parts, drawing, drawn_rest, reassembled
from dew.state import TrainState, apart, together
from dew.telemetry.benchmark import samples


def mesh_stages(mesh: MeshSpec | Mesh) -> int:
    """The pipeline stages `mesh` lays out: a `MeshSpec`'s `stage`, a native mesh's stage axis, one where it
    has none."""
    return mesh.stage if isinstance(mesh, MeshSpec) else mesh.shape.get(STAGE_AXIS, 1)


def stage_microbatches(mesh: Mesh, asked: int | None) -> int:
    """The microbatches a pipelined step feeds through `mesh`'s stage axis: as
    `asked`, else one a stage. A mesh without a stage axis has one stage."""
    stages = mesh.shape.get(STAGE_AXIS, 1)
    if asked is None or asked == stages:
        return stages
    if stages == 1:
        raise ValueError(f"microbatches ({asked}) feed the stage axis, which this mesh does not split; "
                         f"give it stages or leave microbatches unset")
    if asked < stages or asked % stages:
        raise ValueError(f"microbatches must be a positive multiple of the stage axis's {stages}, "
                         f"got {asked}")
    return asked


class Ruled(Protocol):
    layout: Layout


def ruled[Owner: Ruled, **Arguments, Result](
        method: Callable[Concatenate[Owner, Arguments], Result]
) -> Callable[Concatenate[Owner, Arguments], Result]:
    """`method`, run under its owner's layout's rules: they place a batch's
    rows (`batch_axes`) and a run's data shares as they place the activations
    its trace constrains."""
    @functools.wraps(method)
    def run(owner: Owner, *args: Arguments.args, **kwargs: Arguments.kwargs) -> Result:
        with nn.logical_axis_rules(owner.layout.axis_rules):
            return method(owner, *args, **kwargs)
    return run


def link_bandwidth(mesh: Mesh, axis: str, size: int = 1 << 28) -> float:
    """What one device receives a second in an all-gather over `mesh`'s
    `axis` of a `size`-byte result, with every group along the axis
    gathering at once as a step's do: (N - 1) / N of the result over the
    median of five gathers, after two that warm the collective. Every
    process takes the pool's lowest figure, so every process decides from
    the same number and compiles the same program; the slowest group bounds
    the step anyway.

    `dew.nn.sharding.down_projection` reads the tensor axis's to decide
    whether a down-projection of the residual runs on each tensor shard's
    own tokens, and `dew.nn.sharding.split_positions` the sequence axis's.
    The default result, 256 MiB, of which a device receives at least half,
    is the size of the collectives those decisions price where they matter
    (DeepSeek-V3's residual gradient over 16384 tokens is 235 MB in bf16),
    past the sizes where a collective's latency counts: on 4x RTX 3090 an
    NVLink pair's all-gather moved 8.2 GB/s a device at 4 MiB and 31.0 at
    128 MiB.

    A pool keeps its figure beside the persistent compilation cache, by the
    axis and the devices it spans, and a later pool on them takes it, a
    resume among them: the decisions from it shape the step, and a figure
    measured again moves by noise, across a threshold one time in some."""
    topology = link_topology(mesh, axis)
    record = _link_record(topology)
    # Every process gathers, or none does: a pool measures unless each of its processes reads the figure,
    # kept for exactly this placement; a record no process can read (another writer's truncated file) is
    # one it does not see, never an error one process raises while its peers wait in the gather.
    kept = None if record is None else read_kept(record)
    figure = (kept["bytes_per_second"] if isinstance(kept, dict) and kept.get("topology") == topology
              and isinstance(kept.get("bytes_per_second"), int | float) else None)
    if all(from_every_process(figure is not None)):
        return min(from_every_process(float(figure)))
    ways = mesh.shape[axis]
    count = size // 4 // ways * ways
    sharding = NamedSharding(mesh, P(axis))
    source = jax.make_array_from_callback(
        (count,), sharding, lambda index: np.zeros(sharding.shard_shape((count,)), np.float32))
    gather = owned_program(jax.shard_map(
        lambda shard: jax.lax.all_gather(shard, axis, tiled=True), mesh=mesh,
        in_specs=P(axis), out_specs=P(), axis_names={axis}, check_vma=False))
    seconds = samples(gather, source, warmup=2, repeats=5)
    received = min(from_every_process((ways - 1) / ways * count * 4 / statistics.median(seconds)))
    if record is not None and jax.process_index() == 0:
        keep(record, {"topology": topology, "bytes_per_second": received}, "the link's figure")
    return received


def link_figures(bandwidths: Mapping[tuple[Mesh, str], float | None]) -> dict[str, JSON]:
    """The link figures a checkpoint's rung keeps (`StepCompiler.checkpoint_rung`): each figure of
    `bandwidths` a step traced with, by mesh and axis, under the placement it was taken on
    (`link_topology`), None where a step compiled ahead took none (`AheadOfTime.links`)."""
    return {link_topology(mesh, axis): figure for (mesh, axis), figure in bandwidths.items()}


def link_figure(mesh: Mesh, axis: str, resumed: Mapping[str, float | None]) -> float | None:
    """The bandwidth a step over `mesh` decides `axis`'s projections by: the one a resumed checkpoint
    traced with on this very placement (`resumed_link_figures`), None where it was unmeasured, else the
    one measured there (`link_bandwidth`)."""
    topology = link_topology(mesh, axis)
    return resumed[topology] if topology in resumed else link_bandwidth(mesh, axis)


def resumed_link_figures(rung: JSON) -> dict[str, float | None]:
    """`link_figures` as a checkpoint's rung recorded them; none where it recorded none."""
    links = record(record(rung, 'rung').get('links', {}), 'rung links')
    return {topology: None if figure is None else number(figure, f'link of {topology}')
            for topology, figure in links.items()}


def _link_record(topology: str) -> Path | None:
    """Where `link_bandwidth` keeps the figure of `topology` (`link_topology`): beside the persistent
    compilation cache it is configured with (`kept_beside`), or nowhere where the process keeps none or
    keeps it in a bucket."""
    directory = kept_beside("dew-links", configured=True)
    return None if directory is None else directory / f"{hashlib.sha256(topology.encode()).hexdigest()}.json"


def batch_divisor(mesh: Mesh | AbstractMesh, microbatches: int) -> int:
    """Return the row count every global batch must be a multiple of.

    A batch's rows are sharded over the row axes (`batch_axes`) as whole
    rows, and a pipelined step cuts each device's rows again into
    `microbatches`: microbatch m takes rows m, m + M, ..., which is a share
    of every device's rows only where M divides them. A batch that divides
    by neither is refused where it is placed or traced.

    `Trainer.fit` checks a batch against this before it places anything,
    and a batch ramp's every stage, since a later stage would otherwise fail
    an hour in.
    """
    return math.prod(mesh.shape[axis] for axis in batch_axes(mesh)) * microbatches


def check_batch(batch: int, mesh: Mesh, microbatches: int) -> None:
    """Refuse, before anything is placed, a global batch `mesh` cannot split
    (`batch_divisor`): its rows shard over the row axes as whole rows, and a
    pipeline cuts each device's rows again into `microbatches`, where a
    device that holds none of a microbatch's rows computes another's again.
    The message names the row shards, and the batches and the microbatch
    counts that fit."""
    divisor = batch_divisor(mesh, microbatches)
    if batch % divisor == 0:
        return
    count, stages = microbatches, mesh.shape.get(STAGE_AXIS, 1)
    shards = divisor // count
    suggestions = [f"a batch that is a multiple of {divisor} rows, {-(-batch // divisor) * divisor} "
                   f"the nearest above {batch}"]
    rows = batch_axes(mesh)
    axes = " x ".join(f"{axis} {mesh.shape[axis]}" for axis in rows)
    if count == 1:
        raise LayoutRefused(
            f"a global batch of {batch} rows over the {shards} row shards of {axes} leaves some "
            f"device without whole rows; use {suggestions[0]}, or a mesh whose row axes "
            f"({' x '.join(rows)}) divide {batch}")
    if batch % shards == 0:
        fitting = [m for m in range(stages, batch // shards + 1, stages)
                   if (batch // shards) % m == 0]
        if fitting:
            suggestions.append(f"microbatches={fitting[-1]}, the most that divide the "
                               f"{batch // shards} rows a device holds")
    raise LayoutRefused(
        f"a global batch of {batch} rows over {shards} row shards and {count} microbatches "
        f"leaves some microbatch without rows on some device, which then computes another "
        f"microbatch's again; use {' or '.join(suggestions)}")


def check_stream(source, mesh: Mesh, microbatches: int, batch: int, *, rows: bool,
                 resumed: int | None) -> None:
    """Refuse a training stream a run cannot resume or cannot shard.

    `resumed` is the step a resume restored from a checkpoint without a data
    position, which no stream can continue: with no position to restore, one
    that keeps positions (`Checkpointable`) and one declared to replay alike
    would read its data again from the start, so past step 0 it is refused.
    A `batch` the mesh cannot divide fails where it is placed or
    traced (`check_batch`), unless the stream's rows are not the step's
    (`rows`; a rollout's are its own, checked where its step traces), and a
    ramp stage's or a later phase's (`PhasedStream.batches`, `StepPhases.batches`), an hour in. A ramp
    reads each stage's
    batch whole; an accumulation window pools its stages' batches as it pools
    any, but for a loss that replays its window's batches (`dew.training._candidates.prepared_for`).
    """
    if resumed:
        kept = isinstance(source, Checkpointable)
        why = ("it was saved without one (Checkpoints.save's `saved`)" if kept
               else f"{stream_identity(source)} keeps none")
        remedy = ("save the stream's position with the state, as Trainer.fit does" if kept else
                  "give the stream get_state and set_state, read it through dew.data, or declare that "
                  "it reads the same batches each time it opens (dew.data.dataset.replayed, "
                  "from_iter(replay=True))")
        raise ValueError(f"step {resumed}'s checkpoint holds no data position, as {why}, so resuming "
                         f"would read its data again from the start; {remedy}")
    if isinstance(source, PhasedStream | StepPhases):
        if rows:
            for read in sorted(set(source.batches)):
                check_batch(read, mesh, microbatches)
        return
    if not isinstance(source, RampedStream):
        if rows:
            check_batch(batch, mesh, microbatches)
        return
    divisor = batch_divisor(mesh, microbatches)
    refused = [stage.batch for stage in source.stages if stage.batch % divisor]
    if refused:
        raise ValueError(
            f"the batch ramp reads {refused} records a step at some of "
            f"its stages, and {dict(mesh.shape)} with "
            f"{microbatches} microbatch(es) "
            f"holds a batch that is a multiple of {divisor}")


def microbatch(batch: Batch, part: jax.typing.ArrayLike, split: int) -> Batch:
    """The `part`th of the `split` microbatches a step reads `batch` in
    (`Trainer.compile`): every `split`th row from row `part` (`microbatched`),
    a share of each device's rows where `split` divides them, so no row
    crosses devices. A field with no rows is the whole step's, and every
    microbatch reads it whole; one microbatch is the batch as it came,
    whatever its fields' rows (a rollout's groups)."""
    if split == 1:
        return batch
    return jax.tree.map(lambda leaf: jax.lax.dynamic_index_in_dim(
        microbatched(leaf, 0, split), part, 0, keepdims=False) if np.ndim(leaf) else leaf, batch)


def microbatch_shapes(batch: Batch, split: int) -> Batch:
    """The fields of `microbatch`'s microbatches, as a trace reads them: the
    batch itself for one, whose placed fields' types carry their Explicit
    axes, else their shapes."""
    if split == 1:
        return batch
    part = jax.ShapeDtypeStruct((), np.dtype(np.int32))
    return jax.eval_shape(functools.partial(microbatch, split=split), batch, part)


STEP_BATCH = 2
"""Where a compiled train step takes its batch among its arguments: after the moving state and the
frozen tree (`dew.state.apart`, `dew.training._candidates.step_program`), before the part of the step
it runs."""


def step_reads(executable: jax.stages.Compiled, signature: Signature) -> tuple[bool, ...]:
    """Which leaves of its batch (`STEP_BATCH`) a compiled step reads,
    in the leaf order of the batch's own `signature` (`batch_signature`):
    those the executable kept, where JAX pruned the rest
    (`jax.stages.Compiled.input_shardings` names no sharding for those). A
    field the batch holds as None, as an optional part of `Coordinates`, is
    None in the shardings too and no leaf of the batch, so the batch's tree,
    not the shardings', says where its leaves are."""
    arguments, _ = executable.input_shardings
    tree, _ = signature
    placed = arguments[STEP_BATCH]
    # A field the step adds to the batch it fetched (a drawn pass count, `LOOP_PASSES`) is none of its leaves.
    fetched = tree.unflatten(range(tree.num_leaves))
    if isinstance(placed, dict) and isinstance(fetched, dict):
        placed = {name: placed[name] for name in fetched}
    return tuple(sharding is not None for sharding in tree.flatten_up_to(placed))


@dataclasses.dataclass
class Produced:
    """What a training stream declares it hands each step, checked on each batch before its transfer
    (docs/design/training.md, the producer's contract; row 5724): its share's rows of a step, one of the
    global batches its loader states (`Loader.batch`, a schedule's phases' `Loader.batches`, a ramp's
    stages'), and each field its chain fixes at its per-row shape (`Loader.fields`; row 9017). A batch of
    other rows is refused naming both, where it would compile as another step, as is one missing a fixed
    field or holding it at another shape, naming the field; its fields' dtypes may change, each its own
    step's (a mixed-precision stream's). A batch measured in tokens (`Loader.batch_tokens`) holds rows as
    its tokens fit, and states none."""

    rows: frozenset[int] | None
    fields: Mapping[str, tuple[int | None, ...]] = dataclasses.field(default_factory=dict)

    @classmethod
    def of(cls, loader: Loader, partition: DataPartition) -> Produced:
        if loader.batch_tokens is not None or any(tokens is not None for tokens in loader.token_batches):
            return cls(None, loader.fields)
        ramp = loader.ramp
        ramped = () if ramp is None else tuple(
            stage.batch for stage in ramp.stages(loader.batch, partition.window))
        batches = {loader.batch, *loader.batches, *ramped}
        return cls(frozenset(partition.rows(batch) for batch in batches), loader.fields)

    def checked(self, batch: Batch) -> Batch:
        rows = rows_of(batch)
        if self.rows is not None and rows not in self.rows:
            raise ValueError(f"the training stream handed a step {rows} rows on this share, where its loader "
                             f"states {sorted(self.rows)}: a step after the batch changed its rows, which "
                             "would compile another step; batch where the rows are made")
        for name, shape in self.fields.items():
            held = batch.get(name)
            if held is None:
                raise ValueError(f"the training stream handed a step no field {name!r}, which its chain "
                                 "states each row holds: a step after it dropped the field")
            if isinstance(held, np.ndarray) and not _fits(held.shape[1:], shape):
                raise ValueError(f"the training stream handed a step field {name!r} at {held.shape[1:]} a "
                                 f"row, where its chain states {shape}: a step after it changed the shape, "
                                 "which would compile another step")
        return batch


def _fits(held: tuple[int, ...], shape: tuple[int | None, ...]) -> bool:
    """Whether a row's shape `held` is `shape`, each axis it leaves free (None) any length."""
    return len(held) == len(shape) and all(axis is None or axis == size for size, axis in zip(held, shape,
                                                                                               strict=True))


def marked(partition: DataPartition, prepare: Callable[[Batch], Batch] | None = None
           ) -> Callable[[Batch], Batch]:
    """A training batch's host preparation: `prepare`'s, then each row's place among its step's
    records (`EXAMPLE_ROWS`). A share (`DataPartition.of`) reads the records `index :: count`, so its
    `j`th row is the step's record `j * count + index` at every count: a record draws what it draws
    however many processes read it (`Step.sample`)."""
    def prepared(batch: Batch) -> Batch:
        batch = batch if prepare is None else prepare(batch)
        rows = np.arange(rows_of(batch), dtype=np.int32)
        return {**batch, EXAMPLE_ROWS: rows * partition.count + partition.index}

    return prepared


def placed_held[Held](held: Held, variables: Variables, shardings: Placement[Variables],
                      mesh: Mesh, shared: Mapping[int, NamedSharding] | None = None) -> tuple[Held, Held]:
    """A copy of `held` on `mesh` as two trees, each None where the other holds
    an array: the arrays a state of `variables` placed by `shardings` takes
    over, each where the state keeps the variable of its path and shape
    (`Objective.held_variables`), and the rest, replicated, which no output of
    the state takes.

    Handed to the state's JIT as they are, the arrays land below the state it
    builds and leave a hole the checkpoint's size once freed, which splits
    the free memory a step's temporaries need in one block
    (`step_headroom`). Placed where the state keeps them and donated
    (`taking_over`), they become the state's buffers. The copy leaves the
    objective's own arrays alone. The held leaves whose ids `shared` maps to
    where the state keeps them, which it keeps as they are and no step donates
    (its frozen tree, `apart`, wherever its init moves them), are placed there
    as they are, the same arrays where they are placed already, and kept: no
    copy of them is made."""
    where, takes = _placements(held, variables, shardings, mesh, shared)
    shared = shared or {}
    return _split(takes, jax.tree.map(
        lambda leaf, target: (jax.device_put(leaf, target)
                              if id(leaf) in shared or not isinstance(leaf, jax.Array)
                              else transferred_copy(leaf, target)), held, where))


def transferred_copy(leaf: jax.Array, target: Sharding | None = None) -> jax.Array:
    """A copy of `leaf` in buffers of its own, made by a transfer (`may_alias=False`), placed by `target`, or
    where `leaf` is as it is, committed to its devices or not, where None: `jax.device_put` otherwise reuses
    a buffer the target shares with the source, which a step's donation would free, and `jax.Array.copy`
    compiles a program for each shape and placement, so a run handed its programs compiled ahead would
    compile at its start (row 4779). Dew's JAX build copies a resharded source too (from post77; JAX's own
    reshard path dropped the forced copy). The one copy a state takes of an array it does not own: a held
    array (`placed_held`; `Trainer.initial_state`, whose arrays a step then places as it places any
    uncommitted one), and the parameters its average starts from and the loss scale it is given."""
    return jax.device_put(leaf, target, may_alias=False)


def held_shapes[Held](held: Held, variables: Variables, shardings: Placement[Variables], mesh: Mesh,
                      shared: Mapping[int, NamedSharding] | None = None) -> tuple[Held, Held]:
    """`placed_held`'s two trees as the shapes, dtypes and placements of their arrays, none placed: what a
    start compiled ahead compiles over (`start_compiled`)."""
    where, takes = _placements(held, variables, shardings, mesh, shared)
    return _split(takes, placed_shapes(held, where))


def _placements[Held](held: Held, variables: Variables, shardings: Placement[Variables], mesh: Mesh,
                      shared: Mapping[int, NamedSharding] | None) -> tuple[Held, Held]:
    """Where each leaf of `held` goes (`placed_held`), and whether the state takes it over: a leaf of a
    variable's path and shape, placed where the state keeps it, that `shared` does not keep as it is."""
    shared = shared or {}
    kept = {path: (sharding, leaf.shape) for (path, leaf), sharding in zip(
        jax.tree_util.tree_leaves_with_path(variables), jax.tree.leaves(shardings), strict=True)}
    replicated = NamedSharding(mesh, P())

    def variable(path, leaf) -> NamedSharding | None:
        # An initializer binds the checkpoint as `variables`
        # (`Objective.initializer`), so a leaf's path below that keyword
        # is its variable's.
        keys = [key.key if isinstance(key, jax.tree_util.DictKey) else None for key in path]
        sharding, shape = kept.get(path[keys.index('variables') + 1:] if 'variables' in keys else (),
                                   (None, None))
        return sharding if shape == np.shape(leaf) else None

    where = jax.tree_util.tree_map_with_path(
        lambda path, leaf: shared.get(id(leaf)) or variable(path, leaf) or replicated, held)
    takes = jax.tree_util.tree_map_with_path(
        lambda path, leaf: variable(path, leaf) is not None and id(leaf) not in shared, held)
    return where, takes


def _split[Held](takes: Held, placed: Held) -> tuple[Held, Held]:
    """`placed` as the leaves the state takes over (`takes`) and the rest, each tree None where the other
    holds a leaf."""
    taken = jax.tree.map(lambda take, leaf: leaf if take else None, takes, placed)
    return taken, jax.tree.map(lambda one, leaf: leaf if one is None else None, taken, placed,
                               is_leaf=lambda leaf: leaf is None)


def taking_over[Held, Result](program: OuterProgram, taken: Held, kept: Held
                              ) -> Callable[[jax.Array], Result]:
    """The start's `program` (`StartBuild.program`) over the held tree `placed_held` splits: `taken`
    donated, for the outputs to take its buffers over, and `kept` not, as no output takes those and JAX only
    warns of a donation it cannot use. They are freed as the caller drops them. Its draws lowered as a
    fresh state's are (`drawing`)."""

    def taken_over(key: jax.Array) -> Result:
        with drawing():
            return program(_held_leaves(taken), _held_leaves(kept), key)

    return taken_over


def _held_leaves(tree) -> list:
    """One of the held tree's two trees as its places' leaves, None where the other holds the leaf."""
    return jax.tree.leaves(tree, is_leaf=lambda leaf: leaf is None)


START_PROGRAM = "joined"
"""The name a fresh state's start program traces and compiles under (`start_program`), by which a plan's
seconds tell its work from the step's (`dew.training.ahead.AheadOfTime.planned_parts`' `start`)."""


def start_program[Held, Result](function: Callable[[Held, jax.Array], Result], shardings: Placements,
                                like: Held) -> OuterProgram:
    """The program `taking_over` runs, over the held tree's two trees as their leaves (`_held_leaves`) and
    the key, the tree rebuilt within it in the structure of `like`, either of the two: one program
    whichever arrays or shapes it is called or compiled over (`start_compiled`), whose arguments' structure
    holds none of the held tree's own, which an initializer's functions may be part of, so a start compiled
    ahead saves whole (`AheadOfTime.starts`)."""
    structure = jax.tree.structure(like, is_leaf=lambda leaf: leaf is None)

    def joined(taken: list, kept: list, key: jax.Array) -> Result:
        return function(jax.tree.unflatten(structure, [other if one is None else one
                                                       for one, other in zip(taken, kept, strict=True)]), key)

    assert joined.__name__ == START_PROGRAM

    return owned_program(joined, out_shardings=shardings, donate_argnums=0)


@dataclasses.dataclass(frozen=True)
class StartBuild:
    """How a fresh state is built and placed (`start_build`), whether its arrays or their shapes are placed:
    the function its program runs over the held tree and the key, where that program puts its results, the
    held leaves the state keeps as they are (`shared`, by id, where it keeps them), and where the init
    passes some frozen leaves through (`through`), the frozen tree's placements and structure and the
    leaves the program builds (`built`, by index among the frozen tree's leaves). Its program is built once
    (`program`), so the start a run compiles beside its step (`start_compiling`) and the one it places
    (`placed_start`) are one program, whose lowering the compile point keeps."""

    function: Callable[[Initializer, jax.Array], Any]
    out_shardings: Any
    shared: Mapping[int, NamedSharding]
    through: Mapping[int, int]
    frozen: tuple[list[NamedSharding], Any] | None = None
    built: tuple[int, ...] = ()
    programs: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    def program(self, like) -> OuterProgram:
        """The start's program (`start_program`) over a held tree split as `like` is (its arrays or their
        shapes), built the first time it is asked for."""
        structure = jax.tree.structure(like, is_leaf=lambda leaf: leaf is None)
        if structure not in self.programs:
            self.programs[structure] = start_program(self.function, self.out_shardings, like)
        return self.programs[structure]


def start_build(initial_state: Callable[[Initializer, jax.Array], TrainState], initializer: Initializer,
                shardings: Placement[TrainState], through: Mapping[int, int], built: tuple[int, ...], *,
                kept: bool) -> StartBuild:
    """The one way a fresh state is built and placed, which a run's start runs (`placed_start`) and a run
    compiled ahead compiles (`start_compiled`): where its init passes no frozen leaf through, one program
    builds the state; where it does, the passed leaves are the held arrays placed where the state keeps them,
    and the program builds the rest (`drawn_rest`), the frozen leaves at `built` among it (`traced_draw`'s
    `TracedDraw.built`). One build a process keeps for each init it `kept` (a refit's start traced once,
    `traced_draw`; by the object itself), placement and held arrays it passes through, held weakly (`Held`,
    `_cache.holding`), so a trainer's refits compile and lower its program once (PF03): their start's
    program keeps the executable their call runs (`start_compiling`). An init no process keeps is traced
    afresh each fit, so its build is read once and kept by none, where it took a kept one's place among the
    `KEPT_STARTS`."""
    if not kept:
        return _built(initial_state, initializer, shardings, through, built)
    inputs = jax.tree.leaves(initializer)
    where = jax.tree.flatten(shardings)
    passed = tuple((index, Held(inputs[given])) for index, given in sorted(through.items()))
    return holding(_BUILDS, (Held(initial_state), where[1], tuple(where[0]), passed),
                   lambda: _built(initial_state, initializer, shardings, through, built), KEPT_STARTS)


def _built(initial_state: Callable[[Initializer, jax.Array], TrainState], initializer: Initializer,
           shardings: Placement[TrainState], through: Mapping[int, int], built: tuple[int, ...]
           ) -> StartBuild:
    """`start_build`'s build, made."""
    moving, frozen = apart(shardings)
    if not frozen or not through:
        return StartBuild(initial_state, shardings, {}, through)
    inputs = jax.tree.leaves(initializer)
    where, structure = jax.tree.flatten(frozen)
    # Each passed leaf is placed once, where the state keeps it, whatever path the init moves it to.
    return StartBuild(drawn_rest(initial_state, HELD_APART, built),
                      (moving, tuple(where[index] for index in built)),
                      {id(inputs[given]): where[index] for index, given in through.items()}, through,
                      (where, structure), built)


def start_compiled(start: StartBuild, initializer: Initializer, key: jax.ShapeDtypeStruct,
                   abstract: TrainState, shardings: Placement[TrainState], mesh: Mesh) -> jax.stages.Compiled:
    """The program `placed_start` runs to build a fresh state on `mesh` (`start_build`), compiled through
    Dew's compile point over the shapes of its arguments (`held_shapes`, `key`'s): what a run compiled ahead
    compiles of its start, whose devices compile none of it (`dew.training.ahead.AheadOfTime.starts`)."""
    program, arguments = _start_shapes(start, initializer, abstract, shardings, mesh)
    with drawing():
        return program.compiled(*arguments, key)


def start_compiling(start: StartBuild, initializer: Initializer, key: jax.ShapeDtypeStruct,
                    abstract: TrainState, shardings: Placement[TrainState], mesh: Mesh) -> Compiling | None:
    """`start_compiled`'s program lowered and its compile begun in a thread of its own (`compiling`): a run
    compiled ahead compiles its start beside its step's planning, which reads none of it, and takes it
    where `start_compiled` lowers it (`attempting`). A Qwen3.5-4B's start took 97 s to compile for an
    H100 after its step's 110 s. None where the program keeps an executable already (`OuterProgram.keeps`):
    a refit's, whose build its process kept (`start_build`). A build's program is called over one shape
    alone, its build's arguments' (the held tree its key states, placed as its placement states, and the
    key), so the executable it keeps is the one its call runs, nothing lowered or compiled (PF03; a program
    released keeps none, `options.release`)."""
    program, arguments = _start_shapes(start, initializer, abstract, shardings, mesh)
    if program.keeps():
        return None
    with drawing():
        return compiling(program, *arguments, key)


def _start_shapes(start: StartBuild, initializer: Initializer, abstract: TrainState,
                  shardings: Placement[TrainState], mesh: Mesh) -> tuple[OuterProgram, tuple[list, list]]:
    """The start's program (`StartBuild.program`) and its held tree's two trees' shapes (`held_shapes`):
    what `start_compiled` compiles over them and the key."""
    taken, kept = held_shapes(initializer, abstract.variables, shardings.variables, mesh, start.shared)
    return start.program(taken), (_held_leaves(taken), _held_leaves(kept))


KEPT_STARTS = 4
"""The start builds a process keeps for a refit to read (`start_build`), the latest of them."""

HELD_APART: Parts[TrainState, TrainState, dict[str, Variables]] = Parts(apart, together)
"""A fresh state's parts as a draw reads them (`dew.objectives.drawn.Parts`): its moving trees and its frozen
tree, whose leaves may be held arrays its init passes through."""

_BUILDS: collections.OrderedDict[Hashable, _facts.Held[StartBuild]] = collections.OrderedDict()


def placed_start(start: StartBuild, initializer: Initializer, key: jax.Array, abstract: TrainState,
                 shardings: Placement[TrainState], mesh: Mesh) -> TrainState:
    """A fresh state on `mesh`, built by `initial_state` from the objective's held tree (`placed_held`).
    The frozen leaves the objective's init passes through as they are (`through`, `traced_draw`),
    which no step donates (`apart`), are the held arrays themselves, placed where the state keeps
    them, the same arrays where they are placed already: the run and the objective then hold one
    frozen tree between them. One program builds the rest, a frozen leaf the init builds among it,
    taking the other held arrays over (`start_build`), the program the start compiled beside the step
    where it did (`StartBuild.program`)."""
    taken, kept = placed_held(initializer, abstract.variables, shardings.variables, mesh, start.shared)
    built = taking_over(start.program(taken), taken, kept)(key)
    if start.frozen is None:
        return built
    state, made = built
    where, structure = start.frozen
    placed = jax.tree.leaves(jax.tree.map(lambda one, other: other if one is None else one, taken, kept,
                                          is_leaf=lambda leaf: leaf is None))

    def held_at(index: int) -> jax.Array:
        # A held leaf an init passes through at two places of other shardings is placed at the second too.
        held = placed[start.through[index]]
        return held if held.sharding == where[index] else jax.device_put(held, where[index])

    return reassembled(HELD_APART, state, made, start.built, structure.unflatten, len(where), held_at)
