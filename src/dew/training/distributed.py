"""The trainer's side of placement: the step's batch arguments (`STEP_BATCH`, `step_reads`), a stream's
checks and microbatch staging (`check_stream`, `stage_microbatches`), the links a run measures between
processes, and a run's first state traced and placed (`traced_start`, `placed_start`). Where arrays live
is `dew.distributed.placement`'s."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Concatenate, Protocol

import jax
import numpy as np
from flax import linen as nn
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew._coordination import from_every_process
from dew._records import JSON, number, record
from dew.batch import EXAMPLE_ROWS, Batch, Variables
from dew.data.dataset import Checkpointable, DataPartition, RampedStream, rows_of, stream_identity
from dew.distributed.batches import Signature
from dew.distributed.placement import Layout, Placement, link_topology
from dew.files import write_atomically
from dew.mesh import STAGE_AXIS, LayoutRefused, batch_axes
from dew.nn.sharding import microbatched
from dew.objectives.base import Initializer
from dew.state import TrainState, apart, together
from dew.telemetry.benchmark import samples


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
    # Every process gathers, or none does: a pool measures unless each of its processes sees the figure,
    # kept for exactly this placement.
    if record is not None and all(from_every_process(
            record.exists() and json.loads(record.read_text()).get("topology") == topology)):
        return min(from_every_process(float(json.loads(record.read_text())["bytes_per_second"])))
    ways = mesh.shape[axis]
    count = size // 4 // ways * ways
    sharding = NamedSharding(mesh, P(axis))
    source = jax.make_array_from_callback(
        (count,), sharding, lambda index: np.zeros(sharding.shard_shape((count,)), np.float32))
    gather = jax.jit(jax.shard_map(
        lambda shard: jax.lax.all_gather(shard, axis, tiled=True), mesh=mesh,
        in_specs=P(axis), out_specs=P(), axis_names={axis}, check_vma=False))
    seconds = samples(gather, source, warmup=2, repeats=5)
    received = min(from_every_process((ways - 1) / ways * count * 4 / statistics.median(seconds)))
    if record is not None and jax.process_index() == 0:
        record.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(record, json.dumps({"topology": topology, "bytes_per_second": received}))
    return received


def link_figures(bandwidths: Mapping[tuple[Mesh, str], float | None]) -> dict[str, JSON]:
    """The link figures a checkpoint's rung keeps (`Trainer._checkpoint_rung`): each figure of
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
    compilation cache, or nowhere when the process keeps no cache."""
    cache = jax.config.jax_compilation_cache_dir
    if not cache:
        return None
    return Path(cache) / "dew-links" / f"{hashlib.sha256(topology.encode()).hexdigest()}.json"


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
    ramp stage's, for a later stage, an hour in. A ramp reads each stage's
    batch whole; an accumulation window pools its stages' batches as it pools
    any, but for a loss that replays its window's batches (`Trainer._prepared`).
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


def microbatch(batch: Batch, part: jax.Array, split: int) -> Batch:
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
frozen tree (`dew.state.apart`, `Trainer._jitted`), before the part of the step it runs."""


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
    return tuple(sharding is not None for sharding in tree.flatten_up_to(arguments[STEP_BATCH]))


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

    # A copy of an array already on the devices: device_put can reuse a
    # buffer the target shares with the source, which donation would free.
    owned = jax.tree.map(lambda leaf: leaf if id(leaf) in shared or not isinstance(leaf, jax.Array)
                         else leaf.copy(), held)
    placed = jax.device_put(owned, jax.tree_util.tree_map_with_path(
        lambda path, leaf: shared.get(id(leaf)) or variable(path, leaf) or replicated, held))
    keep = {id(placed_leaf) for leaf, placed_leaf in zip(jax.tree.leaves(held), jax.tree.leaves(placed),
                                                         strict=True) if id(leaf) in shared}
    taken = jax.tree_util.tree_map_with_path(
        lambda path, leaf: None if variable(path, leaf) is None or id(leaf) in keep else leaf, placed)
    return taken, jax.tree.map(lambda one, leaf: leaf if one is None else None, taken, placed,
                               is_leaf=lambda leaf: leaf is None)


def taking_over[Held, Result](function: Callable[[Held, jax.Array], Result], taken: Held, kept: Held,
                              shardings: Placement[Result]) -> Callable[[jax.Array], Result]:
    """`function` jitted onto `shardings` over the held tree `placed_held`
    splits: `taken` donated, for the outputs to take its buffers over, and
    `kept` not, as no output takes those and JAX only warns of a donation it
    cannot use. They are freed as the caller drops them."""

    def joined(taken: Held, kept: Held, key: jax.Array) -> Result:
        return function(jax.tree.map(lambda one, other: other if one is None else one, taken, kept,
                                     is_leaf=lambda leaf: leaf is None), key)

    return functools.partial(jax.jit(joined, out_shardings=shardings, donate_argnums=0), taken, kept)


def traced_start(initial_state: Callable[[Initializer, jax.Array], TrainState], initializer: Initializer,
                 key: jax.Array) -> tuple[TrainState, dict[int, int]]:
    """The shapes of the fresh state `initial_state` builds, and which of its frozen leaves are the
    initializer's arrays as they are: each such leaf's index among the frozen tree's leaves, to the
    index of the initializer's leaf it is. An init that returns a held leaf as it is, by reference,
    passes it through (`Objective.held_variables`); one trace finds both."""
    through: dict[int, int] = {}

    def traced(initializer: Initializer, key: jax.Array) -> TrainState:
        state = initial_state(initializer, key)
        inputs = {id(leaf): index for index, leaf in enumerate(jax.tree.leaves(initializer))}
        frozen = jax.tree.leaves(apart(state)[1])
        through.update({index: inputs[id(leaf)] for index, leaf in enumerate(frozen) if id(leaf) in inputs})
        return state

    return jax.eval_shape(traced, initializer, key), through


def placed_start(initial_state: Callable[[Initializer, jax.Array], TrainState], initializer: Initializer,
                 key: jax.Array, abstract: TrainState, shardings: Placement[TrainState],
                 through: Mapping[int, int], mesh: Mesh) -> TrainState:
    """A fresh state on `mesh`, built by `initial_state` from the objective's held tree (`placed_held`).
    The frozen leaves the objective's init passes through as they are (`through`, `traced_start`),
    which no step donates (`apart`), are the held arrays themselves, placed where the state keeps
    them, the same arrays where they are placed already: the run and the objective then hold one
    frozen tree between them. One program builds the rest, a frozen leaf the init builds among it,
    taking the other held arrays over."""
    moving, frozen = apart(shardings)
    if frozen is None or not through:
        taken, kept = placed_held(initializer, abstract.variables, shardings.variables, mesh)
        return taking_over(initial_state, taken, kept, shardings)(key)
    inputs = jax.tree.leaves(initializer)
    where, structure = jax.tree.flatten(frozen)
    # Each passed leaf is placed once, where the state keeps it, whatever path the init moves it to.
    taken, kept = placed_held(initializer, abstract.variables, shardings.variables, mesh,
                              shared={id(inputs[given]): where[index] for index, given in through.items()})
    placed = jax.tree.leaves(jax.tree.map(lambda one, other: other if one is None else one, taken, kept,
                                          is_leaf=lambda leaf: leaf is None))
    built = [index for index in range(len(where)) if index not in through]

    def rest(tree: Initializer, draw: jax.Array) -> tuple[TrainState, tuple[jax.Array, ...]]:
        state, own = apart(initial_state(tree, draw))
        leaves = jax.tree.leaves(own)
        return state, tuple(leaves[index] for index in built)

    state, made = taking_over(rest, taken, kept, (moving, tuple(where[index] for index in built)))(key)
    drawn = dict(zip(built, made, strict=True))

    def leaf(index: int) -> jax.Array:
        if index in drawn:
            return drawn[index]
        # A held leaf an init passes through at two places of other shardings is placed at the second too.
        held = placed[through[index]]
        return held if held.sharding == where[index] else jax.device_put(held, where[index])

    return together(state, jax.tree.unflatten(structure, [leaf(index) for index in range(len(where))]))
