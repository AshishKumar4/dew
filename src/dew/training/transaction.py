"""The objective transaction, independent of where its numerical phases execute.

A resident step traces these phases together. A host-master step compiles the
same reductions and commit on CPU and crosses an eager transport boundary to
realize a batch on the accelerator. Only that orchestration differs: pooling,
replay snapshots, optimizer admission, effects, EMA and the three clocks have
one implementation. A pullback belongs to the current attempt, not TrainState.

The host-master commit is compiled over what it moves. The frozen collection
never changes and, resident beside the accelerator, is not the CPU's to copy:
it is held out of the compiled commit and put back on the state it returns,
the same arrays step after step.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterator, Mapping
from typing import Generic

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import dynamic_scale as dynamic_scale_lib
from typing_extensions import TypeVar

from dew.averaging import ema_update
from dew.batch import EXAMPLE_ROWS, Batch, Variables, dicts, leaves_by_path, merge
from dew.data.dataset import rows_of
from dew.gauss_newton import solves_equations
from dew.held_updates import holds_back, input_taken, refuse_untold, update_applied
from dew.nn.inspection import (
    PROBES,
    Watched,
    cotangent_summaries,
    discovering,
    flattened,
    kept,
    probes_of,
    unscaled,
)
from dew.nn.kernels.options import owned_program
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import auto_part, rows_spec
from dew.objectives.base import (
    EXAMPLE_STREAM,
    Aux,
    Evaluated,
    Objective,
    Ratio,
    Recorded,
    Step,
    recorded_metrics,
    with_projected,
)
from dew.optim import clips_qk, stepped
from dew.state import Accumulation, TrainState, apart, together
from dew.training.compiled_program import gradients_read
from dew.training.inspection import Inspection, Summaries, stacked, summaries
from dew.training.memory import head_fused_of, head_tile_of
from dew.training.model_flops import _TokenCount, step_counts
from dew.training.narrow import forward_variables, narrowed
from dew.training.staging import fetched, in_device, staged

Loss = TypeVar("Loss", default=Ratio | jax.Array | float)
Effects = TypeVar("Effects", default=None)


@dataclasses.dataclass(frozen=True)
class PassesProgress[L, E]:
    """A custom training step (`Trainer(step=)`) that passes the run's progress to a model that reads
    it (`dew.nn.progress`, `Progress(state.updates)`), as the objective's own loss does: `make`, its
    factory, wrapped. A custom step not so declared refuses such a model, which it would otherwise
    train at the run's end."""

    make: Callable[[Objective[L, E], optax.GradientTransformation],
                   Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]]

    def __call__(self, objective: Objective[L, E], optimizer: optax.GradientTransformation
                 ) -> Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]:
        return self.make(objective, optimizer)


def refuse_unpassed_progress[L, E](objective: Objective[L, E], *, passes: bool) -> None:
    """Refuse a custom training step that does not say it passes the run's progress (`passes`,
    `PassesProgress`) over a module of `objective`'s that reads it (`Objective.progress_readers`): it
    would train the module at the run's end."""
    for module in () if passes else objective.progress_readers()[:1]:
        raise ValueError(f"a custom training step (Trainer(step=)) passes a {type(module).__name__}, which "
                         "reads the run's progress (dew.nn.progress), none unless it says it does: pass "
                         "Progress(state.updates) as the model's progress= and wrap the step's factory in "
                         "dew.training.PassesProgress, or it trains at the run's end")


@struct.dataclass
class Committed:
    """Telemetry from one admitted optimizer commit, not objective metrics or checkpoint state.

    `norm` is optax.global_norm of the native-dtype gradient passed to the
    optimizer, before its transforms such as clipping. Accumulation takes
    the pooled gradient at the window's commit, never each microbatch's
    norm. `tokens` is the real tokens the commit's update trained on, the
    window's accepted microbatches', as two uint32 words. Rejected attempts
    and partial windows contribute zero to all three; the loop sums them on
    device until its logging tick.
    """
    norm: jax.Array
    count: jax.Array
    tokens: jax.Array
    inspected: Summaries | None = None
    """A sampled step's summaries as one vector (`dew.training.inspection.Summaries`), zero where it
    committed nothing; None in a step that samples nothing."""


def _chosen(take, taken, kept):
    """`taken` where `take` holds and `kept` elsewhere, leaf by leaf."""
    return jax.tree.map(lambda new, old: jnp.where(take, new, old), taken, kept)


def cleared(acc: Accumulation) -> Accumulation:
    """`acc` emptied for its next window: its sums zero, its QK maxima at their identity."""
    return dataclasses.replace(jax.tree.map(jnp.zeros_like, acc), qk_stats=jax.tree.map(
        lambda x: jnp.full_like(x, -jnp.inf) if jnp.issubdtype(x.dtype, jnp.inexact) else jnp.zeros_like(x),
        acc.qk_stats))


def dropped_window(state, size: int):
    """`state` with its window of `size` microbatches dropped: its accepted microbatches forgotten and
    the next taking its first slot, as a rejected attempt (`Transaction.finish_attempt`) or a batch
    ramp's new stage (`Trainer`'s runner) leaves it, so a committed update pools one window of one
    stage's batches, all drawn under one key (`example_fields`). The window's gradient sum stays in
    place, unread, as a commit leaves it (`prepare_attempt`); the mutable collections its microbatches
    wrote keep what they wrote, as torch's AMP keeps a skipped update's running statistics."""
    if state.accumulation is None:
        return state
    microstep, window = window_dropped(state.microstep, state.accumulation, size)
    return dataclasses.replace(state, microstep=microstep, accumulation=window)


def window_dropped(microstep: jax.Array, acc: Accumulation, size: int) -> tuple[jax.Array, Accumulation]:
    """`dropped_window`'s two fields: the microstep back at its window's start, and the window cleared but
    for its gradient sum. A state's other fields go round it, as a compiled program over them alone
    leaves a host-master state's resident frozen weights where they are."""
    return microstep - microstep % size, dataclasses.replace(
        cleared(dataclasses.replace(acc, gradient=None)), gradient=acc.gradient)


def _only_when(due, taken, kept, operand):
    """`taken(operand)` where `due` holds, else `kept(operand)`, running `taken` then alone.

    XLA's conditional simplifier judges a branch by its ops' kinds, not their
    size: one of elementwise ops and reductions, an optimizer update among
    them, it runs on every call and selects (`TryRemoveConditional`). In a
    Qwen3.5-0.8B fine-tune at 8 x 3072, bf16, that ran the whole AdamW
    update on every microbatch, about 44 ms of a 731 ms step on an A100. The
    barrier, an op it counts as expensive, keeps the branch a branch.
    """
    return jax.lax.cond(due, lambda current: jax.lax.optimization_barrier(taken(current)), kept, operand)


def outcome(current: TrainState, advanced: TrainState, loss: jax.Array, aux: Aux, committed: jax.Array | None,
            batch: Batch, stepped: jax.typing.ArrayLike, consumed: jax.Array) -> tuple:
    """What a compiled step returns (`Trainer.compile`) of the transaction it ran from `current` on
    `batch`, wherever it ran: the advanced state, its step counting `stepped` and its clock the source
    examples its microbatch `consumed` (`TrainState.consumed`, where the run keeps one), the loss, the
    metrics with the batch's counts, and whether the loss is finite, the microbatch accepted and the
    window committed."""
    metrics = {**aux.metrics, **step_counts(batch)}
    if aux.record is not None:
        assert "evaluated" in aux.record, "a recording transaction stacks its evaluations (`recorded`)"
        metrics |= recorded_metrics(aux.record)
    clock = None if current.consumed is None else (current.consumed + consumed).astype(current.consumed.dtype)
    return (dataclasses.replace(advanced, step=current.step + stepped, consumed=clock), loss, metrics,
            jnp.isfinite(loss), advanced.microstep > current.microstep, committed)


def probed_loss(objective: Objective, variables: Variables, batch: Batch, step: Step):
    """The objective's loss statistics with its parameters and its watched outputs' cotangent probes
    (`dew.nn.inspection.tapped`), their pullback giving the parameters' gradient and the probes', and the
    loss's aux. The probes are laid out as the objective's model calls nest their modules, which a trace
    of its loss that discovers them (`discovering`) sows, nothing computed; a probe no model call reads,
    its collection dropped on the way, is refused."""
    with discovering():
        _, discovered = jax.eval_shape(lambda: objective._loss(variables, batch, step))
    probes = probes_of(discovered.inspected or {})
    if not probes:
        raise ValueError("no watched module's output reaches the objective's loss for its cotangent to be "
                         "summarized; name modules its model calls run (Inspection.modules)")

    def loss(trainable, probes):
        return objective._loss({**variables, "params": trainable, PROBES: probes}, batch, step)
    stats, back, aux = jax.vjp(loss, variables["params"], probes, has_aux=True)
    unread = sorted(set(leaves_by_path(probes)) - set(leaves_by_path(probes_of(aux.inspected or {}))))
    if unread:
        raise ValueError(f"the objective's model calls do not pass the step's cotangent probes "
                         f"(variables[{PROBES!r}]) to the watched modules at {unread}; pass every "
                         f"collection of the variables the objective is given through to its model")
    return stats, back, aux


def watched_activations(watched: Watched, aux: Aux) -> dict[str, jax.Array]:
    """The activation summaries a sampled step's latest microbatch sowed (`Aux.inspected`), by their
    logged names (`dew.nn.inspection.flattened`). Refused where the objective's model call opened no
    `inspected` collection (`Step.sown`), or where no module of a watched kind ran in it, either of which
    would log nothing a run asked for."""
    if aux.inspected is None:
        raise ValueError("the run watches activations (Inspection.modules), which its objective's model "
                         "calls do not return: they open Step.sown() and return what it sowed "
                         "(Aux.inspected)")
    found = flattened(aux.inspected, watched.modules)
    if not found:
        raise ValueError(f"no module the watched globs {list(watched.modules)} name runs in the objective's "
                         f"model; name them by class (`*Attention`, `MLP`) or path (`layers_3/*`)")
    return found


def evaluation_of(aux: Aux) -> Evaluated | None:
    """The record one evaluation's loss returned (`Objective._loss`), None where it recorded nothing."""
    record = aux.record
    assert record is None or "evaluated" not in record, "an evaluation's record is its own, unstacked"
    return record


def recorded(evaluations: list[tuple[Evaluated, jax.Array]], ran: jax.Array) -> Recorded:
    """A recording step's record (`Recorded`) of its `evaluations`, each with the places its slot and
    the column its loss saw give its rows (`Transaction._places`), stacked on a first axis: the
    attempt's own evaluation and then the window's other microbatches as the commit replayed them, each
    in its slot, `ran` saying which ran. `column_ok` holds where each row's place is its expected one,
    so rows a layout places apart from their column, or a microbatch read under another's places, fail
    it."""
    values = jax.tree.map(lambda *leaves: jnp.stack(leaves),
                          *[evaluated["values"] for evaluated, _ in evaluations])
    index = jnp.stack([evaluated["example_index"] for evaluated, _ in evaluations])
    expected = jnp.stack([places for _, places in evaluations])
    return Recorded(values=values, example_index=index, column_ok=jnp.all(index == expected, axis=1),
                    evaluated=ran)


COMMIT_FIELDS = ("opt_state", "ema")
"""The state's fields a resident step reads only as it commits, past its gradient: the optimizer state, and
the EMA but where the objective reads it in its forward (`Transaction.fetch`)."""


def optimizer_update(optimizer: optax.GradientTransformation, gradient, opt_state, params, *, qk=None,
                     statistics=None, progress=None) -> tuple:
    """`optimizer`'s update of `gradient` from `opt_state` over `params`, called as a step's commit calls
    it (`Transaction.commit`) and as the planner traces it (`leaf_local`): an update taking extra arguments
    is handed the attention logits' maxima (`qk`), the run's `progress` where it has one (a run bounded in
    examples), and its loss's `statistics` only where it solves the update's system
    (`dew.gauss_newton`)."""
    if not isinstance(optimizer, optax.GradientTransformationExtraArgs):
        return optimizer.update(gradient, opt_state, params)
    clocked = {} if progress is None else {"progress": progress}
    solved = {"statistics": statistics} if solves_equations(opt_state) else {}
    return optimizer.update(gradient, opt_state, params, qk_stats=qk, **solved, **clocked)


def leaf_local(optimizer: optax.GradientTransformation, gradients, state, *, progressed: bool) -> bool:
    """Whether `optimizer`'s update of `gradients` from `state`, over parameters of their shapes, reads each
    leaf of `gradients` into results that read no other gradient's leaf: no reduction, product or
    concatenation across them. Traced as the commit calls it (`optimizer_update`), handed no statistics of
    attention logits and, `progressed`, the run's progress; one solving its objective's system
    (`dew.gauss_newton.solves_equations`) reads every leaf into each result. An equation running a program
    of its own (a call, a loop) is read as reading every operand into each result."""
    if solves_equations(state):
        return False

    def update(gradients, state, params, progress):
        return optimizer_update(optimizer, gradients, state, params,
                                progress=progress if progressed else None)

    program = jax.make_jaxpr(update)(gradients, state, gradients, jax.ShapeDtypeStruct((), np.float32)).jaxpr
    leaves = len(jax.tree.leaves(gradients))
    read = [frozenset([index]) if index < leaves else frozenset() for index in range(len(program.invars))]
    return all(len(leaves_read) <= 1 for leaves_read in gradients_read(program, read))


def pooled_by_mean(stats_shape) -> bool:
    """Whether a window pools its microbatches' statistics `stats_shape` as a mean, a `Ratio`'s or a
    scalar's (`Transaction.shared`), summing their gradients as it goes; any other it replays."""
    return isinstance(stats_shape, (Ratio, jax.ShapeDtypeStruct))


def consumed_where_made[LossT, EffectsT](objective: Objective[LossT, EffectsT],
                                          optimizer: optax.GradientTransformation, state: TrainState,
                                          window: int, *, shared: bool, progressed: bool) -> bool:
    """Whether the step reads each gradient where it is made into the buffers its state donates, so it
    holds none beside another (`footprint.gradient_peak`'s `consumed`): with the state it commits
    (`COMMIT_FIELDS`) in device memory, a window of several microbatches that pools them by mean (`shared`,
    `pooled_by_mean`) sums each into its accumulator's gradient leaf in device memory, where one that
    replays them keeps their gradients for the replay; and a window of one updates each by `optimizer`,
    where its update reads each gradient into its own leaf's results alone (`leaf_local`, `progressed` the
    run handing it its progress). A dynamically scaled loss commits only once every gradient is known
    finite, a committed state fetched from the host is fetched behind one barrier over every gradient
    (`Transaction.past`), and an update that reads them together (clipping by the global norm) holds them
    all until it has read the last. A head that takes its gradient in the backward a tile at a time
    (`head_tile`, unfused) has XLA hold the layers' gradients to the step's end: Qwen3.5-4B's step at 2 x
    2048 tokens under 'minimal', compiled for an H100 at 59.4 GiB, ran every layer's weight gradient and
    update after the whole backward, 20.1 GB of temporaries against the 10.3 derived, where the whole
    logits' step ran them beside the backward; on an H100 itself the 13 layers its backward reached last
    waited (row 3653)."""
    if head_tile_of(objective) is not None and not head_fused_of(objective):
        return False
    if state.scale is not None or not in_device([getattr(state, field) for field in COMMIT_FIELDS]):
        return False
    if window > 1:
        return shared and state.accumulation is not None and in_device(state.accumulation.gradient)
    params, opt_state = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
                                     (state.variables["params"], state.opt_state))
    return leaf_local(optimizer, params, opt_state, progressed=progressed)


def with_ema(params: Variables, ema: Variables | None) -> Variables | None:
    """The variables tree with the averaged leaves in place of the live ones."""
    return None if ema is None else merge(params, ema)


def write_back(params: Variables, variables: Variables | None) -> Variables:
    """Replace nonparameter collections whole; the optimizer owns params. A collection comes back in
    the state's dicts whichever Flax container the objective's `apply` wrote it in (`dew.batch.dicts`)."""
    if variables is None:
        return params
    variables = dicts(variables)
    if "params" in variables:
        raise ValueError("Aux.variables cannot carry the params collection; the optimizer owns it")
    for name, collection in variables.items():
        if name not in params:
            raise ValueError(f"Aux.variables names {name!r}, which is not a collection "
                             f"of the objective's tree {sorted(params)}")
        if jax.tree.structure(collection) != jax.tree.structure(params[name]):
            raise ValueError(f"Aux.variables[{name!r}] does not have the collection's structure")
    return {**params, **variables}


def _unscale(gradient: jax.Array, factor: jax.typing.ArrayLike) -> jax.Array:
    """Divide a gradient by the loss scale it was computed under.

    The division happens in fp32 or wider, so a bf16 gradient does not round
    twice on the way back to its true magnitude.
    """
    dtype = at_least_fp32(gradient.dtype)
    return gradient.astype(dtype) / jnp.asarray(factor, dtype)


def placed_as(tree, placement):
    """`tree` constrained to where `placement` places each of its leaves, over the Auto axes of the mesh in
    context (`auto_part`): an Explicit axis stays where a leaf's type places it, a Manual one is a
    `shard_map`'s own, and no mesh, or one of no Auto axis, leaves `tree` as it is."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or not mesh.auto_axes:
        return tree
    return jax.tree.map(lambda leaf, sharding: jax.lax.with_sharding_constraint(
        leaf, auto_part(sharding.spec, mesh)), tree, placement)


def _descending(gradient: jax.Array, parameter: jax.Array) -> jax.Array:
    """`gradient` in its parameter's dtype, as optax descends along it. JAX's gradient of a real
    loss in a complex parameter is the conjugate of its steepest ascent."""
    return (jnp.conj(gradient) if jnp.iscomplexobj(gradient) else gradient).astype(parameter.dtype)


def _all_finite(tree) -> jax.Array:
    finite = jnp.asarray(a=True)
    for leaf in jax.tree.leaves(tree):
        finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def compact_qk(tree):
    """Reduce every sowed `max_logits` leaf over its rows, keeping the heads, and a layer's geometry
    (`kv_heads`, `head_dim`, `qk_nope`) to the one count it is.

    The clip reads each head's strongest logit alone, so a retained window
    holds one row per layer instead of the batch's. A pipelined layer sows
    its geometry once a microbatch (`StackView`), the same count each time,
    and the clip reads it as one.
    """
    def compact(path, leaf):
        is_maximum = any(isinstance(entry, jax.tree_util.DictKey) and entry.key == "max_logits"
                         for entry in path)
        if is_maximum:
            return jnp.max(leaf, axis=-2, keepdims=True) if leaf.ndim > 1 else leaf
        return jnp.max(leaf) if leaf.ndim else leaf
    return jax.tree_util.tree_map_with_path(compact, tree)


def loss_scale(dynamic_scale: bool | dynamic_scale_lib.DynamicScale) -> dynamic_scale_lib.DynamicScale | None:
    """The dynamic loss scale a trainer's `dynamic_scale` asks for: Flax's at its defaults for True,
    the one given (its growth interval and factors), or none."""
    if isinstance(dynamic_scale, dynamic_scale_lib.DynamicScale):
        return dynamic_scale
    return dynamic_scale_lib.DynamicScale() if dynamic_scale else None


def _advance_scale(scale: dynamic_scale_lib.DynamicScale, finite: jax.Array):
    # Flax's growth comparison uses the incoming streak, including at restart.
    grow = scale.fin_steps == scale.growth_interval
    increased = jnp.minimum(scale.scale * scale.growth_factor, jnp.finfo(jnp.float32).max)
    decreased = scale.scale * scale.backoff_factor
    if scale.minimum_scale is not None:
        decreased = jnp.maximum(decreased, scale.minimum_scale)
    return dataclasses.replace(
        scale,
        scale=jnp.where(finite, jnp.where(grow, increased, scale.scale), decreased),
        fin_steps=jnp.where(grow | ~finite, 0, scale.fin_steps + 1),
    )


@dataclasses.dataclass(frozen=True)
class Realization(Generic[Loss, Effects]):
    """What one objective evaluation produced, with the pullback that undoes it.

    `pullback` closes over the forward's residuals, so it belongs to the
    attempt that made it and never travels on a TrainState.
    """
    stats: Loss
    aux: Aux[Effects]
    pullback: Callable[[Loss], Variables]
    probed: Callable[[Loss], tuple[Variables, Variables]] | None = None
    """Where the step summarizes its watched outputs' cotangents (`Watched.cotangents`), the pullback
    that also gives its probes' gradient (`dew.nn.inspection.tapped`), one backward for both."""


@struct.dataclass
class PendingAttempt(Generic[Loss]):
    """The window after one microbatch, before the commit decides on it.

    `fill` is the slot this microbatch took and `due` whether it closed the
    window. `active` is whether the reduction found any statistical support,
    `local_finite` whether this microbatch alone produced finite values, and
    `candidate` the accumulation the state keeps if the attempt is admitted.
    """
    fill: jax.Array
    due: jax.Array
    active: jax.Array
    loss: jax.Array
    local_finite: jax.Array
    replay_required: jax.Array
    pooled: Loss
    effects: tuple[jax.Array, ...]
    qk: Variables | None
    candidate: Accumulation | None
    gradient: Variables


_GROUPS: contextvars.ContextVar[tuple[int, int | jax.Array] | None] = contextvars.ContextVar(
    "example_groups", default=None)


@contextlib.contextmanager
def example_groups(groups: int, group: int | jax.Array) -> Iterator[None]:
    """Within the block, a step's microbatch is group `group`'s share of `groups` equal ones (DiLoCo's,
    `dew.training.diloco`): its examples are placed and counted among the whole step's (`example_fields`),
    each group's rows the `group`th share of them, as one run of every row places them. `group` is the
    traced axis index within a step, and a host integer where a plan reads one group's shapes."""
    token = _GROUPS.set((groups, group))
    try:
        yield
    finally:
        _GROUPS.reset(token)


def example_places(slot, split: int, rows: int, column: jax.Array | None = None, part=None,
                   first: jax.typing.ArrayLike = 0, length: int | None = None) -> jax.Array:
    """The places among an update's examples of the `rows` rows of the microbatch in window slot `slot`,
    of steps read in `split` microbatches each: the step's place in the window times a step's rows, plus
    each row's place among its step's records, its `column` (`EXAMPLE_ROWS`) where the batch holds one,
    else its row in the step's batch, every `split`th from row `part` (`microbatch`; by default
    `slot % split`).

    A window pools its steps' microbatches in the order it accepts them. A dynamic loss scale's
    rejection drops its window (`dropped_window`), and the next takes its first slot under a key that
    folds in the rejection (`example_fields`), whatever part of its step it reads: a window's slots
    are consecutive parts, so each step's places stay distinct, and no two examples of an update
    draw alike."""
    step = slot // split
    if column is None:
        part = slot % split if part is None else part
        # A group's share (`example_groups`): its `length` rows from the step's row `first`.
        column = part + split * (first + jnp.arange(rows if length is None else length, dtype=jnp.int32))
    return (step * rows * split + column).astype(jnp.int32)


def example_fields(state, slot, split: int, batch, slots: int = 1, part=None, rejected=0,
                   microstep=None) -> dict:
    """The `Step` fields of the microbatch in window slot `slot` of the update `state` commits next, of
    `slots` microbatches, its steps read in `split` microbatches each, this the `part`th, after the run
    rejected `rejected` attempts: the update's key (`Step.example_key`), which folds in the rejections,
    so a window that follows a dropped one draws afresh, each row's place among the update's
    examples (`example_places`), placed as the batch's rows are, so each device draws its own rows'
    examples alone, and how many examples the update pools. A batch with no rows has no examples."""
    rows = local = rows_of(batch)
    if not rows:
        return {}
    groups, group = _GROUPS.get() or (1, 0)
    rows *= groups
    key = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(state.key, EXAMPLE_STREAM),
                                                state.updates), rejected)
    if holds_back(state.opt_state):
        # Each input the optimizer takes, holds or skips draws apart: by the windows accepted before it,
        # read off the microbatches accepted (`microstep`, the state's where none is given) over a
        # window's (`slots`, its parts counted), so a window's slots and parts share its draw however
        # its batch is split.
        accepted = state.microstep if microstep is None else microstep
        key = jax.random.fold_in(key, accepted // slots)
    index = example_places(slot, split, rows, batch.get(EXAMPLE_ROWS), part, group * local, local)
    return {"example_key": key, "example_index": placed_as_rows(index, batch), "example_count": slots * rows}


def attempt_step(state, variables, ema, batch, *, split: int, slots: int, attempt, part, slot,
                 microstep=None, record: bool = False, watched: Watched | None = None) -> Step:
    """The `Step` the loss of attempt `attempt` of the update `state` commits next reads: the step it
    counts, its microbatches accepted (`microstep`, the state's where none is given) over the `split`
    each step is read in; the attempt's own key; the averaged weights `ema` beside `variables`; the
    updates committed, which a model whose computation moves over a run reads (`Objective.progress`);
    and the microbatch's examples, its window `slot` of `slots`, its `part`, after the attempts the run
    rejected (`example_fields`). An attempt and its replay read it, and so do the traces that shape and
    narrow the step (`dew.training.footprint.traced_loss`, `dew.training.narrow.narrow_copies`), which
    record nothing and watch nothing."""
    accepted = state.microstep if microstep is None else microstep
    return Step(accepted // split, jax.random.fold_in(state.key, attempt), with_ema(variables, ema),
                committed=state.updates, record=record, qk_stats=clips_qk(state.opt_state),
                equations=solves_equations(state.opt_state), watched=watched,
                **example_fields(state, slot, split, batch, slots, part, attempt - accepted, microstep))


def placed_as_rows(index: jax.Array, batch) -> jax.Array:
    """`index`, an entry for each of `batch`'s rows, placed as the batch's rows are, so each device
    reads its own rows' entries alone."""
    like = next(leaf for leaf in jax.tree.leaves(batch) if np.ndim(leaf))
    placed = rows_spec(like, 1)
    return index if placed is None else jax.sharding.reshard(index, placed)


class Transaction:
    """Turn one or more microbatches into one optimizer update.

    Pooling, replay, admission, effects, EMA and the three clocks are written
    once here. `step` composes them for a resident run or for a host-master
    run; only where each phase executes differs.
    """
    def __init__(self, objective, optimizer, accumulation: int, shapes, copies=None, split: int = 1,
                 record: bool = False, resting: Mapping[str, object] | None = None,
                 inspection: Inspection | None = None, examples: int | None = None, placed=None):
        """Hold the objective, the optimizer and the shapes a step traces for.

        `accumulation` is the microbatches a window pools, and `split` the
        microbatches each step's batch is read in (`Trainer.compile`), one
        attempt each. `shapes` is the traced result of the objective's loss,
        statistics and `Aux`. Statistics that are a `Ratio` or a bare scalar
        pool into one shared mean, which the window can sum and never has to
        replay. `copies` names the parameters the forward reads through the
        state's narrow copies, which the commit rewrites (`dew.training.narrow`). `record` makes each
        evaluation return what its `Step.sample` and `Step.boundary` calls produced (`recorded`).
        `resting` places the fields of `COMMIT_FIELDS` a resident step's state keeps in pinned host memory,
        each by its placement there: the step reads them only once its gradient is ready, as the commit
        does, so the device holds them through the update and not through the forward and backward, but
        for an EMA the objective reads in its forward. Fetched at the step's start, an A100 held them
        through both: a plain Adam step's peak rose from 2.483 GB to 3.020 GB with its state in host
        memory, and fetched past the gradient it is 2.483 GB again (node receipt TL-a100-offload-1338-c,
        hostmem.json). Fetched whole there, XLA:GPU held the whole state for the update, so keeping it on
        the host freed nothing at a step's peak (a Dew decoder's peaked at 8.427 GB either way): the
        commit takes them a chunk at a time (`past`, `dew.training.staging`). `placed` is the parameters'
        placement, which the commit's gradient is constrained to (`placed_as`). `examples`
        is a run's bound in source examples, where it counts no updates ahead: each update hands its
        optimizer the run's progress, the share of them its state has consumed (`TrainState.consumed`),
        which a schedule over the run reads (`dew.optim.ScheduleBase.progressed`).
        """
        self.resting = dict(resting or {})
        self.placed = placed
        self.record = record
        # What a sampled step summarizes at its commit (`Committed.inspected`); None in every other step.
        self.inspection = inspection
        self.watched = None if inspection is None else inspection.watched
        self.bound = examples
        self.objective = objective
        self.optimizer = optimizer
        self.copies = copies or {}
        self.size = accumulation
        self.split = split
        stats_shape, self.aux_shape = shapes
        self.shared = pooled_by_mean(stats_shape)
        self.stats_tree = jax.tree.structure(stats_shape)
        self.effects_tree = jax.tree.structure(self.aux_shape.effects)

    def fetched(self, state: TrainState) -> TrainState:
        """`state` with its host-resident fields (`resting`) brought to the device whole, for an EMA the
        objective reads in its forward."""
        return dataclasses.replace(state, **{field: fetched(getattr(state, field), placement)
                                             for field, placement in self.resting.items()})

    def past(self, finish: Callable, state: TrainState, gradient, *arguments) -> tuple:
        """`finish(state, *arguments)`, the step's commit and what follows it, its host-resident fields
        (`resting`) brought to the device a chunk at a time past `gradient` (`staged`): the first chunk
        passes an optimization barrier with it, so the device takes none of them before it is ready, and
        the commit's new fields go back to host memory chunk by chunk. Where nothing rests on the host the
        state is the commit's as it is: with the barrier there too, a step that fit by a byte past its own
        rung found no rung (tests/test_narrow.py, at ada6c5686)."""
        if not self.resting:
            return finish(state, *arguments)

        def finished(fields, held, *arguments):
            advanced, *outputs = finish(dataclasses.replace(held, **fields), *arguments)
            return ({field: getattr(advanced, field) for field in fields},
                    (dataclasses.replace(advanced, **dict.fromkeys(fields)), *outputs))

        fields = {field: getattr(state, field) for field in self.resting}
        kept, (advanced, *outputs) = staged(finished, fields, self.resting,
                                            (dataclasses.replace(state, **dict.fromkeys(fields)), *arguments),
                                            gradient)
        return dataclasses.replace(advanced, **kept), *outputs

    def realize(self, variables: Variables, batch: Batch, step_info: Step) -> Realization:
        """Evaluate the objective's loss and keep the pullback of that trace.

        This is the resident implementation, differentiating the whole
        parameter tree where the state lives. `HostExecution.realize` is the
        other one, and `step` takes either.
        """
        watched = step_info.watched
        if watched is None or not watched.cotangents:
            def loss(trainable):
                return self.objective._loss({**variables, "params": trainable}, batch, step_info)
            stats, back, aux = jax.vjp(loss, variables["params"], has_aux=True)
            return Realization(stats, aux, lambda cotangent: back(cotangent)[0])
        stats, back, aux = probed_loss(self.objective, variables, batch, step_info)
        return Realization(stats, aux, lambda cotangent: back(cotangent)[0], back)

    def local_reduction(self, stats, factor):
        """Reduce one microbatch's statistics to its loss and its cotangent.

        The cotangent is seeded with `factor`, the loss scale, so the
        gradient the pullback returns is the scaled one `unscaled` divides.
        """
        loss, back = jax.vjp(lambda s: self.objective.reduce_loss(s)[0], stats)
        return loss, back(jnp.asarray(factor, loss.dtype))[0]

    def prepare_attempt(self, state, stats, aux, loss, gradient, closing=None):
        """Pool this microbatch into the window and weigh up the commit.

        Returns a `PendingAttempt`: the pooled statistics and gradient,
        whether this attempt closes the window, and whether the pooled loss
        needs a replay. A shared mean pools its gradient by mass here and
        never replays. A general statistics tree reduces only once pooled, so
        its gradient has to be recomputed under the pooled cotangent. A
        `closing` attempt, the last of a run whose stream ends inside the
        window (`dew.batch.CLOSES_WINDOW`), closes it with the microbatches it
        holds, its mean theirs.
        """
        if aux.qk_stats is None and clips_qk(state.opt_state):
            raise ValueError(
                f"{type(self.objective).__name__} returns no attention logit maxima (Aux.qk_stats) though "
                "its step asks for them (Step.qk_stats), and the run's optimizer clips by them (a QK-Clip, "
                "muonclip's): its attention would train unclipped")
        previous = state.accumulation
        fill = state.microstep % self.size
        due = (fill + 1) == self.size
        if closing is not None:
            due = due | closing
        local_finite = _all_finite((loss, stats, gradient, aux.variables, aux.effects))
        local_ok = jnp.asarray(a=True) if state.scale is None else local_finite
        qk = compact_qk(aux.qk_stats)
        effects = tuple(jax.tree.leaves(aux.effects))
        candidate, pooled = previous, stats
        replay_required = jnp.asarray(a=False)
        if previous is None:
            _, active = self.objective.reduce_loss(stats)
        else:
            effects = tuple(a + b for a, b in zip(previous.effects, effects, strict=True))
            qk = jax.tree.map(jnp.maximum, previous.qk_stats, qk)
            if self.shared:
                prior_mass, prior_gradient = previous.mass, previous.gradient
                if prior_mass is None or prior_gradient is None:
                    raise ValueError("checkpoint accumulator does not match shared-mean statistics")
                if isinstance(stats, Ratio):
                    mass, total = stats.mass, stats.total
                elif isinstance(stats, (jax.Array, float, int)):
                    mass, total = jnp.asarray(1.), jnp.asarray(stats)
                else:
                    raise TypeError("shared accumulation requires Ratio or a scalar")
                mass = jax.lax.stop_gradient(mass)
                total_mass = prior_mass + mass
                denominator = jnp.where(total_mass > 0, total_mass, 1)
                # A window's first microbatch starts its sum: the sum the last
                # window committed stays in place, unread, rather than being
                # cleared (`finish_attempt`).
                gradient = jax.tree.map(
                    lambda old, new: jnp.where(
                        fill == 0, new, old * jnp.asarray(prior_mass / denominator, old.dtype)
                        + new * jnp.asarray(mass / denominator, new.dtype)), prior_gradient, gradient)
                numerator = previous.statistics[0] + total
                pooled = Ratio(numerator, total_mass)
                value, active = pooled.mean()
                candidate = dataclasses.replace(previous, gradient=gradient, mass=total_mass,
                                                statistics=(numerator,), effects=effects, qk_stats=qk)
            else:
                leaves = tuple(a + b for a, b in zip(
                    previous.statistics, jax.tree.leaves(stats), strict=True))
                pooled = self.stats_tree.unflatten(leaves)
                value, active = self.objective.reduce_loss(pooled)
                candidate = dataclasses.replace(previous, statistics=leaves, effects=effects, qk_stats=qk)
                replay_required = due & local_ok
            loss = jnp.where(due, value, loss)
        return PendingAttempt(fill, due, active, loss, local_finite, replay_required,
                              pooled, effects, qk, candidate, gradient)

    def pooled_cotangent(self, pooled, current_stats, factor):
        """Seed the pooled statistics' cotangent, in the live leaves' dtypes.

        Replay differentiates the pooled loss rather than each microbatch's
        own, so every microbatch of the window contributes under this one
        cotangent.
        """
        value, back = jax.vjp(lambda s: self.objective.reduce_loss(s)[0], pooled)
        return jax.tree.map(
            lambda cot, leaf: cot.astype(leaf.dtype)
            if jnp.issubdtype(leaf.dtype, jnp.inexact) else cot,
            back(jnp.asarray(factor, value.dtype))[0], current_stats)

    def update_seed(self, pooled, current_stats, factor):
        """The cotangent of the update's loss, the window's `pooled` statistics', on this microbatch's
        `current_stats`, seeded with `factor`: `pooled_cotangent`'s, or, for a loss stated as one scalar,
        which a shared window pools as the mean of its microbatches' (`prepare_attempt`), `factor` over
        their count."""
        if isinstance(pooled, Ratio) and not isinstance(current_stats, Ratio):
            return jnp.asarray(factor / pooled.mass, jnp.result_type(current_stats))
        return self.pooled_cotangent(pooled, current_stats, factor)

    def replay_input(self, state, fill, index):
        """Rebuild the inputs of the window's `index`-th microbatch.

        The batch and the mutable collections come from the retained buffers.
        The key folds in the attempt that first read them, so a replayed
        microbatch draws the randomness it drew before.
        """
        previous = state.accumulation
        assert previous is not None and previous.attempts is not None
        batch = jax.tree.map(lambda x: x[index], previous.batches)
        variables = state.variables if previous.variables is None else {
            **state.variables, **jax.tree.map(lambda x: x[index], previous.variables)}
        accepted, attempt = state.microstep - fill + index, previous.attempts[index]
        # The part its input ran as, which an attempt counts (`step` times `split` plus its part).
        step_info = self.attempt(state, variables, batch, attempt, attempt % self.split, index,
                                 microstep=accepted)
        return self.read(variables, state.compute), batch, step_info

    def attempt(self, state, variables, batch, attempt, part, slot, microstep=None) -> Step:
        """`attempt_step` of this transaction's split, window, recording and watch."""
        return attempt_step(state, variables, self.fetched(state).ema, batch, split=self.split,
                            slots=self.size, attempt=attempt, part=part, slot=slot, microstep=microstep,
                            record=self.record, watched=self.watched)

    def _places(self, slot, evaluated: Evaluated, part) -> jax.Array:
        """The places the window's slot `slot` and the column its loss saw give an evaluation's rows
        (`example_places`), which its own are held to (`recorded`)."""
        return example_places(slot, self.split, evaluated["example_index"].shape[0], evaluated.get("column"),
                              part)

    def read(self, variables, compute):
        """The variables a step's forward reads: the narrow copies in place of
        their parameters (`forward_variables`), projected as the step's update
        reads them (`Objective.projected`), whether or not the objective's
        forward trains the model, so the loss and its gradient are taken where
        the update applies. The projection is not differentiated."""
        return with_projected(forward_variables(variables, compute),
                              functools.partial(self.objective.projected, updated=False))

    @staticmethod
    def unscaled(gradient, factor):
        """Divide a whole gradient tree by the loss scale it was taken under."""
        return jax.tree.map(lambda x: _unscale(x, factor), gradient)

    @staticmethod
    def add_contribution(accumulated, contribution, factor):
        """Add one replayed microbatch's unscaled gradient into the running sum."""
        return jax.tree.map(lambda a, b: a + _unscale(b, factor), accumulated, contribution)

    def progress(self, current: TrainState) -> jax.Array:
        """The share of the run's bound in examples `current` has consumed, float32: what its dispatched
        batches consumed before this update (`TrainState.consumed`)."""
        if self.bound is None or current.consumed is None:
            raise ValueError("a schedule over the run's progress reads the examples it consumed, which only "
                             "a run bounded in examples that no plan counts ahead keeps")
        return jnp.asarray(current.consumed, jnp.float32) / self.bound

    def commit(self, current, gradient, effects, qk, statistics):
        """Apply the pooled native gradient and return its finiteness and global norm. An optimizer that
        reads more than the gradient is handed the update's pooled loss `statistics` (a Gauss-Newton
        update's system, `dew.gauss_newton`) and attention logit maxima (`qk`).

        The norm is reduced beside optimizer.update, in the same compiled
        transaction, and leaves it with the loss and auxiliary report. No
        gradient tree crosses to the host or is reconstructed for logging.
        Admission is decided by finish_attempt after checking the updated
        parameters and optimizer state as well.
        """
        # The clocks count the updates the optimizer applies (`update_applied`), which an optimizer that holds
        # some back says by its state; one whose state does not, or a held update whose post-step
        # effects no update would apply, is refused as the step is traced, before any runs.
        refuse_untold(current.opt_state)
        if holds_back(current.opt_state) and self.aux_shape.effects is not None:
            raise ValueError("an objective with post-step effects (Aux.effects) under an optimizer that "
                             "holds updates back, whose held calls' effects no update applies; accumulate "
                             "gradients with Trainer(accumulation=k)")
        if holds_back(current.opt_state) and current.deferred is None:
            raise ValueError("the state holds no tokens waiting for the update its optimizer holds back "
                             "(TrainState.deferred): build it with Trainer.initial_state, or restore it "
                             "through the trainer, which gives one written before them none waiting")
        # The gradient where the parameters are placed (`placed`): left where the backward made it, a
        # stage-local leaf's update came out split over the tensor axis its pipeline's read took, and the
        # commit gathered every parameter and moment back each step (row 3697).
        gradient = gradient if self.placed is None else placed_as(gradient, self.placed)
        native = jax.tree.map(_descending, gradient, current.variables["params"])
        native_finite = _all_finite(native)
        norm = optax.global_norm(native)
        norm = norm.astype(at_least_fp32(norm.dtype))

        def apply(current):
            # The update applies to the parameters the model's constraint projects
            # (`Objective.projected`), and the step stores them projected again.
            stored = self.objective.projected(current.variables["params"], updated=False)
            update, opt_state = optimizer_update(
                self.optimizer, native, current.opt_state, stored, qk=qk, statistics=statistics,
                progress=None if self.bound is None else self.progress(current))
            # The update moves the weights, and each post-step effect of this update (QK-Clip's
            # rescale, a bounded group's clamp) acts on them (`stepped`).
            params = {**current.variables, "params": self.objective.projected(
                stepped(stored, update, current.opt_state, opt_state), updated=True)}
            done = update_applied(current.opt_state, opt_state)
            if self.aux_shape.effects is not None:
                params = write_back(params, self.objective.apply_effects(
                    params, self.effects_tree.unflatten(effects)))
            averaged = current.ema
            if self.objective.ema is not None:
                if averaged is None:
                    raise ValueError("the objective declares EMA but the state carries none")
                # A decay over the run's progress reads it where the run keeps one (`EMASpec.progressed`).
                at = self.progress(current) if self.objective.ema.progressed else current.updates
                averaged = _chosen(self.objective.averages(current.updates) & done,
                                   ema_update(averaged, params, self.objective.ema.decay(at)), averaged)
            inspected = {} if self.inspection is None else summaries(
                self.inspection, stored, native, update, params["params"])
            updates = current.updates + done.astype(current.updates.dtype)
            return dataclasses.replace(current, variables=params, opt_state=opt_state, ema=averaged,
                                       updates=updates), inspected

        if current.scale is None:
            return *apply(current), native_finite, norm
        if self.size == 1:
            applied, inspected = apply(current)
            return _chosen(native_finite, applied, current), inspected, native_finite, norm
        unapplied = jax.tree.map(jnp.zeros_like, jax.eval_shape(apply, current)[1])
        return *_only_when(native_finite, apply, lambda x: (x, unapplied), current), native_finite, norm

    def watched_summaries(self, aux, probed) -> dict[str, jax.Array]:
        """A sampled step's watched modules' summaries: their outputs' (`watched_activations`), and with
        their probes' gradient `probed`, the loss's cotangents there (`dew.nn.inspection.cotangent_summaries`,
        unscaled as the probes read them, `unscaled`)."""
        if self.watched is None:
            return {}
        found = watched_activations(self.watched, aux)
        if probed is None:
            return found
        assert aux.inspected is not None
        logged = kept(aux.inspected, self.watched.modules)
        return {**found, **cotangent_summaries(probed, self.watched.histogram, logged)}

    def finish_attempt(self, state, batch, aux, pending, gradient, attempt, tokens, probed=None):
        """Commit the window when it is due, then advance the state's clocks.

        A rejected attempt costs the same trace as an accepted one. With a
        dynamic scaler an attempt is admitted only when every value it
        produced is finite, the committed parameters and optimizer state
        included; a rejected attempt returns `state` with its window dropped
        (`dropped_window`), its parameters, optimizer state and EMA untouched,
        and the scaler moved.

        A window of one microbatch commits on every step it can, so its
        commit is computed and kept by elementwise selects. A longer window
        commits under `lax.cond`, which skips the optimizer on the
        microbatches that do not close it. A GPU conditional whose branch
        holds a collective, as fsdp's gradient norm is, reads its predicate
        on the host: the host then queues the next step only once the device
        has reached this one's commit, and the device idles through every
        step's host work (15.9 ms a 32.5 ms DiT step on four RTX 3090s).

        A window that is not due retains this microbatch's batch, read
        snapshots and `attempt` in its slot, so a later replay can reproduce
        it; a window that committed is zeroed for the next one, but for its
        gradient sum, which the next window's first microbatch replaces.
        """
        scale, previous = state.scale, state.accumulation
        effects, qk, loss = pending.effects, pending.qk, pending.loss
        finite = pending.local_finite & _all_finite((loss, gradient, effects, qk, aux.variables))
        accepted = jnp.asarray(a=True) if scale is None else finite
        numerical = dataclasses.replace(state, variables=write_back(state.variables, aux.variables),
                                        microstep=state.microstep + 1)

        def commit(current):
            return self.commit(current, gradient, effects, qk, pending.pooled)

        def recopied(current):
            # A window of several microbatches reads one set of copies
            # throughout and rewrites them only as it commits.
            committed, inspected, native_finite, norm = commit(current)
            if committed.compute is not None:
                committed = dataclasses.replace(committed, compute=narrowed(committed.variables, self.copies))
            return committed, inspected, native_finite, norm

        due = pending.due & pending.active & accepted
        # The microbatch's real tokens (`tokens`, two uint32 words), which its window's commit trains on,
        # the window's accepted ones before it too.
        read = tokens
        held = read if previous is None or previous.tokens is None else _TokenCount.add(previous.tokens, read)
        if self.size == 1:
            committed, inspected, native_finite, norm = commit(numerical)
            numerical = _chosen(due, committed, numerical)
            native_finite = native_finite | ~due
        else:
            dtype = jnp.result_type(*jax.tree.leaves(state.variables["params"]), jnp.float32)
            zero = jnp.real(jnp.zeros((), dtype))
            uncommitted = jax.tree.map(jnp.zeros_like, jax.eval_shape(recopied, numerical)[1])
            # The window goes round the commit, which never reads it: a
            # conditional copies what it passes through.
            numerical, inspected, native_finite, norm = _only_when(
                due, recopied, lambda x: (x, uncommitted, jnp.asarray(a=True), zero),
                dataclasses.replace(numerical, accumulation=None))
            numerical = dataclasses.replace(numerical, accumulation=previous)
        inspected = {**inspected, **self.watched_summaries(aux, probed)}
        finite = finite & native_finite
        if scale is not None:
            accepted = finite & _all_finite((numerical.variables, numerical.opt_state, numerical.ema))
        if previous is not None:
            candidate = pending.candidate
            assert candidate is not None

            def retain(acc):
                if acc.tokens is not None:
                    acc = dataclasses.replace(acc, tokens=_TokenCount.add(acc.tokens, read))
                if not self.shared:
                    acc = dataclasses.replace(
                        acc,
                        batches=jax.tree.map(
                            lambda held, x: held.at[pending.fill].set(x), acc.batches, batch
                        ),
                        attempts=acc.attempts.at[pending.fill].set(attempt),
                    )
                    if acc.variables is not None:
                        reads = {name: state.variables[name] for name in acc.variables}
                        acc = dataclasses.replace(acc, variables=jax.tree.map(
                            lambda held, x: held.at[pending.fill].set(x), acc.variables, reads))
                return acc

            # A pooled window's gradient sum (a replayed window keeps none)
            # goes round the reset too, its next first microbatch restarting
            # it (`prepare_attempt`): passed through on every microbatch the
            # window does not commit, it was copied each time (Qwen3.5-0.8B's 1 GB
            # embedding sum, twice a step on an A100).
            window = jax.lax.cond(pending.due, cleared, retain, dataclasses.replace(candidate, gradient=None))
            numerical = dataclasses.replace(
                numerical, accumulation=dataclasses.replace(window, gradient=candidate.gradient))
        # A rejected attempt drops its window (`dropped_window`): the update it would have closed is
        # skipped whole, its accepted microbatches with it.
        advanced = _chosen(accepted, numerical, dropped_window(state, self.size))
        if self.size == 1 and advanced.compute is not None:
            # The copies are the cast of whichever parameters were chosen, so
            # they are cast after the choice. Choosing between the new and the
            # old copies reads the old ones, two bytes a parameter that the
            # update's fused pass otherwise never reads.
            advanced = dataclasses.replace(advanced, compute=narrowed(advanced.variables, self.copies))
        if scale is not None:
            advanced = dataclasses.replace(advanced, scale=_advance_scale(scale, finite))
        # An update the optimizer holds back commits nothing yet: its tokens wait in the state
        # (`TrainState.deferred`) for the update that applies them (`update_applied`).
        # An input the optimizer skipped trains no update, so its tokens wait for none (`input_taken`).
        done = update_applied(state.opt_state, advanced.opt_state)
        counted = due & accepted & done
        if state.deferred is not None:
            taken = input_taken(state.opt_state, advanced.opt_state)
            held = _TokenCount.add(state.deferred, jnp.where(taken, held, 0).astype(jnp.uint32))
            waiting = jnp.where(due & accepted & ~done, held, jnp.where(counted, 0, state.deferred))
            advanced = dataclasses.replace(advanced, deferred=waiting.astype(jnp.uint32))
        return advanced, loss, aux, Committed(
            jnp.where(counted, norm, 0.), counted.astype(jnp.int32),
            jnp.where(counted, held, jnp.zeros((2,), jnp.uint32)), stacked(inspected, counted))

    def step(self, *, realize=None, host=False):
        """Build the function that runs one attempt over the shared phases.

        A host-master step compiles each phase on its own and replays in a
        Python loop, because its realizations cross an eager transport
        boundary. A resident step traces the phases together and replays
        inside `lax` control flow.
        """
        realize = self.realize if realize is None else realize
        # Each host phase is a program over the objective's modules, compiled through Dew's compile point
        # with their options as the step's own compile takes them (`program`).
        towers = self.objective.towers()
        compile_phase = (lambda phase: owned_program(phase, *towers)) if host else (lambda phase: phase)
        reduce = compile_phase(self.local_reduction)
        prepare = compile_phase(self.prepare_attempt)
        pooled = compile_phase(self.pooled_cotangent)
        finish = compile_phase(self.finish_attempt)
        unscale = compile_phase(self.unscaled)
        add = compile_phase(self.add_contribution)

        def run(state, batch, part, tokens, closes=None):
            # The `part`th microbatch of the step's batch is attempt step *
            # split + part, with a draw of its own; the objective's schedules
            # count steps, however many microbatches each is read in. `tokens`
            # is its real tokens, which the trainer reads off the accounting
            # columns the objective's batch no longer holds (`model_batch`).
            attempt = state.step * self.split + part
            # Every attempt the run made and did not accept draws afresh (`attempt_step`'s rejections).
            slot = state.microstep % self.size
            step_info = self.attempt(state, state.variables, batch, attempt, part, slot)
            factor = jnp.asarray(1., jnp.float32) if state.scale is None else state.scale.scale
            # A watched cotangent is read unscaled by the factor its backward is seeded with.
            with unscaled(factor):
                realized = realize(self.read(state.variables, state.compute), batch, step_info)
            loss, cotangent = reduce(realized.stats, factor)
            # A step that summarizes its cotangents takes its probes' gradient from the same backward where
            # the microbatch is its update's whole window; otherwise from the pooled loss's, below.
            probing = realized.probed
            whole = probing is not None and self.size == 1
            gradient, probed = (realized.pullback(cotangent), None) if probing is None or not whole else (
                probing(cotangent))
            if state.compute is None or self.size > 1:
                gradient = unscale(gradient, factor)
            # Otherwise no loss scale divides it (`Trainer._narrow_copies`),
            # and a copied parameter's gradient stays bf16 until the update
            # widens it as it reads it. A window of several microbatches
            # widens it as it pools it, so the window sums in fp32.
            # A run whose stream ends inside the window closes it at the step's last part (`closes`).
            closing = None if closes is None else jnp.logical_and(closes, part == self.split - 1)
            pending = prepare(state, realized.stats, realized.aux, loss, gradient, closing)
            if probing is not None and not whole:
                # The probes summarize the cotangent of the update's loss, the window's pooled one, as replay
                # differentiates it (`pooled_cotangent`), whatever microbatches the plan splits it into.
                probed = probing(self.update_seed(pending.pooled, realized.stats, factor))[1]

            # The record of each of the window's other microbatches the commit replays, in its slot.
            own = evaluation_of(realized.aux)
            unreplayed = None if own is None else [jax.tree.map(jnp.zeros_like, (own, self._places(
                state.microstep % self.size, own, part))) for _ in range(self.size - 1)]

            def replay(_):
                cot = pooled(pending.pooled, realized.stats, factor)
                combined = unscale(realized.pullback(cot), factor)
                def one(index, carried):
                    accumulated, kept = carried
                    replayed, records, step_info = self.replay_input(state, pending.fill, index)
                    realized = realize(replayed, records, step_info)
                    replayed_record = evaluation_of(realized.aux)
                    if kept is not None and replayed_record is not None:
                        attempt = state.accumulation.attempts[index]
                        places = self._places(index, replayed_record, attempt % self.split)
                        evaluated = (replayed_record, places)
                        kept = [jax.tree.map(lambda held, new, slot=slot: jnp.where(slot == index, new, held),
                                             held, evaluated) for slot, held in enumerate(kept)]
                    return add(accumulated, realized.pullback(cot), factor), kept
                # The window's other microbatches stand in its slots before this one's: all of them
                # where the window is whole, those it holds where a closing attempt ends it early.
                if host:
                    carried = (combined, unreplayed)
                    for index in range(int(pending.fill)):
                        carried = one(index, carried)
                    return carried
                return jax.lax.fori_loop(0, pending.fill, one, (combined, unreplayed))

            replayed = jnp.zeros((), bool)
            if state.accumulation is not None and not self.shared:
                replayed = pending.replay_required
                if host:
                    gradient, replays = replay(None) if bool(replayed) else (pending.gradient, unreplayed)
                else:
                    gradient, replays = jax.lax.cond(
                        replayed, replay, lambda _: (pending.gradient, unreplayed), None)
            else:
                gradient, replays = pending.gradient, []
            if own is not None:
                replays = replays or []
                evaluations = [(own, self._places(state.microstep % self.size, own, part)), *replays]
                held = jnp.arange(len(replays)) < pending.fill
                ran = jnp.concatenate([jnp.ones((1,), bool), jnp.logical_and(held, replayed)])
                realized = dataclasses.replace(realized, aux=dataclasses.replace(
                    realized.aux, record=recorded(evaluations, ran)))
            if not host:
                # The commit compiles as the host-master step's does, over values
                # already computed: without the barrier XLA fuses the gradient's
                # producers into the optimizer's arithmetic, which then rounds
                # otherwise than the host's compiled commit on the same platform.
                # One barrier a leaf, so each leaf's update can start as soon as
                # its own gradient is ready.
                aux, pending, gradient = jax.tree.map(
                    jax.lax.optimization_barrier, (realized.aux, pending, gradient))
                return self.past(finish, state, gradient, batch, aux, pending, gradient, attempt, tokens,
                                 probed)
            moving, held = apart(state)
            advanced, loss, aux, committed = finish(moving, batch, realized.aux, pending, gradient, attempt,
                                                    tokens)
            return together(advanced, held), loss, aux, committed
        return run
