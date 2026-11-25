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
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import Generic

import jax
import jax.extend.core as jcore
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import dynamic_scale as dynamic_scale_lib
from typing_extensions import TypeVar

from dew._coordination import read_by_holders
from dew.averaging import ema_update
from dew.batch import EXAMPLE_ROWS, FROZEN, Batch, Variables, dicts, leaves_by_path, merge, under_roots
from dew.data.dataset import rows_of
from dew.gauss_newton import solves_equations
from dew.held_updates import holds_back, input_taken, refuse_untold, update_applied
from dew.nn import in_backward
from dew.nn.dense import NVFP4_HISTORY, rolled
from dew.nn.finite import all_finite, leaf_finite, zeroed_unless
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
from dew.nn.kernels.options import abstract_signature, owned_program, program, trace_context
from dew.nn.moe import fixed_routers
from dew.nn.precision import at_least_fp32
from dew.nn.programs import called, computed
from dew.nn.sharding import constrained, rows_spec
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
from dew.objectives.routing import refuse_fixed_routers
from dew.optim import clips_qk, pooled_update, post_step_order, stepped
from dew.optim_schedules import injected_edited
from dew.state import Accumulation, ReplicaState, TokenCount, TrainState, apart, occupancy, together
from dew.training.distributed import microbatch
from dew.training.inspection import Inspection, Summaries, stacked, summaries
from dew.training.model_flops import step_counts
from dew.training.narrow import forward_variables, narrowed
from dew.training.staging import fetched, in_device, staged

Loss = TypeVar("Loss", default=Ratio | jax.Array | float)
Effects = TypeVar("Effects", default=None)


@dataclasses.dataclass(frozen=True)
class PassesProgress[L, E]:
    """A custom training step (`TrainerAdvanced(step=)`) that passes the run's progress to a model that reads
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
        raise ValueError(f"a custom training step (TrainerAdvanced(step=)) passes a {type(module).__name__}, "
                         "which reads the run's progress (dew.nn.progress), none unless it says it does: "
                         "pass Progress(state.updates) as the model's progress= and wrap the step's factory "
                         "in dew.training.PassesProgress, or it trains at the run's end")


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
    own: Ratio | jax.Array | float | None = None
    """The microbatch's own loss statistics, which a step read in several pools into its batch's loss
    (`dew.training.planning.whole_step`) and lets go; None past the step."""
    unfinite_layer: jax.Array | None = None
    """The first layer the attempt's backward walk met whose gradient was not finite (`Applied`), -1 for
    none; None in a step that updates no layer in its backward, and past the step."""
    unfinite_state: jax.Array | None = None
    """A scaled step's committed state as `unfinite_state` reads it: its first leaf not finite (-1 for none)
    and how many are not (`dew.training.interval.checked_finite` stops the run on one); None where the step
    is not scaled."""
    finite: jax.Array | None = None
    """Whether an attempt with no loss scale was finite, its gradient and forward as well as its loss, which
    the step reports (`outcome`) and a run's streak of steps that are not reads (`dew.training.interval`): a
    finite loss over a gradient that is not trains nothing. None under a loss scale, whose overflow its
    scale backs off from, the step reporting its loss's finiteness, and past the step, whose own finiteness
    the whole of its parts states (`whole_step`)."""


def _chosen(take, taken, kept):
    """`taken` where `take` holds and `kept` elsewhere, leaf by leaf."""
    return jax.tree.map(lambda new, old: jnp.where(take, new, old), taken, kept)


def cleared(acc: Accumulation) -> Accumulation:
    """`acc` emptied for its next window: its sums zero, its QK maxima and statistics stamp at their
    identities."""
    return dataclasses.replace(
        jax.tree.map(jnp.zeros_like, acc),
        qk_stats=jax.tree.map(lambda x: (jnp.full_like(x, -jnp.inf) if jnp.issubdtype(x.dtype, jnp.inexact)
                                         else jnp.zeros_like(x)), acc.qk_stats),
        statistics_stamp=None if acc.statistics_stamp is None else jnp.full_like(acc.statistics_stamp, -1))


def dropped_window(state):
    """`state` with its window dropped: its accepted microbatches forgotten and
    the next taking its first slot, as a rejected attempt (`Transaction.finish_attempt`) or a batch
    ramp's new stage (`Trainer`'s runner) leaves it, so a committed update pools one window of one
    stage's batches, all drawn under one key (`example_fields`). The window's gradient sum stays in
    place, unread, as a commit leaves it (`prepare_attempt`); the mutable collections its microbatches
    wrote keep what they wrote, as torch's AMP keeps a skipped update's running statistics."""
    if state.accumulation is None:
        return state
    microstep, window = window_dropped(state.microstep, state.accumulation)
    return dataclasses.replace(state, microstep=microstep, accumulation=window)


def window_dropped(microstep: jax.Array, acc: Accumulation) -> tuple[jax.Array, Accumulation]:
    """`dropped_window`'s two fields: the microstep back by the microbatches the window held (`occupancy`),
    and the window cleared, its occupancy 0, but for its gradient sum. A state's other fields go round it,
    as a compiled program over them alone leaves a host-master state's resident frozen weights where they
    are."""
    return microstep - occupancy(acc), dataclasses.replace(
        cleared(dataclasses.replace(acc, gradient=None, batch_statistics=None)),
        gradient=acc.gradient, batch_statistics=acc.batch_statistics)


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


def outcome(current: TrainState, advanced: TrainState, loss: jax.Array, aux: Aux, committed: Committed | None,
            batch: Batch, stepped: jax.typing.ArrayLike, consumed: jax.Array, *, split: int,
            part: Batch | None = None, passes: jax.Array | None = None) -> tuple:
    """What a compiled step returns (`Trainer.compile`) of the transaction it ran from `current` on
    `batch`, wherever it ran: the advanced state, its step counting `stepped` and its clock the source
    examples its microbatch `consumed` (`TrainState.consumed`, where the run keeps one), the loss, the
    metrics with the batch's counts, and whether the attempt was finite (`Committed.finite`, its loss's
    where it states none), the microbatch accepted and the window committed. A metric stated as a statistic
    (`Aux.pooled`) is its value where the step is read whole (`split` 1), and its statistic where it is one
    of `split` microbatches, which the step's whole
    pools (`dew.training.planning.whole_step`). A step of a sampled pass count's `passes` counts the work of
    its microbatch `part` at its count (`step_counts`)."""
    step, clock, *rest = outcome_of(current.step, current.consumed, current.microstep, advanced.microstep,
                                    loss, aux, committed, batch, stepped, consumed, split=split, part=part,
                                    passes=passes)
    return dataclasses.replace(advanced, step=step, consumed=clock), *rest


class Read:
    """`read` (`Transaction.read`: the narrow copies merged, the parameters projected) as a host step reads
    its variables, tracing and compiling nothing at a step: its first call, over the variables' shapes (a
    run's startup pass, `options.collecting`), traces it once and keeps which of its results are its inputs as
    given (a leaf no projection changes), the program computing the others alone from that one trace
    (`_unforwarded`, compiled there, `read` traced no second time): each later call over arrays runs that
    program, if any leaf is computed, and hands the others back as the arrays they are, no copy of a
    parameter the projection leaves as it is."""

    def __init__(self, read: Callable):
        self.read = read
        # Per variables' tree: which result is which input as given, the results' tree, and the program.
        self.plans: dict[Hashable, tuple] = {}

    def __call__(self, variables, compute):
        leaves = jax.tree.leaves((variables, compute))
        # Which results are inputs as given is the traced program's: its inputs' types and the trace
        # context decide it (a cast is no copy of a leaf already of its dtype), as they key its lowering.
        key = abstract_signature((variables, compute)), trace_context()
        plan = self.plans.get(key)
        if plan is None:
            traced, shapes = jax.make_jaxpr(self.read, return_shape=True)(variables, compute)
            given = list(traced.jaxpr.invars)
            forwarded = tuple(given.index(out) if isinstance(out, jcore.Var) and out in given else None
                              for out in traced.jaxpr.outvars)
            computed = tuple(index is None for index in forwarded)
            program = owned_program(functools.partial(_unforwarded, traced, computed)) if any(
                computed) else None
            plan = self.plans[key] = (forwarded, jax.tree.structure(shapes), program)
        forwarded, read_structure, program = plan
        made = iter(() if program is None else program(variables, compute))
        return jax.tree.unflatten(read_structure, [leaves[index] if index is not None else next(made)
                                            for index in forwarded])


def _unforwarded(traced: jcore.Jaxpr, computed: tuple[bool, ...], variables, compute) -> tuple:
    """The results of `traced`, a read's one trace (`Read`; a Jaxpr carries its constants on JAX 0.12),
    that are no input as given, in their order: its equations evaluated over the inputs' leaves, the read's
    Python run no second time."""
    read = jcore.jaxpr_as_fun(traced)(*jax.tree.leaves((variables, compute)))
    return tuple(leaf for leaf, made in zip(read, computed, strict=True) if made)


def outcome_of(step: jax.Array, clock: jax.Array | None, microstep: jax.Array, advanced: jax.Array,
               loss: jax.Array, aux: Aux, committed: Committed | None, batch: Batch,
               stepped: jax.typing.ArrayLike, consumed: jax.Array, *, split: int, part: Batch | None = None,
               passes: jax.Array | None = None) -> tuple:
    """`outcome` of the fields it reads of the states (their `step`, `consumed` clock and `microstep`,
    the advanced state's `advanced`), and nothing else of them: the step and clock the advanced state
    takes, then `outcome`'s other results. A host step runs it as a program of its own over those fields
    (`dew.training.compiler`), its states' other leaves read by no program."""
    pooled = aux.pooled if split > 1 else {name: value.mean()[0] for name, value in aux.pooled.items()}
    metrics = {**aux.metrics, **pooled, **step_counts(batch, part, passes)}
    if aux.record is not None:
        assert "evaluated" in aux.record, "a recording transaction stacks its evaluations (`recorded`)"
        metrics |= recorded_metrics(aux.record)
    clock = None if clock is None else (clock + consumed).astype(clock.dtype)
    finite = jnp.isfinite(loss) if committed is None or committed.finite is None else committed.finite
    return (step + stepped, clock, loss, metrics, finite, advanced > microstep,
            dataclasses.replace(committed, finite=None) if committed is not None else None)


def probed_loss(objective: Objective, variables: Variables, batch: Batch, step: Step):
    """The objective's loss statistics with its parameters and its watched outputs' cotangent probes
    (`dew.nn.inspection.tapped`), their pullback giving the parameters' gradient and the probes', and the
    loss's aux. The probes are laid out as the objective's model calls nest their modules, which a trace
    of its loss that discovers them (`discovering`) sows, nothing computed; a probe no model call reads,
    its collection dropped on the way, is refused."""
    probes = discovered_probes(objective._loss, variables, batch, step)

    def loss(trainable, probes):
        return objective._loss({**variables, "params": trainable, PROBES: probes}, batch, step)
    stats, back, aux = jax.vjp(loss, variables["params"], probes, has_aux=True)
    refuse_unread_probes(probes, aux)
    return stats, back, aux


def discovered_probes(loss: Callable, variables: Variables, batch: Batch, step: Step) -> dict:
    """The zero cotangent probes `loss` over `variables`, `batch` and `step` reads, laid out as its model
    calls nest their watched modules (`dew.nn.inspection.discovering`, nothing computed): what a probed
    loss differentiates beside the parameters (`probed_loss`; a host step's, `dew.training.execution`).
    Refused where no watched module's output reaches the loss."""
    with discovering():
        _, discovered = jax.eval_shape(lambda: loss(variables, batch, step))
    probes = probes_of(discovered.inspected or {})
    if not probes:
        raise ValueError("no watched module's output reaches the objective's loss for its cotangent to be "
                         "summarized; name modules its model calls run (Inspection.modules)")
    return probes


def refuse_unread_probes(probes: dict, aux: Aux) -> None:
    """Refuse a probe no model call read (`probes_of` of the loss's sown collection): its collection was
    dropped on its way to the watched module."""
    unread = sorted(set(leaves_by_path(probes)) - set(leaves_by_path(probes_of(aux.inspected or {}))))
    if unread:
        raise ValueError(f"the objective's model calls do not pass the step's cotangent probes "
                         f"(variables[{PROBES!r}]) to the watched modules at {unread}; pass every "
                         f"collection of the variables the objective is given through to its model")


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
    (`dew.gauss_newton`). An update its type declares to take none, a plain transformation wrapping the
    clip, is handed the logits' maxima where its state holds a QK-Clip, which reads the step's own and
    would otherwise step unclipped (row 6552); never the statistics, so a solver in it refuses
    (`dew.gauss_newton`) rather than solve one step's system for the window's update. Where the optimizer
    holds its update back over several calls (`optax.MultiSteps`, which runs its inner update at the
    window's last call alone, or `optax.apply_every`), each call's maxima are pooled in its QK-Clip's state
    (`dew.optim.pooled_update`), so the update that applies is clipped by every microstep's, as the
    trainer's own accumulation pools them."""
    clocked = {} if progress is None else {"progress": progress}

    def update(state) -> tuple:
        if isinstance(optimizer, optax.GradientTransformationExtraArgs):
            solved = {"statistics": statistics} if solves_equations(state) else {}
            return optimizer.update(gradient, state, params, qk_stats=qk, **solved, **clocked)
        clipped = {"qk_stats": qk} if clips_qk(state) else {}
        return optimizer.update(gradient, state, params, **clipped)

    if qk is not None and clips_qk(opt_state) and holds_back(opt_state):
        return pooled_update(update, opt_state, params, qk)
    return update(opt_state)


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


def consumed_where_made(optimizer: optax.GradientTransformation, state: TrainState, window: int, *,
                        shared: bool, progressed: bool) -> bool:
    """Whether the step reads each gradient where it is made into the buffers its state donates, so it
    holds none beside another: with the state it commits
    (`COMMIT_FIELDS`) in device memory, a window of several microbatches that pools them by mean (`shared`,
    `pooled_by_mean`) sums each into its accumulator's gradient leaf in device memory, where one that
    replays them keeps their gradients for the replay; and a window of one updates each by `optimizer`,
    where its update reads each gradient into its own leaf's results alone (`leaf_local`, `progressed` the
    run handing it its progress). A dynamically scaled loss commits only once every gradient is known
    finite, a committed state fetched from the host is fetched behind one barrier over every gradient
    (`Transaction.past`), and an update that reads them together (clipping by the global norm) holds them
    all until it has read the last. A head that takes its gradient in the backward a tile at a time
    (`head_tile`, unfused) has XLA run every layer's weight gradient after the whole backward (row 3653),
    each gradient then read where it is made as any head's."""
    if state.scale is not None or not in_device([getattr(state, field) for field in COMMIT_FIELDS]):
        return False
    if window > 1:
        return shared and state.accumulation is not None and in_device(state.accumulation.gradient)
    params, opt_state = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
                                     (state.variables["params"], state.opt_state))
    return leaf_local(optimizer, params, opt_state, progressed=progressed)


def _placeholder(node) -> bool:
    """Whether `node` is the place an optimizer's per-parameter tree holds for a parameter it keeps no value
    of: a partitioned optimizer's group's for one another group steps (`optax.multi_transform`'s masks), or
    an 8-bit moment's scales' for a parameter it keeps unquantized (None, `dew._moments.int8_moments`)."""
    return node is None or isinstance(node, optax.MaskedNode)


def _laid_as(params) -> Callable[[optax.OptState], bool]:
    """Whether a node of an optimizer's state lays its leaves out as `params`' tree, a placeholder
    (`_placeholder`) a leaf."""
    shape = jax.tree.structure(params)
    return lambda node: jax.tree.structure(node, is_leaf=_placeholder) == shape


def _copy_of(params) -> Callable[[optax.OptState], bool]:
    """Whether a node of an optimizer's state is a copy of `params`' tree (Adam's moments): laid out as it
    (`_laid_as`), at each parameter a float leaf of its shape or a placeholder, one at least a value."""
    laid, shapes = _laid_as(params), [np.shape(leaf) for leaf in jax.tree.leaves(params)]

    def copy(node) -> bool:
        if not laid(node):
            return False
        held = jax.tree.leaves(node, is_leaf=_placeholder)
        valued = [(leaf, want) for leaf, want in zip(held, shapes, strict=True) if not _placeholder(leaf)]
        return bool(valued) and all(np.shape(leaf) == want and jnp.issubdtype(leaf.dtype, jnp.floating)
                                    for leaf, want in valued)
    return copy


def _placeholders_of(params) -> Callable[[optax.OptState], bool]:
    """Whether a node of an optimizer's state is laid out as `params`' tree (`_laid_as`), holding no value."""
    laid = _laid_as(params)
    return lambda node: laid(node) and not jax.tree.leaves(node)


def _param_copies(opt_state, params) -> list[Variables]:
    """The copies of `params`' tree an optimizer's state holds (Adam's moments), in the state's order."""
    copy = _copy_of(params)
    return [node for node in jax.tree.leaves(opt_state, is_leaf=copy) if copy(node)]


def _with_copies(opt_state, params, replaced: Callable[[int, Variables], Variables]):
    """`opt_state` with its `k`th copy of `params`' tree (`_param_copies`) `replaced(k, copy)`."""
    copy = _copy_of(params)
    counted = iter(range(len(_param_copies(opt_state, params))))

    def each(node):
        return replaced(next(counted), node) if copy(node) else node
    return jax.tree.map(each, opt_state, is_leaf=copy)


def _placeholders_at(opt_state, params, path: tuple[str, ...]):
    """`opt_state` with each of its trees laid out as `params` that holds no value (`_placeholders_of`) cut to
    its subtree at `path`, as a layer's update reads them."""
    empty = _placeholders_of(params)
    return jax.tree.map(lambda node: _nested(path, _at(node, path)) if empty(node) else node, opt_state,
                        is_leaf=empty)


def _held_in_copies(opt_state, params) -> bool:
    """Whether every leaf of `opt_state` a parameter owns lies in one of its copies of `params`' tree
    (`_param_copies`), the rest scalars (a count): a factored optimizer's state holds its parameters' rows
    and columns (Adafactor's), and an 8-bit one's its moments' codes and their blocks' scales
    (`dew._moments.int8_moments`), none of which a layer's slice reads as the layer's own, nor a backward
    writes as a float cotangent."""
    rest = _with_copies(opt_state, params, lambda index, copy: {})
    return all(np.ndim(leaf) == 0 for leaf in jax.tree.leaves(rest))


def _nested(path: tuple[str, ...], subtree) -> Variables:
    """`subtree` under `path`."""
    for name in reversed(path):
        subtree = {name: subtree}
    return subtree


def _at(tree, path: tuple[str, ...]):
    """`tree`'s subtree at `path`."""
    for name in path:
        tree = tree[name]
    return tree


def _stage_runs(params: Variables, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """The paths of the runs a one-stage scanned state holds stacked (`dew.training.stages`), whether
    whole or by position of a repeated period: each under a `stages` subtree whose leaves lead with a stage
    axis of one."""
    found = []
    for name, child in params.items():
        if not isinstance(child, Mapping):
            continue
        if name == "stages" and all(np.shape(leaf)[:1] == (1,) for leaf in jax.tree.leaves(child)):
            found += [(*path, name, run) for run in child]
        else:
            found += _stage_runs(child, (*path, name))
    return found


def _state_path(path: tuple[str, ...]) -> tuple[str, ...]:
    """A scanned run's path in a one-stage state (`_stage_runs`) from its module's path in the model."""
    return (*path[:-1], "stages", path[-1])


@struct.dataclass
class Applied:
    """What the runs that updated in their backward (`dew.nn.in_backward`) leave for a step's commit:
    their new parameters (the masters where the step keeps narrow copies) and the new copies of them in
    the optimizer's state (`_param_copies`), each a tree of the runs alone under their paths, and their
    gradients' sum of squares and non-finite count, which the step's norm and finiteness read."""

    params: Variables
    moments: tuple[Variables, ...]
    sumsq: jax.Array
    nonfinite: jax.Array
    unfinite_layer: jax.Array
    """The first layer the backward's walk met whose gradient was not finite (the runs' layers numbered in
    the model's order, the walk taking them from the top), -1 for none: the layers above it moved, it and
    those below took a zero gradient (`dew.nn.in_backward.gated`), which a run's log names."""

    def onto(self, state: TrainState) -> TrainState:
        """`state` with the runs' new parameters and moments in place of theirs."""
        params = state.variables["params"]
        opt_state = _with_copies(state.opt_state, params,
                                 lambda index, copy: merge(copy, self.moments[index]))
        return dataclasses.replace(state, variables={**state.variables, "params": merge(params, self.params)},
                                   opt_state=opt_state)


def _collections_kept(finite: jax.Array, numerical: TrainState, state: TrainState,
                      written: Variables | None) -> TrainState:
    """`numerical` with the forward's collections it wrote back (`written`, a batch norm's statistics) as
    `state` held them where the step, with no loss scale, is not `finite`, by its attempt or its update
    (training.md, a step that is not finite)."""
    if written is None:
        return numerical
    kept = {name: _chosen(finite, numerical.variables[name], state.variables[name])
            for name in dicts(written)}
    return dataclasses.replace(numerical, variables={**numerical.variables, **kept})


def unfinite_state(kept: dict[str, Variables | optax.OptState | None]) -> jax.Array:
    """The first leaf of a committed state's kept trees (`checked`: parameters, optimizer state, averages), in
    their order, that holds a value not finite (-1 for none) and how many such leaves they hold,
    `int32[2]`. A scaled step accepted on its finite gradient can leave them so: an update that overflows a
    parameter, or a gradient near 1e20 that overflows Adam's second moment to infinity, whose parameter then
    stops moving with every loss and gradient after it finite (row 6748). Read off the state once it is
    committed, choosing nothing, so it holds no second state; it reads the kept trees once a step."""
    finite = jnp.stack([all_finite(leaf) for leaf in jax.tree.leaves(kept)])
    return jnp.stack([jnp.where(jnp.all(finite), -1, jnp.argmin(finite)), jnp.sum(~finite)]).astype(jnp.int32)


def _unfinite_layer(runs: Sequence[tuple[jax.Array, int]]) -> jax.Array:
    """The model layer that tops the first scanned slot the backward's walk met a gradient not finite in,
    of `runs`' per-slot counts in the model's order (their stage and slot axes, one stage), each slot of a
    run that many layers (`_period`: one, or a repeated period's): the layer itself for a run of layers, the
    top of the period for a periodic one, whose positions share one count; numbered across the runs. -1 for
    none."""
    first, offset = jnp.asarray(-1, jnp.int32), 0
    for counts, period in runs:
        counts = counts.reshape(-1)
        tops = jnp.arange(counts.size, dtype=jnp.int32) * period + period - 1 + offset
        first = jnp.maximum(first, jnp.max(jnp.where(counts > 0, tops, -1)))
        offset += counts.size * period
    return first


def _period(run: Mapping) -> int:
    """The layers each scanned slot of `run` (its tree in the state) holds: a repeated period's positions
    (`layers_0`, `layers_1`, ...), one each, or one for a run of single layers."""
    keys = [str(key) for key in run]
    return len(keys) if keys and all(key.startswith("layers_") for key in keys) else 1


def _pooling_weights(previous: Accumulation, stats):
    """Shared-mean pooling weights for `stats` joining `previous`: prior mass and gradient, this
    microbatch's mass and total, the combined mass and its non-zero denominator."""
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
    return prior_mass, prior_gradient, mass, total, total_mass, denominator


def _unscaled_beside(gradient: Variables, factor, pooled: Sequence[tuple[str, ...]]) -> Variables:
    """`gradient` unscaled by `factor`, but its runs at `pooled`, which pooled into the window's accumulator
    in their backward unscaled there (`Transaction._planned_accumulation`)."""
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf if under_roots(path, tuple(pooled)) else _unscale(leaf, factor), gradient)


def _pooled_leaf(fill, old_weight, new_weight, old, new):
    return jnp.where(
        fill == 0, new,
        old * jnp.asarray(old_weight, old.dtype) + new * jnp.asarray(new_weight, new.dtype))


@dataclasses.dataclass(frozen=True)
class _Planned:
    """A step's plan for the runs that update or accumulate in their backward (`Transaction.planned`) and
    the collections it adds to what the forward reads: the optimizer's copies of the parameters, the
    masters (or accumulated gradient) where written separately from the forward's weights, and each run's
    norms (`dew.nn.in_backward`)."""

    plan: in_backward.InBackward
    read: dict[str, Variables]
    pooling: _Pooling | None = None
    """Where the runs pool their gradient into the window's accumulator in their backward
    (`Transaction._planned_accumulation`): the weights each layer pools by, set once the forward knows the
    microbatch's share of the window (`Transaction._pulled`); None where they update in it."""
    runs: tuple[tuple[str, ...], ...] = ()
    """The stacked runs the state holds, by their paths in it, which may pool (`Transaction._read_once`)."""


@dataclasses.dataclass
class _Pooling:
    """The weights a window's runs pool each layer's gradient into its accumulator by in their backward:
    whether the window holds a microbatch yet (`fill`), the share the sum so far and this microbatch's keep
    of the pooled mean (`old`, `new`) and the loss scale this microbatch's gradient is unscaled by (`factor`),
    known once its forward is, before its backward is traced."""

    fill: jax.Array | None = None
    old: jax.Array | None = None
    new: jax.Array | None = None
    factor: jax.Array | float | None = None


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
    """`tree` constrained to where `placement` places each of its leaves (`constrained`)."""
    return jax.tree.map(lambda leaf, sharding: constrained(leaf, sharding.spec), tree, placement)


def _descending(gradient: jax.Array, parameter: jax.Array) -> jax.Array:
    """`gradient` in its parameter's dtype, as optax descends along it. JAX's gradient of a real
    loss in a complex parameter is the conjugate of its steepest ascent."""
    return (jnp.conj(gradient) if jnp.iscomplexobj(gradient) else gradient).astype(parameter.dtype)


def checked(state: TrainState | ReplicaState) -> dict[str, Variables | optax.OptState | None]:
    """The trees of `state` (a run's, or DiLoCo's replicas' inner copies) a finite step keeps finite and a
    save refuses non-finite: the parameters, the optimizer state and the averages. The one statement of
    them, which the step's acceptance (`finish_attempt`) and a save's refusal
    (`refuse_non_finite`) read. The optimizer state is read without the
    hyperparameters injected beside it at any depth (`_numerical`): control values a schedule or the user
    sets, not written by a step's gradient, and legally infinite (a clip's bound of `math.inf`)."""
    return {"variables": state.variables, "opt_state": _numerical(state.opt_state), "ema": state.ema}


def _numerical(opt_state: optax.OptState) -> optax.OptState:
    """`opt_state` without the hyperparameters injected beside it at any depth (`injected_edited`): the
    state a step's gradients write."""
    return injected_edited(opt_state, lambda _, node: node._replace(hyperparams={}))



def _finite_each(leaves: list[jax.Array]) -> list[jax.Array]:
    """Whether each of `leaves` is finite throughout, one program over them."""
    return [leaf_finite(leaf) for leaf in leaves]


def finite_flags(leaves: Sequence[jax.Array]) -> list[bool]:
    """Whether each of `leaves`, arrays placed anywhere, is finite throughout (`leaf_finite`), read outside a
    step: one program for each placement the leaves share (their devices and memory kind), as no one
    program spans two meshes (a CPU-owned run's host companion and its accelerator), which hands back one
    flag for them all and, only where it is false, each leaf's; a leaf in host memory (a layout's
    pinned-host field) brought to its devices' memory one leaf at a time (`fetched`). A flag a leaf, read
    each check, took 0.19 s a check of the 4B's 1,493 leaves on a GH200 (UB-6861-FIT2). A placement no device
    of this process's holds (a pipeline stage's mesh another process's devices make, `StageLocal.saved`) is
    read by the processes that hold it, and every process takes their verdicts in one round, so each answers
    alike (row 10780: each rank read each stage's scalar, those outside a stage holding none of it)."""
    groups: dict[tuple, list[int]] = {}
    for index, leaf in enumerate(leaves):
        sharding = leaf.sharding
        groups.setdefault((frozenset(sharding.device_set), sharding.memory_kind), []).append(index)
    # Each placement's flags as this process read them, None where it holds none of its devices (on any
    # backend: a CPU-owned state's companion mesh is no default backend's).
    read: list[list[bool] | None] = []
    for (_, memory), members in groups.items():
        if not leaves[members[0]].sharding.addressable_devices:
            read.append(None)
        elif memory not in (None, "device"):
            read.append([_held_flag(program(leaf_finite)(fetched(leaves[index], leaves[index].sharding)))
                         for index in members])
        else:
            held = [leaves[index] for index in members]
            read.append([True] * len(members) if _held_flag(program(all_finite)(held)) else
                        [_held_flag(finite) for finite in program(_finite_each)(held)])
    flags: list[bool] = [True] * len(leaves)
    for members, held in zip(groups.values(), read_by_holders(read, [devices for devices, _ in groups]),
                             strict=True):
        for index, finite in zip(members, held, strict=True):
            flags[index] = bool(finite)
    return flags


def _held_flag(finite: jax.Array) -> bool:
    """A verdict replicated over its placement's devices, read from one this process holds."""
    return bool(jax.device_get(finite.addressable_data(0)))


def largest_recorded(sown: Variables | None) -> Variables | None:
    """The `nvfp4_amax` collection a step's projections sowed (`Aux.amax`) with each projection's calls'
    records, a tuple, taken to their elementwise largest, its own shape (`[layers]` in a scanned stack):
    what a window pools by its largest and its commit rolls into the projections' histories
    (`dew.nn.dense.rolled`, row 12721). None where the step recorded none."""
    if sown is None:
        return None
    return jax.tree.map(lambda calls: functools.reduce(jnp.maximum, calls), sown,
                        is_leaf=lambda node: isinstance(node, tuple))


def compact_qk(tree, calls: bool = False):
    """Reduce every sowed `max_logits` leaf over its rows, keeping the heads, and a layer's geometry
    (`kv_heads`, `head_dim`, `qk_nope`) to the one count it is; and with `calls`, a layer's calls to one,
    their elementwise largest (`dew.batch.sown_calls`: a looped layer's passes, each over the same
    weights), the window of a run that draws its pass count per microbatch, whose microbatches call a
    looped layer as many times as their counts.

    The clip reads each head's strongest logit alone, so a retained window
    holds one row per layer instead of the batch's. A pipelined layer sows its
    geometry once a microbatch (`StackView`), the same count each time, and
    the clip reads it as one.
    """
    def compact(path, leaf):
        is_maximum = any(isinstance(entry, jax.tree_util.DictKey) and entry.key == "max_logits"
                         for entry in path)
        if is_maximum:
            return jnp.max(leaf, axis=-2, keepdims=True) if leaf.ndim > 1 else leaf
        return jnp.max(leaf) if leaf.ndim else leaf

    def folded(node: Mapping) -> dict:
        return {name: folded(value) if isinstance(value, Mapping) else
                (functools.reduce(jnp.maximum, value),) if isinstance(value, tuple) and value else value
                for name, value in node.items()}

    compacted = jax.tree_util.tree_map_with_path(compact, tree)
    return folded(compacted) if calls and isinstance(compacted, Mapping) else compacted


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
    applied: Applied | None = None
    """What the runs that updated in their backward leave for the commit (`Transaction.planned`)."""
    own: Loss | None = None
    """This microbatch's own statistics, which a step read in several pools into its batch's loss
    (`dew.training.planning.whole_step`)."""
    amax: Variables | None = None
    """The window's largest recorded magnitudes with this microbatch's (`Accumulation.amax`), which the
    commit rolls into the delayed-scaling NVFP4 projections' histories; None in a run that keeps none."""


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
                 microstep=None, record: bool = False, watched: Watched | None = None,
                 batch_statistics: tuple[jax.Array, ...] | None = None) -> Step:
    """The `Step` the loss of attempt `attempt` of the update `state` commits next reads: the step it
    counts, its microbatches accepted (`microstep`, the state's where none is given) over the `split`
    each step is read in; the attempt's own key; the averaged weights `ema` beside `variables`; the
    updates committed, which a model whose computation moves over a run reads (`Objective.progress`);
    and the microbatch's examples, its window `slot` of `slots`, its `part`, after the attempts the run
    rejected (`example_fields`). An attempt and its replay read it, and so do the traces that shape and
    narrow the step (`traced_loss`, `dew.training.narrow.narrow_copies`), which
    record nothing and watch nothing."""
    accepted = state.microstep if microstep is None else microstep
    return Step(accepted // split, jax.random.fold_in(state.key, attempt), with_ema(variables, ema),
                committed=state.updates, record=record, qk_stats=clips_qk(state.opt_state),
                equations=solves_equations(state.opt_state), watched=watched,
                amax=NVFP4_HISTORY in state.variables,
                batch_statistics=batch_statistics,
                **example_fields(state, slot, split, batch, slots, part, attempt - accepted, microstep))


def batch_statistics_shapes[LossT, EffectsT](
        objective: Objective[LossT, EffectsT], state: TrainState, micro: Batch, window: int,
        split: int) -> tuple[jax.ShapeDtypeStruct, ...] | None:
    """The shapes of `objective.batch_statistics` for a step read in `split` microbatches; None where
    `split` is 1 or the objective states none."""
    if split <= 1:
        return None
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

    def traced(variables, compute, ema, micro_batch, step_count, microstep, key, updates):
        current = dataclasses.replace(state, step=step_count, microstep=microstep, key=key,
                                      updates=updates)
        read = with_projected(forward_variables(variables, compute),
                              functools.partial(objective.projected, updated=False))

        def part_at(index: jax.Array) -> tuple[Batch, Step]:
            step = attempt_step(current, variables, ema, micro_batch, split=split, slots=window,
                                attempt=step_count * split + index, part=index, slot=index,
                                microstep=microstep + index)
            return micro_batch, step

        return objective.batch_statistics(read, part_at, split)

    return jax.eval_shape(traced, variables, state.compute, state.ema, micro, state.step, state.microstep,
                          state.key, state.updates)


def placed_as_rows(index: jax.Array | jax.ShapeDtypeStruct, batch: Batch) -> jax.Array | jax.ShapeDtypeStruct:
    """`index`, an entry for each of `batch`'s rows, placed as the batch's rows are, so each device
    reads its own rows' entries alone: an array resharded, its shape (a pass over shapes) carrying the
    placement (`placed_shapes`)."""
    like = next(leaf for leaf in jax.tree.leaves(batch) if np.ndim(leaf))
    placed = rows_spec(like, 1)
    if placed is None:
        return index
    if isinstance(index, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(index.shape, index.dtype, sharding=placed, weak_type=index.weak_type)
    return jax.sharding.reshard(index, placed)


class Transaction:
    """Turn one or more microbatches into one optimizer update.

    Pooling, replay, admission, effects, EMA and the three clocks are written
    once here. `step` composes them for a resident run or for a host-master
    run; only where each phase executes differs.
    """
    def __init__(self, objective, optimizer, accumulation: int, shapes, copies=None, split: int = 1,
                 record: bool = False, resting: Mapping[str, object] | None = None,
                 inspection: Inspection | None = None, examples: int | None = None, placed=None,
                 skip: bool = False, calls: bool = False):
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
        which a schedule over the run reads (`dew.optim_schedules.ScheduleBase.progressed`). `calls` folds
        a looped layer's QK calls into the window as one (`compact_qk`), a run's that draws its pass count
        per microbatch.
        """
        self.calls = calls
        self.resting = dict(resting or {})
        self.placed = placed
        # Whether a step with no loss scale whose loss or gradient is not finite is skipped (the run's opt-in,
        # `TrainerAdvanced(skip_nonfinite=True)`): else it is taken as it is formed and the run stops at its
        # next log, naming it (training.md, a step that is not finite).
        self.skip = skip
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

    def prepare_attempt(self, state, stats, aux, loss, gradient, closing=None,
                        pooled_paths: tuple[tuple[str, ...], ...] = ()):
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
        fill = occupancy(previous)
        due = (fill + 1) == self.size
        if closing is not None:
            due = due | closing
        # A step with no loss scale reads its gradient's finiteness by its commit's norm (`commit`), its
        # pooled gradient carrying every microbatch's; one taken whether finite or not (`skip` off) reads
        # its loss alone, a scalar, the rest of a skipped one's small trees too.
        unscaled = state.scale is None and not holds_back(state.opt_state)
        everything = (loss, stats, aux.variables, aux.effects, *(() if unscaled else (gradient,)))
        local_finite = all_finite(loss) if unscaled and not self.skip else all_finite(everything)
        local_ok = jnp.asarray(a=True) if state.scale is None else local_finite
        if aux.amax is None and NVFP4_HISTORY in state.variables:
            raise ValueError(
                f"{type(self.objective).__name__} returns none of its projections' recorded largest "
                "magnitudes (Aux.amax) though its step asks for them (Step.amax), and the run's "
                "delayed-scaling NVFP4 projections read their amax histories: they would never move")
        qk = compact_qk(aux.qk_stats, self.calls)
        amax = largest_recorded(aux.amax)
        effects = tuple(jax.tree.leaves(aux.effects))
        candidate, pooled = previous, stats
        replay_required = jnp.asarray(a=False)
        if previous is None:
            _, active = self.objective.reduce_loss(stats)
        else:
            effects = tuple(a + b for a, b in zip(previous.effects, effects, strict=True))
            qk = jax.tree.map(jnp.maximum, previous.qk_stats, qk)
            # A window's largest magnitudes the largest of its microbatches', so the history a split step
            # rolls in is the one the step read whole would.
            amax = jax.tree.map(jnp.maximum, previous.amax, amax)
            if self.shared:
                prior_mass, prior_gradient, mass, total, total_mass, denominator = _pooling_weights(
                    previous, stats)
                old_weight, new_weight = prior_mass / denominator, mass / denominator
                # A window's first microbatch starts its sum: the sum the last
                # window committed stays in place, unread, rather than being
                # cleared (`finish_attempt`). Runs already pooled in their backward
                # (`pooled_paths`) pass through without reading `prior_gradient` again.
                if pooled_paths:
                    gradient = jax.tree_util.tree_map_with_path(
                        lambda path, old, new: new if under_roots(path, pooled_paths) else _pooled_leaf(
                            fill, old_weight, new_weight, old, new), prior_gradient, gradient)
                else:
                    gradient = jax.tree.map(
                        lambda old, new: _pooled_leaf(fill, old_weight, new_weight, old, new),
                        prior_gradient, gradient)
                numerator = previous.statistics[0] + total
                pooled = Ratio(numerator, total_mass)
                value, active = pooled.mean()
                candidate = dataclasses.replace(previous, gradient=gradient, mass=total_mass,
                                                statistics=(numerator,), effects=effects, qk_stats=qk,
                                                amax=amax)
            else:
                leaves = tuple(a + b for a, b in zip(
                    previous.statistics, jax.tree.leaves(stats), strict=True))
                pooled = self.stats_tree.unflatten(leaves)
                value, active = self.objective.reduce_loss(pooled)
                candidate = dataclasses.replace(previous, statistics=leaves, effects=effects, qk_stats=qk,
                                                amax=amax)
                replay_required = due & local_ok
            loss = jnp.where(due, value, loss)
        return PendingAttempt(fill, due, active, loss, local_finite, replay_required,
                              pooled, effects, qk, candidate, gradient, own=stats, amax=amax)

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

    def phases(self, host: bool) -> tuple[Callable, ...]:
        """The step's phases (`step`), each a program over the objective's modules for a host step, compiled
        through Dew's compile point as the step's own compile is (`owned_program`); traced in place for a
        resident one. Among them its glue between the others (`begun`, the replay's
        and the records' arithmetic), each a program of its own: an eager operation would trace and compile
        at a host step's first call, where the run steps on the programs its startup compiled
        (`dew.training.compiler`, `options.collecting`)."""
        def compile_phase(phase: Callable) -> Callable:
            return owned_program(phase) if host else phase

        return tuple(compile_phase(phase) for phase in (
            self.local_reduction, self.prepare_attempt, self.pooled_cotangent, self.finish_attempt,
            self.unscaled, self.add_contribution, self.begun, self.replayed_input, self.unreplayed,
            self.recorded_into, self.kept_into))

    def replay_input(self, state, fill, index, phase=None, read=None):
        """Rebuild the inputs of the window's `index`-th microbatch.

        The batch and the mutable collections come from the retained buffers.
        The key folds in the attempt that first read them, so a replayed
        microbatch draws the randomness it drew before. `phase` runs their
        arithmetic (`replayed_input`), and `read` the variables' (`Read`), a
        host step's programs; traced in place where None.
        """
        batch, mutable, step_info = (phase or self.replayed_input)(state, fill, index)
        variables = state.variables if mutable is None else {**state.variables, **mutable}
        step_info = dataclasses.replace(step_info, ema=with_ema(variables, self.fetched(state).ema))
        return (read or self.read)(variables, state.compute), batch, step_info

    def replayed_input(self, state, fill, index):
        """`replay_input`'s arithmetic: the window's `index`-th microbatch, its mutable collections where
        the window keeps them, and its `Step` but for its averaged weights beside its variables
        (`with_ema`), which the replay reads as the state holds them."""
        previous = state.accumulation
        assert previous is not None and previous.attempts is not None
        batch = jax.tree.map(lambda x: x[index], previous.batches)
        mutable = (None if previous.variables is None
                   else jax.tree.map(lambda x: x[index], previous.variables))
        variables = state.variables if mutable is None else {**state.variables, **mutable}
        accepted, attempt = state.microstep - fill + index, previous.attempts[index]
        # The part its input ran as, which an attempt counts (`step` times `split` plus its part).
        replayed_stats = (previous.batch_statistics if previous.replayed_statistics is None
                          else tuple(x[index] for x in previous.replayed_statistics))
        step_info = self.attempt(state, variables, batch, attempt, attempt % self.split, index,
                                 microstep=accepted, batch_statistics=replayed_stats)
        return batch, mutable, dataclasses.replace(step_info, ema=None)

    def begun(self, state, read, batch, part, closes, whole):
        """An attempt's arithmetic before its realization: the window's held batch statistics and their
        stamp where they are computed (`_step_attempt`; None where the window holds none), the attempt's
        `Step` but for its averaged weights beside its variables (`with_ema`), which the step reads as the
        state holds them, the loss scale's factor, the attempt's count (`step` times `split` plus its
        `part`) and whether it closes the window (`closes`, at the step's last part)."""
        # Every attempt the run made and did not accept draws afresh (`attempt_step`'s rejections).
        attempt = state.step * self.split + part
        slot = occupancy(state.accumulation)
        factor = jnp.asarray(1., jnp.float32) if state.scale is None else state.scale.scale
        stepped, step_info = self._step_attempt(state, read, batch, attempt, part, slot, whole)
        held = stepped.accumulation
        statistics = (None if held is state.accumulation or held is None
                      else (held.batch_statistics, held.statistics_stamp))
        # A run whose stream ends inside the window closes it at the step's last part (`closes`).
        closing = None if closes is None else jnp.logical_and(closes, part == self.split - 1)
        return statistics, dataclasses.replace(step_info, ema=None), factor, attempt, closing

    def unreplayed(self, state, own: Evaluated, part):
        """The records of the window's other microbatches before a replay writes them: zeros, in the slots
        it holds."""
        return [jax.tree.map(jnp.zeros_like, (own, self._places(occupancy(state.accumulation), own, part)))
                for _ in range(self.size - 1)]

    def kept_into(self, state, kept, record: Evaluated, index):
        """`kept`, the window's records, with the `index`-th microbatch's replayed `record` in its slot."""
        assert state.accumulation is not None and state.accumulation.attempts is not None
        attempt = state.accumulation.attempts[index]
        evaluated = (record, self._places(index, record, attempt % self.split))
        return [jax.tree.map(lambda held, new, slot=slot: jnp.where(slot == index, new, held), held,
                             evaluated) for slot, held in enumerate(kept)]

    def recorded_into(self, state, part, own, replays, fill, replayed) -> Recorded:
        """The window's records (`recorded`): this microbatch's (`own`, in its slot) and each other one's
        the replay ran (`replays`, those of the `fill` slots the window holds where it `replayed`)."""
        evaluations = [(own, self._places(occupancy(state.accumulation), own, part)), *replays]
        held = jnp.arange(len(replays)) < fill
        ran = jnp.concatenate([jnp.ones((1,), bool), jnp.logical_and(held, replayed)])
        return recorded(evaluations, ran)

    def attempt(self, state, variables, batch, attempt, part, slot, microstep=None, *,
                batch_statistics: tuple[jax.Array, ...] | None = None,
                recorded: bool | None = None, watch: bool = True) -> Step:
        """`attempt_step` of this transaction's split, window, recording and watch."""
        return attempt_step(state, variables, self.fetched(state).ema, batch, split=self.split,
                            slots=self.size, attempt=attempt, part=part, slot=slot, microstep=microstep,
                            record=self.record if recorded is None else recorded,
                            watched=self.watched if watch else None, batch_statistics=batch_statistics)

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

    def _refuse_unheld(self, current) -> None:
        """Refuse, as the step is traced and before any runs, a commit whose clocks could not count the
        updates its optimizer applies (`update_applied`): one whose state does not say which it holds back,
        a held update whose post-step effects no update would apply, and a held one with no tokens
        waiting for it."""
        refuse_untold(current.opt_state)
        if holds_back(current.opt_state) and self.aux_shape.effects is not None:
            raise ValueError("an objective with post-step effects (Aux.effects) under an optimizer that "
                             "holds updates back, whose held calls' effects no update applies; accumulate "
                             "gradients with Trainer(accumulation=k)")
        if holds_back(current.opt_state) and current.deferred is None:
            raise ValueError("the state holds no tokens waiting for the update its optimizer holds back "
                             "(TrainState.deferred): build it with Trainer.initial_state, or restore it "
                             "through the trainer, which gives one written before them none waiting")

    def commit(self, current, gradient, effects, qk, statistics, applied: Applied | None = None,
               attempted: jax.Array | None = None, amax: Variables | None = None):
        """Apply the pooled native gradient and return its finiteness and global norm. An optimizer that
        reads more than the gradient is handed the update's pooled loss `statistics` (a Gauss-Newton
        update's system, `dew.gauss_newton`) and attention logit maxima (`qk`). The window's largest
        recorded magnitudes (`amax`) roll into the delayed-scaling NVFP4 projections' histories as the
        update commits, as its effects apply, no optimizer reading them (row 12721).

        The norm is reduced beside optimizer.update, in the same compiled
        transaction, and leaves it with the loss and auxiliary report. No
        gradient tree crosses to the host or is reconstructed for logging.
        Admission is decided by finish_attempt after checking the updated
        parameters and optimizer state as well.
        """
        self._refuse_unheld(current)
        # The gradient where the parameters are placed (`placed`): left where the backward made it, a
        # stage-local leaf's update came out split over the tensor axis its pipeline's read took, and the
        # commit gathered every parameter and moment back each step (row 3697).
        gradient = gradient if self.placed is None else placed_as(gradient, self.placed)
        native = jax.tree.map(_descending, gradient, current.variables["params"])
        # The gradient's global norm, which the run logs and a clip reads, and its finiteness by it: a value
        # not finite anywhere makes the norm so, with no pass of its own (the owner, 2026-11-16).
        norm = optax.global_norm(native)
        norm = norm.astype(at_least_fp32(norm.dtype))
        native_finite = jnp.isfinite(norm)
        if applied is not None:
            # The runs that updated in their backward hold no gradient here (zeros where theirs would be):
            # their norm and finiteness come from the backward that read them (`Applied`).
            norm = jnp.sqrt(jnp.square(norm) + applied.sumsq.astype(norm.dtype))
            # The whole gradient's norm finite, as the rest's is read, besides each run's own count.
            native_finite = native_finite & (applied.nonfinite == 0) & jnp.isfinite(norm)
        # With no loss scale, a run that skips steps not finite (`skip`, its opt-in) gates one where it is
        # computed: its gradient, the forward's statistics and its update read as zero, so its parameters,
        # update count and EMA stay as they were and its optimizer takes one step of zero gradient, holding
        # no old state beside the new (training.md, a step that is not finite). One that does not takes the
        # step as it is formed. A scaled step's rejection keeps the optimizer, chosen below, as a scale's
        # back-off must, and so does one whose optimizer holds updates back (`holds_back`,
        # `optax.MultiSteps`): a zero input would join its window and close it, where the step's rejection
        # keeps the inputs it holds.
        gated = self.skip and current.scale is None and not holds_back(current.opt_state)
        if gated:
            native_finite = native_finite & all_finite((qk, statistics, effects, amax))
            # The attempt's own finiteness (its loss, its collections, its microbatch), as its acceptance
            # reads it (`finish_attempt`): one gate for every write.
            native_finite = native_finite if attempted is None else native_finite & attempted
            native, qk, statistics, effects, amax = (zeroed_unless(native_finite, value)
                                                     for value in (native, qk, statistics, effects, amax))

        def apply(current):
            # The update applies to the parameters the model's constraint projects
            # (`Objective.projected`), and the step stores them projected again.
            stored = self.objective.projected(current.variables["params"], updated=False)
            progress = None if self.bound is None else self.progress(current)
            update, opt_state = optimizer_update(self.optimizer, native, current.opt_state, stored, qk=qk,
                                                 statistics=statistics, progress=progress)
            update = zeroed_unless(native_finite, update) if gated else update
            # The update moves the weights, and each post-step effect of this update (QK-Clip's
            # rescale, a bounded group's clamp) acts on them (`stepped`): on a gated step's only where it is
            # taken, as the effect's count steps with the zero gradient's step of a rejected one.
            order = post_step_order(self.optimizer, stored)
            moved = stepped(stored, update, current.opt_state, opt_state, order)
            if gated and order:
                moved = _chosen(native_finite, moved, stored)
            params = {**current.variables, "params": moved}
            if applied is not None:
                # The runs that updated in their backward take what their backward wrote, and the update's
                # own of them, read by nothing, is never computed.
                moved = applied.onto(dataclasses.replace(current, variables=params, opt_state=opt_state))
                params, opt_state = moved.variables, moved.opt_state
            params = {**params, "params": self.objective.projected(params["params"], updated=True)}
            done = update_applied(current.opt_state, opt_state) & (native_finite | (not gated))
            if self.aux_shape.effects is not None:
                written = write_back(params, self.objective.apply_effects(
                    params, self.effects_tree.unflatten(effects)))
                # A gated step's forward writes back nothing where it is rejected, for whatever reason: its
                # collections (batch statistics) stay, and so does whatever its effects would move (a
                # router's balancing biases), its parameters among them.
                params = written if not gated else {
                    **written, **{name: _chosen(native_finite, value, params[name])
                                  for name, value in written.items()}}
            if amax is not None:
                # Each history the window recorded a largest for rolls it in, newest, where the update
                # applies (`done`): a gated step's rejection, or an update an optimizer holds back, keeps
                # them as they were.
                params = {**params, NVFP4_HISTORY: _chosen(done, rolled(params[NVFP4_HISTORY], amax),
                                                           params[NVFP4_HISTORY])}
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
            committed, inspected = apply(current)
            return committed, inspected, native_finite, norm

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

    def _retained_window(self, state: TrainState, batch: Batch, pending: PendingAttempt, attempt,
                         read: jax.Array) -> Accumulation:
        """`pending.candidate` cleared when due, else retaining this microbatch's slot, with `gradient`
        and `batch_statistics` bypassing the conditional."""
        candidate = pending.candidate
        assert candidate is not None
        held_stats = candidate.batch_statistics
        held_stamp = candidate.statistics_stamp

        def retain(acc):
            acc = dataclasses.replace(acc, fill=occupancy(acc) + 1)
            if acc.tokens is not None:
                acc = dataclasses.replace(acc, tokens=TokenCount.add(acc.tokens, read))
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
                if acc.replayed_statistics is not None and held_stats is not None:
                    acc = dataclasses.replace(acc, replayed_statistics=tuple(
                        held.at[pending.fill].set(x)
                        for held, x in zip(acc.replayed_statistics, held_stats, strict=True)))
            return acc

        # A pooled window's gradient sum (a replayed window keeps none) and split batch statistics go
        # round the reset too, the next first microbatch replacing them (`prepare_attempt`).
        window = jax.lax.cond(
            pending.due, cleared, retain,
            dataclasses.replace(candidate, gradient=None, batch_statistics=None))
        stamp = (None if held_stamp is None
                 else jnp.where(pending.due, window.statistics_stamp, held_stamp))
        return dataclasses.replace(window, gradient=candidate.gradient, batch_statistics=held_stats,
                                   statistics_stamp=stamp)

    def _finiteness(self, scale, holding: bool, pending, aux, gradient) -> tuple[bool, bool, jax.Array]:
        """Whether the step is gated where it is not finite (no loss scale, `skip`, in its commit), whether
        it is taken as it is formed (no loss scale, not `skip`), and its attempt's finiteness: one taken as
        formed reads its loss alone, its gradient by its commit's norm; one gated its small trees too; a
        scaled one, or one an optimizer holds back, its gradient besides."""
        unscaled = scale is None and not holding
        gated, taken_as_formed = unscaled and self.skip, unscaled and not self.skip
        trees = (pending.effects, pending.qk, pending.amax, aux.variables, *(() if unscaled else (gradient,)))
        read = (pending.loss,) if taken_as_formed else (pending.loss, *trees)
        return gated, taken_as_formed, pending.local_finite & all_finite(read)

    def finish_attempt(self, state, batch, aux, pending, gradient, attempt, tokens, probed=None):
        """Commit the window when it is due, then advance the state's clocks.

        A rejected attempt costs the same trace as an accepted one. With a
        dynamic scaler an attempt is admitted on its loss and gradient finite alone, as GradScaler's
        `found_inf` (row 6748), its committed state's finiteness reported after (`unfinite_state`); one
        whose optimizer holds updates back on its whole new state finite. A rejected attempt returns `state`
        with its window dropped (`dropped_window`), its parameters, optimizer state and EMA untouched, and
        the scaler moved.

        A window of one microbatch commits on every step it can, by elementwise selects; a longer window
        under `lax.cond`, which skips the optimizer on the microbatches that do not close it. A GPU
        conditional whose branch holds a collective (fsdp's gradient norm) reads its predicate on the host,
        which then idles the device through every step's host work (15.9 ms a 32.5 ms DiT step on four
        RTX 3090s).

        A window that is not due retains this microbatch's batch, read
        snapshots and `attempt` in its slot, so a later replay can reproduce
        it; a window that committed is zeroed for the next one, but for its
        gradient sum, which the next window's first microbatch replaces.
        """
        scale, previous = state.scale, state.accumulation
        effects, qk, loss = pending.effects, pending.qk, pending.loss
        holding = holds_back(state.opt_state)
        gated, taken_as_formed, finite = self._finiteness(scale, holding, pending, aux, gradient)
        # A step with no loss scale is gated in its commit (`commit`), but one whose optimizer holds updates
        # back is rejected whole, its optimizer state kept, as a scaled step's is.
        accepted = jnp.asarray(a=True) if scale is None and not holding else finite
        numerical = dataclasses.replace(state, variables=write_back(state.variables, aux.variables),
                                        microstep=state.microstep + 1)
        # Runs that updated in their backward wrote it only where the step commits (`in_backward.gated`).
        numerical = numerical if pending.applied is None else pending.applied.onto(numerical)

        # What the commit's own finiteness (its pooled gradient, effects and logit maxima) leaves out: the
        # attempt's loss and its collections. A microbatch's own gradient (`local_finite`) is in the pooled
        # one; read here, its reduction would hold the window's step back (a tenth more of a split step's
        # need in test_plan's tight start).
        attempted = all_finite((loss, aux.variables)) if gated else None

        def commit(current):
            return self.commit(current, gradient, effects, qk, pending.pooled, pending.applied, attempted,
                               pending.amax)

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
        held = read if previous is None or previous.tokens is None else TokenCount.add(previous.tokens, read)
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
        if gated:
            numerical = _collections_kept(finite, numerical, state, aux.variables)
        if holding or scale is not None:
            # Taken whole or not: holding updates back (`optax.MultiSteps`), on its whole new state finite; a
            # scaled step on its loss and gradient alone, as GradScaler's `found_inf`, in place (row 6748).
            accepted = finite & all_finite(checked(numerical)) if holding else finite
        if previous is not None:
            numerical = dataclasses.replace(
                numerical, accumulation=self._retained_window(state, batch, pending, attempt, read))
        # A rejected attempt drops its window (`dropped_window`): the update it would have closed is
        # skipped whole, its accepted microbatches with it.
        advanced = _chosen(accepted, numerical, dropped_window(state))
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
        # A gated step (no loss scale, skipped where not finite) committed nothing: no update, no tokens
        # (`commit`); one taken as formed committed its update whatever it held.
        done = update_applied(state.opt_state, advanced.opt_state) & (finite | (scale is not None)
                                                                      | taken_as_formed)
        counted = due & accepted & done
        if state.deferred is not None:
            taken = input_taken(state.opt_state, advanced.opt_state)
            held = TokenCount.add(state.deferred, jnp.where(taken, held, 0).astype(jnp.uint32))
            trained = finite | (scale is not None) | taken_as_formed
            waiting = jnp.where(due & accepted & trained & ~done, held, jnp.where(counted, 0, state.deferred))
            advanced = dataclasses.replace(advanced, deferred=waiting.astype(jnp.uint32))
        if self.split == 1:
            # A step that commits no update, a window still filling or one its optimizer holds back
            # (`update_applied`), logs its own batch's loss, its window's pooled at one that commits
            # (training.md, the logged loss); a step read in several parts pools its parts' own
            # (`whole_step`).
            own = self.objective.reduce_loss(pending.own)[0].astype(loss.dtype)
            loss = jnp.where(counted, loss, own)
        return advanced, loss, aux, Committed(
            jnp.where(counted, norm, 0.), counted.astype(jnp.int32),
            jnp.where(counted, held, jnp.zeros((2,), jnp.uint32)), stacked(inspected, counted),
            own=pending.own if self.split > 1 else None,
            finite=(accepted if holding else finite) if scale is None else None,
            unfinite_layer=None if pending.applied is None else pending.applied.unfinite_layer,
            unfinite_state=None if scale is None or holding else unfinite_state(checked(advanced)))

    @staticmethod
    def _run_norms(params: Variables, runs: Sequence[tuple[str, ...]]) -> Variables:
        """Each run's norms the walk writes (`dew.nn.in_backward.NORMS`): a layer's sum of squares and count
        not finite, under its period's where its run is periodic."""
        norms: Variables = {}
        for path in runs:
            run = _at(params, path)
            count = jax.tree.leaves(run)[0].shape[1]
            zeros = jnp.zeros((1, count), jnp.float32)
            held = {"sumsq": zeros, "nonfinite": zeros}
            periodic = all(str(key).startswith("layers_") for key in run)
            norms = merge(norms, _nested(path, {in_backward.PERIOD_NORMS: held} if periodic else held))
        return norms

    def _planned_accumulation(self, state: TrainState) -> _Planned | None:
        """In-backward gradient pooling for a shared-mean accumulating window (`self.size > 1`): each
        scanned layer's unscaled gradient is pooled directly into `state.accumulation.gradient` during the
        backward walk, so no stacked `[L, ...]` gradient buffer is materialized."""
        if (not self.shared or state.scale is not None or holds_back(state.opt_state)
                or FROZEN in state.variables):
            return None
        previous = state.accumulation
        if previous is None or previous.gradient is None or not in_device(previous.gradient):
            return None
        params = state.variables["params"]
        runs = _stage_runs(params)
        if not runs:
            return None
        pooling = _Pooling()

        def accumulate(path, index, gradient, held, moments):
            del path, index, moments
            fill, prior, given, factor = pooling.fill, pooling.old, pooling.new, pooling.factor
            # Set once the step's forward is (`_pulled`): no backward runs before it, a loss differentiating
            # a run itself planned none (`_read_once`).
            assert fill is not None and prior is not None and given is not None and factor is not None

            def pooled(old, new):
                return _pooled_leaf(fill, prior, given, old, _unscale(new, factor))

            return jax.tree.map(pooled, held, gradient), {}

        read = {in_backward.MASTERS: previous.gradient, in_backward.NORMS: self._run_norms(params, runs)}
        return _Planned(in_backward.InBackward(update=accumulate, moments=()), read, pooling=pooling,
                        runs=tuple(runs))

    def planned(self, state: TrainState) -> _Planned | None:
        """The plan by which the runs a one-stage scanned state holds stacked update or accumulate in their
        backward (`dew.nn.in_backward`): for a shared-mean window of several microbatches (`self.size > 1`),
        each layer's unscaled gradient pools in place into `state.accumulation.gradient`
        (`_planned_accumulation`); for a window of one read whole, where the step reads each gradient where
        it is made into the update (`consumed_where_made`), with no post-step effect, sampled summary,
        QK-Clip, held update or adapter (a frozen base), and its optimizer holds what each parameter owns in
        copies of their tree (`_held_in_copies`); None elsewhere. A run the model cannot update so (a routed
        or adapted one) runs as before.

        The backward writes where the step commits (`dew.nn.in_backward.gated`, its loss's `active`): a
        window of one under no loss scale (`consumed_where_made` refuses `TrainState.scale`) commits every
        attempt it takes, so what the backward wrote is what the commit keeps (`Applied.onto`), on the
        state the window leaves as on the one it commits."""
        if self.size > 1:
            # A step that inspects what its layers made reads their scan's own collections, which a layer
            # applied on its own opens none of: it pools no run in its backward.
            return self._planned_accumulation(state) if self.inspection is None else None
        if self.split != 1 or self.inspection is not None:
            return None
        if self.aux_shape.effects is not None or FROZEN in state.variables:
            return None
        if holds_back(state.opt_state) or clips_qk(state.opt_state):
            return None
        # The model's runs and its optimizer's copies first, read off the trees: the optimizer's update is
        # traced (`consumed_where_made`) only for a run that could be updated so (row 6740: a model with no
        # stacked run traced a whole update it then discarded).
        params = state.variables["params"]
        runs = _stage_runs(params)
        if not runs or not _held_in_copies(state.opt_state, params):
            return None
        if not consumed_where_made(self.optimizer, state, 1, shared=self.shared,
                                   progressed=self.bound is not None):
            return None
        copies = _param_copies(state.opt_state, params)
        names = tuple(f"{in_backward.MOMENTS}_{index}" for index in range(len(copies)))
        progress = None if self.bound is None else self.progress(state)

        def update(path, index, gradient, weights, moments):
            # One layer of the run as the step's update reads it: its tree under the run's path with the
            # stage and layer axes of one, so every rule of the update by path reads it as its whole.
            where = _state_path(path)

            def layer(tree):
                return _nested(where, jax.tree.map(lambda leaf: leaf[None, None], tree))

            def unlayer(tree):
                return jax.tree.map(lambda leaf: leaf[0, 0], _at(tree, where))

            held = layer(weights)
            copies = [layer(moments[name]) for name in names]
            before = _with_copies(_placeholders_at(state.opt_state, params, where), params,
                                  lambda index, _: copies[index])
            descent = layer(jax.tree.map(_descending, gradient, weights))

            moved, after = optimizer_update(self.optimizer, descent, before, held, progress=progress)
            stepped_held = stepped(held, moved, before, after, post_step_order(self.optimizer, held))
            # Each copy's new value where the update leaves it in the state it read, which the update keeps
            # laid out as it was: a group with no parameter in this run holds placeholders alone there.
            laid = _laid_as(held)
            slots = [slot for copy in copies
                     for slot, node in enumerate(jax.tree.leaves(before, is_leaf=laid)) if node is copy]
            stepped_copies = [jax.tree.leaves(after, is_leaf=laid)[slot] for slot in slots]
            return unlayer(stepped_held), {name: unlayer(copy) for name, copy in zip(names, stepped_copies,
                                                                                   strict=True)}

        read = {**dict(zip(names, copies, strict=True)), in_backward.NORMS: self._run_norms(params, runs)}
        if state.compute is not None:
            # The masters the step's update applies to, as its constraint projects them (`Transaction.read`).
            read[in_backward.MASTERS] = self.objective.projected(params, updated=False)
        return _Planned(in_backward.InBackward(update=update, moments=names), read, runs=tuple(runs))

    def realize_planned(self, planned: _Planned) -> Callable[[Variables, Batch, Step], Realization]:
        """`realize`, differentiating the collections `planned` adds beside the parameters, whose
        cotangents are what the runs that update in their backward leave (`split_planned`)."""
        def realize(variables, batch, step_info):
            def loss(trainable):
                return self.objective._loss({**variables, **trainable}, batch, step_info)
            stats, back, aux = jax.vjp(loss, {"params": variables["params"], **planned.read}, has_aux=True)
            return Realization(stats, aux, lambda cotangent: back(cotangent)[0])
        return realize

    @staticmethod
    def split_planned(planned: _Planned, cotangents, masters: bool) -> tuple[Variables, Applied | None]:
        """The step's gradient and what the runs that updated in their backward left (`Applied`), from the
        cotangents of `realize_planned`: those runs' gradient zeros, their new state read off the
        cotangents of the parameters (or their `masters`) and moments."""
        gradient = cotangents["params"]
        handled = [_state_path(path) for path in planned.plan.handled]
        if not handled:
            return gradient, None
        source = cotangents[in_backward.MASTERS] if masters else gradient
        norms = cotangents[in_backward.NORMS]
        applied = Applied(
            params=functools.reduce(merge, [_nested(path, _at(source, path)) for path in handled], {}),
            moments=tuple(functools.reduce(merge, [_nested(path, _at(cotangents[name], path))
                                                   for path in handled], {})
                          for name in planned.plan.moments),
            sumsq=sum((jnp.sum(in_backward.norms_of(_at(norms, path))["sumsq"]) for path in handled),
                      jnp.zeros((), jnp.float32)),
            nonfinite=sum((jnp.sum(in_backward.norms_of(_at(norms, path))["nonfinite"]) for path in handled),
                          jnp.zeros((), jnp.float32)),
            unfinite_layer=_unfinite_layer([(in_backward.norms_of(_at(norms, path))["nonfinite"],
                                             _period(_at(gradient, path))) for path in handled]))
        zeros = functools.reduce(merge, [_nested(path, jax.tree.map(jnp.zeros_like, _at(gradient, path)))
                                         for path in handled], {})
        return merge(gradient, zeros), applied

    def _realized_in(self, state: TrainState, realize, read, batch, step_info, factor, *,
                     host: bool) -> tuple[Realization, _Planned | None]:
        """The microbatch realized, inside the plan by which its runs update in their backward where the
        step has one (`planned`; not a host step's, one that summarizes its cotangents, nor one realized
        otherwise), and that plan."""
        watching = step_info.watched is not None and bool(step_info.watched.cotangents)
        planned = None if host or watching or realize != self.realize else self.planned(state)
        if planned is not None and not self._read_once(planned, read, batch, step_info, factor):
            planned = None
        if planned is None:
            return self._realized(realize, read, batch, step_info, factor), None
        with in_backward.planned(planned.plan):
            realized = self._realized(self.realize_planned(planned), read, batch, step_info, factor)
        return realized, planned

    def _lean(self, state: TrainState, planned: _Planned | None, probing, *, host: bool) -> bool:
        """Whether the window the step replays, its loss unscaled and its update taken whole, reads no
        gradient of a microbatch's own (`_pulled`): the commit replays the window's every microbatch, this
        one realized afresh, under the pooled loss's cotangent, so its forward's residuals live through no
        backward and no replay (a decision step whose loss weighs its consistency term, 8 layers of a 4B
        model at split 2: 28.4 GiB a step and 82 s to compile against 18.5 GiB and 43 s). A host step, one
        that updates its runs in their backward or summarizes its cotangents takes its own."""
        return (not host and state.accumulation is not None and not self.shared and state.scale is None
                and not holds_back(state.opt_state) and planned is None and probing is None)

    def _read_once(self, planned: _Planned, read: Variables, batch: Batch, step_info, factor) -> bool:
        """Whether every run the state stacks (`_Planned.runs`) is read by the loss once, in a call of its
        own rule at the loss's own level that nothing else of the loss reads its variables beside
        (`in_backward.read_once`), and whose backward the step's own pullback runs: it then updates or
        pools exactly once a microbatch, its parameters' gradient all its own, the donated state it writes
        read by nothing after it. A run called in a loop or a branch (whose body runs as many times as it
        steps, or none), within a nested jit (whose cached program a later trace reads without calling the
        run), under a batching transform, drawing random bits, whose parameters the loss reads beside it or
        lays out otherwise than the state it writes (a transposed weight), whose output the loss detaches
        (its backward never run), or that the loss does not call at all, is updated by no backward: the step
        runs as it would unscanned. Read off the loss's program traced under a plan of its own, and off the
        step's realization, reduction and pullback (`realize_planned`, `local_reduction`) traced under one
        whose update records the runs it reaches."""
        # A backward its own primal trace takes (a loss differentiating a run itself) calls the update.
        inner: list[tuple[str, ...]] = []

        def differentiated(path, index, gradient, held, moments):
            del index, gradient
            inner.append(path)
            return held, dict(moments)

        probe = dataclasses.replace(planned.plan, handled=[], shapes={}, sources={}, update=differentiated)

        def loss(trainable):
            return self.objective._loss({**read, **trainable}, batch, step_info)

        trainable = {"params": read["params"], **planned.read}
        with unscaled(factor), fixed_routers(), in_backward.planned(probe):
            program = jax.make_jaxpr(loss)(trainable)
        handled = [_state_path(path) for path in probe.handled]
        if inner or sorted(handled) != sorted(planned.runs):
            # A run the loss differentiates itself, does not call, calls where no trace registers it (a
            # nested jit's cached program), or calls more than once (a loss scoring it twice).
            return False
        shapes = {_state_path(path): called for path, called in probe.shapes.items()}
        sources = {_state_path(path): read for path, read in probe.sources.items()}
        roots = {}
        for (path, _), value in zip(jax.tree_util.tree_flatten_with_path(trainable)[0], program.jaxpr.invars,
                                    strict=True):
            names = tuple(str(entry.key) for entry in path[1:] if isinstance(entry, jax.tree_util.DictKey))
            run = next((at for at in handled if names[:len(at)] == at), None)
            if run is not None and isinstance(path[0], jax.tree_util.DictKey):
                roots[value] = in_backward.RunRead(run, shapes[run], sources[run], str(path[0].key),
                                                   names[len(run):])
        if not in_backward.read_once(program.jaxpr, roots, handled):
            return False
        # Live in the step's own pullback: each run's rule reached by the backward the step runs, its update
        # called for exactly the runs registered, never for one detached.
        reached: list[tuple[str, ...]] = []

        def recorded(path, index, gradient, held, moments):
            del index, gradient
            reached.append(_state_path(path))
            return held, dict(moments)

        live = dataclasses.replace(planned, plan=dataclasses.replace(planned.plan, handled=[], shapes={},
                                                                     sources={}, update=recorded))

        def pulled():
            realized = self._realized(self.realize_planned(live), read, batch, step_info, factor)
            _, cotangent = self.local_reduction(realized.stats, factor)
            return realized.pullback(cotangent)

        with in_backward.planned(live.plan):
            jax.make_jaxpr(pulled)()
        return set(reached) == set(handled)

    def _pulled(self, state: TrainState, realized: Realization, cotangent, planned: _Planned | None,
                factor, *, whole: bool, masters: bool, lean: bool = False
                ) -> tuple[Variables, Variables | None, Applied | None, tuple[tuple[str, ...], ...]]:
        """The microbatch's gradient, its probes' where it summarizes them from the same backward (`whole`),
        what its runs that updated in their backward left (`split_planned`), and the run paths already
        unscaled and pooled into `state.accumulation.gradient` in the backward (`_planned_accumulation`);
        zeros of its shape, no backward run, for a lean window, whose commit replays it (`_lean`)."""
        if lean:
            shapes = jax.eval_shape(realized.pullback, cotangent)
            return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes), None, None, ()
        if planned is not None:
            if planned.pooling is not None:
                previous = state.accumulation
                assert previous is not None
                prior_mass, _, mass, _, _, denominator = _pooling_weights(previous, realized.stats)
                planned.pooling.fill, planned.pooling.old = occupancy(previous), prior_mass / denominator
                planned.pooling.new, planned.pooling.factor = mass / denominator, factor
                cotangents = realized.pullback(cotangent)
                gradient = cotangents["params"]
                handled = tuple(_state_path(path) for path in planned.plan.handled)
                if not handled:
                    return gradient, None, None, ()
                # Each run read once, its parameters by nothing else (`_read_once`): its window sum is the
                # one its backward pooled in place, its parameters' own cotangent none.
                accumulated = cotangents[in_backward.MASTERS]
                pooled = functools.reduce(merge, [_nested(path, _at(accumulated, path)) for path in handled],
                                          {})
                return merge(gradient, pooled), None, None, handled
            loss, active = self.objective.reduce_loss(realized.stats)
            # A run that updates in its backward holds no loss scale (`consumed_where_made`), so where it
            # skips steps not finite (`skip`) each is gated layer by layer as the walk reaches it: by the
            # loss's, the forward's statistics' and collections' finiteness, as the commit and the attempt
            # read them, and each layer's own gradient's (`in_backward.gated`); else the walk updates each
            # layer as it is formed.
            finite = (jnp.isfinite(loss) & all_finite((realized.stats, realized.aux.variables))
                      if self.skip else None)
            with in_backward.gated(active, finite):
                gradient, applied = self.split_planned(planned, realized.pullback(cotangent), masters)
            return gradient, None, applied, ()
        if realized.probed is None or not whole:
            return realized.pullback(cotangent), None, None, ()
        gradient, probed = realized.probed(cotangent)
        return gradient, probed, None, ()

    def _realized(self, realize, read, batch, step_info, factor):
        """The attempt `realize`d unscaled by `factor`, refused where its loss weighs a router it trains by a
        fixed weight that no term of it reaches (`refuse_fixed_routers`)."""
        with unscaled(factor), fixed_routers() as fixed:
            realized = realize(read, batch, step_info)
        refuse_fixed_routers(fixed)
        return realized

    def _step_attempt(self, state, read: Variables, batch: Batch, attempt, part, slot,
                      whole: Batch | None):
        """`state` and the `Step` for `part`, computing `batch_statistics` at `part` 0 or whenever the
        update/rejection stamp moved and holding them on `state.accumulation` across parts."""
        held_stats = None if state.accumulation is None else state.accumulation.batch_statistics
        held_stamp = None if state.accumulation is None else state.accumulation.statistics_stamp
        batch_stats = held_stats
        if held_stats is not None and held_stamp is not None and whole is not None:
            rejected = attempt - state.microstep
            stamp = jnp.stack([state.updates, rejected]).astype(jnp.int32)
            base_slot = slot - slot % self.split
            base_microstep = state.microstep - state.microstep % self.split

            def compute_batch_stats(_):
                def part_at(index: jax.Array) -> tuple[Batch, Step]:
                    micro = microbatch(whole, index, self.split)
                    step_i = self.attempt(state, state.variables, micro,
                                          base_microstep + index + rejected, index, base_slot + index,
                                          microstep=base_microstep + index, recorded=False, watch=False)
                    return micro, step_i

                found = self.objective.batch_statistics(jax.lax.stop_gradient(read), part_at, self.split)
                assert found is not None
                return tuple(jax.lax.stop_gradient(x).astype(held.dtype)
                             for x, held in zip(found, held_stats, strict=True))

            stale = (part == 0) | jnp.any(held_stamp != stamp)
            batch_stats = jax.lax.cond(stale, compute_batch_stats, lambda _: held_stats, None)
            assert state.accumulation is not None
            state = dataclasses.replace(
                state, accumulation=dataclasses.replace(state.accumulation, batch_statistics=batch_stats,
                                                        statistics_stamp=stamp))
        return state, self.attempt(state, state.variables, batch, attempt, part, slot,
                                   batch_statistics=batch_stats)

    def step(self, *, realize=None, host=False, frozen: Sequence[tuple[str, ...]] = ((FROZEN,),)):
        """Build the function that runs one attempt over the shared phases.

        A host-master step compiles each phase on its own and replays in a
        Python loop, because its realizations cross an eager transport
        boundary. A resident step traces the phases together and replays
        inside `lax` control flow. A host step's commit reads none of the
        trees at `frozen`, those it keeps beside the accelerator
        (`dew.training.execution.frozen_trees`): they cross to no other
        device and stay the arrays the state holds (`apart`).
        """
        realize = self.realize if realize is None else realize
        (reduce, prepare, pooled, finish, unscale, add, begin, replayed_in, unreplayed_of, recorded_into,
         kept_into) = self.phases(host)
        # The variables a host step's forward reads, their projection's arithmetic a program (`Read`).
        reading = Read(self.read) if host else self.read

        def run(state, batch, part, tokens, closes=None, whole=None):
            # The `part`th microbatch of the step's batch is attempt step *
            # split + part, with a draw of its own; the objective's schedules
            # count steps, however many microbatches each is read in. `tokens`
            # is its real tokens, which the trainer reads off the accounting
            # columns the objective's batch no longer holds (`model_batch`).
            # A watched cotangent is read unscaled by the factor its backward is seeded with.
            read = reading(state.variables, state.compute)
            statistics, step_info, factor, attempt, closing = begin(state, read, batch, part, closes, whole)
            if statistics is not None:
                assert state.accumulation is not None
                state = dataclasses.replace(state, accumulation=dataclasses.replace(
                    state.accumulation, batch_statistics=statistics[0], statistics_stamp=statistics[1]))
            # The averaged weights the attempt reads beside its variables, as the state holds them (`begun`
            # hands none back: `attempt_step`'s `with_ema`).
            step_info = dataclasses.replace(step_info, ema=with_ema(state.variables, self.fetched(state).ema))
            # The runs a one-stage scanned state holds update in their backward where the step's update
            # reads each gradient alone (`planned`); a host step, or one realized otherwise, as before.
            realized, planned = self._realized_in(state, realize, read, batch, step_info, factor, host=host)
            loss, cotangent = reduce(realized.stats, factor)
            # Probes' gradient from this backward where the microbatch is its whole window, else the pooled.
            probing = realized.probed
            whole = probing is not None and self.size == 1
            gradient, probed, applied, pooled_paths = self._pulled(
                state, realized, cotangent, planned, factor, whole=whole, masters=state.compute is not None,
                lean=(lean := self._lean(state, planned, probing, host=host)))
            if state.compute is None or self.size > 1:
                gradient = _unscaled_beside(gradient, factor, pooled_paths) if pooled_paths else unscale(
                    gradient, factor)
            # Otherwise no loss scale divides it (`dew.training._candidates.prepared_for`), and a copied
            # parameter's gradient stays bf16 until the update widens it as it reads it. A window of several
            # microbatches widens it as it pools it, so the window sums in fp32.
            pending = prepare(state, realized.stats, realized.aux, loss, gradient, closing,
                              *((pooled_paths,) if pooled_paths else ())).replace(applied=applied)
            if probing is not None and not whole:
                # The probes summarize the cotangent of the update's loss, the window's pooled one, as replay
                # differentiates it (`pooled_cotangent`), whatever microbatches the plan splits it into.
                probed = probing(self.update_seed(pending.pooled, realized.stats, factor))[1]

            # The record of each of the window's other microbatches the commit replays, in its slot.
            own = evaluation_of(realized.aux)
            unreplayed = None if own is None else unreplayed_of(state, own, part)
            # A host step's pass over shapes (its run's startup, `options.collecting`) reads no value: it
            # takes each arm a value chooses, a replay of one microbatch among them, so every program the
            # step can call compiles there.
            over_shapes = host and isinstance(pending.fill, jax.ShapeDtypeStruct)

            def replay(_):
                cot = pooled(pending.pooled, realized.stats, factor)
                current = self._realized(realize, read, batch, step_info, factor) if lean else realized
                combined = unscale(current.pullback(cot), factor)
                def one(index, carried):
                    accumulated, kept = carried
                    replayed, records, step_info = self.replay_input(state, pending.fill, index,
                                                                     replayed_in if host else None,
                                                                     reading)
                    realized = realize(replayed, records, step_info)
                    replayed_record = evaluation_of(realized.aux)
                    if kept is not None and replayed_record is not None:
                        kept = kept_into(state, kept, replayed_record, index)
                    return add(accumulated, realized.pullback(cot), factor), kept
                # The window's other microbatches stand in its slots before this one's: all of them
                # where the window is whole, those it holds where a closing attempt ends it early.
                if host:
                    carried = (combined, unreplayed)
                    for index in range(1 if over_shapes else int(pending.fill)):
                        carried = one(np.int32(index), carried)
                    return carried
                return jax.lax.fori_loop(0, pending.fill, one, (combined, unreplayed))

            # A constant, which a host step's commit reads with no program of its own.
            replayed = np.zeros((), bool)
            if state.accumulation is not None and not self.shared:
                replayed = pending.replay_required
                if over_shapes:
                    replay(None)
                    gradient, replays = pending.gradient, unreplayed
                elif host:
                    gradient, replays = replay(None) if bool(replayed) else (pending.gradient, unreplayed)
                else:
                    gradient, replays = jax.lax.cond(
                        replayed, replay, lambda _: (pending.gradient, unreplayed), None)
            else:
                gradient, replays = pending.gradient, []
            if own is not None:
                realized = dataclasses.replace(realized, aux=dataclasses.replace(realized.aux, record=(
                    recorded_into(state, part, own, replays or [], pending.fill, replayed))))
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
            moving, held = apart(state, frozen)
            advanced, loss, aux, committed = finish(moving, batch, realized.aux, pending, gradient, attempt,
                                                    tokens, probed)
            return together(advanced, held), loss, aux, committed
        return run


def gradients_read(program: jcore.Jaxpr, read: Sequence[frozenset[int]]) -> list[frozenset[int]]:
    """Which gradient leaves each result of `program` reads, its operands reading `read`: a call's
    (`called`, an optax tree helper's `jit` among them) through its program, any other
    equation each of its results all its operands read."""
    reads = dict(zip(program.invars, read, strict=True))
    for equation in program.eqns:
        operands = [reads.get(value, frozenset()) if computed(value) else frozenset()
                    for value in equation.invars]
        program_called = called(equation)
        if program_called is not None:
            reads.update(zip(equation.outvars, gradients_read(program_called, operands), strict=True))
        else:
            reads.update(dict.fromkeys(equation.outvars, frozenset().union(*operands)))
    return [reads.get(value, frozenset()) if computed(value) else frozenset() for value in program.outvars]


def traced_loss[LossT, EffectsT](objective: Objective[LossT, EffectsT], state: TrainState, window: int,
                                 split: int) -> tuple[Callable, tuple]:
    """The loss the step evaluates for `objective` over `state`, its update a `window` of microbatches,
    each step read in `split`, with the abstract arguments it is traced over (but the batch): the
    variables, the microstep, the key, the step, the EMA and the updates committed. Shapes
    and dtypes only: a resident frozen leaf sits in another memory space than the moving ones, and
    the loss the realization runs reads a snapshot in one space. The step's key is drawn inside the
    trace, so an abstract state compiles a step as a placed one does."""
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

    def loss(variables, batch, microstep, key, step, ema, updates):
        shared = batch_statistics_shapes(objective, state, batch, window, split)
        stats = None if shared is None else tuple(jnp.zeros(x.shape, x.dtype) for x in shared)
        return objective._loss(variables, batch, _first_attempt(state, window, split, None, variables, batch,
                                                                microstep, key, step, ema, updates, stats))

    return loss, (variables, state.microstep, state.key, state.step, state.ema, state.updates)


def _first_attempt(state, window: int, split: int, watched: Watched | None, variables, batch, microstep, key,
                   count, ema, updates, statistics) -> Step:
    """The `Step` of a step's first attempt of its first part over `state` at these of its fields, as the
    transaction reads it (`attempt_step`), watching `watched` where given (a sampled step's): what a trace
    of the step's loss alone takes (`traced_loss`, `loss_arguments`)."""
    current = dataclasses.replace(state, microstep=microstep, key=key, step=count, updates=updates)
    return attempt_step(current, variables, ema, batch, split=split, slots=window, attempt=count * split,
                        part=0, slot=microstep % window, batch_statistics=statistics, watched=watched)


def loss_arguments[LossT, EffectsT](objective: Objective[LossT, EffectsT], state: TrainState, batch: Batch,
                                    window: int, split: int, *, watched: Watched | None = None
                                    ) -> tuple[Variables, Step]:
    """The abstract variables and `Step` a step evaluates `objective`'s loss over (`objective._loss`), for
    `state` and a microbatch of `batch`'s shapes, its update a `window` of microbatches, each step read in
    `split`: its first attempt's of its first part (`_first_attempt`), a sampled step's where it `watched`
    its outputs (`Trainer.inspection`), every value a step varies (its key, its microstep, its examples, its
    pooled batch statistics) among its leaves, shapes and dtypes alone: what a step's loss is compiled over
    apart from the step (a streamed step's priced program, `dew.training.streaming`)."""
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)
    shared = batch_statistics_shapes(objective, state, batch, window, split)
    attempted = functools.partial(_first_attempt, state, window, split, watched)
    abstract = jax.eval_shape(attempted, variables, batch, state.microstep, state.key, state.step, state.ema,
                              state.updates, None if shared is None else tuple(shared))
    return variables, abstract


def first_non_finite(state: TrainState) -> str | None:
    """The first leaf of `state`'s checked trees (`transaction.checked`: the parameters, every optimizer
    leaf and the averages, and under DiLoCo each replica's inner copy of them) holding a value that is not
    finite, by its path; None where none does. Read wherever the leaves are placed, one compiled check a
    placement (`transaction.finite_flags`), the same on every process."""
    trees = {**checked(state), **({} if state.inner is None else
                                  {f"inner.{name}": tree for name, tree in checked(state.inner).items()})}
    leaves = [(f"{name}{jax.tree_util.keystr(path)}", leaf) for name, tree in trees.items()
              for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
              if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact)]
    finite = finite_flags([leaf for _, leaf in leaves])
    return next((path for (path, _), held in zip(leaves, finite, strict=True) if not held), None)


def refuse_non_finite(state: TrainState, step: int, latest: Callable[[], int | None]) -> None:
    """Refuse to save a `state` holding a value that is not finite (`first_non_finite`), on any tier: no
    checkpoint is written and none rotated out, and the run stops, naming the step and the first such leaf,
    so the checkpoint a resume finds is the last finite one (`latest`, read only to say so) (training.md,
    checkpoints and resume). Read at the writers' own boundary (`Checkpoints.save`, `save_local`), so a
    direct save is held to it as a trainer's is (row 9021)."""
    first = first_non_finite(state)
    if first is not None:
        kept = latest()
        said = "no checkpoint" if kept is None else f"the latest checkpoint, step {kept}"
        raise RuntimeError(f"step {step}'s state holds a non-finite value at {first}, so no checkpoint of it "
                           f"is written and {said} is the one a resume finds")
