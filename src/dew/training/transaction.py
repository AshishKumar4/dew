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

import dataclasses
import functools
from collections.abc import Callable
from typing import Generic, overload

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import dynamic_scale as dynamic_scale_lib
from typing_extensions import TypeVar

from dew.objectives.base import FROZEN, Aux, Batch, Ratio, Step, Variables, merge, with_projected
from dew.training.model_flops import _TokenCount, step_counts
from dew.training.narrow import forward_variables, narrowed
from dew.training.state import Accumulation, TrainState

Loss = TypeVar("Loss", default=Ratio | jax.Array | float)
Effects = TypeVar("Effects", default=None)


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


def _chosen(take, taken, kept):
    """`taken` where `take` holds and `kept` elsewhere, leaf by leaf."""
    return jax.tree.map(lambda new, old: jnp.where(take, new, old), taken, kept)


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
            batch: Batch, stepped: jax.typing.ArrayLike) -> tuple:
    """What a compiled step returns (`Trainer.compile`) of the transaction it ran from `current` on
    `batch`, wherever it ran: the advanced state, its step counting `stepped`, the loss, the metrics
    with the batch's counts, and whether the loss is finite, the microbatch accepted and the window
    committed."""
    return (dataclasses.replace(advanced, step=current.step + stepped), loss,
            {**aux.metrics, **step_counts(batch)}, jnp.isfinite(loss), advanced.microstep > current.microstep,
            committed)


def with_ema(params: Variables, ema: Variables | None) -> Variables | None:
    """The variables tree with the averaged leaves in place of the live ones."""
    return None if ema is None else merge(params, ema)


def _project(tree: optax.Params, like: optax.Params) -> optax.Params:
    """The leaves of tree at the key paths like holds, in like's containers."""
    leaves = dict(jax.tree_util.tree_flatten_with_path(tree)[0])
    return jax.tree_util.tree_map_with_path(lambda path, _: leaves[path], like)


@overload
def ema_update(ema: Variables, params: Variables, decay: jax.typing.ArrayLike) -> Variables: ...


@overload
def ema_update(ema: optax.Params, params: optax.Params, decay: jax.typing.ArrayLike) -> optax.Params: ...


def ema_update(ema, params, decay):
    """Update the EMA leaves toward `params`, keeping each leaf's storage dtype.

    Each leaf of `ema` becomes `decay * average + (1 - decay) * live`, where
    the live value is the leaf of `params` at the same path; leaves that
    `ema` does not hold are not averaged. The arithmetic runs in fp32 or
    wider, so fp64 leaves keep fp64 precision. A decay of 1 or more returns
    the average leaf as it was, even when it holds nonfinite values, as a
    frozen reference can.
    """
    def step(average, live):
        dtype = jnp.result_type(average.dtype, live.dtype, decay, jnp.float32)
        if average.dtype != dtype or live.dtype != dtype:
            average, live = jax.lax.optimization_barrier((average, live))
        weight = jnp.asarray(decay, dtype)
        updated = weight * average.astype(dtype) + (1 - weight) * live.astype(dtype)
        stored = updated.astype(average.dtype)
        return jnp.where(jnp.asarray(decay) >= 1.0, average, stored)
    return jax.tree.map(step, ema, _project(params, ema))


def write_back(params: Variables, variables: Variables | None) -> Variables:
    """Replace nonparameter collections whole; the optimizer owns params."""
    if variables is None:
        return params
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
    dtype = jnp.promote_types(gradient.dtype, jnp.float32)
    return gradient.astype(dtype) / jnp.asarray(factor, dtype)


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
    """Reduce every sowed `max_logits` leaf over its rows, keeping the heads.

    The clip reads each head's strongest logit alone, so a retained window
    holds one row per layer instead of the batch's.
    """
    def compact(path, leaf):
        is_maximum = any(isinstance(entry, jax.tree_util.DictKey) and entry.key == "max_logits"
                         for entry in path)
        return jnp.max(leaf, axis=-2, keepdims=True) if is_maximum and leaf.ndim > 1 else leaf
    return jax.tree_util.tree_map_with_path(compact, tree)


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


class Transaction:
    """Turn one or more microbatches into one optimizer update.

    Pooling, replay, admission, effects, EMA and the three clocks are written
    once here. `step` composes them for a resident run or for a host-master
    run; only where each phase executes differs.
    """
    def __init__(self, objective, optimizer, accumulation: int, shapes, copies=None, split: int = 1):
        """Hold the objective, the optimizer and the shapes a step traces for.

        `accumulation` is the microbatches a window pools, and `split` the
        microbatches each step's batch is read in (`Trainer.compile`), one
        attempt each. `shapes` is the traced result of the objective's loss,
        statistics and `Aux`. Statistics that are a `Ratio` or a bare scalar
        pool into one shared mean, which the window can sum and never has to
        replay. `copies` names the parameters the forward reads through the
        state's narrow copies, which the commit rewrites (`dew.training.narrow`).
        """
        self.objective = objective
        self.optimizer = optimizer
        self.copies = copies or {}
        self.size = accumulation
        self.split = split
        stats_shape, self.aux_shape = shapes
        self.shared = isinstance(stats_shape, (Ratio, jax.ShapeDtypeStruct))
        self.stats_tree = jax.tree.structure(stats_shape)
        self.effects_tree = jax.tree.structure(self.aux_shape.effects)

    def realize(self, variables: Variables, batch: Batch, step_info: Step) -> Realization:
        """Evaluate the objective's loss and keep the pullback of that trace.

        This is the resident implementation, differentiating the whole
        parameter tree where the state lives. `HostExecution.realize` is the
        other one, and `step` takes either.
        """
        def loss(trainable):
            return self.objective._loss({**variables, "params": trainable}, batch, step_info)
        stats, back, aux = jax.vjp(loss, variables["params"], has_aux=True)
        return Realization(stats, aux, lambda cotangent: back(cotangent)[0])

    def local_reduction(self, stats, factor):
        """Reduce one microbatch's statistics to its loss and its cotangent.

        The cotangent is seeded with `factor`, the loss scale, so the
        gradient the pullback returns is the scaled one `unscaled` divides.
        """
        loss, back = jax.vjp(lambda s: self.objective.reduce_loss(s)[0], stats)
        return loss, back(jnp.asarray(factor, loss.dtype))[0]

    def prepare_attempt(self, state, stats, aux, loss, gradient):
        """Pool this microbatch into the window and weigh up the commit.

        Returns a `PendingAttempt`: the pooled statistics and gradient,
        whether this attempt closes the window, and whether the pooled loss
        needs a replay. A shared mean pools its gradient by mass here and
        never replays. A general statistics tree reduces only once pooled, so
        its gradient has to be recomputed under the pooled cotangent.
        """
        previous = state.accumulation
        fill = state.microstep % self.size
        due = (fill + 1) == self.size
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
        step_info = Step((state.microstep - fill + index) // self.split,
                    jax.random.fold_in(state.key, previous.attempts[index]),
                    with_ema(variables, state.ema), committed=state.updates)
        return self.read(variables, state.compute), batch, step_info

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

    def commit(self, current, gradient, effects, qk):
        """Apply the pooled native gradient and return its finiteness and global norm.

        The norm is reduced beside optimizer.update, in the same compiled
        transaction, and leaves it with the loss and auxiliary report. No
        gradient tree crosses to the host or is reconstructed for logging.
        Admission is decided by finish_attempt after checking the updated
        parameters and optimizer state as well.
        """
        native = jax.tree.map(_descending, gradient, current.variables["params"])
        native_finite = _all_finite(native)
        norm = optax.global_norm(native)
        norm = norm.astype(jnp.result_type(norm.dtype, jnp.float32))

        def apply(current):
            # The update applies to the parameters the model's constraint projects
            # (`Objective.projected`), and the step stores them projected again.
            stored = self.objective.projected(current.variables["params"], updated=False)
            if isinstance(self.optimizer, optax.GradientTransformationExtraArgs):
                update, opt_state = self.optimizer.update(native, current.opt_state, stored, qk_stats=qk)
            else:
                update, opt_state = self.optimizer.update(native, current.opt_state, stored)
            params = {**current.variables, "params": self.objective.projected(
                optax.apply_updates(stored, update), updated=True)}
            if self.aux_shape.effects is not None:
                params = write_back(params, self.objective.apply_effects(
                    params, self.effects_tree.unflatten(effects)))
            averaged = current.ema
            if self.objective.ema is not None:
                if averaged is None:
                    raise ValueError("the objective declares EMA but the state carries none")
                averaged = _chosen(self.objective.averages(current.updates),
                                   ema_update(averaged, params, self.objective.ema.decay(current.updates)),
                                   averaged)
            return dataclasses.replace(current, variables=params, opt_state=opt_state, ema=averaged,
                                       updates=current.updates + 1)

        if current.scale is None:
            return apply(current), native_finite, norm
        if self.size == 1:
            return _chosen(native_finite, apply(current), current), native_finite, norm
        return _only_when(native_finite, apply, lambda x: x, current), native_finite, norm

    def finish_attempt(self, state, batch, aux, pending, gradient, attempt, tokens):
        """Commit the window when it is due, then advance the state's clocks.

        A rejected attempt costs the same trace as an accepted one. With a
        dynamic scaler an attempt is admitted only when every value it
        produced is finite, the committed parameters and optimizer state
        included; a rejected attempt returns `state` untouched and only the
        scaler moves.

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
            return self.commit(current, gradient, effects, qk)

        def recopied(current):
            # A window of several microbatches reads one set of copies
            # throughout and rewrites them only as it commits.
            committed, native_finite, norm = commit(current)
            if committed.compute is not None:
                committed = dataclasses.replace(committed, compute=narrowed(committed.variables, self.copies))
            return committed, native_finite, norm

        due = pending.due & pending.active & accepted
        # The microbatch's real tokens (`tokens`, two uint32 words), which its window's commit trains on,
        # the window's accepted ones before it too.
        read = tokens
        held = read if previous is None or previous.tokens is None else _TokenCount.add(previous.tokens, read)
        if self.size == 1:
            committed, native_finite, norm = commit(numerical)
            numerical = _chosen(due, committed, numerical)
            native_finite = native_finite | ~due
        else:
            dtype = jnp.result_type(*jax.tree.leaves(state.variables["params"]), jnp.float32)
            zero = jnp.real(jnp.zeros((), dtype))
            # The window goes round the commit, which never reads it: a
            # conditional copies what it passes through.
            numerical, native_finite, norm = _only_when(
                due, recopied, lambda x: (x, jnp.asarray(a=True), zero),
                dataclasses.replace(numerical, accumulation=None))
            numerical = dataclasses.replace(numerical, accumulation=previous)
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

            def clear(acc):
                return dataclasses.replace(jax.tree.map(jnp.zeros_like, acc), qk_stats=jax.tree.map(
                    lambda x: jnp.full_like(x, -jnp.inf) if jnp.issubdtype(x.dtype, jnp.inexact)
                    else jnp.zeros_like(x), acc.qk_stats))
            # A pooled window's gradient sum (a replayed window keeps none)
            # goes round the reset too, its next first microbatch restarting
            # it (`prepare_attempt`): passed through on every microbatch the
            # window does not commit, it was copied each time (Qwen3.5-0.8B's 1 GB
            # embedding sum, twice a step on an A100).
            window = jax.lax.cond(pending.due, clear, retain, dataclasses.replace(candidate, gradient=None))
            numerical = dataclasses.replace(
                numerical, accumulation=dataclasses.replace(window, gradient=candidate.gradient))
        advanced = _chosen(accepted, numerical, state)
        if self.size == 1 and advanced.compute is not None:
            # The copies are the cast of whichever parameters were chosen, so
            # they are cast after the choice. Choosing between the new and the
            # old copies reads the old ones, two bytes a parameter that the
            # update's fused pass otherwise never reads.
            advanced = dataclasses.replace(advanced, compute=narrowed(advanced.variables, self.copies))
        if scale is not None:
            advanced = dataclasses.replace(advanced, scale=_advance_scale(scale, finite))
        counted = due & accepted
        return advanced, loss, aux, Committed(jnp.where(counted, norm, 0.), counted.astype(jnp.int32),
                                              jnp.where(counted, held, jnp.zeros((2,), jnp.uint32)))

    def step(self, *, realize=None, host=False):
        """Build the function that runs one attempt over the shared phases.

        A host-master step compiles each phase on its own and replays in a
        Python loop, because its realizations cross an eager transport
        boundary. A resident step traces the phases together and replays
        inside `lax` control flow.
        """
        realize = self.realize if realize is None else realize
        compile_phase = jax.jit if host else lambda f: f
        reduce = compile_phase(self.local_reduction)
        prepare = compile_phase(self.prepare_attempt)
        pooled = compile_phase(self.pooled_cotangent)
        finish = compile_phase(self.finish_attempt)
        unscale = compile_phase(self.unscaled)
        add = compile_phase(self.add_contribution)

        def run(state, batch, part, tokens):
            # The `part`th microbatch of the step's batch is attempt step *
            # split + part, with a draw of its own; the objective's schedules
            # count steps, however many microbatches each is read in. `tokens`
            # is its real tokens, which the trainer reads off the accounting
            # columns the objective's batch no longer holds (`model_batch`).
            attempt = state.step * self.split + part
            step_info = Step(state.microstep // self.split, jax.random.fold_in(state.key, attempt),
                             with_ema(state.variables, state.ema), committed=state.updates)
            factor = jnp.asarray(1., jnp.float32) if state.scale is None else state.scale.scale
            realized = realize(self.read(state.variables, state.compute), batch, step_info)
            loss, cotangent = reduce(realized.stats, factor)
            gradient = realized.pullback(cotangent)
            if state.compute is None or self.size > 1:
                gradient = unscale(gradient, factor)
            # Otherwise no loss scale divides it (`Trainer._narrow_copies`),
            # and a copied parameter's gradient stays bf16 until the update
            # widens it as it reads it. A window of several microbatches
            # widens it as it pools it, so the window sums in fp32.
            pending = prepare(state, realized.stats, realized.aux, loss, gradient)

            def replay(_):
                cot = pooled(pending.pooled, realized.stats, factor)
                combined = unscale(realized.pullback(cot), factor)
                def one(index, accumulated):
                    replayed, records, step_info = self.replay_input(state, pending.fill, index)
                    realized = realize(replayed, records, step_info)
                    return add(accumulated, realized.pullback(cot), factor)
                if host:
                    for index in range(self.size - 1):
                        combined = one(index, combined)
                    return combined
                return jax.lax.fori_loop(0, self.size - 1, one, combined)

            if state.accumulation is not None and not self.shared:
                if host:
                    gradient = replay(None) if bool(pending.replay_required) else pending.gradient
                else:
                    gradient = jax.lax.cond(pending.replay_required, replay, lambda _: pending.gradient, None)
            else:
                gradient = pending.gradient
            if not host:
                # The commit compiles as the host-master step's does, over values
                # already computed: without the barrier XLA fuses the gradient's
                # producers into the optimizer's arithmetic, which then rounds
                # otherwise than the host's compiled commit on the same platform.
                # One barrier a leaf, so each leaf's update can start as soon as
                # its own gradient is ready.
                aux, pending, gradient = jax.tree.map(
                    jax.lax.optimization_barrier, (realized.aux, pending, gradient))
                return finish(state, batch, aux, pending, gradient, attempt, tokens)
            if FROZEN not in state.variables:
                return finish(state, batch, realized.aux, pending, gradient, attempt, tokens)
            held = state.variables[FROZEN]
            moving = {name: tree for name, tree in state.variables.items() if name != FROZEN}
            advanced, loss, aux, committed = finish(dataclasses.replace(state, variables=moving), batch,
                                                    realized.aux, pending, gradient, attempt, tokens)
            return (dataclasses.replace(advanced, variables={**advanced.variables, FROZEN: held}),
                    loss, aux, committed)
        return run
