"""The trainer: mesh, compiled step, EMA, checkpoints, logging.

What is learned is the objective's business (`dew.objectives.base`). The
trainer materialises the objective's tree on the mesh, compiles one step over
the global batch, keeps the EMA copy on the optimizer's clock, and hands
effects to the capabilities it was given: a `Checkpoints` for disk, a
`Tracker` for numbers and artifacts. Constructing one opens nothing; the mesh,
the compiled step and the capabilities' resources come into being in `fit`.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import dynamic_scale as dynamic_scale_lib
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from termcolor import colored

from dew.artifacts import agree_process_phase
from dew.checkpoints import Checkpoints
from dew.data.dataset import Checkpointable
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Aux, Batch, Effects, Loss, Mean, Metric, Objective, Step, Variables, merge, select, mean_loss
from dew.telemetry.instrumentation import model_flops_utilization, step_flops
from dew.training.distributed import (
    DevicePrefetchIterator, Layout, MeshSpec, Placement, batch_shardings, build_mesh,
    shard_batch,
)
from dew.training.evaluation import Evaluation, evaluate
from dew.training.state import Accumulation, TrainState
from dew.training.tracker import Tracker
from dew.telemetry.records import FitStarted, FitEnded, CheckpointRequested, ProfileWindow, Record

if TYPE_CHECKING:
    from dew.data import Dataset

# Consecutive non-finite losses that stop a run.
BAD_LOSS_STEPS = 5

StepFn = Callable[[TrainState, Batch], tuple[TrainState, jax.Array, Aux]]
"""A compiled step's body: the state and the global batch in, the new state,
the loss and the objective's report out."""

CompiledStep = Callable[
    [TrainState, Batch],
    tuple[TrainState, jax.Array, Mapping[str, jax.Array], jax.Array, jax.Array]]
"""A call returns state, scalar loss, metrics, loss_finite, and accepted."""


class Rollout(Protocol):
    """A host-side batch producer the trainer runs before the compiled step.

    Sampling is effectful and untraceable, so it lives outside `jit`: the
    trainer calls the rollout with the state, the prefetched batch and a key
    folded from the run key and the step, then reshards what comes back with
    `shard_batch`. The returned batch must hold arrays in fixed shapes, so
    the step still compiles once per run."""

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...


@dataclasses.dataclass(frozen=True)
class Profile:
    """One profiler window per fit: `steps` steps traced into `directory`
    after `warmup` steps have run, so the trace holds the loop and not the
    compile."""
    directory: str
    steps: int
    warmup: int = 2


Book = tuple[jax.Array, jax.Array, jax.Array]
"""The loop's device-side counters: the interval's summed loss, the current
streak of non-finite losses, and the longest streak since the last check."""


def fresh_book() -> Book:
    return jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)


@jax.jit
def bookkeep(book: Book, loss: jax.Array, finite: jax.Array) -> Book:
    """The counters after one step, in one dispatch.

    As five eager ops (the cast, the add, the where, the add, the maximum)
    each dispatched an executable of its own, 176 us a step on an i9-12900K
    against 37 us for this one call, measured over 2000 steps on the CPU
    backend with the result blocked on at the end.
    """
    interval_loss, bad_run, worst_bad_run = book
    bad_run = jnp.where(finite, 0, bad_run + 1)
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    return interval_loss + loss.astype(dtype), bad_run, jnp.maximum(worst_bad_run, bad_run)


def goodput(wall: float, first_step: float | None, other: float) -> dict[str, float]:
    """The two goodput numbers of MaxText's report dew can compute locally.

    `first_step` is the time from the start of `fit` to the first step's
    result: the placement or restore, the first batch, the compile and the
    step itself, or None when no step ran. `other` is the time spent
    outside steps after that (evaluations, checkpoint writes and the wait
    for them at the end). The step fraction is what is left of `wall`,
    which counts a step's own data stall as step time, as MaxText's
    start-to-start step time does.
    """
    numbers = {}
    if first_step is not None:
        numbers["goodput/time_to_first_step_s"] = first_step
    steps = wall - (wall if first_step is None else first_step) - other
    numbers["goodput/step_fraction"] = max(steps, 0.0) / wall if wall > 0 else 0.0
    return numbers


def with_ema(params: Variables, ema: Variables | None) -> Variables | None:
    """The variables tree with the averaged leaves in place of the live ones."""
    return None if ema is None else merge(params, ema)


def _project(tree: Variables, like: Variables) -> Variables:
    """The leaves of `tree` at the paths `like` holds, in `like`'s nesting."""
    return {name: _project(tree[name], child) if isinstance(child, Mapping) else tree[name]
            for name, child in like.items()}


def ema_update(ema: Variables, params: Variables,
               decay: jax.typing.ArrayLike) -> Variables:
    """Update selected EMA leaves in their initialized storage dtypes.

    Arithmetic uses at least fp32 and preserves explicit fp64. Unit decay
    selects the original leaf, including nonfinite frozen-reference values.
    """
    def step(average, live):
        dtype = jnp.result_type(average.dtype, live.dtype, decay, jnp.float32)
        if average.dtype != dtype or live.dtype != dtype:
            # Widen stored values without bypassing their producer's rounding.
            average, live = jax.lax.optimization_barrier((average, live))
        weight = jnp.asarray(decay, dtype)
        updated = weight * average.astype(dtype) + (1 - weight) * live.astype(dtype)
        stored = updated.astype(average.dtype)
        return jnp.where(jnp.asarray(decay) >= 1.0, average, stored)

    return jax.tree.map(step, ema, _project(params, ema))


def write_back(params: Variables, variables: Variables | None) -> Variables:
    """`params` with the collections in `variables` replaced whole."""
    if variables is None:
        return params
    if "params" in variables:
        raise ValueError("Aux.variables cannot carry the params collection; "
                         "the optimizer owns it")
    for name, collection in variables.items():
        if name not in params:
            raise ValueError(f"Aux.variables names {name!r}, which is not a collection "
                             f"of the objective's tree {sorted(params)}")
        if jax.tree.structure(collection) != jax.tree.structure(params[name]):
            raise ValueError(f"Aux.variables[{name!r}] does not have the collection's "
                             f"structure")
    return {**params, **variables}


def _unscale(gradient: jax.Array, factor: jax.typing.ArrayLike) -> jax.Array:
    """Working gradients use at least fp32, preserving higher caller precision."""
    dtype = jnp.promote_types(gradient.dtype, jnp.float32)
    return gradient.astype(dtype) / jnp.asarray(factor, dtype)


def _all_finite(tree) -> jax.Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(tree):
        finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def _compact_qk(tree):
    def compact(path, leaf):
        is_maximum = any(getattr(entry, "key", None) == "max_logits" for entry in path)
        return jnp.max(leaf, axis=-2, keepdims=True) if is_maximum and leaf.ndim > 1 else leaf
    return jax.tree_util.tree_map_with_path(compact, tree)


def _advance_scale(scale: dynamic_scale_lib.DynamicScale, finite: jax.Array):
    # Flax's growth comparison uses the incoming streak, including at restart.
    grow = scale.fin_steps == scale.growth_interval
    increased = jnp.minimum(scale.scale * scale.growth_factor, jnp.finfo(jnp.float32).max)
    decreased = scale.scale * scale.backoff_factor
    if scale.minimum_scale is not None:
        decreased = jnp.maximum(decreased, scale.minimum_scale)
    return dataclasses.replace(scale, scale=jnp.where(finite, jnp.where(grow, increased, scale.scale), decreased),
                         fin_steps=jnp.where(grow | ~finite, 0, scale.fin_steps + 1))



class Trainer(Generic[Loss, Effects]):
    """Runs an `Objective`: gradients, sharding, EMA, checkpoints, logging."""

    def __init__(
        self,
        objective: Objective[Loss, Effects],
        optimizer: optax.GradientTransformation,
        *,
        key: jax.Array,
        mesh: MeshSpec = MeshSpec(),
        layout: Layout = Layout(),
        accumulation: int = 1,
        dynamic_scale: bool = False,
        checkpoints: Checkpoints | None = None,
        tracker: Tracker | None = None,
        step: Callable[[Objective[Loss, Effects], optax.GradientTransformation], StepFn] | None = None,
        rollout: Rollout | None = None,
        profile: Profile | None = None,
    ):
        """Accumulate accepted microbatches before an optimizer commit.

        A custom step owns accepted/update clocks, scaler, EMA and mutable
        writes. The compiled wrapper owns attempted-work advancement. Host
        rollout collection runs once per consumed batch, outside replay.
        """
        if accumulation < 1:
            raise ValueError(f"accumulation must be at least 1, got {accumulation}")
        self.objective = objective
        self.optimizer = optimizer
        self.key = key
        self.mesh = mesh
        self.layout = layout
        self.accumulation = accumulation
        self.dynamic_scale = dynamic_scale
        self.checkpoints = checkpoints
        self.tracker = tracker
        self.step = step
        self.rollout = rollout
        self.profile = profile
        # Measured off the compiled step, once per fit.
        self.flops_per_step = None

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def initial_state(self) -> TrainState:
        """The state a fresh run starts from. Pure, so `fit` traces it once
        for its shapes and once, sharded, for its values."""
        init_key, run_key = jax.random.split(self.key)
        params = nn.unbox(self.objective.init(init_key))
        if "params" not in params:
            raise ValueError(
                f"the objective's tree has no params collection, only {sorted(params)}; "
                "the optimizer moves params and treats every other collection as state")
        ema = self.objective.ema
        return TrainState(
            step=jnp.zeros((), jnp.int32),
            microstep=jnp.zeros((), jnp.int32),
            updates=jnp.zeros((), jnp.int32),
            scale=(jax.tree.map(jnp.asarray, dynamic_scale_lib.DynamicScale())
                   if self.dynamic_scale else None),
            window_size=jnp.asarray(self.accumulation, jnp.int32),
            params=params,
            opt_state=self.optimizer.init(params["params"]),
            ema=None if ema is None else select(params, ema.select),
            key=run_key,
        )

    @functools.cached_property
    def device_mesh(self) -> Mesh:
        """The mesh `MeshSpec` describes over this process pool's devices,
        built on first use."""
        return build_mesh(self.mesh)

    def shardings(self, state: TrainState) -> Placement:
        """Parameter gradients follow parameters; replay records follow batches."""
        mesh = self.device_mesh
        placed = self.layout.shardings(mesh, dataclasses.replace(state, accumulation=None))
        accumulation = state.accumulation
        if accumulation is None:
            return placed
        replicated = NamedSharding(mesh, P())
        pending = jax.tree.map(lambda _: replicated, accumulation)
        def buffered_shardings(tree, batches):
            if tree is None:
                return None
            sample = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), tree)
            layout = batch_shardings(mesh, sample) if batches else self.layout.shardings(mesh, sample)
            return jax.tree.map(lambda s: NamedSharding(mesh, P(None, *s.spec)), layout)
        pending = dataclasses.replace(pending,
            gradient=None if accumulation.gradient is None else placed.params["params"],
            batches=buffered_shardings(accumulation.batches, True),
            variables=buffered_shardings(accumulation.variables, False))
        return dataclasses.replace(placed, accumulation=pending)

    def place(self) -> tuple[TrainState, Placement, bytes | None]:
        """The state itself, fresh or restored, on the mesh, with its shardings
        and the data position a resume continues from."""
        abstract = jax.eval_shape(self.initial_state)
        shardings = self.shardings(abstract)
        self.layout.check(abstract.params, shardings.params, self.device_mesh)
        checkpoints = self.checkpoints
        resume = None if checkpoints is None else checkpoints.latest
        if checkpoints is None or resume is None:
            state = jax.jit(self.initial_state, out_shardings=shardings)()
            return state, shardings, None
        abstract = dataclasses.replace(abstract, accumulation=checkpoints.accumulation_template(resume))
        shardings = self.shardings(abstract)
        template = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            abstract, shardings)
        state, position = checkpoints.restore(template, resume)
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        print(f"Resumed from step {resume} in {checkpoints.source(resume)}")
        return state, shardings, position

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _loss_shape(self, state: TrainState, batch: Batch):
        info = Step(state.microstep, jax.random.fold_in(state.key, state.step),
                    with_ema(state.params, state.ema))
        return jax.eval_shape(self.objective.loss, state.params, batch, info)

    def _initialize_accumulation(self, state: TrainState, batch: Batch, shapes, *, shape_only=False):
        if self.accumulation == 1 or self.step is not None or state.accumulation is not None:
            return state
        stats, aux = shapes
        shared = isinstance(stats, (Mean, jax.ShapeDtypeStruct))
        mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
        slots = self.accumulation - 1
        shape = lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
        trainable = jax.tree.map(shape, state.params["params"])
        records = jax.tree.map(shape, batch)
        mutable = (None if shared or aux.variables is None else
                   jax.tree.map(shape, {name: state.params[name] for name in aux.variables}))

        def zeros(leaf):
            dtype = jnp.promote_types(leaf.dtype, jnp.float32) if jnp.issubdtype(leaf.dtype, jnp.inexact) else leaf.dtype
            return jnp.zeros(leaf.shape, dtype)

        def buffer(tree):
            return jax.tree.map(lambda x: jnp.zeros((slots, *x.shape), x.dtype), tree)

        def allocate():
            return Accumulation(
                gradient=jax.tree.map(zeros, trainable) if shared else None,
                mass=jnp.zeros((), mean_dtype) if shared else None,
                statistics=((jnp.zeros((), mean_dtype),) if shared else
                            tuple(zeros(x) for x in jax.tree.leaves(stats))),
                effects=tuple(zeros(x) for x in jax.tree.leaves(aux.effects)),
                qk_stats=jax.tree.map(
                    lambda x: jnp.full(x.shape, -jnp.inf, jnp.promote_types(x.dtype, jnp.float32))
                    if jnp.issubdtype(x.dtype, jnp.inexact) else zeros(x),
                    _compact_qk(jax.tree.map(zeros, aux.qk_stats))),
                batches=None if shared else buffer(records),
                variables=None if mutable is None else buffer(mutable),
                attempts=None if shared else jnp.zeros((slots,), jnp.int32))

        pending = jax.eval_shape(allocate)
        if not shape_only:
            placement = self.shardings(dataclasses.replace(state, accumulation=pending)).accumulation
            # Allocate on the final shards, without a replicated window-sized temporary.
            pending = jax.jit(allocate, out_shardings=placement)()
        return dataclasses.replace(state, accumulation=pending)


    def _default_step(self, shapes):
        objective, optimizer, size = self.objective, self.optimizer, self.accumulation
        stats_shape, aux_shape = shapes
        shared = isinstance(stats_shape, (Mean, jax.ShapeDtypeStruct))
        stats_tree = jax.tree.structure(stats_shape)
        effects_tree = jax.tree.structure(aux_shape.effects)

        def step(state: TrainState, batch: Batch):
            info = Step(state.microstep, jax.random.fold_in(state.key, state.step),
                        with_ema(state.params, state.ema))
            scale = state.scale
            factor = jnp.asarray(1., jnp.float32) if scale is None else scale.scale
            previous = state.accumulation
            fill = state.microstep % size
            due = (fill + 1) == size

            def loss_fn(trainable):
                return objective.loss({**state.params, "params": trainable}, batch, info)

            stats, pullback, aux = jax.vjp(loss_fn, state.params["params"], has_aux=True)
            loss, local_cotangent = jax.vjp(lambda s: objective.reduce_loss(s)[0], stats)
            gradients = jax.tree.map(lambda x: _unscale(x, factor),
                                     pullback(local_cotangent(jnp.asarray(factor, loss.dtype))[0])[0])
            local_finite = _all_finite((loss, stats, gradients, aux.variables, aux.effects))
            local_ok = jnp.asarray(True) if scale is None else local_finite
            qk = _compact_qk(aux.qk_stats)
            effects = tuple(jax.tree.leaves(aux.effects))
            candidate = previous
            if previous is not None:
                effects = tuple(a + b for a, b in zip(previous.effects, effects, strict=True))
                qk = jax.tree.map(jnp.maximum, previous.qk_stats, qk)
                if shared:
                    prior_mass, prior_gradient = previous.mass, previous.gradient
                    if prior_mass is None or prior_gradient is None:
                        raise ValueError("checkpoint accumulator does not match shared-mean statistics")
                    if isinstance(stats, Mean):
                        mass, total = stats.mass, stats.total
                    elif isinstance(stats, (jax.Array, float, int)):
                        mass, total = jnp.asarray(1.), jnp.asarray(stats)
                    else:
                        raise TypeError("shared accumulation requires Mean or a scalar")
                    mass = jax.lax.stop_gradient(mass)
                    total_mass = prior_mass + mass
                    denominator = jnp.where(total_mass > 0, total_mass, 1)
                    gradients = jax.tree.map(
                        lambda old, new: old * jnp.asarray(prior_mass / denominator, old.dtype)
                        + new * jnp.asarray(mass / denominator, new.dtype), prior_gradient, gradients)
                    numerator = previous.statistics[0] + total
                    pooled = Mean(numerator, total_mass)
                    value, active = mean_loss(pooled)
                    candidate = dataclasses.replace(previous, gradient=gradients, mass=total_mass,
                                                 statistics=(numerator,), effects=effects, qk_stats=qk)
                else:
                    pooled_leaves = tuple(a + b for a, b in zip(
                        previous.statistics, jax.tree.leaves(stats), strict=True))
                    pooled = stats_tree.unflatten(pooled_leaves)
                    value, active = objective.reduce_loss(pooled)
                    candidate = dataclasses.replace(previous, statistics=pooled_leaves, effects=effects, qk_stats=qk)

                    def final_gradient(_):
                        attempts = previous.attempts
                        assert attempts is not None
                        _, reduce_pullback = jax.vjp(lambda s: objective.reduce_loss(s)[0], pooled)
                        cotangent = jax.tree.map(
                            lambda cot, leaf: cot.astype(leaf.dtype)
                            if jnp.issubdtype(leaf.dtype, jnp.inexact) else cot,
                            reduce_pullback(jnp.asarray(factor, value.dtype))[0], stats)
                        combined = jax.tree.map(lambda x: _unscale(x, factor),
                                                pullback(cotangent)[0])
                        # Each replay reads its original sequential-mutable snapshot.
                        # Returned Aux is discarded; effects and writes happen once.
                        def replay(index, accumulated):
                            retained = jax.tree.map(lambda x: x[index], previous.batches)
                            variables = state.params if previous.variables is None else {
                                **state.params, **jax.tree.map(lambda x: x[index], previous.variables)}
                            original = Step(state.microstep - fill + index,
                                            jax.random.fold_in(state.key, attempts[index]),
                                            with_ema(variables, state.ema))
                            def forward(trainable):
                                replayed, _ = objective.loss({**variables, "params": trainable}, retained, original)
                                return replayed
                            _, back = jax.vjp(forward, state.params["params"])
                            contribution = back(cotangent)[0]
                            return jax.tree.map(lambda a, b: a + _unscale(b, factor),
                                                accumulated, contribution)
                        return jax.lax.fori_loop(0, size - 1, replay, combined)
                    gradients = jax.lax.cond(due & local_ok, final_gradient, lambda _: gradients, None)
                loss = jnp.where(due, value, loss)
            else:
                _, active = objective.reduce_loss(stats)
            gradient_finite = local_finite & _all_finite((loss, gradients, effects, qk, aux.variables))
            accepted = jnp.asarray(True) if scale is None else gradient_finite
            numerical = dataclasses.replace(state, params=write_back(state.params, aux.variables),
                                      microstep=state.microstep + 1)

            def commit(current):
                # Optax initializes its state from parameter dtypes. Accumulation
                # stays wider; only the completed gradient enters that contract.
                optimizer_gradients = jax.tree.map(
                    lambda gradient, parameter: gradient.astype(parameter.dtype),
                    gradients, current.params["params"])
                native_finite = _all_finite(optimizer_gradients)

                def apply(current):
                    if isinstance(optimizer, optax.GradientTransformationExtraArgs):
                        update, opt_state = optimizer.update(optimizer_gradients, current.opt_state,
                                                            current.params["params"], qk_stats=qk)
                    else:
                        update, opt_state = optimizer.update(optimizer_gradients, current.opt_state,
                                                            current.params["params"])
                    params = {**current.params, "params": optax.apply_updates(current.params["params"], update)}
                    if aux_shape.effects is not None:
                        params = write_back(params, objective.apply_effects(params, effects_tree.unflatten(effects)))
                    averaged = current.ema
                    if objective.ema is not None:
                        if averaged is None:
                            raise ValueError("the objective declares EMA but the state carries none")
                        averaged = ema_update(averaged, params, objective.ema.decay(current.updates))
                    return dataclasses.replace(current, params=params, opt_state=opt_state, ema=averaged,
                                               updates=current.updates + 1)

                current = jax.lax.cond(native_finite if scale is not None else jnp.asarray(True),
                                       apply, lambda x: x, current)
                return current, native_finite

            numerical, native_finite = jax.lax.cond(
                due & active & accepted, commit, lambda x: (x, jnp.asarray(True)), numerical)
            gradient_finite = gradient_finite & native_finite
            if scale is not None:
                accepted = gradient_finite & _all_finite((numerical.params, numerical.opt_state, numerical.ema))
            if previous is not None:
                assert candidate is not None
                def retain(acc):
                    if not shared:
                        acc = dataclasses.replace(acc, batches=jax.tree.map(lambda held, x: held.at[fill].set(x), acc.batches, batch),
                        attempts=acc.attempts.at[fill].set(state.step))
                        if acc.variables is not None:
                            reads = {name: state.params[name] for name in acc.variables}
                            acc = dataclasses.replace(acc, variables=jax.tree.map(
                                lambda held, x: held.at[fill].set(x), acc.variables, reads))
                    return acc
                def clear(acc):
                    return dataclasses.replace(jax.tree.map(jnp.zeros_like, acc), qk_stats=jax.tree.map(
                        lambda x: jnp.full_like(x, -jnp.inf) if jnp.issubdtype(x.dtype, jnp.inexact)
                        else jnp.zeros_like(x), acc.qk_stats))
                numerical = dataclasses.replace(numerical, accumulation=jax.lax.cond(due, clear, retain, candidate))
            result = jax.lax.cond(accepted, lambda _: numerical, lambda _: state, None)
            if scale is not None:
                result = dataclasses.replace(result, scale=_advance_scale(scale, gradient_finite))
            return result, loss, aux
        return step

    def compile(self, state: TrainState, batch: Batch) -> CompiledStep:
        """Compile a transaction over state and one already-produced global batch.

        Retained records and asynchronous checkpoints can own old array leaves.
        The call therefore does not donate input state or batch buffers.
        """
        if int(state.window_size) != self.accumulation:
            raise ValueError("checkpoint accumulation window_size differs from this trainer")
        if (state.scale is not None) != self.dynamic_scale:
            raise ValueError("checkpoint dynamic-scaler configuration differs from this trainer")
        mesh = self.device_mesh
        with jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches):
            shapes = None if self.step is not None else self._loss_shape(state, batch)
            prepared = self._initialize_accumulation(state, batch, shapes, shape_only=True)
            body = self.step(self.objective, self.optimizer) if self.step is not None else self._default_step(shapes)
            shardings = self.shardings(prepared)
            replicated = NamedSharding(mesh, P())

            def step(current, batch):
                result, loss, aux = body(current, batch)
                return (dataclasses.replace(result, step=current.step + 1), loss, aux.metrics,
                        jnp.isfinite(loss), result.microstep > current.microstep)

            jitted = jax.jit(step, in_shardings=(shardings, batch_shardings(mesh, batch)),
                             out_shardings=(shardings, replicated, replicated, replicated, replicated))
            prepared = jax.tree.map(
                lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), prepared, shardings)
            self.flops_per_step = step_flops(jitted, prepared, batch)

        def run(current, batch):
            with jax.set_mesh(mesh), pipeline_microbatches(self.mesh.microbatches):
                if current.accumulation is None and self.accumulation > 1 and self.step is None:
                    current = self._initialize_accumulation(current, batch, shapes)
                    current = jax.device_put(current, shardings)
                return jitted(current, batch)
        return run

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def fit(self, data: "Dataset", *, steps: int, log_every: int = 100,
            eval_every: int | None = None, checkpoint_every: int | None = None,
            metrics: Sequence[Metric] = (), preview: bool = False) -> TrainState:
        """Train to `steps` total steps, resuming from the checkpoints' latest
        step when the directory holds one.

        Every `log_every` steps the tracker receives the loss, the objective's
        metrics and the throughput. Every `eval_every` steps, and at the end,
        the validation split is scored: the objective's artifacts go to the
        tracker and to `metrics`, whose reductions are logged as `val/<name>`.
        Every `checkpoint_every` steps, and at the end, the state and the data
        position are written; every `checkpoints.local_every` steps they are
        written to the local directory as well.
        Preview generation is opt-in through preview=True, independent of scalar sinks.
        """
        started = time.perf_counter()
        profile, checkpoints = self.profile, self.checkpoints
        source = train = None
        tracing, traced = False, 0
        loss = None
        other = 0.0
        current = 0
        first_step = None
        process_zero = jax.process_index() == 0
        try:
            mesh = self.device_mesh
            process_zero = jax.process_index() == 0
            state, shardings, position = self.place()
            current = int(state.step)
            self._report(FitStarted(current, steps,
                checkpoints.source(current) if checkpoints is not None and position is not None else None,
                sum(leaf.size for leaf in jax.tree.leaves(state.params["params"])), mesh.devices.size,
                jax.devices()[0].device_kind, jax.process_count(), dict(mesh.shape)), current)
            if current > steps:
                raise ValueError(f"the run is at step {current}, past the {steps} asked for")

            if checkpoint_every and checkpoints is None:
                raise ValueError(
                    "checkpoint_every asks for checkpoints and this trainer has no "
                    "checkpointer; pass Checkpoints(directory) to write any")
            if current == steps and checkpoints is not None and checkpoints.latest is not None:
                return state
            local_every = None if checkpoints is None else checkpoints.local_every
            train_step = None
            # Rebound once the step is compiled, so the first tick measures steps,
            # not the compile.
            last_log_time = time.time()
            last_saved = current if checkpoints is not None and checkpoints.latest is not None else None
            interval_steps = 0
            steps_since_log = 0
            # Seconds spent sampling this interval, logged under
            # train/rollout_seconds when a rollout is set.
            rollout_seconds = 0.0
            # The interval's loss and both bad-loss counters live on device, so the
            # loop never blocks on a result, and move together in one dispatch.
            # `worst_bad_run` remembers the longest streak of non-finite losses
            # seen since the last host check; the host check reads it to decide
            # whether to stop.
            book = fresh_book()
            seen = 0
            first_step = None

            if current < steps:
                source = data.train()
                if (checkpoint_every or local_every) and not isinstance(source, Checkpointable):
                    raise ValueError(
                        f"checkpoint_every needs a training stream with get_state and "
                        f"set_state, and {type(source).__name__} lacks one; a checkpoint "
                        f"written without the data position would replay the data on "
                        f"resume. Train it with checkpoint_every=None "
                        f"(--trainer.checkpoint-every None)")
                train = DevicePrefetchIterator(source, mesh, source_state=position)
                source = None  # Lifetime transferred to the prefetch worker.

            error = None
            try:
                if process_zero:
                    print(f"Training from step {current} to {steps} on "
                          f"{dict(mesh.shape)} ({jax.process_count()} process(es))")
            except BaseException as failure:
                error = failure
            agree_process_phase(error, phase="training announcement")
            while current < steps:
                assert train is not None
                batch = next(train)
                if self.rollout is not None:
                    # Host-side and untraceable: sampling, scoring, advantages.
                    # The key folds the step key once more, keeping the
                    # rollout's draws off the step's stream; both are
                    # checkpointed, so a resumed run samples forward. Fixed
                    # shapes mean the compile below traces once.
                    began = time.perf_counter()
                    key = jax.random.fold_in(
                        jax.random.fold_in(state.key, state.step), 1)
                    batch = shard_batch(mesh, self.rollout(state, batch, key))
                    rollout_seconds += time.perf_counter() - began
                if train_step is None:
                    train_step = self.compile(state, batch)
                    last_log_time = time.time()
                if (profile is not None and not tracing and traced == 0
                        and seen >= profile.warmup):
                    jax.profiler.start_trace(profile.directory)
                    tracing = True

                state, loss, aux, finite, accepted = train_step(state, batch)
                position = train.source_state
                current += 1
                seen += 1
                steps_since_log += 1
                interval_steps += 1
                book = bookkeep(book, loss, finite)
                if first_step is None:
                    loss.block_until_ready()
                    first_step = time.perf_counter() - started

                if tracing and profile is not None:
                    traced += 1
                    if traced == profile.steps:
                        tracing = False
                        self._stop_trace(traced, loss, profile, step=current)

                if current % log_every == 0:
                    interval_loss, _, worst_bad_run = book
                    self._check_finite(worst_bad_run, current)
                    book = (interval_loss, book[1], jnp.zeros((), jnp.int32))
                    error = None
                    try:
                        if process_zero:
                            # The interval's numbers need the loss on the host, so
                            # this is where the loop waits on the device.
                            loss.block_until_ready()
                            now = time.time()
                            scalars = {"train/step": current, "train/loss": float(loss),
                                       **{f"train/{k}": float(v) for k, v in aux.items()},
                                       **self._throughput(now - last_log_time, steps_since_log,
                                                          data.batch)}
                            scalars["train/accepted"] = float(accepted)
                            if state.scale is not None:
                                scalars["train/loss_scale"] = float(state.scale.scale)
                            if self.rollout is not None:
                                scalars["train/rollout_seconds"] = rollout_seconds
                            print(f"step {current}: loss {scalars['train/loss']:.4f}")
                            if self.tracker is not None:
                                self.tracker.log(scalars, current)
                            last_log_time, steps_since_log, rollout_seconds = now, 0, 0.0

                    except BaseException as failure:
                        error = failure
                    agree_process_phase(error, phase="training reporting")

                if eval_every and current % eval_every == 0 and current < steps:
                    paused = time.perf_counter()
                    self._report_evaluation(evaluate(
                        self.objective, state.params, data.val, metrics=metrics, key=state.key,
                        step=state.step, schedule_step=state.microstep,
                        averaged=with_ema(state.params, state.ema),
                        preview=preview, mesh=mesh))
                    other += time.perf_counter() - paused

                # On its own clock, not the logging one: nested inside the log
                # tick, a cadence that did not divide log_every never fired at all.
                if (checkpoint_every and checkpoints is not None
                        and current % checkpoint_every == 0 and current < steps):
                    paused = time.perf_counter()
                    checkpoints.save(current, state, position,
                                     {"loss": float(book[0] / interval_steps)})
                    self._report(CheckpointRequested(checkpoints.directory), current)
                    other += time.perf_counter() - paused
                    last_saved = current
                    book = (jnp.zeros_like(book[0]), book[1], book[2])
                    interval_steps = 0
                if (local_every and checkpoints is not None
                        and current % local_every == 0 and current < steps):
                    paused = time.perf_counter()
                    checkpoints.save_local(current, state, position)
                    self._report(CheckpointRequested(str(checkpoints.local_directory), local=True), current)
                    other += time.perf_counter() - paused

            paused = time.perf_counter()
            if train is not None:
                train.close()
                train = None
            other += time.perf_counter() - paused
            if tracing and profile is not None:
                tracing = False
                # The window outlived the run, and a trace left running takes the
                # next one down with it.
                self._stop_trace(traced, loss, profile, step=current)
            interval_loss, _, worst_bad_run = book
            self._check_finite(worst_bad_run, current)
            if loss is not None:
                # The last step has to land before the wall time is read.
                loss.block_until_ready()
            paused = time.perf_counter()
            if eval_every:
                self._report_evaluation(evaluate(
                    self.objective, state.params, data.val, metrics=metrics, key=state.key,
                    step=state.step, schedule_step=state.microstep,
                    averaged=with_ema(state.params, state.ema),
                    preview=preview, mesh=mesh))
            if checkpoints is not None and last_saved != current:
                # The in-loop saves are conditional, so the state the run ends on
                # may never have been written. It goes out under its real step,
                # because a step-0 checkpoint holding the final weights would make
                # a resume restart the schedule from the beginning.
                checkpoints.save(
                    current, state, position,
                    {"loss": float(interval_loss / interval_steps)} if interval_steps else None)
                self._report(CheckpointRequested(checkpoints.directory), current)
            other += time.perf_counter() - paused
        finally:
            paused = time.perf_counter()
            primary = sys.exception()
            error = primary
            close = train.close if train is not None else getattr(source, "close", None)
            stop_trace = None
            if tracing and profile is not None:
                tracing = False
                stop_trace = lambda: self._stop_trace(traced, loss, profile, step=current)
            for label, cleanup in (
                ("Training iterator", close),
                ("Profiler", stop_trace),
                ("Checkpoint wait", None if checkpoints is None else checkpoints.wait),
            ):
                if cleanup is not None:
                    try:
                        cleanup()
                    except BaseException as failure:
                        if error is None:
                            error = failure
                        else:
                            error.add_note(f"{label} cleanup failed: {failure!r}")
            source = train = close = cleanup = None
            other += time.perf_counter() - paused
            try:
                agree_process_phase(error, phase="fit cleanup")
            except BaseException as failure:
                if error is None:
                    error = failure
            if error is None:
                try:
                    if process_zero:
                        scalars = goodput(time.perf_counter() - started, first_step, other)
                        print(f"Goodput: first step after {scalars.get('goodput/time_to_first_step_s', 0.0):.2f} s, "
                              f"{scalars['goodput/step_fraction']:.1%} of the wall time in steps")
                        if self.tracker is not None:
                            self.tracker.log(scalars, current)
                except BaseException as failure:
                    error = failure
                try:
                    agree_process_phase(error, phase="goodput reporting")
                except BaseException as failure:
                    error = failure
            try:
                self._report(FitEnded.outcome(time.perf_counter() - started, error), current)
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f'Reporting fit outcome failed: {failure!r}')
            if primary is None and error is not None:
                raise error
        return state

    def _report(self, value: Record, step: int) -> None:
        error = None
        if jax.process_index() == 0 and self.tracker is not None:
            try:
                self.tracker.artifact(value, step)
            except BaseException as failure:
                error = failure
        agree_process_phase(error, phase=type(value).__name__)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _report_evaluation(self, result: Evaluation) -> None:
        error = None
        if jax.process_index() == 0:
            try:
                print(f"Evaluation {result.split} at step {result.step}: "
                      f"{result.coordinated_batches} coordinated batches, {result.records} records, "
                      f"uneven_shards={result.uneven_shards}, event_key={result.event_key}: {result.scores}")
                if self.tracker is not None:
                    for artifact in result.previews:
                        self.tracker.artifact(artifact, result.step)
                    scalars = result.scalars
                    if scalars:
                        self.tracker.log(scalars, result.step)
            except BaseException as failure:
                error = failure
        agree_process_phase(error, phase="evaluation reporting")


    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _stop_trace(self, traced: int, loss, profile: Profile, *, step: int = 0) -> None:
        """Stop every owned trace before reporting its window on process zero."""
        error = None
        try:
            try:
                if loss is not None:
                    loss.block_until_ready()
            finally:
                primary = sys.exception()
                try:
                    jax.profiler.stop_trace()
                except BaseException as failure:
                    if primary is None:
                        raise
                    primary.add_note(f"Profiler stop failed: {failure!r}")
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="profile stop")
        self._report(ProfileWindow(profile.directory, traced), step)
        error = None
        try:
            if jax.process_index() == 0:
                print(f"Wrote profile for {traced} steps to {profile.directory}")
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="profile announcement")

    def _check_finite(self, worst_bad_run, step: int):
        """Raise RuntimeError once the loss has been non-finite for BAD_LOSS_STEPS steps.

        Deferred to the logging cadence so the step loop never synchronises;
        detection is late by at most that many steps, never missed.
        """
        streak = int(worst_bad_run)
        if streak >= BAD_LOSS_STEPS:
            raise RuntimeError(
                f"Loss has been non-finite for {streak} consecutive steps "
                f"ending near step {step}, stopping")
        if streak:
            print(colored(f"Non-finite loss for {streak} step(s) before {step}", 'red'))

    def _throughput(self, elapsed: float, steps: int, batch: int) -> dict[str, float]:
        if elapsed <= 0 or steps <= 0:
            return {}
        step_time = elapsed / steps
        scalars = {"train/step_time_ms": step_time * 1000,
                   "train/samples_per_sec": batch / step_time}
        mfu = model_flops_utilization(self.flops_per_step, step_time)
        if mfu is not None:
            scalars["train/mfu"] = mfu
        return scalars
