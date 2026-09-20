"""The objective transaction, independent of where its numerical phases execute.

A resident step traces these phases together. A host-master step compiles the
same reductions and commit on CPU and crosses an eager transport boundary to
realize a batch on the accelerator. Only that orchestration differs: pooling,
replay snapshots, optimizer admission, effects, EMA and the three clocks have
one implementation. A pullback belongs to the current attempt, not TrainState.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Generic

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import dynamic_scale as dynamic_scale_lib

from dew.objectives.base import Aux, Batch, Effects, Loss, Mean, Step, Variables, mean_loss, merge
from dew.training.state import Accumulation


def with_ema(params: Variables, ema: Variables | None) -> Variables | None:
    """The variables tree with the averaged leaves in place of the live ones."""
    return None if ema is None else merge(params, ema)


def _project(tree: Variables, like: Variables) -> Variables:
    """The leaves of tree at the paths like holds, in like's nesting."""
    return {name: _project(tree[name], child) if isinstance(child, Mapping) else tree[name]
            for name, child in like.items()}


def ema_update(ema: Variables, params: Variables, decay: jax.typing.ArrayLike) -> Variables:
    """Update selected EMA leaves in their initialized storage dtypes.

    Arithmetic uses at least fp32 and preserves explicit fp64. Unit decay
    selects the original leaf, including nonfinite frozen-reference values.
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
    """Working gradients use at least fp32, preserving higher precision."""
    dtype = jnp.promote_types(gradient.dtype, jnp.float32)
    return gradient.astype(dtype) / jnp.asarray(factor, dtype)


def _all_finite(tree) -> jax.Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(tree):
        finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def compact_qk(tree):
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


@dataclasses.dataclass(frozen=True)
class Realization(Generic[Loss, Effects]):
    stats: Loss
    aux: Aux[Effects]
    pullback: Callable[[Loss], Variables]


@struct.dataclass
class PendingAttempt(Generic[Loss]):
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
    """The existing full-tree update and replay algebra, staged at its seams."""
    def __init__(self, objective, optimizer, accumulation: int, shapes):
        self.objective = objective
        self.optimizer = optimizer
        self.size = accumulation
        stats_shape, self.aux_shape = shapes
        self.shared = isinstance(stats_shape, (Mean, jax.ShapeDtypeStruct))
        self.stats_tree = jax.tree.structure(stats_shape)
        self.effects_tree = jax.tree.structure(self.aux_shape.effects)

    def realize(self, variables: Variables, batch: Batch, info: Step) -> Realization:
        def loss(trainable):
            return self.objective.loss({**variables, "params": trainable}, batch, info)
        stats, back, aux = jax.vjp(loss, variables["params"], has_aux=True)
        return Realization(stats, aux, lambda cotangent: back(cotangent)[0])

    def local_reduction(self, stats, factor):
        loss, back = jax.vjp(lambda s: self.objective.reduce_loss(s)[0], stats)
        return loss, back(jnp.asarray(factor, loss.dtype))[0]

    def prepare_attempt(self, state, stats, aux, loss, gradient):
        previous = state.accumulation
        fill = state.microstep % self.size
        due = (fill + 1) == self.size
        local_finite = _all_finite((loss, stats, gradient, aux.variables, aux.effects))
        local_ok = jnp.asarray(True) if state.scale is None else local_finite
        qk = compact_qk(aux.qk_stats)
        effects = tuple(jax.tree.leaves(aux.effects))
        candidate, pooled = previous, stats
        replay_required = jnp.asarray(False)
        if previous is None:
            _, active = self.objective.reduce_loss(stats)
        else:
            effects = tuple(a + b for a, b in zip(previous.effects, effects, strict=True))
            qk = jax.tree.map(jnp.maximum, previous.qk_stats, qk)
            if self.shared:
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
                gradient = jax.tree.map(
                    lambda old, new: old * jnp.asarray(prior_mass / denominator, old.dtype)
                    + new * jnp.asarray(mass / denominator, new.dtype), prior_gradient, gradient)
                numerator = previous.statistics[0] + total
                pooled = Mean(numerator, total_mass)
                value, active = mean_loss(pooled)
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
        value, back = jax.vjp(lambda s: self.objective.reduce_loss(s)[0], pooled)
        return jax.tree.map(
            lambda cot, leaf: cot.astype(leaf.dtype)
            if jnp.issubdtype(leaf.dtype, jnp.inexact) else cot,
            back(jnp.asarray(factor, value.dtype))[0], current_stats)

    @staticmethod
    def replay_input(state, fill, index):
        previous = state.accumulation
        assert previous is not None and previous.attempts is not None
        batch = jax.tree.map(lambda x: x[index], previous.batches)
        variables = state.params if previous.variables is None else {
            **state.params, **jax.tree.map(lambda x: x[index], previous.variables)}
        info = Step(state.microstep - fill + index,
                    jax.random.fold_in(state.key, previous.attempts[index]),
                    with_ema(variables, state.ema))
        return variables, batch, info

    @staticmethod
    def unscaled(gradient, factor):
        return jax.tree.map(lambda x: _unscale(x, factor), gradient)

    @staticmethod
    def add_contribution(accumulated, contribution, factor):
        return jax.tree.map(lambda a, b: a + _unscale(b, factor), accumulated, contribution)

    def finish_attempt(self, state, batch, aux, pending, gradient):
        scale, previous = state.scale, state.accumulation
        effects, qk, loss = pending.effects, pending.qk, pending.loss
        finite = pending.local_finite & _all_finite((loss, gradient, effects, qk, aux.variables))
        accepted = jnp.asarray(True) if scale is None else finite
        numerical = dataclasses.replace(state, params=write_back(state.params, aux.variables),
                                        microstep=state.microstep + 1)

        def commit(current):
            native = jax.tree.map(lambda g, p: g.astype(p.dtype), gradient, current.params["params"])
            native_finite = _all_finite(native)

            def apply(current):
                if isinstance(self.optimizer, optax.GradientTransformationExtraArgs):
                    update, opt_state = self.optimizer.update(
                        native, current.opt_state, current.params["params"], qk_stats=qk)
                else:
                    update, opt_state = self.optimizer.update(native, current.opt_state, current.params["params"])
                params = {**current.params, "params": optax.apply_updates(current.params["params"], update)}
                if self.aux_shape.effects is not None:
                    params = write_back(params, self.objective.apply_effects(
                        params, self.effects_tree.unflatten(effects)))
                averaged = current.ema
                if self.objective.ema is not None:
                    if averaged is None:
                        raise ValueError("the objective declares EMA but the state carries none")
                    averaged = ema_update(averaged, params, self.objective.ema.decay(current.updates))
                return dataclasses.replace(current, params=params, opt_state=opt_state, ema=averaged,
                                           updates=current.updates + 1)

            current = jax.lax.cond(native_finite if scale is not None else jnp.asarray(True),
                                   apply, lambda x: x, current)
            return current, native_finite

        numerical, native_finite = jax.lax.cond(
            pending.due & pending.active & accepted, commit, lambda x: (x, jnp.asarray(True)), numerical)
        finite = finite & native_finite
        if scale is not None:
            accepted = finite & _all_finite((numerical.params, numerical.opt_state, numerical.ema))
        if previous is not None:
            candidate = pending.candidate
            assert candidate is not None

            def retain(acc):
                if not self.shared:
                    acc = dataclasses.replace(
                        acc, batches=jax.tree.map(lambda held, x: held.at[pending.fill].set(x), acc.batches, batch),
                        attempts=acc.attempts.at[pending.fill].set(state.step))
                    if acc.variables is not None:
                        reads = {name: state.params[name] for name in acc.variables}
                        acc = dataclasses.replace(acc, variables=jax.tree.map(
                            lambda held, x: held.at[pending.fill].set(x), acc.variables, reads))
                return acc

            def clear(acc):
                return dataclasses.replace(jax.tree.map(jnp.zeros_like, acc), qk_stats=jax.tree.map(
                    lambda x: jnp.full_like(x, -jnp.inf) if jnp.issubdtype(x.dtype, jnp.inexact)
                    else jnp.zeros_like(x), acc.qk_stats))
            numerical = dataclasses.replace(
                numerical, accumulation=jax.lax.cond(pending.due, clear, retain, candidate))
        result = jax.lax.cond(accepted, lambda _: numerical, lambda _: state, None)
        if scale is not None:
            result = dataclasses.replace(result, scale=_advance_scale(scale, finite))
        return result, loss, aux

    def step(self, *, realize=None, host=False):
        """One orchestration over the shared phases; host replay crosses eager
        execution boundaries, resident replay remains inside lax control flow."""
        realize = self.realize if realize is None else realize
        compile_phase = jax.jit if host else lambda f: f
        reduce = compile_phase(self.local_reduction)
        prepare = compile_phase(self.prepare_attempt)
        pooled = compile_phase(self.pooled_cotangent)
        finish = compile_phase(self.finish_attempt)
        unscale = compile_phase(self.unscaled)
        add = compile_phase(self.add_contribution)

        def run(state, batch):
            info = Step(state.microstep, jax.random.fold_in(state.key, state.step),
                        with_ema(state.params, state.ema))
            factor = jnp.asarray(1., jnp.float32) if state.scale is None else state.scale.scale
            realized = realize(state.params, batch, info)
            loss, cotangent = reduce(realized.stats, factor)
            gradient = unscale(realized.pullback(cotangent), factor)
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
            return finish(state, batch, realized.aux, pending, gradient)
        return run
