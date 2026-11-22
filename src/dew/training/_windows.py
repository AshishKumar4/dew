"""A training state's accumulation window, built empty (`empty_window`): what a step of a window of
microbatches pools its gradients, statistics and replayed batches in, which a run's first step of a window
builds and a step compiled ahead compiles (`dew.training.trainer.Trainer`'s stepping)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew.batch import Batch
from dew.nn.kernels.options import owned_program
from dew.nn.precision import at_least_fp32
from dew.state import Accumulation, TrainState, kept_window, paired, taken
from dew.training.model_flops import TOKENS
from dew.training.planning import released
from dew.training.transaction import batch_statistics_shapes, compact_qk, pooled_by_mean

if TYPE_CHECKING:
    from dew.objectives.base import Objective
    from dew.training.trainer import Trainer


def empty_window[LossT, EffectsT](trainer: Trainer[LossT, EffectsT], state: TrainState, batch: Batch,
                                  shapes, *, window: int, split: int = 1, shape_only=False,
                                  donor: Accumulation | None = None,
                                  objective: Objective[LossT, EffectsT]) -> TrainState:
    """`trainer`'s state with the accumulation a `window` of microbatches of `batch`'s shapes pools in, empty;
    unchanged under a custom step or a held one that replays such batches (`Accumulation.replays`): a
    ramp's stage begins with a window (`Ramp.stages`). Under a window of one, the state holds none: an
    empty one it holds of another split (a phase planned afresh, `StepCompiler.afresh`) is the one its
    step lets go as it is handed the state (`laid_out`), so the step is planned over the state without
    it. Built over `donor`, the empty window it replaces, where given: each of its leaves of a new
    leaf's shape and dtype donated to it, its buffer the new one's, zeroed, the rest released
    (`taken`)."""
    if trainer.step is not None:
        return state
    if window == 1:
        return state if state.accumulation is None else dataclasses.replace(state, accumulation=None)
    stats, aux = shapes
    shared = pooled_by_mean(stats)
    mean_dtype = jnp.result_type(jnp.float32, *(x.dtype for x in jax.tree.leaves(stats)))
    slots = window - 1
    def shape(leaf: jax.Array) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)

    trainable = jax.tree.map(shape, state.variables["params"])
    records = jax.tree.map(shape, trainer._model_shapes(batch))
    mutable = (None if shared or aux.variables is None else
               jax.tree.map(shape, {name: state.variables[name] for name in aux.variables}))
    batch_stats = batch_statistics_shapes(objective, state, records, window, split)

    def zeros(leaf):
        wide = at_least_fp32(leaf.dtype)
        return jnp.zeros(leaf.shape, wide if jnp.issubdtype(leaf.dtype, jnp.inexact) else leaf.dtype)

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
                lambda x: jnp.full(x.shape, -jnp.inf, at_least_fp32(x.dtype))
                if jnp.issubdtype(x.dtype, jnp.inexact) else zeros(x),
                compact_qk(jax.tree.map(zeros, aux.qk_stats), trainer.phase_passes.sampled is not None)),
            batches=None if shared else buffer(records),
            variables=None if mutable is None else buffer(mutable),
            attempts=None if shared else jnp.zeros((slots,), jnp.int32),
            tokens=jnp.zeros((2,), jnp.uint32) if TOKENS in batch else None,
            batch_statistics=None if batch_stats is None else tuple(zeros(x) for x in batch_stats),
            replayed_statistics=(None if shared or batch_stats is None
                                 else tuple(buffer(zeros(x)) for x in batch_stats)),
            statistics_stamp=None if batch_stats is None else jnp.full((2,), -1, jnp.int32),
            fill=jnp.zeros((), jnp.int32))

    pending = jax.eval_shape(allocate)
    # The microbatches the state's window holds are read only where it is replaced (`kept_window`).
    if kept_window(state.accumulation, pending, read=not shape_only):
        return state
    if not shape_only:
        placement = trainer.shardings(dataclasses.replace(state, accumulation=pending)).accumulation
        # Allocate on the final shards, without a replicated window-sized temporary, over the donor's
        # buffers where the old window's leaves have the new ones' shapes: none allocated twice.
        # The window it replaces: one laid out for another split (`donor`), or the state's own, a ramp's
        # stage's of other shapes.
        donor = state.accumulation if donor is None else donor
        released(taken(donor, pending)[1])
        pairs = paired(donor, pending)
        given = jax.tree.leaves(donor)
        donated = [given[index] for index in sorted(pairs.values())]
        place = {index: sorted(pairs.values()).index(held) for index, held in pairs.items()}

        def built(held):
            # Each leaf a donated one takes the place of is written over it: the donated buffer is an
            # operand, so XLA writes the new window's zeros into it.
            leaves, structure = jax.tree.flatten(allocate())
            for index, at in place.items():
                leaves[index] = held[at].at[...].set(leaves[index])
            return jax.tree.unflatten(structure, leaves)

        pending = owned_program(built, out_shardings=placement, donate_argnums=0)(donated)
    return dataclasses.replace(state, accumulation=pending)
