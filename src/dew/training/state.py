"""The values that cross `jit` in a training run."""

from __future__ import annotations

from flax import struct
import jax
import optax

from dew.objectives.base import Aux, Step, Variables

__all__ = ["Aux", "Step", "TrainState", "Variables"]


@struct.dataclass
class TrainState:
    """What a run carries from step to step, and what a checkpoint holds.

    `params` is the objective's whole variables tree, every collection; the
    optimizer moves its `params` collection and the objective rewrites the
    others through `Aux.variables`. `ema` holds the leaves the objective's
    `EMASpec` selected, in the same nesting, or None. Step keys are
    `jax.random.fold_in(key, step)`.
    """
    step: jax.Array
    params: Variables
    opt_state: optax.OptState
    ema: Variables | None
    key: jax.Array
