"""Numerical state and retained accumulation records that cross jit."""

from __future__ import annotations

import jax
import optax
from flax import struct
from flax.training.dynamic_scale import DynamicScale

from dew.objectives.base import Aux, Batch, Step, Variables, merge

__all__ = ["Accumulation", "Aux", "Step", "TrainState", "Variables"]


@struct.dataclass
class Accumulation:
    """Hold the microbatches pooled so far, waiting for an optimizer commit.

    It keeps sums, not tapes: no forward residuals and no statistic
    Jacobians, so it survives a checkpoint. Statistics and effects retain
    their array leaves in canonical tree order, and the objective's traced
    result supplies their PyTree structure.

    Replay buffers have a leading window-slot dimension, then the original
    batch or mutable-collection dimensions. Only the collections
    `Aux.variables` rewrites need a per-record read snapshot.
    """
    gradient: Variables | None
    mass: jax.Array | None
    statistics: tuple[jax.Array, ...]
    effects: tuple[jax.Array, ...]
    qk_stats: Variables | None
    batches: Batch | None
    variables: Variables | None
    attempts: jax.Array | None



@struct.dataclass
class TrainState:
    """Hold everything a run must checkpoint to resume where it stopped.

    Three clocks count separately. `step` counts attempts, and with the
    immutable root key it determines the next training draw. `microstep`
    counts accepted microbatches and indexes the objective's schedules.
    `updates` counts committed updates and indexes the optimizer's and the
    EMA's schedules.

    The scaler and the retained partial window are numerical state, and
    travel through the same checkpoint as the parameters.
    """
    step: jax.Array
    microstep: jax.Array
    updates: jax.Array
    params: Variables
    opt_state: optax.OptState
    ema: Variables | None
    key: jax.Array
    scale: DynamicScale | None
    window_size: jax.Array
    accumulation: Accumulation | None = None

    @property
    def averaged(self) -> Variables:
        """The objective's EMA leaves merged into the live variables."""
        if self.ema is None:
            raise ValueError(
                "the objective keeps no EMA, so there are no averaged weights; "
                "read state.params")
        return merge(self.params, self.ema)
