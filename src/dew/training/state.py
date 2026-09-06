"""Numerical state and retained accumulation records that cross jit."""

from __future__ import annotations

from flax import struct
from flax.training.dynamic_scale import DynamicScale
import jax
import optax

from dew.objectives.base import Aux, Batch, Step, Variables, merge

__all__ = ["Accumulation", "Aux", "Step", "TrainState", "Variables"]


@struct.dataclass
class Accumulation:
    """Partial accepted work, without forward tapes or statistic Jacobians.

    Statistics and effects retain their array leaves in canonical tree order;
    the objective's traced result supplies their PyTree structure. Replay
    buffers have a leading window-slot dimension, followed by the original
    batch or mutable-collection dimensions. Only collections rewritten by
    Aux.variables need per-record read snapshots.
    """
    gradient: Variables | None
    mass: jax.Array
    statistics: tuple[jax.Array, ...]
    effects: tuple[jax.Array, ...]
    qk_stats: Variables | None
    batches: Batch | None
    variables: Variables | None
    attempts: jax.Array | None
    schedules: jax.Array | None


@struct.dataclass
class TrainState:
    """Completed attempts, accepted microbatches, and committed updates.

    The immutable root key and attempted step determine the next training
    draw. microstep indexes objective schedules; updates indexes optimizer
    and EMA schedules. The scaler and retained partial window are numerical
    state and travel through the same checkpoint as the parameters.
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
