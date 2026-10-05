"""The train state and the accumulation record that jitted steps take and return."""

from __future__ import annotations

import jax
import optax
from flax import struct
from flax.training.dynamic_scale import DynamicScale

from dew.objectives.base import Aux, Batch, Step, Variables, merge

__all__ = ["Accumulation", "Aux", "Step", "TrainState", "Variables"]


@struct.dataclass
class Accumulation:
    """Holds the microbatches pooled so far, until the optimizer commits them.

    It keeps only sums, with no forward residuals and no Jacobians of the
    statistics, so it can be saved in a checkpoint. `statistics` and
    `effects` hold their array leaves in canonical tree order, and the
    objective's traced result supplies their tree structure.

    The replay buffers, `batches` and `variables`, have a leading dimension
    of window slots, followed by the original batch or mutable-collection
    dimensions. Only the collections that `Aux.variables` rewrites need a
    snapshot of what each record read.
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
    """Holds everything a run must checkpoint to resume where it stopped.

    It keeps three separate counters. `step` counts attempts, and together
    with the root `key`, which never changes, it determines the next
    training draw. `microstep` counts accepted microbatches, and the
    objective's schedules read it. `updates` counts committed updates, and
    the optimizer's and the EMA's schedules read it.

    The loss scaler (`scale`) and the partial accumulation window
    (`accumulation`) are numerical state too, so they are saved in the same
    checkpoint as the parameters.
    """
    step: jax.Array
    microstep: jax.Array
    updates: jax.Array
    variables: Variables
    opt_state: optax.OptState
    ema: Variables | None
    key: jax.Array
    scale: DynamicScale | None
    window_size: jax.Array
    accumulation: Accumulation | None = None
    compute: Variables | None = None
    """The lower-precision copies of the parameters that the forward pass reads, such as bf16 copies
    of fp32 weights, written by the update (`dew.training.narrow`).

    They are derived from `variables`, so no checkpoint holds them and a
    restored state starts without them.
    """

    @property
    def averaged(self) -> Variables:
        """The objective's EMA leaves merged into the live variables.

        Raises `ValueError` when the objective keeps no EMA.
        """
        if self.ema is None:
            raise ValueError(
                "the objective keeps no EMA, so there are no averaged weights; "
                "read state.variables")
        return merge(self.variables, self.ema)
