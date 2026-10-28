"""A candidate step's device bytes, derived from its shapes at trace cost (rows 2113, 2114).

The fit search admits a step where its temporaries, the new outputs it does not alias and what its
first call builds beside it (`Candidate.held`) take a free block of every device (`step_fits`); the
state it reads is already resident. A footprint states those terms before any compile: the
residuals the loss's pullback keeps between forward and backward at one microbatch, read off the
pullback's traced program (`residual_bytes`), and the trained parameters' gradient, each a device's
share. XLA's workspace beyond them is no term of it: what XLA's buffer assignment and scheduling make
of a program is no function of its residuals (tools/benchmark_decoder_remat.py, RTX 4080: 326 MB of
residuals against 138 MB of temporaries without recomputation, 13.5 against 31.9 at 'full'), so the
run's one compile confirms the derivation, and the plan records each term beside what the compile
measured.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import jax
import jax.extend.core as jcore
import numpy as np
from jax.core import ShapedArray

from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.placement import placed_bytes
from dew.mesh import batch_axes
from dew.objectives.base import Objective, Step
from dew.optim import clips_qk
from dew.state import TrainState
from dew.training.transaction import with_ema


@dataclasses.dataclass(frozen=True)
class Footprint:
    """What a candidate step takes of a device beside the state it reads, derived from its shapes:
    `residuals`, the bytes its loss's pullback keeps from forward to backward at one microbatch;
    `gradients`, the trained parameters' gradient; and `held`, what its first call builds beside the
    resident state (`Candidate.held`), each a device's share."""

    residuals: int
    gradients: int
    held: int

    @property
    def temporaries(self) -> int:
        """The temporaries the derivation names: the residuals and the gradient."""
        return self.residuals + self.gradients

    def fits(self, free: int) -> bool:
        """Whether a device with `free` bytes in one block holds the derived terms."""
        return free - self.held - self.temporaries >= 0

    def record(self) -> dict[str, JSON]:
        return {"residuals": self.residuals, "gradients": self.gradients, "held": self.held}


def residual_bytes(loss: Callable, params, arguments: Sequence, rows: int, shards: int) -> int:
    """The device bytes the pullback of `loss(params, *arguments)` in `params` keeps, a device's share:
    each value its traced program (`jax.make_jaxpr` of `jax.vjp`'s pullback) closes over, once,
    that is neither one of its inputs, which the state and the batch already hold, nor a literal,
    and lives in device memory, an offloaded one on the host. A residual whose leading axis is the batch's
    `rows` is split over the `shards` the batch is read in; any other is counted whole."""
    def pullback(params, *arguments):
        return jax.vjp(lambda trainable: loss(trainable, *arguments), params, has_aux=True)[1]

    program = jax.make_jaxpr(pullback)(params, *arguments).jaxpr
    seen = set(program.invars)
    total = 0
    for value in program.outvars:
        if isinstance(value, jcore.Literal) or value in seen:
            continue
        seen.add(value)
        aval = value.aval
        if not isinstance(aval, ShapedArray):
            raise TypeError(f"a pullback keeps {aval}, which is no array")
        if aval.memory_space != jax.memory.Space.Device:
            continue
        size = int(np.prod(aval.shape)) * np.dtype(aval.dtype).itemsize
        total += size // shards if aval.shape and aval.shape[0] == rows else size
    return total


def traced_loss[LossT, EffectsT](objective: Objective[LossT, EffectsT], state: TrainState, window: int
                                 ) -> tuple[Callable, tuple]:
    """The loss the step evaluates for `objective` over `state`, with the abstract arguments it is
    traced over (but the batch): the variables, the microstep, the key, the step and the EMA. Shapes
    and dtypes only: a resident frozen leaf sits in another memory space than the moving ones, and
    the loss the realization runs reads a snapshot in one space. The step's key is drawn inside the
    trace, so an abstract state compiles a step as a placed one does."""
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

    def loss(variables, batch, microstep, key, step, ema):
        return objective._loss(variables, batch, Step(
            microstep, jax.random.fold_in(key, step), with_ema(variables, ema),
            qk_stats=clips_qk(state.opt_state), example_count=window * rows_of(batch) or None))

    return loss, (variables, state.microstep, state.key, state.step, state.ema)


def candidate_footprint[LossT, EffectsT](objective: Objective[LossT, EffectsT], state: TrainState,
                                          micro: Batch, window: int, mesh, prepared: TrainState,
                                          held: int) -> Footprint:
    """What a step over microbatches of `micro`'s shapes, its state `prepared` as placed, takes of a
    device beside that state (`Footprint`): its loss's residuals at one microbatch, split over the
    axes its rows split over (`batch_axes`), its trained parameters' gradient, laid out as they are,
    and `held`."""
    loss, (variables, *rest) = traced_loss(objective, state, window)

    def trained(params, variables, batch, *rest):
        return loss({**variables, "params": params}, batch, *rest)

    shards = int(np.prod([mesh.shape[axis] for axis in batch_axes(mesh)]))
    residuals = residual_bytes(trained, variables["params"], (variables, micro, *rest), rows_of(micro),
                               shards)
    return Footprint(residuals, placed_bytes(prepared.variables["params"]), held)
