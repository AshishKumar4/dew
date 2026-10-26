"""The train state and the accumulation record that jitted steps take and return."""

from __future__ import annotations

import dataclasses

import jax
import optax
from flax import struct
from flax.training.dynamic_scale import DynamicScale

from dew.batch import FROZEN, Batch, Variables, merge

__all__ = ["CHECKPOINTED", "MODEL_STATE", "Accumulation", "TrainState"]


@struct.dataclass
class Accumulation:
    """Holds the microbatches pooled so far, until the optimizer commits them.

    It keeps only sums, with no forward residuals and no Jacobians of the
    statistics, so it can be saved in a checkpoint. `gradient` is the
    window's from its first microbatch (`microstep % accumulation == 0`),
    which replaces what the last window left there. `statistics` and
    `effects` hold their array leaves in canonical tree order, and the
    objective's traced result supplies their tree structure.

    The replay buffers, `batches` and `variables`, have a leading dimension
    of window slots, followed by the original batch or mutable-collection
    dimensions. Only the collections that `Aux.variables` rewrites need a
    snapshot of what each record read. `tokens` is the real tokens of the
    window's accepted microbatches, as two uint32 words, which its commit
    counts toward a run's tokens; None where the objective counts none.
    """
    gradient: Variables | None
    mass: jax.Array | None
    statistics: tuple[jax.Array, ...]
    effects: tuple[jax.Array, ...]
    qk_stats: Variables | None
    batches: Batch | None
    variables: Variables | None
    attempts: jax.Array | None
    tokens: jax.Array | None = None

    def replays(self, other: Accumulation | None) -> bool:
        """Whether this window's replay slots hold batches of the shapes `other`'s do, so a step
        compiled for `other` reads it: a batch ramp's stage changes them, and a pooled mean,
        which keeps no batch, replays none. A batch's few fields are compared leaf by leaf, without
        building a tree, as a ramped run's runner does each step."""
        held, compiled = self.batches, None if other is None else other.batches
        return jax.tree.structure(held) == jax.tree.structure(compiled) and all(
            (mine.shape, mine.dtype) == (theirs.shape, theirs.dtype)
            for mine, theirs in zip(jax.tree.leaves(held), jax.tree.leaves(compiled), strict=True))


def kept_window(held: Accumulation | None, pending: Accumulation, filled: int | None) -> bool:
    """Whether a state's window (`held`) is the one a step compiled for `pending`'s shapes reads.
    One that replays other shapes is replaced by an empty one where `filled`, the microbatches it
    holds, is 0 or unread (None, a step's shapes alone), and refused while it holds any."""
    if held is None or held.replays(pending):
        return held is not None
    if filled:
        raise ValueError(f"the state's part-filled accumulation window ({filled} in) holds batches of "
                         f"other shapes than the step's, which it cannot replay")
    return False



@struct.dataclass
class TrainState:
    """Holds everything a run must checkpoint to resume where it stopped.

    It keeps three separate counters. `step` counts attempted steps, and
    together with the root `key`, which never changes, and a microbatch's
    place in its step, it determines each training draw. `microstep` counts
    accepted microbatches, which a step's batch is read in one or more of
    (`Trainer.compile`), and the objective's schedules read it per step; a
    dropped window (`dropped_window`) takes back the ones it held.
    `updates` counts committed updates, and the optimizer's and the EMA's
    schedules read it.

    The loss scaler (`scale`) and the partial accumulation window
    (`accumulation`) are numerical state too, so they are saved in the same
    checkpoint as the parameters.
    """
    step: jax.Array
    microstep: jax.Array
    updates: jax.Array
    variables: Variables = struct.field(metadata={'variables': True})
    opt_state: optax.OptState
    ema: Variables | None = struct.field(metadata={'variables': True})
    key: jax.Array
    scale: DynamicScale | None
    window_size: jax.Array
    accumulation: Accumulation | None = None
    compute: Variables | None = struct.field(default=None,
                                             metadata={'checkpointed': False, 'variables': True})
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


CHECKPOINTED = tuple(field.name for field in dataclasses.fields(TrainState)
                     if field.metadata.get('checkpointed', True))
"""The train-state fields a checkpoint holds, by their names; a field its
declaration marks `checkpointed=False` is rebuilt on resume."""

MODEL_STATE = ("step", "microstep", "updates", "variables", "ema")
"""The train-state fields that say which model a step holds, its counters, variables and EMA:
what `restore_best` returns of the best step and a weights-only snapshot keeps."""

VARIABLE_FIELDS = tuple(field.name for field in dataclasses.fields(TrainState)
                        if field.metadata.get('variables'))
"""The train-state fields that each hold a variables tree, keyed by collection (`params`,
`FROZEN`, ...), as their declarations mark them; the optimizer's state mirrors the trained
parameters' paths instead."""


def apart[State: TrainState](state: State) -> tuple[State, Variables | None]:
    """`state` less its frozen tree (`variables[FROZEN]`), and the frozen tree, None where it holds
    none: a step donates the first and reads the second, so the frozen arrays are one set of buffers
    a run shares with the tree its objective holds (`together`). A state's shardings or shapes split
    the same way."""
    variables = state.variables
    if FROZEN not in variables:
        return state, None
    moving = {collection: tree for collection, tree in variables.items() if collection != FROZEN}
    return dataclasses.replace(state, variables=moving), variables[FROZEN]


def together[State: TrainState](moving: State, frozen: Variables | None) -> State:
    """The state `apart` split, its frozen tree the very arrays given."""
    return moving if frozen is None else dataclasses.replace(moving, variables={**moving.variables,
                                                                                FROZEN: frozen})
