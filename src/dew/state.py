"""The train state and the accumulation record that jitted steps take and return."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.dynamic_scale import DynamicScale
from jax.typing import DTypeLike

from dew.batch import FROZEN, Batch, Variables, merge
from dew.optim import parameter_name

if TYPE_CHECKING:
    from dew.objectives.base import Objective

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


def kept_window(held: Accumulation | None, pending: Accumulation, filled: Callable[[], int] | None) -> bool:
    """Whether a state's window (`held`) is the one a step compiled for `pending`'s shapes reads.
    One that replays other shapes is replaced by an empty one where `filled`, the microbatches it
    holds, read only then, is 0 or unread (None, a step's shapes alone), and refused while it holds
    any."""
    if held is None or held.replays(pending):
        return held is not None
    if filled is not None and (count := filled()):
        raise ValueError(f"the state's part-filled accumulation window ({count} in) holds batches of "
                         f"other shapes than the step's, which it cannot replay")
    return False



REPLICA: Mapping[str, bool] = MappingProxyType({"replica": True})
"""The metadata marking a `TrainState` field each DiLoCo replica holds its own of (`Replicas`): the one
declaration of the replicas' state, from which `Replicas` and `REPLICA_FIELDS` are made."""


class ReplicaState(Protocol):
    """DiLoCo's inner copies (`dew.training.diloco`) as their readers are typed: each replica group's own
    state stacked on a first axis that the mesh's diloco axis splits, its clocks of accepted microbatches
    and committed updates, its variables, optimizer state, EMA, loss scale, pooling window and held
    tokens, each as a `TrainState` field of one run holds them. `Replicas` is made from the fields
    `TrainState` marks `REPLICA`, which tests/test_diloco.py holds this view to; it is a dataclass, which
    `dataclasses.replace` takes."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]
    microstep: jax.Array
    updates: jax.Array
    variables: Variables
    opt_state: optax.OptState
    ema: Variables | None
    scale: DynamicScale | None
    accumulation: Accumulation | None
    deferred: jax.Array | None


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
    microstep: jax.Array = struct.field(metadata=REPLICA)
    updates: jax.Array = struct.field(metadata=REPLICA)
    variables: Variables = struct.field(metadata={'variables': True, **REPLICA})
    opt_state: optax.OptState = struct.field(metadata=REPLICA)
    ema: Variables | None = struct.field(metadata={'variables': True, 'averages': 'variables', **REPLICA})
    key: jax.Array
    scale: DynamicScale | None = struct.field(metadata=REPLICA)
    window_size: jax.Array
    accumulation: Accumulation | None = struct.field(default=None, metadata=REPLICA)
    consumed: jax.Array | None = dataclasses.field(default=None, metadata={"optional": True})
    """The source examples the run's dispatched batches consumed, int32, in a run bounded in examples
    (`dew.batch.EXAMPLES_CONSUMED`): each step adds what its microbatch's real rows were the last rows of,
    whether or not the update it joins commits, so the run's schedules and cadences read its progress
    exactly, the same on every process; None in a run counted otherwise, which a checkpoint that holds
    none restores (`OPTIONAL`)."""
    deferred: jax.Array | None = dataclasses.field(default=None, metadata={"optional": True, **REPLICA})
    """The real tokens of the calls whose updates the optimizer holds back
    (`dew.held_updates.update_applied`), two uint32 words (`dew.training.model_flops._TokenCount`),
    committed with the update that applies them; None for an optimizer that applies every update, which
    a checkpoint that holds none restores."""
    inner: ReplicaState | None = dataclasses.field(default=None, metadata={"optional": True})
    """DiLoCo's replicas' inner state (`dew.training.diloco`), where the variables, the EMA and the
    optimizer state above are the outer model's and its outer optimizer's; None in a run of one copy,
    which a checkpoint that holds none restores."""
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

    @property
    def window(self) -> Accumulation | None:
        """The pooling window a step reads: the run's own, or under DiLoCo its groups', stacked
        (`Replicas.accumulation`)."""
        return self.accumulation if self.inner is None else self.inner.accumulation

    @property
    def pending_tokens(self) -> jax.Array | None:
        """The real tokens consumed but not yet committed, two words, which a run's clocks count pending
        (`TokenClock`): an update's open window's (`Accumulation.tokens`) and those the optimizer holds back
        for the update that applies them (`deferred`), the run's own, or under DiLoCo the first group's,
        each group counting the whole batch's tokens as the run commits them; None where it counts neither
        (row 3930: the held tokens left out, a budget's bound fell short of what the next update commits)."""
        window = self.window
        held = self.deferred if self.inner is None else self.inner.deferred
        windowed = None if window is None or window.tokens is None else window.tokens
        if self.inner is not None:
            windowed, held = (None if words is None else words[0] for words in (windowed, held))
        if windowed is None or held is None:
            return held if windowed is None else windowed
        from dew.training.model_flops import _TokenCount

        return _TokenCount.add(windowed, held)

    def with_window(self, window: Accumulation | None) -> TrainState:
        """This state with `window` as the window its step reads (`window`)."""
        if self.inner is None:
            return dataclasses.replace(self, accumulation=window)
        return dataclasses.replace(self, inner=dataclasses.replace(self.inner, accumulation=window))


def without_empty_window(state: TrainState) -> TrainState:
    """`state` as a checkpoint holds it: its window (`window`) left out where it holds no microbatch, at
    an update's boundary, under DiLoCo every group's. A commit clears every field of the window but its
    gradient sum, which the next window's first microbatch replaces unread (`dew.training.transaction`),
    so the empty window a resume allocates (`Trainer._initialize_accumulation`) is the same window, and
    the sum, a trainable-size float32 tree, is no checkpoint's (row 4309: ~36 GB of a 9B run's ~112 GB).
    A window part filled is saved whole."""
    if state.window is None:
        return state
    from dew._coordination import collective_host

    size = int(state.window_size)
    # The groups' clocks are stacked over the groups, which a pool's processes split.
    clocks = collective_host(state.microstep if state.inner is None else state.inner.microstep,
                             phase="checkpoint window")
    return state.with_window(None) if int(jnp.max(jnp.asarray(clocks) % size)) == 0 else state


CHECKPOINTED = tuple(field.name for field in dataclasses.fields(TrainState)
                     if field.metadata.get('checkpointed', True))
"""The train-state fields a checkpoint holds, by their names; a field its
declaration marks `checkpointed=False` is rebuilt on resume."""

OPTIONAL = tuple(field.name for field in dataclasses.fields(TrainState) if field.metadata.get('optional'))
"""The checkpointed train-state fields a run may hold none of, as a run on another clock holds no
examples consumed: a checkpoint that holds none restores None (`dew.checkpoints`), whichever Dew wrote it."""

REPLICA_FIELDS = tuple(field.name for field in dataclasses.fields(TrainState)
                       if all(field.metadata.get(key) == value for key, value in REPLICA.items()))
"""The `TrainState` fields each DiLoCo replica holds its own of, in their order (`REPLICA`)."""

Replicas = struct.dataclass(type("Replicas", (), {
    "__module__": __name__,
    "__doc__": "DiLoCo's inner copies (`ReplicaState`), a pytree of the `REPLICA_FIELDS`, each replica "
               "group's own stacked on a first axis.",
    "__annotations__": {field.name: field.type for field in dataclasses.fields(TrainState)
                        if field.name in REPLICA_FIELDS}}))
"""DiLoCo's inner copies (`ReplicaState`), made as `TrainState` declares the `REPLICA_FIELDS`: a class of
their annotations alone, which `struct.dataclass` makes the pytree."""

MODEL_STATE = ("step", "microstep", "updates", "variables", "ema")
"""The train-state fields that say which model a step holds, its counters, variables and EMA:
what `restore_best` returns of the best step and a weights-only snapshot keeps."""

VARIABLE_FIELDS = tuple(field.name for field in dataclasses.fields(TrainState)
                        if field.metadata.get('variables'))
"""The train-state fields that each hold a variables tree, keyed by collection (`params`,
`FROZEN`, ...), as their declarations mark them; the optimizer's state mirrors the trained
parameters' paths instead."""

AVERAGES: Mapping[str, str] = MappingProxyType({
    field.name: field.metadata['averages'] for field in dataclasses.fields(TrainState)
    if 'averages' in field.metadata})
"""Each train-state field that averages another's leaves, by name, to the field it averages: the EMA of
the variables, which a checkpoint stores as its difference from them (`dew.checkpoints`), the run's and
each DiLoCo group's."""


def drawn_in[LossT, EffectsT](variables: Variables, dtype: DTypeLike | None,
                               objective: Objective[LossT, EffectsT]) -> Variables:
    """`variables` with each floating parameter the run draws fresh stored in `dtype`, as Flax's
    `param_dtype` would draw it: every leaf of `params` its objective holds no loaded tree for, and
    each module drawn beside a loaded base (`Objective.fresh_modules`). What the base loaded stays as its
    source stored it (`dew.config.RunConfig.source_keywords`), and an adapter's factors as they are
    stored (`dew.lora.FACTOR_DTYPE`). None leaves every leaf as drawn."""
    if dtype is None:
        return variables
    held = objective.held_variables() or {}
    kept = {path for path, _ in jax.tree_util.tree_leaves_with_path(held.get("params", {}))}
    fresh = tuple(f"{module}/" for module in objective.fresh_modules())

    def stored(path, leaf):
        drawn = path not in kept or f"{parameter_name(path)}/".startswith(fresh)
        return leaf.astype(dtype) if drawn and jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
    return {**variables, "params": jax.tree_util.tree_map_with_path(stored, variables["params"])}


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
