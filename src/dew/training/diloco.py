"""DiLoCo: replica groups that each train an inner copy of the model, averaged into the outer model by an
outer optimizer every few updates (Douillard et al., 2023, "DiLoCo: Distributed Low-Communication Training
of Language Models", Algorithm 1; MaxText's `diloco.py`).

A run of DiLoCo has a mesh whose outermost axis, `diloco` (`MeshSpec.diloco`), counts the groups. Each
group holds its own inner state (`dew.state.Replicas`), stacked on a first axis that axis splits, and reads
its own share of every batch's rows (`dew.mesh.batch_axes`). Its step is the run's own transaction over
that share, run under `jax.shard_map` over the diloco axis alone (`DiLoCo.step`): each group's conditions
stay its own, and the collectives the model takes inside it never cross groups. Every `every` windows of
updates the groups meet (`DiLoCo.synced`): the outer gradient is the outer parameters less the groups'
mean, the outer optimizer steps the outer parameters in float32, and every group starts again from them,
each keeping its inner optimizer's state.

The state's variables, EMA and optimizer state are the outer model's and the outer optimizer's, which an
evaluation, an export and a checkpoint's model read; its clocks and loss scale are the first group's,
whose windows commit with the others' but where a group's loss is not finite.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.distributed.placement import Placement
from dew.mesh import DILOCO_AXIS
from dew.state import REPLICA_FIELDS, Replicas, TrainState
from dew.training.transaction import example_groups

__all__ = ["DiLoCo", "grouped", "on_groups", "rated", "rates_set", "regrouped", "shared_by_groups",
           "window_of", "windowed", "with_window"]


@dataclasses.dataclass(frozen=True)
class DiLoCo:
    """DiLoCo's schedule and outer optimizer: the groups meet every `every` windows of updates, and the
    outer step is Nesterov SGD of the outer gradient at `rate` with `momentum`, 0.7 and 0.9 by default (the
    paper's, MaxText's `diloco_outer_lr` and `diloco_outer_momentum`). The groups are the mesh's diloco
    axis (`MeshSpec.diloco`)."""

    every: int
    """Windows of updates between the groups' meetings (MaxText's `diloco_sync_period`)."""
    rate: float = 0.7
    momentum: float = 0.9

    @property
    def outer(self) -> optax.GradientTransformation:
        """The outer optimizer: Nesterov SGD at `rate` with `momentum`."""
        return optax.sgd(self.rate, momentum=self.momentum, nesterov=True)

    def __post_init__(self):
        if self.every < 1:
            raise ValueError(f"DiLoCo's groups meet every so many windows of updates, at least 1, got "
                             f"{self.every}")

    def started(self, state: TrainState, mesh: Mesh, host: Sequence[str] = ()) -> TrainState:
        """The state a DiLoCo run over `mesh`'s groups starts from: each group's inner state a copy of
        `state`'s, the outer variables `state`'s with their parameters in float32, the outer optimizer's
        state fresh. A layout that keeps fields on the host (`host`, `Layout.host`) is refused: its
        offload places the outer fields, and no placement of a group's own is carried to its step."""
        if host:
            raise ValueError(f"DiLoCo keeps each group's state on its devices; this layout keeps "
                             f"{list(host)} on the host, which places the outer model's alone")
        replicas = mesh.shape[DILOCO_AXIS]
        inner = Replicas(**{field: jax.tree.map(lambda leaf: jnp.stack([leaf] * replicas),
                                                getattr(state, field)) for field in REPLICA_FIELDS})
        params = jax.tree.map(_wide, state.variables["params"])
        return dataclasses.replace(state, variables={**state.variables, "params": params},
                                   opt_state=self.outer.init(params), deferred=None, inner=inner)

    def step(self, body: Callable, mesh: Mesh, split: int) -> Callable:
        """`body`, a step over one group's state and rows (`Trainer._step_body`), run by every group on its
        own share of the batch under `jax.shard_map` over the diloco axis alone, then the groups met where
        the step closes their `every`th window (`synced`). Its loss, floating metrics and commit telemetry
        are the groups' mean, the rest the first group's (`_agreed`): its commits' count and tokens, which
        each group counts of the whole batch."""
        groups = mesh.shape[DILOCO_AXIS]

        def replica(inner, shared, batch, part, tokens, closes):
            own = jax.tree.map(lambda leaf: leaf[0], inner)
            group = jax.lax.axis_index(DILOCO_AXIS)
            # Each group draws its own; one group draws as a run of one copy does.
            key = shared.key if groups == 1 else jax.random.fold_in(shared.key, group)
            view = dataclasses.replace(shared, **{field: getattr(own, field) for field in REPLICA_FIELDS},
                                       key=key)
            with example_groups(groups, group):
                advanced, loss, aux, committed = body(view, batch, part, tokens, closes)
            stacked = Replicas(**{field: jax.tree.map(lambda leaf: leaf[None], getattr(advanced, field))
                                  for field in REPLICA_FIELDS})
            agreed = jax.tree.map(_agreed, (loss, aux, committed))
            return stacked, *agreed

        def run(current: TrainState, batch, part, tokens, closes=None):
            shared = dataclasses.replace(current, variables={}, opt_state=None, ema=None, inner=None)
            # A batch input of no rows, a scalar, every group reads whole.
            rows = jax.tree.map(lambda leaf: P(DILOCO_AXIS) if jnp.ndim(leaf) else P(), batch)
            inner, loss, aux, committed = jax.shard_map(
                replica, mesh=mesh, axis_names={DILOCO_AXIS}, check_vma=False,
                in_specs=(P(DILOCO_AXIS), P(), rows, P(), P(), P()),
                out_specs=(P(DILOCO_AXIS), P(), P(), P()))(current.inner, shared, batch, part, tokens, closes)
            first = jax.tree.map(lambda leaf: leaf[0], inner)
            advanced = dataclasses.replace(current, inner=inner, microstep=first.microstep,
                                           updates=first.updates, scale=first.scale)
            attempt = current.step * split + part
            meets = (attempt + 1) % (current.window_size * self.every) == 0
            return jax.lax.cond(meets, self.synced, lambda state: state, advanced), loss, aux, committed

        return run

    def synced(self, state: TrainState) -> TrainState:
        """`state` with its groups met: the outer gradient the outer parameters less the groups' mean,
        stepped by the outer optimizer in float32, every group's parameters the outer ones again; the
        other collections and the EMA the groups' mean where they are floating, the first group's
        otherwise. Each group keeps its inner optimizer's state."""
        inner = state.inner
        assert inner is not None, "a DiLoCo step's state holds its groups' inner state"
        outer = state.variables["params"]
        mean = jax.tree.map(lambda leaf: jnp.mean(_wide(leaf), axis=0), inner.variables["params"])
        gradient = jax.tree.map(jnp.subtract, outer, mean)
        updates, opt_state = self.outer.update(gradient, state.opt_state, outer)
        outer = optax.apply_updates(outer, updates)
        collections = {name: jax.tree.map(_met, tree) for name, tree in inner.variables.items()
                       if name != "params"}
        variables = {**collections, "params": outer}
        restarted = jax.tree.map(lambda kept, met: jnp.broadcast_to(met.astype(kept.dtype), kept.shape),
                                 inner.variables, {**collections, "params": outer})
        return dataclasses.replace(state, variables=variables, opt_state=opt_state,
                                   ema=None if inner.ema is None else jax.tree.map(_met, inner.ema),
                                   inner=dataclasses.replace(inner, variables=restarted))


def regrouped(inner: Replicas, change: Callable[[Replicas], Replicas]) -> Replicas:
    """`inner` with each group's state as `change` makes it of that group's alone, restacked: a host-side
    change of a supplied state (`dew.training._fit.afresh`), once a run."""
    groups = jax.tree.leaves(inner.variables)[0].shape[0]
    each = [change(jax.tree.map(lambda leaf, group=group: leaf[group], inner)) for group in range(groups)]
    return jax.tree.map(lambda *made: jnp.stack(made), *each)


def rated(state: TrainState) -> optax.OptState:
    """The optimizer state whose learning rates a run reads and sets (`dew.optim.learning_rate`,
    `held_rates`, `with_rates`): its own, or under DiLoCo the groups' inner one, its rates stacked, every
    group's alike, which those read at the first group's and set at every group's."""
    return state.opt_state if state.inner is None else state.inner.opt_state


def rates_set(state: TrainState, setting: Callable[[optax.OptState], optax.OptState]) -> TrainState:
    """`state` with its rated optimizer state (`rated`) as `setting` makes it, its other leaves its own."""
    if state.inner is None:
        return dataclasses.replace(state, opt_state=setting(state.opt_state))
    return dataclasses.replace(state, inner=dataclasses.replace(state.inner,
                                                                opt_state=setting(state.inner.opt_state)))


def window_of(state: TrainState):
    """The pooling window a step reads (`TrainState.accumulation`): a run's own, or under DiLoCo its groups'
    inner ones, stacked."""
    return state.accumulation if state.inner is None else state.inner.accumulation


def with_window(state: TrainState, window) -> TrainState:
    """`state` with `window` as the pooling window its step reads (`window_of`)."""
    if state.inner is None:
        return dataclasses.replace(state, accumulation=window)
    return dataclasses.replace(state, inner=dataclasses.replace(state.inner, accumulation=window))


def windowed(initialize: Callable, state: TrainState, batch) -> TrainState:
    """`initialize` (the trainer's window of a batch's shapes, `Trainer._initialize_accumulation`) of
    `state` and `batch`: under DiLoCo of one group's view, its inner state and its share of the batch's
    rows, the window stacked over the groups as the inner state's; a run of one copy's as it is."""
    if state.inner is None:
        return initialize(state, batch)
    groups = jax.tree.leaves(state.inner.variables)[0].shape[0]
    view = dataclasses.replace(state, inner=None, **{field: jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype), getattr(state.inner, field))
        for field in REPLICA_FIELDS})
    # The groups' clocks agree; the first's says how full the window is.
    microstep = state.inner.microstep
    view = dataclasses.replace(view, microstep=microstep if isinstance(microstep, jax.ShapeDtypeStruct)
                               else microstep[0])
    share = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(
        (jnp.shape(leaf)[0] // groups, *jnp.shape(leaf)[1:]) if jnp.ndim(leaf) else (), leaf.dtype), batch)
    window = initialize(view, share).accumulation
    stacked = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct((groups, *leaf.shape), leaf.dtype)
                           if isinstance(leaf, jax.ShapeDtypeStruct) else jnp.stack([leaf] * groups), window)
    return dataclasses.replace(state, inner=dataclasses.replace(state.inner, accumulation=stacked))


def grouped(shardings: Callable[[TrainState], Placement[TrainState]],
            state: TrainState) -> Placement[TrainState]:
    """Where a DiLoCo state's fields sit (`Trainer.shardings`, `shardings`): each group's inner state as one
    run's state, stacked over the groups (`on_groups`), and the outer model, its optimizer state and its
    EMA as one run's, further split over the groups where a dimension divides (`shared_by_groups`): only
    the groups' meeting reads them, so each group holds a share."""
    inner = shardings(dataclasses.replace(state, inner=None, **{field: jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype), getattr(state.inner, field))
        for field in REPLICA_FIELDS}))
    outer = shardings(dataclasses.replace(state, inner=None))
    outer = dataclasses.replace(outer, **{field: jax.tree.map(shared_by_groups, getattr(outer, field),
                                                              getattr(state, field))
                                          for field in ("variables", "opt_state", "ema")})
    return dataclasses.replace(outer, inner=Replicas(
        **{field: jax.tree.map(on_groups, getattr(inner, field)) for field in REPLICA_FIELDS}))


def shared_by_groups(sharding: NamedSharding, leaf) -> NamedSharding:
    """`sharding` of a leaf of the outer state further split over the diloco axis on its first dimension
    that the groups and the axes already splitting it divide; as it is where none does."""
    groups = sharding.mesh.shape[DILOCO_AXIS]
    shape = jnp.shape(leaf)
    spec = list(sharding.spec) + [None] * (len(shape) - len(sharding.spec))
    for dim, size in enumerate(shape):
        names = () if spec[dim] is None else spec[dim] if isinstance(spec[dim], tuple) else (spec[dim],)
        if groups > 1 and size % (groups * math.prod(sharding.mesh.shape[name] for name in names)) == 0:
            spec[dim] = (DILOCO_AXIS, *names)
            return NamedSharding(sharding.mesh, P(*spec), memory_kind=sharding.memory_kind)
    return sharding


def on_groups(sharding: NamedSharding) -> NamedSharding:
    """Where a group's leaf placed as `sharding` sits stacked over the groups: its first axis on the
    diloco axis, the rest as `sharding` places them inside each group."""
    def kept(entry):
        names = entry if isinstance(entry, tuple) else (entry,)
        inside = tuple(name for name in names if name not in (None, DILOCO_AXIS))
        return inside if len(inside) > 1 else inside[0] if inside else None
    return NamedSharding(sharding.mesh, P(DILOCO_AXIS, *map(kept, sharding.spec)),
                         memory_kind=sharding.memory_kind)


def _wide(leaf: jax.Array) -> jax.Array:
    """A floating leaf in at least float32, the outer model's precision; another as it is."""
    if jnp.issubdtype(leaf.dtype, jnp.floating) and jnp.finfo(leaf.dtype).bits < 32:
        return leaf.astype(jnp.float32)
    return leaf


def _met(stacked: jax.Array) -> jax.Array:
    """The groups' `stacked` leaf met: their mean, in its dtype, where it is floating; the first group's
    otherwise."""
    if jnp.issubdtype(stacked.dtype, jnp.floating):
        return jnp.mean(stacked, axis=0, dtype=jnp.float32).astype(stacked.dtype)
    return stacked[0]


def _agreed(leaf):
    """A group's metric leaf as the step reports it, inside the groups' `shard_map`: the groups' mean
    where it is floating, the first group's otherwise."""
    if not isinstance(leaf, jax.Array):
        return leaf
    if jnp.issubdtype(leaf.dtype, jnp.floating):
        return jax.lax.pmean(leaf, DILOCO_AXIS)
    first = jax.lax.axis_index(DILOCO_AXIS) == 0
    taken = jax.lax.psum(jnp.where(first, leaf, jnp.zeros_like(leaf)).astype(jnp.int32), DILOCO_AXIS)
    return taken.astype(leaf.dtype)
