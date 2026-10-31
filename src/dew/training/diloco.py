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
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.distributed.placement import Placement
from dew.mesh import DILOCO_AXIS
from dew.state import REPLICA_FIELDS, Replicas, TrainState

__all__ = ["DiLoCo", "grouped", "on_groups", "rated", "rates_set"]


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

    def started(self, state: TrainState, mesh: Mesh) -> TrainState:
        """The state a DiLoCo run over `mesh`'s groups starts from: each group's inner state a copy of
        `state`'s, the outer variables `state`'s with their parameters in float32, the outer optimizer's
        state fresh."""
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
        def replica(inner, shared, batch, part, tokens, closes):
            own = jax.tree.map(lambda leaf: leaf[0], inner)
            group = jax.lax.axis_index(DILOCO_AXIS)
            view = dataclasses.replace(shared, **{field: getattr(own, field) for field in REPLICA_FIELDS},
                                       key=jax.random.fold_in(shared.key, group))
            advanced, loss, aux, committed = body(view, batch, part, tokens, closes)
            stacked = Replicas(**{field: jax.tree.map(lambda leaf: leaf[None], getattr(advanced, field))
                                  for field in REPLICA_FIELDS})
            agreed = jax.tree.map(_agreed, (loss, aux, committed))
            return stacked, *agreed

        def run(current: TrainState, batch, part, tokens, closes=None):
            shared = dataclasses.replace(current, variables={}, opt_state=None, ema=None, inner=None)
            inner, loss, aux, committed = jax.shard_map(
                replica, mesh=mesh, axis_names={DILOCO_AXIS}, check_vma=False,
                in_specs=(P(DILOCO_AXIS), P(), P(DILOCO_AXIS), P(), P(), P()),
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


def rated(state: TrainState) -> optax.OptState:
    """The optimizer state whose learning rates a run reads (`dew.optim.learning_rate`, `held_rates`): its
    own, or under DiLoCo the first group's inner one, every group's rates alike."""
    if state.inner is None:
        return state.opt_state
    return jax.tree.map(lambda leaf: leaf[0], state.inner.opt_state)


def rates_set(state: TrainState, setting: Callable[[optax.OptState], optax.OptState]) -> TrainState:
    """`state` with its rated optimizer state (`rated`) as `setting` makes it: under DiLoCo each group's
    inner one, its other leaves its own, placed where it was."""
    if state.inner is None:
        return dataclasses.replace(state, opt_state=setting(state.opt_state))
    held = state.inner.opt_state
    groups = range(jax.tree.leaves(held)[0].shape[0])
    each = [setting(jax.tree.map(lambda leaf, group=group: leaf[group], held)) for group in groups]
    def restacked(kept, *made):
        return jax.device_put(jnp.stack(made).astype(kept.dtype), kept.sharding)
    stacked = jax.tree.map(restacked, held, *each)
    return dataclasses.replace(state, inner=dataclasses.replace(state.inner, opt_state=stacked))


def grouped(shardings: Callable[[TrainState], Placement[TrainState]],
            state: TrainState) -> Placement[TrainState]:
    """Where a DiLoCo state's fields sit (`Trainer.shardings`, `shardings`): the outer fields as one run's,
    and each group's inner state as one run's state, stacked over the groups (`on_groups`)."""
    inner = shardings(dataclasses.replace(state, inner=None, **{field: jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype), getattr(state.inner, field))
        for field in REPLICA_FIELDS}))
    return dataclasses.replace(shardings(dataclasses.replace(state, inner=None)), inner=Replicas(
        **{field: jax.tree.map(on_groups, getattr(inner, field)) for field in REPLICA_FIELDS}))


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
