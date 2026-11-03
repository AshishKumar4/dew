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

import contextlib
import dataclasses
import functools
import math
from collections.abc import Callable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.batch import Batch
from dew.distributed.placement import Placement
from dew.mesh import DILOCO_AXIS
from dew.nn.kernels.options import owned_program
from dew.optim import learning_rate
from dew.state import REPLICA_FIELDS, Replicas, TrainState
from dew.training.transaction import cleared, example_groups, window_dropped


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

    def admits(self, host: Sequence[str], record: bool) -> None:
        """Refuse what a DiLoCo run cannot express, before any state is made or supplied: a layout that
        keeps fields on the host (`host`, `Layout.host`), whose offload and host-master step place the
        outer model's fields and run no group's, and a record of each example's draws (`record`), which
        no single tape holds of several groups' own draws."""
        if host:
            raise ValueError(f"DiLoCo keeps each group's state on its devices; this layout keeps "
                             f"{list(host)} on the host, which places and steps the outer model's alone")
        if record:
            raise ValueError("a DiLoCo run's groups draw their own; no one record of a step's examples "
                             "holds them")

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
        met = dataclasses.replace(state, variables=variables, opt_state=opt_state,
                                  ema=None if inner.ema is None else jax.tree.map(_met, inner.ema),
                                  inner=dataclasses.replace(inner, variables=restarted))
        # A group that rejected an attempt holds a part-filled window of the parameters it leaves: dropped,
        # as a rejection drops one, so its next update pools the met parameters' gradients alone.
        return _dropped(met, state.window_size)


def regrouped(inner: Replicas, change: Callable[[Replicas], Replicas]) -> Replicas:
    """`inner` with each group's state as `change` makes it of that group's alone, restacked: a host-side
    change of a supplied state (`dew.training._fit.afresh`), once a run."""
    groups = jax.tree.leaves(inner.variables)[0].shape[0]
    each = [change(jax.tree.map(lambda leaf, group=group: leaf[group], inner)) for group in range(groups)]
    return jax.tree.map(lambda *made: jnp.stack(made), *each)


def _groups_dropped(first, microstep, window, size):
    """`window_dropped` of each group's `microstep` and `window`, and the first group's microstep, the
    run's (`first`'s place)."""
    del first
    microstep, window = jax.vmap(window_dropped, in_axes=(0, 0, None))(microstep, window, size)
    return microstep[0], microstep, window


def _dropped(state: TrainState, size) -> TrainState:
    """`state` with each group's window of `size` microbatches dropped, traced in a step."""
    inner = state.inner
    if inner is None or inner.accumulation is None:
        return state
    first, microstep, window = _groups_dropped(state.microstep, inner.microstep, inner.accumulation, size)
    return dataclasses.replace(state, microstep=first, inner=dataclasses.replace(
        inner, microstep=microstep, accumulation=window))


def dropped(state: TrainState, size: int) -> TrainState:
    """`state` with its window of `size` microbatches dropped (`window_dropped`), each group's under DiLoCo,
    as a ramp's stage begins (`Trainer._staged`). Only the windows' fields are compiled, so a host-master
    state's resident weights stay where they are."""
    if state.inner is None:
        fields = (state.microstep, state.accumulation)
        microstep, window = owned_program(functools.partial(window_dropped, size=size),
                                    out_shardings=jax.tree.map(lambda x: x.sharding, fields))(*fields)
        return dataclasses.replace(state, microstep=microstep, accumulation=window)
    fields = (state.microstep, state.inner.microstep, state.inner.accumulation)
    first, microstep, window = owned_program(functools.partial(_groups_dropped, size=size),
                                       out_shardings=jax.tree.map(lambda x: x.sharding, fields))(*fields)
    return dataclasses.replace(state, microstep=first, inner=dataclasses.replace(
        state.inner, microstep=microstep, accumulation=window))


def rated(state: TrainState) -> optax.OptState:
    """The optimizer state whose learning rates a run reads and sets (`dew.optim.learning_rate`,
    `held_rates`, `with_rates`): its own, or under DiLoCo the groups' inner one, its rates stacked, every
    group's alike, which those read at the first group's and set at every group's."""
    return state.opt_state if state.inner is None else state.inner.opt_state


def learning_rate_of(state: TrainState) -> jax.typing.ArrayLike | None:
    """The learning rate a run's optimizer state holds (`dew.optim.learning_rate`), under DiLoCo the first
    group's, every group's alike."""
    rate = learning_rate(rated(state))
    return rate if state.inner is None or rate is None else jnp.asarray(rate)[0]


def rates_set(state: TrainState, setting: Callable[[optax.OptState], optax.OptState]) -> TrainState:
    """`state` with its rated optimizer state (`rated`) as `setting` makes it, its other leaves its own."""
    if state.inner is None:
        return dataclasses.replace(state, opt_state=setting(state.opt_state))
    return dataclasses.replace(state, inner=dataclasses.replace(state.inner,
                                                                opt_state=setting(state.inner.opt_state)))


@contextlib.contextmanager
def group_of(state: TrainState, batch: Batch) -> Iterator[tuple[TrainState, Batch]]:
    """The state and batch one group's step reads, its shapes, and within the block its examples placed
    among the whole step's as its step places them (`example_groups`, the first group's): under DiLoCo its
    inner state, the first group's clock as it is (every group's window holds as many microbatches), and
    its share of the batch's rows; a run of one copy's as they are."""
    if state.inner is None:
        yield state, batch
        return
    groups = jax.tree.leaves(state.inner.variables)[0].shape[0]
    view = dataclasses.replace(state, inner=None, **{field: jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype), getattr(state.inner, field))
        for field in REPLICA_FIELDS})
    if not isinstance(state.inner.microstep, jax.ShapeDtypeStruct):
        view = dataclasses.replace(view, microstep=state.inner.microstep[0])
    share = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(
        (jnp.shape(leaf)[0] // groups, *jnp.shape(leaf)[1:]) if jnp.ndim(leaf) else (), leaf.dtype), batch)
    with example_groups(groups, np.int32(0)):
        yield view, share


def windowed(initialize: Callable, state: TrainState, batch,
             place: Callable[[TrainState], Placement[TrainState]] | None = None) -> TrainState:
    """`initialize(state, batch, shape_only)` (the trainer's window of a batch's shapes,
    `Trainer._initialize_accumulation`) of `state` and `batch`, its shapes alone unless `place` gives the
    state's placement (`Trainer.shardings`) to allocate it on. Under DiLoCo, of one group's view, its
    inner state and its share of the batch's rows, the window stacked over the groups as the inner
    state's: allocated, each group's empty (`cleared`), where `place` puts the stacked window, so no
    group's rows split over the groups."""
    if state.inner is None:
        return initialize(state, batch, shape_only=place is None)
    groups = jax.tree.leaves(state.inner.variables)[0].shape[0]
    with group_of(state, batch) as (view, share):
        held = initialize(view, share, shape_only=True)
    if held.accumulation is view.accumulation:
        return state
    shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct((groups, *leaf.shape), leaf.dtype),
                          held.accumulation)
    if place is None:
        return state.with_window(shapes)
    placement = place(state.with_window(shapes)).window
    # The zeros are made inside the program, so they are placed as `placement` places them, never whole.
    def made():
        return jax.vmap(cleared)(jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes))
    window = owned_program(made, out_shardings=placement)()
    return state.with_window(window)


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
