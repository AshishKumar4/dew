"""Device mesh, parameter layout, host-to-device prefetch, and the values a
process pool has to agree on."""

from __future__ import annotations

import dataclasses
import json
import math
import queue
import threading
from collections.abc import Mapping
from typing import Any, Iterator, Optional, Protocol, TypeAlias, runtime_checkable

import jax
import numpy as np
from flax import linen as nn
from flax.linen import spmd
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from dew.nn.sharding import LogicalAxes, declared_axes
from dew.objectives.base import Batch, Variables

DATA_AXIS = 'data'
FSDP_AXIS = 'fsdp'
EXPERT_AXIS = 'expert'

# The two axes a parameter can be split over. A dimension named 'exp' takes
# the expert axis, everything else takes fsdp.
PARAMETER_AXES = (EXPERT_AXIS, FSDP_AXIS)

# Batches are split across every device, whichever axis it sits on; only
# parameters distinguish the axes.
BATCH_SPEC = P((DATA_AXIS, EXPERT_AXIS, FSDP_AXIS))

MeshAxes: TypeAlias = str | tuple[str, ...] | None
Placement: TypeAlias = Any
"""A pytree shaped like what it places, with a `NamedSharding` at every leaf.
Python has no way to say "this tree's structure with those leaves", so the
name carries what the annotation cannot."""

LogicalAxisRules: TypeAlias = tuple[tuple[str, MeshAxes], ...]

# Rule order is precedence when two logical dimensions target the one fsdp axis.
# It reproduces the largest-axis choice for the declared model shapes while
# giving a config one place to redirect model semantics onto a future mesh.
DEFAULT_RULES: LogicalAxisRules = (
    ("vocab", FSDP_AXIS),
    ("mlp", FSDP_AXIS),
    ("modulation", FSDP_AXIS),
    ("attention", FSDP_AXIS),
    ("embed", FSDP_AXIS),
    ("head_dim", FSDP_AXIS),
    ("heads", FSDP_AXIS),
    ("kv", FSDP_AXIS),
    ("output", FSDP_AXIS),
    ("exp", EXPERT_AXIS),
    ("batch", None),
    ("sequence", None),
    ("stage", None),
)


@dataclasses.dataclass(frozen=True)
class MeshSpec:
    """How many devices the parameter axes take; data parallelism fills the rest."""
    fsdp: int = 1
    expert: int = 1
    """Devices the expert dimension of an MoE layer is split over."""


def _mesh_axes(assignment: MeshAxes) -> tuple[str, ...]:
    """One entry of a spec or a rule as the mesh axes it names."""
    if assignment is None:
        return ()
    return (assignment,) if isinstance(assignment, str) else tuple(assignment)


def build_mesh(spec: MeshSpec = MeshSpec(), devices: Optional[list] = None) -> Mesh:
    """Three-axis device mesh: parameters shard over 'fsdp' and 'expert',
    batches over all three.

    An MoE layer's expert dimension is the one dimension no dense model has,
    and splitting it is what expert parallelism is, so it gets its own axis
    rather than competing with the model's widths for 'fsdp'. Sizes of 1
    degenerate to plain data parallelism, so the same code path serves every
    topology without a flag. Axes are Auto so GSPMD infers the collectives
    rather than us writing them by hand.
    """
    devices = list(devices) if devices is not None else jax.devices()
    sharded = spec.fsdp * spec.expert
    if spec.fsdp < 1 or spec.expert < 1 or len(devices) % sharded:
        raise ValueError(
            f"fsdp {spec.fsdp} times expert {spec.expert} must be a positive "
            f"divisor of device count {len(devices)}")
    return jax.make_mesh(
        (len(devices) // sharded, spec.expert, spec.fsdp),
        (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS),
        devices=devices,
        axis_types=(AxisType.Auto, AxisType.Auto, AxisType.Auto),
    )


def parameter_spec(shape: tuple, fsdp_size: int, min_shard_size: int) -> P:
    """Shard the largest evenly-divisible axis over 'fsdp', else replicate.

    Applied to every leaf of the train state, not just params: optimizer moments
    and EMA copies have the same shapes as the params they track, so they pick
    up the same spec without anyone having to describe the optimizer's layout.
    """
    if fsdp_size == 1 or int(np.prod(shape, dtype=np.int64)) < min_shard_size:
        return P()
    for axis in sorted(range(len(shape)), key=lambda i: -shape[i]):
        if shape[axis] % fsdp_size == 0:
            return P(*([None] * axis), FSDP_AXIS)
    return P()


def _mesh_spec(shape: tuple, axes: LogicalAxes, rules: LogicalAxisRules, mesh: Mesh) -> P:
    """The spec these logical axes ask for, reduced to one the shape can take.

    A mesh axis of size 1 shards nothing, so it is dropped rather than left in
    the spec where it would only obscure what is replicated. A dimension its
    assigned axes do not divide evenly cannot be split at all, so its name is
    dropped and the rules hand the axis to the next dimension that names it:
    an odd vocabulary shards the embedding on its width instead of taking the
    whole table out of the layout. Only a parameter no named dimension can
    split stays whole, which the tolerance check turns into an error when it
    matters.
    """
    names: list[str | None] = list(axes)
    while True:
        mapped = spmd.logical_to_mesh_axes(tuple(names), rules)
        if mapped is None:
            raise ValueError(
                f"the rules {rules} give the logical axes {tuple(names)} no mesh "
                "assignment, so the parameter they name cannot be placed")
        assigned = [
            tuple(axis for axis in _mesh_axes(assignment) if mesh.shape[axis] > 1)
            for assignment in mapped]
        blocked = [
            dimension for dimension, mesh_axes in enumerate(assigned)
            if shape[dimension] % math.prod(mesh.shape[axis] for axis in mesh_axes)]
        if not blocked:
            break
        for dimension in blocked:
            names[dimension] = None
    entries = [mesh_axes[0] if len(mesh_axes) == 1 else mesh_axes or None
               for mesh_axes in assigned]
    while entries and entries[-1] is None:
        entries.pop()
    return P(*entries)


@dataclasses.dataclass(frozen=True)
class Layout:
    """How a train state is placed on a mesh.

    `rules` map the logical axes the modules declare (`dew.nn.sharding`) onto
    mesh axes, in precedence order; a rule that names a mesh axis this mesh
    does not have is dropped, which is what lets one table carry a future
    'tensor' axis. Below `min_shard` elements a parameter costs more in
    collectives than it saves in memory, so it stays replicated. `tolerance`
    is the fraction of shardable parameter elements a layout may leave
    replicated before `check` refuses it.
    """
    rules: LogicalAxisRules | Mapping[str, MeshAxes] = DEFAULT_RULES
    min_shard: int = 2 ** 16
    tolerance: float = 0.02

    def __post_init__(self):
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(
                f"sharding tolerance must be between 0 and 1, got {self.tolerance}")
        # Rules are written as a mapping or arrive as lists from a JSON
        # record; the table is a tuple of pairs, in precedence order.
        items = self.rules.items() if isinstance(self.rules, Mapping) else self.rules
        object.__setattr__(self, "rules", tuple(
            (name, axes if axes is None or isinstance(axes, str) else tuple(axes))
            for name, axes in items))

    def _rules_for(self, mesh: Mesh) -> LogicalAxisRules:
        normalized = []
        for logical_axis, mesh_axes in self.rules:
            axes = tuple(axis for axis in _mesh_axes(mesh_axes) if axis in mesh.axis_names)
            normalized.append(
                (logical_axis, axes[0] if len(axes) == 1 else axes or None))
        return tuple(normalized)

    def shardings(self, mesh: Mesh, tree: Any) -> Placement:
        """A NamedSharding per leaf of `tree`, from the declared parameter axes.

        A leaf whose path no module declares takes the largest-divisible-axis
        heuristic, so a model family can be declared at a time. Flax metadata,
        if a caller's own module attached any, is removed here, because the
        state the trainer materialises against this tree carries plain arrays.
        """
        rules = self._rules_for(mesh)
        fsdp_size = mesh.shape[FSDP_AXIS]
        sharded_devices = math.prod(mesh.shape[axis] for axis in PARAMETER_AXES)

        def leaf_sharding(path, value):
            axes = declared_axes(path, value.ndim)
            size = int(np.prod(value.shape, dtype=np.int64))
            if axes is None:
                spec = parameter_spec(value.shape, fsdp_size, self.min_shard)
            elif sharded_devices == 1 or size < self.min_shard:
                spec = P()
            else:
                spec = _mesh_spec(value.shape, axes, rules, mesh)
            return NamedSharding(mesh, spec)

        return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(tree))

    def check(self, params: Variables, shardings: Placement, mesh: Mesh) -> None:
        """Reject a layout that left too much of the model replicated.

        MaxText's guardrail (base.yml sharding_tolerance) against a mesh whose
        parameter axes divide none of the model's dimensions, which the shape
        heuristic otherwise absorbs in silence.

        MaxText measures excess per-chip memory over perfect sharding across
        every parameter. Here the same ratio is taken over the parameters the
        threshold policy meant to shard: anything below min_shard is
        replicated on purpose, so counting it would fire on models that are
        merely small.
        """
        if all(mesh.shape[axis] == 1 for axis in PARAMETER_AXES):
            return

        path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        shardable_elements = 0
        replicated = []
        for (path, param), sharding in zip(
                path_leaves, jax.tree.leaves(shardings), strict=True):
            elements = int(np.prod(param.shape, dtype=np.int64))
            if elements < self.min_shard:
                continue
            shardable_elements += elements
            if any(axis in _mesh_axes(assignment)
                   for assignment in sharding.spec for axis in PARAMETER_AXES):
                continue
            replicated.append((elements, jax.tree_util.keystr(path), param.shape))

        if not shardable_elements:
            return
        fraction = sum(elements for elements, _, _ in replicated) / shardable_elements
        if fraction <= self.tolerance:
            return

        details = "\n".join(
            f"  {name}: shape={tuple(shape)}, elements={elements}"
            for elements, name, shape in sorted(replicated, reverse=True)[:5])
        raise ValueError(
            f"{fraction:.2%} of shardable parameter elements are replicated, over "
            f"the sharding tolerance of {self.tolerance:.2%}.\n"
            f"Largest replicated parameters:\n{details}")


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, BATCH_SPEC)


def shard_batch(sharding: NamedSharding, batch: Batch) -> Batch:
    """Assemble this process's slice of each array into a globally sharded one."""
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(sharding, np.asarray(x)), batch)


@runtime_checkable
class Checkpointable(Protocol):
    """A data stream that can say where it stopped and be put back there.

    grain's iterators satisfy this; a plain iterator does not, which is what
    `fit` refuses when a run asks for checkpoints.
    """

    def get_state(self) -> Any: ...

    def set_state(self, state: Any) -> None: ...


class DevicePrefetchIterator:
    """Runs the host-to-device batch transfer a few batches ahead of the loop.

    Without this the transfer sits on the critical path between steps, because
    the loop only starts moving batch N+1 after step N has been dispatched.
    """

    def __init__(self, iterator: Iterator, sharding: NamedSharding, depth: int = 2,
                 source_state: Optional[bytes] = None):
        self._iterator = iter(iterator)
        self._sharding = sharding
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._terminal: Optional[BaseException] = None
        self._source = self._iterator if isinstance(self._iterator, Checkpointable) else None
        self.checkpointable = self._source is not None
        if source_state is not None:
            if self._source is None:
                raise TypeError(
                    f"{type(self._iterator).__name__} cannot resume from a saved position")
            self._source.set_state(self._position_for(self._source, source_state))
        # Position of the source iterator as of the batch most recently handed
        # out, so a checkpoint resumes at the next unseen batch rather than at
        # whatever the prefetch thread has already raced ahead to.
        self.source_state = source_state
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._thread.start()

    def _position_as_bytes(self, state) -> bytes:
        """A position a checkpoint can carry: grain's DataLoader iterator
        reports JSON bytes, its Dataset iterator (the packed loader) a dict,
        and the checkpoint holds one uint8 array either way."""
        return state if isinstance(state, bytes) else json.dumps(state).encode()

    def _position_for(self, source: Checkpointable, saved: bytes):
        """`saved` back in the shape this iterator's set_state reads."""
        return saved if isinstance(source.get_state(), bytes) else json.loads(saved)

    def _prefetch(self):
        try:
            while True:
                batch = next(self._iterator)
                state = (self._position_as_bytes(self._source.get_state())
                         if self._source is not None else None)
                self._queue.put((shard_batch(self._sharding, batch), state))
        except StopIteration:
            self._queue.put(StopIteration())
        except BaseException as error:  # surfaced on the consumer's thread
            self._queue.put(error)

    def __iter__(self):
        return self

    def __next__(self):
        if self._terminal is not None:
            raise self._terminal
        item = self._queue.get()
        if isinstance(item, BaseException):
            self._terminal = item
            raise item
        batch, self.source_state = item
        return batch


# --------------------------------------------------------------------------
# What every process has to agree on
# --------------------------------------------------------------------------

def broadcast_from_process_zero(value):
    """`value` as process 0 holds it, on every process.

    JSON-encodable values only. The bytes go out behind their length, because
    a collective needs one shape on every process and the others do not know
    how long process 0's value is.
    """
    payload = np.frombuffer(json.dumps(value).encode(), np.uint8)
    length = int(multihost_utils.broadcast_one_to_all(np.asarray(len(payload), np.int64)))
    if jax.process_index() != 0:
        payload = np.zeros(length, np.uint8)
    return json.loads(multihost_utils.broadcast_one_to_all(payload).tobytes())


def minimum_across_processes(count: int) -> int:
    """The smallest `count` any process holds."""
    return int(multihost_utils.process_allgather(np.asarray(count, np.int64)).min())
