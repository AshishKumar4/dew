"""Device mesh, parameter sharding and host-to-device prefetch."""

import queue
import threading
from collections.abc import Mapping, Sequence
from typing import Iterator, Optional, TypeAlias

import jax
import numpy as np
from flax import linen as nn
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

DATA_AXIS = 'data'
FSDP_AXIS = 'fsdp'

# Batches are split across every device, whichever axis it sits on; only
# parameters distinguish the two axes.
BATCH_SPEC = P((DATA_AXIS, FSDP_AXIS))

# Below this many elements a parameter costs more in collectives than it saves
# in memory, so it stays replicated.
DEFAULT_MIN_SHARD_SIZE = 2 ** 16
DEFAULT_SHARDING_TOLERANCE = 0.02

MeshAxes: TypeAlias = str | tuple[str, ...] | None
LogicalAxisRules: TypeAlias = tuple[tuple[str, MeshAxes], ...]
LogicalAxisRuleConfig: TypeAlias = (
    Mapping[str, str | Sequence[str] | None] | LogicalAxisRules)

# Rule order is precedence when two logical dimensions target the one fsdp axis.
# It reproduces the largest-axis choice for the annotated model shapes while
# giving a config one place to redirect model semantics onto a future mesh.
DEFAULT_LOGICAL_AXIS_RULES: LogicalAxisRules = (
    ("vocab", FSDP_AXIS),
    ("mlp", FSDP_AXIS),
    ("modulation", FSDP_AXIS),
    ("attention", FSDP_AXIS),
    ("embed", FSDP_AXIS),
    ("head_dim", FSDP_AXIS),
    ("heads", FSDP_AXIS),
    ("kv", FSDP_AXIS),
    ("output", FSDP_AXIS),
    ("expert", FSDP_AXIS),
    ("batch", None),
    ("sequence", None),
    ("stage", None),
)


def _normalize_logical_axis_rules(
    rules: Optional[LogicalAxisRuleConfig], mesh: Mesh,
) -> LogicalAxisRules:
    """Rules as flax wants them, minus mesh axes this mesh does not have.

    Dropping absent axes is what lets one table name a future 'tensor' or
    'expert' mesh axis and still apply on today's (data, fsdp) mesh.
    """
    items = (DEFAULT_LOGICAL_AXIS_RULES if rules is None
             else rules.items() if isinstance(rules, Mapping) else rules)
    normalized = []
    for logical_axis, mesh_axes in items:
        axes = ((mesh_axes,) if isinstance(mesh_axes, str)
                else () if mesh_axes is None else tuple(mesh_axes))
        axes = tuple(axis for axis in axes if axis in mesh.axis_names)
        normalized.append(
            (logical_axis, axes[0] if len(axes) == 1 else axes or None))
    return tuple(normalized)


def build_mesh(fsdp_size: int = 1, devices: Optional[list] = None) -> Mesh:
    """Two-axis device mesh: parameters shard over 'fsdp', replicate over 'data'.

    fsdp_size=1 degenerates to plain data parallelism, so the same code path
    serves both without a flag. Axes are Auto so GSPMD infers the collectives
    rather than us writing them by hand.
    """
    devices = list(devices) if devices is not None else jax.devices()
    if fsdp_size < 1 or len(devices) % fsdp_size:
        raise ValueError(
            f"fsdp_size {fsdp_size} must be a positive divisor of device count {len(devices)}")
    return jax.make_mesh(
        (len(devices) // fsdp_size, fsdp_size),
        (DATA_AXIS, FSDP_AXIS),
        devices=devices,
        axis_types=(AxisType.Auto, AxisType.Auto),
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


def _canonical_mesh_spec(shape: tuple, spec: P, mesh: Mesh) -> P:
    """Reduce a rules-derived spec to one this parameter can actually take.

    A mesh axis of size 1 shards nothing, so it is dropped rather than left in
    the spec where it would only obscure what is replicated. A dimension the
    assigned axes do not divide evenly cannot be split at all, and replicating
    the whole parameter is the honest answer: the tolerance check below is what
    turns that into an error when it matters.
    """
    entries = []
    for dimension, assignment in enumerate(spec):
        axes = ((assignment,) if isinstance(assignment, str)
                else tuple(assignment) if assignment is not None else ())
        axes = tuple(axis for axis in axes if mesh.shape[axis] > 1)
        shards = int(np.prod([mesh.shape[axis] for axis in axes], dtype=np.int64))
        if shards > 1 and shape[dimension] % shards:
            return P()
        entries.append(axes[0] if len(axes) == 1 else axes or None)
    while entries and entries[-1] is None:
        entries.pop()
    return P(*entries)


def state_sharding_tree(
    mesh: Mesh,
    abstract_state,
    min_shard_size: int = DEFAULT_MIN_SHARD_SIZE,
    logical_axis_rules: Optional[LogicalAxisRuleConfig] = None,
):
    """Derive physical shardings from logical metadata, with shape fallback.

    Flax metadata is removed from the returned tree. Unannotated leaves retain
    the previous largest-divisible-axis heuristic so models can be annotated a
    family at a time.
    """
    unboxed_state = nn.unbox(abstract_state)
    logical_specs = nn.get_partition_spec(abstract_state)
    logical_shardings = nn.logical_to_mesh_sharding(
        logical_specs, mesh, _normalize_logical_axis_rules(logical_axis_rules, mesh))
    fsdp_size = mesh.shape[FSDP_AXIS]

    def leaf_sharding(value, logical_spec, logical_sharding):
        size = int(np.prod(value.shape, dtype=np.int64))
        if logical_spec == P():
            spec = parameter_spec(value.shape, fsdp_size, min_shard_size)
        elif fsdp_size == 1 or size < min_shard_size:
            spec = P()
        else:
            spec = _canonical_mesh_spec(value.shape, logical_sharding.spec, mesh)
        return NamedSharding(mesh, spec)

    return jax.tree.map(
        leaf_sharding, unboxed_state, logical_specs, logical_shardings)


def assert_params_sufficiently_sharded(
    params, shardings, mesh: Mesh,
    tolerance: float = DEFAULT_SHARDING_TOLERANCE,
    min_shard_size: int = DEFAULT_MIN_SHARD_SIZE,
) -> None:
    """Reject an FSDP layout that left too much of the model replicated.

    MaxText's guardrail (base.yml sharding_tolerance) against a mesh whose
    fsdp axis divides none of the model's dimensions, which the shape
    heuristic otherwise absorbs in silence.

    MaxText measures excess per-chip memory over perfect sharding across every
    parameter. Here the same ratio is taken over the parameters the threshold
    policy meant to shard: anything below min_shard_size is replicated on
    purpose, so counting it would fire on models that are merely small.
    """
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError(
            f"sharding_tolerance must be between 0 and 1, got {tolerance}")
    if mesh.shape[FSDP_AXIS] == 1:
        return

    path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    shardable_elements = 0
    replicated = []
    for (path, param), sharding in zip(
            path_leaves, jax.tree.leaves(shardings), strict=True):
        elements = int(np.prod(param.shape, dtype=np.int64))
        if elements < min_shard_size:
            continue
        shardable_elements += elements
        if any(assignment == FSDP_AXIS
               or isinstance(assignment, tuple) and FSDP_AXIS in assignment
               for assignment in sharding.spec):
            continue
        replicated.append((elements, jax.tree_util.keystr(path), param.shape))

    if not shardable_elements:
        return
    fraction = sum(elements for elements, _, _ in replicated) / shardable_elements
    if fraction <= tolerance:
        return

    details = "\n".join(
        f"  {name}: shape={tuple(shape)}, elements={elements}"
        for elements, name, shape in sorted(replicated, reverse=True)[:5])
    raise ValueError(
        f"{fraction:.2%} of shardable parameter elements are replicated, over "
        f"the sharding_tolerance of {tolerance:.2%}.\n"
        f"Largest replicated parameters:\n{details}")


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, BATCH_SPEC)


def shard_batch(sharding: NamedSharding, batch):
    """Assemble this process's slice of each array into a globally sharded one."""
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(sharding, np.asarray(x)), batch)


class DevicePrefetchIterator:
    """Runs the host-to-device batch transfer a few batches ahead of the loop.

    Without this the transfer sits on the critical path between steps, because
    the loop only starts moving batch N+1 after step N has been dispatched.
    """

    def __init__(self, iterator: Iterator, sharding: NamedSharding, depth: int = 2,
                 source_state=None):
        self._iterator = iter(iterator)
        self._sharding = sharding
        self._queue = queue.Queue(maxsize=depth)
        self._terminal: Optional[BaseException] = None
        self._checkpointable = hasattr(self._iterator, 'get_state')
        if source_state is not None:
            if not self._checkpointable:
                raise TypeError(
                    f"{type(self._iterator).__name__} cannot resume from a saved position")
            self._iterator.set_state(source_state)
        # Position of the source iterator as of the batch most recently handed
        # out, so a checkpoint resumes at the next unseen batch rather than at
        # whatever the prefetch thread has already raced ahead to.
        self.source_state = source_state
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._thread.start()

    def _prefetch(self):
        try:
            while True:
                batch = next(self._iterator)
                state = self._iterator.get_state() if self._checkpointable else None
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
