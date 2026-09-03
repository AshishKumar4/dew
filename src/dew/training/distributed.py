"""Device mesh, parameter sharding and host-to-device prefetch."""

import json
import math
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
EXPERT_AXIS = 'expert'

# The two axes a parameter can be split over. A dimension named 'exp' takes
# the expert axis, everything else takes fsdp.
PARAMETER_AXES = (EXPERT_AXIS, FSDP_AXIS)

# Batches are split across every device, whichever axis it sits on; only
# parameters distinguish the axes.
BATCH_SPEC = P((DATA_AXIS, EXPERT_AXIS, FSDP_AXIS))

# Below this many elements a parameter costs more in collectives than it saves
# in memory, so it stays replicated.
DEFAULT_MIN_SHARD_SIZE = 2 ** 16
DEFAULT_SHARDING_TOLERANCE = 0.02

MeshAxes: TypeAlias = str | tuple[str, ...] | None
LogicalAxes: TypeAlias = tuple[Optional[str], ...]
LogicalAxisRules: TypeAlias = tuple[tuple[str, MeshAxes], ...]
LogicalAxisRuleConfig: TypeAlias = (
    Mapping[str, str | Sequence[str] | None] | LogicalAxisRules)

# Rule order is precedence when two logical dimensions target the one fsdp axis.
# It reproduces the largest-axis choice for the declared model shapes while
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
    ("exp", EXPERT_AXIS),
    ("batch", None),
    ("sequence", None),
    ("stage", None),
)

# What a parameter's dimensions are, keyed by the module path that ends in
# these names, outermost dimension first. A parameter takes the trailing names
# its rank can hold, so a kernel takes all of them and its bias the output
# ones. Declaring them here rather than on the initializers keeps the models
# plain flax modules whose init returns arrays, and reaches optimizer moments
# and EMA copies for free: their paths end in their parameter's.
DEFAULT_LOGICAL_PARAM_AXES: Mapping[tuple[str, ...], LogicalAxes] = {
    ("embed_tokens",): ("vocab", "embed"),
    ("lm_head",): ("embed", "vocab"),
    ("q_proj",): ("embed", "heads"),
    ("k_proj",): ("embed", "kv"),
    ("v_proj",): ("embed", "kv"),
    ("o_proj",): ("attention", "embed"),
    ("gate_proj",): ("embed", "mlp"),
    ("up_proj",): ("embed", "mlp"),
    ("down_proj",): ("mlp", "embed"),
    # A sparse layer's experts are stacked on one leaf, so the expert
    # dimension is named here and the longer path wins over the dense
    # projection above it.
    ("experts", "gate_proj"): ("exp", "embed", "mlp"),
    ("experts", "up_proj"): ("exp", "embed", "mlp"),
    ("experts", "down_proj"): ("exp", "mlp", "embed"),
    ("gate",): ("embed", "exp"),
    ("patch_embed", "Conv_0"): (None, None, None, "embed"),
    ("to_q",): ("embed", "heads", "head_dim"),
    ("to_k",): ("embed", "heads", "head_dim"),
    ("to_v",): ("embed", "heads", "head_dim"),
    ("to_out_0",): ("heads", "head_dim", "embed"),
    ("ada_proj",): ("embed", "modulation"),
    ("final_ada_proj",): ("embed", "modulation"),
    ("final_proj",): ("embed", "output"),
    ("mlp", "layers_0"): ("embed", "mlp"),
    ("mlp", "layers_2"): ("mlp", "embed"),
    ("time_embed", "layers_2"): ("mlp", "embed"),
}


def _mesh_axes(assignment: MeshAxes) -> tuple[str, ...]:
    """One entry of a spec or a rule as the mesh axes it names."""
    if assignment is None:
        return ()
    return (assignment,) if isinstance(assignment, str) else tuple(assignment)


def _parameter_path(path) -> tuple[str, ...]:
    """The parameter's own path: the trailing run of dict keys under a leaf.

    An optimizer state nests a copy of the parameter tree inside its own
    structure, so what identifies a parameter is where its path ends.
    """
    names = []
    for entry in reversed(path):
        if not isinstance(entry, jax.tree_util.DictKey) or not isinstance(entry.key, str):
            break
        names.append(entry.key)
    return tuple(reversed(names))


def logical_axes(path, ndim: int) -> Optional[LogicalAxes]:
    """The declared axes of the parameter at `path`, or None for an unnamed one."""
    module = _parameter_path(path)[:-1]
    for length in range(len(module), 0, -1):
        axes = DEFAULT_LOGICAL_PARAM_AXES.get(module[-length:])
        if axes is None:
            continue
        if ndim > len(axes):
            raise ValueError(
                f"{'/'.join(module[-length:])} is declared {axes}, which cannot "
                f"name the {ndim} dimensions of {'/'.join(_parameter_path(path))}")
        return axes[len(axes) - ndim:]
    return None


def _normalize_logical_axis_rules(
    rules: Optional[LogicalAxisRuleConfig], mesh: Mesh,
) -> LogicalAxisRules:
    """Rules as flax wants them, minus mesh axes this mesh does not have.

    Dropping absent axes is what lets one table name a future 'tensor' axis
    and still apply on today's (data, expert, fsdp) mesh.
    """
    items = (DEFAULT_LOGICAL_AXIS_RULES if rules is None
             else rules.items() if isinstance(rules, Mapping) else rules)
    normalized = []
    for logical_axis, mesh_axes in items:
        axes = tuple(axis for axis in _mesh_axes(mesh_axes) if axis in mesh.axis_names)
        normalized.append(
            (logical_axis, axes[0] if len(axes) == 1 else axes or None))
    return tuple(normalized)


def build_mesh(fsdp_size: int = 1, expert_size: int = 1,
               devices: Optional[list] = None) -> Mesh:
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
    sharded = fsdp_size * expert_size
    if fsdp_size < 1 or expert_size < 1 or len(devices) % sharded:
        raise ValueError(
            f"fsdp_size {fsdp_size} times expert_size {expert_size} must be a "
            f"positive divisor of device count {len(devices)}")
    return jax.make_mesh(
        (len(devices) // sharded, expert_size, fsdp_size),
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
    names = list(axes)
    while True:
        assigned = [
            tuple(axis for axis in _mesh_axes(assignment) if mesh.shape[axis] > 1)
            for assignment in nn.logical_to_mesh_axes(tuple(names), rules)]
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


def state_sharding_tree(
    mesh: Mesh,
    abstract_state,
    min_shard_size: int = DEFAULT_MIN_SHARD_SIZE,
    logical_axis_rules: Optional[LogicalAxisRuleConfig] = None,
):
    """Derive physical shardings from the declared parameter axes.

    A leaf whose path no entry names retains the largest-divisible-axis
    heuristic, so a model family can be declared at a time. Flax metadata, if a
    caller's own module attached any, is removed here, because the state the
    trainer materialises against this tree carries plain arrays.
    """
    rules = _normalize_logical_axis_rules(logical_axis_rules, mesh)
    fsdp_size = mesh.shape[FSDP_AXIS]
    sharded_devices = math.prod(mesh.shape[axis] for axis in PARAMETER_AXES)

    def leaf_sharding(path, value):
        axes = logical_axes(path, value.ndim)
        size = int(np.prod(value.shape, dtype=np.int64))
        if axes is None:
            spec = parameter_spec(value.shape, fsdp_size, min_shard_size)
        elif sharded_devices == 1 or size < min_shard_size:
            spec = P()
        else:
            spec = _mesh_spec(value.shape, axes, rules, mesh)
        return NamedSharding(mesh, spec)

    return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(abstract_state))


def assert_params_sufficiently_sharded(
    params, shardings, mesh: Mesh,
    tolerance: float = DEFAULT_SHARDING_TOLERANCE,
    min_shard_size: int = DEFAULT_MIN_SHARD_SIZE,
) -> None:
    """Reject a layout that left too much of the model replicated.

    MaxText's guardrail (base.yml sharding_tolerance) against a mesh whose
    parameter axes divide none of the model's dimensions, which the shape
    heuristic otherwise absorbs in silence.

    MaxText measures excess per-chip memory over perfect sharding across every
    parameter. Here the same ratio is taken over the parameters the threshold
    policy meant to shard: anything below min_shard_size is replicated on
    purpose, so counting it would fire on models that are merely small.
    """
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError(
            f"sharding_tolerance must be between 0 and 1, got {tolerance}")
    if all(mesh.shape[axis] == 1 for axis in PARAMETER_AXES):
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
        if any(axis in _mesh_axes(assignment)
               for assignment in sharding.spec for axis in PARAMETER_AXES):
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
            self._iterator.set_state(self._position_for(source_state))
        # Position of the source iterator as of the batch most recently handed
        # out, so a checkpoint resumes at the next unseen batch rather than at
        # whatever the prefetch thread has already raced ahead to.
        self.source_state = source_state
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._thread.start()

    def _position_as_bytes(self, state):
        """A position a checkpoint can carry: grain's DataLoader iterator
        reports JSON bytes, its Dataset iterator (the packed loader) a dict,
        and the checkpoint holds one uint8 array either way."""
        return state if isinstance(state, bytes) else json.dumps(state).encode()

    def _position_for(self, saved: bytes):
        """`saved` back in the shape this iterator's set_state reads."""
        return saved if isinstance(self._iterator.get_state(), bytes) else json.loads(saved)

    def _prefetch(self):
        try:
            while True:
                batch = next(self._iterator)
                state = (self._position_as_bytes(self._iterator.get_state())
                         if self._checkpointable else None)
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
