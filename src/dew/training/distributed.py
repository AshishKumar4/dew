"""Device mesh, parameter layout, and host-to-device prefetch."""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import functools
import json
import logging
import math
import queue
import threading
from collections.abc import Mapping
from typing import Iterator

import jax
import numpy as np
from flax import linen as nn
from jax.experimental import mesh_utils
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P

from dew.data.dataset import Budgeted, Checkpointable, Closeable, DataPartition, Stoppable
from dew.nn.inputs import filled_validity
from dew.nn.sharding import (
    BATCH_AXES,
    DEFAULT_RULES,
    EXPERT_AXIS,
    FSDP_AXIS,
    MESH_AXES,
    SEQUENCE_AXIS,
    TENSOR_AXIS,
    LogicalAxisRules,
    MeshAxes,
    declared_axes,
    logical_spec,
    mesh_axes,
)
from dew.objectives.base import Batch, Variables
from dew.telemetry.profile import region

# The axes a parameter can be split over. A dimension named 'exp' takes the
# expert axis, the widths of Megatron's split take tensor, everything else
# takes fsdp. The data and sequence axes split the batch, and the stage axis
# holds the pipeline's stages of the layer stack. None of the three ever
# places a parameter, which is why `Layout` refuses a parameter placed there.
PARAMETER_AXES = (EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS)

# The batch's rows split over the batch axes and its sequence dimension over
# the sequence axis. Every stage sees the whole batch, since the pipeline
# hands its microbatches from stage to stage itself.
BATCH_SPEC = P(BATCH_AXES, SEQUENCE_AXIS)

type Placement[TreeT] = TreeT
"""`TreeT`'s own structure, with a `NamedSharding` at every leaf.

A placement is built by mapping over the tree it places, so it is that tree's
type: the same dataclass with the same fields, the same records under the same
keys. Only the leaves differ, and Python has no way to say "this structure
with those leaves". So the parameter carries the structure and this name
carries the leaves. Every caller reads the leaves as shardings, through
`jax.jit`, `device_put` or `Layout.check`."""

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MeshSpec:
    """Says how many devices each sharding axis takes; data parallelism fills the rest."""
    fsdp: int = 1
    expert: int = 1
    """Devices the expert dimension of an MoE layer is split over."""
    tensor: int = 1
    """Devices the mlp, head and vocabulary widths are split over, beside the
    fsdp axis they also take; 1 keeps every width on fsdp alone."""
    sequence: int = 1
    """Devices the batch's sequence dimension is split over; 1 keeps whole sequences."""
    stage: int = 1
    """Pipeline stages the layer stack is split into, each on its own devices; 1
    runs the stack whole on every device."""
    microbatches: int | None = None
    """Microbatches a step feeds through the stages, a multiple of `stage`; None
    is one per stage, the smallest schedule. A stage runs one microbatch while
    the next runs the one before it. So more microbatches shrink the idle time
    at either end of the step, and make each iteration's matmuls smaller."""
    replicas: int = 1
    """Groups of hosts the data axis spans, for hybrid sharded data
    parallelism: every other axis, fsdp included, stays inside one group,
    so the parameter gathers and gradient reduce-scatters run over the fast
    links and only the gradient all-reduce between replicas crosses the
    slow one. A group is a whole number of granules: whatever the devices'
    slice_index groups (a TPU slice, and on multi-host GPU a host or an
    NVLink domain), or the process where every device shares one slice.
    1 lets `jax.make_mesh` place every device."""

    def __post_init__(self):
        if self.stage < 1:
            raise ValueError(f"stage counts pipeline stages, got {self.stage}")
        if self.replicas < 1:
            raise ValueError(f"replicas counts host groups, got {self.replicas}")
        if self.microbatches is None:
            return
        if self.stage == 1:
            raise ValueError(
                f"microbatches ({self.microbatches}) feed the stage axis, and "
                "stage is 1; set stage above 1 or leave microbatches unset")
        if self.microbatches < self.stage or self.microbatches % self.stage:
            raise ValueError(
                f"microbatches must be a positive multiple of stage "
                f"({self.stage}), got {self.microbatches}")


def _rule_table(rules: LogicalAxisRules | Mapping[str, MeshAxes]) -> LogicalAxisRules:
    """Return `rules` as the tuple of pairs flax reads, in precedence order.

    They are written as a mapping in code, as pairs on the command line, and
    arrive as lists from a JSON record."""
    pairs = rules.items() if isinstance(rules, Mapping) else rules
    return tuple((name, axes if axes is None or isinstance(axes, str) else tuple(axes))
                 for name, axes in pairs)



def build_mesh(spec: MeshSpec = MeshSpec(), devices: list | None = None) -> Mesh:
    """Build the six-axis device mesh `spec` describes.

    Parameters shard over 'fsdp', 'expert' and 'tensor', batches over the
    first four with their sequence dimension on 'sequence', and the layer
    stack over 'stage'.

    An MoE layer's expert dimension is the one dimension no dense model has,
    and splitting it is what expert parallelism is. So it gets its own axis
    and leaves 'fsdp' to the model's widths. The tensor axis is where the
    mlp, head and vocabulary widths split beside fsdp, as `DEFAULT_RULES`
    places them. The sequence axis is where long sequences split. The stage
    axis is where a decoder's layers split into pipeline stages, each stage
    on its own devices.

    Sizes of 1 degenerate to plain data parallelism, so the same code path
    serves every topology without a flag. Axes are Auto so GSPMD infers the
    collectives.

    `spec.replicas` above 1 builds the mesh the way MaxText builds a
    multislice one, through `mesh_utils.create_hybrid_device_mesh`: the
    data axis takes `replicas` groups of granules as its outer factor, and
    each group lays out the rest of the mesh over its own devices. A group
    of more than one granule splits fsdp across them, the one axis whose
    traffic, a gather and a reduce-scatter per layer, tolerates it.
    """
    devices = list(devices) if devices is not None else jax.devices()
    sharded = spec.fsdp * spec.expert * spec.tensor * spec.sequence * spec.stage
    if (spec.fsdp < 1 or spec.expert < 1 or spec.tensor < 1
            or spec.sequence < 1 or len(devices) % sharded):
        raise ValueError(
            f"fsdp {spec.fsdp} times expert {spec.expert} times tensor "
            f"{spec.tensor} times sequence {spec.sequence} times stage "
            f"{spec.stage} must be a positive divisor of device count {len(devices)}")
    shape = (len(devices) // sharded, spec.expert, spec.fsdp, spec.tensor, spec.sequence,
             spec.stage)
    # `jax.make_mesh` lays out one slice; devices on several, every GPU host
    # its own, take the hybrid layout even as one replica.
    if spec.replicas == 1 and len({_slice(device) for device in devices}) == 1:
        return jax.make_mesh(shape, MESH_AXES, devices=devices, axis_types=(AxisType.Auto,) * 6)
    return Mesh(hybrid_devices(spec, shape, devices), MESH_AXES, axis_types=(AxisType.Auto,) * 6)


def _slice(device) -> int:
    """The slice a device sits on; one outside any process pool, such as a
    lone CPU process's, carries no slice_index and sits on the one there is."""
    return device.slice_index if hasattr(device, "slice_index") else 0


def hybrid_devices(spec: MeshSpec, shape: tuple[int, ...], devices: list) -> np.ndarray:
    """The device array of a mesh whose data axis spans `spec.replicas` host groups.

    A granule is the unit the slow network joins: whatever the devices'
    `slice_index` groups where they report more than one, and the process
    where they all share one. A multislice TPU run reports one slice index
    per TPU slice. XLA numbers GPU slices per host boot or per NVLink fabric
    (`BuildGlobalTopology`, xla/pjrt/distributed/topology_util.cc), so on
    several GPU hosts a granule is a host or an NVLink domain however many
    processes each runs. CPU pools, the hosts of one TPU slice, and
    several processes on one machine share slice 0, and there the process
    is the granule.

    Devices on several slices with no replicas asked for put the slices on
    the data axis where it divides, as MaxText's default DCN data
    parallelism does, so only the gradient sum crosses the slow network;
    otherwise the slices split fsdp.
    """
    by_process = len({_slice(device) for device in devices}) == 1
    granules = len({device.process_index if by_process else device.slice_index
                    for device in devices})
    data_width = shape[0]
    replicas = spec.replicas
    if replicas == 1 and not by_process and data_width % granules == 0:
        replicas = granules
    if granules % replicas or data_width % replicas:
        raise ValueError(
            f"replicas {replicas} must divide both the {granules} granules "
            f"(slices, or processes) the devices form and the data axis of {data_width}")
    per_replica = granules // replicas
    if spec.fsdp % per_replica:
        raise ValueError(
            f"each of the {replicas} replicas spans {per_replica} granules, "
            f"which only the fsdp axis may cross, and fsdp {spec.fsdp} does not "
            "divide over them")
    dcn = (replicas, 1, per_replica, 1, 1, 1)
    ici = tuple(size // outer for size, outer in zip(shape, dcn, strict=True))
    return mesh_utils.create_hybrid_device_mesh(
        ici, dcn, devices, process_is_granule=by_process, allow_split_physical_axes=True)


def batch_divisor(mesh: Mesh, spec: MeshSpec) -> int:
    """Return the row count every global batch must be a multiple of.

    A batch's rows are sharded over the batch axes as whole rows, and a
    pipelined step cuts them again into `spec`'s microbatches, one per stage
    when unset. A batch that divides by neither fails where it is placed or
    traced.

    A run whose batch never changes hits that on its first step. A batch ramp
    is checked against this before it reads, since a later stage would
    otherwise fail an hour in.
    """
    shards = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
    return math.lcm(shards, spec.microbatches or spec.stage)


def parameter_spec(shape: tuple, fsdp_size: int, min_shard_size: int) -> P:
    """Shard the largest evenly-divisible axis over 'fsdp', else replicate.

    Applied to every leaf of the train state, params, optimizer moments and
    EMA copies alike. The moments and the copies have the same shapes as the
    params they track, so they pick up the same spec without anyone having to
    describe the optimizer's layout.
    """
    if fsdp_size == 1 or int(np.prod(shape, dtype=np.int64)) < min_shard_size:
        return P()
    for axis in sorted(range(len(shape)), key=lambda i: -shape[i]):
        if shape[axis] % fsdp_size == 0:
            return P(*([None] * axis), FSDP_AXIS)
    return P()


HOST_RESIDENT = ("params", "opt_state", "ema")
"""The train-state fields a layout may keep in pinned host memory between
steps. Naming params selects a CPU-owned complete transaction state, including
optimizer, EMA and accumulation. The accelerator scan fetches parameter rows
and rematerialization refetches them in backward. Naming only opt_state or
ema retains accelerator execution with pinned-host storage."""


def _variable_path(path) -> str:
    """Spell a leaf's logical path as `host_parameters` patterns are written:
    `params/layers_3/self_attn/q_proj/kernel`, the collection first."""
    return "/".join(
        entry.key if isinstance(entry, jax.tree_util.DictKey) else str(entry.idx)
        for entry in path)


def host_selected(patterns: tuple[str, ...], path: str) -> bool:
    """Whether `path` is one of the variables `patterns` names.

    A pattern is an `fnmatch` glob over the path, and a pattern that names a
    subtree covers it whole, so `params/layers_3` selects that layer's every
    leaf and `params/layers_*` the stack's.
    """
    return any(fnmatch.fnmatchcase(path, pattern)
               or fnmatch.fnmatchcase(path, f"{pattern}/*") for pattern in patterns)


@dataclasses.dataclass(frozen=True)
class Layout:
    """Says how a train state is placed on a mesh.

    `rules` map the logical axes the modules declare (`dew.nn.sharding`) onto
    the mesh, in precedence order, for the parameters and for the activations
    a compiled step constrains (the trainer puts them in context). An axis of
    size 1 shards nothing, so the same table serves every topology.

    A parameter the rules place on the data, sequence or stage axis is
    refused. The first two split the batch, and a parameter placed on either
    would be gathered on every use. The stage axis holds the layer stack's
    pipeline stages, which the decoder places itself from the stored tree.

    Below `min_shard` elements a parameter costs more in collectives than it
    saves in memory, so it stays replicated. `tolerance` is the fraction of
    shardable parameter elements a layout may leave replicated before `check`
    refuses it.

    `host` names the fields of `HOST_RESIDENT` kept in pinned host memory
    between steps. The step fetches them to the device, updates them as it
    would have, and writes them back, so what they hold is the same and only
    where changes. Naming params instead selects canonical CPU ownership of
    the entire TrainState, including optimizer, EMA and accumulation. The
    full logical optimizer transaction then runs on a CPU companion of this
    mesh, and accelerator parameter banks are immutable execution snapshots,
    never another master. The runtime CPU device count must match the
    accelerator count on every process before JAX initializes; neither this
    layout nor the trainer changes it.

    `host_parameters` names the variables an inference placement keeps in
    pinned host memory, as globs over their logical paths
    (`params/layers_*`). Where a parameter sits is independent of how it
    splits: a selected leaf keeps the spec the rules give it and changes only
    its memory kind. Only `offloaded` reads the patterns, because only the
    stack fetches a layer's parameters as it reaches it. `check` refuses a
    layout that names them to any other placement, rather than place the
    weights somewhere nothing brings them back from.
    """
    rules: LogicalAxisRules | Mapping[str, MeshAxes] = DEFAULT_RULES
    min_shard: int = 2 ** 16
    tolerance: float = 0.02
    host: tuple[str, ...] = ()
    host_parameters: tuple[str, ...] = ()

    def __post_init__(self):
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(
                f"sharding tolerance must be between 0 and 1, got {self.tolerance}")
        rules = _rule_table(self.rules)
        for name, axes in rules:
            unknown = [axis for axis in mesh_axes(axes) if axis not in MESH_AXES]
            if unknown:
                raise ValueError(
                    f"rule {name!r} names {unknown}, which no mesh has; the axes are "
                    f"{list(MESH_AXES)}")
        object.__setattr__(self, "rules", rules)
        host = tuple(self.host)
        unknown = sorted(set(host) - set(HOST_RESIDENT))
        if unknown:
            raise ValueError(
                f"host names the train-state fields kept in pinned host memory, "
                f"{list(HOST_RESIDENT)}, got {unknown}")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "host_parameters", tuple(self.host_parameters))

    @property
    def axis_rules(self) -> LogicalAxisRules:
        """The rules as flax reads them, pairs in precedence order."""
        rules = self.rules
        assert isinstance(rules, tuple), "__post_init__ keeps the rules as pairs"
        return rules

    def shardings[TreeT](self, mesh: Mesh, tree: TreeT) -> Placement[TreeT]:
        """Derive a NamedSharding per leaf of `tree` from the declared axes.

        A leaf whose path no module declares takes the largest-divisible-axis
        heuristic, so a model family can be declared at a time. Flax
        metadata, if a caller's own module attached any, is removed here,
        because the state the trainer materialises against this tree carries
        plain arrays.
        """
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
                spec = logical_spec(axes, value.shape, rules=self.axis_rules, mesh=mesh)
            outside = sorted({axis for entry in spec for axis in mesh_axes(entry)}
                             - set(PARAMETER_AXES))
            if outside:
                raise ValueError(
                    f"the rules place {_variable_path(path)} on {outside}; parameters "
                    f"split over {list(PARAMETER_AXES)}, the data and sequence axes "
                    f"split the batch, and the stage axis holds the pipeline")
            return NamedSharding(mesh, spec)

        return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(tree))

    def offloaded[TreeT](self, mesh: Mesh, tree: TreeT) -> Placement[TreeT]:
        """Return `shardings`, with the memory kind each leaf's path asks for.

        The spec is the one the rules give a leaf either way, so a selected
        parameter is the same shard in another memory space and the
        collectives its layer issues are the ones a resident run issues.

        Every pattern has to name something. One that matches nothing is a
        typo or a stale path, and a placement that quietly kept those weights
        on the device would run, with only the memory it did not save to say
        so. So each pattern is checked on its own, not the table as a whole.
        """
        placed = self.shardings(mesh, tree)
        if not self.host_parameters:
            return placed
        paths = [_variable_path(path)
                 for path, _ in jax.tree_util.tree_flatten_with_path(placed)[0]]
        unmatched = [pattern for pattern in self.host_parameters
                     if not any(host_selected((pattern,), path) for path in paths)]
        if unmatched:
            raise ValueError(
                f"host_parameters {unmatched} names none of this tree's "
                f"{len(paths)} variables; the paths start {sorted(paths)[:3]}")
        return jax.tree_util.tree_map_with_path(
            lambda path, sharding: (
                sharding.with_memory_kind("pinned_host")
                if host_selected(self.host_parameters, _variable_path(path)) else sharding),
            placed)

    def check(self, params: Variables, shardings: Placement[Variables], mesh: Mesh) -> None:
        """Reject a layout that left too much of the model replicated, or that
        asked for host-resident parameters where nothing fetches them.

        This is MaxText's guardrail (base.yml sharding_tolerance) against a
        mesh whose parameter axes divide none of the model's dimensions,
        which the shape heuristic otherwise absorbs by replicating
        everything.

        MaxText measures excess per-chip memory over perfect sharding across
        every parameter. Here the same ratio is taken over the parameters the
        threshold policy meant to shard. Anything below min_shard is
        replicated on purpose, so counting it would fire on models that are
        merely small.
        """
        if self.host_parameters and not any(
                sharding.memory_kind == "pinned_host"
                for sharding in jax.tree.leaves(shardings)):
            raise ValueError(
                f"host_parameters {list(self.host_parameters)} keeps those weights "
                f"in pinned host memory, which only a stack that fetches a layer's "
                f"parameters as it reaches it reads; this placement keeps every "
                f"parameter on the device. Place the weights for generation with "
                f"dew.inference.host_banked, or use host=('params',) for a "
                f"CPU-owned training transaction and drop the inference-only patterns")
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
            if any(axis in mesh_axes(assignment)
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


def batch_shardings(mesh: Mesh | AbstractMesh, batch: Batch) -> Placement[Batch]:
    """Derive a sharding per leaf of `batch` from the spec and the leaf's shape.

    Rows split over the data, expert, fsdp, and tensor axes. A leaf of rank 2
    or 3 is a sequence per row: token ids, segment ids, positions, encoded
    tokens. Its second dimension splits over the sequence axis when the axis
    divides it, and otherwise stays whole, the way `logical_spec` drops a name
    no dimension can split. An image or a video is not a sequence, so only
    its rows split.

    Placement is all this decides: values never change, and until a model
    constrains its attention to the axis, the sequence splits here and
    gathers there. Only the shape is read, so a leaf that is already a global
    array costs no transfer.
    """
    rows, sequence = BATCH_SPEC
    sequence_size = mesh.shape[SEQUENCE_AXIS]

    def leaf_sharding(leaf):
        shape = np.shape(leaf)
        if not shape:
            return NamedSharding(mesh, P())
        if len(shape) in (2, 3) and shape[1] % sequence_size == 0:
            return NamedSharding(mesh, P(rows, sequence))
        return NamedSharding(mesh, P(rows))

    return jax.tree.map(leaf_sharding, batch)


@functools.cache
def data_partition(mesh: Mesh) -> DataPartition:
    """The share of every global batch this process reads on `mesh`.

    A batch's rows split over the batch axes and no others (`BATCH_SPEC`):
    the sequence axis splits positions, the tensor axis widths, and the stage
    axis holds a pipeline's stages. So the processes whose devices hold the
    same row shards need the same rows, and the processes fall into groups by
    the rows they hold.
    Each group reads one share, numbered by the first row shard it holds,
    and every process of the group reads it (`readers`).

    Groups whose rows overlap without being the same rows, which a device
    order built by hand can produce, leave no share each could read whole,
    so they are refused.
    """
    shards = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
    held: dict[int, set[int]] = {}
    placement = NamedSharding(mesh, P(BATCH_AXES)).devices_indices_map((shards,))
    for device, index in placement.items():
        held.setdefault(device.process_index, set()).add(index[0].start or 0)
    groups = sorted({frozenset(rows) for rows in held.values()}, key=min)
    if (sum(len(group) for group in groups) != shards
            or len({len(group) for group in groups}) != 1):
        raise ValueError(
            f"the processes of this mesh hold the row shards "
            f"{ {process: sorted(rows) for process, rows in sorted(held.items())} }, "
            f"which overlap without being the same; each group of processes has to "
            f"hold rows no other group holds, so it can read them as its own share")
    mine = frozenset(held[jax.process_index()])
    return DataPartition(index=groups.index(mine), count=len(groups),
                         readers=sum(frozenset(rows) == mine for rows in held.values()))


def shard_batch(mesh: Mesh, batch: Batch) -> Batch:
    """Assemble this process's share of each array into a globally sharded one.

    The share is the one `data_partition(mesh)` names: that share's rows, each
    whole in every other dimension. A pool assembles one leaf per process,
    so every process has to hand this
    the same tree. Validity is the one optional token field, and whether a
    process's own rows needed padding is rank-local. So a pool materializes
    it at every `ModelInputs` of the batch that lacks it, before the
    traversal below reads the leaves. The schema then depends on the process
    count and the batch's structure and never on which rows this process
    drew. One process changes nothing and keeps the omission the fused
    attention kernel wants.

    The rule is deliberately local. Placement runs on
    `DevicePrefetchIterator`'s worker thread while the step's collectives run
    on the caller's, and a collective issued from here would have to be
    ordered against those across every process. Agreeing which sites
    actually carry validity, the way `agreed_validity` does for a generation
    request, belongs where the caller's own collectives are issued.
    """
    batch = filled_validity(batch) if jax.process_count() > 1 else batch
    count = data_partition(mesh).count

    def place(leaf, sharding: NamedSharding) -> jax.Array:
        # The share holds whole rows: `count` shares make the rows, and every
        # other dimension is already whole, so a device of a sequence or a
        # stage that spans processes picks its own slice out of it.
        local = leaf if isinstance(leaf, jax.Array) else np.asarray(leaf)
        shape = np.shape(local)
        return jax.make_array_from_process_local_data(
            sharding, local, (shape[0] * count, *shape[1:]) if shape else ())

    return jax.tree.map(place, batch, batch_shardings(mesh, batch))


class DevicePrefetchIterator:
    """Reads batches on a worker thread and places them on the mesh ahead of the step.

    The iterator owns its source: use it as a context manager, or call
    `close`, even after a loop that ended early. At most `depth` batches are
    queued, plus the one being read and placed.

    Every call into the source runs on the worker thread: `next`, the
    checkpoint position, and the final close. An optional `request_stop` hook
    runs on the closing thread instead, and must be thread-safe and
    non-blocking.

    If thread creation itself fails, caller-thread finalization runs before
    re-raising; no worker has touched the source. That synchronous error path
    requires a cooperative finalizer. Active-worker close reports a timeout
    when arbitrary source code cannot be stopped.
    """

    def __init__(self, iterator: Iterator, mesh: Mesh, depth: int = 2,
                 source_state: bytes | None = None):
        if depth <= 0:
            raise ValueError("prefetch depth must be positive")
        self._iterator: Iterator | None = iter(iterator)
        self._source_name = type(self._iterator).__name__
        self._mesh: Mesh | None = mesh
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._start_lock = threading.Lock()
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self.source_state = source_state
        self._thread = threading.Thread(target=self._prefetch, name="dew-prefetch", daemon=True)
        # No source work may start until the caller owns this object. In
        # particular, an interrupt during restoration must unwind through
        # this iterator's close, never the caller's untransferred-source path.

    def _start(self) -> None:
        """Start the worker on first use, finalizing here if it cannot start."""
        with self._start_lock:
            if self._done.is_set():
                return
            if self._thread.ident is None:
                try:
                    self._thread.start()
                except RuntimeError as error:
                    if self._thread.ident is None:
                        # Thread creation failed before any source operation.
                        self._stop.set()
                        try:
                            if isinstance(self._iterator, Stoppable):
                                self._iterator.request_stop()
                        except BaseException as failure:
                            error.add_note(f"Source cancellation failed: {failure!r}")
                        self._prefetch()
                    raise

    def _prefetch(self):
        """Read, place and enqueue batches until the source drains or a stop.

        This is the worker thread's whole body, and it owns the source's
        finalization too, so the source is touched from one thread only. A
        drained iterator ends the loop through `StopIteration`, any other
        failure is published to the consumer, and the `finally` closes the
        source either way.

        `_start` calls this on the caller's thread when the worker could not
        be created. The stop flag is already set there, so the loop does
        nothing but finalize.
        """
        iterator, mesh = self._iterator, self._mesh
        assert iterator is not None and mesh is not None
        source = iterator if isinstance(iterator, Checkpointable) else None
        batch = state = placed = None
        try:
            if not self._stop.is_set() and self.source_state is not None:
                if source is None:
                    raise TypeError(f"{self._source_name} cannot resume from a saved position")
                saved = self.source_state
                source.set_state(saved if isinstance(source.get_state(), bytes)
                                 else json.loads(saved))
            while not self._stop.is_set():
                with region("data.read"):
                    batch = next(iterator)
                if self._stop.is_set():
                    break
                with region("data.position"):
                    state = source.get_state() if source is not None else None
                    if source is not None and not isinstance(state, bytes):
                        state = json.dumps(state).encode()
                with region("data.place"):
                    placed = shard_batch(mesh, batch)
                batch = None
                with region("data.enqueue"):
                    while not self._stop.is_set():
                        try:
                            self._queue.put((placed, state), timeout=0.05)
                            break
                        except queue.Full:
                            continue
                placed = state = None
        except StopIteration:
            # A drained source is how a prefetch thread ends; the finally
            # below publishes the end, and no error goes with it.
            _log.debug("%s drained; the prefetch thread is done", self._source_name)
        except BaseException as error:
            self._error = error
        finally:
            batch = placed = state = None
            try:
                close = iterator.close if isinstance(iterator, Closeable) else None
                if close is not None:
                    close()
            except BaseException as error:
                self._cleanup_error = error
            finally:
                self._iterator = self._mesh = None
                # No bound close method or checkpointable alias may retain a
                # Grain pipeline after the worker exits.
                iterator = source = mesh = close = None
                if self._stop.is_set():
                    self._discard()
                self._done.set()

    def _discard(self) -> None:
        """Drop every batch already queued, without waiting for more."""
        with contextlib.suppress(queue.Empty):
            while True:
                self._queue.get_nowait()

    def close(self, *, timeout: float | None = None) -> None:
        """Cancel and join, discarding unread batches, not consumed position.

        The join waits `timeout`, else the source's own `stop_seconds` (a
        grain pipeline joins one worker process after another), else 5 s. A
        timeout leaves cancellation requested; a later close may join again.
        It does not mean the source's in-flight work or buffers were released.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("a prefetch worker cannot close itself")
        if timeout is None:
            seconds = self._iterator.stop_seconds if isinstance(self._iterator, Budgeted) else None
            timeout = 5.0 if seconds is None else float(seconds)
        if timeout < 0:
            raise ValueError("close timeout must be nonnegative")
        first_stop = not self._stop.is_set()
        self._stop.set()
        error = None
        try:
            if first_stop and isinstance(self._iterator, Stoppable):
                self._iterator.request_stop()
        except BaseException as failure:
            error = failure
        self._discard()
        self._start()  # Even an unused iterator finalizes on its worker.
        if self._thread.ident is not None:
            self._thread.join(timeout)
        self._discard()
        self._error = None  # Unconsumed speculative errors are discarded too.
        failure = None
        if self._thread.is_alive():
            failure = TimeoutError(
                f"{self._source_name} did not stop within {timeout}s: its next, "
                "placement or finalization is uncooperative; the worker is still alive")
        elif self._cleanup_error is not None:
            failure, self._cleanup_error = self._cleanup_error, None
        if error is not None:
            if failure is not None:
                error.add_note(f"Prefetch cleanup also failed: {failure!r}")
            raise error
        if failure is not None:
            raise failure

    def __enter__(self) -> DevicePrefetchIterator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"Prefetch cleanup failed: {error!r}")

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop.is_set():
            raise StopIteration
        self._start()
        while not self._stop.is_set():
            try:
                batch, position = self._queue.get(timeout=0.05)
            except queue.Empty:
                if not self._done.is_set():
                    continue
                # The final publication may race the timed-out get.
                try:
                    batch, position = self._queue.get_nowait()
                except queue.Empty:
                    error, self._error = self._error, None
                    cleanup, self._cleanup_error = self._cleanup_error, None
                    if error is not None:
                        if cleanup is not None:
                            error.add_note(f"Source cleanup failed: {cleanup!r}")
                        raise error from None
                    if cleanup is not None:
                        raise cleanup from None
                    raise StopIteration from None
            if self._stop.is_set():
                break
            self.source_state = position
            return batch
        raise StopIteration
