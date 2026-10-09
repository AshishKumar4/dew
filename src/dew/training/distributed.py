"""Device mesh, parameter layout, and host-to-device prefetch."""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import json
import logging
import math
import queue
import statistics
import threading
import time
from collections.abc import Callable, Iterator, Mapping

import jax
import numpy as np
from flax import linen as nn
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P

from dew.coordination import agreed, broadcast_from_process_zero, from_every_process, stop_at_exit
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
    LayoutRefused,
    LogicalAxisRules,
    MeshAxes,
    boxed,
    boxed_axes,
    current_boxes,
    declared_axes,
    logical_spec,
    mesh_axes,
    ruled_boxes,
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
    """Device counts for the mesh's sharding axes; the data axis takes the devices left over."""
    fsdp: int = 1
    expert: int = 1
    """Devices the expert dimension of an MoE layer is split over."""
    tensor: int = 1
    """Devices the mlp, head and vocabulary widths are split over, in addition
    to fsdp. 1 keeps every width on fsdp alone."""
    sequence: int = 1
    """Devices the batch's sequence dimension is split over; 1 keeps whole sequences."""
    stage: int = 1
    """Pipeline stages the layer stack is split into, each on its own devices.
    1 runs the whole stack on every device."""
    microbatches: int | None = None
    """Microbatches a step feeds through the stages, a positive multiple of
    `stage`. None means one per stage, the smallest schedule, and any other
    value needs `stage` above 1. Each stage works on one microbatch while
    the next stage works on the previous one, so more microbatches shrink
    the idle time at the start and end of a step, and make each iteration's
    matmuls smaller."""
    replicas: int = 1
    """Groups of hosts the data axis spans, for hybrid sharded data
    parallelism. Every other axis, fsdp included, stays inside one group. So
    the parameter gathers and gradient reduce-scatters run over the fast
    links, and only the gradient all-reduce between replicas crosses the
    slow network. A group is a whole number of granules. A granule is the
    devices that share a slice_index (a TPU slice, and on multi-host GPU a
    host or an NVLink domain), or one process when every device reports the
    same slice. At 1, `jax.make_mesh` places the devices of a single slice;
    devices on several slices still take the hybrid layout, with each slice
    as one replica when the slice count divides the data axis."""
    explicit: tuple[str, ...] = ()
    """The axes whose sharding an array's type carries (JAX's Explicit mode),
    where every other axis is left to the partitioner (Auto). An Explicit
    axis is placed where an array is made and follows the operations from
    there; a model's sharding constraints name only the Auto ones."""

    def __post_init__(self):
        unknown = sorted(set(self.explicit) - set(MESH_AXES))
        if unknown:
            raise ValueError(f"explicit names the mesh's axes {list(MESH_AXES)}, got {unknown}")
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

    def build(self, devices: list | None = None) -> Mesh:
        """Build the six-axis device mesh this spec describes.

        Parameters shard over 'fsdp', 'expert' and 'tensor'. A batch's rows
        shard over 'data', 'expert' and 'fsdp', its sequence dimension over
        'sequence', and the layer stack over 'stage'.

        Expert parallelism splits an MoE layer's expert dimension, which no
        dense model has. So that dimension gets its own axis, and 'fsdp' is
        left for the model's widths. The tensor axis splits the mlp, head and
        vocabulary widths alongside fsdp, as `DEFAULT_RULES` places them. The
        sequence axis splits long sequences, and the stage axis splits a
        decoder's layers into pipeline stages, each stage on its own devices.

        An axis of size 1 splits nothing, and with every size at 1 the mesh is
        plain data parallelism. So the same code path serves every topology
        without a flag. The axes are Auto, so GSPMD infers the collectives,
        but for those `explicit` names.

        `devices` lists the devices of one slice in the order to use them. The
        mesh takes them row-major over `MESH_AXES`, so neighbouring entries
        land on the last axes. When it is unset, the mesh uses every device, in
        the order `jax.make_mesh` gives the platform's topology.

        When `replicas` is above 1, or the devices are on several slices, the
        mesh is built the way MaxText builds a multislice one, through
        `mesh_utils.create_hybrid_device_mesh`. The data axis takes `replicas`
        groups of granules as its outer factor, and each group lays out the
        rest of the mesh over its own devices. A group of more than one granule
        splits fsdp across them, because fsdp's traffic (a gather and a
        reduce-scatter per layer) is the only kind that tolerates the slower
        link.

        Raises `LayoutRefused` when the expert, fsdp, tensor, sequence and
        stage sizes multiply to a number that does not divide the device count,
        or when `replicas` does not divide both the granule count and the data
        axis.
        """
        named = devices is not None
        devices = list(devices) if devices is not None else jax.devices()
        sizes = {"expert": self.expert, "fsdp": self.fsdp, "tensor": self.tensor,
                 "sequence": self.sequence, "stage": self.stage}
        sharded = math.prod(sizes.values())
        if any(size < 1 for size in sizes.values()):
            raise LayoutRefused(f"every axis of a mesh is at least 1, and {sizes} is not")
        if len(devices) % sharded:
            split = " x ".join(f"{axis} {size}" for axis, size in sizes.items() if size > 1)
            raise LayoutRefused(f"{split} is {sharded} devices a data replica, which does not divide "
                                f"the {len(devices)} devices")
        shape = (len(devices) // sharded, self.expert, self.fsdp, self.tensor, self.sequence,
                 self.stage)
        # `jax.make_mesh` lays one slice out by the platform's topology, which on
        # GPU is the devices sorted by id whatever order they came in; a list the
        # caller names is the layout itself, filled in its own order. Devices on
        # several slices, every GPU host its own, take the hybrid layout even as
        # one replica.
        types = tuple(AxisType.Explicit if axis in self.explicit else AxisType.Auto for axis in MESH_AXES)
        if self.replicas == 1 and len({_slice(device) for device in devices}) == 1:
            if named:
                return Mesh(np.asarray(devices).reshape(shape), MESH_AXES, axis_types=types)
            return jax.make_mesh(shape, MESH_AXES, devices=devices, axis_types=types)
        return Mesh(hybrid_devices(self, shape, devices), MESH_AXES, axis_types=types)


def _rule_table(rules: LogicalAxisRules | Mapping[str, MeshAxes]) -> LogicalAxisRules:
    """Return `rules` as the tuple of pairs flax reads, in precedence order.

    They are written as a mapping in code, as pairs on the command line, and
    arrive as lists from a JSON record."""
    pairs = rules.items() if isinstance(rules, Mapping) else rules
    return tuple((name, axes if axes is None or isinstance(axes, str) else tuple(axes))
                 for name, axes in pairs)



def link_bandwidth(mesh: Mesh, axis: str, size: int = 1 << 28) -> float:
    """What one device receives a second in an all-gather over `mesh`'s
    `axis` of a `size`-byte result, with every group along the axis
    gathering at once as a step's do: (N - 1) / N of the result over the
    median of five gathers, after two that warm the collective. Every
    process takes the pool's lowest figure, so every process decides from
    the same number and compiles the same program; the slowest group bounds
    the step anyway.

    `dew.nn.sharding.down_projection` reads the tensor axis's to decide
    whether a down-projection of the residual runs on each tensor shard's
    own tokens, and `dew.nn.sharding.split_positions` the sequence axis's.
    The default result, 256 MiB, of which a device receives at least half,
    is the size of the collectives those decisions price where they matter
    (DeepSeek-V3's residual gradient over 16384 tokens is 235 MB in bf16),
    past the sizes where a collective's latency counts: on 4x RTX 3090 an
    NVLink pair's all-gather moved 8.2 GB/s a device at 4 MiB and 31.0 at
    128 MiB."""
    ways = mesh.shape[axis]
    count = size // 4 // ways * ways
    sharding = NamedSharding(mesh, P(axis))
    source = jax.make_array_from_callback(
        (count,), sharding, lambda index: np.zeros(sharding.shard_shape((count,)), np.float32))
    gather = jax.jit(jax.shard_map(
        lambda shard: jax.lax.all_gather(shard, axis, tiled=True), mesh=mesh,
        in_specs=P(axis), out_specs=P(), axis_names={axis}, check_vma=False))
    seconds = []
    for attempt in range(7):
        began = time.perf_counter()
        jax.block_until_ready(gather(source))
        if attempt >= 2:
            seconds.append(time.perf_counter() - began)
    received = (ways - 1) / ways * count * 4 / statistics.median(seconds)
    return min(from_every_process(received))


def _slice(device) -> int:
    """The slice a device sits on; one outside any process pool, such as a
    lone CPU process's, carries no slice_index and sits on the one there is.
    jax's Device declares no such field, so it is read at this boundary."""
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
        raise LayoutRefused(
            f"replicas {replicas} must divide both the {granules} granules "
            f"(slices, or processes) the devices form and the data axis of {data_width}")
    per_replica = granules // replicas
    if spec.fsdp % per_replica:
        raise LayoutRefused(
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
    pipelined step cuts each device's rows again into `spec`'s microbatches,
    one per stage when unset: microbatch m takes rows m, m + M, ..., which
    is a share of every device's rows only where M divides them. A batch
    that divides by neither is refused where it is placed or traced.

    `Trainer.fit` checks a batch against this before it places anything,
    and a batch ramp's every stage, since a later stage would otherwise fail
    an hour in.
    """
    shards = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
    return shards * (spec.microbatches or spec.stage)


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


HOST_RESIDENT = ("variables", "opt_state", "ema")
"""The train-state fields a layout may keep in pinned host memory between
steps. Naming variables selects a CPU-owned complete transaction state, including
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
    """Decides how a train state is placed on a mesh.

    `rules` map the logical axis names that modules declare (`dew.nn.sharding`)
    onto mesh axes, in precedence order. They place the parameters, and the
    trainer sets them in context so they also place the activations a compiled
    step constrains. An axis of size 1 splits nothing, so the same table serves
    every topology.

    `shardings` refuses rules that place a parameter on the data, sequence or
    stage axis. The data and sequence axes split the batch, so a parameter on
    either would be gathered on every use. The stage axis holds the layer
    stack's pipeline stages, and the decoder places those itself from the
    stored tree.

    Below `min_shard` elements a parameter costs more in collectives than it
    saves in memory, so it stays replicated. `tolerance` is the fraction of
    shardable parameter elements a layout may leave replicated before `check`
    refuses it.

    `host` names the fields of `HOST_RESIDENT` to keep in pinned host memory
    between steps. For `opt_state` and `ema`, the step fetches the field to the
    device, updates it as usual and writes it back, so the values are the same
    and only where they are stored changes. Naming `variables` instead moves
    the whole TrainState to the CPU, including the optimizer state, the EMA and
    gradient accumulation. The optimizer update then runs on a CPU companion of
    this mesh, and the parameter copies on the accelerators are read-only
    snapshots for running the step, never a second master copy. Every process
    must have as many CPU devices as accelerators, set before JAX initializes;
    neither this layout nor the trainer changes the count.

    `host_parameters` lists the variables an inference placement keeps in
    pinned host memory, as globs over their logical paths (`params/layers_*`).
    A selected leaf keeps the spec the rules give it, and only its memory kind
    changes. Only `offloaded` reads the patterns, because only the layer stack
    fetches each layer's parameters as it reaches that layer. When
    `host_parameters` is set, `check` refuses a placement that keeps every
    parameter on the device, so the patterns are never silently ignored.
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
        """The rules as the (name, axes) pairs flax reads, in precedence order."""
        rules = self.rules
        assert isinstance(rules, tuple), "__post_init__ keeps the rules as pairs"
        return rules

    def shardings[TreeT](self, mesh: Mesh, tree: TreeT) -> Placement[TreeT]:
        """Return a NamedSharding for each leaf of `tree`, from the axes its module declared.

        A parameter its module boxed (`nn.with_logical_partitioning`) takes the
        axes it carries, in `tree` or in the trainer's `boxed` table, when the
        rules place one of them (`ruled_boxes`). A leaf
        whose path no module declares falls back to a shape heuristic,
        which splits its largest dimension that fsdp divides evenly over fsdp.
        So the axes of one model family can be declared at a time. A leaf below
        `min_shard` elements is replicated either way.

        Flax metadata that a caller's own module attached is removed, because
        the state the trainer creates from these shardings holds plain arrays.
        Raises ValueError when the rules place a leaf on the data, sequence or
        stage axis.
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

        with boxed(ruled_boxes({**current_boxes(), **boxed_axes(tree)}, self.axis_rules)):
            return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(tree))

    def offloaded[TreeT](self, mesh: Mesh, tree: TreeT) -> Placement[TreeT]:
        """Return `shardings`, with the memory kind each leaf's path asks for.

        Leaves that `host_parameters` selects go to pinned host memory. Each
        leaf keeps the spec the rules give it, so a selected parameter is the
        same shard in another memory space, and its layer issues the same
        collectives as when the parameter stays on the device.

        Every pattern must match at least one variable, or this raises
        ValueError. A pattern that matches nothing is a typo or a stale path.
        Without the error the run would keep those weights on the device, and
        the only sign would be the memory it did not save. So each pattern is
        checked on its own.
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
        """Refuse a placement that replicates too much of the model or ignores `host_parameters`.

        This is MaxText's guardrail (base.yml sharding_tolerance) against a
        mesh whose parameter axes divide none of the model's dimensions. On
        such a mesh the shape heuristic replicates every parameter, and nothing
        else reports it.

        MaxText measures the excess per-chip memory over perfect sharding
        across every parameter. This check takes the same ratio over the
        parameters of at least `min_shard` elements only. Smaller ones are
        replicated on purpose, so counting them would fail models that are
        merely small.

        Raises `LayoutRefused` when the replicated fraction is above
        `tolerance`, naming the five largest replicated parameters, and
        ValueError when `host_parameters` is set but every parameter in
        `shardings` stays on the device. A mesh whose parameter axes all have
        size 1 passes the replication check.
        """
        if self.host_parameters and not any(
                sharding.memory_kind == "pinned_host"
                for sharding in jax.tree.leaves(shardings)):
            raise ValueError(
                f"host_parameters {list(self.host_parameters)} keeps those weights "
                f"in pinned host memory, which only a stack that fetches a layer's "
                f"parameters as it reaches it reads; this placement keeps every "
                f"parameter on the device. Place the weights for generation with a "
                f"dew.inference.LayerBanks source's place, or use host=('variables',) for a "
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
        raise LayoutRefused(
            f"{fraction:.2%} of shardable parameter elements are replicated, over "
            f"the sharding tolerance of {self.tolerance:.2%}.\n"
            f"Largest replicated parameters:\n{details}")


def batch_shardings(mesh: Mesh | AbstractMesh, batch: Batch) -> Placement[Batch]:
    """Return a sharding for each leaf of `batch`, chosen from the leaf's shape.

    Rows split over the data, expert and fsdp axes. A leaf of rank 2 or 3 holds
    a sequence per row: token ids, segment ids, positions, encoded tokens. Its
    second dimension splits over the sequence axis when the axis divides it,
    and otherwise stays whole, the way `logical_spec` drops a name no dimension
    can split. An image or a video is not a sequence, so only its rows split.

    This only decides placement and never changes values. Until a model
    constrains its attention to the sequence axis, a sequence split here is
    gathered again inside the model. Only the shape is read, so a leaf that is
    already a global array costs no transfer.
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


def shard_batch(mesh: Mesh, batch: Batch) -> Batch:
    """Assemble this process's share of each array into a globally sharded one.

    The share is the one `DataPartition.of(mesh)` names: that share's rows, each
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
    count = DataPartition.of(mesh).count
    shardings = batch_shardings(mesh, batch)
    if jax.process_count() == 1:
        # The whole batch is this process's share, placed by one device_put
        # for the tree: 74 us of host time a batch on an RTX 4080, and 174 on
        # 8 CPU devices, where a call per leaf took 105 and 222.
        return jax.device_put(jax.tree.map(
            lambda leaf: leaf if isinstance(leaf, jax.Array) else np.asarray(leaf), batch), shardings)

    def place(leaf, sharding: NamedSharding) -> jax.Array:
        # The share holds whole rows: `count` shares make the rows, and every
        # other dimension is already whole, so a device of a sequence or a
        # stage that spans processes picks its own slice out of it.
        local = leaf if isinstance(leaf, jax.Array) else np.asarray(leaf)
        shape = np.shape(local)
        return jax.make_array_from_process_local_data(
            sharding, local, (shape[0] * count, *shape[1:]) if shape else ())

    return jax.tree.map(place, batch, shardings)


def first_reader_batch(mesh: Mesh, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The batch the first reader of this process's share read, on every reader of it.

    The processes that read one share (`DataPartition.of(mesh).readers`) hand
    `shard_batch` rows the same devices hold, so their batches must be the
    same. A source whose reads differ between them, as independent draws
    from an engine do, reads on the share's first reader alone
    (`DataPartition.reader` 0). Every process of the pool then calls this
    on the thread its other collectives run on, as a rollout runs: each
    first reader's batch, laid out as process 0's is, reaches every
    process, and each takes its share's. A later reader's `batch` is not
    read. On a mesh whose shares have one reader each this is `batch`.
    """
    partition = DataPartition.of(mesh)
    if partition.readers == 1:
        return dict(batch)
    layout = broadcast_from_process_zero(
        {name: [list(np.shape(leaf)), str(np.asarray(leaf).dtype)] for name, leaf in batch.items()})

    def held() -> dict[str, np.ndarray]:
        if partition.reader:
            return {name: np.zeros(shape, np.dtype(dtype)) for name, (shape, dtype) in layout.items()}
        mine = {name: np.asarray(leaf) for name, leaf in batch.items()}
        if {name: [list(leaf.shape), str(leaf.dtype)] for name, leaf in mine.items()} != layout:
            raise ValueError("the first readers' batches are not laid out alike; every process of a pool "
                             "hands shard_batch the same tree")
        return mine

    rows = agreed("first reader batch", held)
    gathered = multihost_utils.process_allgather(
        {"share": np.asarray([partition.index, partition.reader], np.int32), **rows})
    source = next(process for process, (index, reader) in enumerate(gathered["share"].tolist())
                  if index == partition.index and reader == 0)
    return {name: np.asarray(gathered[name][source]) for name in layout}


# The batches a `DevicePrefetchIterator` queues ahead of the step by default.
PREFETCH_DEPTH = 2


def prefetched_bytes(batch: Batch, shardings: Placement[Batch]) -> int:
    """Return the bytes a device holds of the batches `fit` places while a step runs.

    These are the batches next to the one the step reads: the `PREFETCH_DEPTH` it
    queues and the one it is placing, each laid out as `shardings` places `batch`.
    """
    shares = jax.tree.map(lambda leaf, sharding: math.prod(sharding.shard_shape(np.shape(leaf)))
                          * np.dtype(leaf.dtype).itemsize, batch, shardings)
    return (PREFETCH_DEPTH + 1) * sum(jax.tree.leaves(shares))


class DevicePrefetchIterator:
    """Reads batches on a worker thread and places them on the mesh ahead of the step.

    The iterator closes its source when the source runs out or the iterator is
    closed, so use it as a context manager or call `close`, even after a loop
    that ended early. At most `depth` batches are queued, plus the one being
    read and placed.

    Every call into the source runs on the worker thread: `next`, reading the
    checkpoint position, and the final close. The exception is the source's
    optional `request_stop` hook, which runs on the thread that closes the
    iterator, so it must be thread-safe and must not block.

    `source_state`, when given, is a saved position the worker restores before
    its first read; if the source is not checkpointable, the first `next`
    raises TypeError. After each batch the iterator returns, `source_state`
    holds the source's position after that batch.

    If the worker thread cannot be created, the source is finalized on the
    caller's thread before the error is re-raised; no worker has touched it at
    that point. That path runs the source's close synchronously, so the close
    has to return on its own. When a running worker is stuck in source code
    that cannot be stopped, `close` raises TimeoutError.
    """

    def __init__(self, iterator: Iterator, mesh: Mesh, depth: int = PREFETCH_DEPTH,
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
        self._withdraw_exit: Callable[[], None] | None = None
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
                # A worker still placing a batch when Python finalizes would abort the process.
                self._withdraw_exit = stop_at_exit(self._thread, self._cancel, timeout=5.0)

    def _cancel(self) -> None:
        """Ask the worker to stop between batches, and a stoppable source to stop its own work."""
        first = not self._stop.is_set()
        self._stop.set()
        if first and isinstance(self._iterator, Stoppable):
            self._iterator.request_stop()

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
        """Stop the worker and wait for it, discarding unread batches but keeping `source_state`.

        The wait lasts `timeout` seconds, else the source's own `stop_seconds`
        (a grain pipeline joins one worker process after another), else 5 s. If
        the worker is still alive after that, this raises TimeoutError.
        Cancellation stays requested and a later close may wait again, but the
        source's in-flight work and buffers may not have been released. An
        error from the source's own close is raised here too.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("a prefetch worker cannot close itself")
        if timeout is None:
            seconds = self._iterator.stop_seconds if isinstance(self._iterator, Budgeted) else None
            timeout = 5.0 if seconds is None else float(seconds)
        error = None
        try:
            self._cancel()
        except BaseException as failure:
            error = failure
        self._discard()
        self._start()  # Even an unused iterator finalizes on its worker.
        if self._thread.ident is not None:
            self._thread.join(timeout)
        self._discard()
        self._error = None  # Unconsumed speculative errors are discarded too.
        if not self._thread.is_alive() and self._withdraw_exit is not None:
            self._withdraw_exit()
            self._withdraw_exit = None
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


__all__ = ["DevicePrefetchIterator", "Layout", "MeshSpec", "batch_shardings"]
