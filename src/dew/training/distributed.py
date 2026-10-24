"""Device mesh, parameter layout, and host-to-device prefetch."""

from __future__ import annotations

import collections
import dataclasses
import fnmatch
import functools
import hashlib
import json
import logging
import math
import statistics
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Concatenate, Protocol, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P

from dew.coordination import agreed, broadcast_from_process_zero, from_every_process, stop_at_exit
from dew.data.dataset import (
    Budgeted,
    Checkpointable,
    Closeable,
    DataPartition,
    RampedStream,
    Stoppable,
    rows_of,
    stream_identity,
)
from dew.files import write_atomically
from dew.nn.inputs import filled_validity
from dew.nn.sharding import (
    DATA_AXIS,
    DEFAULT_RULES,
    EXPERT_AXIS,
    FSDP_AXIS,
    MESH_AXES,
    SEQUENCE_AXIS,
    STAGE_AXIS,
    LayoutRefused,
    LogicalAxes,
    LogicalAxisRules,
    MeshAxes,
    Suffix,
    batch_axes,
    boxed,
    boxed_axes,
    current_boxes,
    declarations,
    declared_axes,
    logical_spec,
    mesh_axes,
    microbatched,
    ruled_boxes,
)
from dew.objectives.base import EXAMPLE_ROWS, FROZEN, Batch, Initializer, Variables
from dew.records import JSON, number, record
from dew.telemetry.benchmark import samples
from dew.telemetry.profile import region
from dew.training.state import VARIABLE_FIELDS, TrainState, apart, together

# The role axes no parameter splits over: data holds a batch's replicas, and
# stage the pipeline's stages of the layer stack, which the decoder places
# itself. `Layout` refuses a parameter placed there. Every other axis a mesh
# has, Dew's own (expert, fsdp, tensor, sequence) or a native mesh's, holds a
# parameter where the rules put one.
UNSPLIT_AXES = (DATA_AXIS, STAGE_AXIS)

STATE_AXES = (FSDP_AXIS, SEQUENCE_AXIS)
"""The axes whose devices compute their own share of one batch with the same
weights, fsdp's rows and sequence's positions, and so each hold one share of
the persistent state, as fully sharded data parallelism defines it. What the
rules place on fsdp, `Layout` splits over both where together they divide it
(`state_spec`): a sequence-parallel run then holds its parameters, optimizer
moments, EMA and gradients in shares rather than whole on every device of a
sequence, which torchtitan's FSDP mesh does over dp_shard and cp alike
(`resolve_fsdp_mesh`). Leaves below `Layout.min_shard`, or that the pair
cannot divide, keep their own placement. A step reads a weight split so as
it reads an fsdp split."""


def parameter_axes(names) -> tuple[str, ...]:
    """The mesh axes among `names` (a mesh's `axis_names`) a parameter can
    split over: all but `UNSPLIT_AXES`."""
    return tuple(axis for axis in names if axis not in UNSPLIT_AXES)


def state_spec(spec_on: Callable[[Mesh | AbstractMesh], P], mesh: Mesh | AbstractMesh) -> P:
    """The spec of a leaf of the persistent state on `mesh`, from `spec_on`, its spec on a mesh.

    What `spec_on` places on fsdp splits over fsdp and sequence together
    (`STATE_AXES`): the spec is read on `mesh` with the two as one fsdp axis
    of both sizes, so the rules' divisibility and fallbacks judge the
    group, and each fsdp entry then names both. Where the group divides no
    dimension fsdp would take, the group's spec moves a split of another axis
    (a vocabulary the group times tensor cannot divide), the rules place the
    leaf on sequence already or sequence is Explicit, whose placement the
    program's types carry and the partitioner does not choose, the spec is
    `spec_on`'s on `mesh`."""
    plain = spec_on(mesh)
    sequence = mesh.shape.get(SEQUENCE_AXIS, 1)
    if (sequence == 1 or SEQUENCE_AXIS in _spec_axes(plain)
            or SEQUENCE_AXIS in explicit_axes(mesh)):
        return plain
    fsdp = mesh.shape.get(FSDP_AXIS, 1)
    sizes = {**mesh.shape, SEQUENCE_AXIS: 1, FSDP_AXIS: fsdp * sequence}
    added = () if FSDP_AXIS in mesh.axis_names else (FSDP_AXIS,)
    names = (*mesh.axis_names, *added)
    abstract = mesh.abstract_mesh if isinstance(mesh, Mesh) else mesh
    grouped = spec_on(abstract.update(axis_sizes=tuple(sizes[name] for name in names), axis_names=names,
                                      axis_types=(*mesh.axis_types, *(AxisType.Auto for _ in added))))
    def others(spec: P) -> list[MeshAxes]:
        entries = [_without_state_axes(entry) for entry in spec]
        return entries + [None] * (len(grouped) + len(plain) - len(entries))

    if FSDP_AXIS not in _spec_axes(grouped) or others(grouped) != others(plain):
        return plain
    group = (FSDP_AXIS,) * (fsdp > 1) + (SEQUENCE_AXIS,)

    def held(entry: MeshAxes) -> MeshAxes:
        axes = tuple(name for axis in mesh_axes(entry) for name in (group if axis == FSDP_AXIS else (axis,)))
        return None if not axes else axes[0] if len(axes) == 1 else axes

    return P(*map(held, grouped))


def _spec_axes(spec: P) -> set[str]:
    """Every mesh axis `spec` names."""
    return {axis for entry in spec for axis in mesh_axes(entry)}


def device_mesh(mesh: MeshSpec | Mesh) -> Mesh:
    """The device mesh `mesh` names: a native `jax.sharding.Mesh` as it is,
    whatever its axes, or the one a `MeshSpec` builds."""
    return mesh if isinstance(mesh, Mesh) else mesh.build()


def stage_microbatches(mesh: Mesh, asked: int | None) -> int:
    """The microbatches a pipelined step feeds through `mesh`'s stage axis: as
    `asked`, else one a stage. A mesh without a stage axis has one stage."""
    stages = mesh.shape.get(STAGE_AXIS, 1)
    if asked is None or asked == stages:
        return stages
    if stages == 1:
        raise ValueError(f"microbatches ({asked}) feed the stage axis, which this mesh does not split; "
                         f"give it stages or leave microbatches unset")
    if asked < stages or asked % stages:
        raise ValueError(f"microbatches must be a positive multiple of the stage axis's {stages}, "
                         f"got {asked}")
    return asked


class Ruled(Protocol):
    layout: Layout


def ruled[Owner: Ruled, **Arguments, Result](
        method: Callable[Concatenate[Owner, Arguments], Result]
) -> Callable[Concatenate[Owner, Arguments], Result]:
    """`method`, run under its owner's layout's rules: they place a batch's
    rows (`batch_axes`) and a run's data shares as they place the activations
    its trace constrains."""
    @functools.wraps(method)
    def run(owner: Owner, *args: Arguments.args, **kwargs: Arguments.kwargs) -> Result:
        with nn.logical_axis_rules(owner.layout.axis_rules):
            return method(owner, *args, **kwargs)
    return run


def explicit_axes(mesh: MeshSpec | Mesh | AbstractMesh) -> tuple[str, ...]:
    """The axes of `mesh` whose sharding the program's types carry (`AxisType.Explicit`)."""
    if isinstance(mesh, MeshSpec):
        return tuple(mesh.explicit)
    return tuple(axis for axis, kind in zip(mesh.axis_names, mesh.axis_types, strict=True)
                 if kind == AxisType.Explicit)

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
    """Devices the batch's sequence dimension is split over; 1 keeps whole
    sequences. The persistent state splits over them with fsdp (`STATE_AXES`)."""
    stage: int = 1
    """Pipeline stages the layer stack is split into, each on its own devices.
    1 runs the whole stack on every device. How many microbatches a step
    feeds through them is the trainer's (`Trainer.microbatches`)."""
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
    128 MiB.

    A pool keeps its figure beside the persistent compilation cache, by the
    axis and the devices it spans, and a later pool on them takes it, a
    resume among them: the decisions from it shape the step, and a figure
    measured again moves by noise, across a threshold one time in some."""
    topology = link_topology(mesh, axis)
    record = _link_record(topology)
    # Every process gathers, or none does: a pool measures unless each of its processes sees the figure,
    # kept for exactly this placement.
    if record is not None and all(from_every_process(
            record.exists() and json.loads(record.read_text()).get("topology") == topology)):
        return min(from_every_process(float(json.loads(record.read_text())["bytes_per_second"])))
    ways = mesh.shape[axis]
    count = size // 4 // ways * ways
    sharding = NamedSharding(mesh, P(axis))
    source = jax.make_array_from_callback(
        (count,), sharding, lambda index: np.zeros(sharding.shard_shape((count,)), np.float32))
    gather = jax.jit(jax.shard_map(
        lambda shard: jax.lax.all_gather(shard, axis, tiled=True), mesh=mesh,
        in_specs=P(axis), out_specs=P(), axis_names={axis}, check_vma=False))
    seconds = samples(gather, source, warmup=2, repeats=5)
    received = min(from_every_process((ways - 1) / ways * count * 4 / statistics.median(seconds)))
    if record is not None and jax.process_index() == 0:
        record.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(record, json.dumps({"topology": topology, "bytes_per_second": received}))
    return received


def link_topology(mesh: Mesh, axis: str) -> str:
    """What a figure of `mesh`'s `axis` is a figure of: the axis, the mesh's axes and their widths in
    order, and its devices in the mesh's order, each by id, process, kind and slice. A device's id is
    its place in its system, so two meshes of one shape over other devices, or over the same devices
    grouped otherwise along the axis, are other links; one built again over the same placement is the
    same."""
    devices = [[device.id, device.process_index, device.device_kind, _slice(device)]
               for device in mesh.devices.flat]
    return json.dumps([axis, [[name, width] for name, width in mesh.shape.items()], devices])


def link_figures(bandwidths: Mapping[tuple[Mesh, str], float | None]) -> dict[str, JSON]:
    """The link figures a checkpoint's rung keeps (`Trainer._checkpoint_rung`): each measured figure of
    `bandwidths`, by mesh and axis, under the placement it was measured on (`link_topology`)."""
    return {link_topology(mesh, axis): figure for (mesh, axis), figure in bandwidths.items()
            if figure is not None}


def resumed_link_figures(rung: JSON) -> dict[str, float]:
    """`link_figures` as a checkpoint's rung recorded them; none where it recorded none."""
    links = record(record(rung, 'rung').get('links', {}), 'rung links')
    return {topology: number(figure, f'link of {topology}') for topology, figure in links.items()}


def _link_record(topology: str) -> Path | None:
    """Where `link_bandwidth` keeps the figure of `topology` (`link_topology`): beside the persistent
    compilation cache, or nowhere when the process keeps no cache."""
    cache = jax.config.jax_compilation_cache_dir
    if not cache:
        return None
    return Path(cache) / "dew-links" / f"{hashlib.sha256(topology.encode()).hexdigest()}.json"


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


def batch_divisor(mesh: Mesh | AbstractMesh, microbatches: int) -> int:
    """Return the row count every global batch must be a multiple of.

    A batch's rows are sharded over the row axes (`batch_axes`) as whole
    rows, and a pipelined step cuts each device's rows again into
    `microbatches`: microbatch m takes rows m, m + M, ..., which is a share
    of every device's rows only where M divides them. A batch that divides
    by neither is refused where it is placed or traced.

    `Trainer.fit` checks a batch against this before it places anything,
    and a batch ramp's every stage, since a later stage would otherwise fail
    an hour in.
    """
    return math.prod(mesh.shape[axis] for axis in batch_axes(mesh)) * microbatches


def check_batch(batch: int, mesh: Mesh, microbatches: int) -> None:
    """Refuse, before anything is placed, a global batch `mesh` cannot split
    (`batch_divisor`): its rows shard over the row axes as whole rows, and a
    pipeline cuts each device's rows again into `microbatches`, where a
    device that holds none of a microbatch's rows computes another's again.
    The message names the row shards, and the batches and the microbatch
    counts that fit."""
    divisor = batch_divisor(mesh, microbatches)
    if batch % divisor == 0:
        return
    count, stages = microbatches, mesh.shape.get(STAGE_AXIS, 1)
    shards = divisor // count
    suggestions = [f"a batch that is a multiple of {divisor} rows, {-(-batch // divisor) * divisor} "
                   f"the nearest above {batch}"]
    rows = batch_axes(mesh)
    axes = " x ".join(f"{axis} {mesh.shape[axis]}" for axis in rows)
    if count == 1:
        raise LayoutRefused(
            f"a global batch of {batch} rows over the {shards} row shards of {axes} leaves some "
            f"device without whole rows; use {suggestions[0]}, or a mesh whose row axes "
            f"({' x '.join(rows)}) divide {batch}")
    if batch % shards == 0:
        fitting = [m for m in range(stages, batch // shards + 1, stages)
                   if (batch // shards) % m == 0]
        if fitting:
            suggestions.append(f"microbatches={fitting[-1]}, the most that divide the "
                               f"{batch // shards} rows a device holds")
    raise LayoutRefused(
        f"a global batch of {batch} rows over {shards} row shards and {count} microbatches "
        f"leaves some microbatch without rows on some device, which then computes another "
        f"microbatch's again; use {' or '.join(suggestions)}")


def check_stream(source, mesh: Mesh, microbatches: int, batch: int, *, rows: bool,
                 resumed: int | None) -> None:
    """Refuse a training stream a run cannot resume or cannot shard.

    `resumed` is the step a resume restored from a checkpoint without a data
    position, which no stream can continue: with no position to restore, one
    that keeps positions (`Checkpointable`) and one declared to replay alike
    would read its data again from the start, so past step 0 it is refused.
    A `batch` the mesh cannot divide fails where it is placed or
    traced (`check_batch`), unless the stream's rows are not the step's
    (`rows`; a rollout's are its own, checked where its step traces), and a
    ramp stage's, for a later stage, an hour in. A ramp reads each stage's
    batch whole; an accumulation window pools its stages' batches as it pools
    any, but for a loss that replays its window's batches (`Trainer._prepared`).
    """
    if resumed:
        kept = isinstance(source, Checkpointable)
        why = ("it was saved without one (Checkpoints.save's `saved`)" if kept
               else f"{stream_identity(source)} keeps none")
        remedy = ("save the stream's position with the state, as Trainer.fit does" if kept else
                  "give the stream get_state and set_state, read it through dew.data, or declare that "
                  "it reads the same batches each time it opens (dew.data.dataset.replayed, "
                  "from_iter(replay=True))")
        raise ValueError(f"step {resumed}'s checkpoint holds no data position, as {why}, so resuming "
                         f"would read its data again from the start; {remedy}")
    if not isinstance(source, RampedStream):
        if rows:
            check_batch(batch, mesh, microbatches)
        return
    divisor = batch_divisor(mesh, microbatches)
    refused = [stage.batch for stage in source.stages if stage.batch % divisor]
    if refused:
        raise ValueError(
            f"the batch ramp reads {refused} records a step at some of "
            f"its stages, and {dict(mesh.shape)} with "
            f"{microbatches} microbatch(es) "
            f"holds a batch that is a multiple of {divisor}")


def microbatch(batch: Batch, part: jax.Array, split: int) -> Batch:
    """The `part`th of the `split` microbatches a step reads `batch` in
    (`Trainer.compile`): every `split`th row from row `part` (`microbatched`),
    a share of each device's rows where `split` divides them, so no row
    crosses devices. A field with no rows is the whole step's, and every
    microbatch reads it whole; one microbatch is the batch as it came,
    whatever its fields' rows (a rollout's groups)."""
    if split == 1:
        return batch
    return jax.tree.map(lambda leaf: jax.lax.dynamic_index_in_dim(
        microbatched(leaf, 0, split), part, 0, keepdims=False) if np.ndim(leaf) else leaf, batch)


def microbatch_shapes(batch: Batch, split: int) -> Batch:
    """The fields of `microbatch`'s microbatches, as a trace reads them: the
    batch itself for one, whose placed fields' types carry their Explicit
    axes, else their shapes."""
    if split == 1:
        return batch
    part = jax.ShapeDtypeStruct((), np.dtype(np.int32))
    return jax.eval_shape(functools.partial(microbatch, split=split), batch, part)


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
    return jax.tree_util.keystr(path, simple=True, separator="/")


def _within_variables(path) -> tuple | None:
    """`path` inside the variables tree its leaf belongs to: as it is in a
    variables tree, past the field in a train state's (`VARIABLE_FIELDS`), and
    None in another field of one, which mirrors the parameters (the optimizer's
    state, the accumulation) or holds none (the counters, the key)."""
    if path and isinstance(path[0], jax.tree_util.GetAttrKey):
        return path[1:] if path[0].name in VARIABLE_FIELDS else None
    return path


def host_selected(patterns: tuple[str, ...], path: str) -> bool:
    """Whether `path` is one of the variables `patterns` names.

    A pattern is an `fnmatch` glob over the path, and a pattern that names a
    subtree covers it whole, so `params/layers_3` selects that layer's every
    leaf and `params/layers_*` the stack's.
    """
    return any(fnmatch.fnmatchcase(path, pattern)
               or fnmatch.fnmatchcase(path, f"{pattern}/*") for pattern in patterns)


@dataclasses.dataclass(frozen=True)
class LeafPlacement:
    """Where one leaf of a placed state lives (`Validation.leaves`)."""
    spec: P
    """The mesh axes each of its dimensions splits over."""
    shard_shape: tuple[int, ...]
    """The part of it each device holds."""
    memory_kind: str
    """The memory that part sits in: `device`, or the host memory a layout keeps it in
    (`pinned_host`), which takes none of the device's."""


@dataclasses.dataclass(frozen=True)
class Validation:
    """What a placed state holds on a mesh before the first step (`Layout.validate`)."""
    leaves: dict[str, LeafPlacement]
    """Each leaf's placement by its path (`variables/params/...`), in the state's order."""
    device_bytes: dict[int, int]
    """The bytes each device of the mesh holds of the state in its own memory between steps,
    by the device's id."""
    capacity: dict[int, int]
    """The bytes each device has, by its id, as the caller stated them or the device reports
    them; a device that reports none, such as a CPU, is absent."""
    batch_axes: tuple[str, ...]
    """The mesh axes a batch's rows split over (`batch_axes`)."""
    collectives: dict[str, tuple[str, ...]]
    """The collectives a training step runs over each axis of the mesh above size one, by the
    role the placement gives the axis, in HLO's names (`_implied_collectives`)."""
    unbound: dict[str, tuple[str, ...]]
    """Each logical axis the variables declare that no rule names, with the leaves that
    declare it. A leaf keeps the dimensions it names whole on every device where a rule
    places another of its names, and takes the shape heuristic where none does (`shardings`)."""


@dataclasses.dataclass(frozen=True)
class Layout:
    """Decides how a train state is placed on a mesh.

    `rules` map the logical axis names that modules declare (`dew.nn.sharding`)
    onto mesh axes, in precedence order. They place the parameters, and the
    trainer sets them in context so they also place the activations a compiled
    step constrains. An axis of size 1 splits nothing, so the same table serves
    every topology.

    What the rules place on fsdp also splits over the sequence axis, whose
    devices compute their own positions with the same weights (`STATE_AXES`).
    `shardings` refuses rules that place a parameter on the data or stage
    axis. The data axis holds the batch's replicas, each with the whole
    state. The stage axis holds the layer stack's pipeline stages, and the
    decoder places those itself from the stored tree.

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
    needs at least as many CPU devices as accelerators, set before JAX
    initializes; neither this layout nor the trainer changes the count, and a
    process's surplus CPUs, and the processes the mesh leaves out, stay out of
    the companion (`companion_mesh`).

    `host_parameters` lists the variables an inference placement keeps in
    pinned host memory, as globs over their logical paths (`params/layers_*`).
    A selected leaf keeps the spec the rules give it, and only its memory kind
    changes. Only `offloaded` reads the patterns, because only the layer stack
    fetches each layer's parameters as it reaches that layer. When
    `host_parameters` is set, `check` refuses a placement that keeps every
    parameter on the device, so the patterns are never silently ignored.

    `frozen_whole` keeps the frozen collection (`FROZEN`, a LoRA run's base)
    whole over fsdp and sequence (`STATE_AXES`), split only over the other
    axes the rules give it (a tensor split stays). A frozen leaf has no
    gradient and no optimizer state, so a split over them saves only its
    memory, and the step gathers it whole forward and again backward.
    `check` counts no leaf it keeps whole, and a leaf `host_parameters` also
    names is kept so in pinned host memory.
    """
    rules: LogicalAxisRules | Mapping[str, MeshAxes] = DEFAULT_RULES
    min_shard: int = 2 ** 16
    tolerance: float = 0.02
    host: tuple[str, ...] = ()
    host_parameters: tuple[str, ...] = ()
    frozen_whole: bool = False
    data_axes: tuple[str, ...] = ()
    """The axes a batch's rows split over, ahead of the rules' own
    `activation_batch` entries: a native mesh's names for its data-parallel
    axes, which the run's data shares, its batch placement and its row
    counts read (`dew.nn.sharding.batch_axes`). Empty leaves the rules as
    they are."""

    def __post_init__(self):
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(
                f"sharding tolerance must be between 0 and 1, got {self.tolerance}")
        object.__setattr__(self, "rules", _rule_table(self.rules))
        object.__setattr__(self, "data_axes", tuple(self.data_axes))
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
        """The rules as the (name, axes) pairs flax reads, in precedence order,
        `data_axes` first."""
        rules = self.rules
        assert isinstance(rules, tuple), "__post_init__ keeps the rules as pairs"
        return ((("activation_batch", self.data_axes),) if self.data_axes else ()) + rules

    def _refuse_unknown(self, mesh: Mesh | AbstractMesh) -> None:
        """Refuse rules naming an axis `mesh` lacks. Dew's own rules (`DEFAULT_RULES`) name its role
        axes (`MESH_AXES`) for any mesh, and one without such an axis takes it as size 1; a rule of
        your own names an axis you mean to split over, and one the mesh lacks would split nothing."""
        defaults = set(_rule_table(DEFAULT_RULES))
        named = {axis for rule in self.axis_rules for axis in mesh_axes(rule[1])
                 if rule not in defaults or axis not in MESH_AXES}
        unknown = sorted(named - set(mesh.axis_names))
        if unknown:
            raise ValueError(f"the rules name the axes {unknown}, which this mesh, with axes "
                             f"{list(mesh.axis_names)}, does not have")

    def shardings[TreeT](self, mesh: Mesh, tree: TreeT) -> Placement[TreeT]:
        """Return a NamedSharding for each leaf of `tree`, from the axes its module declared.

        A parameter its module boxed (`nn.with_logical_partitioning`) takes the
        axes it carries, in `tree` or in the trainer's `boxed` table, when the
        rules place one of them (`ruled_boxes`). A leaf
        whose path no module declares falls back to a shape heuristic,
        which splits its largest dimension that fsdp divides evenly over fsdp.
        So the axes of one model family can be declared at a time. Either
        way, what lands on fsdp splits over the sequence axis too where the
        two divide it (`state_spec`). A leaf below `min_shard` elements is
        replicated either way.

        Flax metadata that a caller's own module attached is removed, because
        the state the trainer creates from these shardings holds plain arrays.
        Raises ValueError when the rules place a leaf on the data or stage
        axis (`UNSPLIT_AXES`) or on an Explicit axis, or name an axis
        the mesh lacks.
        """
        self._refuse_unknown(mesh)
        sharded_devices = math.prod(mesh.shape[axis] for axis in parameter_axes(mesh.axis_names))
        explicit = set(explicit_axes(mesh))
        declared = declarations(tree)

        def leaf_sharding(path, value):
            axes = self._axes(path, value.ndim, declared)
            size = int(np.prod(value.shape, dtype=np.int64))
            if axes is None:
                spec = state_spec(lambda placed: parameter_spec(
                    value.shape, placed.shape.get(FSDP_AXIS, 1), self.min_shard), mesh)
            elif sharded_devices == 1 or size < self.min_shard:
                spec = P()
            else:
                spec = state_spec(lambda placed: logical_spec(axes, value.shape, rules=self.axis_rules,
                                                              mesh=placed), mesh)
            if self._kept_whole(path):
                spec = P(*map(_without_state_axes, spec))
            placed = _spec_axes(spec)
            if outside := sorted(placed & set(UNSPLIT_AXES)):
                raise ValueError(
                    f"the rules place {_variable_path(path)} on {outside}; the data axis "
                    f"holds the batch's replicas, and the stage axis the pipeline")
            if typed := sorted(placed & explicit):
                raise ValueError(
                    f"the rules place {_variable_path(path)} on the Explicit axes {typed}; the "
                    f"partitioner places parameters, over Auto axes (jax.make_mesh's axis_types)")
            return NamedSharding(mesh, spec)

        with self._declared(tree):
            return jax.tree_util.tree_map_with_path(leaf_sharding, nn.unbox(tree))

    def _declared(self, tree):
        """The context in which `declared_axes` reads the axes `tree`'s
        modules declared, as the rules place them."""
        return boxed(ruled_boxes({**current_boxes(), **boxed_axes(tree)}, self.axis_rules))

    def _axes(self, path, ndim: int, declared: Mapping[Suffix, LogicalAxes]) -> LogicalAxes | None:
        """The logical axes of the leaf at `path`: those its module declared, or None for the shape
        heuristic. Placement, its minimum size, `frozen_whole` and its checks follow from them. A layout
        that names axes otherwise overrides it (torchax's `TorchLayout`, by a module's torch names)."""
        return declared_axes(path, ndim, declared)

    def _splitting(self, path, value, declared: Mapping[Suffix, LogicalAxes]) -> set[str]:
        """The parameter axes the rules could split the leaf at `path` over:
        those its declared names' rules name, or fsdp, which the shape
        heuristic splits an undeclared leaf over."""
        axes = self._axes(path, value.ndim, declared)
        if axes is None:
            return {FSDP_AXIS}
        return set(parameter_axes(axis for name, assignment in self.axis_rules if name in axes
                                  for axis in mesh_axes(assignment)))

    def _kept_whole(self, path) -> bool:
        """Whether a leaf sits in the frozen collection `frozen_whole` keeps
        whole: `FROZEN` heads a variables tree, the tree placed or one of a
        train state's (`VARIABLE_FIELDS`). The optimizer's state mirrors the
        trained parameters' paths, so a module or a group named so is none."""
        within = _within_variables(path)
        return self.frozen_whole and within is not None and within[:1] == (jax.tree_util.DictKey(FROZEN),)

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
        parameters of at least `min_shard` elements whose rules name a
        parameter axis this mesh splits. Smaller ones are replicated on
        purpose, so counting them would fail models that are merely small;
        one no rule splits over this mesh's axes is replicated by the mesh's
        choice, as a data axis replicates every parameter, so an expert-only
        mesh keeps a mixture's dense weights whole as data parallelism does.
        The sequence axis splits only where it divides what the rules place
        on fsdp (`state_spec`), so a leaf it cannot split is the rules' to
        count, and a mesh of sequence alone refuses none.

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
        if all(mesh.shape[axis] == 1 for axis in parameter_axes(mesh.axis_names)):
            return

        path_leaves, _ = jax.tree_util.tree_flatten_with_path(nn.unbox(params))
        splits = {axis for axis in parameter_axes(mesh.axis_names) if mesh.shape[axis] > 1}
        declared = declarations(params)
        with self._declared(params):
            splittable = [bool(self._splitting(path, param, declared) & splits)
                          for path, param in path_leaves]
        shardable_elements = 0
        replicated = []
        for (path, param), sharding, has_rule in zip(
                path_leaves, jax.tree.leaves(shardings), splittable, strict=True):
            elements = int(np.prod(param.shape, dtype=np.int64))
            if elements < self.min_shard or not has_rule or self._kept_whole(path):
                continue
            shardable_elements += elements
            if any(axis in mesh_axes(assignment)
                   for assignment in sharding.spec for axis in parameter_axes(mesh.axis_names)):
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

    def validate(self, mesh: Mesh, state, *, capacity: int | None = None) -> Validation:
        """Report what `state` holds on `mesh` before the first step, and refuse a state a
        device cannot hold.

        MaxText prints each parameter's logical and physical sharding before it trains
        (`print_shardings_params`), and Megatron reports the weight and optimizer bytes a GPU
        holds from a formula over the model's config (`report_theoretical_memory`). Here both
        are read off the placement itself, so they hold for any rules, mesh and optimizer.

        `state` is placed, or is about to be: each leaf an array, or a `jax.ShapeDtypeStruct`
        carrying the `NamedSharding` it is placed with (`shardings`, `Trainer.shardings`), so
        nothing is allocated. The report (`Validation`) gives each leaf's split and the part each
        device holds, the bytes each device of `mesh` keeps between steps, the axes a batch's rows
        split over, the collectives a step runs over each axis (`Validation.collectives`), and the
        logical axes the variables declare that no rule names (`Validation.unbound`). A leaf in
        host memory (`host`), or on devices outside `mesh` (a CPU-owned state's companion), takes
        none of a device's memory. Rules naming an axis `mesh` lacks raise ValueError here, as
        `shardings` refuses them.

        `capacity` is the bytes each device of `mesh` has. Unset, each of this process's devices
        gives its allocator's limit (`Device.memory_stats`' `bytes_limit`) where JAX reports one,
        which is what an allocation fails against, under whatever memory fraction the process
        set. A CPU device reports none, so a CPU mesh standing in for accelerators takes theirs
        from the caller.

        Raises `LayoutRefused`, naming the device and what it holds by field, when a device
        would hold more than its capacity between steps: the state cannot be placed there,
        before any of the step's temporaries, which the plan checks against the compiled step
        (`dew.training.planning`).
        """
        self._refuse_unknown(mesh)
        leaves: dict[str, LeafPlacement] = {}
        devices = set(mesh.devices.flat)
        # The ids of `mesh`'s devices each leaf's mesh shares, matched as devices, not ids: a CPU
        # companion's devices number themselves as the accelerators do.
        shared: dict[Mesh, frozenset[int]] = {}
        # What the devices' own memory holds, leaf by leaf: its field, its devices, a device's bytes of it.
        held: list[tuple[str, frozenset[int], int]] = []
        # The axes of `mesh` that split some variable.
        split: set[str] = set()
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
            sharding = leaf.sharding
            if sharding.mesh == mesh and _within_variables(path) is not None:
                split |= _spec_axes(sharding.spec)
            # A sharding that names no memory places in its devices' default, their own.
            kind = sharding.memory_kind or "device"
            shard = sharding.shard_shape(leaf.shape)
            leaves[_variable_path(path)] = LeafPlacement(sharding.spec, shard, kind)
            if kind == "device":
                on = sharding.mesh
                if on not in shared:
                    shared[on] = frozenset(device.id for device in devices.intersection(on.devices.flat))
                held.append((_variable_path(path[:1]), shared[on], _shard_bytes(leaf)))
        totals = collections.Counter()
        for _, ids, size in held:
            totals[ids] += size
        device_bytes = dict.fromkeys(mesh.device_ids.ravel().tolist(), 0)
        for ids, size in totals.items():
            for device in ids:
                device_bytes[device] += size
        limits = (dict.fromkeys(device_bytes, capacity) if capacity is not None else
                  {device.id: limit for device in mesh.local_devices
                   if (limit := (device.memory_stats() or {}).get("bytes_limit")) is not None})
        _refuse_overfull(held, device_bytes, limits)
        with nn.logical_axis_rules(self.axis_rules):
            rows = batch_axes(mesh)
        return Validation(leaves, device_bytes, limits, rows, _implied_collectives(mesh, rows, split),
                          self._unbound(state))

    def _unbound(self, state) -> dict[str, tuple[str, ...]]:
        """The logical axes the variables of `state` declare that no rule names, each with
        the leaves that declare it, read from every declaration: a box none of whose names
        a rule places too, which `shardings` leaves to the shape heuristic (`ruled_boxes`)."""
        named = {name for name, _ in self.axis_rules}
        found: dict[str, list[str]] = collections.defaultdict(list)
        declared = declarations(state)
        with boxed({**current_boxes(), **boxed_axes(state)}):
            for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
                if _within_variables(path) is None:
                    continue
                for name in self._axes(path, leaf.ndim, declared) or ():
                    if name is not None and name not in named:
                        found[name].append(_variable_path(path))
        return {name: tuple(paths) for name, paths in found.items()}


def _implied_collectives(mesh: Mesh, rows: tuple[str, ...], split: set[str]) -> dict[str, tuple[str, ...]]:
    """The collectives a training step runs over each axis of `mesh` above size one, by the role a
    placement gives the axis, in HLO's names (`dew.telemetry.instrumentation.COLLECTIVES`).

    An axis whose devices each compute their own share of a batch, its rows (`rows`) or the
    sequence axis's positions, sums the step's loss and the gradients of what it holds whole
    (all-reduce). Where it also splits a variable (`split`), it gathers the variable whole to use
    it (all-gather), and a row axis sums each gradient into its shares (reduce-scatter), as fully
    sharded data parallelism does. An axis that splits a variable and no batch, tensor, sums the
    partial products of the width it splits (all-reduce). The stage axis hands each microbatch's
    activations on (collective-permute). How the experts on the expert axis meet their tokens, an
    all-to-all or a gather by the layer's dispatch (`dew.nn.moe`), and the attention's exchange
    over the sequence axis, an all-to-all or a gather by its heads and lengths
    (`dew.nn.attention.sequence_parallel_attention`), are the model's choice, not the
    placement's, and are not named."""
    implied = {}
    for axis, size in mesh.shape.items():
        if size == 1:
            continue
        shares = axis in rows or axis == SEQUENCE_AXIS
        ops = ["all-reduce"] if shares or axis in split else []
        if shares and axis in split and axis != EXPERT_AXIS:
            ops.append("all-gather")
            if axis in rows:
                ops.append("reduce-scatter")
        if axis == STAGE_AXIS:
            ops.append("collective-permute")
        if ops:
            implied[axis] = tuple(ops)
    return implied


def _refuse_overfull(held: list[tuple[str, frozenset[int], int]], device_bytes: Mapping[int, int],
                     limits: Mapping[int, int]) -> None:
    """Refuse a state some device would hold more of than its `limits`, naming the device
    furthest over and its bytes by field, from `held` (`Layout.validate`); both by device id."""
    over = [device for device, limit in limits.items() if device_bytes[device] > limit]
    if not over:
        return
    fullest = max(over, key=lambda device: device_bytes[device] - limits[device])
    fields = collections.Counter()
    for field, ids, size in held:
        if fullest in ids:
            fields[field] += size
    parts = ", ".join(f"{field} {size / 2 ** 30:.2f}" for field, size in fields.most_common())
    raise LayoutRefused(
        f"device {fullest} would hold {device_bytes[fullest] / 2 ** 30:.2f} GiB of the state between steps "
        f"({parts} GiB), over the {limits[fullest] / 2 ** 30:.2f} GiB it has. Split the state over more "
        f"devices (a larger fsdp axis), keep the optimizer state and the EMA in pinned host memory "
        f"(Layout(host=('opt_state', 'ema'))), or keep Adam's moments in bfloat16 "
        f"(OptimConfig.state_dtype)")


def placeable(field) -> bool:
    """Whether a device can hold every leaf of `field`: arrays of numbers,
    not strings, bytes or Python objects, such as a caption or an image's
    catalog URL."""
    return all(isinstance(leaf, jax.Array) or (
        not isinstance(leaf, (str, bytes)) and np.asarray(leaf).dtype.kind not in "OSUV")
        for leaf in jax.tree.leaves(field))


@overload
def split_fields(batch: Batch) -> tuple[Batch, dict]: ...
@overload
def split_fields(batch: np.ndarray | jax.Array) -> tuple[np.ndarray | jax.Array, dict]: ...
def split_fields(batch: Batch | np.ndarray | jax.Array) -> tuple[Batch | np.ndarray | jax.Array, dict]:
    """`batch` split in one walk into the fields a device can hold
    (`placeable`), the batch a step and an objective's programs are called
    on, and those none can, which host code reads and a compiled step cannot
    take, each in the mappings that hold them (`with_host` joins them back).
    A batch with no host field, a bare array among them, is its own device
    part."""
    if not isinstance(batch, Mapping):
        return batch, {}
    device: dict = {}
    host: dict = {}
    for name, value in batch.items():
        if isinstance(value, Mapping):
            device[name], nested = split_fields(value)
            if nested:
                host[name] = nested
        elif placeable(value):
            device[name] = value
        else:
            host[name] = value
    return (device if host else batch), host


def with_host(placed: Batch, host: Batch) -> Batch:
    """`placed` with the host fields put back where they were."""
    return {**placed, **{name: with_host(placed[name], value) if isinstance(value, Mapping) else value
                         for name, value in host.items()}}


Signature = tuple[jax.tree_util.PyTreeDef, tuple[jax.ShapeDtypeStruct, ...]]
"""A batch's abstract signature, the key a compiled step is held under: its
tree, which holds each node's static fields, and each leaf's shape and
dtype. A fixed batch has one; a ramped batch has one per stage of its ramp,
each compiled on the first step that reads it."""


def batch_signature(batch: Batch) -> Signature:
    """The abstract signature of a batch of arrays or their shapes (`jax.ShapeDtypeStruct`)."""
    leaves, tree = jax.tree.flatten(batch)
    return tree, tuple(jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)) for leaf in leaves)


def batch_layout(batch: Batch) -> str:
    """A batch's fields, each with its dtype and global shape; the batch may be
    arrays or their abstract shapes (`jax.ShapeDtypeStruct`)."""
    return ", ".join(f"{jax.tree_util.keystr(path)} {jnp.result_type(leaf)}{list(np.shape(leaf))}"
                     for path, leaf in jax.tree_util.tree_leaves_with_path(batch))


STEP_BATCH = 2
"""Where a compiled train step takes its batch among its arguments: after the moving state and the
frozen tree (`dew.training.state.apart`, `Trainer._jitted`), before the part of the step it runs."""


def step_reads(executable: jax.stages.Compiled, signature: Signature) -> tuple[bool, ...]:
    """Which leaves of its batch (`STEP_BATCH`) a compiled step reads,
    in the leaf order of the batch's own `signature` (`batch_signature`):
    those the executable kept, where JAX pruned the rest
    (`jax.stages.Compiled.input_shardings` names no sharding for those). A
    field the batch holds as None, as an optional part of `Coordinates`, is
    None in the shardings too and no leaf of the batch, so the batch's tree,
    not the shardings', says where its leaves are."""
    arguments, _ = executable.input_shardings
    tree, _ = signature
    return tuple(sharding is not None for sharding in tree.flatten_up_to(arguments[STEP_BATCH]))


def batch_shardings(mesh: Mesh | AbstractMesh, batch: Batch) -> Placement[Batch]:
    """Return a sharding for each leaf of `batch`, chosen from the leaf's shape.

    Rows split over the row axes, those the rules in context give
    `activation_batch` (`batch_axes`): data, expert and fsdp by default. A
    leaf of rank 2 or 3 holds a sequence per row: token ids, segment ids,
    positions, encoded tokens. Its second dimension splits over the sequence
    axis when the mesh has one that divides it, and otherwise stays whole,
    the way `logical_spec` drops a name no dimension can split. An image or
    a video is not a sequence, so only its rows split.

    This only decides placement and never changes values. Until a model
    constrains its attention to the sequence axis, a sequence split here is
    gathered again inside the model. Only the shape is read, so a leaf that is
    already a global array costs no transfer.
    """
    rows = batch_axes(mesh) or None
    sequence = SEQUENCE_AXIS if SEQUENCE_AXIS in mesh.axis_names else None
    sequence_size = mesh.shape.get(SEQUENCE_AXIS, 1)

    def leaf_sharding(leaf):
        shape = np.shape(leaf)
        if not shape:
            return NamedSharding(mesh, P())
        if sequence is not None and len(shape) in (2, 3) and shape[1] % sequence_size == 0:
            return NamedSharding(mesh, P(rows, sequence))
        return NamedSharding(mesh, P(rows))

    return jax.tree.map(leaf_sharding, batch)


def shard_batch(mesh: Mesh, batch: Batch, reads: Mapping[Signature, tuple[bool, ...]] | None = None) -> Batch:
    """Assemble this process's share of each array into a globally sharded one.

    The placed batch owns its memory because reader buffers are borrowed until
    the next batch. CPU placement copies each borrowed host leaf once, since
    JAX 0.11.2 drops copy_semantics in pxla._shard_np_array and ignores
    may_alias=False for NumPy inputs. Accelerator transfers copy by construction
    and release the borrowed host buffer when the transfer completes.

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

    `reads` holds, by the signature of a batch as placed (`batch_signature`
    of what this returns), which of its leaves the step compiled for it reads
    (`step_reads`); one it does not read stays on the host, no transfer
    made, the reader's buffer still released at the next read: a copy of its
    own where this process holds every row, else a stand-in of the placed
    shape, which the executable checks and drops, its values unread. A
    batch whose placed signature it lacks is placed whole. The lookup is
    here, where the placed shapes are made, so the step that publishes a
    mask and the placement that reads it name one batch alike.

    The rule is deliberately local. Placement runs on
    `DevicePrefetchIterator`'s worker thread while the step's collectives run
    on the caller's, and a collective issued from here would have to be
    ordered against those across every process. Agreeing which sites
    actually carry validity, the way `agreed_validity` does for a generation
    request, belongs where the caller's own collectives are issued.
    """
    device, host = split_fields(batch)
    if host:
        # A field no device holds (`placeable`) stays on the host, where host code reads it.
        return with_host(shard_batch(mesh, device, reads), host)
    batch = filled_validity(batch) if jax.process_count() > 1 else batch
    cpu = all(device.platform == "cpu" for device in mesh.devices.flat)
    leaves, tree = jax.tree.flatten(batch)
    count = DataPartition.of(mesh).count

    def placed(shape: tuple[int, ...]) -> tuple[int, ...]:
        # The share holds whole rows: `count` shares make the rows, and every other dimension is whole.
        return (shape[0] * count, *shape[1:]) if shape else ()

    signature = tree, tuple(jax.ShapeDtypeStruct(placed(np.shape(leaf)), jnp.result_type(leaf))
                            for leaf in leaves)
    read = (reads or {}).get(signature, (True,) * len(leaves))
    # An unread leaf is still an argument the executable checks against the placed shapes before it
    # drops it, so in a pool, whose process holds a share of the rows, it is a zero-cost stand-in
    # of the placed shape; a process that holds every row keeps its own copy.
    held = {index: np.array(leaf, copy=True) if placed(np.shape(leaf)) == np.shape(leaf)
            else np.broadcast_to(np.zeros((), np.asarray(leaf).dtype), placed(np.shape(leaf)))
            for index, (leaf, kept) in enumerate(zip(leaves, read, strict=True))
            if not kept and not isinstance(leaf, jax.Array)}
    moving = [leaf if isinstance(leaf, jax.Array) else np.array(leaf, copy=True) if cpu else np.asarray(leaf)
              for index, leaf in enumerate(leaves) if index not in held]
    every: list[NamedSharding] = jax.tree.leaves(batch_shardings(mesh, batch))
    shardings = [sharding for index, sharding in enumerate(every) if index not in held]
    if jax.process_count() == 1:
        # The whole batch is this process's share, placed by one device_put
        # for the tree: 74 us of host time a batch on an RTX 4080, and 174 on
        # 8 CPU devices, where a call per leaf took 105 and 222.
        moved = iter(jax.device_put(moving, shardings))
    else:
        def place(leaf, sharding: NamedSharding) -> jax.Array:
            # A device of a sequence or a stage that spans processes picks its own slice out of it.
            return jax.make_array_from_process_local_data(sharding, leaf, placed(np.shape(leaf)))

        moved = iter([place(leaf, sharding) for leaf, sharding in zip(moving, shardings, strict=True)])
    return tree.unflatten([held[index] if index in held else next(moved) for index in range(len(leaves))])


def marked(partition: DataPartition, prepare: Callable[[Batch], Batch] | None = None
           ) -> Callable[[Batch], Batch]:
    """A training batch's host preparation: `prepare`'s, then each row's place among its step's
    records (`EXAMPLE_ROWS`). A share (`DataPartition.of`) reads the records `index :: count`, so its
    `j`th row is the step's record `j * count + index` at every count: a record draws what it draws
    however many processes read it (`Step.sample`)."""
    def prepared(batch: Batch) -> Batch:
        batch = batch if prepare is None else prepare(batch)
        rows = np.arange(rows_of(batch), dtype=np.int32)
        return {**batch, EXAMPLE_ROWS: rows * partition.count + partition.index}

    return prepared


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


def _without_state_axes(entry: MeshAxes) -> MeshAxes:
    """A spec entry with the state axes (`STATE_AXES`) dropped from it."""
    kept = tuple(axis for axis in mesh_axes(entry) if axis not in STATE_AXES)
    return kept if kept else None


def _shard_bytes(leaf) -> int:
    """The bytes each device its sharding places it on holds of `leaf`, an
    array or an abstract one that carries its sharding."""
    return math.prod(leaf.sharding.shard_shape(np.shape(leaf))) * leaf.dtype.itemsize


def placed_bytes(tree) -> int:
    """The bytes a device holds of `tree`'s arrays, each laid out as its
    sharding places it (abstract arrays carry their sharding)."""
    return sum(map(_shard_bytes, jax.tree.leaves(tree)))


def placed_shapes[TreeT](tree: TreeT, placement: Placement[TreeT]) -> TreeT:
    """`tree`'s arrays as their shapes, each carrying where `placement` puts it: what a
    lowering, a restore and `Layout.validate` read of a state not yet placed."""
    return jax.tree.map(lambda leaf, sharding: jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype,
                                                                    sharding=sharding), tree, placement)


def placed_held[Held](held: Held, variables: Variables, shardings: Placement[Variables],
                      mesh: Mesh, shared: Mapping[int, NamedSharding] | None = None) -> tuple[Held, Held]:
    """A copy of `held` on `mesh` as two trees, each None where the other holds
    an array: the arrays a state of `variables` placed by `shardings` takes
    over, each where the state keeps the variable of its path and shape
    (`Objective.held_variables`), and the rest, replicated, which no output of
    the state takes.

    Handed to the state's JIT as they are, the arrays land below the state it
    builds and leave a hole the checkpoint's size once freed, which splits
    the free memory a step's temporaries need in one block
    (`step_headroom`). Placed where the state keeps them and donated
    (`taking_over`), they become the state's buffers. The copy leaves the
    objective's own arrays alone. The held leaves whose ids `shared` maps to
    where the state keeps them, which it keeps as they are and no step donates
    (its frozen tree, `apart`, wherever its init moves them), are placed there
    as they are, the same arrays where they are placed already, and kept: no
    copy of them is made."""
    shared = shared or {}
    kept = {path: (sharding, leaf.shape) for (path, leaf), sharding in zip(
        jax.tree_util.tree_leaves_with_path(variables), jax.tree.leaves(shardings), strict=True)}
    replicated = NamedSharding(mesh, P())

    def variable(path, leaf) -> NamedSharding | None:
        # An initializer binds the checkpoint as `variables`
        # (`Objective.initializer`), so a leaf's path below that keyword
        # is its variable's.
        keys = [key.key if isinstance(key, jax.tree_util.DictKey) else None for key in path]
        sharding, shape = kept.get(path[keys.index('variables') + 1:] if 'variables' in keys else (),
                                   (None, None))
        return sharding if shape == np.shape(leaf) else None

    # A copy of an array already on the devices: device_put can reuse a
    # buffer the target shares with the source, which donation would free.
    owned = jax.tree.map(lambda leaf: leaf if id(leaf) in shared or not isinstance(leaf, jax.Array)
                         else leaf.copy(), held)
    placed = jax.device_put(owned, jax.tree_util.tree_map_with_path(
        lambda path, leaf: shared.get(id(leaf)) or variable(path, leaf) or replicated, held))
    keep = {id(placed_leaf) for leaf, placed_leaf in zip(jax.tree.leaves(held), jax.tree.leaves(placed),
                                                         strict=True) if id(leaf) in shared}
    taken = jax.tree_util.tree_map_with_path(
        lambda path, leaf: None if variable(path, leaf) is None or id(leaf) in keep else leaf, placed)
    return taken, jax.tree.map(lambda one, leaf: leaf if one is None else None, taken, placed,
                               is_leaf=lambda leaf: leaf is None)


def taking_over[Held, Result](function: Callable[[Held, jax.Array], Result], taken: Held, kept: Held,
                              shardings: Placement[Result]) -> Callable[[jax.Array], Result]:
    """`function` jitted onto `shardings` over the held tree `placed_held`
    splits: `taken` donated, for the outputs to take its buffers over, and
    `kept` not, as no output takes those and JAX only warns of a donation it
    cannot use. They are freed as the caller drops them."""

    def joined(taken: Held, kept: Held, key: jax.Array) -> Result:
        return function(jax.tree.map(lambda one, other: other if one is None else one, taken, kept,
                                     is_leaf=lambda leaf: leaf is None), key)

    return functools.partial(jax.jit(joined, out_shardings=shardings, donate_argnums=0), taken, kept)


def traced_start(initial_state: Callable[[Initializer, jax.Array], TrainState], initializer: Initializer,
                 key: jax.Array) -> tuple[TrainState, dict[int, int]]:
    """The shapes of the fresh state `initial_state` builds, and which of its frozen leaves are the
    initializer's arrays as they are: each such leaf's index among the frozen tree's leaves, to the
    index of the initializer's leaf it is. An init that returns a held leaf as it is, by reference,
    passes it through (`Objective.held_variables`); one trace finds both."""
    through: dict[int, int] = {}

    def traced(initializer: Initializer, key: jax.Array) -> TrainState:
        state = initial_state(initializer, key)
        inputs = {id(leaf): index for index, leaf in enumerate(jax.tree.leaves(initializer))}
        frozen = jax.tree.leaves(apart(state)[1])
        through.update({index: inputs[id(leaf)] for index, leaf in enumerate(frozen) if id(leaf) in inputs})
        return state

    return jax.eval_shape(traced, initializer, key), through


def placed_start(initial_state: Callable[[Initializer, jax.Array], TrainState], initializer: Initializer,
                 key: jax.Array, abstract: TrainState, shardings: Placement[TrainState],
                 through: Mapping[int, int], mesh: Mesh) -> TrainState:
    """A fresh state on `mesh`, built by `initial_state` from the objective's held tree (`placed_held`).
    The frozen leaves the objective's init passes through as they are (`through`, `traced_start`),
    which no step donates (`apart`), are the held arrays themselves, placed where the state keeps
    them, the same arrays where they are placed already: the run and the objective then hold one
    frozen tree between them. One program builds the rest, a frozen leaf the init builds among it,
    taking the other held arrays over."""
    moving, frozen = apart(shardings)
    if frozen is None or not through:
        taken, kept = placed_held(initializer, abstract.variables, shardings.variables, mesh)
        return taking_over(initial_state, taken, kept, shardings)(key)
    inputs = jax.tree.leaves(initializer)
    where, structure = jax.tree.flatten(frozen)
    # Each passed leaf is placed once, where the state keeps it, whatever path the init moves it to.
    taken, kept = placed_held(initializer, abstract.variables, shardings.variables, mesh,
                              shared={id(inputs[given]): where[index] for index, given in through.items()})
    placed = jax.tree.leaves(jax.tree.map(lambda one, other: other if one is None else one, taken, kept,
                                          is_leaf=lambda leaf: leaf is None))
    built = [index for index in range(len(where)) if index not in through]

    def rest(tree: Initializer, draw: jax.Array) -> tuple[TrainState, tuple[jax.Array, ...]]:
        state, own = apart(initial_state(tree, draw))
        leaves = jax.tree.leaves(own)
        return state, tuple(leaves[index] for index in built)

    state, made = taking_over(rest, taken, kept, (moving, tuple(where[index] for index in built)))(key)
    drawn = dict(zip(built, made, strict=True))

    def leaf(index: int) -> jax.Array:
        if index in drawn:
            return drawn[index]
        # A held leaf an init passes through at two places of other shardings is placed at the second too.
        held = placed[through[index]]
        return held if held.sharding == where[index] else jax.device_put(held, where[index])

    return together(state, jax.tree.unflatten(structure, [leaf(index) for index in range(len(where))]))


def prefetched_bytes(batch: Batch, shardings: Placement[Batch], depth: int = PREFETCH_DEPTH) -> int:
    """Return the bytes a device holds of the batches `fit` places while a step runs.

    These are the batches next to the one the step reads: the `depth` it queues
    (`fit(device_prefetch=)`) and the one it is placing, each laid out as
    `shardings` places `batch`.
    """
    return (depth + 1) * placed_bytes(placed_shapes(batch, shardings))


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

    `prepare`, when supplied, transforms the source's host batch before
    placement, on the same worker and after its checkpoint position is read.
    `reads` holds, by the placed batch's signature, which leaves a compiled
    step reads (`step_reads`); a batch of a signature it lacks, its step not
    yet compiled, is placed whole (`shard_batch`).
    Analytical per-row telemetry therefore travels with its own batch and
    never requires a synchronous device-to-host read in the step loop.

    If the worker thread cannot be created, the source is finalized on the
    caller's thread before the error is re-raised; no worker has touched it at
    that point. That path runs the source's close synchronously, so the close
    has to return on its own. When a running worker is stuck in source code
    that cannot be stopped, `close` raises TimeoutError.
    """

    def __init__(self, iterator: Iterator, mesh: Mesh, depth: int = PREFETCH_DEPTH,
                 source_state: bytes | None = None, *, prepare: Callable[[Batch], Batch] | None = None,
                 reads: Mapping[Signature, tuple[bool, ...]] | None = None):
        if depth <= 0:
            raise ValueError("prefetch depth must be positive")
        self._iterator: Iterator | None = iter(iterator)
        self._source_name = type(self._iterator).__name__
        self._mesh: Mesh | None = mesh
        # The placed batches waiting for the consumer, at most `depth`; `_ready` wakes whoever waits on
        # them, on an enqueue, a dequeue, a stop and the worker's end, so neither side polls.
        self._queued: collections.deque[tuple[Batch, bytes | None]] = collections.deque()
        self._depth = depth
        self._ready = threading.Condition()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._start_lock = threading.Lock()
        # Held while a stop request reaches the source and while the worker lets the source go:
        # a request reaches the source it read (during its close too, which may wait on it), and
        # the worker releases the source only after that request returns.
        self._handover = threading.Lock()
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self.source_state = source_state
        self._prepare = prepare
        # The worker places batches by the rules its creator reads, which flax
        # keeps per thread (`batch_shardings`).
        self._rules = nn.get_logical_axis_rules()
        self._reads = {} if reads is None else reads
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
                            self._stop_source()
                        except BaseException as failure:
                            error.add_note(f"Source cancellation failed: {failure!r}")
                        self._prefetch()
                    raise
                # A worker still placing a batch when Python finalizes would abort the process.
                self._withdraw_exit = stop_at_exit(self._thread, self._cancel, timeout=self._budget())

    def _budget(self) -> float | None:
        """The seconds the source gives itself to stop (`Budgeted.stop_seconds`), None for a
        source that sets no deadline of its own."""
        source = self._iterator
        seconds = source.stop_seconds if isinstance(source, Budgeted) else None
        return None if seconds is None else float(seconds)

    def _cancel(self) -> None:
        """Ask the worker to stop between batches, and a stoppable source to stop its own work."""
        first = not self._stop.is_set()
        self._stop.set()
        with self._ready:
            self._ready.notify_all()
        if first:
            self._stop_source()

    def _stop_source(self) -> None:
        """Ask the source to stop its own work, unless the worker has already let it go."""
        with self._handover:
            source = self._iterator
            if isinstance(source, Stoppable):
                source.request_stop()

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
        batch = state = read = placed = None
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
                    read = source.get_state() if source is not None else None
                    state = read if read is None or isinstance(read, bytes) else json.dumps(read).encode()
                with region("data.place"), nn.logical_axis_rules(self._rules):
                    if self._prepare is not None:
                        batch = self._prepare(batch)
                    placed = shard_batch(mesh, batch, self._reads)
                batch = None
                with region("data.enqueue"), self._ready:
                    self._ready.wait_for(lambda: self._stop.is_set() or len(self._queued) < self._depth)
                    if not self._stop.is_set():
                        self._queued.append((placed, state))
                        self._ready.notify_all()
                placed = state = read = None
        except StopIteration:
            # A drained source is how a prefetch thread ends; the finally
            # below publishes the end, and no error goes with it.
            _log.debug("%s drained; the prefetch thread is done", self._source_name)
        except BaseException as error:
            self._error = error
        finally:
            batch = placed = state = read = None
            try:
                close = iterator.close if isinstance(iterator, Closeable) else None
                if close is not None:
                    close()
            except BaseException as error:
                self._cleanup_error = error
            finally:
                with self._handover:
                    self._iterator = None
                self._mesh = self._prepare = None
                # No bound close method or checkpointable alias may retain a
                # Grain pipeline after the worker exits.
                iterator = source = mesh = close = None
                if self._stop.is_set():
                    self._discard()
                with self._ready:
                    self._done.set()
                    self._ready.notify_all()

    def _discard(self) -> None:
        """Drop every batch already queued, without waiting for more."""
        with self._ready:
            self._queued.clear()
            self._ready.notify_all()

    def close(self, *, timeout: float | None = None) -> None:
        """Stop the worker and wait for it, discarding unread batches but keeping `source_state`.

        The wait lasts `timeout` seconds, else the source's own `stop_seconds`
        (a grain pipeline joins one worker process after another), else until
        the worker ends: the source, the external reader, owns its deadline. If
        the worker is still alive after a deadline, this raises TimeoutError.
        Cancellation stays requested and a later close may wait again, but the
        source's in-flight work and buffers may not have been released. An
        error from the source's own close is raised here too.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("a prefetch worker cannot close itself")
        if timeout is None:
            timeout = self._budget()
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
        with self._ready:
            self._ready.wait_for(lambda: self._queued or self._done.is_set() or self._stop.is_set())
            if self._stop.is_set():
                raise StopIteration
            if self._queued:
                batch, position = self._queued.popleft()
                self._ready.notify_all()
                self.source_state = position
                return batch
        # The worker ended, every batch it published read: with its error, if it had one.
        error, self._error = self._error, None
        cleanup, self._cleanup_error = self._cleanup_error, None
        if error is not None:
            if cleanup is not None:
                error.add_note(f"Source cleanup failed: {cleanup!r}")
            raise error from None
        if cleanup is not None:
            raise cleanup from None
        raise StopIteration


__all__ = ["DevicePrefetchIterator", "Layout", "LeafPlacement", "MeshSpec", "Validation", "batch_shardings"]
