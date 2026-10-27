"""Where a run's arrays live: the device mesh (`MeshSpec`), the parameter and state layout over it
(`Layout`, from the logical axes every module declares, `dew.nn.sharding`, and the rule table,
`dew.mesh`), and what a placed tree holds (`placed_bytes`, `placed_shapes`). A run's batches on the
devices are `dew.distributed.batches`'. It sits below the objectives and the trainer, which place
their state by it, and imports nothing of theirs."""

from __future__ import annotations

import collections
import dataclasses
import fnmatch
import json
import math
from collections.abc import Callable, Mapping, Sequence

import jax
import numpy as np
from flax import linen as nn
from jax.experimental import mesh_utils
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P

from dew.batch import FROZEN, Variables
from dew.mesh import (
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
    batch_axes,
    logical_spec,
    mesh_axes,
)
from dew.nn.sharding import Suffix, boxed, boxed_axes, current_boxes, declarations, declared_axes, ruled_boxes
from dew.state import VARIABLE_FIELDS

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

    def build(self, devices: Sequence | None = None) -> Mesh:
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

        `devices` lists the devices of one slice in the order to use them, any
        devices a mesh holds (a compile-only GPU's among them). The mesh takes
        them row-major over `MESH_AXES`, so neighbouring entries land on the
        last axes. When it is unset, the mesh uses every device, in the order
        `jax.make_mesh` gives the platform's topology.

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
    counts read (`dew.mesh.batch_axes`). Empty leaves the rules as
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

    def keeping(self, fields: Sequence[str]) -> Layout:
        """This layout with `fields` of `HOST_RESIDENT` kept in pinned host memory beside its own."""
        kept = (field for field in fields if field not in self.host)
        return dataclasses.replace(self, host=(*self.host, *kept))

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
    it (all-gather) and sums each gradient into its shares (reduce-scatter), as fully sharded data
    parallelism does: a row axis, and the sequence axis, which holds the state in shares beside
    fsdp (`STATE_AXES`). These are the GPU's step's: XLA:CPU partitions a gradient's sum into
    shares as a whole sum each device slices (all-reduce), which no GPU program runs (a sequence
    axis of two H100s reduce-scatters, tests/test_layout_validation.py). An axis that splits a
    variable and no batch, tensor, sums the
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
            ops += ["all-gather", "reduce-scatter"]
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
    """Whether a device can hold every leaf of `field`: arrays of numbers, or
    their shapes (`jax.ShapeDtypeStruct`) where a run reads a batch before
    placing it, not strings, bytes or Python objects, such as a caption or
    an image's catalog URL."""
    return all(isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) or (
        not isinstance(leaf, (str, bytes)) and np.asarray(leaf).dtype.kind not in "OSUV")
        for leaf in jax.tree.leaves(field))


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


def link_topology(mesh: Mesh, axis: str) -> str:
    """What a figure of `mesh`'s `axis` is a figure of: the axis, the mesh's axes and their widths in
    order, and its devices in the mesh's order, each by id, process, kind and slice. A device's id is
    its place in its system, so two meshes of one shape over other devices, or over the same devices
    grouped otherwise along the axis, are other links; one built again over the same placement is the
    same."""
    devices = [[device.id, device.process_index, device.device_kind, _slice(device)]
               for device in mesh.devices.flat]
    return json.dumps([axis, [[name, width] for name, width in mesh.shape.items()], devices])


__all__ = ["Layout", "LeafPlacement", "MeshSpec", "Validation"]
