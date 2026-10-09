"""Where a parameter is split across devices, declared on the module that creates it.

A module names the logical axes of the parameters its submodules create,
keyed by the trailing module path, outermost dimension first:

    @logical_axes({("q_proj",): ("embed", "heads"), ("o_proj",): ("attention", "embed")})
    class CausalSelfAttention(nn.Module): ...

A parameter takes as many of the trailing names as its rank holds, so a
kernel takes all of them and its bias takes the output ones. The
declarations of every decorated module merge into one table. The `Layout`
reads that table when it places a train state, and Muon reads it when it
picks a parameter's matrix axes. The models stay plain Flax modules whose
init returns arrays. The optimizer's moments and the EMA copy have paths
that end in their parameter's path, so one declaration covers them as well.

A parameter that no declaration names (a convolution, a state matrix, a
projection with no side worth naming) gets the shape heuristic, which
places it on its largest divisible axis.

This module also defines the mesh axis names and the functions that read
the mesh in context: `pipeline_stages` for the decoder's stage count,
`microbatches` for the schedule the trainer sets in context around its
compiled step, `sequence_shards` for how many ways attention and the Mamba-2
mixer split a sequence, `row_axes` for the axes their `shard_map`s split
rows over, and `manual_map` for those maps.

`DEFAULT_RULES` maps the logical names onto the mesh, for parameters and
activations alike. `logical_spec` reads it (or the rules a layout sets in
context) to place a parameter or an activation, and `constrain` applies that
placement to an activation.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import math
import types
from collections.abc import Callable, Iterator, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import spmd

type LogicalAxes = tuple[str | None, ...]
type Suffix = tuple[str, ...]

DATA_AXIS = 'data'
EXPERT_AXIS = 'expert'
FSDP_AXIS = 'fsdp'
TENSOR_AXIS = 'tensor'
SEQUENCE_AXIS = 'sequence'
STAGE_AXIS = 'stage'
"""The axes of a mesh `MeshSpec.build` builds, plus the stage axis a
pipeline mesh adds. They are named here because the attention seam and the
decoder read them off the mesh in context: the sequence axis attention splits
its queries over, and the tensor and stage axes, which hold a width and a
pipeline stage and never a row."""

MESH_AXES = (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS, SEQUENCE_AXIS, STAGE_AXIS)
"""Every mesh's axes, in the order `MeshSpec.build` lays them out."""

BATCH_AXES = (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS)
"""The mesh axes a batch's rows split over, in mesh order. Sequence holds a
slice of the positions, stage the pipeline's stages that hand one batch's
microbatches along, and tensor the widths of Megatron's split, so every
tensor shard computes its share of the width for every row."""


class LayoutRefused(ValueError):
    """Raised for a mesh layout that a model, an objective or a device set does not run, by design.

    The message names the axis, says why it cannot hold what it would split,
    and gives a layout that runs. Any other error on a layout is a defect.
    It is a ValueError, so a caller that catches ValueError catches it too.
    """

type MeshAxes = str | tuple[str, ...] | None
type LogicalAxisRules = tuple[tuple[str, MeshAxes], ...]

# Rule order is precedence when two logical dimensions target the one mesh
# axis. A name written twice is an ordered pair of choices, the form flax
# documents for `logical_to_mesh_axes`: the second places a dimension the
# first could not, because another dimension of the same array took that
# axis, or because its axes do not divide the dimension (`logical_spec`).
#
# The tensor axis carries Megatron's split. That is the mlp's hidden width
# ('mlp'), the attention's query heads ('heads') and its grouped key and value
# heads ('kv'), the attention width o_proj reads back ('attention'), and the
# vocabulary of the embedding table and the output head ('vocab'). Each of
# those is the output side of one matmul and the input side of the next, so
# splitting it splits both and leaves the block one reduction.
#
# 'embed', the residual width every norm, residual add, rotation and the
# loss share, stays whole on the tensor axis and keeps fsdp: sharding it puts
# a collective between every pair of sublayers. A kernel holding both splits
# in two dimensions as MaxText's rules do, the residual width over fsdp and
# the Megatron width over tensor (the mlp and attention widths name tensor
# first and fsdp second). Split over fsdp and tensor together, those widths
# had GSPMD repeat products on both devices of every pair (1.21 times one
# device's FLOPs, layout_parity's dense decoder on fsdp2_tensor2). The
# vocabulary, which shares no array with a residual, splits fsdp times
# tensor ways.
#
# The activation_ names place what a step computes rather than what it
# stores, by the same table: rows over the batch axes, positions over the
# sequence axis, and the widths of Megatron's split over tensor. The residual
# width (activation_embed) has no rule, so it stays whole.
DEFAULT_RULES: LogicalAxisRules = (
    ("vocab", (FSDP_AXIS, TENSOR_AXIS)),
    ("mlp", TENSOR_AXIS),
    ("mlp", FSDP_AXIS),
    ("modulation", FSDP_AXIS),
    ("attention", TENSOR_AXIS),
    ("attention", FSDP_AXIS),
    # The gated delta net's projected width (keys, values and their gate),
    # placed like the attention's: the width over the model dimension.
    ("linear", FSDP_AXIS),
    ("embed", FSDP_AXIS),
    ("head_dim", FSDP_AXIS),
    ("heads", (FSDP_AXIS, TENSOR_AXIS)),
    ("kv", (FSDP_AXIS, TENSOR_AXIS)),
    # The latent widths multi-head latent attention compresses through and
    # the sparse indexer's head dim: model-width-like, so they ride fsdp.
    ("index", FSDP_AXIS),
    ("kvlora", FSDP_AXIS),
    ("qlora", FSDP_AXIS),
    ("output", FSDP_AXIS),
    ("exp", EXPERT_AXIS),
    # A projection from the residual width to the heads, q_proj's shape: the
    # rule above gave 'embed' fsdp, so the heads take the tensor axis by
    # itself rather than fall back to replication.
    ("heads", TENSOR_AXIS),
    ("kv", TENSOR_AXIS),
    ("activation_batch", BATCH_AXES),
    ("activation_length", SEQUENCE_AXIS),
    # The positions of a stretch of per-token work that splits no width the
    # tensor axis splits, which every tensor shard would otherwise compute
    # whole (`SPREAD`): over the tensor axis beside the sequence axis, as
    # Megatron's sequence parallelism splits the positions of its norms;
    # over the sequence axis alone where the tensor axis cannot divide them.
    ("activation_spread", (SEQUENCE_AXIS, TENSOR_AXIS)),
    ("activation_spread", SEQUENCE_AXIS),
    ("activation_heads", TENSOR_AXIS),
    ("activation_kv", TENSOR_AXIS),
    ("activation_mlp", TENSOR_AXIS),
    ("activation_vocab", TENSOR_AXIS),
    # A served paged cache's pool splits its pages as the served rows split,
    # one part of the pool per group of rows (`dew.inference.serving`).
    ("pages", BATCH_AXES),
    # Rows too few for every batch axis split over the first ones.
    ("activation_batch", (DATA_AXIS, EXPERT_AXIS)),
    ("activation_batch", DATA_AXIS),
)

RESIDUAL: LogicalAxes = ("activation_batch", "activation_length", "activation_embed")
"""The logical axes of a `[batch, length, width]` activation between sublayers,
which is also each sublayer's input. Under a tensor axis the width stays
whole, so every tensor shard reads every row it computes its share of the
width for."""
HEADS: LogicalAxes = ("activation_batch", "activation_length", "activation_heads", None)
"""A `[batch, length, heads, head_dim]` query, or the attention's output."""
KV_HEADS: LogicalAxes = ("activation_batch", "activation_length", "activation_kv", None)
"""A `[batch, length, kv_heads, head_dim]` key or value. Grouped heads the
tensor axis does not divide are computed whole on every tensor shard, the
way Megatron repeats them."""
SPREAD: LogicalAxes = ("activation_batch", "activation_spread", None)
"""A `[batch, length, width]` activation whose per-token work splits no
tensor width, multi-head latent attention's down-projections and their
latents: its positions split over the tensor axis too, so each tensor shard
computes its own tokens, and the result is gathered back into `RESIDUAL`'s
placement for the work that splits a width."""
MLP_HIDDEN: LogicalAxes = ("activation_batch", "activation_length", "activation_mlp")
"""A `[batch, length, hidden]` feed-forward activation."""
LOGITS: LogicalAxes = ("activation_batch", "activation_length", "activation_vocab")
"""`[batch, length, vocab]` scores."""


DECLARED: dict[type, dict[Suffix, LogicalAxes]] = {}
"""Each decorated module class's own declarations."""
_OWNERS: dict[Suffix, tuple[LogicalAxes, type]] = {}
"""Every declaration by its suffix, with the class that made it: the table
`declared_axes` matches, one set of axes per suffix across all classes."""

@dataclasses.dataclass
class Schedule:
    """The microbatch count a step feeds the stage axis, and whether a model in the step ran a pipeline.

    `microbatches` sets `pipelined` when the pipeline reads the count.
    """

    count: int | None
    pipelined: bool = False


_SCHEDULE: contextvars.ContextVar[Schedule | None] = contextvars.ContextVar(
    'pipeline_schedule', default=None)


def pipeline_stages() -> int:
    """How many pipeline stages the mesh in context has, 1 with no mesh or no
    such axis.

    The trainer runs its compiled step under `jax.set_mesh`, and that puts
    the mesh in context while the step traces; a model called outside it
    runs its layer stack whole.
    """
    mesh = jax.sharding.get_abstract_mesh()
    return 1 if mesh.empty else mesh.shape.get(STAGE_AXIS, 1)


@contextlib.contextmanager
def pipeline_microbatches(count: int | None) -> Iterator[Schedule]:
    """How many microbatches a step feeds the stage axis, for the model that
    traces inside. None leaves one microbatch per stage, the smallest schedule
    a pipeline runs. The schedule yielded says, once the step has traced,
    whether a pipeline ran."""
    schedule = Schedule(count)
    token = _SCHEDULE.set(schedule)
    try:
        yield schedule
    finally:
        _SCHEDULE.reset(token)


def microbatches() -> int:
    """The microbatch count in context, or one per stage of the mesh in
    context. A pipeline reads it as it runs."""
    schedule = _SCHEDULE.get()
    if schedule is None:
        return pipeline_stages()
    schedule.pipelined = True
    return pipeline_stages() if schedule.count is None else schedule.count


@dataclasses.dataclass
class Link:
    """A mesh axis's interconnect, one device's dense bf16 peak, and whether a projection split over the axis.

    The trainer measures the interconnect when it places a step
    (`dew.training.distributed.link_bandwidth`). `down_projection` and
    `split_positions` set `spread` when they decide to split a projection in
    the step over the axis.
    """

    bytes_per_second: float | None
    """The bytes one device receives per second in an all-gather over the axis:
    (N - 1) / N of the result, divided by the time the gather took. None where
    nothing was measured: on a CPU mesh, or on a device the peak table does not
    name."""
    flops_per_second: float | None
    """One device's dense bf16 peak in FLOPs per second, or None for hardware
    the peak table does not name (`dew.telemetry.instrumentation.peak_flops`)."""
    platform: str
    spread: bool = False


_LINKS: contextvars.ContextVar[Mapping[str, Link]] = contextvars.ContextVar(
    'links', default=types.MappingProxyType({}))


@contextlib.contextmanager
def measured_links(links: Mapping[str, Link]) -> Iterator[Mapping[str, Link]]:
    """Trace with `links`, each mesh axis's measured interconnect by axis
    name, for the projections that decide from them; each link yielded says,
    once the step has traced, whether a projection spread over its axis."""
    token = _LINKS.set(links)
    try:
        yield links
    finally:
        _LINKS.reset(token)


def _spreads(axis: str, x: jax.Array, latent: int, tokens: float, *, padding: float = 0.0) -> bool:
    """Whether a projection of `x`'s tokens to `latent` features, which would
    otherwise run on every token of every `axis` shard, runs split over
    `axis`, each shard on its own share of the `tokens` one `axis` group
    computes. It splits where that cannot slow the step.

    Each of N shards currently projects all tokens; a split projects
    (tokens + padding) / N on each. The saved work excludes padded rows,
    but their input gradients and outputs still cross the link. Their
    gathers move (N - 1) / N of the padded arrays, and a ring sum moves
    twice that fraction of the fp32 weight gradient. It cannot lose where
    the axis's link moves those bytes in no more time than the device's
    peak takes for the saved FLOPs. The weight gradient is a microbatch's,
    so it weighs more at fewer tokens. A CPU mesh's devices share one host's
    memory, which moves a byte in less time than a CPU's matmul spends on a
    thousand FLOPs, and it splits. Without a measured link, or with a device
    the peak table does not name, the projection runs on every token."""
    link = _LINKS.get().get(axis)
    if link is None:
        return False
    shards = jax.sharding.get_abstract_mesh().shape[axis]
    saved = tokens - (tokens + padding) / shards
    if saved <= 0:
        return False
    width = x.shape[-1]
    flops = 6 * width * latent * saved
    moved = (1 - 1 / shards) * ((width + latent) * x.dtype.itemsize * (tokens + padding)
                                + 2 * width * latent * 4)
    if link.platform == 'cpu':
        spreads = True
    elif link.bytes_per_second is None or link.flops_per_second is None:
        spreads = False
    else:
        spreads = link.bytes_per_second * flops >= link.flops_per_second * moved
    link.spread = link.spread or spreads
    return spreads


def down_projection(x: jax.Array, latent: int) -> LogicalAxes:
    """Where the per-token projections of `x`, the residual or any other
    `[batch, ..., tokens, width]` input whose width the tensor axis does not
    split, to `latent` features in all run: `SPREAD`, each tensor shard on
    its own tokens, where the tensor axis's link pays for it (`_spreads`),
    else `RESIDUAL`, every tensor shard on every token. At DeepSeek-V3's widths
    in bf16 and 16384 tokens the link must move 20.3 GB/s for an RTX 3090
    (an NVLink pair's 31.0 meets it, PCIe 3.0's 5.8 does not) and 283 GB/s for
    an H100."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or mesh.shape.get(TENSOR_AXIS, 1) == 1 or TENSOR_AXIS in mesh.manual_axes:
        return RESIDUAL
    # The tokens one tensor group computes: the rows and positions the
    # residual's other axes leave it.
    split = math.prod(mesh.shape[axis] for entry in logical_spec(RESIDUAL[:2], x.shape[:2])
                      for axis in mesh_axes(entry))
    tokens = math.prod(x.shape[:-1]) / split
    return SPREAD if _spreads(TENSOR_AXIS, x, latent, tokens) else RESIDUAL


def sequence_shards() -> int:
    """How many ways the mesh in context splits the sequence axis, 1 with no
    mesh or no such axis.

    The trainer runs its compiled step under `jax.set_mesh`, so the mesh is
    in context while the step traces; a model called outside
    it sees whole sequences. So does code inside a `shard_map` that took the
    sequence axis manual: each of its instances holds whatever it was handed,
    which is how Ulysses's exchange runs a whole-sequence kernel.
    """
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or SEQUENCE_AXIS in mesh.manual_axes:
        return 1
    return mesh.shape.get(SEQUENCE_AXIS, 1)


def split_positions(x: jax.Array, head_shape: tuple[int, int],
                    project: Callable[[jax.Array], tuple[jax.Array, jax.Array]]
                    ) -> tuple[jax.Array, jax.Array]:
    """Project a context's keys and values on each sequence shard's positions.

    `project(x)` returns the two HEADS-shaped projections, each ending in
    `head_shape`. Bytes and FLOPs are priced at the head width the active rules
    leave on each device. Where the sequence link pays, pad to a multiple of
    its shard count, project, and trim; otherwise project the context
    unchanged. A cross-attention's context (SD's 77 text tokens) is the input
    this serves; its weight-gradient sum usually outweighs the saved
    projection, so a GPU's link rarely pays (an H100 would need 2.47 TB/s at
    SDXL's widths), while a CPU mesh splits it."""
    shards = sequence_shards()
    length = x.shape[1]
    if shards == 1 or length % shards == 0:
        return project(x)
    padded_length = length + -length % shards
    mesh = jax.sharding.get_abstract_mesh()
    rows = x.shape[0] / math.prod(mesh.shape[axis] for axis in row_axes(x.shape[0]))
    projected = logical_spec(HEADS, (x.shape[0], padded_length, *head_shape))
    feature_shards = math.prod(mesh.shape[axis] for entry in projected[2:] for axis in mesh_axes(entry))
    features = 2 * math.prod(head_shape) // feature_shards
    if not _spreads(SEQUENCE_AXIS, x, features, rows * length,
                    padding=rows * (padded_length - length)):
        return project(x)
    padded = jnp.pad(x, ((0, 0), (0, padded_length - length)) + ((0, 0),) * (x.ndim - 2))
    return jax.tree.map(lambda out: out[:, :length], project(padded))


def row_axes(batch: int, *, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None
             ) -> tuple[str, ...]:
    """The mesh axes a `shard_map` over the sequence axis splits `batch` rows
    over: the ones `activation_batch` takes for `batch` rows on `mesh`, the
    mesh in context by default (`logical_spec`). A batch too small for them is
    computed alike on those axes' shards, the way GSPMD replicates a dimension
    it cannot split."""
    spec = logical_spec(("activation_batch",), (batch,), mesh=mesh)
    return mesh_axes(spec[0]) if spec else ()


def manual_map(local, in_specs, out_specs):
    """`local` in a `shard_map` over every axis the context leaves automatic.

    Those the specs do not name go manual too, and the operands are
    replicated over them. A Mosaic kernel (splash, the Mamba-2 SSD scan)
    refuses to lower where any axis is still left to the partitioner,
    whatever its size, and cuDNN's partitioning rule refuses queries split
    unlike their keys (`_check_qkv_bias_mask_spec`,
    jax/_src/cudnn/fused_attention_stablehlo.py), so the kernel has to see
    local arrays. The pipeline also needs it: it vmaps its stages with
    spmd_axis_name=stage, and a vmapped shard_map can only split the new
    dimension over an axis it holds manual.

    Pallas kernels state no varying-manual-axes type for their outputs, so
    one inside the map needs the check off, as MaxText wraps splash. Every
    operand is split on the axes the specs name and nothing is reduced over
    another, so the check has nothing to catch.
    """
    mesh = jax.sharding.get_abstract_mesh()
    manual = {axis for axis in mesh.axis_names if axis not in mesh.manual_axes}
    return jax.shard_map(local, in_specs=in_specs, out_specs=out_specs, axis_names=manual,
                         check_vma=False)


def batch_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple[str, ...]:
    """The axes of `mesh` a batch's rows split over: every axis
    `activation_batch` takes, each of which divides `mesh.size` rows."""
    return row_axes(mesh.size, mesh=mesh)


def mesh_axes(assignment: MeshAxes) -> tuple[str, ...]:
    """Read one entry of a spec or a rule as the mesh axes it names."""
    if assignment is None:
        return ()
    return (assignment,) if isinstance(assignment, str) else tuple(assignment)


def axis_rules() -> LogicalAxisRules:
    """The rules in context, `flax.linen.logical_axis_rules`'s, where the
    trainer puts its layout's; the default table outside one."""
    return tuple(nn.get_logical_axis_rules()) or DEFAULT_RULES


def logical_spec(axes: LogicalAxes, shape: tuple[int, ...], *,
                 rules: LogicalAxisRules | None = None,
                 mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None
                 ) -> jax.sharding.PartitionSpec:
    """Return the spec that `rules` give, on `mesh`, an array of `shape` whose dimensions `axes` names.

    `rules` and `mesh` default to the ones in context. A mesh axis of size 1
    shards nothing, and a manual axis belongs to the `shard_map` in context,
    so both are dropped from the rules before they are read. Such an axis
    claims no dimension, and a name whose rule named only such axes takes
    its next rule. With no tensor axis, 'mlp' passes its tensor rule to
    fsdp. A rule whose axes do not divide its dimension evenly cannot split
    it, so that rule is set aside for this array: the name takes its next
    rule, or none, and the mesh axis goes to the next dimension that names
    it. For example, an odd vocabulary shards the embedding on its width and
    keeps the table in the layout.
    """
    rules = axis_rules() if rules is None else rules
    mesh = jax.sharding.get_abstract_mesh() if mesh is None else mesh

    def splitting(assignment: MeshAxes) -> tuple[str, ...]:
        return tuple(axis for axis in mesh_axes(assignment)
                     if axis not in mesh.manual_axes and mesh.shape.get(axis, 1) > 1)

    # A rule of None keeps its name whole and stands; one whose axes all
    # split nothing is dropped.
    left: list[tuple[str, MeshAxes]] = []
    for name, assignment in rules:
        if assignment is None:
            left.append((name, None))
        elif used := splitting(assignment):
            left.append((name, used[0] if len(used) == 1 else used))
    while True:
        mapped = spmd.logical_to_mesh_axes(axes, tuple(left))
        assert mapped is not None, "flax answers None for array_dim_names=None only"
        assigned = [mesh_axes(assignment) for assignment in mapped]
        blocked = {(name, assignment) for name, assignment, used, size
                   in zip(axes, mapped, assigned, shape, strict=True)
                   if size % math.prod(mesh.shape[axis] for axis in used)}
        if not blocked:
            break
        left = [rule for rule in left if rule not in blocked]
    entries = [used[0] if len(used) == 1 else used or None for used in assigned]
    while entries and entries[-1] is None:
        entries.pop()
    return jax.sharding.PartitionSpec(*entries)


def constrain(x: jax.Array, axes: LogicalAxes) -> jax.Array:
    """`x` placed as `logical_spec` places the dimensions `axes` names.

    Without it GSPMD picks each activation's placement from the weights
    around it, and splits the residual width wherever fsdp splits the
    matrices that read it: every projection then sums partial products
    across the fsdp axis, rounded before the sum, instead of gathering the
    weight. With it fsdp gathers weights and splits rows, as fully sharded
    data parallelism is defined. No mesh in context leaves `x` as it is, and
    under a pipeline's vmap the stage axis takes the vmapped dimension. An
    Explicit axis stays where `x`'s type places it (`auto_part`)."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return x
    return jax.lax.with_sharding_constraint(x, auto_part(logical_spec(axes, x.shape), mesh))


def auto_part(spec: jax.sharding.PartitionSpec,
              mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> jax.sharding.PartitionSpec:
    """`spec` over `mesh`'s Auto axes alone, which is what a sharding
    constraint may name: an array's type carries its Explicit axes
    (`MeshSpec.explicit`), placed by the operations that made it."""
    explicit = set(mesh.explicit_axes)
    if not explicit:
        return spec
    kept = [tuple(axis for axis in mesh_axes(entry) if axis not in explicit) for entry in spec]
    return jax.sharding.PartitionSpec(*(axes[0] if len(axes) == 1 else axes or None for axes in kept))


def explicit_spec(x: jax.Array, trailing: int = 0) -> jax.sharding.NamedSharding | None:
    """Where `x`'s type places it, over its dimensions and `trailing` more
    left whole, when an Explicit axis splits it (`MeshSpec.explicit`); None
    otherwise. An operation JAX cannot place from its operands' types (a
    gather, a scatter, a reshape that splits a split dimension, a repeat)
    takes it as its `out_sharding`; None leaves the operation as it was."""
    sharding = jax.typeof(x).sharding
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return None
    explicit = set(sharding.mesh.explicit_axes)
    spec = tuple(sharding.spec)
    if not any(axis in explicit for entry in spec for axis in mesh_axes(entry)):
        return None
    padded = (*spec, *(None,) * (x.ndim - len(spec) + trailing))
    return jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(*padded))


def rows_spec(like: jax.Array, ndim: int) -> jax.sharding.NamedSharding | None:
    """A placement for an array of `ndim` dimensions whose rows are `like`'s:
    the Explicit axis that splits `like`'s first dimension on its first, the
    rest whole; None where no Explicit axis splits `like`."""
    placed = explicit_spec(like)
    if placed is None:
        return None
    rows = jax.sharding.PartitionSpec(placed.spec[0], *(None,) * (ndim - 1))
    return jax.sharding.NamedSharding(placed.mesh, rows)


def whole_spec(like: jax.Array, ndim: int) -> jax.sharding.NamedSharding | None:
    """A whole placement for an array of `ndim` dimensions on `like`'s mesh,
    where an Explicit axis splits `like`: where a contraction over its split
    rows lands its sum, which JAX asks to be told, or where its rows are
    gathered whole; None where no Explicit axis splits `like`."""
    placed = explicit_spec(like)
    if placed is None:
        return None
    return jax.sharding.NamedSharding(placed.mesh, jax.sharding.PartitionSpec(*(None,) * ndim))


def rows_like(x: jax.Array, like: jax.Array) -> jax.Array:
    """`x` with its rows placed as `like`'s are, where an Explicit axis splits
    them: a whole `x` is cut to each device's rows, without a collective."""
    placed = rows_spec(like, x.ndim)
    return x if placed is None else jax.sharding.reshard(x, placed)


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def logical_axes(declared: Mapping[Suffix, LogicalAxes]):
    """Declare the parameter axes of the modules `cls` creates.

    A suffix names a parameter's trailing module names, matched in every
    model in the process, so two classes that declare one suffix differently
    are refused here, both named. A class outside Dew qualifies a one-name
    suffix with the name its own module goes by, (`head`, `readout`) rather
    than (`readout`,), which would place any model's `readout`.
    """
    declared = {tuple(suffix): tuple(axes) for suffix, axes in declared.items()}
    for suffix, axes in declared.items():
        names = [name for name in axes if name is not None]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            # The rules hand a mesh axis to a logical name once per array.
            raise ValueError(
                f"{'/'.join(suffix)} is declared {axes}, which names "
                f"{', '.join(repeated)} twice; a kernel whose two sides share a "
                f"width names one of them or leaves the other None")

    def decorate(cls):
        single = sorted('/'.join(suffix) for suffix in declared if len(suffix) == 1)
        if single and cls.__module__.partition('.')[0] != 'dew':
            raise ValueError(
                f"{_qualified(cls)} declares the one-name suffixes {single}, which place a "
                f"module of that name in any model; qualify each with the name of the "
                f"module that holds it, e.g. ('head', '{single[0]}')")
        for suffix, axes in declared.items():
            held = _OWNERS.get(suffix)
            if held is not None and held[0] != axes:
                raise ValueError(
                    f"{'/'.join(suffix)} is declared {axes} by {_qualified(cls)} and "
                    f"{held[0]} by {_qualified(held[1])}; a suffix has one set of axes")
        for suffix, axes in declared.items():
            _OWNERS.setdefault(suffix, (axes, cls))
        DECLARED.setdefault(cls, {}).update(declared)
        return cls

    return decorate


def parameter_path(path) -> Suffix:
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


def _matching(table, names: Suffix):
    for length in range(len(names), 0, -1):
        if names[-length:] in table:
            return names[-length:]
    return None


_BOXED: contextvars.ContextVar[Mapping[Suffix, LogicalAxes]] = contextvars.ContextVar(
    "boxed", default=types.MappingProxyType({}))
"""The axes modules boxed on their own parameters (`boxed_axes`), which
`declared_axes` reads ahead of the suffix table while `boxed` holds them."""


def boxed_axes(tree) -> dict[Suffix, LogicalAxes]:
    """The axes each parameter a module boxed itself carries
    (`nn.with_logical_partitioning`), by the parameter's full path,
    collection first; `nn.unbox` drops them."""
    table: dict[Suffix, LogicalAxes] = {}

    def visit(path, leaf):
        if isinstance(leaf, nn.LogicallyPartitioned):
            table[parameter_path(path)] = tuple(leaf.names)

    jax.tree_util.tree_map_with_path(visit, tree,
                                     is_leaf=lambda leaf: isinstance(leaf, nn.LogicallyPartitioned))
    return table


def ruled_boxes(table: Mapping[Suffix, LogicalAxes], rules: LogicalAxisRules) -> dict[Suffix, LogicalAxes]:
    """`table`'s boxes that name an axis `rules` places. A box whose names
    the rules never mention keeps the shape heuristic, as an undeclared
    parameter does, where its names alone would leave it whole: a Flax model
    boxing with its own names (MaxText's 'activation_*', 'kv')."""
    named = {name for name, _ in rules}
    return {path: names for path, names in table.items() if named.intersection(names)}


def current_boxes() -> Mapping[Suffix, LogicalAxes]:
    """The boxed axes `declared_axes` reads now."""
    return _BOXED.get()


@contextlib.contextmanager
def boxed(table: Mapping[Suffix, LogicalAxes]) -> Iterator[None]:
    """Read `table`'s boxed axes ahead of the suffix table while the block runs."""
    token = _BOXED.set(dict(table))
    try:
        yield
    finally:
        _BOXED.reset(token)


def declared_axes(path, ndim: int) -> LogicalAxes | None:
    """The declared axes of the parameter at `path`, or None for an unnamed one.

    A parameter its module boxed with its axes (`boxed`) takes exactly
    those, matched by its whole path: a variables leaf by its own, a leaf of
    a tree that mirrors the params collection (an optimizer moment, a
    gradient) under `params`. Otherwise a declaration names a module, whose
    parameters share its axes, or one parameter under its module, for a
    module whose leaves have different axes (GPT OSS's fused experts). The
    parameter's own path is tried first.
    """
    names = parameter_path(path)
    held = _BOXED.get()
    own = held.get(names) or held.get(("params", *names))
    if own is not None:
        if ndim > len(own):
            raise ValueError(f"{'/'.join(names)} is boxed {own}, which cannot name its {ndim} dimensions")
        return own[len(own) - ndim:]
    suffix = _matching(_OWNERS, names) or _matching(_OWNERS, names[:-1])
    if suffix is None:
        return None
    axes, owner = _OWNERS[suffix]
    if ndim > len(axes):
        raise ValueError(
            f"{'/'.join(suffix)} is declared {axes} by {_qualified(owner)}, which cannot "
            f"name the {ndim} dimensions of {'/'.join(names)}")
    return axes[len(axes) - ndim:]


__all__ = ["DATA_AXIS", "FSDP_AXIS", "RESIDUAL", "LayoutRefused", "Link", "LogicalAxes", "LogicalAxisRules",
           "Schedule", "logical_axes", "logical_spec"]
