"""Where a parameter splits, declared on the module that owns it.

A module names the logical axes of the parameters its submodules create,
keyed by the trailing module path, outermost dimension first:

    @logical_axes({("q_proj",): ("embed", "heads"), ("o_proj",): ("attention", "embed")})
    class CausalSelfAttention(nn.Module): ...

A parameter takes the trailing names its rank can hold, so a kernel takes all
of them and its bias the output ones. The declarations of every decorated
module merge into one table the `Layout` reads when it places a train state
and Muon reads when it picks a parameter's matrix axes. The models stay
plain Flax modules whose init returns arrays. The optimizer's
moments and the EMA copy have paths ending in their parameter's, so one
declaration reaches them as well.

`heuristic` lists what a class leaves to the shape heuristic on purpose (a
convolution, a state matrix, a projection with no side worth naming): their
parameters are placed on their largest divisible axis. Each entry is a run
of `fnmatch` patterns matched against consecutive names of the parameter
path, so `("time_embed",)` covers every parameter under that module and
`("up_dense_*",)` a numbered family. The coverage test in
tests/test_architectures.py reports a declared or heuristic name that no
parameter carries any more, so a renamed submodule fails there.

The mesh axis names live here too, with the readers of the mesh in context:
`pipeline_stages` for the decoder's stage count, `microbatches` for the
schedule the trainer puts in context around its compiled step,
`sequence_shards` for how many ways attention and the Mamba-2 mixer split a
sequence, `row_axes` for the axes their `shard_map`s split rows over, and
`manual_map` for those maps.

`DEFAULT_RULES` maps the logical names onto the mesh, parameters and
activations alike: `logical_spec` reads it (or the rules a layout puts in
context) for a parameter's placement and for an activation's, which
`constrain` pins.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import fnmatch
import math
import types
from collections.abc import Callable, Iterable, Iterator, Mapping

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
"""The axes of a mesh `dew.training.build_mesh` builds, plus the stage axis a
pipeline mesh adds. They are named here because the attention seam and the
decoder read them off the mesh in context: the sequence axis attention splits
its queries over, and the tensor and stage axes, which hold a width and a
pipeline stage and never a row."""

MESH_AXES = (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS, SEQUENCE_AXIS, STAGE_AXIS)
"""Every mesh's axes, in the order `dew.training.build_mesh` lays them out."""

BATCH_AXES = (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS)
"""The mesh axes a batch's rows split over, in mesh order. Sequence holds a
slice of the positions, stage the pipeline's stages that hand one batch's
microbatches along, and tensor the widths of Megatron's split, so every
tensor shard computes its share of the width for every row."""


class LayoutRefused(ValueError):
    """A mesh layout that a model, an objective or a device set does not run,
    by design. The message names the axis, why it cannot hold what it would
    split, and a layout that runs. Any other error on a layout is a defect.
    It is a ValueError, so a caller that catches the one catches the other."""

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
# 'embed', the residual width, is the side those matmuls share with every
# norm, residual add, rotary rotation and the loss. It stays whole on the
# tensor axis and keeps fsdp, because sharding it would put a collective
# between every pair of sublayers, the cost tensor parallelism is arranged
# to avoid. In a kernel that holds both, the residual width takes fsdp and
# the Megatron width tensor alone, a split in two dimensions, as MaxText's
# rules make it; the mlp and attention widths name tensor first and fsdp
# second, and the heads take tensor through the pairs at the end. Split
# over fsdp and tensor together while the residual width stayed whole, the
# mlp's and o_proj's widths had GSPMD gather a block's activations over
# fsdp and repeat their products on both devices of every pair: 1.21 times
# one device's matmul FLOPs for layout_parity's dense decoder under
# fsdp2_tensor2. A width no residual shares an array with, the vocabulary
# of the embedding table, composes the two axes and splits fsdp times
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
"""A `[batch, length, width]` activation between sublayers, and a sublayer's
input: under a tensor axis every tensor shard reads every row it computes
its share of the width for."""
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


DECLARED: dict[Suffix, LogicalAxes] = {}
"""Every decorated module's declarations, merged."""

HEURISTIC: set[Suffix] = set()
"""Runs of name patterns whose parameters take the shape heuristic on purpose."""

@dataclasses.dataclass
class Schedule:
    """The microbatch count a step feeds the stage axis, and whether a model
    in the step ran a pipeline over it, which `microbatches` notes as the
    pipeline reads the count."""

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
    """A mesh axis's interconnect as the trainer measured it when it placed a
    step (`dew.training.distributed.link_bandwidth`), one device's dense bf16
    peak, and whether a projection in the step ran split over the axis, which
    `down_projection` and `split_positions` note as they decide."""

    bytes_per_second: float | None
    """What one device receives a second in an all-gather over the axis:
    (N - 1) / N of the result, over the time it took. None where nothing was
    measured: a CPU mesh, or a device the peak table does not name."""
    flops_per_second: float | None
    """One device's dense bf16 peak, None for hardware the peak table does
    not name (`dew.telemetry.instrumentation.peak_flops`)."""
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
    else `RESIDUAL`, every tensor shard on every token.

    At DeepSeek-V3's widths in bf16 (7168 into 1536 + 512 + 64) and 16384
    tokens the link must move 20.3 GB/s for an RTX 3090, which an NVLink
    pair's 31.0 meets and a PCIe 3.0 pair's 5.8 does not (4x RTX 3090, PCIe
    3.0, one host), and 283 GB/s for an H100, under NVLink 4's nominal 450
    a direction; at 4096 tokens an H100 needs 524."""
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


def split_positions[T](x: jax.Array, features: int, project: Callable[[jax.Array], T]) -> T:
    """`project(x)`, a projection of a `[batch, length, width]` input whose
    positions the sequence axis does not divide to `features` output
    features in all, run on each sequence shard's share of the positions
    where the sequence axis's link pays for it (`_spreads`, the rule
    `down_projection` decides by): `x` padded with zero rows to a multiple
    of the shard count, so `project`'s own constraints split its positions,
    and every array `project` returns cut back to `length`. `project` must
    work position by position, as a projection without a sequence mixer
    does. With the sequence axis at one, a length it divides, or a link that
    does not pay, it runs on `x` as it is.

    A cross-attention's context is such an input: SD's 77 text tokens. Left
    whole, every sequence shard projects all of them into keys and values,
    and the conditional UNet's step on sequence4 computed 1.57 times one
    device's FLOPs on a CPU mesh, which splits them; split, the pad rows'
    projections are what is repeated, 3 of 80 rows at SD's length. The
    projections are small beside the step and the sum of their weight
    gradients the split adds is not, so a GPU's link rarely pays: at SDXL's
    widths in bf16 (2048 into 2 x 1280) and 8 rows a group, an H100 would
    need 2.47 TB/s, including the padded positions' extra work."""
    shards = sequence_shards()
    length = x.shape[1]
    if shards == 1 or length % shards == 0:
        return project(x)
    padded_length = length + -length % shards
    mesh = jax.sharding.get_abstract_mesh()
    rows = x.shape[0] / math.prod(mesh.shape[axis] for axis in row_axes(x.shape[0]))
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
    """The spec `rules` give an array of `shape` whose dimensions `axes`
    names, on `mesh`: the rules and the mesh in context by default.

    A mesh axis of size 1 shards nothing, and a manual one belongs to the
    `shard_map` in context, so both are dropped from the rules before they
    are read: such an axis claims no dimension, and a name whose rule named
    only such axes takes its next rule. With no tensor axis, 'mlp' passes
    its tensor rule to fsdp. A rule whose axes do not divide its dimension
    evenly cannot split it, so that rule is set aside for this array and the
    name takes its next rule, or none, and the axis goes to the next
    dimension that names it. An odd vocabulary shards the embedding on its
    width and keeps the table in the layout.
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
    under a pipeline's vmap the stage axis takes the vmapped dimension."""
    if jax.sharding.get_abstract_mesh().empty:
        return x
    return jax.lax.with_sharding_constraint(x, logical_spec(axes, x.shape))


def logical_axes(declared: Mapping[Suffix, LogicalAxes], *,
                 heuristic: Iterable[Suffix] = ()):
    """Declare the parameter axes of the modules `cls` creates."""
    declared = {tuple(suffix): tuple(axes) for suffix, axes in declared.items()}
    heuristic = tuple(tuple(suffix) for suffix in heuristic)
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
        for suffix, axes in declared.items():
            held = DECLARED.get(suffix)
            if held is not None and held != axes:
                raise ValueError(
                    f"{'/'.join(suffix)} is declared {axes} by {cls.__name__} and "
                    f"{held} elsewhere; one module path has one set of axes")
            DECLARED[suffix] = axes
        HEURISTIC.update(heuristic)
        cls.__logical_axes__ = declared
        cls.__heuristic_axes__ = heuristic
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


def declared_axes(path, ndim: int) -> LogicalAxes | None:
    """The declared axes of the parameter at `path`, or None for an unnamed one.

    A declaration names a module, whose parameters share its axes, or one
    parameter under its module, for a module whose leaves have different
    axes (GPT OSS's fused experts). The parameter's own path is tried first.
    """
    names = parameter_path(path)
    suffix = _matching(DECLARED, names) or _matching(DECLARED, names[:-1])
    if suffix is None:
        return None
    axes = DECLARED[suffix]
    if ndim > len(axes):
        raise ValueError(
            f"{'/'.join(suffix)} is declared {axes}, which cannot name the "
            f"{ndim} dimensions of {'/'.join(parameter_path(path))}")
    return axes[len(axes) - ndim:]


def is_heuristic(path) -> bool:
    """Whether the parameter at `path` is one a module left to the shape heuristic."""
    names = parameter_path(path)
    for pattern in HEURISTIC:
        for start in range(len(names) - len(pattern) + 1):
            if all(fnmatch.fnmatchcase(name, glob)
                   for name, glob in zip(names[start:], pattern, strict=False)):
                return True
    return False
