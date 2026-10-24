"""The mesh facts every layer reads, below the models and the data alike: the names of a mesh's axes
(`MESH_AXES`), the ones a batch's rows split over (`BATCH_AXES`), the default table of rules that
places each logical axis on them (`DEFAULT_RULES`, or a layout's in flax's context, `axis_rules`),
the spec those rules give an array (`logical_spec`), which axes a batch's rows split over on a mesh
(`row_axes`, `batch_axes`), and the refusal of a layout that does not run (`LayoutRefused`). The
models' layout (`dew.nn.sharding`) constrains and maps their arrays by these, and a run's data shares
(`dew.data.dataset.DataPartition.of`) read the rows' axes here."""

from __future__ import annotations

import math

import jax
from flax import linen as nn
from flax.linen import spmd

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


type LogicalAxes = tuple[str | None, ...]
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
# fsdp alone stores a split: a parameter it splits is gathered whole for
# every use, so which dimension it takes changes no product, only how the
# shards lie in memory. Among the dimensions a rule names it alone for, it
# takes the array's leading one that no other axis took (`logical_spec`),
# whatever the rule order: each shard is then one block of the row-major
# array, so its gather concatenates and its gradient's reduce-scatter cuts,
# as PyTorch's FSDP2 splits every parameter on dim 0 (`fully_shard`).
# Split on the mlp width, the gate and up projections transposed their
# gradients for every update: Llama-3.1-8B over FSDP 2 on two H100 SXMs
# stepped in 537.8 ms, and in 488.0 split on the residual width.
# A name whose rule pairs fsdp with tensor (the vocabulary, the heads) splits
# for a computation, and its rule order stands.
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
    keeps the table in the layout. fsdp, placed by a name whose first rule
    names it alone, then moves to the leading dimension of such a name that
    no other axis took and that it divides (`DEFAULT_RULES`).
    """
    rules = axis_rules() if rules is None else rules
    mesh = jax.sharding.get_abstract_mesh() if mesh is None else mesh

    def splitting(assignment: MeshAxes) -> tuple[str, ...]:
        return tuple(axis for axis in mesh_axes(assignment)
                     if axis not in mesh.manual_axes and mesh.shape.get(axis, 1) > 1)

    # A rule of None keeps its name whole and stands; one whose axes all
    # split nothing is dropped. `stored` names those whose first rule left
    # names fsdp alone.
    left: list[tuple[str, MeshAxes]] = []
    stored: set[str] = set()
    for name, assignment in rules:
        used = splitting(assignment)
        if assignment is not None and not used:
            continue
        if all(name != kept for kept, _ in left) and mesh_axes(assignment) == (FSDP_AXIS,):
            stored.add(name)
        left.append((name, None if assignment is None else used[0] if len(used) == 1 else used))
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
    # fsdp alone takes the leading such dimension no other axis took (`DEFAULT_RULES`).
    leading = [dim for dim, (name, used, size) in enumerate(zip(axes, assigned, shape, strict=True))
               if name in stored and used in ((), (FSDP_AXIS,)) and size % mesh.shape.get(FSDP_AXIS, 1) == 0]
    if (FSDP_AXIS,) in assigned and assigned.index((FSDP_AXIS,)) in leading:
        assigned[assigned.index((FSDP_AXIS,))], assigned[leading[0]] = (), (FSDP_AXIS,)
    entries = [used[0] if len(used) == 1 else used or None for used in assigned]
    while entries and entries[-1] is None:
        entries.pop()
    return jax.sharding.PartitionSpec(*entries)


def row_axes(batch: int, *, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None
             ) -> tuple[str, ...]:
    """The mesh axes a `shard_map` over the sequence axis splits `batch` rows
    over: the ones `activation_batch` takes for `batch` rows on `mesh`, the
    mesh in context by default (`logical_spec`). A batch too small for them is
    computed alike on those axes' shards, the way GSPMD replicates a dimension
    it cannot split."""
    spec = logical_spec(("activation_batch",), (batch,), mesh=mesh)
    return mesh_axes(spec[0]) if spec else ()


def batch_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple[str, ...]:
    """The axes of `mesh` a batch's rows split over: every axis
    `activation_batch` takes, each of which divides `mesh.size` rows."""
    return row_axes(mesh.size, mesh=mesh)


__all__ = ["BATCH_AXES", "DATA_AXIS", "DEFAULT_RULES", "EXPERT_AXIS", "FSDP_AXIS", "MESH_AXES",
           "SEQUENCE_AXIS", "STAGE_AXIS", "TENSOR_AXIS", "LayoutRefused", "LogicalAxes", "LogicalAxisRules",
           "MeshAxes", "axis_rules", "batch_axes", "logical_spec", "mesh_axes", "row_axes"]
