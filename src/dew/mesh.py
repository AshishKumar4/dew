"""The mesh facts every layer reads below the models and the data alike: the names of a mesh's
axes (`MESH_AXES`), the ones a batch's rows split over (`BATCH_AXES`), the rule entries that place a
batch's rows (`BATCH_RULES`) and which axes a batch's rows split over on a mesh (`row_axes`,
`batch_axes`), read from the rules a layout puts in flax's context or these defaults. The models'
layout (`dew.nn.sharding`) places everything else by the full rule table, which composes these
entries; a run's data shares (`dew.data.dataset.DataPartition.of`) read the rows' axes here."""

from __future__ import annotations

import math

import jax
from flax import linen as nn

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


type MeshAxes = str | tuple[str, ...] | None
type LogicalAxisRules = tuple[tuple[str, MeshAxes], ...]


BATCH_RULES: LogicalAxisRules = (
    ("activation_batch", BATCH_AXES),
    ("activation_batch", (DATA_AXIS, EXPERT_AXIS)),
    ("activation_batch", DATA_AXIS),
)
"""How the default rules (`dew.nn.sharding.DEFAULT_RULES`) place a batch's rows: over every batch
axis, and rows too few for them over the first ones."""


def mesh_axes(assignment: MeshAxes) -> tuple[str, ...]:
    """Read one entry of a spec or a rule as the mesh axes it names."""
    if assignment is None:
        return ()
    return (assignment,) if isinstance(assignment, str) else tuple(assignment)


def row_axes(batch: int, *, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None
             ) -> tuple[str, ...]:
    """The mesh axes `batch` rows split over on `mesh`, the mesh in context by default: those of the
    first `activation_batch` entry of the rules in context (a layout's, `flax.linen.logical_axis_rules`)
    or of `BATCH_RULES` that splits them, as `dew.nn.sharding.logical_spec` places an array's
    leading rows. An axis of size 1 or held manual splits nothing and drops from its entry, an entry
    left with none is passed over, one whose axes do not divide `batch` rows is set aside for the
    next, and an entry of None keeps the rows whole. A batch too small for them is computed alike
    on those axes' shards, the way GSPMD replicates a dimension it cannot split."""
    mesh = jax.sharding.get_abstract_mesh() if mesh is None else mesh
    for name, assignment in tuple(nn.get_logical_axis_rules()) or BATCH_RULES:
        if name != "activation_batch":
            continue
        if assignment is None:
            return ()
        used = tuple(axis for axis in mesh_axes(assignment)
                     if axis not in mesh.manual_axes and mesh.shape.get(axis, 1) > 1)
        if used and not batch % math.prod(mesh.shape[axis] for axis in used):
            return used
    return ()


def batch_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple[str, ...]:
    """The axes of `mesh` a batch's rows split over: every axis
    `activation_batch` takes, each of which divides `mesh.size` rows."""
    return row_axes(mesh.size, mesh=mesh)


__all__ = ["BATCH_AXES", "BATCH_RULES", "DATA_AXIS", "EXPERT_AXIS", "FSDP_AXIS", "MESH_AXES", "SEQUENCE_AXIS",
           "STAGE_AXIS", "TENSOR_AXIS", "LogicalAxisRules", "MeshAxes", "batch_axes", "mesh_axes", "row_axes"]
