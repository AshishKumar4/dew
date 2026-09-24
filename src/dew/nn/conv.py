"""The convolution Dew's models build on: `flax.linen.Conv` with its input and
output placed so that XLA partitions its kernel gradient correctly.

XLA's SPMD partitioner (jax 0.11.2) scales a convolution's kernel gradient
by a power of two in some layouts where a mesh axis holds the convolution's
input or output replicated. In the case we traced, it split the kernel
gradient's computation over one axis and all-reduced the partial kernels
over every device, so the devices along the replicated axis added the same
partial sum again. An input batch split over data with the output's image
rows split over sequence doubles it, and so does a grouped convolution's
batch split over one axis beside another. In our checks the gradient came
out right whenever the input and the output were split over every mesh axis
or over none, and a 1x1 convolution and every dot_general came out right in
every layout we tried. The input and bias gradients were right throughout.

`Conv` exists only to work around that bug, reported as
https://github.com/openxla/xla/issues/49382. When XLA partitions the kernel
gradient correctly, this module goes and every model uses `flax.linen.Conv`
again.
"""

import math

import jax
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from .sharding import SEQUENCE_AXIS, STAGE_AXIS, logical_spec, mesh_axes


def _unreplicated(x: jax.Array, spatial: int) -> jax.Array:
    """`x` `[*batch, *spatial, features]` constrained so that every mesh
    axis above size one splits it.

    The first batch dimension splits as `activation_batch` splits rows
    (`logical_spec`). Every other automatic mesh axis above size one then
    splits the first dimension after it that it divides evenly, the features
    last. The sequence axis goes first, since a sequence of positions lays
    its rows out along the first spatial dimension, so a patch embedding's
    output already sits where its tokens go. If an axis divides no dimension,
    `x` is replicated over the whole mesh. The stage axis is left out:
    inside a pipeline the stage vmap holds it, and the trainer refuses a
    stage axis for a model without one. With no automatic axis above size
    one (no mesh, one device, or inside a `shard_map` that holds them all)
    `x` is left as it is."""
    mesh = jax.sharding.get_abstract_mesh()
    automatic = [axis for axis in mesh.axis_names if mesh.shape[axis] > 1
                 and axis not in mesh.manual_axes and axis != STAGE_AXIS]
    if not automatic:
        return x
    batched = x.ndim > spatial + 1
    rows = logical_spec(("activation_batch",), x.shape[:1]) if batched else P()
    entries: list[list[str]] = [list(mesh_axes(rows[0])) if rows else []]
    entries += [[] for _ in x.shape[1:]]
    for axis in sorted(automatic, key=lambda axis: axis != SEQUENCE_AXIS):
        if axis in entries[0]:
            continue
        for dimension in range(1 if batched else 0, x.ndim):
            held = math.prod(mesh.shape[other] for other in entries[dimension])
            if x.shape[dimension] % (held * mesh.shape[axis]) == 0:
                entries[dimension].append(axis)
                break
        else:
            return jax.lax.with_sharding_constraint(x, P())
    return jax.lax.with_sharding_constraint(
        x, P(*(tuple(entry) if entry else None for entry in entries)))


class Conv(nn.Conv):
    """`flax.linen.Conv`, with its input and output constrained by
    `_unreplicated` under a mesh, for the XLA partitioner bug in the module
    docstring. The parameters, their names and the result are flax's, so
    checkpoints and one-device runs are unchanged. Remove the class when XLA
    fixes the bug."""

    def __call__(self, inputs: jax.Array) -> jax.Array:
        spatial = 1 if isinstance(self.kernel_size, int) else len(self.kernel_size)
        return _unreplicated(super().__call__(_unreplicated(inputs, spatial)), spatial)
