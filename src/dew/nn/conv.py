"""The convolution Dew's models build on: `flax.linen.Conv` with its input,
output and kernel placed so that XLA partitions it correctly.

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

The same partitioner computes a convolution's output wrong where a mesh
axis splits its image rows or columns, so that each shard needs a halo of
its neighbours', and another axis splits the kernel's input features: a
3x3 convolution of an 8x8 image, its rows over one axis of a 2x2 mesh and
the kernel's input features over the other, came out 23.9 off one
device's (jax 0.11.2, CPU). A 1x1 kernel, a kernel whole or split on its
output features, or whole image rows came out right. A layout that splits a kernel over fsdp stores it on
whichever width its heuristic picks, so the kernel is used whole, as fully
sharded data parallelism gathers a layer's weights before it computes.

On TPU, XLA's space-to-batch rewrite also corrupts a convolution feeding
a strided 2D convolution at small batches (Dew issue #5). A barrier at the
strided convolution's input fixes both the forward and the VJP. It applies
regardless of the global batch: partitioning may make a shard's batch small.
CPU and GPU lower the input unchanged. The barrier's derivative and batching
rules are identities; JAX's primitive defines neither.

`Conv` exists only to work around these bugs. The partitioner bugs are
reported at https://github.com/openxla/xla/issues/49382 and drafted in
~/.cache/dew/verification-evidence/upstream-reports/xla-conv-halo-kernel-split/.
When XLA fixes these lowerings, every model can use `flax.linen.Conv` again.
"""

import math

import jax
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PromoteDtypeFn
from jax.custom_batching import custom_vmap
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike

from .sharding import SEQUENCE_AXIS, STAGE_AXIS, logical_spec, mesh_axes


@custom_vmap
def _barrier_value(x: jax.Array) -> jax.Array:
    return jax.lax.optimization_barrier(x)


@_barrier_value.def_vmap
def _barrier_batch(axis_size, in_batched, x):
    del axis_size
    return _barrier_value(x), in_batched[0]


@jax.custom_jvp
def _barrier(x: jax.Array) -> jax.Array:
    return _barrier_value(x)


@_barrier.defjvp
def _barrier_jvp(primals, tangents):
    return _barrier(primals[0]), tangents[0]


def _automatic_axes(mesh: jax.sharding.AbstractMesh) -> list[str]:
    """The mesh axes above size one that GSPMD places: neither a
    `shard_map`'s manual axes nor the stage axis, which a pipeline's stage
    vmap holds."""
    return [axis for axis in mesh.axis_names if mesh.shape[axis] > 1
            and axis not in mesh.manual_axes and axis != STAGE_AXIS]


def _promoted_whole(*arrays: jax.Array | None, dtype: DTypeLike | None = None,
                    inexact: bool = True) -> list[jax.Array | None]:
    """flax's dtype promotion of a convolution's input, kernel and bias,
    with the kernel then placed whole on every device (the module
    docstring's second bug). Where no automatic mesh axis is above size
    one, the kernel is left as it is."""
    inputs, kernel, bias = promote_dtype(*arrays, dtype=dtype, inexact=inexact)
    if _automatic_axes(jax.sharding.get_abstract_mesh()):
        kernel = jax.lax.with_sharding_constraint(kernel, P())
    return [inputs, kernel, bias]


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
    automatic = _automatic_axes(mesh)
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
    """Flax's convolution with the placement and TPU input barriers above.

    The kernel is placed where Flax reads its promoted operands, through
    `promote_dtype`. Parameters and names remain Flax's, so checkpoints do
    not change. Remove the workarounds when XLA fixes the lowerings.
    """

    promote_dtype: PromoteDtypeFn = _promoted_whole

    def __call__(self, inputs: jax.Array) -> jax.Array:
        spatial = 1 if isinstance(self.kernel_size, int) else len(self.kernel_size)
        inputs = _unreplicated(inputs, spatial)
        if spatial == 2 and self.strides is not None:
            strided = (self.strides > 1 if isinstance(self.strides, int)
                       else any(stride > 1 for stride in self.strides))
            if strided:
                inputs = jax.lax.platform_dependent(inputs, tpu=_barrier, default=lambda x: x)
        return _unreplicated(super().__call__(inputs), spatial)
