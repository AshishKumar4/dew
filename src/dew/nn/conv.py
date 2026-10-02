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

The partitioner bugs are reported at
https://github.com/openxla/xla/issues/49382 and drafted in
Dew issue #4 (https://github.com/AshishKumar4/dew/issues/4).

A dilated 3x3 depthwise convolution runs on CUDA as an undilated one over
its interleaved grids (`_polyphase_depthwise_3x3`): cuDNN's dilated grouped
forward and weight-gradient kernels are slow, its dilation-one depthwise
kernels are not. On the RTX 4080 at 16x16x768 and batch 16 the bf16
forward and VJP take 0.097 ms at dilation 2 and 0.089 at dilation 3,
against 1.6 ms through the dilated convolution and 0.31-0.33 through the
fp32 shifted products this replaced; the 176M hybrid DiT's bf16 step at
that batch runs 70.2 ms against 75.3. cuDNN's fp32 depthwise kernels are
slower, and the same step in fp32 runs 113.7 ms against 109.7 with the
shifted products. The accumulation stays fp32,
including for bf16 inputs; ordinary JAX differentiation keeps forward-mode
and higher-order derivatives. Other backends retain lax.
`tools/benchmark_depthwise.py` measures each path's forward and backward
separately, as well as the hybrid DiT training step.
Under Qwix quantization (`dew.training.quantization`) a convolution that a
rule quantizes goes through Qwix's provider, which computes it with lax, and
so does not get this depthwise speedup; one that no rule quantizes, such as
an excluded or weight-only one, computes here as unwrapped.
"""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PromoteDtypeFn
from flax.typing import ConvGeneralDilatedT
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike

from .sharding import SEQUENCE_AXIS, STAGE_AXIS, logical_spec, mesh_axes

# A linear boundary must survive in tangents and cotangents too: dropping it
# from either leaves the same convolution chain open to the faulty rewrite.
_barrier_p = core.Primitive("dew_conv_barrier")
_barrier_p.def_impl(lambda x: x)
_barrier_p.def_abstract_eval(lambda x: x)
mlir.register_lowering(
    _barrier_p, mlir.lower_fun(jax.lax.optimization_barrier, multiple_results=False))
ad.deflinear(_barrier_p, lambda cotangent: (_barrier_p.bind(cotangent),))
batching.defvectorized(_barrier_p)


def _barrier(x: jax.Array) -> jax.Array:
    return _barrier_p.bind(x)


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


def _polyphase_depthwise_3x3(lhs: jax.Array, rhs: jax.Array, dilation: int) -> jax.Array:
    """The dilated depthwise convolution as an undilated one over its
    `dilation ** 2` interleaved grids.

    A pixel's dilated taps are its neighbours in the grid of pixels that
    share its row and column residues modulo `dilation`, so the image splits
    into those grids along the batch, each grid takes the dilation-1
    convolution with its own zero border, and the grids interleave back.
    The image is padded with zeros to a multiple of `dilation` first, which
    are the zeros 'SAME' padding reads there anyway. The convolution runs at
    full precision, so an fp32 model keeps fp32 accumulation and bf16 rounds
    once, from fp32."""
    batch, height, width, features = lhs.shape
    rows, columns = -(-height // dilation), -(-width // dilation)
    padded = jnp.pad(lhs, ((0, 0), (0, rows * dilation - height), (0, columns * dilation - width), (0, 0)))
    grids = padded.reshape(batch, rows, dilation, columns, dilation, features).transpose(0, 2, 4, 1, 3, 5)
    output = jax.lax.conv_general_dilated(
        grids.reshape(batch * dilation * dilation, rows, columns, features), rhs, (1, 1), 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=features,
        precision=jax.lax.Precision.HIGHEST)
    output = output.reshape(batch, dilation, dilation, rows, columns, features).transpose(0, 3, 1, 4, 2, 5)
    return output.reshape(batch, rows * dilation, columns * dilation, features)[:, :height, :width]


def _conv_general_dilated(
        lhs: jax.Array, rhs: jax.Array, window_strides: Sequence[int],
        padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None,
        rhs_dilation: Sequence[int] | None = None,
        dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
        feature_group_count: int = 1, batch_group_count: int = 1,
        precision: jax.lax.PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        out_sharding: jax.sharding.NamedSharding | None = None) -> jax.Array:
    def convolve(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.lax.conv_general_dilated(
            x, w, window_strides, padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers, feature_group_count=feature_group_count,
            batch_group_count=batch_group_count, precision=precision,
            preferred_element_type=preferred_element_type, out_sharding=out_sharding)

    dilation = (1, 1) if rhs_dilation is None else tuple(rhs_dilation)
    if (lhs.ndim == rhs.ndim == 4 and rhs.shape[:3] == (3, 3, 1)
            and lhs.shape[-1] == rhs.shape[-1] == feature_group_count
            and tuple(window_strides) == (1, 1)
            and (lhs_dilation is None or tuple(lhs_dilation) == (1, 1))
            and dilation in ((1, 1), (2, 2), (3, 3))
            and (padding == 'SAME' or padding == ((dilation[0], dilation[0]),) * 2)
            and batch_group_count == 1 and preferred_element_type is None and out_sharding is None
            and lhs.dtype == rhs.dtype and lhs.dtype in (jnp.float32, jnp.bfloat16)
            and jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
            == jax.lax.ConvDimensionNumbers((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))):
        return jax.lax.platform_dependent(
            lhs, rhs,
            cuda=convolve if dilation[0] == 1 else lambda x, w: _polyphase_depthwise_3x3(x, w, dilation[0]),
            default=convolve)
    return convolve(lhs, rhs)


class Conv(nn.Conv):
    """Flax's convolution with safe placement and backend-specific depthwise work.

    The kernel is placed where Flax reads its promoted operands, through
    `promote_dtype`. Parameters and names remain Flax's, so checkpoints do
    not change. Other grouped and ordinary convolutions remain lax calls.
    """

    promote_dtype: PromoteDtypeFn = _promoted_whole
    conv_general_dilated: ConvGeneralDilatedT | None = _conv_general_dilated

    def __call__(self, inputs: jax.Array) -> jax.Array:
        spatial = 1 if isinstance(self.kernel_size, int) else len(self.kernel_size)
        inputs = _unreplicated(inputs, spatial)
        if spatial == 2 and self.strides is not None:
            strided = (self.strides > 1 if isinstance(self.strides, int)
                       else any(stride > 1 for stride in self.strides))
            if strided:
                inputs = jax.lax.platform_dependent(inputs, tpu=_barrier, default=lambda x: x)
        return _unreplicated(super().__call__(inputs), spatial)
