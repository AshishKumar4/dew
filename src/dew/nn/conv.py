"""The convolution Dew's models build on: `flax.linen.Conv` with its input,
output and kernel placed so that XLA partitions it correctly.

XLA's SPMD partitioner (jax 0.11.2) scales a convolution's kernel gradient by
a power of two where a mesh axis holds the input or output replicated: it
splits the gradient over one axis and all-reduces over every device, so the
replicas add the same partial again (a batch over data with output rows over
sequence doubles it). The gradient is right when input and output are split
over every mesh axis or none; 1x1 convolutions, dot_generals and the input
and bias gradients are right throughout. The same partitioner computes the
output wrong where one axis splits the image rows or columns (a halo) and
another the kernel's input features (23.9 off on an 8x8 3x3 over a 2x2 CPU
mesh), so a kernel is used whole, as FSDP gathers a layer's weights. Both
are https://github.com/openxla/xla/issues/49382 and
https://github.com/AshishKumar4/dew/issues/4.

On TPU, XLA's space-to-batch rewrite corrupts a convolution feeding a strided
2D convolution at small batches (Dew issue #5), so a barrier sits at the
strided convolution's input, whatever the global batch, since a shard's may
be small; its derivative and batching rules are identities.

A dilated 3x3 depthwise convolution runs on CUDA as an undilated one over
its interleaved grids (`_polyphase_depthwise_3x3`), since cuDNN's dilated
grouped kernels are slow: on the RTX 4080 at 16x16x768, batch 16, bf16
forward and VJP take 0.097 ms (dilation 2) against 1.6 ms dilated, and the
176M hybrid DiT's bf16 step runs 70.2 ms against 75.3
(`tools/benchmark_depthwise.py`). Accumulation stays fp32 and ordinary JAX
differentiation keeps higher-order derivatives; other backends use lax, as
does a convolution a Qwix rule quantizes (`dew.training.quantization`).
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

from .sharding import SEQUENCE_AXIS, STAGE_AXIS, auto_part, logical_spec, mesh_axes

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
    `shard_map`'s manual axes, an Explicit axis, which an array's type
    places, nor the stage axis, which a pipeline's stage vmap holds."""
    return [axis for axis in mesh.axis_names if mesh.shape[axis] > 1 and axis not in mesh.manual_axes
            and axis not in mesh.explicit_axes and axis != STAGE_AXIS]


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
    axis above size one splits it: the first batch dimension as
    `activation_batch` splits rows, then every other automatic axis the first
    later dimension it divides, the sequence axis first (a patch embedding's
    rows already sit where its tokens go) and the features last; an axis that
    divides nothing replicates. The stage axis is left to the pipeline's vmap,
    and with no automatic axis above one `x` is left as it is."""
    mesh = jax.sharding.get_abstract_mesh()
    automatic = _automatic_axes(mesh)
    if not automatic:
        return x
    batched = x.ndim > spatial + 1
    rows = auto_part(logical_spec(("activation_batch",), x.shape[:1]), mesh) if batched else P()
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


def _shifted_depthwise_3x3(lhs: jax.Array, rhs: jax.Array, dilation: int) -> jax.Array:
    """The depthwise convolution as its nine shifted products, summed in the
    kernel's row-major order in fp32 and rounded once to the input's dtype.

    The optimization barrier rounds each product to fp32 before it is
    added, where XLA would otherwise contract it into a fused multiply-add.
    For more than 16 features that is XLA:CPU's library convolution's
    arithmetic (YNNPACK, jax 0.11.2), bit for bit in fp32 and bf16, which on
    the hybrid DiT's 2 x 16 x 16 x 768 maps takes 6.4 ms a call against 1.4
    here; it sums 16 features or fewer otherwise, so those keep the
    convolution. If a jax or YNNPACK release changes that summation,
    tests/test_depthwise_conv.py's
    test_cpu_depthwise_is_the_library_convolution_bit_for_bit fails."""
    height, width = lhs.shape[1:3]
    padded = jnp.pad(lhs, ((0, 0), (dilation, dilation), (dilation, dilation), (0, 0))).astype(jnp.float32)
    kernel = rhs.astype(jnp.float32)
    products = jax.lax.optimization_barrier([
        padded[:, i * dilation:i * dilation + height, j * dilation:j * dilation + width, :] * kernel[i, j, 0]
        for i in range(3) for j in range(3)])
    total = products[0]
    for product in products[1:]:
        total = total + product
    return total.astype(lhs.dtype)


def _materialized_depthwise_3x3(lhs: jax.Array, rhs: jax.Array, dilation: int) -> jax.Array:
    """The depthwise convolution as its nine shifted products in fp32, with
    its fp32 input, kernel and output held in memory (`_barrier`, in the
    cotangents too), so XLA fuses the products and their weight reductions
    into kernels of their own rather than into a neighbour's.

    CUDA runs a dilated one in this form in fp32 and in the polyphase form
    in bf16: the 176M hybrid DiT's training step on an RTX 4080, ms, with
    each form for its dilation-2 and -3 layers (docs/performance.md):

        dtype   batch   polyphase   this form
        bf16      16       60.1        65.4
        bf16      32      100.6       107.0
        fp32      16      105.9       101.6
        fp32      32      265.6       254.5

    In fp32 cuDNN runs the polyphase convolutions with its grouped direct
    kernels, 3.8 ms of the 5.2 ms a b16 step spent in those layers, and the
    interleaving transposes 1.3 ms more."""
    height, width = lhs.shape[1:3]
    border = ((0, 0), (dilation, dilation), (dilation, dilation), (0, 0))
    padded = jnp.pad(_barrier(lhs.astype(jnp.float32)), border)
    kernel = _barrier(rhs.astype(jnp.float32))
    output = jnp.zeros(lhs.shape, jnp.float32)
    for row in range(3):
        for column in range(3):
            top, left = row * dilation, column * dilation
            output = output + padded[:, top:top + height, left:left + width, :] * kernel[row, column, 0, :]
    return _barrier(output).astype(lhs.dtype)


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
        # A dilated one on CUDA takes the form measured faster for its dtype
        # (`_materialized_depthwise_3x3`).
        dilated = _polyphase_depthwise_3x3 if lhs.dtype == jnp.bfloat16 else _materialized_depthwise_3x3
        return jax.lax.platform_dependent(
            lhs, rhs,
            cuda=convolve if dilation[0] == 1 else lambda x, w: dilated(x, w, dilation[0]),
            cpu=((lambda x, w: _shifted_depthwise_3x3(x, w, dilation[0]))
                 if feature_group_count > 16 else convolve),
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
