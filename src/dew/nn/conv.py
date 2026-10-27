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

A dilated 3x3 depthwise convolution runs on CUDA in the form measured faster
for its dtype, since cuDNN's dilated grouped kernels are slow: in bf16 as an
undilated one over its interleaved grids (`_polyphase_depthwise_3x3`), whose
forward and VJP on the RTX 4080 at 16x16x768, batch 16, take 0.097 ms at
dilation 2 against 0.31 ms as nine shifted products, the 176M hybrid DiT's
bf16 step 70.2 ms against 75.3; in fp32 as those nine products held in memory
(`_materialized_depthwise_3x3`), where cuDNN's grouped direct kernels make the
polyphase form slower (docs/performance.md, `tools/benchmark_depthwise.py`).
Accumulation stays fp32 and ordinary JAX differentiation keeps higher-order
derivatives; other backends use lax, as does a convolution a Qwix rule
quantizes (`dew.quantization`).
"""

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PromoteDtypeFn, canonicalize_padding
from flax.typing import ConvGeneralDilatedT
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike

from dew.mesh import SEQUENCE_AXIS, STAGE_AXIS, logical_spec, mesh_axes

from .flops import conv_flops
from .precision import biased, float32_products, products_dtype
from .sharding import auto_part, in_pipeline_stage

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
    places, nor, inside a pipeline's stage (`in_pipeline_stage`), the stage
    axis, which the stage's vmap holds. Outside one, a convolution runs on
    every stage, and the stage axis replicating its input and output doubled
    its kernel gradient as the first bug does (a JEPA's patch embedding over
    stage 2 and sequence 2, tests/test_stage_blocks.py), so it splits them
    like any other axis."""
    return [axis for axis in mesh.axis_names if mesh.shape[axis] > 1 and axis not in mesh.manual_axes
            and axis not in mesh.explicit_axes and not (axis == STAGE_AXIS and in_pipeline_stage())]


def _whole(kernel: jax.Array) -> jax.Array:
    """`kernel` placed whole on every device (the module docstring's second
    bug). Where no automatic mesh axis is above size one, it is left as it is."""
    if _automatic_axes(jax.sharding.get_abstract_mesh()):
        return jax.lax.with_sharding_constraint(kernel, P())
    return kernel


def _promoted_whole(*arrays: jax.Array | None, dtype: DTypeLike | None = None,
                    inexact: bool = True) -> list[jax.Array | None]:
    """flax's dtype promotion of a convolution's input, kernel and bias,
    with the kernel then placed whole (`_whole`)."""
    inputs, kernel, bias = promote_dtype(*arrays, dtype=dtype, inexact=inexact)
    return [inputs, _whole(kernel), bias]


def _unreplicated(x: jax.Array, spatial: int) -> jax.Array:
    """`x` `[*batch, *spatial, features]` constrained so that every mesh
    axis above size one splits it: the first batch dimension as
    `activation_batch` splits rows, then every other automatic axis the first
    later dimension it divides, the sequence axis first (a patch embedding's
    rows already sit where its tokens go) and the features last; an axis that
    divides nothing replicates. Inside a pipeline's stage the stage axis is
    left to its vmap, and with no automatic axis above one `x` is left as it is."""
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


def _taps(padded: jax.Array, kernel: jax.Array, dilation: int, height: int, width: int) -> list[jax.Array]:
    """The depthwise convolution's nine fp32 products, in the kernel's row-major order: each tap of the
    padded input, shifted by its row and column times `dilation`, times the kernel's weight there. Each
    schedule (`_shifted_depthwise_3x3`, `_materialized_depthwise_3x3`) sums them as it rounds."""
    return [padded[:, row * dilation:row * dilation + height, column * dilation:column * dilation + width, :]
            * kernel[row, column, 0] for row in range(3) for column in range(3)]


def _accumulated(total: jax.Array, products: Sequence[jax.Array]) -> jax.Array:
    """`total` plus each of `products` in order, one fp32 add at a time: the depthwise schedules'
    accumulation, each giving its own seed (the first product, or zeros)."""
    for product in products:
        total = total + product
    return total


def _shifted_depthwise_3x3(lhs: jax.Array, rhs: jax.Array, dilation: int) -> jax.Array:
    """The depthwise convolution's taps (`_taps`), each product rounded to fp32 before it is added, the
    sum from the first, rounded once to the input's dtype.

    XLA:CPU runs a depthwise convolution of more than 16 features this way,
    as exact as its library convolution (tests/test_depthwise_conv.py holds
    it to check 1), which on the hybrid DiT's 2 x 16 x 16 x 768 maps takes
    6.4 ms a call against 1.4 here; 16 features or fewer keep the
    convolution. An output whose nine taps are all negative zero is the
    sum's negative zero here, where the library gives a zero of either sign
    by its kernel and the host it runs on."""
    height, width = lhs.shape[1:3]
    padded = jnp.pad(lhs, ((0, 0), (dilation, dilation), (dilation, dilation), (0, 0))).astype(jnp.float32)
    products = jax.lax.optimization_barrier(_taps(padded, rhs.astype(jnp.float32), dilation, height, width))
    return _accumulated(products[0], products[1:]).astype(lhs.dtype)


def _materialized_depthwise_3x3(lhs: jax.Array, rhs: jax.Array, dilation: int) -> jax.Array:
    """The depthwise convolution's taps (`_taps`) summed onto zeros, its fp32 input, kernel and output
    held in memory (`_barrier`, in the cotangents too), so XLA fuses the products and their weight
    reductions into kernels of their own rather than into a neighbour's.

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
    # Onto a positive zero, as the form has summed: XLA:CPU keeps it, so an output whose nine taps
    # are all negative zero is positive zero there, and XLA:CUDA, where this form runs, drops it (A100
    # and H100), so the output is the sum's negative zero, as the shifted form's is.
    products = _taps(padded, _barrier(rhs.astype(jnp.float32)), dilation, height, width)
    return _barrier(_accumulated(jnp.zeros(lhs.shape, jnp.float32), products)).astype(lhs.dtype)


def _conv_general_dilated(
        lhs: jax.Array, rhs: jax.Array, window_strides: Sequence[int],
        padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None,
        rhs_dilation: Sequence[int] | None = None,
        dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
        feature_group_count: int = 1, batch_group_count: int = 1,
        precision: jax.lax.PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        out_sharding: jax.sharding.NamedSharding | None = None, *,
        dilated: Callable[[jax.Array, jax.Array, int], jax.Array] | None = None) -> jax.Array:
    """`lax.conv_general_dilated` with Dew's depthwise forms. A dilated 3x3
    depthwise one runs on CUDA as `dilated(x, kernel, dilation)`, by default
    the form measured faster for its dtype; `tools/benchmark_depthwise.py`
    passes lax's own to time against it."""

    def convolve(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.lax.conv_general_dilated(
            x, w, window_strides, padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers, feature_group_count=feature_group_count,
            batch_group_count=batch_group_count, precision=float32_products(products_dtype(x, w), precision),
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
        # bf16 is faster polyphase, fp32 materialized (`_materialized_depthwise_3x3`).
        if dilated is None:
            dilated = _polyphase_depthwise_3x3 if lhs.dtype == jnp.bfloat16 else _materialized_depthwise_3x3
        return jax.lax.platform_dependent(
            lhs, rhs,
            cuda=convolve if dilation[0] == 1 else lambda x, w: dilated(x, w, dilation[0]),
            cpu=((lambda x, w: _shifted_depthwise_3x3(x, w, dilation[0]))
                 if feature_group_count > 16 else convolve),
            default=convolve)
    return convolve(lhs, rhs)


def _placed(inputs: jax.Array, spatial: int, strides: Sequence[int],
            convolve: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """`convolve(inputs)` with its input and output split over every mesh
    axis (the module docstring's first bug) and, on TPU, a barrier at a
    strided 2D convolution's input (its third)."""
    inputs = _unreplicated(inputs, spatial)
    if spatial == 2 and any(stride > 1 for stride in strides):
        inputs = jax.lax.platform_dependent(inputs, tpu=_barrier, default=lambda x: x)
    return _unreplicated(convolve(inputs), spatial)


def convolution(inputs: jax.Array, kernel: jax.Array, padding: Sequence[tuple[int, int]]) -> jax.Array:
    """`Conv`'s stride-1 convolution of `inputs` `[batch, *spatial, features]`
    by a `[*spatial, in, out]` kernel of the inputs' dtype that the caller
    holds, placed and dispatched as `Conv` places and dispatches its own:
    the product of a layer that transforms its kernel first
    (`dew.nn.mp.MPConv`)."""
    spatial = kernel.ndim - 2
    letters = "HWD"[:spatial]
    return _placed(inputs, spatial, (1,) * spatial, lambda x: _conv_general_dilated(
        x, _whole(kernel), (1,) * spatial, padding,
        dimension_numbers=(f"N{letters}C", f"{letters}IO", f"N{letters}C")))


class Conv(nn.Conv):
    """Flax's convolution with safe placement, backend-specific depthwise work
    and its bias gradient summed at fp32.

    The kernel is placed where Flax reads its promoted operands, through
    `promote_dtype`. Parameters and names remain Flax's, so checkpoints do
    not change. Other grouped and ordinary convolutions remain lax calls.

    The call is flax 0.12's, its padding, batch flattening, mask and
    promotion alike, with the bias added by `biased`: flax adds it in its
    storage dtype after the product, so JAX sums a bfloat16 bias's gradient
    over every output position in bfloat16 (jax-ml/jax#30310), where torch's
    convolution sums it in float32 (`dew.nn.dense` says the same of a dense
    layer's).
    """

    promote_dtype: PromoteDtypeFn = _promoted_whole
    conv_general_dilated: ConvGeneralDilatedT | None = _conv_general_dilated

    @nn.nowrap
    def spatial_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """The native convolution's output geometry, without allocating its arrays."""
        spatial = shape[:-1]
        kernel = (self.kernel_size,) if isinstance(self.kernel_size, int) else tuple(self.kernel_size)
        strides = ((self.strides,) * len(kernel) if isinstance(self.strides, int) else
                   (1,) * len(kernel) if self.strides is None else tuple(self.strides))
        dilation = ((self.kernel_dilation,) * len(kernel) if isinstance(self.kernel_dilation, int) else
                    (1,) * len(kernel) if self.kernel_dilation is None else tuple(self.kernel_dilation))
        effective = tuple((tap - 1) * rate + 1 for tap, rate in zip(kernel, dilation, strict=True))
        padding = canonicalize_padding(self.padding, len(kernel))
        if isinstance(padding, str):
            if padding == 'VALID':
                padding = ((0, 0),) * len(kernel)
            else:
                return (*tuple((size + step - 1) // step
                               for size, step in zip(spatial, strides, strict=True)), self.features)
        sizes = tuple((size + sum(pad) - tap) // step + 1
                      for size, tap, step, pad in zip(spatial, effective, strides, padding, strict=True))
        return (*sizes, self.features)

    @nn.nowrap
    def model_flops(self, lengths, input_channels: int) -> float:
        """Learned convolution products at the caller's real output positions."""
        taps = self.kernel_size if isinstance(self.kernel_size, int) else math.prod(self.kernel_size)
        return conv_flops(lengths, input_channels, self.features, taps, groups=self.feature_group_count)

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        kernel_size = (self.kernel_size,) if isinstance(self.kernel_size, int) else tuple(self.kernel_size)
        spatial = len(kernel_size)
        strides = 1 if self.strides is None else self.strides
        strides = (strides,) * spatial if isinstance(strides, int) else tuple(strides)
        return _placed(inputs, spatial, strides, lambda placed: self._convolved(placed, kernel_size))

    @nn.nowrap
    def _convolved(self, inputs: jax.Array, kernel_size: tuple[int, ...]) -> jax.Array:
        """Flax's shared-weight convolution of `inputs`, its bias added by `biased`."""
        def each(value: int | Sequence[int] | None) -> tuple[int, ...]:
            value = 1 if value is None else value
            return (value,) * len(kernel_size) if isinstance(value, int) else tuple(value)

        # Every batch dimension in one, as flax convolves them.
        batch_dims = inputs.ndim - len(kernel_size) - 1
        batch_shape = inputs.shape[:batch_dims]
        if batch_dims != 1:
            inputs = jnp.reshape(inputs, (-1, *inputs.shape[batch_dims:]))
        strides, input_dilation, kernel_dilation = (
            each(self.strides), each(self.input_dilation), each(self.kernel_dilation))
        padding = canonicalize_padding(self.padding, len(kernel_size))
        if padding in ('CIRCULAR', 'REFLECT'):
            dilated = [(size - 1) * rate + 1 for size, rate in zip(kernel_size, kernel_dilation, strict=True)]
            pads = [(0, 0), *(((size - 1) // 2, size // 2) for size in dilated), (0, 0)]
            inputs = jnp.pad(inputs, pads, mode='wrap' if padding == 'CIRCULAR' else 'reflect')
            padding = 'VALID'
        elif padding == 'CAUSAL':
            if len(kernel_size) != 1:
                raise ValueError('Causal padding is only implemented for 1D convolutions.')
            inputs = jnp.pad(inputs, [(0, 0), (kernel_dilation[0] * (kernel_size[0] - 1), 0), (0, 0)])
            padding = 'VALID'
        width = inputs.shape[-1]
        if width % self.feature_group_count:
            raise ValueError(f"{width} input features do not split into {self.feature_group_count} groups")
        kernel_shape = (*kernel_size, width // self.feature_group_count, self.features)
        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError('Mask needs to have the same shape as weights. '
                             f'Shapes are: {self.mask.shape}, {kernel_shape}')
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        if self.mask is not None:
            kernel *= self.mask
        bias = (self.param('bias', self.bias_init, (self.features,), self.param_dtype)
                if self.use_bias else None)
        promoted, kernel, bias = self.promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        if promoted is None or kernel is None:
            raise ValueError("a convolution's promotion must return its input and kernel")
        if self.conv_general_dilated_cls is not None:
            convolve = self.conv_general_dilated_cls()
        elif self.conv_general_dilated is not None:
            convolve = self.conv_general_dilated
        else:
            convolve = jax.lax.conv_general_dilated
        # Channels last, as flax lays them out: [N, *spatial, C] and a [*spatial, I, O] kernel.
        ndim = promoted.ndim
        lhs = (0, ndim - 1, *range(1, ndim - 1))
        dimensions = jax.lax.ConvDimensionNumbers(lhs, (ndim - 1, ndim - 2, *range(ndim - 2)), lhs)
        y = convolve(promoted, kernel, strides, padding, lhs_dilation=input_dilation,
                     rhs_dilation=kernel_dilation, dimension_numbers=dimensions,
                     feature_group_count=self.feature_group_count, precision=self.precision)
        if bias is not None:
            # The convolution rounds to its dtype and the bias adds as flax's do; its gradient sums at
            # fp32 (`biased`).
            y = biased(y, bias)
        return y if batch_dims == 1 else jnp.reshape(y, (*batch_shape, *y.shape[1:]))
