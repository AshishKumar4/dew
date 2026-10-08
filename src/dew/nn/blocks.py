"""The convolutional and embedding pieces the UNets and the DiT sandwich share."""

import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.typing import DTypeLike

from .attention import RMSNorm
from .conv import Conv
from .precision import at_least_fp32
from .sharding import constrain


def normal_kernel(std: float | None, default: Callable | None = None) -> dict:
    """The `kernel_init` keyword for a normal draw of `std`, as lm-engine
    initialises its linears (`nn.init.normal_`, init_utils.py at 45b6b57b).

    None keeps `default`, or the module's own initializer when that is None
    too, so a model that states no std builds exactly what it built before.
    """
    if std is not None:
        return {"kernel_init": nn.initializers.normal(std)}
    return {} if default is None else {"kernel_init": default}


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def table_rows(table: jax.Array, ids: jax.Array, dtype: Dtype) -> jax.Array:
    """`table[ids]` in `dtype`, the table's rows gathered before the cast so
    the gradient accumulates in the table's dtype: casting the table first
    would scatter-add repeated tokens' cotangents in bf16."""
    return jnp.take(table, ids, axis=0).astype(dtype)


def _table_rows_forward(table: jax.Array, ids: jax.Array, dtype: Dtype):
    return table_rows(table, ids, dtype), (table, ids)


def _table_rows_backward(dtype: Dtype, residuals: tuple[jax.Array, jax.Array],
                         cotangent: jax.Array) -> tuple[jax.Array, None]:
    del dtype
    table, ids = residuals
    # The gradient adds each token's cotangent to its row. Under a batch
    # split GSPMD scattered each device's tokens into a table-sized buffer
    # and summed the buffers across devices: a table's worth of traffic
    # decided by the rows. Where the rows are the smaller, every device
    # gathers all of them, in the compute dtype, and scatters them itself,
    # into its copy of the table or its shard of the vocabulary, and the
    # gradient needs no sum.
    if cotangent.size * cotangent.dtype.itemsize < table.size * table.dtype.itemsize:
        cotangent = constrain(cotangent, (None,) * cotangent.ndim)
        ids = constrain(ids, (None,) * ids.ndim)
    return jnp.zeros_like(table).at[ids].add(cotangent.astype(table.dtype)), None


table_rows.defvjp(_table_rows_forward, _table_rows_backward)


class TokenEmbedding(nn.Embed):
    """Token lookup with cotangent accumulation in the parameter dtype."""

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        if self.num_embeddings == 1:
            values = jnp.broadcast_to(self.embedding, (*inputs.shape, self.features))
            promoted, = self.promote_dtype(values, dtype=self.dtype, inexact=False)
            if promoted is None:
                raise ValueError("Embedding dtype promotion must return an array")
            return promoted
        return table_rows(self.embedding, inputs, self.dtype or self.embedding.dtype)


class FourierEmbedding(nn.Module):
    """Random Fourier features of a scalar per example: `[B]` to `[B, features]`,
    sines then cosines of the input against fixed Gaussian frequencies.

    The frequencies, already multiplied by `scale`, are the `constants`
    variable `frequencies`, so a checkpoint carries the table its weights
    learned against. `init` draws it from numpy's RandomState(42), which gives
    the same table on every jax version. A checkpoint converted from
    elsewhere brings its own: FlaxDiff 0.2 (commit 3e3497e, the code flaxdiff
    0.2.8 shipped) drew its with `jax.random.normal`, whose stream changed in
    jax 0.5.0, and FlaxDiff's main branch draws numpy's, as Dew does (commit
    63f2427).
    """
    features: int
    scale: float = 16
    dtype: Dtype | None = None
    """The model's compute dtype; the features are computed in
    `at_least_fp32` of it."""

    def setup(self):
        self.frequencies = self.variable(
            "constants", "frequencies",
            lambda: jnp.asarray(np.random.RandomState(42).normal(size=(self.features // 2,)),
                                dtype=jnp.float32) * self.scale)

    def __call__(self, x):
        x = jax.lax.convert_element_type(x, at_least_fp32(self.dtype))
        # 2 pi times the table is one rounded vector, as it was when the table
        # was a trace-time constant. Without the barrier XLA folds the 2 pi
        # into whatever scalar the caller scaled `x` by (EDM's 1/4) and the
        # sin/cos arguments round differently.
        angular = jax.lax.optimization_barrier(2 * jnp.pi * self.frequencies.value)
        emb = x[:, None] * angular[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


def sinusoidal_time(time, features: int, *, dtype: DTypeLike, shift: float = 0, cosine_first: bool = True):
    """Embed a scalar timestep as `features` sinusoids: `[B]` to `[B, features]`,
    computed in `dtype` (`at_least_fp32` of the model's).

    `cosine_first` puts the cosines in the leading half, as SD1/2 and SDXL
    store them. `shift` moves the lowest frequency, the reference's
    `freq_shift`.
    """
    half = features // 2
    if features % 2 or half <= shift:
        raise ValueError("Time embedding width must be even and exceed twice the frequency shift")
    # Scale the whole exponent before dividing, as the published models do;
    # the other order moves a sine by 1e-5 near timestep 1000. Take exp on
    # the host in float64 so the table does not vary with the backend.
    width = np.dtype(dtype).type
    exponent = np.arange(half, dtype=width) * width(-math.log(10000.0)) / width(half - shift)
    frequencies = jnp.asarray(np.exp(exponent.astype(np.float64)).astype(width))
    phase = jnp.asarray(time, width).reshape(-1, 1) * frequencies[None]
    first, second = (jnp.cos(phase), jnp.sin(phase)) if cosine_first else (jnp.sin(phase), jnp.cos(phase))
    return jnp.concatenate([first, second], axis=-1)


class TimeProjection(nn.Module):
    """Two dense layers with the activation after each."""
    features: int
    activation: Callable = jax.nn.gelu
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        dense = partial(nn.DenseGeneral, self.features,
                        dtype=self.dtype, precision=self.precision)
        x = self.activation(dense()(x))
        return self.activation(dense()(x))


def torch_nearest_resize(x, height: int, width: int):
    """Resample `[B, H, W, C]` to `height` by `width`, torch's nearest rule.

    PyTorch's 'nearest' reads source index `floor(dst * src / dst_size)`,
    not the half-pixel coordinates of `jax.image.resize`, so a converted
    checkpoint only reproduces its reference through this arithmetic.
    """
    rows = jnp.arange(height) * x.shape[1] // height
    columns = jnp.arange(width) * x.shape[2] // width
    return jnp.take(jnp.take(x, rows, axis=1), columns, axis=2)


def _torch_bicubic_weights(size_in: int, size_out: int, antialias: bool) -> np.ndarray:
    """The `[size_out, size_in]` matrix of torch's bicubic interpolation along
    one axis, `align_corners=False`.

    Without antialiasing each output reads four taps at half-pixel source
    coordinates, clamped at the border, with the cubic's a = -0.75; the
    coordinate is computed in float32 as torch's CPU kernel computes it, since
    at 256 pixels its rounding moves a weight by 1e-5. With antialiasing the
    kernel is a = -0.5, widened by the scale when shrinking, and normalized."""
    scale = np.float32(size_in) / np.float32(size_out)
    matrix = np.zeros((size_out, size_in), np.float64)
    if antialias:
        stretch = max(float(scale), 1.0)
        for i in range(size_out):
            center = float(scale) * (i + 0.5)
            low, high = max(int(center - 2 * stretch + 0.5), 0), min(int(center + 2 * stretch + 0.5), size_in)
            x = np.abs((np.arange(low, high) - center + 0.5) / stretch)
            weights = np.where(
                x < 1, (1.5 * x - 2.5) * x * x + 1, np.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0)
            )
            matrix[i, low:high] = weights / weights.sum()
        return matrix.astype(np.float32)
    a, one = np.float32(-0.75), np.float32(1)

    def near(x):
        return ((a + 2) * x - (a + 3)) * x * x + one

    def far(x):
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    for i in range(size_out):
        source = scale * (np.float32(i) + np.float32(0.5)) - np.float32(0.5)
        index = int(np.floor(source))
        t = np.float32(source - np.float32(index))
        for tap, weight in zip(
            range(index - 1, index + 3), (far(t + one), near(t), near(one - t), far(2 * one - t)), strict=True
        ):
            matrix[i, min(max(tap, 0), size_in - 1)] += weight
    return matrix.astype(np.float32)


def torch_bicubic_resize(x, height: int, width: int, *, antialias: bool = False):
    """Resample `[B, H, W, C]` to `height` by `width` as torch's
    `F.interpolate(mode="bicubic", align_corners=False)` does.

    `jax.image.resize`'s cubic is Keys' a = -0.5 and always antialiases a
    shrink; torch's is a = -0.75 and antialiases only when asked, so a vision
    encoder that resizes its input or its position table reproduces its
    reference only through this arithmetic."""
    rows = _torch_bicubic_weights(x.shape[1], height, antialias)
    columns = _torch_bicubic_weights(x.shape[2], width, antialias)
    return jnp.einsum("oh,bhwc,pw->bopc", rows, x, columns, precision=jax.lax.Precision.HIGHEST)


class Upsample(nn.Module):
    """Nearest-neighbour upsampling by `scale`, then a 3x3 convolution to `features`."""
    features: int
    scale: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        out = jax.image.resize(x, (B, H * self.scale, W * self.scale, C), method="nearest")
        return Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1),
                    dtype=self.dtype, precision=self.precision)(out)


class Downsample(nn.Module):
    """A stride-2 3x3 convolution to `features`."""
    features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        return Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2),
                    dtype=self.dtype, precision=self.precision)(x)


class ResidualBlock(nn.Module):
    """Norm, activation, convolution, the projected time embedding added,
    norm, activation, convolution, plus the input (through a 1x1 convolution
    when the width changes). `norm_groups` of 0 swaps the group norms for
    RMS norms."""
    features: int
    kernel_size: tuple = (3, 3)
    activation: Callable = jax.nn.swish
    norm_groups: int = 8
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-4
    dropout: float = 0.0

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups, epsilon=self.norm_epsilon, dtype=self.dtype)
        else:
            norm = partial(RMSNorm, epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm1 = norm()
        self.norm2 = norm()

    @nn.compact
    def __call__(self, x: jax.Array, temb: jax.Array, *, train: bool = False):
        conv = partial(Conv, features=self.features, kernel_size=self.kernel_size,
                       strides=(1, 1), dtype=self.dtype, precision=self.precision)
        out = conv(name="conv1")(self.activation(self.norm1(x)))

        temb = nn.DenseGeneral(features=self.features, name="temb_projection",
                               dtype=self.dtype, precision=self.precision)(temb)
        out = out + temb[:, None, None, :]

        out = self.activation(self.norm2(out))
        if self.dropout:
            out = nn.Dropout(self.dropout)(out, deterministic=not train)
        out = conv(name="conv2")(out)

        residual = x
        if residual.shape != out.shape:
            residual = conv(kernel_size=(1, 1), name="residual_conv")(residual)
        return out + residual
