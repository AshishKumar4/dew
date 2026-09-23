"""The convolutional and embedding pieces the UNets and the DiT sandwich share."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .attention import RMSNorm
from .conv import Conv
from .sharding import constrain, logical_axes


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
    sines then cosines of the input against fixed Gaussian frequencies."""
    features: int
    scale: int = 16

    def setup(self):
        # Fixed frequencies via numpy so they are identical across jax versions
        # (jax 0.5.0 changed the default PRNG, which changed these).
        freqs = np.random.RandomState(42).normal(size=(self.features // 2,))
        self.freqs = jnp.asarray(freqs, dtype=jnp.float32) * self.scale

    def __call__(self, x):
        x = jax.lax.convert_element_type(x, jnp.float32)
        emb = x[:, None] * (2 * jnp.pi * self.freqs)[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


@logical_axes({}, heuristic=(("DenseGeneral_*",),))
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


@logical_axes({}, heuristic=(("Conv_*",),))
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


@logical_axes({}, heuristic=(("conv1",), ("conv2",), ("residual_conv",), ("temb_projection",)))
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
