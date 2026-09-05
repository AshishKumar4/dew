"""The convolutional and embedding pieces the UNets and the DiT sandwich share."""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .sharding import logical_axes


class FourierEmbedding(nn.Module):
    """Random Fourier features of a scalar per example: `[B]` to `[B, features]`,
    sines then cosines of the input against fixed Gaussian frequencies."""
    features: int
    scale: int = 16

    def setup(self):
        # Fixed frequencies via numpy so they are identical across jax versions
        # (jax 0.5.0 changed the default PRNG and silently altered these)
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

    @nn.compact
    def __call__(self, x):
        x = self.activation(nn.DenseGeneral(self.features)(x))
        return self.activation(nn.DenseGeneral(self.features)(x))


@logical_axes({}, heuristic=(("Conv_*",),))
class Upsample(nn.Module):
    """Nearest-neighbour upsampling by `scale`, then a 3x3 convolution to `features`."""
    features: int
    scale: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        out = jax.image.resize(x, (B, H * self.scale, W * self.scale, C), method="nearest")
        return nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1),
                       dtype=self.dtype, precision=self.precision)(out)


class Downsample(nn.Module):
    """A stride-2 3x3 convolution to `features`."""
    features: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        return nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2),
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
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-4

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups, epsilon=self.norm_epsilon)
        else:
            norm = partial(nn.RMSNorm, epsilon=self.norm_epsilon)
        self.norm1 = norm()
        self.norm2 = norm()

    @nn.compact
    def __call__(self, x: jax.Array, temb: jax.Array):
        conv = partial(nn.Conv, features=self.features, kernel_size=self.kernel_size,
                       strides=(1, 1), dtype=self.dtype, precision=self.precision)
        out = conv(name="conv1")(self.activation(self.norm1(x)))

        temb = nn.DenseGeneral(features=self.features, name="temb_projection",
                               dtype=self.dtype, precision=self.precision)(temb)
        out = out + temb[:, None, None, :]

        out = conv(name="conv2")(self.activation(self.norm2(out)))

        residual = x
        if residual.shape != out.shape:
            residual = conv(kernel_size=(1, 1), name="residual_conv")(residual)
        return out + residual
