"""EDM2's magnitude-preserving layers.

The layers follow Karras et al. 2024, "Analyzing and Improving the Training
Dynamics of Diffusion Models", section 3 and appendix B. Every layer keeps
the expected magnitude of its activations at one. The weights are normalized
in the forward pass and scaled by the fan-in (Eq. 47), the nonlinearity is
divided by its expected output magnitude (Eq. 81), and a sum or a
concatenation weights its operands so the result has unit magnitude
(Eqs. 88, 103). The layers are NVlabs/edm2's `training/networks_edm2.py`, in
Flax's channels-last layout.

Forced weight normalization (Eq. 66) also keeps each stored weight at unit
magnitude, so an update's relative size, and with it the effective learning
rate, does not decay as the weights grow. The paper applies it in place in
the training forward pass. Here it is `forced_weight_normalization`, an
optimizer step that renormalizes every `mp_kernel` after its update;
`OptimConfig.build` appends it when `OptimConfig.forced_weight_normalization`
is set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.nn.activations import silu
from dew.nn.conv import convolution
from dew.nn.precision import at_least_fp32, compensated_sum, divided, float32_matmul

MP_KERNEL = "mp_kernel"
"""The parameter name of a magnitude-preserving weight. Forced weight
normalization keeps these weights at unit magnitude."""


def normalize(x: jax.Array, axes: Sequence[int] | None = None, eps: float = 1e-4, *,
              compensated: bool = False) -> jax.Array:
    """Return `x` scaled to unit root-mean-square magnitude over `axes`.

    `axes` defaults to all but the last, which normalizes each output channel
    of a kernel, or each feature vector. `compensated` sums the squares with
    `compensated_sum`, as a kernel's initialization does (`drawn_kernel`).
    """
    axes = tuple(range(x.ndim - 1)) if axes is None else tuple(axes)
    count = math.prod(x.shape[axis] for axis in axes)
    squares = jnp.square(x.astype(at_least_fp32(x.dtype)))
    total = (jnp.expand_dims(compensated_sum(squares, axes), axes) if compensated
             else jnp.sum(squares, axis=axes, keepdims=True))
    return x / (eps + jnp.sqrt(total) / math.sqrt(count)).astype(x.dtype)


def drawn_kernel(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """A standard normal kernel of `shape` scaled to unit magnitude per output channel by
    `normalize`, its sum of squares compensated: a kernel split over its fan-in sums each shard
    apart, and the compensated sum rounds once at the end, so a split draw has matched one
    device's bit for bit where a plain sum's last bits followed the split (tools/layout_parity.py
    holds each split draw to one device's)."""
    return normalize(jax.random.normal(key, shape), compensated=True)


def mp_silu(x: jax.Array) -> jax.Array:
    """Return SiLU divided by its expected output magnitude on a unit normal input.

    This is Eq. 81 of the EDM2 paper.
    """
    # IEEE division, as EDM2's PyTorch divides: XLA takes a division by a constant as a product with
    # its rounded reciprocal.
    return divided(silu(x), 0.596)


def mp_sum(a: jax.Array, b: jax.Array, t: float = 0.5) -> jax.Array:
    """Return the interpolation (1 - t) a + t b, scaled to unit magnitude.

    This is Eq. 88 of the EDM2 paper.
    """
    return (a + t * (b - a)) / math.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a: jax.Array, b: jax.Array, t: float = 0.5) -> jax.Array:
    """Return the channel concatenation of `a` and `b`, weighted by `t` and scaled to unit magnitude.

    This is Eq. 103 of the EDM2 paper.
    """
    na, nb = a.shape[-1], b.shape[-1]
    # Python floats, which take the operands' dtype rather than promoting it.
    scale = math.sqrt((na + nb) / ((1 - t) ** 2 + t ** 2))
    return jnp.concatenate([a * (scale / math.sqrt(na) * (1 - t)),
                            b * (scale / math.sqrt(nb) * t)], axis=-1)


class MPFourier(nn.Module):
    """Computes Fourier features of a scalar at unit magnitude.

    The random frequencies and phases are drawn once and held in the
    `constants` collection. This is Eq. 75 of the EDM2 paper.
    """

    channels: int
    bandwidth: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        frequencies = self.variable(
            "constants", "frequencies",
            lambda: 2 * np.pi * self.bandwidth * jax.random.normal(
                self.make_rng("params"), (self.channels,), jnp.float32))
        phases = self.variable(
            "constants", "phases",
            lambda: 2 * np.pi * jax.random.uniform(self.make_rng("params"), (self.channels,),
                                                   jnp.float32))
        x = jnp.asarray(x)
        wide = at_least_fp32(x.dtype)
        y = x.astype(wide)[..., None] * frequencies.value.astype(wide) + phases.value.astype(wide)
        return (jnp.cos(y) * math.sqrt(2)).astype(x.dtype)


class MPConv(nn.Module):
    """Applies a convolution whose weight is normalized per output channel and scaled by the fan-in.

    An empty `kernel_size` makes it a dense layer. This is Eq. 47 of the
    EDM2 paper. The weight is `mp_kernel`, with the spatial axes first, then
    the input and output channels. A convolution pads to keep the size, as
    the reference's odd kernels do, and runs as `Conv`'s does
    (`dew.nn.conv.convolution`).
    """

    features: int
    kernel_size: tuple[int, ...] = ()

    @nn.nowrap
    def spatial_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (*shape[:-1], self.features)

    @nn.nowrap
    def model_flops(self, lengths, input_features: int) -> float:
        """Dense or convolution products; weight normalization is pointwise work."""
        from dew.nn.flops import conv_flops
        return conv_flops(lengths, input_features, self.features, math.prod(self.kernel_size))

    @nn.compact
    def __call__(self, x: jax.Array, gain: jax.Array | float = 1.0) -> jax.Array:
        shape = (*self.kernel_size, x.shape[-1], self.features)
        weight = self.param(MP_KERNEL, lambda key: drawn_kernel(key, shape))
        weight = normalize(weight.astype(at_least_fp32(x.dtype)))
        weight = weight * (gain / math.sqrt(math.prod(shape[:-1])))
        weight = weight.astype(x.dtype)
        if not self.kernel_size:
            return float32_matmul(x, weight)
        return convolution(x, weight, [(size // 2, size // 2) for size in self.kernel_size])


class Uncertainty(nn.Module):
    """Computes EDM2's learned uncertainty u(sigma), one log-variance per example.

    Magnitude-preserving Fourier features of the model time go through one
    magnitude-preserving dense layer. This is Eq. 21 of the EDM2 paper, and
    the logvar head of the reference's `Precond`.
    """

    channels: int = 128

    @nn.compact
    def __call__(self, time: jax.Array) -> jax.Array:
        return MPConv(1, name="linear")(MPFourier(self.channels, name="fourier")(time))[..., 0]


def forced_weight_normalization() -> optax.GradientTransformation:
    """Return an optax transformation that keeps every `mp_kernel` at unit magnitude.

    It changes each `mp_kernel` update so that the kernel has unit magnitude
    per output channel once the update is applied (Eq. 66 of the EDM2
    paper). Every other update passes through unchanged. The transformation
    needs the parameters and raises `ValueError` without them.
    """

    def update(updates, state, params=None):
        if params is None:
            raise ValueError("forced weight normalization needs the parameters")

        def project(path, change, value):
            last = path[-1]
            if not (isinstance(last, jax.tree_util.DictKey) and last.key == MP_KERNEL):
                return change
            return (normalize(value + change) - value).astype(change.dtype)

        return jax.tree_util.tree_map_with_path(project, updates, params), state

    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


__all__ = ["MP_KERNEL", "MPConv", "MPFourier", "Uncertainty", "forced_weight_normalization",
           "mp_cat", "mp_silu", "mp_sum", "normalize"]
