"""EDM2's magnitude-preserving layers (Karras et al. 2024, "Analyzing and
Improving the Training Dynamics of Diffusion Models", section 3 and
appendix B).

Every layer keeps the expected magnitude of its activations at one: the
weights are normalized in the forward pass and scaled by the fan-in (Eq. 47),
the nonlinearity is divided by its expected output magnitude (Eq. 81), and a
sum or a concatenation weighs its operands so the result has unit magnitude
(Eqs. 88, 103). The layers are NVlabs/edm2's `training/networks_edm2.py`, in
Flax's channels-last layout.

Forced weight normalization (Eq. 66) keeps each stored weight at unit
magnitude as well, so an update's relative size, and with it the effective
learning rate, does not decay as the weights grow. The paper applies it in
the training forward pass, in place; here it is `forced_weight_normalization`,
an optimizer step that renormalizes every `mp_kernel` after its update, which
`dew.training.optim.build_optimizer` appends under
`OptimConfig.forced_weight_normalization`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

MP_KERNEL = "mp_kernel"
"""The parameter name of a magnitude-preserving weight, which forced weight
normalization keeps at unit magnitude."""


def normalize(x: jax.Array, axes: Sequence[int] | None = None, eps: float = 1e-4) -> jax.Array:
    """`x` over `axes` at unit root-mean-square magnitude, all but the last
    by default: each output channel of a kernel, or each feature vector."""
    axes = tuple(range(x.ndim - 1)) if axes is None else tuple(axes)
    count = math.prod(x.shape[axis] for axis in axes)
    norm = jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=axes, keepdims=True))
    return x / (eps + norm / np.sqrt(count)).astype(x.dtype)


def mp_silu(x: jax.Array) -> jax.Array:
    """SiLU divided by its expected output magnitude on a unit normal (Eq. 81)."""
    return jax.nn.silu(x) / 0.596


def mp_sum(a: jax.Array, b: jax.Array, t: float = 0.5) -> jax.Array:
    """The interpolation (1 - t) a + t b at unit magnitude (Eq. 88)."""
    return (a + t * (b - a)) / np.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a: jax.Array, b: jax.Array, t: float = 0.5) -> jax.Array:
    """The channel concatenation of `a` and `b`, weighted by `t` and at unit
    magnitude (Eq. 103)."""
    na, nb = a.shape[-1], b.shape[-1]
    scale = np.sqrt((na + nb) / ((1 - t) ** 2 + t ** 2))
    return jnp.concatenate([a * (scale / np.sqrt(na) * (1 - t)),
                            b * (scale / np.sqrt(nb) * t)], axis=-1)


class MPFourier(nn.Module):
    """Fourier features of a scalar at unit magnitude (Eq. 75): random
    frequencies and phases, drawn once and held under `constants`."""

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
        y = jnp.asarray(x, jnp.float32)[..., None] * frequencies.value + phases.value
        return (jnp.cos(y) * np.sqrt(2)).astype(jnp.asarray(x).dtype)


class MPConv(nn.Module):
    """A convolution, or a dense layer for an empty `kernel_size`, whose
    weight is normalized per output channel and scaled by the fan-in (Eq. 47).

    The weight is `mp_kernel`, spatial axes first, then input and output
    channels. A convolution pads to keep the size, as the reference's odd
    kernels do.
    """

    features: int
    kernel_size: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, x: jax.Array, gain: jax.Array | float = 1.0) -> jax.Array:
        shape = (*self.kernel_size, x.shape[-1], self.features)
        weight = self.param(MP_KERNEL, lambda key: normalize(jax.random.normal(key, shape)))
        weight = normalize(weight.astype(jnp.float32))
        weight = weight * (gain / np.sqrt(math.prod(shape[:-1])))
        weight = weight.astype(x.dtype)
        if not self.kernel_size:
            return x @ weight
        pad = [(size // 2, size // 2) for size in self.kernel_size]
        return jax.lax.conv_general_dilated(x, weight, (1,) * len(self.kernel_size), pad,
                                            dimension_numbers=("NHWC", "HWIO", "NHWC"))


class Uncertainty(nn.Module):
    """EDM2's learned uncertainty u(sigma) (Eq. 21; `Precond`'s logvar head):
    magnitude-preserving Fourier features of the model time through one
    magnitude-preserving dense layer, one log-variance per example."""

    channels: int = 128

    @nn.compact
    def __call__(self, time: jax.Array) -> jax.Array:
        return MPConv(1, name="linear")(MPFourier(self.channels, name="fourier")(time))[..., 0]


def forced_weight_normalization() -> optax.GradientTransformation:
    """The update that leaves every `mp_kernel` at unit magnitude per output
    channel once applied (Eq. 66); every other update passes through."""

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
