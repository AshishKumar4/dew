"""Fourier feature pyramids: learnable periodic feature grids at coarse-to-fine resolutions, read at any
coordinate by Fourier interpolation.

The pyramid is beignet's (Zhao et al., "Fourier Feature Pyramids for Physics-Informed Neural Networks",
arXiv 2605.24278): level l holds a grid of `features` channels at its own resolution, read at a
coordinate x in [0, 1)^d by its trigonometric interpolant,

    g_l(x) = Re( sum_k theta_hat_l[k] exp(2 pi i k . x) ),

theta_hat_l the grid's DFT over its size, k over each axis's signed FFT modes {-N/2, ..., N/2 - 1}. The
levels' features, each scaled by `level_scale` to its index and all by `global_scale` (the paper's
per-level and global preconditioners), are concatenated for a decoder to read. A level's resolution bounds
the frequencies it represents, so the finest grid is the representation's bandlimit. The interpolant is
smooth in x, so a residual's derivatives in the coordinates are the spectral derivatives of each level
(the paper's Eq. 8), which automatic differentiation of this sum computes exactly.
"""

from __future__ import annotations

import math
import string

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.typing import DTypeLike

from dew.nn.precision import float32_einsum


def interpolated(grid: jax.Array, coordinates: jax.Array) -> jax.Array:
    """The trigonometric interpolant of `grid` ([*sizes, channels], periodic on [0, 1)^d) at
    `coordinates` ([..., d]): Re of the sum over each axis's signed FFT modes of the grid's DFT over its
    size times exp(2 pi i k . x), [..., channels]. At a grid point i / N it is the grid's own value.

    The modes are summed one axis at a time, each a complex product written as real products through
    `float32_einsum`, so a float32 grid multiplies at float32's own precision on every backend (a GPU's
    complex product would take TF32) and a float64 one in float64."""
    sizes = grid.shape[:-1]
    if coordinates.shape[-1] != len(sizes):
        raise ValueError(f"a grid over {len(sizes)} axes reads coordinates of {len(sizes)}, not "
                         f"{coordinates.shape[-1]}")
    spectrum = jnp.fft.fftn(grid, axes=tuple(range(len(sizes)))) / math.prod(sizes)
    points = coordinates.reshape(-1, len(sizes))
    real, imaginary = spectrum.real, spectrum.imag
    # Contracted axis by axis: [modes..., channels], then [points, modes..., channels] with the points in
    # front once the first axis is summed.
    for axis, size in enumerate(sizes):
        angle = 2 * jnp.pi * jnp.fft.fftfreq(size, 1 / size).astype(points.dtype)[:, None] * points[:, axis]
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        rest = string.ascii_uppercase[:len(sizes) - axis - 1]
        subscripts = f"k{rest}c,kn->n{rest}c" if axis == 0 else f"nk{rest}c,kn->n{rest}c"
        last = axis == len(sizes) - 1
        real, imaginary = (float32_einsum(subscripts, real, cos) - float32_einsum(subscripts, imaginary, sin),
                           None if last else
                           float32_einsum(subscripts, real, sin) + float32_einsum(subscripts, imaginary, cos))
    return real.reshape(*coordinates.shape[:-1], grid.shape[-1])


class FourierPyramid(nn.Module):
    """Learnable periodic feature grids at the resolutions `grids` (one size per axis a level, or one int
    for a 1-D level), each of `features` channels, read at coordinates in [0, 1)^d by Fourier
    interpolation (`interpolated`) and concatenated, [..., levels * features].

    Level l's features are scaled by `level_scale` ** l and all by `global_scale`: a decoder reads features
    of order one while the grids train at the scale of `init_noise`, the half-width of the uniform draw
    every grid value starts from. Coordinates are periodic with period one; a problem on another domain
    maps onto [0, 1) first, onto half of it where its solution is not periodic.
    """

    grids: tuple[int | tuple[int, ...], ...]
    features: int
    global_scale: float = 1.0
    level_scale: float = 1.0
    init_noise: float = 1e-4
    param_dtype: DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, coordinates: jax.Array) -> jax.Array:
        noise = self.init_noise

        def drawn(key, shape, dtype):
            return (jax.random.uniform(key, shape, dtype) - 0.5) * 2 * noise

        levels = []
        for index, size in enumerate(self.grids):
            shape = (size,) if isinstance(size, int) else tuple(size)
            grid = self.param(f"level_{index}", drawn, (*shape, self.features), self.param_dtype)
            levels.append(interpolated(grid, coordinates.astype(grid.dtype)) * self.level_scale ** index)
        return jnp.concatenate(levels, axis=-1) * self.global_scale
