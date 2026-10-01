"""Rectified flow schedules: the linear path and its resolution shift."""

import math
from typing import Literal

import jax
import jax.numpy as jnp

from .continuous import ContinuousNoiseScheduler

Density = Literal["logit_normal", "mode", "cosmap", "uniform"]
"""The training time densities of Esser et al. 2024 ("Scaling Rectified Flow
Transformers for High-Resolution Image Synthesis", section 3.1)."""


class FlowMatchingScheduler(ContinuousNoiseScheduler):
    """Rectified flow / conditional flow matching on the linear path.

    x_t = (1 - t) * x_0 + t * epsilon for t in [0, 1], so alpha + sigma = 1 and
    the model input needs no scaling. Training times are drawn from one of
    SD3's densities (`density`):

    - `logit_normal`: t = sigmoid(logit_mean + logit_std * n), n ~ N(0, 1),
      which concentrates training on the middle of the trajectory, where the
      velocity is hardest to predict.
    - `mode`: t = 1 - u - s (cos^2(pi u / 2) - 1 + u), u ~ U(0, 1), with
      s = `mode_scale` in [-1, 2 / (pi - 2)] (Eq. 20): a density with a mode
      in the middle that keeps weight on both ends, uniform at s = 0.
    - `cosmap`: t = 1 - 1 / (tan(pi u / 2) + 1) (Eq. 21).
    - `uniform`: t = u.

    The draw is of the unshifted time; `shift` maps it, and the sampling
    grid, to the resolution's noise levels.
    """

    def __init__(self, shift: float = 1.0, logit_mean: float = 0.0, logit_std: float = 1.0,
                 density: Density = "logit_normal", mode_scale: float = 1.29):
        if density not in ("logit_normal", "mode", "cosmap", "uniform"):
            raise ValueError(f"density is logit_normal, mode, cosmap or uniform, not {density!r}")
        if density == "mode" and not -1.0 <= mode_scale <= 2.0 / (math.pi - 2.0):
            raise ValueError(f"mode_scale is in [-1, 2 / (pi - 2)], where the map is monotone; "
                             f"got {mode_scale}")
        self.shift = shift
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.density = density
        self.mode_scale = mode_scale

    def shift_timesteps(self, t) -> jax.Array:
        return self.shift * t / (1 + (self.shift - 1) * t)

    def sample_t(self, key, n):
        if self.density == "logit_normal":
            normal = jax.random.normal(key, (n,), dtype=jnp.float32)
            return jax.nn.sigmoid(normal * self.logit_std + self.logit_mean)
        u = jax.random.uniform(key, (n,), dtype=jnp.float32)
        if self.density == "mode":
            return 1 - u - self.mode_scale * (jnp.cos(jnp.pi / 2 * u) ** 2 - 1 + u)
        if self.density == "cosmap":
            return 1 - 1 / (jnp.tan(jnp.pi / 2 * u) + 1)
        return u

    def rates(self, t):
        t = self.shift_timesteps(jnp.asarray(t, jnp.float32))
        return 1 - t, t

    def weight(self, t):
        return jnp.ones_like(jnp.asarray(t, jnp.float32))

    def model_time(self, t):
        # Trained flow checkpoints are conditioned on the shifted time times
        # 1000. The factor is part of the training convention, not of the
        # embedder: SimpleDiT's Fourier embedding takes an input of order one
        # either way.
        return self.shift_timesteps(jnp.asarray(t, jnp.float32)) * 1000
