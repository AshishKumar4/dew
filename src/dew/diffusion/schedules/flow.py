"""Rectified flow schedules: the linear path and its resolution shift."""

import math
from typing import Literal

import jax
import jax.numpy as jnp

from .common import ContinuousNoiseScheduler, times

Density = Literal["logit_normal", "mode", "cosmap", "uniform"]
"""The training time densities of Esser et al. 2024 ("Scaling Rectified Flow
Transformers for High-Resolution Image Synthesis", section 3.1)."""


def token_mu(tokens: int, base_tokens: int, max_tokens: int, base_shift: float, max_shift: float) -> float:
    """mu linear in a token count, `base_shift` at `base_tokens` and
    `max_shift` at `max_tokens`: Diffusers' Flux `calculate_shift`, operation
    for operation, so its double is the pipelines' own."""
    slope = (max_shift - base_shift) / (max_tokens - base_tokens)
    return tokens * slope + (base_shift - slope * base_tokens)


class RectifiedFlow:
    """A schedule on the rectified-flow path, x_t = (1 - sigma) x_0 + sigma
    epsilon: its `rates(t)` are `(1 - sigma, sigma)`. A sampler whose step
    holds only on that path (`dew.sampling.FlowSDE`) asks for this, whether
    the schedule is Dew's own (`FlowMatchingScheduler`) or a published
    source's grid (`dew.diffusion.schedules.source_grids.FlowGrid`)."""


class FlowMatchingScheduler(ContinuousNoiseScheduler, RectifiedFlow):
    """Rectified flow / conditional flow matching on the linear path.

    x_t = (1 - t) * x_0 + t * epsilon for t in [0, 1], so alpha + sigma = 1 and
    the model input needs no scaling. Training times are drawn from one of
    SD3's densities (`density`):

    - `logit_normal`: t = sigmoid(logit_mean + logit_std * n), n ~ N(0, 1),
      which concentrates training on the middle of the trajectory, where the
      velocity is hardest to predict.
    - `mode`: t = 1 - u - s (cos^2(pi u / 2) - 1 + u), u ~ U(0, 1), with
      s = `mode_scale` in [-1, 2 / (pi - 2)] (Eq. 20). This density has a
      mode in the middle, keeps weight on both ends, and is uniform at
      s = 0.
    - `cosmap`: t = 1 - 1 / (tan(pi u / 2) + 1) (Eq. 21).
    - `uniform`: t = u.

    Training draws the unshifted time. `shift` then maps it, and the
    sampling grid, to the resolution's noise levels with
    shift t / (1 + (shift - 1) t), and the model is conditioned on the
    shifted time times 1000. An unknown `density`, or a `mode_scale` outside
    its range for `mode`, raises `ValueError`.
    """

    def __init__(self, shift: float = 1.0, logit_mean: float = 0.0, logit_std: float = 1.0,
                 density: Density = "logit_normal", mode_scale: float = 1.29):
        if density not in ("logit_normal", "mode", "cosmap", "uniform"):
            raise ValueError(f"density is logit_normal, mode, cosmap or uniform, not {density!r}")
        if density == "mode" and not -1.0 <= mode_scale <= 2.0 / (math.pi - 2.0):
            raise ValueError(f"mode_scale is in [-1, 2 / (pi - 2)], where the map is monotone; "
                             f"got {mode_scale}")
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"a rectified-flow timestep shift must be finite and positive, got {shift}")
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
        t = self.shift_timesteps(times(t))
        return 1 - t, t

    def weight(self, t):
        return jnp.ones_like(times(t))

    def model_time(self, t):
        # Trained flow checkpoints are conditioned on the shifted time times
        # 1000. The factor is part of the training convention, not of the
        # embedder: SimpleDiT's Fourier embedding takes an input of order one
        # either way.
        return self.shift_timesteps(times(t)) * 1000
