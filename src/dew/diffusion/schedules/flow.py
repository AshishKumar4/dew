"""Rectified flow schedules: the linear path and its resolution shift."""

import jax
import jax.numpy as jnp

from .continuous import ContinuousNoiseScheduler


class FlowMatchingScheduler(ContinuousNoiseScheduler):
    """Rectified flow / conditional flow matching on the linear path.

    x_t = (1 - t) * x_0 + t * epsilon for t in [0, 1], so alpha + sigma = 1 and
    the model input needs no scaling. Timesteps are drawn logit-normal as in
    SD3, which concentrates training on the middle of the trajectory where the
    velocity is hardest to predict.
    """

    def __init__(self, shift: float = 1.0, logit_mean: float = 0.0, logit_std: float = 1.0):
        self.shift = shift
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def shift_timesteps(self, t) -> jax.Array:
        return self.shift * t / (1 + (self.shift - 1) * t)

    def sample_t(self, key, n):
        normal = jax.random.normal(key, (n,), dtype=jnp.float32)
        return jax.nn.sigmoid(normal * self.logit_std + self.logit_mean)

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
