import math

import jax
import jax.numpy as jnp

from .continuous import ContinuousNoiseScheduler


def compute_resolution_shift(sequence_length, base_seq_len=256, max_seq_len=4096,
                             base_shift=0.5, max_shift=1.15) -> float:
    """Flux-style resolution dependent timestep shift.

    Longer token sequences carry more redundancy, so the trajectory has to
    spend more of its budget at high noise for the global structure to settle.
    mu is interpolated linearly in sequence length and the shift is exp(mu).
    """
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = base_shift + slope * (sequence_length - base_seq_len)
    return math.exp(mu)


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
        # The flow models were trained on the shifted time times 1000, so that
        # is what a trained one reads. SimpleDiT's embedder is EDM's random
        # Fourier features (blocks.FourierEmbedding), which take an input of
        # order one and do not ask for the DiT sinusoidal range; the factor
        # stays because changing it changes what every trained flow model is
        # conditioned on.
        return self.shift_timesteps(jnp.asarray(t, jnp.float32)) * 1000
