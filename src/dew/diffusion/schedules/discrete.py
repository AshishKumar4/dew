"""The tabulated variance-preserving schedule every beta table shares."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .common import NoiseScheduler


class DiscreteNoiseScheduler(NoiseScheduler):
    """A variance preserving schedule tabulated from betas, DDPM style.

    signal_rate^2 + noise_rate^2 = 1 at every index, and t is the index into
    the table, so T is the number of entries. The loss weight is the P2
    weight of Choi et al. 2022, (k + SNR)^-gamma, with
    k = `p2_loss_weight_k` and gamma = `p2_loss_weight_gamma`. At the
    defaults k = 1, gamma = 1 it is 1 / (1 + SNR). On a v-prediction loss,
    whose error is 1 + SNR times the x_0 error, that is exactly an
    unweighted x_0 loss. All fixed tables, including the weight, are
    prepared in host float64 and rounded once.
    """

    def __init__(self, betas: np.ndarray | Sequence[float],
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        self.betas = tuple(float(beta) for beta in np.asarray(betas, np.float64))
        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.T = len(self.betas)
        # The table is fixed at construction. Device float32 prefix products
        # and roots introduce backend-dependent error into every later step.
        alpha_cumprod = np.cumprod(1 - np.asarray(betas, np.float64), axis=0)
        self.sqrt_alpha_cumprod = jnp.asarray(np.sqrt(alpha_cumprod), jnp.float32)
        self.sqrt_one_minus_alpha_cumprod = jnp.asarray(np.sqrt(1 - alpha_cumprod), jnp.float32)
        noise_variance = 1 - alpha_cumprod
        # This form of (k + SNR)^-gamma also preserves the zero-noise limit.
        weights = (
            noise_variance / (p2_loss_weight_k * noise_variance + alpha_cumprod)
        ) ** p2_loss_weight_gamma
        self.p2_loss_weights = jnp.asarray(weights, jnp.float32)

    def index(self, t) -> jax.Array:
        """Return `t` as a table index.

        A time grid may reach T itself, which maps to the last entry.
        """
        return jnp.clip(jnp.asarray(t).astype(jnp.int32), 0, self.T - 1)

    def rates(self, t):
        index = self.index(t)
        return self.sqrt_alpha_cumprod[index], self.sqrt_one_minus_alpha_cumprod[index]

    def sample_t(self, key, n):
        return jax.random.randint(key, (n,), 0, self.T)

    def weight(self, t):
        return self.p2_loss_weights[self.index(t)]
