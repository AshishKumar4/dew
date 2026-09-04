import jax
import jax.numpy as jnp

from .common import NoiseScheduler


class DiscreteNoiseScheduler(NoiseScheduler):
    """A variance preserving schedule tabulated from betas, DDPM style.

    signal_rate^2 + noise_rate^2 = 1 at every index, and t is the index into
    the table, so T is the number of entries. The loss weight is the P2 weight
    of Choi et al. 2022, (k + SNR)^-gamma. At the defaults k = 1, gamma = 1 it
    is 1 / (1 + SNR), which on a v-prediction loss (whose error is 1 + SNR
    times the x_0 error) is exactly an unweighted x_0 loss.
    """

    def __init__(self, timesteps: int, beta_start: float, beta_end: float, schedule_fn,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        self.T = timesteps
        betas = schedule_fn(timesteps, beta_start, beta_end)
        alphas = 1 - betas
        alpha_cumprod = jnp.cumprod(alphas, axis=0)

        self.betas = jnp.array(betas, dtype=jnp.float32)
        self.alphas = alphas.astype(jnp.float32)
        self.alpha_cumprod = alpha_cumprod.astype(jnp.float32)
        self.sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod).astype(jnp.float32)
        self.sqrt_one_minus_alpha_cumprod = jnp.sqrt(1 - alpha_cumprod).astype(jnp.float32)
        self.p2_loss_weights = (
            p2_loss_weight_k + self.alpha_cumprod / (1 - self.alpha_cumprod)
        ) ** -p2_loss_weight_gamma

    def index(self, t) -> jax.Array:
        """`t` as a table index; a time grid may reach T itself, which is the
        last entry."""
        return jnp.clip(jnp.asarray(t).astype(jnp.int32), 0, self.T - 1)

    def rates(self, t):
        index = self.index(t)
        return self.sqrt_alpha_cumprod[index], self.sqrt_one_minus_alpha_cumprod[index]

    def sample_t(self, key, n):
        return jax.random.randint(key, (n,), 0, self.T)

    def weight(self, t):
        return self.p2_loss_weights[self.index(t)]
