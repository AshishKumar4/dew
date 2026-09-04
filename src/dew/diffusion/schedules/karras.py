import jax
import jax.numpy as jnp

from .common import GeneralizedNoiseScheduler


class KarrasVENoiseScheduler(GeneralizedNoiseScheduler):
    """Sigmas placed along t with the rho spacing of Karras et al. 2022 (Eq. 5):
    sigma(t) = (sigma_max^(1/rho) + (1 - t) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho,
    so a uniform grid in t is the paper's sampling grid in sigma."""

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0,
                 sigma_data: float = 0.5):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)
        self.rho = rho
        self.min_inv_rho = sigma_min ** (1 / rho)
        self.max_inv_rho = sigma_max ** (1 / rho)

    def sigmas(self, t):
        ramp = jnp.clip(1 - jnp.asarray(t, jnp.float32) / self.T, 0.0, 1.0)
        return (self.max_inv_rho + ramp * (self.min_inv_rho - self.max_inv_rho)) ** self.rho

    def t_of_sigma(self, sigma):
        inv_rho = jnp.asarray(sigma, jnp.float32) ** (1 / self.rho)
        ramp = jnp.clip((inv_rho - self.max_inv_rho) / (self.min_inv_rho - self.max_inv_rho),
                        0.0, 1.0)
        return (1 - ramp) * self.T


class EDMNoiseScheduler(GeneralizedNoiseScheduler):
    """Training sigmas drawn from exp(N(P_mean, P_std^2)): t is the standard
    normal draw and sigma(t) = exp(P_mean + P_std t).

    Defaults are EDM2's (Karras et al. 2024); EDM1's -1.2/1.2 concentrated too
    much mass on low noise levels for larger models. Pass them explicitly to
    reproduce an EDM1 run.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, sigma_data: float = 0.5,
                 P_mean: float = -0.4, P_std: float = 1.0):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)
        self.P_mean = P_mean
        self.P_std = P_std

    def sigmas(self, t):
        return jnp.exp(jnp.asarray(t, jnp.float32) / self.T * self.P_std + self.P_mean)

    def t_of_sigma(self, sigma):
        return (jnp.log(jnp.asarray(sigma, jnp.float32)) - self.P_mean) / self.P_std * self.T

    def sample_t(self, key, n):
        return jax.random.normal(key, (n,), dtype=jnp.float32)
