import math

import jax.numpy as jnp
import numpy as np

from .common import GeneralizedNoiseScheduler
from .continuous import ContinuousNoiseScheduler
from .discrete import DiscreteNoiseScheduler


def cosine_beta_schedule(timesteps, start_angle=0.008, end_angle=0.999):
    """Nichol and Dhariwal 2021, Eq. 17: the cumulative alpha follows
    cos^2((t / T + s) / (1 + s) * pi / 2), s = start_angle, and each beta is
    clipped at end_angle."""
    ts = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
    alphas_bar = np.cos((ts + start_angle) / (1 + start_angle) * np.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, end_angle)


class CosineNoiseScheduler(DiscreteNoiseScheduler):
    """The cosine beta table of Nichol and Dhariwal 2021."""

    def __init__(self, timesteps: int, beta_start: float = 0.008, beta_end: float = 0.999,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(cosine_beta_schedule(timesteps, beta_start, beta_end),
                         p2_loss_weight_k=p2_loss_weight_k,
                         p2_loss_weight_gamma=p2_loss_weight_gamma)


class CosineGeneralNoiseScheduler(GeneralizedNoiseScheduler):
    """Sigmas placed so that log-SNR runs along a cosine in t, on the
    variance exploding form."""

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 80.0, kappa: float = 1.0,
                 sigma_data: float = 0.5):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)
        self.kappa = kappa
        logsnr_max = 2 * (math.log(self.kappa) - math.log(self.sigma_max))
        self.theta_max = math.atan(math.exp(-0.5 * logsnr_max))
        logsnr_min = 2 * (math.log(self.kappa) - math.log(self.sigma_min))
        self.theta_min = math.atan(math.exp(-0.5 * logsnr_min))

    def sigmas(self, t):
        return jnp.tan(self.theta_min + t * (self.theta_max - self.theta_min)) / self.kappa

    def t_of_sigma(self, sigma):
        theta = jnp.arctan(jnp.asarray(sigma, jnp.float32) * self.kappa)
        return (theta - self.theta_min) / (self.theta_max - self.theta_min)


class CosineContinuousNoiseScheduler(ContinuousNoiseScheduler):
    """alpha = cos(pi t / 2), sigma = sin(pi t / 2), weighted by sigma^2, which
    is 1 / (1 + SNR)."""

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        return jnp.cos(jnp.pi * t / 2), jnp.sin(jnp.pi * t / 2)

    def weight(self, t):
        alpha, sigma = self.rates(t)
        return 1 / (1 + (alpha ** 2 / sigma ** 2))
