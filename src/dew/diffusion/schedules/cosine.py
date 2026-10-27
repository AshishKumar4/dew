"""The cosine schedules: Nichol and Dhariwal's beta table, the continuous variance-preserving cosine
and the tangent sigma grid of a variance-exploding cosine."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from .common import ContinuousNoiseScheduler, GeneralizedNoiseScheduler, times
from .discrete import BetaTable


def cosine_beta_schedule(timesteps, start_angle=0.008, end_angle=0.999):
    """Return the cosine beta table of Nichol and Dhariwal 2021.

    The table is their Eq. 17. The cumulative alpha follows
    f(t) = cos^2((t + s) / (1 + s) * pi / 2) at t = i / T, with
    T = `timesteps` and s = `start_angle`, and
    beta_i = 1 - f((i + 1) / T) / f(i / T), clipped at `end_angle`. The
    arithmetic is improved-diffusion's `betas_for_alpha_bar`, operation for
    operation and in scalar `math.cos` (NumPy's vectorized cosine rounds a
    few arguments differently), so the table matches the authors' to the
    bit.
    """
    def alpha_bar(t):
        return math.cos((t + start_angle) / (1 + start_angle) * math.pi / 2) ** 2
    return np.array([min(1 - alpha_bar((i + 1) / timesteps) / alpha_bar(i / timesteps), end_angle)
                     for i in range(timesteps)], np.float64)


class CosineNoiseScheduler(BetaTable):
    """The cosine beta table of Nichol and Dhariwal 2021.

    `beta_start` is the offset s of `cosine_beta_schedule`, and `beta_end`
    is the value each beta is clipped at. The P2 weight arguments are those
    of `DiscreteNoiseScheduler`.
    """

    table = staticmethod(cosine_beta_schedule)

    def __init__(self, timesteps: int, beta_start: float = 0.008, beta_end: float = 0.999,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(timesteps, beta_start, beta_end, p2_loss_weight_k, p2_loss_weight_gamma)



class CosineContinuousNoiseScheduler(ContinuousNoiseScheduler):
    """The continuous variance-preserving cosine schedule, FlaxDiff's (15c55b001).

    alpha(t) = cos(pi t / 2) and sigma(t) = sin(pi t / 2) on t in [0, 1], so alpha^2 + sigma^2 = 1
    and the SNR is cot^2(pi t / 2). The loss weight is 1 / (1 + SNR), sigma^2 itself, as the P2
    weight at k = 1, gamma = 1 of a beta table is: zero at t = 0, one at t = 1. Training draws t
    uniformly and the model reads t.
    """

    def rates(self, t):
        angle = jnp.pi * times(t) / (2 * self.T)
        return jnp.cos(angle), jnp.sin(angle)

    def weight(self, t):
        alpha, sigma = self.rates(t)
        # FlaxDiff's 1 / (1 + alpha^2 / sigma^2), with no division by zero at t = 0.
        return sigma ** 2 / (alpha ** 2 + sigma ** 2)


class CosineGeneralNoiseScheduler(GeneralizedNoiseScheduler):
    """The variance-exploding cosine schedule, FlaxDiff's tangent sigma grid (15c55b001).

    Each end's log-SNR, 2 (log kappa - log sigma), gives its angle theta = atan(exp(-logSNR / 2)),
    atan(sigma / kappa), and the grid is uniform in the angle between them:

        sigma(t) = tan(theta_min + t (theta_max - theta_min)) / kappa,

    alpha = 1. At kappa = 1 the grid runs from `sigma_min` to `sigma_max`; another kappa divides both
    ends by kappa^2, as FlaxDiff's does. It keeps FlaxDiff's own loss weight, its generalized schedule's
    1 + sigma^2 / sigma_max^2, and tells the model the thousandth of the trajectory t falls in,
    floor(1000 t), as FlaxDiff's generalized schedule does; `sigma_data` is that schedule's default, 1.
    """

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 80.0, kappa: float = 1.0,
                 sigma_data: float = 1.0):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data)
        self.kappa = kappa
        # The ends' angles, through their log-SNRs as FlaxDiff takes them, so the doubles are its own.
        self.theta_max = math.atan(math.exp(-0.5 * (2 * (math.log(kappa) - math.log(sigma_max)))))
        self.theta_min = math.atan(math.exp(-0.5 * (2 * (math.log(kappa) - math.log(sigma_min)))))

    def sigmas(self, t):
        angle = self.theta_min + times(t) / self.T * (self.theta_max - self.theta_min)
        return jnp.tan(angle) / self.kappa

    def t_of_sigma(self, sigma):
        angle = jnp.arctan(self.kappa * times(sigma))
        return (angle - self.theta_min) / (self.theta_max - self.theta_min) * self.T

    def weight(self, t):
        return 1 + self.sigmas(t) ** 2 / self.sigma_max ** 2

    def model_time(self, t) -> jax.Array:
        return jnp.floor(times(t) / self.T * 1000)
