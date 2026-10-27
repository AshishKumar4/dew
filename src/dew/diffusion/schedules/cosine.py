"""The cosine schedules: Nichol and Dhariwal 2021's beta table, and FlaxDiff's continuous cosine and its
tangent grid of sigmas."""

import math

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
    """FlaxDiff's continuous cosine schedule, variance preserving: alpha = cos(pi t / 2) and
    sigma = sin(pi t / 2), each step weighted 1 / (1 + SNR), which is sigma^2. The arithmetic is
    FlaxDiff's `CosineContinuousNoiseScheduler` (flaxdiff/schedulers/cosine.py at 15c55b001),
    operation for operation in float32."""

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        return jnp.cos((jnp.pi * t) / 2), jnp.sin((jnp.pi * t) / 2)

    def weight(self, t):
        alpha, sigma = self.rates(t)
        return 1 / (1 + (alpha ** 2 / sigma ** 2))


class CosineGeneralNoiseScheduler(GeneralizedNoiseScheduler):
    """FlaxDiff's tangent grid of sigmas, variance exploding: sigma(t) = tan(theta(t)) / kappa, theta
    moving linearly from atan(sigma_min / kappa) at t = 0 to atan(sigma_max / kappa) at t = 1, the
    angles of FlaxDiff's log-SNR bounds. With kappa 1 the grid runs from sigma_min to sigma_max; with
    another it runs from sigma_min / kappa^2, as FlaxDiff's divides the tangent by kappa again. The
    arithmetic is FlaxDiff's `CosineGeneralNoiseScheduler` (flaxdiff/schedulers/cosine.py at
    15c55b001); its weight and model time are the EDM family's (`GeneralizedNoiseScheduler`)."""

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 80.0, kappa: float = 1.0,
                 sigma_data: float = 0.5):
        super().__init__(sigma_min, sigma_max, sigma_data)
        self.kappa = kappa
        logsnr_max = 2 * (math.log(self.kappa) - math.log(self.sigma_max))
        self.theta_max = math.atan(math.exp(-0.5 * logsnr_max))
        logsnr_min = 2 * (math.log(self.kappa) - math.log(self.sigma_min))
        self.theta_min = math.atan(math.exp(-0.5 * logsnr_min))

    def sigmas(self, t):
        return jnp.tan(self.theta_min + times(t) * (self.theta_max - self.theta_min)) / self.kappa

    def t_of_sigma(self, sigma):
        return (jnp.arctan(jnp.asarray(sigma, jnp.float32) * self.kappa) - self.theta_min) / (
            self.theta_max - self.theta_min)
