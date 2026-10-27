"""The cosine schedules: Nichol and Dhariwal 2021's beta table, and FlaxDiff's continuous cosine and its
tangent grid of sigmas."""

import math

import jax.numpy as jnp
import numpy as np

from dew.nn.precision import divided

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
    operation for operation, in float32 or a float64 run's float64 (`times`)."""

    def rates(self, t):
        t = times(t)
        return jnp.cos((jnp.pi * t) / 2), jnp.sin((jnp.pi * t) / 2)

    def weight(self, t):
        alpha, sigma = self.rates(t)
        return 1 / (1 + (alpha ** 2 / sigma ** 2))


class CosineGeneralNoiseScheduler(GeneralizedNoiseScheduler):
    """FlaxDiff's tangent grid of sigmas, variance exploding: sigma(t) = tan(theta(t)) / kappa, theta
    moving linearly from atan(sigma_min / kappa) at t = 0 to atan(sigma_max / kappa) at t = 1, the
    angles of FlaxDiff's log-SNR bounds. With kappa 1 the grid runs from sigma_min to sigma_max; with
    another it runs from sigma_min / kappa^2 to sigma_max / kappa^2, as FlaxDiff's divides the tangent
    by kappa again. The schedule's `sigma_min` and `sigma_max` are the grid's own ends, which a solver
    keeps its sigmas inside (Heun's churn, `BrownianTree`); `nominal` holds the two arguments.

    The arithmetic is FlaxDiff's `CosineGeneralNoiseScheduler` and the `GeneralizedNoiseScheduler`
    it extends (flaxdiff/schedulers/cosine.py and common.py at 15c55b001): its grid operation for
    operation, each step weighted 1 + sigma^2 / sigma_max^2, its nominal sigma_max (`weight`), the
    model conditioned on the step's progress in whole thousandths (`model_time`), and sigma_data 1,
    not the EDM family's weight, log-sigma conditioning and 0.5. In float32, or a float64 run's
    float64 (`times`). The weight is FlaxDiff's function written without its cancellation: FlaxDiff
    forms sigma^2 as 1 / (1 + (1 - sigma^2) / sigma^2), whose sum cancels as sigma grows, so its
    float32 weight is 4e-4 off at the default grid's top and 2% off at a sigma of 640.

    `t_of_sigma`, which FlaxDiff leaves unwritten, divides as IEEE division does (`divided`), not as
    XLA's product with the reciprocal of the constant: that product rounds the time 1.7 times as far
    from float64 as the formula in NumPy's float32 at a kappa of 2."""

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 80.0, kappa: float = 1.0,
                 sigma_data: float = 1.0):
        self.kappa, self.nominal = kappa, (sigma_min, sigma_max)
        logsnr_max = 2 * (math.log(self.kappa) - math.log(sigma_max))
        self.theta_max = math.atan(math.exp(-0.5 * logsnr_max))
        logsnr_min = 2 * (math.log(self.kappa) - math.log(sigma_min))
        self.theta_min = math.atan(math.exp(-0.5 * logsnr_min))
        super().__init__(math.tan(self.theta_min) / kappa, math.tan(self.theta_max) / kappa, sigma_data)

    def sigmas(self, t):
        return jnp.tan(self.theta_min + times(t) * (self.theta_max - self.theta_min)) / self.kappa

    def t_of_sigma(self, sigma):
        angle = jnp.arctan(times(sigma) * self.kappa)
        return divided(angle - self.theta_min, self.theta_max - self.theta_min)

    def weight(self, t):
        return 1 + divided(jnp.square(self.sigmas(t)), self.nominal[1] ** 2)

    def model_time(self, t):
        # FlaxDiff's `transform_inputs`: the progress in whole thousandths, truncated as its int32 cast is.
        return jnp.trunc(times(t) * 1000)
