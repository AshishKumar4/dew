"""The cosine beta table of Nichol and Dhariwal 2021."""

import math

import numpy as np

from .discrete import DiscreteNoiseScheduler


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


class CosineNoiseScheduler(DiscreteNoiseScheduler):
    """The cosine beta table of Nichol and Dhariwal 2021.

    `beta_start` is the offset s of `cosine_beta_schedule`, and `beta_end`
    is the value each beta is clipped at. The P2 weight arguments are those
    of `DiscreteNoiseScheduler`.
    """

    def __init__(self, timesteps: int, beta_start: float = 0.008, beta_end: float = 0.999,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(cosine_beta_schedule(timesteps, beta_start, beta_end),
                         p2_loss_weight_k=p2_loss_weight_k,
                         p2_loss_weight_gamma=p2_loss_weight_gamma)
        self.timesteps, self.beta_start, self.beta_end = timesteps, beta_start, beta_end

