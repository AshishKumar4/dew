"""The linear beta table of Ho et al. 2020."""

import numpy as np

from .discrete import DiscreteNoiseScheduler


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Return DDPM's betas, linear from `beta_start` to `beta_end` over 1000 steps.

    The table is Ho et al. 2020's. For another step count, the betas are
    scaled by 1000 / `timesteps` so the cumulative alpha stays roughly the
    same.
    """
    scale = 1000 / timesteps
    return np.linspace(scale * beta_start, scale * beta_end, timesteps, dtype=np.float64)


class LinearNoiseScheduler(DiscreteNoiseScheduler):
    """The linear beta table of DDPM, scaled to the step count.

    DDPM is Ho et al. 2020.
    """

    def __init__(self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(linear_beta_schedule(timesteps, beta_start, beta_end),
                         p2_loss_weight_k=p2_loss_weight_k,
                         p2_loss_weight_gamma=p2_loss_weight_gamma)
        self.timesteps, self.beta_start, self.beta_end = timesteps, beta_start, beta_end
