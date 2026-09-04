import numpy as np

from .discrete import DiscreteNoiseScheduler


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


class LinearNoiseScheduler(DiscreteNoiseScheduler):
    """The linear beta table of Ho et al. 2020, scaled to the step count."""

    def __init__(self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(timesteps, beta_start, beta_end, schedule_fn=linear_beta_schedule,
                         p2_loss_weight_k=p2_loss_weight_k,
                         p2_loss_weight_gamma=p2_loss_weight_gamma)
