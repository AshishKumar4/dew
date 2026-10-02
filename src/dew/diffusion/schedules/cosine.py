"""The cosine beta table of Nichol and Dhariwal 2021."""

import numpy as np

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

