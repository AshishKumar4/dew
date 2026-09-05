import numpy as np

from .discrete import DiscreteNoiseScheduler


def exp_beta_schedule(timesteps, beta_end=0.999):
    """Betas whose cumulative alpha decays as exp(-12 t), each clipped at beta_end."""
    ts = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
    alphas_bar = np.exp(ts * -12.0)
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, beta_end)


class ExpNoiseScheduler(DiscreteNoiseScheduler):
    """A beta table whose cumulative alpha decays as exp(-12 t)."""

    def __init__(self, timesteps: int, beta_end: float = 0.999,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(exp_beta_schedule(timesteps, beta_end),
                         p2_loss_weight_k=p2_loss_weight_k,
                         p2_loss_weight_gamma=p2_loss_weight_gamma)
