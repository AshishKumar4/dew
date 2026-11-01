"""The exponential beta table of FlaxDiff."""

import numpy as np

from .discrete import BetaTable


def exp_beta_schedule(timesteps, beta_start=0.008, beta_end=0.999):
    """Return FlaxDiff's exponential betas: the cumulative alpha is exp(-12 t) on t = i / T, T =
    `timesteps`, and beta_i = 1 - alpha_bar((i + 1) / T) / alpha_bar(i / T), clipped to [0, `beta_end`].

    `beta_start` is FlaxDiff's argument, which its table reads nothing of. The arithmetic is FlaxDiff's
    `exp_beta_schedule` (flaxdiff/schedulers/exp.py at 15c55b001), operation for operation in float64
    NumPy, so the table matches its to the bit; Diffusers' exponential table, which caps every beta at
    0.999 whatever its `beta_end`, is `source_grids.published_betas`'.
    """
    ts = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
    alphas_bar = np.exp(ts * -12.0)
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, beta_end)


class ExpNoiseScheduler(BetaTable):
    """FlaxDiff's exponential beta table (`exp_beta_schedule`); the P2 weight arguments are those of
    `DiscreteNoiseScheduler`."""

    table = staticmethod(exp_beta_schedule)

    def __init__(self, timesteps: int, beta_start: float = 0.008, beta_end: float = 0.999,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(timesteps, beta_start, beta_end, p2_loss_weight_k, p2_loss_weight_gamma)
