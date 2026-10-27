"""The exponential beta table, FlaxDiff's."""

import numpy as np

from .discrete import BetaTable


def exp_beta_schedule(timesteps, start_angle=0.008, end_angle=0.999):
    """Return the betas of the cumulative alpha exp(-12 t) over `timesteps` steps, FlaxDiff's (15c55b001).

    t runs over linspace(0, 1, timesteps + 1), and each beta, 1 - exp(-12 / timesteps) before
    rounding, is clipped to [0, `end_angle`]. `start_angle` is the offset FlaxDiff's signature
    shares with the cosine table, which this table does not read. The arithmetic is FlaxDiff's,
    operation for operation in float64, so the table is its own to the bit.
    """
    ts = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
    alphas_bar = np.exp(ts * -12.0)
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, end_angle)


class ExpNoiseScheduler(BetaTable):
    """The exponential beta table, FlaxDiff's: the cumulative alpha falls as exp(-12 t).

    `beta_end` is the value each beta is clipped at, and `beta_start` the table's unread offset,
    kept as its record is. The P2 weight arguments are those of `DiscreteNoiseScheduler`.
    """

    table = staticmethod(exp_beta_schedule)

    def __init__(self, timesteps: int, beta_start: float = 0.008, beta_end: float = 0.999,
                 p2_loss_weight_k: float = 1, p2_loss_weight_gamma: float = 1):
        super().__init__(timesteps, beta_start, beta_end, p2_loss_weight_k, p2_loss_weight_gamma)
