import jax.numpy as jnp
from .common import NoiseScheduler


class ContinuousNoiseScheduler(NoiseScheduler):
    """Base for schedulers whose timestep is a continuous variable, not an index.

    Subclasses interpret ``steps`` however their parameterization requires (a
    [0, 1] progress variable, a raw sigma, ...) and implement ``get_rates``.
    ``timesteps`` is kept for interface compatibility with the discrete
    schedulers. It is the upper end of the timestep domain (1.0 by default,
    i.e. "fully noised") and doubles as the switch that makes
    ``generate_timesteps`` sample a float in [0, timesteps) instead of an
    integer index into a table. Continuous schedulers have no beta table, so
    sampling goes through ``get_rates``.
    """
    def __init__(self, dtype=jnp.float32, clip_min=-1.0, clip_max=1.0, min_snr_gamma=None, prediction_transform=None):
        super().__init__(timesteps=1, dtype=dtype, clip_min=clip_min, clip_max=clip_max,
                         min_snr_gamma=min_snr_gamma, prediction_transform=prediction_transform)
