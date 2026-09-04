import jax

from .common import NoiseScheduler


class ContinuousNoiseScheduler(NoiseScheduler):
    """A schedule whose time is a fraction of the trajectory, not an index.

    T is 1.0, so t = 1 is fully noised, and training draws t uniformly. A
    subclass gives the rates and the weight of its parameterization.
    """

    T = 1.0

    def sample_t(self, key, n):
        return jax.random.uniform(key, (n,), minval=0.0, maxval=self.T)
