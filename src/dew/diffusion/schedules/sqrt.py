import jax.numpy as jnp

from .continuous import ContinuousNoiseScheduler


class SqrtContinuousNoiseScheduler(ContinuousNoiseScheduler):
    """Square-root schedule from Diffusion-LM (Li et al. 2022).

    alpha(t) = sqrt(1 - t) and sigma(t) = sqrt(t) for t in [0, 1], so it is
    variance preserving (alpha^2 + sigma^2 = 1) with SNR(t) = (1 - t) / t.
    Noise ramps up much faster near t = 0 than the cosine schedule, which is
    what the paper wanted for discrete/embedding data: the low-noise end
    carries little signal about the token identity, so spending fewer steps
    there is a win. The paper trains the plain x_0 loss, so the weight is one.
    """

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        return jnp.sqrt(1 - t), jnp.sqrt(t)

    def weight(self, t):
        return jnp.ones_like(jnp.asarray(t, jnp.float32))
