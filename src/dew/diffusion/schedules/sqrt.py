"""The square-root schedule of Diffusion-LM."""

import jax.numpy as jnp

from .common import ContinuousNoiseScheduler


class SqrtContinuousNoiseScheduler(ContinuousNoiseScheduler):
    """Square-root schedule from Diffusion-LM.

    Diffusion-LM is Li et al. 2022. The cumulative alpha is 1 - sqrt(t + s),
    s = 1e-4, normalized to one at t = 0 the way Diffusion-LM's
    `betas_for_alpha_bar` normalizes it:

        alpha^2(t) = (1 - sqrt(t + s)) / (1 - sqrt(s)),   sigma^2 = 1 - alpha^2,

    so at Diffusion-LM's step k of T the rates are its table's at
    t = (k + 1) / T. Noise rises like t^(1/4) from t = 0, much faster than
    in the cosine schedule, so fewer steps are spent where an embedding
    carries little noise. The cumulative alpha reaches zero just before
    t = 1, and the rates stay at (0, 1) from there; Diffusion-LM's last step
    instead clips its beta at 0.999. At t >= 1 - s, alpha's derivative in t
    is infinite, so an objective that differentiates the rates in time
    (MeanFlow, consistency training) must not draw t from the last 1e-4.
    The paper trains the plain x_0 loss, so the weight is one.
    """

    s: float = 1e-4

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        alpha_bar = jnp.clip((1 - jnp.sqrt(t + self.s)) / (1 - jnp.sqrt(self.s)), 0.0, 1.0)
        return jnp.sqrt(alpha_bar), jnp.sqrt(1 - alpha_bar)

    def weight(self, t):
        return jnp.ones_like(jnp.asarray(t, jnp.float32))
