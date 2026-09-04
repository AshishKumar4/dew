"""Noise schedules as values.

A schedule is the forward process x_t = alpha(t) x_0 + sigma(t) eps over a
time domain [0, T]: how training draws t, how the loss weights a draw and what
the model is told about t. It holds no random state and knows nothing about
the parameterization the model predicts in; `Process` pairs the two.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


def expand(coefficient, x):
    """A per-example coefficient `[B]` shaped to broadcast against `x` `[B, ...]`."""
    return jnp.reshape(coefficient, (-1,) + (1,) * (x.ndim - 1))


class NoiseScheduler(ABC):
    """The forward process on [0, T], with t = T the fully noised end."""

    T: float

    @abstractmethod
    def rates(self, t) -> tuple[jax.Array, jax.Array]:
        """`(alpha, sigma)` at `t`, shaped like `t`."""

    @abstractmethod
    def sample_t(self, key, n: int) -> jax.Array:
        """`n` training times drawn the way this schedule trains."""

    @abstractmethod
    def weight(self, t) -> jax.Array:
        """The schedule's own loss weight at `t`, in the space its paired
        parameterization computes the loss in."""

    def model_time(self, t) -> jax.Array:
        """What the model is conditioned on at `t`; the time itself unless the
        schedule says otherwise."""
        return jnp.asarray(t, jnp.float32)

    def snr(self, t) -> jax.Array:
        alpha, sigma = self.rates(t)
        return (alpha / sigma) ** 2


class GeneralizedNoiseScheduler(NoiseScheduler):
    """The variance exploding family of Karras et al. 2022 ("Elucidating the
    Design Space of Diffusion-Based Generative Models").

    alpha is 1 and the model input is scaled by the paired preconditioning
    instead. Every member conditions the model on c_noise = log(sigma) / 4
    and weights the loss with lambda(sigma) = (sigma^2 + sigma_data^2) /
    (sigma sigma_data)^2 (Eq. 8 of the paper), written in a form that needs
    no epsilon guard; a subclass places the sigmas along t and inverts that
    placement for the solvers that step in sigma.
    """

    T = 1.0

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0,
                 sigma_data: float = 0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    @abstractmethod
    def sigmas(self, t) -> jax.Array:
        """The noise level at `t`."""

    @abstractmethod
    def t_of_sigma(self, sigma) -> jax.Array:
        """The inverse of `sigmas`."""

    def rates(self, t):
        sigma = self.sigmas(jnp.asarray(t, jnp.float32))
        return jnp.ones_like(sigma), sigma

    def sample_t(self, key, n):
        return jax.random.uniform(key, (n,), minval=0.0, maxval=self.T)

    def weight(self, t):
        sigma = self.sigmas(jnp.asarray(t, jnp.float32))
        return 1 / self.sigma_data ** 2 + 1 / sigma ** 2

    def model_time(self, t):
        return jnp.log(self.sigmas(jnp.asarray(t, jnp.float32))) / 4
