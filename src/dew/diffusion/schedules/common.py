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
    """Return a per-example coefficient `[B]` reshaped to broadcast against `x` `[B, ...]`."""
    return jnp.reshape(coefficient, (-1,) + (1,) * (x.ndim - 1))


class NoiseScheduler(ABC):
    """Defines a forward process x_t = alpha(t) x_0 + sigma(t) eps on the time domain [0, T].

    t = T is the fully noised end.
    """

    T: float

    @abstractmethod
    def rates(self, t) -> tuple[jax.Array, jax.Array]:
        """Return `(alpha, sigma)` at `t`, shaped like `t`."""

    @abstractmethod
    def sample_t(self, key, n: int) -> jax.Array:
        """Draw `n` training times from this schedule's training distribution."""

    @abstractmethod
    def weight(self, t) -> jax.Array:
        """Return the schedule's own loss weight at `t`.

        The weight is in the space where the schedule's paired
        parameterization computes the loss.
        """

    def model_time(self, t) -> jax.Array:
        """Return the time value the model is conditioned on at `t`.

        By default this is `t` itself, and a schedule can override it.
        """
        return jnp.asarray(t, jnp.float32)

    def prior_scale(self) -> jax.Array:
        """Return the standard deviation of the initial Gaussian draw.

        By default this is sqrt(alpha^2 + sigma^2) at T.
        """
        alpha, sigma = self.rates(jnp.asarray(self.T))
        return jnp.sqrt(alpha ** 2 + sigma ** 2)

    def step_interval(self, t, t_next) -> jax.Array:
        """Return the transfer interval from `t` to `t_next`.

        An ordinary grid advances to its next point, so the interval is
        `t - t_next`.
        """
        return jnp.asarray(t, jnp.float32) - jnp.asarray(t_next, jnp.float32)

    def half_interval(self, t, t_next) -> jax.Array:
        """Return half a grid interval, where a Runge-Kutta stage places its intermediate points."""
        return (jnp.asarray(t, jnp.float32) - jnp.asarray(t_next, jnp.float32)) / 2

    def snr(self, t) -> jax.Array:
        alpha, sigma = self.rates(t)
        return (alpha / sigma) ** 2


class GeneralizedNoiseScheduler(NoiseScheduler):
    """The variance-exploding schedule family of the EDM paper.

    The paper is Karras et al. 2022, "Elucidating the Design Space of
    Diffusion-Based Generative Models". alpha is 1, and the paired
    preconditioning scales the model input. Every member conditions the
    model on c_noise = log(sigma) / 4 and weights the loss with
    lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma sigma_data)^2 (Eq. 8
    of the paper). A subclass places the sigmas along t (`sigmas`) and
    inverts that placement (`t_of_sigma`) for the solvers that step in
    sigma.
    """

    T = 1.0

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0,
                 sigma_data: float = 0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    @abstractmethod
    def sigmas(self, t) -> jax.Array:
        """Return the noise level at `t`."""

    @abstractmethod
    def t_of_sigma(self, sigma) -> jax.Array:
        """Return the time at which `sigmas` gives `sigma`, the inverse of `sigmas`."""

    def rates(self, t):
        sigma = self.sigmas(jnp.asarray(t, jnp.float32))
        return jnp.ones_like(sigma), sigma

    def sample_t(self, key, n):
        return jax.random.uniform(key, (n,), minval=0.0, maxval=self.T)

    def weight(self, t):
        sigma = self.sigmas(jnp.asarray(t, jnp.float32))
        # Eq. 8's lambda, written as a sum that needs no epsilon guard.
        return 1 / self.sigma_data ** 2 + 1 / sigma ** 2

    def model_time(self, t):
        return jnp.log(self.sigmas(jnp.asarray(t, jnp.float32))) / 4
