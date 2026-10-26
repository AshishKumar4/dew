"""Noise schedules as values.

A schedule is the forward process x_t = alpha(t) x_0 + sigma(t) eps over a
time domain [0, T]: how training draws t, how the loss weights a draw and what
the model is told about t. It holds no random state and knows nothing about
the parameterization the model predicts in; `Process` pairs the two.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from dew.nn.precision import at_least_fp32, correctly_rounded_divide


def expand(coefficient, x):
    """Return a per-example coefficient `[B]` reshaped to broadcast against `x` `[B, ...]`."""
    return jnp.reshape(coefficient, (-1,) + (1,) * (x.ndim - 1))


def ancestral_variance(sigma, following):
    """following^2 (sigma^2 - following^2) / sigma^2, the variance of the fresh noise an ancestral
    step from `sigma` to `following` adds (k-diffusion's `get_ancestral_step` at eta 1, before its
    root), on a number or an array of either library: each caller takes its own root, cap or clamp."""
    return following ** 2 * (sigma ** 2 - following ** 2) / sigma ** 2


def divided(numerator, rate):
    """`numerator / rate` as a source written in PyTorch divides a state by a rate: IEEE division,
    whatever the backend's own division does. A float32, bfloat16 or float16 quotient is
    `dew.nn.precision.correctly_rounded_divide`'s of the operands widened to float32, rounded to its
    dtype, as PyTorch's float32 opmath gives it, under any compiler flags. A float64 one is the plain
    division."""
    numerator, rate = jnp.asarray(numerator), jnp.asarray(rate)
    dtype = jnp.result_type(numerator, rate)
    if not jnp.issubdtype(dtype, jnp.floating):
        return numerator / rate
    if dtype == jnp.float64:
        return numerator / rate
    return correctly_rounded_divide(numerator.astype(jnp.float32), rate.astype(jnp.float32)).astype(dtype)


def times(t) -> jax.Array:
    """`t` as an array of at least float32: a float64 run's times stay float64."""
    t = jnp.asarray(t)
    return t.astype(at_least_fp32(t.dtype))

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
        """Draw `n` training times from this schedule's training distribution, independent and alike:
        an objective draws each example's time as `sample_t(its key, 1)` (`Step.sample`), so a draw
        that couples the `n` (stratified, antithetic) is the objective's, over the update's places
        (`Step.places`), as MDLM's is."""

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
        return times(t)

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
        return times(t) - times(t_next)

    def half_interval(self, t, t_next) -> jax.Array:
        """Return half a grid interval, where a Runge-Kutta stage places its intermediate points."""
        return (times(t) - times(t_next)) / 2

    def variances(self, t) -> tuple[jax.Array, jax.Array]:
        """Return `(alpha^2, sigma^2)` at `t`, shaped like `t`: the rates squared, unless the
        schedule holds the squares themselves, as a beta table's cumulative products are, whose
        differences a transfer between two of them reads more exactly than the squares of its
        rounded roots (`DiscreteNoiseScheduler`)."""
        alpha, sigma = self.rates(t)
        return alpha ** 2, sigma ** 2

    def snr(self, t) -> jax.Array:
        alpha, sigma = self.rates(t)
        return (alpha / sigma) ** 2


class ContinuousNoiseScheduler(NoiseScheduler):
    """A schedule whose time is a fraction of the trajectory, not an index.

    T is 1.0, so t = 1 is fully noised, and training draws t uniformly. A
    subclass provides the rates and the weight of its parameterization.
    """

    T = 1.0

    def sample_t(self, key, n):
        return jax.random.uniform(key, (n,), jnp.float32, 0.0, self.T)


class GeneralizedNoiseScheduler(ContinuousNoiseScheduler):
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
        sigma = self.sigmas(times(t))
        return jnp.ones_like(sigma), sigma

    def weight(self, t):
        sigma = self.sigmas(times(t))
        # Eq. 8's lambda, written as a sum that needs no epsilon guard.
        return 1 / self.sigma_data ** 2 + 1 / sigma ** 2

    def model_time(self, t):
        return jnp.log(self.sigmas(times(t))) / 4
