"""What the model predicts, and how a loss weight crosses between spaces.

A `PredictionTransform` is the parameterization: the target the model is
trained to output at `(x_t, t)` and the way `x_0` and `epsilon` are read back
out of an output. `rates` throughout is the schedule's `(alpha, sigma)` pair,
already shaped to broadcast against the batch.

A `Weighting` turns a schedule and a parameterization into the per-example
loss weight. The schedule's own weight is stated in the space its paired
parameterization computes the loss in; min-SNR is defined on the x_0 loss
and is converted with `target_error_scale`.
"""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from dew.diffusion.schedules import NoiseScheduler, expand


class PredictionTransform:
    """The identity parameterization: the model outputs x_0's target space."""

    def pred_transform(self, x_t, preds, rates) -> jax.Array:
        return preds

    def forward_diffusion(self, x_0, epsilon, rates) -> tuple[jax.Array, jax.Array, jax.Array]:
        """`(x_t, c_in, target)`: the noised sample, the model input scale and
        what the model should output for it."""
        signal_rate, noise_rate = rates
        x_t = signal_rate * x_0 + noise_rate * epsilon
        return x_t, self.get_input_scale(rates), self.get_target(x_0, epsilon, rates)

    def backward_diffusion(self, x_t, preds, rates) -> tuple[jax.Array, jax.Array]:
        """`(x_0, epsilon)` read out of a prediction in target space."""
        raise NotImplementedError

    def get_target(self, x_0, epsilon, rates) -> jax.Array:
        return x_0

    def get_input_scale(self, rates):
        return 1

    def target_error_scale(self, snr) -> jax.Array:
        """||target error||^2 / ||x_0 error||^2 at the given SNR.

        Loss weights (min-SNR-gamma and friends) are defined on the x_0 loss;
        dividing by this converts them into the space the model is trained in.
        """
        return 1.0


class EpsilonPredictionTransform(PredictionTransform):
    def backward_diffusion(self, x_t, preds, rates):
        signal_rates, noise_rates = rates
        return (x_t - preds * noise_rates) / signal_rates, preds

    def get_target(self, x_0, epsilon, rates):
        return epsilon

    def target_error_scale(self, snr):
        return snr


class DirectPredictionTransform(PredictionTransform):
    def backward_diffusion(self, x_t, preds, rates):
        signal_rate, noise_rate = rates
        return preds, (x_t - preds * signal_rate) / noise_rate


class VPredictionTransform(PredictionTransform):
    """v = alpha eps - sigma x_0, normalized by the total variance."""

    def backward_diffusion(self, x_t, preds, rates):
        signal_rate, noise_rate = rates
        variance = signal_rate ** 2 + noise_rate ** 2
        v = preds * jnp.sqrt(variance)
        x_0 = signal_rate * x_t - noise_rate * v
        eps_0 = signal_rate * v + noise_rate * x_t
        return x_0 / variance, eps_0 / variance

    def get_target(self, x_0, epsilon, rates):
        signal_rate, noise_rate = rates
        v = signal_rate * epsilon - noise_rate * x_0
        variance = signal_rate ** 2 + noise_rate ** 2
        return v / jnp.sqrt(variance)

    def target_error_scale(self, snr):
        return snr + 1


class FlowMatchPredictionTransform(PredictionTransform):
    """Rectified flow velocity: the model predicts u = epsilon - x_0, the
    constant velocity of the linear path, so both endpoints are one step away.
    """

    def backward_diffusion(self, x_t, preds, rates):
        signal_rate, noise_rate = rates
        return x_t - noise_rate * preds, x_t + signal_rate * preds

    def get_target(self, x_0, epsilon, rates):
        return epsilon - x_0

    def target_error_scale(self, snr):
        # x_0 error is t times the velocity error, and t = 1 / (1 + sqrt(SNR))
        return (1 + jnp.sqrt(snr)) ** 2


class KarrasPredictionTransform(PredictionTransform):
    """The EDM preconditioning (Karras et al. 2022, Table 1): the model sees
    c_in x_t and its raw output F is read as x_0 = c_skip x_t + c_out F."""

    def __init__(self, sigma_data: float = 0.5) -> None:
        self.sigma_data = sigma_data

    def backward_diffusion(self, x_t, preds, rates):
        signal_rate, noise_rate = rates
        return preds, (x_t - preds * signal_rate) / noise_rate

    def pred_transform(self, x_t, preds, rates, epsilon=1e-8):
        _, sigma = rates
        c_out = sigma * self.sigma_data / (jnp.sqrt(self.sigma_data ** 2 + sigma ** 2) + epsilon)
        c_skip = self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2 + epsilon)
        return c_out * preds + c_skip * x_t

    def get_input_scale(self, rates, epsilon=1e-8):
        _, sigma = rates
        return 1 / (jnp.sqrt(self.sigma_data ** 2 + sigma ** 2) + epsilon)

    def target_error_scale(self, snr):
        # x_0 error is c_out times the raw error, and alpha = 1 here so
        # sigma^2 = 1 / SNR
        return 1 / self.sigma_data ** 2 + snr


class Weighting(Protocol):
    def __call__(self, schedule: NoiseScheduler, prediction: PredictionTransform, t) -> jax.Array:
        """Per-example loss weight at `t`, shaped like `t`."""


@dataclass(frozen=True)
class ScheduleWeighting:
    """The schedule's own weight."""

    def __call__(self, schedule, prediction, t):
        return schedule.weight(t)


@dataclass(frozen=True)
class MinSNR:
    """min-SNR-gamma (Hang et al. 2023): min(SNR, gamma) on the x_0 loss,
    converted into the space the model is trained in. It replaces the
    schedule's own weight rather than stacking on top of it."""

    gamma: float

    def __call__(self, schedule, prediction, t):
        snr = schedule.snr(t)
        return jnp.minimum(snr, self.gamma) / prediction.target_error_scale(snr)


def broadcast_rates(schedule: NoiseScheduler, t, x) -> tuple[jax.Array, jax.Array]:
    """The schedule's rates at `t`, shaped to broadcast against `x`."""
    alpha, sigma = schedule.rates(t)
    return expand(alpha, x), expand(sigma, x)
