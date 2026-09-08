"""What the model predicts, and how a loss weight crosses between spaces.

A `PredictionTransform` is the parameterization: the target the model is
trained to output at `(x_t, t)` and the way `x_0` and `epsilon` are read back
out of an output. `rates` throughout is the schedule's `(alpha, sigma)` pair,
already shaped to broadcast against the batch; `pred_transform` also sees
`t` itself, for a parameterization whose scaling is a function of the time
rather than of the rates.

A `Weighting` turns a schedule and a parameterization into the per-example
loss weight. The schedule's own weight is stated in the space its paired
parameterization computes the loss in; min-SNR is defined on the x_0 loss
and is converted with `target_error_scale`.
"""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from dew.diffusion.schedules import NoiseScheduler, expand


class PredictionTransform:
    """The identity parameterization: the model outputs x_0's target space."""
    normalize_input: bool = False

    def __init__(self, *, normalize_input: bool = False):
        self.normalize_input = normalize_input

    def pred_transform(self, x_t, preds, rates, t) -> jax.Array:
        """The model's raw output at `(x_t, t)` as a prediction in target space."""
        return preds

    def forward_diffusion(self, x_0, epsilon,
                          rates) -> tuple[jax.Array, ArrayLike, jax.Array]:
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

    def get_input_scale(self, rates) -> ArrayLike:
        if self.normalize_input:
            signal, noise = rates
            return jax.lax.rsqrt(signal ** 2 + noise ** 2)
        return 1

    def target_error_scale(self, snr) -> ArrayLike:
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
    c_in x_t and its raw output F is read as x_0 = c_skip x_t + c_out F. Every
    denominator is at least sigma_data, so none needs a guard.

    `velocity` is Diffusers 0.34.0's EDM `prediction_type="v_prediction"`,
    whose `precondition_outputs` negates c_out: the model's output is the
    velocity of the preconditioned path rather than its endpoint offset.
    """

    def __init__(self, sigma_data: float = 0.5, *, velocity: bool = False) -> None:
        self.sigma_data = sigma_data
        self.velocity = velocity

    def backward_diffusion(self, x_t, preds, rates):
        signal_rate, noise_rate = rates
        return preds, (x_t - preds * signal_rate) / noise_rate

    def pred_transform(self, x_t, preds, rates, t):
        _, sigma = rates
        c_out = sigma * self.sigma_data / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_skip = self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)
        return (-c_out if self.velocity else c_out) * preds + c_skip * x_t

    def get_input_scale(self, rates):
        _, sigma = rates
        return 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)

    def target_error_scale(self, snr):
        # x_0 error is c_out times the raw error, and alpha = 1 here so
        # sigma^2 = 1 / SNR
        return 1 / self.sigma_data ** 2 + snr


class ConsistencyBoundary(PredictionTransform):
    """The boundary parameterization of a latent consistency model (Luo et
    al. 2023, arXiv 2310.04378), as Diffusers' `LCMScheduler` reads it: the
    model predicts x_0 in `inner`'s space, and the consistency function is
    f = c_skip x_t + c_out x_0 with c_skip = sigma_data^2 / (s^2 +
    sigma_data^2) and c_out = s / sqrt(s^2 + sigma_data^2) at the scaled time
    s = `timestep_scaling` t, so f is x_t itself at t = 0. `x_0` and
    `epsilon` are read out of f."""

    def __init__(self, inner: PredictionTransform, timestep_scaling: float = 10.0,
                 sigma_data: float = 0.5) -> None:
        self.inner = inner
        self.timestep_scaling = timestep_scaling
        self.sigma_data = sigma_data

    def pred_transform(self, x_t, preds, rates, t):
        preds = self.inner.pred_transform(x_t, preds, rates, t)
        x_0, _ = self.inner.backward_diffusion(x_t, preds, rates)
        scaled = expand(jnp.asarray(t, jnp.float32) * self.timestep_scaling, x_t)
        c_skip = self.sigma_data ** 2 / (scaled ** 2 + self.sigma_data ** 2)
        c_out = scaled / (scaled ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out * x_0 + c_skip * x_t

    def backward_diffusion(self, x_t, preds, rates):
        signal_rate, noise_rate = rates
        return preds, (x_t - preds * signal_rate) / noise_rate

    def get_input_scale(self, rates):
        return self.inner.get_input_scale(rates)


class SourceLimitedPrediction(PredictionTransform):
    """A published scheduler's limit on x_0, in the conversion order its
    `step` applies it.

    `inner` reads x_0 out of the model's output; then dynamic thresholding
    (Saharia et al. 2022: clamp each sample to its own `ratio` quantile of
    |x_0|, never below 1 and never above `maximum`, and divide by it) or a
    plain clamp to `clip` limits it. Thresholding wins where a source
    declares both, the way its `step` tests them.

    `recompute_epsilon` is whether the source re-derives epsilon from the
    limited x_0. DDPM's posterior, DEIS and the noise-prediction DPM-Solver
    algorithms do, so their update carries the limit; DDIM keeps the model's
    own output as its epsilon and only its x_0 term is limited.

    The limit is not linear in the model's output, so it belongs to the
    conversion a guided walk runs once on the combined output rather than to
    each guidance branch.
    """

    def __init__(self, inner: PredictionTransform, *, clip: float | None = None,
                 threshold: tuple[float, float] | None = None,
                 recompute_epsilon: bool = True) -> None:
        if clip is None and threshold is None:
            raise ValueError("a limited prediction needs a clip range or a thresholding ratio")
        self.inner = inner
        self.clip = clip
        self.threshold = threshold
        self.recompute_epsilon = recompute_epsilon

    def _limit(self, x_0) -> jax.Array:
        if self.threshold is not None:
            ratio, maximum = self.threshold
            flat = jnp.reshape(x_0, (x_0.shape[0], -1))
            level = jnp.quantile(jnp.abs(flat), ratio, axis=1)
            level = expand(jnp.clip(level, 1.0, maximum), x_0)
            return jnp.clip(x_0, -level, level) / level
        return jnp.clip(x_0, -self.clip, self.clip)

    def pred_transform(self, x_t, preds, rates, t):
        return self.inner.pred_transform(x_t, preds, rates, t)

    def backward_diffusion(self, x_t, preds, rates):
        x_0, epsilon = self.inner.backward_diffusion(x_t, preds, rates)
        limited = self._limit(x_0)
        if not self.recompute_epsilon:
            return limited, epsilon
        signal_rate, noise_rate = rates
        return limited, (x_t - signal_rate * limited) / noise_rate

    def get_target(self, x_0, epsilon, rates):
        return self.inner.get_target(x_0, epsilon, rates)

    def get_input_scale(self, rates):
        return self.inner.get_input_scale(rates)

    def target_error_scale(self, snr):
        return self.inner.target_error_scale(snr)


class Weighting(Protocol):
    def __call__(self, schedule: NoiseScheduler, prediction: PredictionTransform,
                 t) -> jax.Array:
        """Per-example loss weight at `t`, shaped like `t`."""
        ...


@dataclass(frozen=True)
class ScheduleWeighting:
    """The schedule's own weight."""

    def __call__(self, schedule, prediction, t):
        return schedule.weight(t)


@dataclass(frozen=True)
class MinSNR:
    """min-SNR-gamma (Hang et al. 2023). min(SNR, gamma) on the x_0 loss,
    converted into the space the model is trained in. It is used in place of
    the schedule's own weight."""

    gamma: float

    def __call__(self, schedule, prediction, t):
        snr = schedule.snr(t)
        return jnp.minimum(snr, self.gamma) / prediction.target_error_scale(snr)


def broadcast_rates(schedule: NoiseScheduler, t, x) -> tuple[jax.Array, jax.Array]:
    """The schedule's rates at `t`, shaped to broadcast against `x`."""
    alpha, sigma = schedule.rates(t)
    return expand(alpha, x), expand(sigma, x)
