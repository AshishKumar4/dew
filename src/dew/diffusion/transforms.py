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


def from_clean(x_t, prediction, rates) -> tuple[jax.Array, jax.Array]:
    """`(x_0, epsilon)` where the prediction is already x_0 itself.

    epsilon is what is left of `x_t` once the clean part is taken out of it,
    which is the reading every x_0-space parameterization shares.
    """
    signal_rate, noise_rate = rates
    return prediction, (x_t - prediction * signal_rate) / noise_rate


class PredictionTransform:
    """Defines what the model predicts and how x_0 and epsilon are recovered from its output.

    The base class provides the x_0 target and the identity output
    transform. A subclass must implement `backward_diffusion`, because the
    base raises `NotImplementedError`. With `normalize_input`, the model's
    input is scaled by 1 / sqrt(alpha^2 + sigma^2).

    `rates` in every method is the schedule's `(alpha, sigma)` pair, already
    shaped to broadcast against the batch.
    """

    normalize_input: bool = False

    def __init__(self, *, normalize_input: bool = False):
        self.normalize_input = normalize_input

    def pred_transform(self, x_t, prediction, rates, t) -> jax.Array:
        """Return the model's raw output at `(x_t, t)` as a prediction in target space.

        `t` is passed for a parameterization whose scaling depends on the
        time itself and not on the rates, such as `ConsistencyBoundary`.
        """
        return prediction

    def forward_diffusion(self, x_0, epsilon,
                          rates) -> tuple[jax.Array, ArrayLike, jax.Array]:
        """Return `(x_t, c_in, target)` for a clean sample and its noise.

        x_t is the noised sample alpha x_0 + sigma epsilon, c_in is the scale
        applied to the model's input, and target is what the model should
        output for x_t.
        """
        signal_rate, noise_rate = rates
        x_t = signal_rate * x_0 + noise_rate * epsilon
        return x_t, self.get_input_scale(rates), self.get_target(x_0, epsilon, rates)

    def backward_diffusion(self, x_t, prediction, rates) -> tuple[jax.Array, jax.Array]:
        """Return `(x_0, epsilon)` recovered from a prediction in target space."""
        raise NotImplementedError

    def get_target(self, x_0, epsilon, rates) -> jax.Array:
        return x_0

    def get_input_scale(self, rates) -> ArrayLike:
        if self.normalize_input:
            signal, noise = rates
            return jax.lax.rsqrt(signal ** 2 + noise ** 2)
        return 1

    def target_error_scale(self, snr) -> ArrayLike:
        """Return ||target error||^2 / ||x_0 error||^2 at the given SNR.

        min-SNR-gamma and the other loss weights are defined on the x_0 loss.
        Dividing such a weight by this ratio converts it into the space the
        model trains in.
        """
        return 1.0


class EpsilonPredictionTransform(PredictionTransform):
    def backward_diffusion(self, x_t, prediction, rates):
        signal_rates, noise_rates = rates
        return (x_t - prediction * noise_rates) / signal_rates, prediction

    def get_target(self, x_0, epsilon, rates):
        return epsilon

    def target_error_scale(self, snr):
        return snr


class DirectPredictionTransform(PredictionTransform):
    def backward_diffusion(self, x_t, prediction, rates):
        return from_clean(x_t, prediction, rates)


class VPredictionTransform(PredictionTransform):
    """The model predicts v = alpha eps - sigma x_0, divided by sqrt(alpha^2 + sigma^2)."""

    def backward_diffusion(self, x_t, prediction, rates):
        signal_rate, noise_rate = rates
        variance = signal_rate ** 2 + noise_rate ** 2
        v = prediction * jnp.sqrt(variance)
        x_0 = signal_rate * x_t - noise_rate * v
        epsilon = signal_rate * v + noise_rate * x_t
        return x_0 / variance, epsilon / variance

    def get_target(self, x_0, epsilon, rates):
        signal_rate, noise_rate = rates
        v = signal_rate * epsilon - noise_rate * x_0
        variance = signal_rate ** 2 + noise_rate ** 2
        return v / jnp.sqrt(variance)

    def target_error_scale(self, snr):
        return snr + 1


class FlowMatchPredictionTransform(PredictionTransform):
    """The model predicts the rectified flow velocity u = epsilon - x_0.

    This is the constant velocity of the linear path, so x_0 and epsilon are
    each one step from x_t along it.
    """

    def backward_diffusion(self, x_t, prediction, rates):
        signal_rate, noise_rate = rates
        return x_t - noise_rate * prediction, x_t + signal_rate * prediction

    def get_target(self, x_0, epsilon, rates):
        return epsilon - x_0

    def target_error_scale(self, snr):
        # x_0 error is t times the velocity error, and t = 1 / (1 + sqrt(SNR))
        return (1 + jnp.sqrt(snr)) ** 2


class KarrasPredictionTransform(PredictionTransform):
    """The EDM preconditioning of Karras et al. 2022, Table 1.

    The model's input is c_in x_t, and its raw output F gives
    x_0 = c_skip x_t + c_out F. The loss compares that x_0 with the clean
    sample, so the target is x_0, and a weight defined on the x_0 loss, such
    as min-SNR's, applies without conversion. EDM's lambda = 1 / c_out^2 is
    the schedule's own weight on this loss.

    `velocity` matches Diffusers 0.34.0's EDM
    `prediction_type="v_prediction"`. Its `precondition_outputs` negates
    c_out, so the model outputs the velocity of the preconditioned path
    instead of the offset to its endpoint.
    """

    def __init__(self, sigma_data: float = 0.5, *, velocity: bool = False) -> None:
        self.sigma_data = sigma_data
        self.velocity = velocity

    def backward_diffusion(self, x_t, prediction, rates):
        return from_clean(x_t, prediction, rates)

    def pred_transform(self, x_t, prediction, rates, t):
        _, sigma = rates
        # Every denominator is at least sigma_data, so none needs a guard.
        c_out = sigma * self.sigma_data / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_skip = self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)
        return (-c_out if self.velocity else c_out) * prediction + c_skip * x_t

    def get_input_scale(self, rates):
        _, sigma = rates
        return 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)


class ConsistencyBoundary(PredictionTransform):
    """The boundary parameterization of a latent consistency model, as in Diffusers' `LCMScheduler`.

    The method is from Luo et al. 2023 (arXiv 2310.04378). `inner` converts
    the model's output to x_0, and the consistency function is
    f = c_skip x_t + c_out x_0, with
    c_skip = sigma_data^2 / (s^2 + sigma_data^2) and
    c_out = s / sqrt(s^2 + sigma_data^2) at the scaled time
    s = `timestep_scaling` t. So f is x_t itself at t = 0. The returned `x_0`
    is f, and `epsilon` is recovered from it.
    """

    def __init__(self, inner: PredictionTransform, timestep_scaling: float = 10.0,
                 sigma_data: float = 0.5) -> None:
        self.inner = inner
        self.timestep_scaling = timestep_scaling
        self.sigma_data = sigma_data

    def pred_transform(self, x_t, prediction, rates, t):
        prediction = self.inner.pred_transform(x_t, prediction, rates, t)
        x_0, _ = self.inner.backward_diffusion(x_t, prediction, rates)
        scaled = expand(jnp.asarray(t, jnp.float32) * self.timestep_scaling, x_t)
        c_skip = self.sigma_data ** 2 / (scaled ** 2 + self.sigma_data ** 2)
        c_out = scaled / (scaled ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out * x_0 + c_skip * x_t

    def backward_diffusion(self, x_t, prediction, rates):
        return from_clean(x_t, prediction, rates)

    def get_input_scale(self, rates):
        return self.inner.get_input_scale(rates)


class SourceLimitedPrediction(PredictionTransform):
    """Limits x_0 the way a published scheduler's `step` does.

    `inner` recovers x_0 from the model's output, and then either dynamic
    thresholding or a plain clamp to `clip` limits it. Thresholding, given
    as `threshold=(ratio, maximum)`, follows Saharia et al. 2022. Each
    sample's level is its own `ratio` quantile of |x_0|, kept between 1 and
    `maximum`, and x_0 is clamped to that level and then divided by it. When
    a source declares both limits, thresholding is used, because that is the
    order in which the source's `step` tests them. Giving neither raises
    `ValueError`.

    `recompute_epsilon` is whether the source recomputes epsilon from the
    limited x_0. DDPM's posterior, DEIS and the noise-prediction DPM-Solver
    algorithms do, so the limit reaches their update through epsilon as
    well. DDIM keeps the model's own output as its epsilon, so only its x_0
    term is limited.

    The limit is not linear in the model's output. So a guided walk applies
    it once, when it converts the combined output, and not to each guidance
    branch.
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
        assert self.clip is not None  # `__init__` refuses both limits unset
        return jnp.clip(x_0, -self.clip, self.clip)

    def pred_transform(self, x_t, prediction, rates, t):
        return self.inner.pred_transform(x_t, prediction, rates, t)

    def backward_diffusion(self, x_t, prediction, rates):
        x_0, epsilon = self.inner.backward_diffusion(x_t, prediction, rates)
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
    """Weights a per-example loss, given the schedule and the prediction transform."""

    def __call__(self, schedule: NoiseScheduler, prediction: PredictionTransform,
                 t) -> jax.Array:
        """Return the per-example loss weight at `t`, shaped like `t`."""
        ...


@dataclass(frozen=True)
class ScheduleWeighting:
    """Weights the loss with the schedule's own weight."""

    def __call__(self, schedule, prediction, t):
        return schedule.weight(t)


@dataclass(frozen=True)
class MinSNR:
    """Weights the loss with min-SNR-gamma (Hang et al. 2023).

    The weight is min(SNR, gamma) on the x_0 loss, converted into the space
    the model trains in, and it replaces the schedule's own weight. At zero
    SNR (a table whose last step keeps no signal) the epsilon conversion is
    0 / 0, so the weight there is one, as the authors' code sets it.
    """

    gamma: float

    def __call__(self, schedule, prediction, t):
        snr = schedule.snr(t)
        scale = prediction.target_error_scale(snr)
        weight = jnp.minimum(snr, self.gamma) / jnp.where(snr == 0, 1.0, scale)
        return jnp.where(snr == 0, 1.0, weight)


@dataclass(frozen=True)
class VelocityLoss:
    """Scores a clean-sample prediction in velocity space on the linear path, as JiT's loss does.

    JiT is Li & He 2025, "Back to Basics: Let Denoising Generative Models
    Denoise". With x_t = (1 - sigma) x_0 + sigma eps, the velocity is
    (x_0 - x_t) / sigma, so its squared error is the squared x_0 error over
    sigma^2. sigma is floored at `t_eps` in both the target and the
    prediction, as the reference clamps it. The weight replaces the
    schedule's own weight. Calling it on a process that does not predict
    with `DirectPredictionTransform` raises `ValueError`.
    """

    t_eps: float = 0.05

    def __call__(self, schedule, prediction, t):
        if not isinstance(prediction, DirectPredictionTransform):
            raise ValueError("the velocity loss scores a clean-sample prediction; "
                             f"the process predicts with {type(prediction).__name__}")
        _, sigma = schedule.rates(t)
        return 1.0 / jnp.square(jnp.maximum(sigma, self.t_eps))


def broadcast_rates(schedule: NoiseScheduler, t, x) -> tuple[jax.Array, jax.Array]:
    """Return the schedule's rates `(alpha, sigma)` at `t`, shaped to broadcast against `x`."""
    alpha, sigma = schedule.rates(t)
    return expand(alpha, x), expand(sigma, x)
