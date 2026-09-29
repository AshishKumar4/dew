"""Named conventions, as the dataclasses a run's `run.json` stores.

A preset is a frozen dataclass of the numbers that define a convention, and
calling it builds the `Process`. Both training and inference build from the
same preset, so a model is always sampled with the convention it was trained
with. A record that holds the preset's fields rebuilds it exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from dew.diffusion.process import Process
from dew.diffusion.schedules import (
    CosineNoiseScheduler,
    EDMNoiseScheduler,
    FlowMatchingScheduler,
    KarrasVENoiseScheduler,
    SqrtContinuousNoiseScheduler,
)
from dew.diffusion.schedules.flow import Density
from dew.diffusion.transforms import (
    DirectPredictionTransform,
    FlowMatchPredictionTransform,
    KarrasPredictionTransform,
    MinSNR,
    ScheduleWeighting,
    VelocityLoss,
    VPredictionTransform,
    Weighting,
)
from dew.registry import presets

if TYPE_CHECKING:
    from dew.diffusion.discrete import DiscreteProcess


@runtime_checkable
class Preset(Protocol):
    """Every member of the `presets` registry is a frozen dataclass of a
    convention's numbers, callable to the `Process` it describes."""

    def __call__(self) -> Process | DiscreteProcess: ...


def _weighting(min_snr_gamma: float | None) -> Weighting:
    return ScheduleWeighting() if min_snr_gamma is None else MinSNR(min_snr_gamma)


@presets("edm")
@dataclass(frozen=True)
class EDM:
    """Log-normal training sigmas, the EDM preconditioning and lambda
    weighting, sampled on the rho-spaced Karras grid.

    The sigma distribution depends on the space the model denoises:
    `regime="pixel"` draws Karras et al. 2022's exp(N(-1.2, 1.2^2)) for
    pixels in [-1, 1], and `regime="latent"` EDM2's exp(N(-0.4, 1.0^2))
    (Karras et al. 2024) for an autoencoder's latents. `P_mean` and `P_std`
    set explicitly override the regime's, and a record that stores them
    rebuilds without one. A preset with neither builds no process: a single
    default silently trained pixel models on latent sigmas (Oxford Flowers
    at 64px, 6k steps, flip-only both sides: FID 160 on EDM2's values against 133 on 2022's).
    `DiffusionRunConfig` fills the regime from whether it has an autoencoder.
    """

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    regime: Literal["pixel", "latent"] | None = None
    P_mean: float | None = None
    P_std: float | None = None
    min_snr_gamma: float | None = None

    def __post_init__(self) -> None:
        if self.regime not in (None, "pixel", "latent"):
            raise ValueError(f"EDM's regime is 'pixel' or 'latent', got {self.regime!r}")

    def lognormal(self) -> tuple[float, float]:
        """The training sigmas' log mean and log standard deviation."""
        if self.P_mean is not None and self.P_std is not None:
            return self.P_mean, self.P_std
        if self.regime is None:
            raise ValueError(
                "EDM draws its training sigmas for one space: set regime='pixel' (Karras 2022, "
                "P_mean -1.2, P_std 1.2) or regime='latent' (EDM2, P_mean -0.4, P_std 1.0), or "
                "P_mean and P_std themselves")
        mean, std = _LOGNORMAL[self.regime]
        return (mean if self.P_mean is None else self.P_mean, std if self.P_std is None else self.P_std)

    def __call__(self) -> Process:
        P_mean, P_std = self.lognormal()
        return Process(
            schedule=EDMNoiseScheduler(
                sigma_min=self.sigma_min, sigma_max=self.sigma_max, sigma_data=self.sigma_data,
                P_mean=P_mean, P_std=P_std),
            prediction=KarrasPredictionTransform(sigma_data=self.sigma_data),
            weighting=_weighting(self.min_snr_gamma),
            sampling=KarrasVENoiseScheduler(
                sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho,
                sigma_data=self.sigma_data))


_LOGNORMAL = {"pixel": (-1.2, 1.2), "latent": (-0.4, 1.0)}
"""Each regime's (P_mean, P_std): Karras et al. 2022 for pixels, EDM2 for latents."""


@presets("karras")
@dataclass(frozen=True)
class Karras:
    """The EDM preconditioning trained on sigmas drawn uniformly along the
    rho-spaced grid it samples on."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    min_snr_gamma: float | None = None

    def __call__(self) -> Process:
        return Process(
            schedule=KarrasVENoiseScheduler(
                sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho,
                sigma_data=self.sigma_data),
            prediction=KarrasPredictionTransform(sigma_data=self.sigma_data),
            weighting=_weighting(self.min_snr_gamma))


@presets("cosine")
@dataclass(frozen=True)
class Cosine:
    """The cosine beta table with v-prediction.

    The table's P2 weight at its defaults (k = 1, gamma = 1) is 1 / (1 + SNR),
    which makes the v loss an unweighted x_0 loss. `p2_loss_weight_gamma`
    changes that.
    """

    timesteps: int = 1000
    beta_end: float = 1.0
    p2_loss_weight_k: float = 1.0
    p2_loss_weight_gamma: float = 1.0
    min_snr_gamma: float | None = None

    def __call__(self) -> Process:
        return Process(
            schedule=CosineNoiseScheduler(
                self.timesteps, beta_end=self.beta_end,
                p2_loss_weight_k=self.p2_loss_weight_k,
                p2_loss_weight_gamma=self.p2_loss_weight_gamma),
            prediction=VPredictionTransform(),
            weighting=_weighting(self.min_snr_gamma))


@dataclass(frozen=True)
class ResolutionShift:
    """Flux's shift by resolution (the pipelines' `calculate_shift`): mu is
    linear in a token count, `base_shift` at `base_tokens` and `max_shift`
    at `max_tokens`, and the shift is exp(mu).

    `at` counts the image's 16 x 16-pixel cells, the grid Flux's constants
    are stated on (an 8x autoencoder under 2x2 patches). That is a reference
    grid, not every model's own token count: a model that tokenizes the
    image otherwise, a 32x autoencoder under 1x1 patches for one, takes
    `tokens` set to its own count. `DiffusionRunConfig` fills an unset
    `tokens` from the data's resolution on that grid.
    """

    base_shift: float = 0.5
    max_shift: float = 1.15
    base_tokens: int = 256
    max_tokens: int = 4096
    tokens: int | None = None

    def at(self, height: int, width: int) -> ResolutionShift:
        """This shift at an image of `height` x `width` pixels, counted in
        16 x 16-pixel cells."""
        return replace(self, tokens=(height // 16) * (width // 16))

    def shift(self) -> float:
        if self.tokens is None:
            raise ValueError("a resolution shift needs the image's token count; set tokens "
                             "or build through DiffusionRunConfig, which fills it")
        slope = (self.max_shift - self.base_shift) / (self.max_tokens - self.base_tokens)
        return math.exp(self.base_shift + slope * (self.tokens - self.base_tokens))


@presets("flow")
@dataclass(frozen=True)
class Flow:
    """Rectified flow on the linear path with velocity prediction.

    `density` is SD3's training time density (`FlowMatchingScheduler`):
    logit-normal at `logit_mean` and `logit_std`, the heavy-tailed mode
    density at `mode_scale`, cosmap or uniform. `shift` is SD3's static
    resolution shift; `resolution_shift` sets it from the image size instead,
    for training and sampling alike.
    """

    shift: float = 1.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    density: Density = "logit_normal"
    mode_scale: float = 1.29
    resolution_shift: ResolutionShift | None = None
    min_snr_gamma: float | None = None

    def __post_init__(self) -> None:
        if self.resolution_shift is not None and self.shift != 1.0:
            raise ValueError("shift is static and resolution_shift sets it from the image "
                             "size; name one")

    def __call__(self) -> Process:
        shift = self.shift if self.resolution_shift is None else self.resolution_shift.shift()
        return Process(
            schedule=FlowMatchingScheduler(
                shift=shift, logit_mean=self.logit_mean, logit_std=self.logit_std,
                density=self.density, mode_scale=self.mode_scale),
            prediction=FlowMatchPredictionTransform(),
            weighting=_weighting(self.min_snr_gamma))


@presets("jit")
@dataclass(frozen=True)
class JiT:
    """JiT (Li & He 2025, "Back to Basics: Let Denoising Generative Models
    Denoise"): rectified flow on the linear path in which the model
    predicts the clean sample and is scored in velocity space
    (`VelocityLoss`). Its training times are LTH14/JiT's logit-normal at
    P_mean -0.8 and P_std 0.8 in its clean-at-one time, which is Dew's
    noise-at-one time at `logit_mean` 0.8. The reference's `noise_scale` is
    1, its value at 256 pixels.
    """

    logit_mean: float = 0.8
    logit_std: float = 0.8
    t_eps: float = 0.05

    def __call__(self) -> Process:
        return Process(
            schedule=FlowMatchingScheduler(logit_mean=self.logit_mean, logit_std=self.logit_std),
            prediction=DirectPredictionTransform(),
            weighting=VelocityLoss(self.t_eps))


@presets("sqrt")
@dataclass(frozen=True)
class Sqrt:
    """Diffusion-LM (Li et al. 2022): the square-root schedule with the plain
    x_0 loss."""

    min_snr_gamma: float | None = None

    def __call__(self) -> Process:
        return Process(
            schedule=SqrtContinuousNoiseScheduler(),
            prediction=DirectPredictionTransform(),
            weighting=_weighting(self.min_snr_gamma))
