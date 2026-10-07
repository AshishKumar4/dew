"""Named diffusion conventions, stored as dataclasses in a run's `run.json`.

A convention is a choice of noise schedule, prediction target, loss
weighting and sampling grid. A preset is a frozen dataclass of the numbers
that define one convention. An objective builds the preset's `Process` when
it is constructed. You can also call a preset to build its process yourself,
to inspect the schedule or to sample at a low level. A record that holds a
preset's fields rebuilds the preset exactly.
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
from dew.diffusion.schedules.flow import Density, _token_mu
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

if TYPE_CHECKING:
    from dew.diffusion.discrete import DiscreteProcess


@runtime_checkable
class Preset(Protocol):
    """The interface every member of the `presets` registry follows.

    Each member is a frozen dataclass of a convention's numbers, and calling
    it returns the process it describes.
    """

    def __call__(self) -> Process | DiscreteProcess: ...


def build_process(convention: Process | Preset) -> Process:
    """Resolve a Gaussian convention for a diffusion objective."""
    process = convention if isinstance(convention, Process) else convention()
    if not isinstance(process, Process):
        kind = type(convention)
        name = kind.__name__
        raise ValueError(
            f"preset {name!r} builds a {type(process).__name__}; "
            "image diffusion needs a Gaussian Process. Masked diffusion trains "
            "through LMRunConfig's --objective masked_diffusion")
    return process


def _weighting(min_snr_gamma: float | None) -> Weighting:
    return ScheduleWeighting() if min_snr_gamma is None else MinSNR(min_snr_gamma)


@dataclass(frozen=True)
class EDM:
    """Trains on log-normal sigmas with the EDM preconditioning and lambda weighting.

    It samples on the rho-spaced Karras grid. The sigma distribution depends
    on the space the model denoises. `regime="pixel"` draws Karras et al.
    2022's exp(N(-1.2, 1.2^2)) for pixels in [-1, 1], and `regime="latent"`
    draws EDM2's exp(N(-0.4, 1.0^2)) (Karras et al. 2024) for an
    autoencoder's latents. `P_mean` and `P_std`, when set, override the
    regime's values, and a record that stores both rebuilds without a
    regime. A preset with neither a regime nor both values builds no
    process. `DiffusionRunConfig` sets the regime from whether the run has
    an autoencoder.
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
        """Return the training sigmas' log mean and log standard deviation.

        Raises `ValueError` when neither `regime` nor both `P_mean` and
        `P_std` are set.
        """
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


@dataclass(frozen=True)
class Karras:
    """Trains the EDM preconditioning on sigmas drawn uniformly along the rho-spaced grid it samples on."""

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


@dataclass(frozen=True)
class Cosine:
    """The cosine beta table with v-prediction.

    The table is improved-diffusion's (`CosineNoiseScheduler`), with its
    betas clipped at 0.999 so the last step keeps some signal. At its
    defaults (k = 1, gamma = 1), the P2 weight is 1 / (1 + SNR), which makes
    the v loss an unweighted x_0 loss. Another `p2_loss_weight_gamma`
    changes that.
    """

    timesteps: int = 1000
    beta_end: float = 0.999
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
    """Flux's time shift by resolution, as the Flux pipelines' `calculate_shift` computes it.

    mu is linear in a token count, equal to `base_shift` at `base_tokens`
    and `max_shift` at `max_tokens`, and the shift is exp(mu). `shift()`
    returns it and raises `ValueError` while `tokens` is unset.

    `at` counts the image's 16 x 16-pixel cells, which is the grid Flux's
    constants are stated on (an 8x autoencoder under 2x2 patches). That grid
    is a reference and does not match every model's own token count. A model
    that tokenizes the image differently, such as a 32x autoencoder under
    1x1 patches, should set `tokens` to its own count. `DiffusionRunConfig`
    sets an unset `tokens` from the data's resolution on the 16 x 16 grid.
    """

    base_shift: float = 0.5
    max_shift: float = 1.15
    base_tokens: int = 256
    max_tokens: int = 4096
    tokens: int | None = None

    def at(self, height: int, width: int) -> ResolutionShift:
        """Return this shift for an image of `height` x `width` pixels, counted in 16 x 16-pixel cells."""
        return replace(self, tokens=(height // 16) * (width // 16))

    def shift(self) -> float:
        if self.tokens is None:
            raise ValueError("a resolution shift needs the image's token count; set tokens "
                             "or build through DiffusionRunConfig, which fills it")
        return math.exp(_token_mu(self.tokens, self.base_tokens, self.max_tokens, self.base_shift,
                                  self.max_shift))


@dataclass(frozen=True)
class Flow:
    """Rectified flow on the linear path with velocity prediction.

    `density` is SD3's density of training times (see
    `FlowMatchingScheduler`): logit-normal at `logit_mean` and `logit_std`,
    the heavy-tailed mode density at `mode_scale`, cosmap, or uniform.
    `shift` is SD3's static resolution shift. `resolution_shift` sets the
    shift from the image size instead, for both training and sampling, so
    giving it together with a `shift` other than 1.0 raises `ValueError`.
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


@dataclass(frozen=True)
class MeanFlow:
    """Rectified flow on the linear path, with a model that predicts the average velocity over an interval.

    This is MeanFlow's convention (Geng et al. 2025, "Mean Flows for One-step
    Generative Modeling"), and its process sets `Process.interval`. The
    training times are Gsunshine/meanflow's logit-normal at P_mean -0.4 and
    P_std 1.0, which measures time the same way Dew does, with noise at 1.
    It trains under `MeanFlowObjective`, and one Euler step over the whole
    grid samples it.
    """

    logit_mean: float = -0.4
    logit_std: float = 1.0

    def __call__(self) -> Process:
        return Process(
            schedule=FlowMatchingScheduler(logit_mean=self.logit_mean, logit_std=self.logit_std),
            prediction=FlowMatchPredictionTransform(), interval=True)


@dataclass(frozen=True)
class Shortcut:
    """Rectified flow on the linear path, with a model that predicts the velocity of one step of a given size.

    This is a shortcut model's convention (Frans et al. 2025, "One Step
    Diffusion via Shortcut Models"), and its process sets
    `Process.interval`. It trains under `ShortcutObjective`, which draws its
    own times on dyadic grids.
    """

    def __call__(self) -> Process:
        return Process(schedule=FlowMatchingScheduler(density="uniform"),
                       prediction=FlowMatchPredictionTransform(), interval=True)


@dataclass(frozen=True)
class JiT:
    """Rectified flow where the model predicts the clean sample and is scored in velocity space (JiT).

    JiT is Li & He 2025, "Back to Basics: Let Denoising Generative Models
    Denoise", and the loss is `VelocityLoss` with `t_eps`. The training
    times are LTH14/JiT's logit-normal at P_mean -0.8 and P_std 0.8. The
    reference puts the clean sample at time 1, so in Dew's time, with noise
    at 1, that is `logit_mean` 0.8. The reference's `noise_scale` is 1, its
    value at 256 pixels.
    """

    logit_mean: float = 0.8
    logit_std: float = 0.8
    t_eps: float = 0.05

    def __call__(self) -> Process:
        return Process(
            schedule=FlowMatchingScheduler(logit_mean=self.logit_mean, logit_std=self.logit_std),
            prediction=DirectPredictionTransform(),
            weighting=VelocityLoss(self.t_eps))


@dataclass(frozen=True)
class Sqrt:
    """The square-root schedule with the plain x_0 loss, from Diffusion-LM.

    Diffusion-LM is Li et al. 2022.
    """

    min_snr_gamma: float | None = None

    def __call__(self) -> Process:
        return Process(
            schedule=SqrtContinuousNoiseScheduler(),
            prediction=DirectPredictionTransform(),
            weighting=_weighting(self.min_snr_gamma))


__all__ = [
    "EDM",
    "Cosine",
    "Flow",
    "JiT",
    "Karras",
    "MeanFlow",
    "Preset",
    "ResolutionShift",
    "Shortcut",
    "Sqrt",
]
