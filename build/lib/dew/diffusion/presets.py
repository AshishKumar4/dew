"""Named conventions, as the dataclasses a run's `run.json` stores.

A preset is a frozen dataclass of the numbers that define a convention;
calling it builds the `Process`. Both training and inference build from the
same preset, so a model is always sampled with the convention it was trained
with, and a record that holds the preset's fields rebuilds it exactly.
"""

from dataclasses import dataclass

from dew.diffusion.process import Process
from dew.diffusion.schedules import (
    CosineNoiseScheduler, EDMNoiseScheduler, FlowMatchingScheduler, KarrasVENoiseScheduler,
    SqrtContinuousNoiseScheduler,
)
from dew.diffusion.transforms import (
    DirectPredictionTransform, FlowMatchPredictionTransform, KarrasPredictionTransform,
    MinSNR, ScheduleWeighting, VPredictionTransform, Weighting,
)
from dew.registry import presets


def _weighting(min_snr_gamma: float | None) -> Weighting:
    return ScheduleWeighting() if min_snr_gamma is None else MinSNR(min_snr_gamma)


@presets("edm")
@dataclass(frozen=True)
class EDM:
    """Karras et al. 2022: log-normal training sigmas, the EDM preconditioning
    and lambda weighting, sampled on the rho-spaced Karras grid."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    P_mean: float = -0.4
    P_std: float = 1.0
    min_snr_gamma: float | None = None

    def __call__(self) -> Process:
        return Process(
            schedule=EDMNoiseScheduler(
                sigma_min=self.sigma_min, sigma_max=self.sigma_max, sigma_data=self.sigma_data,
                P_mean=self.P_mean, P_std=self.P_std),
            prediction=KarrasPredictionTransform(sigma_data=self.sigma_data),
            weighting=_weighting(self.min_snr_gamma),
            sampling=KarrasVENoiseScheduler(
                sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho,
                sigma_data=self.sigma_data))


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
    """The cosine beta table with v-prediction. The table's P2 weight at its
    defaults (k = 1, gamma = 1) is 1 / (1 + SNR), which makes the v loss an
    unweighted x_0 loss; `p2_loss_weight_gamma` changes that."""

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


@presets("flow")
@dataclass(frozen=True)
class Flow:
    """Rectified flow on the linear path, velocity prediction, logit-normal
    times, with SD3's resolution shift."""

    shift: float = 1.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    min_snr_gamma: float | None = None

    def __call__(self) -> Process:
        return Process(
            schedule=FlowMatchingScheduler(
                shift=self.shift, logit_mean=self.logit_mean, logit_std=self.logit_std),
            prediction=FlowMatchPredictionTransform(),
            weighting=_weighting(self.min_snr_gamma))


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
