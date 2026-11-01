"""Forward processes, noise schedules and the parameterizations over them.

`Process` pairs a schedule with a prediction transform, and a run's
objective and every solver read that process. `presets` holds the named
combinations, and `discrete` has the masked-token process.

Each name below is imported from its own module when first read, so a discrete text process reads none of
the image schedules and presets.
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.diffusion import discrete, presets
    from dew.diffusion.process import Denoiser, Process
    from dew.diffusion.schedules import (
        ContinuousNoiseScheduler,
        CosineNoiseScheduler,
        DiscreteNoiseScheduler,
        EDMNoiseScheduler,
        FlowMatchingScheduler,
        GeneralizedNoiseScheduler,
        KarrasVENoiseScheduler,
        LinearNoiseScheduler,
        NoiseScheduler,
        SqrtContinuousNoiseScheduler,
        cosine_beta_schedule,
        expand,
        linear_beta_schedule,
    )
    from dew.diffusion.transforms import (
        ConsistencyBoundary,
        DirectPredictionTransform,
        EpsilonPredictionTransform,
        FlowMatchPredictionTransform,
        KarrasPredictionTransform,
        MinSNR,
        PredictionTransform,
        ScheduleWeighting,
        SourceLimitedPrediction,
        VelocityLoss,
        VPredictionTransform,
        Weighting,
        broadcast_rates,
    )

_EXPORTS = {
    "discrete": "dew.diffusion.discrete",
    "presets": "dew.diffusion.presets",
    "Denoiser": "dew.diffusion.process",
    "Process": "dew.diffusion.process",
    "ContinuousNoiseScheduler": "dew.diffusion.schedules",
    "CosineNoiseScheduler": "dew.diffusion.schedules",
    "DiscreteNoiseScheduler": "dew.diffusion.schedules",
    "EDMNoiseScheduler": "dew.diffusion.schedules",
    "FlowMatchingScheduler": "dew.diffusion.schedules",
    "GeneralizedNoiseScheduler": "dew.diffusion.schedules",
    "KarrasVENoiseScheduler": "dew.diffusion.schedules",
    "LinearNoiseScheduler": "dew.diffusion.schedules",
    "NoiseScheduler": "dew.diffusion.schedules",
    "SqrtContinuousNoiseScheduler": "dew.diffusion.schedules",
    "cosine_beta_schedule": "dew.diffusion.schedules",
    "expand": "dew.diffusion.schedules",
    "linear_beta_schedule": "dew.diffusion.schedules",
    "ConsistencyBoundary": "dew.diffusion.transforms",
    "DirectPredictionTransform": "dew.diffusion.transforms",
    "EpsilonPredictionTransform": "dew.diffusion.transforms",
    "FlowMatchPredictionTransform": "dew.diffusion.transforms",
    "KarrasPredictionTransform": "dew.diffusion.transforms",
    "MinSNR": "dew.diffusion.transforms",
    "PredictionTransform": "dew.diffusion.transforms",
    "ScheduleWeighting": "dew.diffusion.transforms",
    "SourceLimitedPrediction": "dew.diffusion.transforms",
    "VelocityLoss": "dew.diffusion.transforms",
    "VPredictionTransform": "dew.diffusion.transforms",
    "Weighting": "dew.diffusion.transforms",
    "broadcast_rates": "dew.diffusion.transforms",
}

__getattr__, __dir__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "ConsistencyBoundary",
    "ContinuousNoiseScheduler",
    "CosineNoiseScheduler",
    "Denoiser",
    "DirectPredictionTransform",
    "DiscreteNoiseScheduler",
    "EDMNoiseScheduler",
    "EpsilonPredictionTransform",
    "FlowMatchPredictionTransform",
    "FlowMatchingScheduler",
    "GeneralizedNoiseScheduler",
    "KarrasPredictionTransform",
    "KarrasVENoiseScheduler",
    "LinearNoiseScheduler",
    "MinSNR",
    "NoiseScheduler",
    "PredictionTransform",
    "Process",
    "ScheduleWeighting",
    "SourceLimitedPrediction",
    "SqrtContinuousNoiseScheduler",
    "VPredictionTransform",
    "VelocityLoss",
    "Weighting",
    "broadcast_rates",
    "cosine_beta_schedule",
    "discrete",
    "expand",
    "linear_beta_schedule",
    "presets",
]
