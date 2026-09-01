from .common import DiffusionSampler
from .ddim import DDIMSampler
from .ddpm import DDPMSampler, SimpleDDPMSampler
from .euler import EulerSampler, SimplifiedEulerSampler, EulerAncestralSampler
from .heun_sampler import HeunSampler
from .rk4_sampler import RK4Sampler
from .multistep_dpm import MultiStepDPM
# Pipeline/loading names resolve lazily: pipelines imports the trainer, which
# imports the diffusion objective, which imports the samplers above - an eager
# import here would close that loop.
_LAZY = {
    "InferencePipeline": ".pipelines",
    "DiffusionInferencePipeline": ".pipelines",
    "parse_config": ".loading",
    "load_from_checkpoint": ".loading",
    "load_from_wandb_run": ".loading",
    "load_from_wandb_registry": ".loading",
    "get_wandb_run": ".loading",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name], __name__), name)
    if name in ("pipelines", "loading"):
        from importlib import import_module
        return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
