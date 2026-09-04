"""Dew: one registry, one objective, one trainer.

Naming anything the package exports fills the registries, so `dew.models`,
`dew.presets.EDM` and `dew.datasets["oxford_flowers102"]` resolve after
`import dew` with nothing else imported. The fill happens on that first name
rather than at import, so `import dew.training` stays inside the training
layer and pulls in no modality, no encoder and no tracker backend, which is
the layering rule tests/test_api_surface.py checks. Nothing here opens a JAX
backend or loads an optional dependency; encoders, decoders and datasets
fetch what they need when they are built.

`objectives` is not exported: `dew.objectives` is the package holding the
Objective classes, and a registry cannot share its name. It is
`dew.registry.objectives`.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the surface above, with its types, for checkers and editors
    from dew.artifacts import (
        ImageGrid, Representations, TextSamples, TokenScores, VideoGrid,
    )
    from dew.data import Dataset
    from dew.diffusion import Process
    from dew.inputs import Condition, Field, InputSpec
    from dew.objectives import Objective
    from dew.objectives.base import Aux, EMASpec, Step
    from dew.registry import datasets, encoders, metrics, models, presets, samplers
    from dew.sampling import CFG, sample
    from dew.training import (
        Checkpoints, Layout, MeshSpec, Tracker, Trainer, TrainState, WandbTracker,
    )

__version__ = "0.1.0"

# Importing any of these registers its members with the registries.
_REGISTERS = (
    "dew.nn.backbones", "dew.diffusion.presets", "dew.sampling.solvers",
    "dew.data", "dew.inputs", "dew.eval",
    "dew.objectives.lm", "dew.objectives.jepa", "dew.objectives.diffusion",
)

_REGISTRIES = ("models", "presets", "samplers", "datasets", "encoders", "metrics")

_EXPORTS = {
    **{name: "dew.registry" for name in _REGISTRIES},
    "Trainer": "dew.training", "TrainState": "dew.training", "Step": "dew.training",
    "Aux": "dew.training", "EMASpec": "dew.training", "MeshSpec": "dew.training",
    "Layout": "dew.training", "Checkpoints": "dew.training", "Tracker": "dew.training",
    "WandbTracker": "dew.training",
    "Objective": "dew.objectives",
    "Dataset": "dew.data",
    "Process": "dew.diffusion",
    "InputSpec": "dew.inputs", "Field": "dew.inputs", "Condition": "dew.inputs",
    "sample": "dew.sampling", "CFG": "dew.sampling",
    "ImageGrid": "dew.artifacts", "VideoGrid": "dew.artifacts",
    "TextSamples": "dew.artifacts", "Representations": "dew.artifacts",
    "TokenScores": "dew.artifacts",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _REGISTRIES:
        for register in _REGISTERS:
            import_module(register)
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return list(__all__)


# Written out rather than derived from _EXPORTS so a type checker, an editor
# and `from dew import *` can all read the public surface without running the
# lazy lookup above. tests/test_api_surface.py holds the two in agreement.
__all__ = [
    "__version__",
    "Aux", "CFG", "Checkpoints", "Condition", "Dataset", "EMASpec", "Field",
    "ImageGrid", "InputSpec", "Layout", "MeshSpec", "Objective", "Process",
    "Representations", "Step", "TextSamples", "TokenScores", "Tracker",
    "TrainState", "Trainer", "VideoGrid", "WandbTracker",
    "datasets", "encoders", "metrics", "models", "presets", "sample", "samplers",
]
