"""Dew: one objective, one trainer.

Each name below is imported from its own module when it is first read, not at
`import dew`, so `import dew.training` stays inside the training layer and
pulls in no modality, no encoder and no tracker backend. Nothing here opens a
JAX backend or loads an optional dependency; encoders, decoders and datasets
fetch what they need when they are built.
"""

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

from dew.logging import configure as _configure_logging

_configure_logging()

if TYPE_CHECKING:  # the surface above, with its types, for checkers and editors
    from dew.artifacts import ImageGrid, Representations, TextSamples, TokenScores, VideoGrid
    from dew.data import Dataset
    from dew.diffusion import Process
    from dew.eval import Mean
    from dew.inference import pipeline
    from dew.inputs import Condition, Field, InputSpec
    from dew.objectives import Objective
    from dew.objectives.base import Aux, EMASpec, Step
    from dew.sampling import CFG, sample
    from dew.telemetry.profile import Profiler
    from dew.training import (
        Best,
        Checkpoints,
        Evaluation,
        Keep,
        Layout,
        LocalTracker,
        MeshSpec,
        MLflowTracker,
        Plateau,
        ProfileWindow,
        TensorBoardTracker,
        Tracker,
        Trackers,
        Trainer,
        TrainState,
        WandbTracker,
    )

__version__ = "0.1.0"

_EXPORTS = {
    "Best": "dew.training", "Keep": "dew.training", "Plateau": "dew.training",
    "Trainer": "dew.training", "TrainState": "dew.training", "Step": "dew.training",
    "Aux": "dew.training", "EMASpec": "dew.training", "MeshSpec": "dew.training",
    "Layout": "dew.training", "Checkpoints": "dew.training", "Tracker": "dew.training",
    "WandbTracker": "dew.training", "LocalTracker": "dew.training", "Trackers": "dew.training",
    "MLflowTracker": "dew.training", "TensorBoardTracker": "dew.training",
    "Evaluation": "dew.training",
    "ProfileWindow": "dew.training",
    "Objective": "dew.objectives",
    "Dataset": "dew.data",
    "Process": "dew.diffusion",
    "Mean": "dew.eval",
    "InputSpec": "dew.inputs", "Field": "dew.inputs", "Condition": "dew.inputs",
    "sample": "dew.sampling", "CFG": "dew.sampling",
    "pipeline": "dew.inference",
    "Profiler": "dew.telemetry.profile",
    "ImageGrid": "dew.artifacts", "VideoGrid": "dew.artifacts",
    "TextSamples": "dew.artifacts", "Representations": "dew.artifacts",
    "TokenScores": "dew.artifacts",
}


def __getattr__(name: str) -> type | Callable:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return list(__all__)


# Written out, not derived from _EXPORTS, so a type checker, an editor and
# `from dew import *` can all read the public surface without running the
# lazy lookup above.
__all__ = [
    "CFG",
    "Aux",
    "Best",
    "Checkpoints",
    "Condition",
    "Dataset",
    "EMASpec",
    "Evaluation",
    "Field",
    "ImageGrid",
    "InputSpec",
    "Keep",
    "Layout",
    "LocalTracker",
    "MLflowTracker",
    "Mean",
    "MeshSpec",
    "Objective",
    "Plateau",
    "Process",
    "ProfileWindow",
    "Profiler",
    "Representations",
    "Step",
    "TensorBoardTracker",
    "TextSamples",
    "TokenScores",
    "Tracker",
    "Trackers",
    "TrainState",
    "Trainer",
    "VideoGrid",
    "WandbTracker",
    "__version__",
    "pipeline",
    "sample",
]
