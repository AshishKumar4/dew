"""Dew: one objective, one trainer.

Each name exported here is imported from its own module when you first
access it, not at `import dew`. So `import dew.training` loads only the
training layer, with no modality, encoder or tracker backend. Importing
`dew` opens no JAX backend and loads no optional dependency; encoders,
decoders and datasets load what they need when they are built.

`import dew` does set two XLA flags before the backend opens.
`--xla_allow_excess_precision=false` makes XLA round values declared in a
narrow dtype such as bf16 where the program rounds them. When JAX's CUDA
plugin is installed, `--xla_gpu_enable_allocator_spatial_partitioning=false`
keeps a preallocated GPU pool in one piece for a step's temporaries. If you
set either flag yourself in XLA_FLAGS, your value is kept. If the JAX
backend has already opened, the flags cannot take effect, and Dew logs a
warning.
"""

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

from dew.logging import configure as _configure_logging
from dew.telemetry.devices import (
    keep_roundings as _keep_roundings,
    unpartition_gpu_pool as _unpartition_gpu_pool,
)

_configure_logging()
_keep_roundings()
_unpartition_gpu_pool()

if TYPE_CHECKING:  # the surface above, with its types, for checkers and editors
    from dew.artifacts import ImageGrid, Representations, TextSamples, TokenScores, VideoGrid
    from dew.data import Dataset
    from dew.diffusion import Process
    from dew.eval import Mean
    from dew.inference import pipeline
    from dew.inputs import Condition, Field, InputSpec
    from dew.objectives import Objective
    from dew.objectives.base import Aux, EMASpec, Step
    from dew.objectives.supervised import Supervised
    from dew.sampling import CFG, sample
    from dew.telemetry.profile import Profiler
    from dew.training import (
        Best,
        Checkpoints,
        EvalSuite,
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
    "Best": "dew.training", "Keep": "dew.training", "Plateau": "dew.training", "EvalSuite": "dew.training",
    "Trainer": "dew.training", "TrainState": "dew.training", "Step": "dew.training",
    "Aux": "dew.training", "EMASpec": "dew.training", "MeshSpec": "dew.training",
    "Layout": "dew.training", "Checkpoints": "dew.training", "Tracker": "dew.training",
    "WandbTracker": "dew.training", "LocalTracker": "dew.training", "Trackers": "dew.training",
    "MLflowTracker": "dew.training", "TensorBoardTracker": "dew.training",
    "Evaluation": "dew.training",
    "ProfileWindow": "dew.training",
    "Objective": "dew.objectives", "Supervised": "dew.objectives.supervised",
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
    "EvalSuite",
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
    "Supervised",
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
