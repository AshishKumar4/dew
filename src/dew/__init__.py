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

`import dew` also makes the JAX user contexts the kernel autotuner and kernel
choices key traces on (`dew.cache.KERNEL_TUNES`, `KERNEL_PINS`,
`KERNEL_CHOICES`): JAX makes one only beside no other JAX call, so before any
thread starts.

`import dew` leaves the process's loggers alone and asks nothing of its
hardware. Dew's log handler, and the refusal of a machine with an NVIDIA GPU
whose JAX opened on the CPU, come with the process's setup: `prepare_process`,
or the first `Trainer.fit` of a program that never called it
(`dew.training.runtime.ready_process`).

`import dew` marks absl's flags parsed when nothing has parsed them, as
`absl.app` would: Grain's worker processes read absl flags as they start, and
a program that never ran `absl.app` would stop them there. A program that
parses its flags itself, through `absl.app` or `FLAGS(argv)`, still does.
"""

from typing import TYPE_CHECKING

from absl import flags as _flags

from dew import cache as cache
from dew._exports import lookup as _lookup
from dew.devices import (
    divide_exactly as _divide_exactly,
    keep_roundings as _keep_roundings,
    unpartition_gpu_pool as _unpartition_gpu_pool,
)

_keep_roundings()
_divide_exactly()
_unpartition_gpu_pool()


def _parse_absl_flags() -> None:
    """Mark absl's flags parsed where nothing has parsed them, as `absl.app` would, for Grain,
    which reads its own flags as it starts worker processes."""
    if not _flags.FLAGS.is_parsed():
        _flags.FLAGS.mark_as_parsed()


_parse_absl_flags()

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
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
        Observe,
        Pause,
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
    "Observe": "dew.training", "Pause": "dew.training",
    "Trainer": "dew.training", "TrainState": "dew.training", "Step": "dew.objectives.base",
    "Aux": "dew.objectives.base", "EMASpec": "dew.objectives.base", "MeshSpec": "dew.training",
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


__getattr__ = _lookup(__name__, _EXPORTS)


def __dir__() -> list[str]:
    return list(__all__)


# Written from _EXPORTS (tools/gen_exports.py), so a type checker, an editor
# and `from dew import *` can all read the public surface without running
# the lazy lookup above.
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
    "Observe",
    "Pause",
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
