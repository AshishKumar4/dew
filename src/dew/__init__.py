"""Dew: one objective, one trainer.

Each name exported here is imported from its own module when you first
access it, not at `import dew`. So `import dew.training` loads only the
training layer, with no modality, encoder or tracker backend. Importing
`dew` opens no JAX backend and loads no optional dependency; encoders,
decoders and datasets load what they need when they are built.

`import dew` does set XLA flags, before anything it imports loads jaxlib, so
they hold for every XLA runtime the process loads (`dew._devices.flags_hold`):
one everywhere and, when JAX's CUDA plugin is installed, two more.
`--xla_allow_excess_precision=false` makes XLA round values declared in a
narrow dtype such as bf16 where the program rounds them. On CUDA,
`--xla_gpu_enable_allocator_spatial_partitioning=false` keeps a preallocated
GPU pool in one piece for a step's temporaries, and
`--xla_backend_extra_options=-nvptx-prec-divf32=2` divides fp32 to the nearest
float, as a CPU and the references Dew is held to do (`dew._devices`). If you
set any of them yourself in XLA_FLAGS, your value is kept. Where jax was
imported before Dew, XLA may have parsed its flags already and Dew's are
unproven; where the JAX backend has already opened, they cannot take effect,
and Dew logs a warning.

`import dew` also makes the JAX user contexts the kernel autotuner and kernel
choices key traces on (`dew._cache.KERNEL_TUNES`, `KERNEL_PINS`,
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

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

import egglog as _egglog
from absl import flags as _flags

from dew._devices import (
    connect_at_creation as _connect_at_creation,
    divide_exactly as _divide_exactly,
    keep_roundings as _keep_roundings,
    unpartition_gpu_pool as _unpartition_gpu_pool,
)
from dew._exports import lookup as _lookup

# egglog's builtin declarations keep the frame of the module that first imports egglog, and through it every
# caller's, until each is first used: imported here, it holds `import dew`'s importer alone, never the locals
# of a call that compiles a process's first program through the algebra pass (dew.algebra).
del _egglog
_keep_roundings()
_divide_exactly()
_unpartition_gpu_pool()
_connect_at_creation()


def _parse_absl_flags() -> None:
    """Mark absl's flags parsed where nothing has parsed them, as `absl.app` would, for Grain,
    which reads its own flags as it starts worker processes."""
    if not _flags.FLAGS.is_parsed():
        _flags.FLAGS.mark_as_parsed()


_parse_absl_flags()

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew._version import __version__
    from dew.artifacts import ImageGrid, Representations, TextSamples, TokenScores, VideoGrid
    from dew.data import Dataset
    from dew.diffusion import Process
    from dew.distributed import Layout, MeshSpec
    from dew.eval import Mean
    from dew.inference import pipeline
    from dew.inputs import Condition, Field, InputSpec
    from dew.nn.kernels.generation import GpuFacts, gpu_facts
    from dew.objectives import Objective
    from dew.objectives.base import Aux, EMASpec, Step
    from dew.objectives.supervised import Supervised
    from dew.sampling import CFG, sample
    from dew.state import TrainState
    from dew.telemetry.profile import Profiler
    from dew.training import (
        Best,
        Checkpoints,
        EvalSuite,
        Evaluation,
        Keep,
        LocalTracker,
        MLflowTracker,
        Observe,
        Pause,
        Plateau,
        ProfileWindow,
        TensorBoardTracker,
        Tracker,
        Trackers,
        Trainer,
        WandbTracker,
    )

_EXPORTS = {
    "Best": "dew.training", "Keep": "dew.training", "Plateau": "dew.training", "EvalSuite": "dew.training",
    "Observe": "dew.training", "Pause": "dew.training",
    "Trainer": "dew.training", "TrainState": "dew.state", "Step": "dew.objectives.base",
    "Aux": "dew.objectives.base", "EMASpec": "dew.objectives.base", "MeshSpec": "dew.distributed",
    "Layout": "dew.distributed", "Checkpoints": "dew.training", "Tracker": "dew.training",
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
    "gpu_facts": "dew.nn.kernels.generation", "GpuFacts": "dew.nn.kernels.generation",
    "ImageGrid": "dew.artifacts", "VideoGrid": "dew.artifacts",
    "TextSamples": "dew.artifacts", "Representations": "dew.artifacts",
    "TokenScores": "dew.artifacts",
    "__version__": "dew._version",
}


__getattr__, __dir__ = _lookup(__name__, _EXPORTS)


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
    "GpuFacts",
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
    "gpu_facts",
    "pipeline",
    "sample",
]

if TYPE_CHECKING:
    from dew import _cache as _cache
else:
    # `dew._cache` makes JAX's user contexts at `import dew`, before any thread starts, and its `import jax`
    # loads jaxlib: after Dew's flags are in XLA_FLAGS, where every runtime XLA loads parses them
    # (`dew._devices`).
    _import_module("dew._cache")
