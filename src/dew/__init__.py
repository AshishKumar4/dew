"""Dew: one registry, one objective, one trainer.

Importing the package fills the registries, so `models["simple_dit"]`,
`presets.EDM` and `datasets["oxford_flowers102"]` resolve after `import dew`
with nothing else imported. That import opens no JAX backend and loads no
optional dependency; encoders, decoders and datasets fetch what they need
when they are built.
"""

__version__ = "0.1.0"

from dew.registry import (  # noqa: E402
    datasets, encoders, metrics, models, objectives, presets, samplers,
)

# Each of these registers its members with the registries above as a side
# effect of being imported; the names are re-exported for the API's nouns.
import dew.nn.backbones  # noqa: E402,F401
import dew.diffusion.presets  # noqa: E402,F401
import dew.sampling.solvers  # noqa: E402,F401
import dew.data  # noqa: E402,F401
import dew.inputs  # noqa: E402,F401
import dew.eval  # noqa: E402,F401
import dew.objectives.lm  # noqa: E402,F401
import dew.objectives.jepa  # noqa: E402,F401
import dew.objectives.diffusion  # noqa: E402,F401

from dew.artifacts import ImageGrid, Representations, TextSamples, TokenScores, VideoGrid  # noqa: E402
from dew.data import Dataset  # noqa: E402
from dew.diffusion import Process  # noqa: E402
from dew.inputs import Condition, Field, InputSpec  # noqa: E402
from dew.objectives import Objective  # noqa: E402
from dew.sampling import CFG, sample  # noqa: E402
from dew.training import (  # noqa: E402
    Aux, Checkpoints, EMASpec, Layout, MeshSpec, Step, Tracker, TrainState, Trainer,
    WandbTracker,
)

__all__ = [
    "__version__",
    "models", "presets", "samplers", "datasets", "encoders", "metrics", "objectives",
    "Trainer", "TrainState", "Step", "Aux", "EMASpec", "MeshSpec", "Layout",
    "Checkpoints", "Tracker", "WandbTracker",
    "Objective", "Dataset", "Process", "InputSpec", "Field", "Condition",
    "sample", "CFG",
    "ImageGrid", "VideoGrid", "TextSamples", "Representations", "TokenScores",
]
