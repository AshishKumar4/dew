"""The trainer and what it is built from.

`dew.training` knows no modality: it imports nothing from `dew.diffusion`,
`dew.inputs` or `dew.sampling`, and wandb only when a `WandbTracker` logs.
"""

from dew.checkpoints import Checkpoints
from dew.objectives.base import Aux, EMASpec, Metric, Objective, Step, everything, under
from .distributed import DEFAULT_RULES, Layout, MeshSpec, build_mesh
from .optim import build_optimizer
from .runtime import prepare_process, run_timestamp
from .state import TrainState
from .tracker import Tracker, WandbTracker
from .trainer import Profile, Trainer, ema_update, write_back

__all__ = [
    "Aux", "Checkpoints", "DEFAULT_RULES", "EMASpec", "Layout", "MeshSpec", "Metric",
    "Objective", "Profile", "Step", "Tracker", "TrainState", "Trainer", "WandbTracker",
    "build_mesh", "build_optimizer", "ema_update", "everything", "prepare_process",
    "run_timestamp", "under", "write_back",
]
