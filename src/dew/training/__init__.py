"""The trainer and what it is built from.

`dew.training` knows no modality. It imports nothing from `dew.diffusion`,
`dew.inputs` or `dew.sampling`, and wandb only when a `WandbTracker` logs.
"""

from dew.checkpoints import Checkpoints
from dew.objectives.base import Aux, EMASpec, Metric, Objective, Step, everything, under

from .distributed import DEFAULT_RULES, Layout, MeshSpec, build_mesh, data_partition
from .evaluation import Evaluation, evaluate
from .optim import build_optimizer
from .quantization import Quantization, apply_quantization
from .runtime import Preempted, prepare_process, run_timestamp
from .state import TrainState
from .tracker import LocalTracker, MLflowTracker, TensorBoardTracker, Tracker, Trackers, WandbTracker
from .trainer import ProfileWindow, Rollout, Trainer
from .transaction import ema_update, write_back

__all__ = ["DEFAULT_RULES", "Aux", "Checkpoints", "EMASpec", "Evaluation", "Layout", "LocalTracker",
           "MLflowTracker", "MeshSpec", "Metric", "Objective", "Preempted", "ProfileWindow", "Quantization",
           "Rollout", "Step",
           "TensorBoardTracker", "Tracker", "Trackers", "TrainState", "Trainer", "WandbTracker",
           "apply_quantization", "build_mesh", "build_optimizer", "data_partition", "ema_update",
           "evaluate", "everything",
           "prepare_process", "run_timestamp", "under", "write_back"]
