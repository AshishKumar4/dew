"""The trainer and what it is built from.

`dew.training` knows no modality. It imports nothing from `dew.diffusion`,
`dew.inputs` or `dew.sampling`, and wandb only when a `WandbTracker` logs.
"""

from dew.checkpoints import Checkpoints, Keep
from dew.objectives.base import Aux, EMASpec, Metric, Objective, Step, everything, under

from .distributed import DEFAULT_RULES, Layout, MeshSpec
from .evaluation import Evaluation
from .quantization import Quantization
from .runtime import Preempted, prepare_process, run_timestamp
from .selection import Best
from .state import TrainState
from .tracker import LocalTracker, MLflowTracker, TensorBoardTracker, Tracker, Trackers, WandbTracker
from .trainer import Plateau, ProfileWindow, Rollout, Trainer
from .transaction import ema_update, write_back

__all__ = ["DEFAULT_RULES", "Aux", "Best", "Checkpoints", "EMASpec", "Evaluation", "Keep", "Layout",
           "LocalTracker", "MLflowTracker", "MeshSpec", "Metric", "Objective", "Plateau", "Preempted",
           "ProfileWindow", "Quantization", "Rollout", "Step", "TensorBoardTracker", "Tracker", "Trackers",
           "TrainState", "Trainer", "WandbTracker", "ema_update", "everything", "prepare_process",
           "run_timestamp", "under", "write_back"]
