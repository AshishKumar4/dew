"""The trainer and the pieces it is built from.

`dew.training` contains nothing specific to a modality. It imports nothing from
`dew.diffusion`, `dew.inputs` or `dew.sampling` at runtime, and imports wandb
only when a `WandbTracker` logs.
"""

from dew.checkpoints import Checkpoints, Keep
from dew.distributed.placement import Layout, MeshSpec
from dew.objectives.base import Aux, EMASpec, Metric, Objective, Step

from .budget import Pause, Plateau
from .evaluation import EvalSuite, Evaluation
from .observe import Observe
from .profiling import ProfileWindow
from .runtime import Preempted, prepare_process, run_timestamp
from .selection import Best
from .tracker import LocalTracker, MLflowTracker, TensorBoardTracker, Tracker, Trackers, Wandb, WandbTracker
from .trainer import Rollout, Trainer
from .transaction import ema_update

__all__ = ["Aux", "Best", "Checkpoints", "EMASpec", "EvalSuite", "Evaluation", "Keep",
           "Layout", "LocalTracker", "MLflowTracker", "MeshSpec", "Metric", "Objective", "Observe", "Pause",
           "Plateau", "Preempted", "ProfileWindow", "Rollout", "Step", "TensorBoardTracker",
           "Tracker", "Trackers", "Trainer", "Wandb", "WandbTracker", "ema_update",
           "prepare_process", "run_timestamp"]
