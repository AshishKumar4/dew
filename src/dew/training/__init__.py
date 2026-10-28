"""The trainer and the pieces it is built from.

`dew.training` contains nothing specific to a modality. It imports nothing from
`dew.diffusion`, `dew.inputs` or `dew.sampling` at runtime, and imports wandb
only when a `WandbTracker` logs.

Each name is imported from its own module when it is first used (`dew._exports.lookup`), so
`import dew.training.budget` reads that module, not the trainer."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.averaging import ema_update
    from dew.checkpoints import Checkpoints, Keep
    from dew.distributed.placement import Layout, MeshSpec
    from dew.objectives.base import Aux, EMASpec, Metric, Objective, Step
    from dew.training.budget import Pause, Plateau
    from dew.training.cooldown import Cooldowns
    from dew.training.evaluation import EvalSuite, Evaluation
    from dew.training.observe import Observe
    from dew.training.profiling import ProfileWindow
    from dew.training.runtime import Preempted, prepare_process, run_timestamp
    from dew.training.selection import Best
    from dew.training.tracker import (
        LocalTracker,
        MLflowTracker,
        TensorBoardTracker,
        Tracker,
        Trackers,
        Wandb,
        WandbTracker,
    )
    from dew.training.trainer import Rollout, Trainer

_EXPORTS = {
    "Aux": "dew.objectives.base",
    "Best": "dew.training.selection",
    "Checkpoints": "dew.checkpoints",
    "Cooldowns": "dew.training.cooldown",
    "EMASpec": "dew.objectives.base",
    "EvalSuite": "dew.training.evaluation",
    "Evaluation": "dew.training.evaluation",
    "Keep": "dew.checkpoints",
    "Layout": "dew.distributed.placement",
    "LocalTracker": "dew.training.tracker",
    "MLflowTracker": "dew.training.tracker",
    "MeshSpec": "dew.distributed.placement",
    "Metric": "dew.objectives.base",
    "Objective": "dew.objectives.base",
    "Observe": "dew.training.observe",
    "Pause": "dew.training.budget",
    "Plateau": "dew.training.budget",
    "Preempted": "dew.training.runtime",
    "ProfileWindow": "dew.training.profiling",
    "Rollout": "dew.training.trainer",
    "Step": "dew.objectives.base",
    "TensorBoardTracker": "dew.training.tracker",
    "Tracker": "dew.training.tracker",
    "Trackers": "dew.training.tracker",
    "Trainer": "dew.training.trainer",
    "Wandb": "dew.training.tracker",
    "WandbTracker": "dew.training.tracker",
    "ema_update": "dew.averaging",
    "prepare_process": "dew.training.runtime",
    "run_timestamp": "dew.training.runtime",
}


__getattr__ = _lookup(__name__, _EXPORTS)


def __dir__() -> list[str]:
    return list(__all__)


# Written from _EXPORTS (tools/gen_exports.py).
__all__ = [
    "Aux",
    "Best",
    "Checkpoints",
    "Cooldowns",
    "EMASpec",
    "EvalSuite",
    "Evaluation",
    "Keep",
    "Layout",
    "LocalTracker",
    "MLflowTracker",
    "MeshSpec",
    "Metric",
    "Objective",
    "Observe",
    "Pause",
    "Plateau",
    "Preempted",
    "ProfileWindow",
    "Rollout",
    "Step",
    "TensorBoardTracker",
    "Tracker",
    "Trackers",
    "Trainer",
    "Wandb",
    "WandbTracker",
    "ema_update",
    "prepare_process",
    "run_timestamp",
]
