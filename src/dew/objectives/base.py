"""What the trainer is optimizing.

The trainer owns training mechanics - sharding, EMA bookkeeping, checkpoints,
logging, the loops. An Objective owns what is being learned: the parameters it
holds, the loss it computes from a batch, the telemetry that loss reports, and
the artifacts validation scores. Swapping the objective swaps the research
question without touching any of the mechanics.

The loss always returns auxiliary metrics alongside the scalar, so an objective
with several loss terms or with per-step diagnostics (JEPA's collapse
telemetry) can surface them without the trainer knowing what they mean.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax


@dataclass
class EMASpec:
    """Which slice of the parameters the EMA copy tracks, and how fast.

    decay is a step-indexed schedule rather than a constant because momentum
    ramps are load-bearing for some objectives (I-JEPA anneals 0.996 -> 1.0).
    path selects a subtree; the empty path means the whole parameter tree.
    """
    decay: Callable[[Any], Any]
    path: Tuple[str, ...] = ()


class Objective(ABC):
    """The learning problem: parameters, loss, EMA policy, validation artifacts."""

    tag: str = "objective"  # names the checkpoint artifact this run publishes
    ema: EMASpec

    @abstractmethod
    def init_params(self, rng: jax.Array) -> Any:
        """Build the full parameter tree, which may hold several sub-modules."""

    @abstractmethod
    def loss(self, params: Any, ema_params: Any, batch: Any, rng: jax.Array,
             step: Any) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Scalar loss plus auxiliary metrics, differentiated w.r.t. params."""

    @abstractmethod
    def make_validation_step(self, **kwargs) -> Callable[[Any, Any], Any]:
        """Build (val_state, batch) -> artifacts, which eval metrics score."""

    def log_validation_artifacts(self, wandb, artifacts, step: int):
        """Visualize the artifacts. Nothing to draw unless an objective says so."""
