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
from typing import Any, Callable, Dict, Optional, Tuple

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


def shape_and_dtype(entry) -> Tuple[Tuple[int, ...], Any]:
    """Split an `input_shapes` entry into its shape and the dtype it inits as.

    An entry is a plain shape, which inits as float32, or a `(shape, dtype)`
    pair for an input that is not: a language model is fed int32 token ids.
    """
    if len(entry) == 2 and isinstance(entry[0], (tuple, list)) and not isinstance(entry[1], int):
        return tuple(entry[0]), entry[1]
    return tuple(entry), None


class Objective(ABC):
    """The learning problem: parameters, loss, EMA policy, validation artifacts."""

    tag: str = "objective"  # names the checkpoint artifact this run publishes
    ema: EMASpec
    input_shapes: Optional[Dict[str, Any]] = None
    """Shapes of the init batch, for a trainer given no input_config; each entry
    is a shape, or a `(shape, dtype)` pair when float32 is the wrong dtype."""

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
