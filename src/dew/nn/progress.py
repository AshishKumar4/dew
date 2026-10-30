"""The training step's position, which a module whose computation changes over a run reads: an annealed
coefficient, a schedule of its own.

It is a declared input, never ambient state: a model that reads it says so (`ReadsProgress`), takes it as
its call's `progress` keyword, and carries it to its layers in the call's `AttentionMetadata`, so a
compiled forward keys on it as on any argument and a cached one never holds a stale value (ledger 2537).
A run's training loss passes the updates the run committed before the step (`Step.committed`) to a model
that reads it, and so does a derivative that runs the model again (node perturbation's estimate); a call
that passes none reads the run's end (`END`), no updates and past every anneal: a validation's loss and
its evaluation alike, as the model the run would hand on, and evaluation, inference and serving after it.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import jax
from flax import linen as nn, struct


@struct.dataclass
class Progress:
    """Where a run's training stands at a step: a pytree, its updates a leaf a compiled call takes as an
    input and keys on, the run's end (no updates) a tree of its own."""

    updates: jax.Array | None
    """The updates the run committed before the step, an int32 scalar the step traces; None past the
    run's training (evaluation, inference, serving), where every schedule a module keeps has ended."""


END = Progress(None)
"""The run's end: a module's schedules all ended, as a model is evaluated and served."""


@runtime_checkable
class ReadsProgress(Protocol):
    """A model whose computation changes over a run (`Progress`): it takes the step's progress as its
    call's `progress` keyword where `reads_progress` holds, which a run's training loss passes it."""

    def reads_progress(self) -> bool: ...


class ProgressKeywords(TypedDict, total=False):
    """The keywords a model call takes for the run's progress (`progressed`)."""

    progress: Progress


def progressed(model: nn.Module, progress: Progress) -> ProgressKeywords:
    """The keywords a call of `model` takes for `progress`: `{"progress": progress}` where the model
    reads it (`ReadsProgress`), else none, so a model that reads none is called as it always is."""
    return {"progress": progress} if isinstance(model, ReadsProgress) and model.reads_progress() else {}


__all__ = ["END", "Progress", "ProgressKeywords", "ReadsProgress", "progressed"]
