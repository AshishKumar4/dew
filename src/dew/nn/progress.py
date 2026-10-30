"""The training step's position, which a module whose computation changes over a run reads (`progress`):
an annealed coefficient, a schedule of its own.

A run's training loss supplies it (`dew.objectives.base.Objective._loss`, the updates the run committed
before the step, `Step.committed`), and so does a derivative that runs the model again (node
perturbation's estimate); a step the trainer traces otherwise, a custom one, supplies none, and a module
that reads it there is refused, never handed a stand-in. Outside the training step it is the run's end, no
updates and past every anneal: a validation's loss and its evaluation alike, as the model the run would
hand on, and evaluation, inference and serving after it.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterator
from typing import Literal

import jax


@dataclasses.dataclass(frozen=True)
class Progress:
    """Where a run's training stands at a step."""

    updates: jax.Array | None
    """The updates the run committed before the step, an int32 scalar the step traces; None past the
    run's training (evaluation, inference, serving), where every schedule a module keeps has ended."""


END = Progress(None)
"""The run's end: a module's schedules all ended, as a model is evaluated and served."""

_TRAINING: Literal["training"] = "training"
"""A step the trainer traces, whose loss has not supplied its progress."""

_CURRENT: contextvars.ContextVar[Progress | Literal["training"] | None] = contextvars.ContextVar(
    "dew_progress", default=None)


def progress() -> Progress:
    """The step's progress where a run's loss supplied it (`stepping`), the run's end (`END`) outside
    any training step; refused within a training step that supplied none (`training`), a custom
    step's, where any value would be a guess."""
    current = _CURRENT.get()
    if current == _TRAINING:
        raise ValueError("a module reads the training step's progress (dew.nn.progress), which this step "
                         "supplies none of: train it through an objective's loss, which supplies it")
    return END if current is None else current


@contextlib.contextmanager
def stepping(value: Progress) -> Iterator[None]:
    """Within the block, modules read `value` as the step's progress (`progress`)."""
    token = _CURRENT.set(value)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def training[**P, R](step: Callable[P, R]) -> Callable[P, R]:
    """`step`, traced as a training step: a module reading the step's progress within it reads what its
    loss supplies (`stepping`), and is refused where nothing supplies it."""
    @functools.wraps(step)
    def traced(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _CURRENT.set(_TRAINING)
        try:
            return step(*args, **kwargs)
        finally:
            _CURRENT.reset(token)
    return traced
