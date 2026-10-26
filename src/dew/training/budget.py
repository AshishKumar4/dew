"""How long a run trains and what ends it early: its one budget, in steps, epochs, examples or tokens
(`run_length`, `run_steps`), and the stop policies `Trainer.fit` takes beside it (`Plateau`, `Pause`)."""

from __future__ import annotations

import dataclasses
import math
import types
from typing import Literal

from dew.data.dataset import Loader
from dew.objectives.base import Metric, TrainingScalar
from dew.training.evaluation import Summary


def run_length(steps: int | None, epochs: int | None, tokens: int | None, examples: int | None = None, *,
               required: bool = True) -> None:
    """Refuse a run whose length is named twice, by two of `steps`, `epochs` and `examples`, or, where
    `required`, not at all; `tokens` caps any of them or bounds a run alone."""
    named = [name for name, value in (("steps", steps), ("epochs", epochs), ("examples", examples))
             if value is not None]
    if len(named) > 1:
        raise ValueError(f"{' and '.join(named)} each name the run length; set one")
    if required and not named and tokens is None:
        raise ValueError("the run length is steps, epochs or examples, or tokens, which also caps any of "
                         "them")


def run_steps(loader: Loader, steps: int | None, epochs: int | None, tokens: int | None,
              accumulation: int, examples: int | None = None) -> int | None:
    """The microsteps a run takes over `loader`, the one policy for a fit's and a config's run length:
    `steps`, `epochs` of it, whose ramp stages begin where updates do, as the stream cuts them, or the
    steps that read the `examples` it is bounded to (`Loader.example_steps`), or None where `tokens`
    alone bounds it. Refuses a run with no length or two, and epochs of a stream with no record count."""
    run_length(steps, epochs, tokens, examples)
    if examples is not None:
        return loader.example_steps(examples, accumulation)
    if epochs is None:
        return steps
    return loader.epoch_steps(epochs, accumulation)


@dataclasses.dataclass(frozen=True)
class Plateau:
    """Stops training after `evals` eligible evaluations without an improvement larger than `min_delta`. A
    `Summary` is eligible at the evaluations that form it, those that scored every split it weighs."""
    metric: Metric | TrainingScalar | Summary | types.MethodType
    evals: int = 5
    min_delta: float = 0.0
    mode: Literal['min', 'max'] | None = None
    split: str | None = None

    def __post_init__(self):
        if self.evals < 1 or self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("Plateau needs evals >= 1 and min_delta >= 0")


@dataclasses.dataclass(frozen=True)
class Pause:
    """Ends the run once it has trained `fraction` of its budget, its state and data position
    written in full, so a run resumed from that checkpoint trains on as one that never paused: the
    budget still sizes the learning-rate schedule. A budget of steps, epochs or examples pauses at
    the whole update nearest the fraction, a token budget at the first update whose tokens reach
    it, and with both the first reached. Resume without it, or with a later fraction, to go on;
    with the same one the run is already where it pauses, so nothing trains."""
    fraction: float

    def __post_init__(self):
        if not 0 < self.fraction < 1:
            raise ValueError(f"Pause needs a fraction of the budget in (0, 1), got {self.fraction}")

    def at(self, steps: int | None, tokens: int | None, accumulation: int) -> tuple[int | None, int | None]:
        """The step and the real tokens a run of `steps` microsteps, `accumulation` an update, and of
        `tokens` pauses at, each None where that budget is not set. Refuses a pause before the first
        update or at the budget's end, where the run ends anyway."""
        updates = None if steps is None else steps // accumulation
        step = None if updates is None else round(self.fraction * updates) * accumulation
        real = None if tokens is None else math.ceil(self.fraction * tokens)
        if step == 0 or (updates is not None and step is not None and step >= updates * accumulation) or (
                tokens is not None and real is not None and real >= tokens):
            raise ValueError(f"Pause({self.fraction}) of {steps} steps and {tokens} tokens falls before the "
                             "first update or at the budget's end; pause between them")
        return step, real


__all__ = ["Pause", "Plateau", "run_length", "run_steps"]
