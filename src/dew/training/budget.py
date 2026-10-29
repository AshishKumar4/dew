"""How long a run trains and what ends it early: its one budget, in steps, epochs, examples or tokens
(`run_length`, `run_steps`), the stop policies `Trainer.fit` takes beside it (`Plateau`, `Pause`), and
its cadences resolved against it (`every_step`)."""

from __future__ import annotations

import dataclasses
import datetime
import math
import types
from typing import Annotated, Literal

import tyro

from dew._records import JSON, is_share, recorded_duration, share
from dew.data.dataset import Loader
from dew.objectives.base import Metric, TrainingScalar
from dew.training.cadence import Cadence, Tokens
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


def budget_tokens(loader: Loader, tokens: int | None) -> int | None:
    """The run's token budget: `tokens`, or where a schedule's last stage ends the data at its tokens
    (`dew.data.pipeline.schedule`), the first reached; what a fit's and a config's run length read."""
    last = None if loader.token_ends is None else loader.token_ends[-1]
    return last if tokens is None else tokens if last is None else min(tokens, last)


def run_tokens(loader: Loader, tokens: int | None, *, counts_tokens: bool) -> int | None:
    """The run's token budget (`budget_tokens`), refused where the objective counts no tokens."""
    if (tokens is not None or loader.token_ends is not None) and not counts_tokens:
        raise ValueError("tokens= and a schedule's stages need an objective that counts real tokens")
    return budget_tokens(loader, tokens)


@dataclasses.dataclass(frozen=True)
class Plateau:
    """Stops training after `evals` eligible evaluations without an improvement larger than `min_delta`. A
    `Summary` is eligible at the evaluations that form it, those that scored every split it weighs. The
    metric is the object fit scores, or its recorded name, as `Best` takes one (`val/loss`)."""
    # A command line names the metric (`--trainer.stop.metric val/loss`).
    metric: Annotated[Metric | TrainingScalar | Summary | types.MethodType | str,
                      tyro.conf.arg(constructor=str)]
    evals: int = 5
    min_delta: float = 0.0
    mode: Literal['min', 'max'] | None = None
    split: str | None = None

    def __post_init__(self):
        if self.evals < 1 or self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("Plateau needs evals >= 1 and min_delta >= 0")


def every_step(value: int | str | None, steps: int | None, tokens: int | None, accumulation: int, *,
               name: str) -> Cadence:
    """`value`, a cadence of steps or a share of the run such as `5%` (`dew._records.share`), resolved
    once against the run's planned length: a share of `steps` microsteps, `accumulation` an update,
    the whole updates nearest it and at least one; of a token budget alone, a cadence of that many
    committed tokens (`Tokens`). A share of a run with no planned length is refused by `name`."""
    if not isinstance(value, str):
        return value
    if not is_share(value):
        raise ValueError(f"{name}={value!r} is no cadence: give steps or a share of the run, such as 5%")
    fraction = share(value)
    if steps is not None:
        return max(1, round(fraction * (steps // accumulation))) * accumulation
    if tokens is not None:
        return Tokens(max(1, math.ceil(fraction * tokens)))
    raise ValueError(f"{name}={value!r} is a share of the run, which has no planned length: give it "
                     "steps=, epochs=, examples= or tokens=")


def recorded(cadence: Cadence | str | datetime.timedelta) -> JSON:
    """`cadence` as a run's plan records it: its steps, its committed tokens or its duration."""
    if isinstance(cadence, Tokens):
        return {"tokens": cadence.every}
    if isinstance(cadence, datetime.timedelta):
        return recorded_duration(cadence)
    return cadence


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


__all__ = ["Pause", "Plateau", "every_step", "recorded", "run_length", "run_steps",
           "run_tokens"]
