"""How long a run trains and what ends it early: its one budget, in steps, epochs, examples or tokens
(`Horizon`, `run_steps`), the stop policies `Trainer.fit` takes beside it (`Plateau`, `Pause`), and
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
from dew.scores import Summary
from dew.training.cadence import Cadence, Counted


@dataclasses.dataclass(frozen=True)
class Horizon:
    """How long a run trains, named once (row 5725): `steps`; `epochs`, passes over its data; `tokens`, the
    targets its loss scores across every microbatch, a resume's before it among them, counted without
    padding as committed updates trained on them (`train/committed_tokens`), the final microbatch crossing
    it; or `examples`, the data's first source examples, each read once with every row it makes, counted
    before they are packed and before an objective makes several rows of one (a decision layout's row a
    question). Unnamed, the run is as long as its data, or a schedule's last stage ends it at its tokens
    (`budget_tokens`), as it bounds a run of any length beside it. A token-only run needs a constant
    learning rate or a schedule with its own step horizon. `max_tokens` caps a run of any other length at
    tokens counted as `tokens` counts them, the first of the two reached ending it."""

    steps: int | None = None
    epochs: int | None = None
    tokens: int | None = None
    examples: int | None = None
    max_tokens: int | None = None

    def __post_init__(self):
        if len(self.named()) > 1:
            raise ValueError(f"{' and '.join(self.named())} each name the run length; set one")
        if self.tokens is not None and self.max_tokens is not None:
            raise ValueError("a run of tokens is capped at them; max_tokens caps a run of another length")
        for name in (*self.named(), "max_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"a run's {name} is a count, not {value}")

    def named(self) -> tuple[str, ...]:
        """The length it names, its one field of steps, epochs, tokens and examples set, none where it
        names none."""
        return tuple(name for name in ("steps", "epochs", "tokens", "examples")
                     if getattr(self, name) is not None)

    @property
    def token_cap(self) -> int | None:
        """The tokens the run stops at: its length in tokens, or its cap beside another length."""
        return self.tokens if self.tokens is not None else self.max_tokens


def run_steps(loader: Loader, horizon: Horizon, accumulation: int, tokens: int | None) -> int | None:
    """The microsteps a run takes over `loader`, the one policy for a fit's and a config's run length:
    `horizon`'s steps, its epochs of it, whose ramp stages begin where updates do, as the stream cuts them,
    or the steps that read the examples it is bounded to (`Loader.example_steps`), a read with no record
    count bounded to its epochs' source examples among them, or None where `tokens` alone bounds it.
    Refuses a run with no length, and epochs of a stream with no count of either. `tokens` is the run's
    token budget (`budget_tokens`), which a schedule's data bounds beside a length named otherwise."""
    steps, epochs, examples = horizon.steps, horizon.epochs, horizon.examples
    if tokens is None and not horizon.named():
        raise ValueError("the run length is steps, epochs, examples or tokens; set one")
    if examples is not None:
        return loader.example_steps(examples, accumulation)
    if epochs is None:
        return steps
    if loader.records is None and loader.epochs is not None and loader.examples is not None:
        # A read with no record count, bounded to `examples`, its passes over its source examples (`fitted`).
        if loader.epochs != epochs:
            raise ValueError(f"this data was compiled for a run of {loader.epochs} epochs, bounded to its "
                             f"{loader.examples} examples, and the run asks for {epochs}: give the run the "
                             f"epochs its data was compiled for, or pass the chain itself")
        return loader.example_steps(loader.examples, accumulation)
    return loader.epoch_steps(epochs, accumulation)


def budget_tokens(loader: Loader, tokens: int | None) -> int | None:
    """The run's token budget: `tokens`, or where a schedule's last stage ends the data at its tokens
    (`dew.data.phases` on `Tokens`), the first reached; what a fit's and a config's run length read."""
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


def every_step(value: Cadence | str, steps: int | None, tokens: int | None, accumulation: int, *,
               name: str, examples: int | None = None) -> Cadence:
    """`value`, a cadence of steps or a share of the run such as `5%` (`dew._records.share`), resolved
    once against the run's planned length: a share of `steps` microsteps, `accumulation` an update,
    the whole updates nearest it and at least one; of a token budget alone, a cadence of that many
    committed tokens (`Counted`); of a run bounded to `examples` no pass counts steps ahead of, a cadence
    of that many consumed examples. A share of a run with no planned length is refused by
    `name`."""
    if not isinstance(value, str):
        return value
    if not is_share(value):
        raise ValueError(f"{name}={value!r} is no cadence: give steps or a share of the run, such as 5%")
    fraction = share(value)
    if steps is not None:
        return max(1, round(fraction * (steps // accumulation))) * accumulation
    if tokens is not None:
        return Counted(max(1, math.ceil(fraction * tokens)), "tokens")
    if examples is not None:
        return Counted(max(1, math.ceil(fraction * examples)), "examples")
    raise ValueError(f"{name}={value!r} is a share of the run, which has no planned length: give it "
                     "steps=, epochs=, examples= or tokens=")


def recorded(cadence: Cadence | str | datetime.timedelta) -> JSON:
    """`cadence` as a run's plan records it: its steps, its committed tokens or its duration."""
    if isinstance(cadence, Counted):
        return {cadence.clock: cadence.every}
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
    with the same one the run is already where it pauses, so nothing trains. A run bounded in examples
    that counts no steps ahead pauses at the first step whose batch brings its consumed examples to the
    fraction."""
    fraction: float

    def __post_init__(self):
        if not 0 < self.fraction < 1:
            raise ValueError(f"Pause needs a fraction of the budget in (0, 1), got {self.fraction}")

    def at(self, steps: int | None, tokens: int | None, accumulation: int, examples: int | None = None
           ) -> tuple[int | None, int | None, int | None]:
        """The step, the real tokens and the consumed examples a run of `steps` microsteps,
        `accumulation` an update, of `tokens` and, where no pass counts its steps ahead, bounded to
        `examples`, pauses at, each None where that budget is not set. Refuses a pause before the first
        update or at the budget's end, where the run ends anyway."""
        updates = None if steps is None else steps // accumulation
        step = None if updates is None else round(self.fraction * updates) * accumulation
        real = None if tokens is None else math.ceil(self.fraction * tokens)
        consumed = None if examples is None or steps is not None else math.ceil(self.fraction * examples)
        if step == 0 or (updates is not None and step is not None and step >= updates * accumulation) or (
                tokens is not None and real is not None and real >= tokens) or (
                examples is not None and consumed is not None and consumed >= examples):
            raise ValueError(f"Pause({self.fraction}) of {steps} steps and {tokens} tokens falls before the "
                             "first update or at the budget's end; pause between them")
        return step, real, consumed


__all__ = [
    "Horizon",
    "Pause",
    "Plateau",
    "every_step",
    "recorded",
    "run_steps",
    "run_tokens",
]
