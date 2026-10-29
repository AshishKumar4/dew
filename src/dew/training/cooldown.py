"""Cooldown branches of a run (Hägele et al. 2024, "Scaling Laws and Compute-Optimal Training Beyond Fixed
Training Durations", arXiv:2405.18392): at chosen points of a run on a constant or slowly decaying rate,
a branch from the run's state trains on with its rate decayed to zero, is scored by the run's validation
suites as `cooldown_<split>/<metric>`, and is discarded, the run continuing as though none ran. A
warmup-stable-decay schedule (`dew.optim.Linear`'s `decay_start`) so reads what a run annealed at each
point would score, without a run for each."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from dew import _records
from dew._records import JSON, is_share, share
from dew.training.budget import every_step, recorded
from dew.training.cadence import Clock, Counted


@dataclasses.dataclass(frozen=True)
class Cooldowns:
    """Where a run branches into cooldowns, and how each decays.

    Each of `at` is a step, or a share of the run such as `20%`, resolved as a cadence's is
    (`dew.training.budget.every_step`): the whole updates nearest that share of its planned steps, or on
    a token budget alone the step whose commit passes that share of its tokens. A branch trains `length`
    more steps, or a share of the steps the run has taken, such as `20%`, the whole updates nearest it:
    the run's own batches from where it stands, at its own rate times a decay from 1 to 0 over the
    branch's updates, `1-sqrt` (1 - sqrt(u / U), the paper's best) or `linear` (1 - u / U, WSD's linear
    decay: `Linear(decay_start=S, decay_steps=S + U)` from a constant rate is the same rates).
    """

    at: tuple[int | str, ...]
    length: int | str = "20%"
    shape: Literal["1-sqrt", "linear"] = "1-sqrt"

    def __post_init__(self):
        object.__setattr__(self, "at", tuple(self.at))
        if not self.at:
            raise ValueError("cooldowns branch at one point or more")
        for written in (*self.at, self.length):
            if isinstance(written, str):
                share(written)
            elif isinstance(written, bool) or written < 1:
                raise ValueError(f"a cooldown's point and length are steps, 1 or more, or a share of the run "
                                 f"such as 20%, not {written!r}")
        if self.shape not in ("1-sqrt", "linear"):
            raise ValueError(f"a cooldown decays as 1-sqrt or linear, not {self.shape!r}")

    def branches(self, steps: int | None, tokens: int | None, accumulation: int, examples: int | None = None
                 ) -> tuple[Branch, ...]:
        """Each point resolved against a run of `steps` microsteps, a budget of `tokens` alone, or a bound
        of `examples` no pass counts steps ahead of, `accumulation` microsteps an update (`every_step`); a
        point of steps inside an update, whose branch would begin with that update's window part filled,
        is refused."""
        inside = [at for at in self.at if isinstance(at, int) and at % accumulation]
        if inside:
            raise ValueError(f"cooldowns at steps {inside} branch inside an update of {accumulation} steps")
        points = [every_step(at, steps, tokens, accumulation, name="cooldowns.at", examples=examples)
                  for at in self.at]
        return tuple(Branch(point, self) for point in points if point is not None)

    def factor(self, update: int, updates: int) -> float:
        """The share of the run's rate a branch of `updates` updates trains its `update`th at."""
        return decay(self.shape, update, updates)

    def steps(self, current: int, accumulation: int) -> int:
        """The steps of a branch from step `current`: `length`, or its share of the steps so far, the
        whole updates nearest it and at least one. A length of steps no whole number of updates is
        refused, as its last microsteps would commit none."""
        if is_share(self.length):
            assert isinstance(self.length, str)
            return max(1, round(share(self.length) * (current // accumulation))) * accumulation
        assert isinstance(self.length, int)
        if self.length % accumulation:
            raise ValueError(f"a cooldown of {self.length} steps is no whole number of updates of "
                             f"{accumulation} steps")
        return self.length


@dataclasses.dataclass(frozen=True)
class Branch:
    """One cooldown point as a run resolved it: a step, or a committed-token count whose passing step
    branches (`Counted`)."""

    at: int | Counted
    cooldowns: Cooldowns

    def pending(self, current: int, accumulation: int) -> Pending:
        """The branch from step `current` as it runs, resolved: its step, steps and decay, which a run
        cut short inside it keeps to run it again (`Checkpoints.cooldown`), whatever its config then."""
        return {"step": current, "steps": self.cooldowns.steps(current, accumulation),
                "shape": self.cooldowns.shape}

    def due(self, step: int, counts: Mapping[Clock, tuple[int, int]] | None = None) -> bool:
        """Whether the step `step`, which brought each counted clock's count from the second of its
        `counts` to the first, branches: at its step, or where it crosses its point."""
        if isinstance(self.at, Counted):
            return self.at.crossed(counts or {})
        return step == self.at

    def recorded(self) -> JSON:
        """The point as a run's start records it (`dew.telemetry.FitStarted.cadences`)."""
        return recorded(self.at)


type Pending = Mapping[str, object]
"""A cooldown branch resolved at the `step` it leaves from: its `steps` and its decay's `shape`."""


def resolved(pending: Pending, accumulation: int) -> tuple[int, str]:
    """A pending branch's steps and decay, refused where its steps are no whole number of the run's
    updates of `accumulation` steps, as a resume under another accumulation may find them."""
    steps = _records.integer(pending["steps"], "cooldown steps")
    if steps % accumulation:
        raise ValueError(f"a cooldown of {steps} steps is no whole number of this run's updates of "
                         f"{accumulation} steps")
    return steps, _records.text(pending["shape"], "cooldown shape")


def schedule_ended(token_ends: Sequence[int | None] | None, committed: Callable[[], int]) -> bool:
    """Whether a branch's `committed` tokens reached the end of a schedule whose last stage ends its
    data at its tokens (`dew.data.pipeline.schedule`), as the run's own budget ends it there; the count,
    read from the devices, is asked only of such a schedule."""
    return token_ends is not None and token_ends[-1] is not None and committed() >= token_ends[-1]


def decay(shape: str, update: int, updates: int) -> float:
    """The share of a run's rate the `update`th of a branch's `updates` updates trains at: 1 - sqrt(u / U)
    for `1-sqrt`, 1 - u / U for `linear`."""
    done = update / updates
    return 1.0 - done if shape == "linear" else 1.0 - math.sqrt(done)


def check_branches(branching: bool, suites: Mapping[str, object], *, settable: bool) -> None:
    """Refuse cooldown branches, where the run is `branching`, that it could not score or decay: one
    with no validation suite to score them, and an optimizer whose state holds no learning rate a
    branch can scale (`settable`, `dew.optim.settable_rates`)."""
    if not branching:
        return
    if not suites:
        raise ValueError("a cooldown branch is scored by the run's validation suites, and this run "
                         "scores none: pass validation=, or eval_every= or metrics= over its val split")
    if not settable:
        raise ValueError("a cooldown scales the learning rate the optimizer's state holds, as an "
                         "OptimConfig's solvers and optax.inject_hyperparams of a constant hold it; this "
                         "state holds none, or one Optax computes from a step count at each update")


__all__ = ["Branch", "Cooldowns"]
