"""Cooldown branches of a run (Hägele et al. 2024, "Scaling Laws and Compute-Optimal Training Beyond Fixed
Training Durations", arXiv:2405.18392): at chosen points of a run on a constant or slowly decaying rate,
a branch from the run's state trains on with its rate decayed to zero, is scored by the run's validation
suites as `cooldown_<split>/<metric>`, and is discarded, the run continuing as though none ran. A
warmup-stable-decay schedule (`dew.optim.Linear`'s `decay_start`) so reads what a run annealed at each
point would score, without a run for each."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Literal

from dew._records import JSON, is_share, share
from dew.training.budget import every_step, recorded
from dew.training.cadence import Tokens


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

    def branches(self, steps: int | None, tokens: int | None, accumulation: int) -> tuple[Branch, ...]:
        """Each point resolved against a run of `steps` microsteps, or a budget of `tokens` alone,
        `accumulation` microsteps an update (`every_step`); a point of steps inside an update, whose
        branch would begin with that update's window part filled, is refused."""
        inside = [at for at in self.at if isinstance(at, int) and at % accumulation]
        if inside:
            raise ValueError(f"cooldowns at steps {inside} branch inside an update of {accumulation} steps")
        points = [every_step(at, steps, tokens, accumulation, name="cooldowns.at") for at in self.at]
        return tuple(Branch(point, self) for point in points if point is not None)

    def factor(self, update: int, updates: int) -> float:
        """The share of the run's rate a branch of `updates` updates trains its `update`th at."""
        done = update / updates
        return 1.0 - done if self.shape == "linear" else 1.0 - math.sqrt(done)

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
    branches (`Tokens`)."""

    at: int | Tokens
    cooldowns: Cooldowns

    def due(self, step: int, real: int, before: int) -> bool:
        """Whether the step `step`, which brought the committed tokens from `before` to `real`, branches."""
        if isinstance(self.at, Tokens):
            return before < self.at.every <= real
        return step == self.at

    def recorded(self) -> JSON:
        """The point as a run's start records it (`dew.telemetry.FitStarted.cadences`)."""
        return recorded(self.at)


def scaled(rates: Mapping[str | None, Mapping[str, float]], factor: float
           ) -> dict[str | None, dict[str, float]]:
    """A config's rates (`OptimConfig.rates`), each group's every rate-valued field times `factor`:
    the rates a branch trains at, which `dew.optim.with_rates` sets in the optimizer's state. A
    schedule's values are proportional to its rate-valued fields (`ScheduleBase.values`), so its
    every value is the run's times `factor`."""
    return {group: {name: value * factor for name, value in held.items()} for group, held in rates.items()}


def check_branches(branches: tuple[Branch, ...], suites: Mapping[str, object], *, configured: bool,
                   rollout: bool, scheduled: bool) -> None:
    """Refuse cooldown branches a run could not score, decay or read as it reads on: one with no
    validation suite to score them; an optimizer that is no `OptimConfig`, whose rates the branch could
    not scale (`configured`); a rollout's (`rollout`), whose acquisition one reader at a time owns; and
    a schedule's of stages that end on the committed-token clock (`scheduled`), whose stages the
    branch's reads would not end."""
    if not branches:
        return
    if not suites:
        raise ValueError("a cooldown branch is scored by the run's validation suites, and this run "
                         "scores none: pass validation=, or eval_every= or metrics= over its val split")
    if not configured:
        raise ValueError("a cooldown scales the rates an OptimConfig's solvers hold; this trainer's "
                         "optimizer is an optax transformation")
    if rollout:
        raise ValueError("a cooldown branch reads a stream of its own, and a rollout's acquisition is "
                         "one reader's at a time: cooldowns of rollout runs are not built")
    if scheduled:
        raise ValueError("a cooldown branch of a schedule's stages ending on the committed-token clock "
                         "would not end them as the run does: cooldowns of token schedules are not built")


__all__ = ["Branch", "Cooldowns"]
