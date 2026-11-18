"""A run whose looped model runs a count of passes a phase of its data (`PhasePasses`, `Trainer`'s
`phase_passes`): each phase's step compiled with its own count over the one state.

A curriculum over a looped model's depth grows its pass count as its data moves on, one or two passes
while it learns the task's shape, more once it can use them. Each phase's step is the objective's own with
its looped modules run that many passes (`Looping.with_passes`), the same parameters and optimizer state,
compiled once a count; between steps the objective holds the last phase's, the model the run validates and
hands on.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Collection, Hashable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar, Protocol

import numpy as np
from flax import linen as nn

from dew.batch import PHASE, Batch
from dew.nn.protocols import Looped, Looping
from dew.objectives.base import Objective


class Holding[L, E](Protocol):
    """What holds the objective a step is built from: a `Trainer`, and the `StepCompiler` that compiles
    it."""

    objective: Objective[L, E]


def _looped(module: nn.Module) -> bool:
    """Whether `module` runs a block of its layers in passes it can be given (`Looped`, `Looping`)."""
    return isinstance(module, Looped) and isinstance(module, Looping) and module.loop is not None


def _horizons[L, E](objective: Objective[L, E]) -> tuple[int | None, ...]:
    """The final passes each of `objective`'s modules (`program_key`) trains through
    (`Loop.backprop_steps`), None for one that loops through every pass or loops not at all."""
    return tuple(entry.module.loop.backprop_steps if _looped(entry.module) else None
                 for entry in objective.program_key())


def _counted[L, E](objective: Objective[L, E], passes: int,
                   horizons: Sequence[int | None]) -> list[nn.Module]:
    """`objective`'s modules (`program_key`) with each looped one run `passes` times, each training through
    its run's configured final passes (`horizons`, `_horizons` of the run's own objective) or all of them
    where `passes` is fewer: a module a rung restored at another count keeps the run's horizon, not the one
    that count clamped it to."""
    return [_passed(entry.module, passes, horizon)
            for entry, horizon in zip(objective.program_key(), horizons, strict=True)]


def _passed(module: nn.Module, passes: int, trained: int | None) -> nn.Module:
    """`module` run `passes` times where it loops (`Looping.with_passes`), else as it is, training through
    its final `trained` passes (`Loop.backprop_steps`), or all of them where `passes` is fewer."""
    if not _looped(module):
        return module
    assert isinstance(module, Looping)
    passed = module.with_passes(passes, backprop_steps=None if trained is None else min(trained, passes))
    assert isinstance(passed, nn.Module), "a looped module keeps its loop"
    return passed


@dataclass
class PhasePasses[L, E]:
    """The run's own objective (`base`), a pass count a phase of its data (`counts`, none where every step
    runs the objective's own), the count the step being compiled runs (`current`, None for the
    objective's own), and the objective at each count a step was built at (`objective_for`)."""

    base: Objective[L, E]
    counts: tuple[int, ...] = ()
    current: int | None = None
    built: dict[int, Objective[L, E]] = field(default_factory=dict)
    horizons: tuple[int | None, ...] = ()
    """The final passes each of `base`'s modules trains through (`_horizons`), which every count keeps."""
    sampled: SampledPasses | None = None
    """The law a run draws its pass count from per microbatch, in place of a count a phase (`SampledPasses`):
    its step one program a count from 1 to its cap, its cap the count the objective holds, whatever phase
    its data reads (`count`), with no later phase's step of its own (`counts` none)."""

    @classmethod
    def of(cls, objective: Objective[L, E], stated: Sequence[int] | None,
           sampled: SampledPasses | None = None) -> PhasePasses[L, E]:
        """`stated` checked: each count at least one, the objective's modules looping, the last phase's count
        the one they hold, the model the run hands on. No counts where it states none; a run that draws its
        count per microbatch (`sampled`) holds its cap's, and states no count a phase beside it."""
        if sampled is not None:
            if stated is not None:
                raise ValueError("a run states its loop's pass counts once: one a phase of its data "
                                 "(phase_passes) or one drawn per microbatch (sampled_passes)")
            # Its loop holding the cap, checked as a phase's count is; no phase's count beside it.
            return replace(cls.of(objective, (sampled.cap,)), counts=(), current=sampled.cap, sampled=sampled)
        if stated is None:
            return cls(objective)
        counts = tuple(int(count) for count in stated)
        if not counts or min(counts) < 1:
            raise ValueError(f"phase_passes runs each phase at least one pass, got {list(counts)}")
        looped = [entry.module for entry in objective.program_key() if _looped(entry.module)]
        if not looped:
            raise ValueError(f"phase_passes runs a looped model's passes, and {type(objective).__name__} "
                             "runs none (dew.nn.protocols.Looped)")
        held = {module.loop.steps for module in looped
                if isinstance(module, Looped) and module.loop is not None}
        if held != {counts[-1]}:
            raise ValueError(f"the objective's loop runs {sorted(held)} passes and the last phase "
                             f"{counts[-1]}: it holds the count the run hands on")
        return cls(objective, counts, horizons=_horizons(objective))

    def count(self, batch: Batch) -> int | None:
        """The count of the phase `batch` was read in (its host `PHASE`, phase 0 where its data has none);
        None where the run states no counts; the cap where it draws one per microbatch, whatever phase its
        data reads. A phase past the counts is refused, naming both."""
        if self.sampled is not None:
            return self.sampled.cap
        if not self.counts:
            return None
        phase = batch[PHASE].index if PHASE in batch else 0
        if phase >= len(self.counts):
            raise ValueError(f"a batch of the run's phase {phase} meets phase_passes of {len(self.counts)} "
                             "phase(s): one count a phase")
        return self.counts[phase]

    def drawn_count(self) -> int | None:
        """The count the step being compiled runs where the run draws one per microbatch
        (`StepSettings.sampled_passes`); None for any other run."""
        return None if self.sampled is None else self.current

    def counted_at(self) -> list[Objective[L, E]] | None:
        """The run's objective at each count from 1 to its cap where it draws one per microbatch, whose work
        a batch counts at each (`prepare_counts`'s `at_counts`); None for any other run."""
        return None if self.sampled is None else [self.objective_for(count)
                                                  for count in range(1, self.sampled.cap + 1)]

    def begins(self, compiled: Collection[Hashable]) -> bool:
        """Whether the step being compiled (`current`) begins a phase whose count no step of `compiled`
        (keys of `key`) ran, past the run's first: a step planned afresh, at its own count's split, where
        the run is at an update's end (`Trainer._planned_afresh`)."""
        if self.current is None or not compiled:
            return False
        return all(not isinstance(held, tuple) or len(held) != 2 or held[1] != self.current
                   for held in compiled)

    def at(self, batch: Batch) -> int | None:
        """The count of the phase `batch` was read in (`count`), taken as the step being compiled's
        (`current`)."""
        self.current = self.count(batch)
        return self.current

    def key(self, signature: Hashable) -> Hashable:
        """The compiled step's key for a batch's `signature`: with the count it runs, one program a count."""
        return self.key_at(signature, self.current)

    @staticmethod
    def key_at(signature: Hashable, passes: int | None) -> Hashable:
        """The key of a step at `passes` (None for the objective's own) for a batch's `signature` (`key`), a
        later phase's as the loop keys it (`dew.training.ahead.later_phases`)."""
        return signature if passes is None else (signature, passes)

    def run(self, objective: Objective[L, E], passes: int) -> None:
        """Run `objective`'s looped modules `passes` times, each keeping its other choices, the rest as is."""
        objective.substitute(_counted(objective, passes, self.horizons))

    def restored(self, objective: Objective[L, E]) -> None:
        """`objective`'s looped modules at the step's count again (`current`), where a rung restored the
        modules it was taken with; nothing where the step runs the objective's own."""
        if self.current is not None:
            self.run(objective, self.current)

    @contextlib.contextmanager
    def scoped(self, *holders: Holding[L, E]) -> Iterator[None]:
        """Each of `holders`' objective at the step's count while a step is built (`objective_for`), the one
        the step and its host runner hold; the run's own again after."""
        for holder in holders:
            holder.objective = self.objective_for(self.current)
        try:
            yield
        finally:
            for holder in holders:
                holder.objective = self.base

    def apart(self, passes: int) -> PhasePasses[L, E]:
        """These counts with their step at `passes` over an objective of its own: `base`'s copy at that
        count, kept by no one else (`built` holds none of it), which a compile off the loop builds and
        climbs its step's rungs on (`dew.training.ahead.LaterPlans`) while the loop's objectives stay as
        they are."""
        objective = self.base.substituted(_counted(self.base, passes, self.horizons))
        return PhasePasses(objective, self.counts, passes, dict.fromkeys(self.counts, objective),
                           self.horizons)

    def begun(self, first: int, phases: Sequence[int]) -> list[int]:
        """Those of `phases` whose step is planned at its boundary, after the run's `first` phase: each the
        first of a pass count no phase before it ran (`begins`), one plan a count."""
        seen = {self.counts[first]}
        begun = []
        for phase in phases:
            if self.counts[phase] not in seen:
                seen.add(self.counts[phase])
                begun.append(phase)
        return begun

    def objective_for(self, passes: int | None) -> Objective[L, E]:
        """The run's objective (`base`, at the last phase's count) at `passes` (a phase's `count`, the
        step's `current`): a copy of it whose looped modules run that many passes, kept for every step of
        the count, which a step, its host runner and its batches' counts (`Trainer._counted_batch`, on the
        prefetch worker) hold while validation and the model the run hands on read `base`. Read off
        `base`, never off a holder that holds a count's copy meanwhile; one copy a count, whichever thread
        asks first."""
        if passes is None or passes == (self.counts[-1] if self.sampled is None else self.sampled.cap):
            return self.base
        if passes not in self.built:
            self.built.setdefault(passes, self.base.substituted(_counted(self.base, passes, self.horizons)))
        return self.built[passes]


@dataclass(frozen=True)
class SampledPasses:
    """A looped model's pass count drawn afresh for each microbatch a run steps (`count`), from a named law
    bounded to `cap` (row 7660): `law` 'lognormal-poisson', the count `1 + N`, `N ~ Poisson(Λ)`,
    `log Λ ~ Normal(log(mean - 1) - dispersion² / 2, dispersion²)`, so `mean` is the count's mean before its
    bound, conditioned on 1 to `cap` by rejection, as recurrent-depth training draws it (Geiping et al. 2025,
    arXiv 2502.05171). A draw is a function of the run's key and the microbatch's global position alone,
    its place among every microbatch the run dispatched: every process draws alike, and a resume, a replan
    or a step of another split draws on as the run would have."""

    mean: float
    dispersion: float
    cap: int
    law: str = "lognormal-poisson"

    LAWS: ClassVar[tuple[str, ...]] = ("lognormal-poisson",)

    def __post_init__(self) -> None:
        if self.law not in self.LAWS:
            raise ValueError(f"a sampled pass count follows one of the laws {list(self.LAWS)}, "
                             f"not {self.law!r}")
        if not (math.isfinite(self.mean) and self.mean > 1):
            raise ValueError("a log-normal Poisson pass count's mean is above 1, its least count, "
                             f"not {self.mean}")
        if not (math.isfinite(self.dispersion) and self.dispersion >= 0):
            raise ValueError("a log-normal Poisson pass count's dispersion is at least 0, "
                             f"not {self.dispersion}")
        if self.cap < 1:
            raise ValueError(f"a sampled pass count's cap is at least one pass, not {self.cap}")

    def count(self, key: Sequence[int], position: int) -> int:
        """The pass count of the run's microbatch at global `position` under the run's `key` (its words,
        `jax.random.key_data`): a draw of the bounded law, a count past `cap` drawn again."""
        draws = np.random.Generator(np.random.Philox(np.random.SeedSequence([*map(int, key), int(position)])))
        location = math.log(self.mean - 1) - self.dispersion ** 2 / 2
        while True:
            drawn = 1 + int(draws.poisson(math.exp(draws.normal(location, self.dispersion))))
            if drawn <= self.cap:
                return drawn
