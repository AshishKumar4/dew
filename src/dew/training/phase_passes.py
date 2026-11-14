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
from collections.abc import Collection, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

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


def _counted[L, E](objective: Objective[L, E], passes: int) -> list[nn.Module]:
    """`objective`'s modules (`program_key`) with each looped one run `passes` times."""
    return [_passed(entry.module, passes) for entry in objective.program_key()]


def _passed(module: nn.Module, passes: int) -> nn.Module:
    """`module` run `passes` times where it loops (`Looping.with_passes`), else as it is."""
    if not _looped(module):
        return module
    assert isinstance(module, Looping)
    passed = module.with_passes(passes)
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

    @classmethod
    def of(cls, objective: Objective[L, E], stated: Sequence[int] | None) -> PhasePasses[L, E]:
        """`stated` checked: each count at least one, the objective's modules looping, the last phase's count
        the one they hold, the model the run hands on. No counts where it states none."""
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
        return cls(objective, counts)

    def count(self, batch: Batch) -> int | None:
        """The count of the phase `batch` was read in (its host `PHASE`, phase 0 where its data has none);
        None where the run states no counts. A phase past the counts is refused, naming both."""
        if not self.counts:
            return None
        phase = batch[PHASE].index if PHASE in batch else 0
        if phase >= len(self.counts):
            raise ValueError(f"a batch of the run's phase {phase} meets phase_passes of {len(self.counts)} "
                             "phase(s): one count a phase")
        return self.counts[phase]

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
        return signature if self.current is None else (signature, self.current)

    def run(self, objective: Objective[L, E], passes: int) -> None:
        """Run `objective`'s looped modules `passes` times, each keeping its other choices, the rest as is."""
        objective.substitute(_counted(objective, passes))

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
        objective = self.base.substituted(_counted(self.base, passes))
        return PhasePasses(objective, self.counts, passes, dict.fromkeys(self.counts, objective))

    def objective_for(self, passes: int | None) -> Objective[L, E]:
        """The run's objective (`base`, at the last phase's count) at `passes` (a phase's `count`, the
        step's `current`): a copy of it whose looped modules run that many passes, kept for every step of
        the count, which a step, its host runner and its batches' counts (`Trainer._counted_batch`, on the
        prefetch worker) hold while validation and the model the run hands on read `base`. Read off
        `base`, never off a holder that holds a count's copy meanwhile; one copy a count, whichever thread
        asks first."""
        if passes is None or passes == self.counts[-1]:
            return self.base
        if passes not in self.built:
            self.built.setdefault(passes, self.base.substituted(_counted(self.base, passes)))
        return self.built[passes]
