"""Observing a run while it trains: `Trainer.fit(observe=Observe(every, call))`.

A tutorial that shows what a model learns at stages, a sample from each fifth of the run or the
next-token probabilities after each, called `fit` once a stage, each stage resuming the last one's
checkpoint. An observer takes those values inside one fit instead: at every `every`-th step, and at
the run's last, the trainer hands `call` the state the step just produced and keeps what it returns,
copied home (`dew._coordination.collective_host`) before the next step donates the state's buffers.

Every process of a pool calls `fit` with an observer of the same cadence, runs `call` and keeps the
values, as every process takes part in an evaluation: a call may run the model on the mesh, and a
global array's shards come home through every process. The values live in memory, a ledger that a
later fit on the same observer continues; a process that restarts begins a new one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dew._coordination import agree_process_phase, agreed, agreed_same, collective_host

if TYPE_CHECKING:
    from dew.state import TrainState


@dataclasses.dataclass
class Observe:
    """`call(state)` at every `every`-th step of a fit and at its last, its result kept in `values`.

    The state is lent for the call alone: what `call` returns is copied to the host, so a returned
    tree of the state's own arrays (`lambda state: state.averaged`) is a snapshot, not a view of
    buffers the next step reuses. A failure in `call` on any process fails the fit on all of them,
    at that step. `values` holds `(step, value)` in step order; a fit that resumes from a checkpoint
    drops what it holds past the restored step, which a failed run observed and the resumed one
    retrains, and a fit that finds the run already complete observes its final state once.
    """

    every: int
    call: Callable[[TrainState], Any]
    values: list[tuple[int, Any]] = dataclasses.field(default_factory=list)
    # The step this fit last observed, which a run's last step need not observe again.
    _last: int | None = dataclasses.field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.every < 1:
            raise ValueError(f"an observer observes every so many steps, at least one, not {self.every}")

    def opened(self, step: int, state: TrainState, *, complete: bool) -> None:
        """A fit begins at `step`, restored or fresh: agree the cadence and forget later values. A
        fit with nothing to train, its run `complete` at `step`, observes the state there where any
        process holds no value for it, every process alike."""
        agreed_same("observation cadence", lambda: self.every)
        self.values = [(at, value) for at, value in self.values if at <= step]
        self._last = None
        missing = all(at != step for at, _ in self.values)
        # Whether to enter the call and its gather is the pool's decision, never one ledger's.
        if complete and agree_process_phase(None, phase="observation at the restored step",
                                            available=missing):
            self.take(step, state)

    def at(self, step: int, state: TrainState) -> None:
        """After `step`, where the cadence asks for it."""
        if step % self.every == 0:
            self.take(step, state)

    def last(self, step: int, state: TrainState) -> None:
        """The run ended at `step`, done or stopped by a preemption: observe it unless this fit
        just did."""
        if self._last != step:
            self.take(step, state)

    def take(self, step: int, state: TrainState) -> None:
        """`call(state)` on every process, its result copied home and kept as `step`'s value. A fit's
        steps only rise past what `opened` kept, so the value is the ledger's new tail or replaces
        it."""
        value = collective_host(agreed("observation", lambda: self.call(state)), phase="observation")
        if self.values and self.values[-1][0] == step:
            self.values[-1] = (step, value)
        else:
            self.values.append((step, value))
        self._last = step

