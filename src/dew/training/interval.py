"""What a fit sums between its logs and between its checkpoints.

Every step adds its loss, its finiteness, its analytical work and its
optimizer commit to counters that stay on the devices, in one dispatch, so
the step loop never waits on a number; the loop reads them at its logging
cadence and restarts the ones each cadence owns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.data.dataset import rows_of
from dew.objectives.base import Batch
from dew.telemetry.instrumentation import NO_FLOPS, StepFlops
from dew.training.display import TrainingDisplay
from dew.training.model_flops import TOKENS, TRAINING_FLOPS
from dew.training.transaction import Committed

# Consecutive non-finite losses that stop a run.
BAD_LOSS_STEPS = 5


class Book(NamedTuple):
    """The loop's device-side counters.

    `loss` sums the loss since the last checkpoint. `bad_run` is the current
    streak of non-finite losses and `worst_bad_run` the longest since the last
    check. `training` and `tokens` sum the objective's analytical training FLOPs
    and real tokens, and `norms` and `commits` the committed gradient norms
    and their count, since the last log.
    """

    loss: jax.Array
    bad_run: jax.Array
    worst_bad_run: jax.Array
    training: jax.Array
    tokens: jax.Array
    norms: jax.Array
    commits: jax.Array


def fresh_book(loss: jax.Array) -> Book:
    """Return the counters a fresh logging interval starts from, on the device the step returns `loss` on.

    Counters on any other device would move to the step's devices at every count,
    and `bookkeep` would compile again.
    """
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    zeros = (np.zeros((), dtype), np.zeros((), np.int32), np.zeros((), np.int32),
             np.zeros((), dtype), np.zeros((), dtype), np.zeros((), dtype), np.zeros((), np.int32))
    return Book(*jax.device_put(zeros, loss.sharding))


@jax.jit
def bookkeep(book: Book, loss: jax.Array, finite: jax.Array, committed: Committed | None, training=0.0,
             tokens=0.0) -> Book:
    """Advance the loop's counters for one step in a single dispatch.

    Loss counters and commit telemetry advance together, without reading
    either on the host between logging ticks.
    """
    bad_run = jnp.where(finite, 0, book.bad_run + 1)
    dtype = jnp.promote_types(loss.dtype, jnp.float32)
    norms, commits = book.norms, book.commits
    if committed is not None:
        norms, commits = norms + committed.norm.astype(dtype), commits + committed.count
    return Book(book.loss + loss.astype(dtype), bad_run, jnp.maximum(book.worst_bad_run, bad_run),
                book.training + training, book.tokens + tokens, norms, commits)


@dataclasses.dataclass
class Interval:
    """What `fit` sums between logs and between checkpoints.

    `book` holds the device-side counters (`Book`). They move together in one
    dispatch; the host reads them at the logging cadence, never blocking the
    step loop on a result. The first step places them where its loss is, and
    `fresh` keeps them at zero there, for a counter's restart to reuse rather
    than build again.
    `steps` counts since the last checkpoint, the rest since the last log.
    The records and FLOPs are summed per step rather than taken off the
    dataset's batch, so a ramped interval reports the records it read and
    their FLOPs. `model_counted` and `tokens_counted` say whether a step
    since the last log reported its analytical work and real tokens.
    `rollout_seconds` is the time spent sampling, logged under
    train/rollout_seconds when a rollout is set. `waited` is the time the
    loop spent waiting on its next batch and `compiling` the time it spent
    compiling the step for a shape it had not met, both in seconds, since
    the last log. `compiles` counts StepCompiled records over the whole fit,
    so logging and checkpointing never reset it."""

    last_log_time: float
    last_saved: int | None
    tokens_before_log: int = 0
    book: Book | None = None
    fresh: Book | None = None
    steps: int = 0
    since_log: int = 0
    samples: int = 0
    flops: StepFlops | None = NO_FLOPS
    model_counted: bool = False
    tokens_counted: bool = False
    rollout_seconds: float = 0.0
    waited: float = 0.0
    compiling: float = 0.0
    compiles: int = 0

    def count(self, batch: Batch, flops: StepFlops | None, loss: jax.Array, finite: jax.Array,
              committed: Committed | None, aux: Mapping[str, jax.Array] | None = None) -> None:
        """Add one step: its records, its FLOPs, its loss and its optimizer commit."""
        self.since_log += 1
        self.steps += 1
        self.samples += rows_of(batch)
        self.flops = None if self.flops is None or flops is None else self.flops + flops
        if self.book is None:
            self.book = self.fresh = fresh_book(loss)
        aux = {} if aux is None else aux
        self.model_counted |= TRAINING_FLOPS in aux
        self.tokens_counted |= TOKENS in aux
        assert self.fresh is not None
        self.book = bookkeep(self.book, loss, finite, committed,
                             aux.get(TRAINING_FLOPS, self.fresh.training), aux.get(TOKENS, self.fresh.tokens))

    def check_finite(self, step: int, display: TrainingDisplay) -> None:
        """Raise RuntimeError once the loss has been non-finite for
        BAD_LOSS_STEPS steps, and start the longest streak over.

        Deferred to the logging cadence so the step loop never synchronises;
        detection is late by at most that many steps, never missed.
        """
        if self.book is None or self.fresh is None:
            return
        try:
            streak = int(jax.device_get(self.book.worst_bad_run))
        except jax.errors.JaxRuntimeError as error:
            # The counters read every step's loss, so a step that failed on
            # the devices surfaces here, under the counters' program.
            raise RuntimeError(f"a training step before step {step} failed on the devices; JAX reports it "
                               "at the loss counters that read its loss (jit_bookkeep)") from error
        if streak >= BAD_LOSS_STEPS:
            raise RuntimeError(
                f"Loss has been non-finite for {streak} consecutive steps "
                f"ending near step {step}, stopping")
        if streak:
            display.note(f"Non-finite loss for {streak} step(s) before {step}", style="red")
        self.book = self.book._replace(worst_bad_run=self.fresh.worst_bad_run)

    @property
    def real_tokens(self) -> int:
        """Lifetime real tokens from completed logs and the current device counter."""
        current = (int(jax.device_get(self.book.tokens))
                   if self.tokens_counted and self.book is not None else 0)
        return self.tokens_before_log + current

    def logged(self, now: float) -> None:
        """Start the next logging interval at `now`."""
        self.tokens_before_log = self.real_tokens
        self.last_log_time, self.since_log, self.samples = now, 0, 0
        self.flops, self.rollout_seconds, self.waited, self.compiling = NO_FLOPS, 0.0, 0.0, 0.0
        self.model_counted = self.tokens_counted = False
        if self.book is not None and self.fresh is not None:
            self.book = self.book._replace(training=self.fresh.training, tokens=self.fresh.tokens,
                                           norms=self.fresh.norms, commits=self.fresh.commits)

    def saved(self, step: int) -> None:
        """Start the next checkpoint interval after the one saved at `step`."""
        self.last_saved, self.steps = step, 0
        if self.book is not None and self.fresh is not None:
            self.book = self.book._replace(loss=self.fresh.loss)
