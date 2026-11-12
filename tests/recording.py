"""A tracker that keeps everything a run reports, for a test to read back, an observer that
preempts the run it observes, and a run's state as arrays for a bit-for-bit comparison, whose numbers are
finite first."""

import dataclasses
import os
import signal
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from dew.state import TrainState
from dew.training.observe import Observe


class RecordingTracker:
    """Every `log` as `(step, scalars)` and every `artifact` as `(step, value)`, in order."""

    def __init__(self):
        self.scalars: list[tuple[int, dict]] = []
        self.artifacts: list[tuple[int, object]] = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))


@dataclasses.dataclass
class Preempting(Observe):
    """An observer that sends this process the scheduler's SIGTERM, once, after step `at`, as the loop
    observes it (`Observe.at`), and observes as `Observe` does at its own cadence (`every`, `call`;
    unset, never). The loop reads a notice as the next step ends, so it stops the run after step
    `at` + 1, at the same step every run, as a tracker's log, written off the loop, would not."""

    every: int = 10 ** 9
    call: Callable[[TrainState], object] = lambda state: None
    at_step: int | None = None

    def at(self, step, state):
        if step == self.at_step:
            self.at_step = None
            os.kill(os.getpid(), signal.SIGTERM)
        super().at(step, state)


def preempting(at: int, **observed) -> Preempting:
    """`Preempting` after step `at`."""
    return Preempting(at_step=at, **observed)


def finite(tree) -> bool:
    """Whether every floating or complex entry of `tree`'s leaves (a value, or a tree of them) is finite, as
    a bit-for-bit comparison needs first: NaN equals NaN to an array comparison that matches NaNs, and an
    infinity equals itself, so two runs or two requests gone to NaN or overflow would pass as one.
    Integers, booleans and text hold none."""
    return all(np.asarray(leaf).dtype.kind not in "fc" or bool(np.all(np.isfinite(np.asarray(leaf))))
               for leaf in jax.tree.leaves(tree))


def bits(state) -> list[np.ndarray]:
    """Every leaf of a run's state on the host, a key as its raw bits, for a bit-for-bit comparison, each
    finite first (`finite`): two runs gone to NaN compare alike bit for bit."""
    held = [np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf)
            for leaf in jax.tree.leaves(state)]
    assert finite(held), "a leaf of the run's state is not finite"
    return held

