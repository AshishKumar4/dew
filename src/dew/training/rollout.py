"""What a trainer reads of a rollout: its protocol, the report of its latest call and how the display
shows what it and the validation metrics report."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import jax
from jax.sharding import Mesh

from dew.batch import Batch
from dew.objectives.base import Metric, Shown

if TYPE_CHECKING:
    from dew.state import TrainState
    from dew.training.interval import Interval


class Rollout(Protocol):
    """Produces a batch on the host, before the compiled step reads it.

    Sampling has side effects and cannot be traced, so it runs outside `jit`. The
    trainer calls the rollout with the state, the prefetched batch and a key
    folded from the run key and the step, then reshards the result with
    `shard_batch`. The returned batch must hold arrays of fixed shapes, so the
    step still compiles once per run.

    A rollout may have `metrics`, a mapping of names to floats about its latest
    call (rewards, lag, truncation); each logging interval sends them to the
    tracker and the display as `rollout/<name>`. It may declare how the display
    shows them in `shown`, as an objective does.

    A rollout whose calls change more than the batch they return (its metrics,
    a log, rollouts in flight) may have `branch()`, a context manager a cooldown
    branch's calls run inside, on the branch's states, after which the rollout
    stands as the run left it (`_branched`). One without it is called alike there.
    """

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...


def _rollout_metrics(rollout: Rollout) -> Mapping[str, float]:
    """The rollout's report of its latest call. A rollout is any callable,
    and a plain function reports nothing, so the trainer reads `metrics` at this boundary."""
    return getattr(rollout, 'metrics', {})


def rolled_out(rollout: Rollout | None, state: TrainState, batch: Batch, mesh: Mesh,
               marked: Callable[[Batch], Batch], interval: Interval) -> Batch:
    """`batch` sampled through `rollout`, its host preparation `marked` and placed on `mesh`
    (`shard_batch`), its seconds counted in `interval`; without a rollout, the batch as it is.

    Host-side and untraceable: sampling, scoring, advantages. The key folds the
    step key once more, keeping the rollout's draws off the step's stream; both
    are checkpointed, so a resumed run samples forward. Fixed shapes mean the
    step still compiles once.
    """
    from dew.distributed.batches import shard_batch

    if rollout is None:
        return batch
    began = time.perf_counter()
    key = jax.random.fold_in(jax.random.fold_in(state.key, state.step), 1)
    batch = shard_batch(mesh, marked(rollout(state, batch, key)))
    interval.rollout_seconds += time.perf_counter() - began
    return batch


def _branched(rollout: Rollout | None) -> contextlib.AbstractContextManager:
    """The context a cooldown branch's rollout calls run in: the rollout's `branch()`, which a rollout
    is any callable and a plain function has none of, so the trainer reads it at this boundary."""
    branch = getattr(rollout, "branch", None)
    return contextlib.nullcontext() if branch is None else branch()


def _reported(rollout: Rollout | None, metrics: Sequence[Metric]) -> dict[str, Shown]:
    """How the display shows what the rollout and the validation metrics
    report. Both may declare `shown`, which their protocols leave optional,
    so the trainer reads it at this boundary."""
    return {
        **getattr(rollout, "shown", {}),
        **{metric.name: shown for metric in metrics if (shown := getattr(metric, "shown", None)) is not None},
    }
