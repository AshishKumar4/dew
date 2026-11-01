"""What a trainer reads of a rollout: its protocol, the report of its latest call and how the display
shows what it and the validation metrics report."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import jax
import numpy as np
from jax.sharding import Mesh

from dew.batch import CLOSES_WINDOW, CONSUMED_COUNT, EXAMPLES_CONSUMED, VALID_ROWS, Batch
from dew.data.dataset import rows_of
from dew.objectives.base import Metric, Shown

if TYPE_CHECKING:
    from dew.state import TrainState
    from dew.training._fit import FitRun
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

    A rollout whose `draws_further` is true may read data batches past the
    step's, as DAPO's dynamic sampling does until its training batch is full:
    the trainer calls it with `further`, a read of the training stream's next
    batch, None where it ran out (`FitRun.further`), every process reading
    as many, whose position the step's checkpoint saves as any read's.

    A rollout whose calls change more than the batch they return (its metrics,
    a log, a journal, rollouts in flight) may have `branch(name)`, a context
    manager a cooldown branch's calls run inside, on the branch's states, `name`
    the branch's own (`cooldown-<step>`, the step it branches at), after which
    the rollout stands as the run left it (`_branched`). One without it is
    called alike there.
    """

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...


def _rollout_metrics(rollout: Rollout) -> Mapping[str, float]:
    """The rollout's report of its latest call. A rollout is any callable,
    and a plain function reports nothing, so the trainer reads `metrics` at this boundary."""
    return getattr(rollout, 'metrics', {})


def rolled_out(rollout: Rollout | None, state: TrainState, batch: Batch, mesh: Mesh,
               marked: Callable[[Batch], Batch], interval: Interval, run: FitRun) -> Batch:
    """`batch` sampled through `rollout`, its host preparation `marked` and placed on `mesh`
    (`shard_batch`), its seconds counted in `interval`, `run`'s stream placing nothing meanwhile
    (`FitRun.held`); without a rollout, the batch as it is. In a run on the examples clock the
    batch it makes consumes the examples of the batch it read (`clocked_rollout`).

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
    further = {"further": run.further(interval)} if _draws_further(rollout) else {}
    with run.held():
        made = rollout(state, _unclocked(batch), key, **further)
        batch = shard_batch(mesh, counted(marked(clocked_rollout(made, batch)), batch))
    interval.rollout_seconds += time.perf_counter() - began
    return batch


CLOCK = (EXAMPLES_CONSUMED, CONSUMED_COUNT, CLOSES_WINDOW)
"""The fields a batch of a run on the examples clock carries for the run's progress, which the batch it
makes of it carries instead (`clocked_rollout`): none reaches the rollout."""


def _unclocked(batch: Batch) -> Batch:
    """`batch` as a rollout reads it: without the fields of the run's clock (`CLOCK`)."""
    return {name: value for name, value in batch.items() if name not in CLOCK}


def counted(batch: Batch, read: Batch) -> Batch:
    """A rollout's prepared `batch` holding what the batch it `read` consumed, as it counted it
    (`ConsumedCount`, a pool's whole count where its reads are pooled); as it is in a run on another
    clock."""
    return batch if CONSUMED_COUNT not in read else {**batch, CONSUMED_COUNT: read[CONSUMED_COUNT]}


def _held(column) -> np.ndarray:
    """This process's rows of a `column`: a placed column's shards it holds, each rows' once, or a host
    column whole."""
    if not isinstance(column, jax.Array):
        return np.asarray(column)
    shards = {str(shard.index): np.asarray(shard.data) for shard in column.addressable_shards}
    return np.concatenate([held.reshape(-1) for held in shards.values()])


def clocked_rollout(made: Batch, read: Batch) -> Batch:
    """The batch a rollout `made` of the batch it `read`, as a run on the examples clock steps it:
    consuming the examples the batch consumed (`EXAMPLES_CONSUMED`), all on its first row, as a step sums
    them over its rows, and closing the update window where the batch does (`CLOSES_WINDOW`); a batch with
    no real row on this process (a pool's share whose stream ran out, `dew.training._fit.pooled_read`)
    makes one with none. Each process counts the rows it holds, so the step's sum over the pool's rows
    counts each once. A run on another clock reads no count: its rollout's batch is as made."""
    if CONSUMED_COUNT not in read:
        return made
    consumed = np.zeros(rows_of(made), np.int32)
    consumed[0] = int(np.sum(_held(read[EXAMPLES_CONSUMED])))
    clocked = {**made, EXAMPLES_CONSUMED: consumed}
    if CLOSES_WINDOW in read:
        clocked[CLOSES_WINDOW] = read[CLOSES_WINDOW]
    if VALID_ROWS in read and not np.any(_held(read[VALID_ROWS])):
        clocked[VALID_ROWS] = np.zeros(len(consumed), bool)
    return clocked


def _draws_further(rollout: Rollout) -> bool:
    """Whether the rollout reads data batches past its step's (`Rollout`'s `draws_further`), which a
    rollout is any callable and a plain function declares none of, so the trainer reads it at this
    boundary."""
    return getattr(rollout, "draws_further", False)


def _branched(rollout: Rollout | None, name: str) -> contextlib.AbstractContextManager:
    """The context the cooldown branch `name`'s rollout calls run in: the rollout's `branch(name)`,
    which a rollout is any callable and a plain function has none of, so the trainer reads it at this
    boundary."""
    branch = getattr(rollout, "branch", None)
    return contextlib.nullcontext() if branch is None else branch(name)


def _reported(rollout: Rollout | None, metrics: Sequence[Metric]) -> dict[str, Shown]:
    """How the display shows what the rollout and the validation metrics
    report. Both may declare `shown`, which their protocols leave optional,
    so the trainer reads it at this boundary."""
    return {
        **getattr(rollout, "shown", {}),
        **{metric.name: shown for metric in metrics if (shown := getattr(metric, "shown", None)) is not None},
    }
