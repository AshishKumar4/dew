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

from dew.batch import CLOSES_WINDOW, CONSUMED_COUNT, EXAMPLES_CONSUMED, VALID_ROWS, Batch, ConsumedCount
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

    A rollout whose next batches depend on state it carries between calls (its
    environments mid episode, a collection it serves in minibatches, a replay)
    may have `saved()`, that state as a mapping of arrays (host arrays of 32
    bits or fewer a leaf, which the trainer places), None before its first
    call; `saved_shapes()`, its leaves' shapes and dtypes, readable before any
    call; `restored(tree)`, which carries on from a saved tree; and
    `branch(name)`, so a cooldown branch leaves that state as the run left it.
    Each checkpoint of the run holds the run's rollout's tree beside its state
    (`dew.training._fit._rollout_saved`), and a resume hands it back, placed
    as the run's key is (`_rollout_resumed`), so a resumed run samples the
    batches the run that never stopped did. It is the pool's: a run of several
    processes is refused one as the trainer is made (`_rollout_admitted`). The
    contract is the post-training spec's (docs/design/post-training.md, §14).
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
    batch it makes consumes the examples of the batches it read (`clocked_rollout`).

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
    read = [batch]
    further = {"further": _kept(run.further(interval), read)} if _draws_further(rollout) else {}
    with run.held():
        made = rollout(state, _unclocked(batch), key, **further)
        batch = shard_batch(mesh, counted(marked(clocked_rollout(made, read)), read))
    interval.rollout_seconds += time.perf_counter() - began
    return batch


CLOCK = (EXAMPLES_CONSUMED, CONSUMED_COUNT, CLOSES_WINDOW)
"""The fields a batch of a run on the examples clock carries for the run's progress, which the batch it
makes of it carries instead (`clocked_rollout`): none reaches the rollout."""


def _unclocked(batch: Batch) -> Batch:
    """`batch` as a rollout reads it: without the fields of the run's clock (`CLOCK`)."""
    return {name: value for name, value in batch.items() if name not in CLOCK}


def _kept(read: Callable[[], Batch | None], kept: list[Batch]) -> Callable[[], Batch | None]:
    """`read`, each batch it reads kept in `kept` and handed on without the clock's fields."""
    def reading() -> Batch | None:
        batch = read()
        if batch is None:
            return None
        kept.append(batch)
        return _unclocked(batch)
    return reading


def counted(batch: Batch, read: Sequence[Batch]) -> Batch:
    """A rollout's prepared `batch` holding what the batches it `read` consumed, as they counted it
    (`ConsumedCount`, a pool's whole count where its reads are pooled); as it is in a run on another
    clock."""
    if CONSUMED_COUNT not in read[0]:
        return batch
    return {**batch, CONSUMED_COUNT: ConsumedCount(sum(each[CONSUMED_COUNT].examples for each in read))}


def _held(column) -> int:
    """The examples this process's rows of a consumed `column` hold: a placed column's shards it holds,
    each rows' once, or a host column whole."""
    if not isinstance(column, jax.Array):
        return int(np.sum(np.asarray(column)))
    counts = {str(shard.index): shard.data for shard in column.addressable_shards}
    return int(sum(np.sum(np.asarray(held)) for held in counts.values()))


def clocked_rollout(made: Batch, read: Sequence[Batch]) -> Batch:
    """The batch a rollout `made` of the batches it `read` (the step's, then each it drew further), as a
    run on the examples clock steps it: consuming the examples they consumed (`EXAMPLES_CONSUMED`, each
    read batch's `ConsumedCount`), all on its first row, as a step sums them over its rows, and closing
    the update window where the step's batch does (`CLOSES_WINDOW`); a batch read with no real row (a
    pool's share whose stream ran out, `dew.training._fit.pooled_read`) makes one with none. A pool's
    process counts the rows it holds, so the step's sum over the pool's rows counts each once. A run on
    another clock reads no count: its rollout's batch is as made."""
    first = read[0]
    if CONSUMED_COUNT not in first:
        return made
    consumed = np.zeros(rows_of(made), np.int32)
    consumed[0] = sum(_held(each[EXAMPLES_CONSUMED]) for each in read)
    clocked = {**made, EXAMPLES_CONSUMED: consumed}
    if CLOSES_WINDOW in first:
        clocked[CLOSES_WINDOW] = first[CLOSES_WINDOW]
    if VALID_ROWS in first and not np.any(np.asarray(first[VALID_ROWS])):
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
