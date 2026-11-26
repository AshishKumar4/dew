"""What a trainer reads of a rollout: its protocol, the report of its latest call and how the display
shows what it and the validation metrics report."""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from dew.batch import (
    CLOSES_WINDOW,
    CONSUMED_COUNT,
    EXAMPLES_CONSUMED,
    PHASE,
    VALID_ROWS,
    Batch,
    ConsumedCount,
)
from dew.data.dataset import rows_of
from dew.nn.inputs import local_rows
from dew.objectives.base import Metric, Shown

if TYPE_CHECKING:
    from dew.objectives.base import Objective
    from dew.state import TrainState
    from dew.training._fit import FitRun
    from dew.training.interval import Interval


class Rollout(Protocol):
    """Produces a batch on the host, before the compiled step reads it.

    The trainer calls the rollout on the host, before it dispatches the update, with
    the state, the prefetched batch and a key folded from the run key and the step,
    then reshards the result with `shard_batch`. A rollout may run compiled device
    programs of its own (Dreamer's imagined rollout and replay draw, PPO's critic);
    a simulator's or a service's calls and other host effects stay outside the
    update's `jit`. A batch of one signature reuses the update's program for that
    signature and plan, as any batch does.

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
    processes with a checkpointer is refused one as its fit starts
    (`_rollout_admitted`). The
    contract is the post-training spec's (docs/design/post-training.md, §14).

    `objective` is the one the step it feeds trains: the run's own, or, where its data's phases set the loop's
    pass count (`TrainerAdvanced(phase_passes=)`), the copy at the count of the phase the step's batch was
    read in (`PhasePasses.objective_for`), which a policy drawn for the step samples as. A rollout that acts
    with a policy of its own compiled once refuses one other than its own (`own_objective`).
    """

    def __call__[LossT, EffectsT](self, state: TrainState, batch: Batch, key: jax.Array, *,
                                  objective: Objective[LossT, EffectsT]) -> Batch: ...


def _rollout_metrics(rollout: Rollout) -> Mapping[str, float]:
    """The rollout's report of its latest call. A rollout is any callable,
    and a plain function reports nothing, so the trainer reads `metrics` at this boundary."""
    return getattr(rollout, 'metrics', {})


def rolled_out[LossT, EffectsT](rollout: Rollout | None, state: TrainState, batch: Batch, mesh: Mesh,
                                marked: Callable[[Batch], Batch], interval: Interval, run: FitRun, *,
                                objective: Objective[LossT, EffectsT] | None = None,
                                examples: int | None = None, closes: bool = False
                                ) -> tuple[Batch, bytes | None]:
    """`batch` sampled through `rollout` from `state`, as the run holds it, which its policy reads off
    the mesh (`decoder_stack.off_stages`), its host preparation `marked` and placed on `mesh`
    (`shard_batch`), its seconds counted in `interval`, `run`'s stream placing nothing meanwhile
    (`FitRun.held`); without a rollout, the batch as it is. The batch it makes carries the run's metadata of
    the read (`carried`): the data phase its step's batch was read in, and in a run on the examples clock the
    examples of the batches it read, its step's and each it drew further. `objective` is the one the step
    trains (`Rollout`), which a run with a rollout passes. Where it `closes` a step of the run
    bounded to `examples`, the batch is closed (`FitRun.closing`) after the rollout's last draw, so the
    look-ahead that finds the stream's end, and the position the step saves, come after every batch it read,
    in a pool each a pooled read (`FitRun.further`). With the batch, the position the step saves, or None.

    Run on the host before the update: sampling, scoring, advantages, each of which
    may dispatch compiled programs of its own. The key folds the step key once more,
    keeping the rollout's draws off the step's stream; both are checkpointed, so a
    resumed run samples forward.
    """
    from dew.distributed.batches import shard_batch

    if rollout is None:
        return run.closing(interval, batch, examples) if closes else (batch, None)
    assert objective is not None, "a rollout draws for the objective its step trains"
    began = time.perf_counter()
    key = jax.random.fold_in(jax.random.fold_in(state.key, state.step), 1)
    consumed = _Consumed(batch)
    further = {"further": consumed.reading(run.further(interval))} if _draws_further(rollout) else {}
    with run.held():
        policy, key, held = on_devices(state, key, mesh)
        # The copy's bytes on a device, each call's (`Interval.rollout_copied`): what a run that keeps its
        # variables on the host moves to sample, the step aside.
        interval.rollout_copied += max((copy for copy, _ in held.values()), default=0)
        with programs_admitted(mesh, held):
            made = carried(rollout(policy, _unread(batch), key, **further, objective=objective), batch,
                           consumed)
        prepared = marked(made)
        # The count the rollout's batch consumes is what it read, as `ConsumedCount`s count it.
        batch = shard_batch(mesh, prepared if CONSUMED_COUNT not in made else {
            **prepared, CONSUMED_COUNT: made[CONSUMED_COUNT]})
    interval.rollout_seconds += time.perf_counter() - began
    # Placed, as a step's read batch is closed: what the stream's end leaves goes to the pool's first row.
    return run.closing(interval, batch, examples) if closes else (batch, None)


GIB = 2 ** 30

type Held = Mapping[Hashable, tuple[int, int]]
"""Each device's bytes, by the device (a `jax.Device`, which JAX states no type of), of the run's weights a
rollout's policy reads there (`policy_bytes`): the copy of the host's, and the rest it holds."""


def on_devices(state: TrainState, key: jax.Array, mesh: Mesh
               ) -> tuple[TrainState, jax.Array, Held]:
    """`state` and its rollout's `key` as the rollout's policy reads them on `mesh`, the devices the run's
    step runs on: each leaf of its weights (its variables, its EMA's) and the key held on other
    devices (a host run's, `Layout(host=("variables",))`, on its CPU companion) copied there, placed as a
    device run places it (its partition spec, `execution.spec_on`), once a call; every leaf `mesh`'s devices
    hold already (a frozen tree beside the accelerator, a device run's whole state) as it is. With each
    device's bytes of the weights it holds in its own memory, the copy's and the rest's. A copy the room
    its devices' allocators leave beside the rest cannot hold is refused by name before it is made
    (`refuse_past`, the planner's admission of the copy beside them)."""
    from dew.distributed.host import transfer
    from dew.training.execution import spec_on

    devices = set(mesh.devices.flat)
    weights = (state.variables, state.ema)
    counted = policy_bytes(weights, jax.tree.map(lambda leaf: leaf.sharding, weights), mesh)
    refuse_past(counted)

    def placed(leaf):
        return leaf if leaf.sharding.device_set <= devices else transfer(leaf, spec_on(mesh)(leaf))

    variables, ema = jax.tree.map(placed, weights)
    return dataclasses.replace(state, variables=variables, ema=ema), placed(key), counted


def policy_bytes(weights, shardings, mesh: Mesh) -> Held:
    """Each of `mesh`'s devices' bytes, in its own memory, of `weights` (arrays or shapes, each placed by
    `shardings`' leaf) as a rollout's policy reads them there (`on_devices`): of the leaves held on other
    devices, copied onto it by their partition spec (`execution.spec_on`), and of the rest, as they sit: the
    one count the copy is admitted by and its run logs (`Interval.rollout_copied`)."""
    from dew.training.execution import spec_on

    devices = set(mesh.devices.flat)
    held = {device: [0, 0] for device in devices}
    for leaf, sharding in zip(jax.tree.leaves(weights), jax.tree.leaves(shardings), strict=True):
        copied = not sharding.device_set <= devices
        shaped = jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)
        target = spec_on(mesh)(shaped) if copied else sharding
        if target.memory_kind in (None, "device"):
            shard = int(np.prod(target.shard_shape(leaf.shape))) * jnp.dtype(leaf.dtype).itemsize
            for device in target.device_set & devices:
                held[device][0 if copied else 1] += shard
    return {device: (copy, rest) for device, (copy, rest) in held.items()}


def refuse_past(held: Held, executable: jax.stages.Compiled | None = None, called: str = "") -> None:
    """Refuse, by name, what a rollout's policy holds on each device of `held` (`on_devices`: the copy of
    the run's host-held weights it reads there beside the rest of the run's the device holds) past the
    room its allocator leaves (`dew._devices.device_memory`: a stated GPU's limit with those resident, an
    attached one's own in use), as the planner admits a step (`dew.training.memory.admitted`): the copy
    before it is made, beside the rest; or, where `executable`, the program `called` its policy runs there
    once the copy is made, its outputs and temporaries beside the weights (a generation's KV cache and
    activations), its temporaries in one free block. Nothing where no copy is made, the run's weights the
    devices' whole, or the device states no limit (a CPU's)."""
    from dew.telemetry.instrumentation import compiled_memory
    from dew.training.memory import admitted

    memory = None if executable is None else compiled_memory(executable)
    if executable is not None and memory is None:
        return
    for device, (copy, rest) in held.items():
        if not copy:
            continue
        if memory is None:
            headroom = admitted([device], rest, copy, 0, placed=False)
            what = ""
        else:
            headroom = admitted([device], rest + copy, memory.output - memory.alias, memory.temp)
            own = memory.output - memory.alias + memory.temp
            what = (f" and its program {called}'s {own / GIB:.2f} GiB of outputs and temporaries (a "
                    "generation's KV cache and activations)")
        if headroom is not None and headroom < 0:
            raise ValueError(
                f"a rollout's policy needs the {copy / GIB:.2f} GiB copy of the run's host-held weights on "
                f"{device}{what}, beside the {rest / GIB:.2f} GiB the device holds of the run (its frozen "
                f"trees), {-headroom / GIB:.2f} GiB past what its allocator holds: a run that keeps its "
                "variables on the host (Layout(host=('variables',))) samples from a copy of them on its "
                "devices")


@contextlib.contextmanager
def programs_admitted(mesh: Mesh, held: Held):
    """Within the block, a rollout's call, each program Dew's compile point runs on `mesh`'s devices
    (`options.running`: its policy's generation among them, compiled at the call or before) is admitted,
    once an executable, beside what each device holds of the run's weights (`held`, `on_devices`), refused
    by name before it runs where its devices cannot hold it (`refuse_past`)."""
    from dew.nn.kernels.options import running

    if not any(copy for copy, _ in held.values()):
        yield
        return
    devices = set(mesh.devices.flat)
    # Each executable admitted, held so its id names it alone.
    seen: dict[int, jax.stages.Compiled] = {}

    @contextlib.contextmanager
    def heard(program, executable, args, kwargs):
        if id(executable) not in seen:
            placed = {device for sharding in (*jax.tree.leaves(executable.input_shardings),
                                              *jax.tree.leaves(executable.output_shardings))
                      for device in sharding.device_set} & devices
            refuse_past({device: held[device] for device in placed}, executable, program.called)
            seen[id(executable)] = executable
        yield

    with running(heard):
        yield


READ = (EXAMPLES_CONSUMED, CONSUMED_COUNT, CLOSES_WINDOW, PHASE)
"""The fields a read batch carries of its read for the run, not for its rows: its progress on the examples
clock and the data phase it was read in, which the batch a rollout makes of it carries instead
(`carried`): none reaches the rollout."""


def _unread(batch: Batch) -> Batch:
    """`batch` as a rollout reads it: without the run's fields of its read (`READ`)."""
    return {name: value for name, value in batch.items() if name not in READ}


class _Consumed:
    """The examples the batches a rollout reads consume, added as each arrives, none of them kept: this
    process's rows' (`local_rows`, `EXAMPLES_CONSUMED`) and their counts (`ConsumedCount`); a run on
    another clock's batches carry none."""

    def __init__(self, first: Batch):
        self.clocked, self.rows, self.count = CONSUMED_COUNT in first, 0, 0
        self.add(first)

    def add(self, batch: Batch) -> None:
        if self.clocked:
            self.rows += int(np.sum(local_rows(batch[EXAMPLES_CONSUMED])))
            self.count += batch[CONSUMED_COUNT].examples

    def reading(self, read: Callable[[], Batch | None]) -> Callable[[], Batch | None]:
        """`read`, each batch it reads added and handed on without the run's fields of its read."""
        def drawn() -> Batch | None:
            batch = read()
            if batch is None:
                return None
            self.add(batch)
            return _unread(batch)
        return drawn


def carried(made: Batch, read: Batch, consumed: _Consumed) -> Batch:
    """The batch a rollout `made` of the batch it `read` and those it drew further, carrying the run's
    fields of the read (`READ`): the data phase `read` was read in, which the step's pass count reads
    (`dew.training.phase_passes`), and as a run on the examples clock steps it: consuming the examples they
    consumed (`consumed`), all on its first row, as a step sums them over its rows, counted as they counted
    them (`ConsumedCount`), and closing the update window where the read batch does (`CLOSES_WINDOW`), where
    the run closed it before the rollout; a batch with no real row on this process (a pool's share whose
    stream ran out, `dew.training._fit.pooled_read`) makes one with none. Each process counts the rows it
    holds, so the step's sum over the pool's rows counts each once. A run on another clock reads no
    count."""
    made = {**made, PHASE: read[PHASE]} if PHASE in read else made
    if not consumed.clocked:
        return made
    held = np.zeros(rows_of(made), np.int32)
    held[0] = consumed.rows
    clocked = {**made, EXAMPLES_CONSUMED: held, CONSUMED_COUNT: ConsumedCount(consumed.count)}
    if CLOSES_WINDOW in read:
        clocked[CLOSES_WINDOW] = read[CLOSES_WINDOW]
    if VALID_ROWS in read and not np.any(local_rows(read[VALID_ROWS])):
        clocked[VALID_ROWS] = np.zeros(len(held), bool)
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
