"""A run's writes off its step loop (`dew.training.trainer.Trainer`): its checkpoints, each with the envelope
every tier saves and its report, and its logging intervals, read on the host and reported on the run's
writer, and the run's outcome reported as its fit ends."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax

from dew._coordination import agree_process_phase, agreed
from dew.checkpoints import Checkpoints, Envelope, Ranking, _Laid
from dew.data.dataset import DataPartition
from dew.state import TrainState
from dew.telemetry.profile import region
from dew.telemetry.records import CheckpointRequested, FitEnded
from dew.training._fit import FitRun, _rollout_saved
from dew.training._lowering import Lowered
from dew.training.display import goodput
from dew.training.interval import Interval, Reading, Taken, logged_scalars
from dew.training.profiling import reported_window

if TYPE_CHECKING:
    from dew.training.trainer import Trainer

_input_log = logging.getLogger("dew.training.input")
"""Where a run says what its input costs its steps, once (`written`)."""


def saved_checkpoint(
    trainer: Trainer,
    checkpoints: Checkpoints,
    step: int,
    state: TrainState,
    position: bytes | None,
    interval: Interval,
    *,
    scores: Mapping[str, float] | None = None,
    ranking: Sequence[Ranking] = (),
    training_best: bool = True,
    run: FitRun,
    weights_only: bool = False,
) -> float:
    """Write one checkpoint with the interval's mean loss and `run`'s stop control, its stream
    placing nothing meanwhile (`FitRun.held`), report it and start the next interval.

    Returns the seconds it took, the reading of the interval's loss included: that read waits
    on the device, and the wait is time the steps did not have."""
    paused = time.perf_counter()
    trainer._display.status("writing a checkpoint")
    metadata = dict(scores or {})
    if interval.steps and interval.book is not None:
        metadata.setdefault('train/loss', float(jax.device_get(interval.book.loss) / interval.steps))
        if not ranking and training_best:
            ranking = (Ranking('train/loss', metadata['train/loss']),)
    envelope = checkpoint_envelope(trainer, state, position, interval, run, weights_only=weights_only)
    # The run holds the tokenizer its declaration states the files of (`Checkpoints.save`).
    write = functools.partial(checkpoints.save, step, metrics=metadata, ranking=ranking,
                              weights_only=weights_only, tokenizer=trainer.objective.run_tokenizer(),
                              **envelope)
    with run.outside_window(), run.held():
        trainer._report(CheckpointRequested(checkpoints.directory), step, write)
    interval.saved(step, resumable=not weights_only)
    trainer._display.status("")
    return time.perf_counter() - paused


def checkpoint_envelope(trainer: Trainer, state: TrainState, position: bytes | None, interval: Interval,
                        run: FitRun, *, weights_only: bool = False) -> Envelope:
    """What a checkpoint of `state` holds on every tier, each tier's save adding its own metadata: one
    envelope a persistent and a local save submit (row 4126)."""
    control, share = {**run.stop_control, **interval.token_control}, DataPartition.of(trainer.device_mesh)
    # The rung beside the layout's host fields, which a resume lays out alike (`resumed_layout`).
    laid = _Laid(rung={**trainer._compiler.checkpoint_rung(), "host": list(trainer.layout.host)},
                 training=trainer.phase_passes.training_record(trainer.objective),
                 boxed=trainer._boxed_axes(), held_apart=trainer._staging.apart)
    return {"state": trainer._staging.saved(state), "saved": position, "share": share, "control": control,
            "artifact": trainer._artifact(),
            "rollout": None if weights_only else _rollout_saved(trainer.rollout), "_laid": laid}


def saved_local_checkpoint(trainer: Trainer, checkpoints: Checkpoints, step: int, state: TrainState,
                           position: bytes | None, interval: Interval, run: FitRun) -> float:
    """Write one checkpoint to the local directory with `run`'s stop control, its stream placing
    nothing meanwhile (`FitRun.held`), and report it.

    Returns the seconds it took. The local copy carries no metadata; it is the one a restarted
    node reads back, not the run's record."""
    paused = time.perf_counter()
    trainer._display.status("writing a local checkpoint")
    write = functools.partial(checkpoints.save_local, step,
                              **checkpoint_envelope(trainer, state, position, interval, run))
    with run.outside_window(), run.held():
        trainer._report(CheckpointRequested(str(checkpoints.local_directory), local=True), step, write)
    trainer._display.status("")
    return time.perf_counter() - paused


def read_taken(trainer: Trainer, run: FitRun, *, agree: bool = True, wait: bool = True) -> None:
    """Hand the log the loop took (`Taken`) to the run's writer, which reads it on the host and reports
    one logging interval off the step loop (`LogWriter`, `_written`), and raise a write's failure: of
    every write, waiting for each, where `wait`, as before a compile, an evaluation, a save or the end,
    which read the run's state anyway; else of those done, as each logging tick does, which never waits.

    The failures are agreed here, on the loop, at a point every process reaches alike, so a tracker that
    failed on rank zero, or a losses' streak, stops its peers here rather than at their next collective;
    a failed run's cleanup raises without agreeing, waiting only on writes that wait on no device, since
    its peers may never agree and its steps never end."""
    taken, run.taken = run.taken, None
    if taken is not None:
        run.writer.submit(functools.partial(read_log, trainer, run, taken),
                          functools.partial(written, trainer, run, taken), taken.ready)
    elif not (wait and run.writer.unchecked):
        return
    check = functools.partial(run.writer.raise_failure, wait=wait, ready_only=not agree)
    with region("log"):
        if agree:
            agreed("training reporting", check)
        else:
            check()


def read_log(trainer: Trainer, run: FitRun, taken: Taken) -> tuple[Reading, float]:
    """`taken` on the host, its losses checked, and its interval's seconds, on the writer's reader
    thread (`LogWriter`), whose clock it moves."""
    read, elapsed, run.writer.since = taken.read(trainer._display, run.writer.since)
    return read, elapsed


def written(trainer: Trainer, run: FitRun, taken: Taken, reading: tuple[Reading, float]) -> None:
    """Report `taken`'s interval's scalars (`logged_scalars`) on rank zero, on the writer's reporter
    thread (`_read_taken`); every other rank has nothing to report."""
    if jax.process_index() != 0:
        return
    read, elapsed = reading
    scalars = {**logged_scalars(read, taken.counts, elapsed, trainer.device_mesh.devices.size,
                                rollout=trainer.rollout is not None, update_flops=run.update_flops),
               **run.startup.scalars()}
    if run.unworked and elapsed > 0 and taken.counts.waited > elapsed / 2:
        # Said once: a per-share stream read in the process starves the step.
        run.unworked = False
        _input_log.warning("the step waits on its input for more than half its time, read by a per-share "
                           "stream in the training process: state its loading workers (.workers(n)), or "
                           "read it .elastic(), whose order any count of workers keeps")
    trainer._display.interval(taken.step, scalars)
    if trainer.tracker is not None:
        trainer.tracker.log(scalars, taken.step)


def reported_outcome(trainer: Trainer, error: BaseException | None, run: FitRun) -> BaseException | None:
    """Agree the run's cleanup, report its goodput and its outcome.

    Every rank reaches the cleanup agreement, so a rank that failed alone
    is heard here. Goodput is only reported for a run that reached this
    point without an error, since its numbers describe a run that ran.
    """
    try:
        agree_process_phase(error, phase="fit cleanup")
    except BaseException as failure:
        if error is None:
            error = failure
    error = reported_window(trainer._report, trainer._display.note, run.profiled, error)
    if error is None:
        def report_goodput() -> None:
            if jax.process_index() == 0:
                wall = time.perf_counter() - run.started
                # A run that logged no step reports its start here (`Startup.scalars`).
                scalars = {**goodput(wall, run.first_step, run.other),
                           **run.startup.scalars()}
                trainer._display.summary(run.current, wall, scalars,
                                      None if run.loss is None else float(jax.device_get(run.loss)))
                if trainer.tracker is not None:
                    trainer.tracker.log(scalars, run.current)

        try:
            agreed("goodput reporting", report_goodput)
        except BaseException as failure:
            error = failure
    try:
        outcome = FitEnded.outcome(time.perf_counter() - run.started, error,
                                   preempted=run.preempted is not None,
                                   lowered=isinstance(error, Lowered))
        trainer._report(outcome, run.current)
    except BaseException as failure:
        if error is None:
            error = failure
        else:
            error.add_note(f'Reporting fit outcome failed: {failure!r}')
    return error
