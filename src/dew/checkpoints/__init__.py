"""A run's checkpoints: the train state and the data position, through orbax.

A checkpoint holds `step`, `params`, `opt_state`, `ema`, `key` and, when the
data iterator can report one, `position`. Nothing else: metrics, the loss
scale and epoch counters are the loop's business and are rebuilt on resume.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from etils import epath
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation

STATE_LEAVES = ("step", "params", "opt_state", "ema", "key")


def is_uri(path: str) -> bool:
    """A `<scheme>://` location, such as a gs:// bucket, which has no local form.

    The package's one test for it, because the two callers need opposite
    things from the answer: orbax wants an absolute path for a local
    directory, and a tracker uploads a directory it can open while a bucket
    is referenced where it already lies.
    """
    return '://' in path


def location(directory: str) -> epath.Path:
    """Where a run's files go: a bucket URI as given, a local path absolute.

    `epath` reads and writes both, but `Path.resolve` turns `gs://bucket/run`
    into a local `gs:/bucket/run`, so the absolute step is for a path with no
    scheme.
    """
    path = epath.Path(directory)
    return path if is_uri(directory) else path.resolve()


def _processes(count: int) -> str:
    return f"{count} process" + ("es" if count != 1 else "")


def _loss(metrics):
    return metrics['loss']


def gather_positions(position: bytes) -> dict:
    """Every process's iterator position as one table, the checkpoint's `position`.

    'rows' is a uint8 [process_count, longest] array with one row per
    process, 'lengths' the unpadded length of each. A process holds the
    position of its own shard of the data, and orbax writes a host array from
    process 0 alone, so the rows are gathered onto every process before a
    save. They differ in length, which is why the lengths ride along.
    """
    lengths = multihost_utils.process_allgather(np.asarray(len(position), np.int64))
    row = np.zeros(int(lengths.max()), np.uint8)
    row[:len(position)] = np.frombuffer(position, np.uint8)
    return {'rows': multihost_utils.process_allgather(row), 'lengths': lengths}


def own_position(table: dict) -> bytes:
    """This process's row of a `gather_positions` table, without its padding."""
    index = jax.process_index()
    row = np.asarray(table['rows'][index], np.uint8)
    return row[:int(table['lengths'][index])].tobytes()


class Checkpoints:
    """The checkpoints of one run, in one directory.

    Constructing one opens nothing; the orbax manager is created on first use.
    The directory keeps the latest `keep` steps, so a resume has something
    recent, plus the one step with the lowest `loss` metric a save reported.
    A save without metrics can never become the best step.
    """

    def __init__(self, directory: str, *, keep: int = 2):
        self.directory = str(location(directory))
        self.keep = keep
        self._manager = None

    def _open(self) -> ocp.CheckpointManager:
        if self._manager is None:
            options = ocp.CheckpointManagerOptions(
                preservation_policy=preservation.AnyPreservationPolicy([
                    preservation.LatestN(n=self.keep),
                    preservation.BestN(get_metric_fn=_loss, n=1,
                                       keep_checkpoints_without_metrics=False,
                                       reverse=True),
                ]),
                best_fn=_loss, best_mode='min',
                create=True, enable_async_checkpointing=True)
            self._manager = ocp.CheckpointManager(
                self.directory, options=options,
                item_handlers=ocp.PyTreeCheckpointHandler())
        return self._manager

    @property
    def latest(self) -> int | None:
        return self._open().latest_step()

    @property
    def best(self) -> int | None:
        """The step with the lowest reported loss, or None when no save carried one."""
        return self._open().best_step()

    def path(self, step: int) -> str:
        return str(epath.Path(self.directory) / str(step))

    def save(self, step: int, state, position: bytes | None,
             metrics: Mapping[str, float] | None = None) -> None:
        """Write `state` under `step`, asynchronously.

        Sharded arrays go straight to orbax: gathering them onto the host
        first would serialise the whole state through one process and undo
        the point of an async checkpointer. Grain reports its position as
        JSON bytes, which tensorstore has no dtype for; the raw bytes ride
        along as uint8 rows instead, one per process, so each resumes its own
        shard where it left it. A write that fails surfaces from `wait`, which
        is deliberately unguarded: a checkpoint that did not land is data loss.
        """
        item = {name: getattr(state, name) for name in STATE_LEAVES}
        if position is not None:
            item['position'] = gather_positions(position)
        self._open().save(step, args=ocp.args.PyTreeSave(item), metrics=metrics, force=True)

    def restore(self, template=None, step: int | None = None) -> tuple[Any, bytes | None]:
        """The state at `step` (the latest by default) and this process's data position.

        `template` is a pytree of `jax.ShapeDtypeStruct` naming the state
        leaves to restore; a leaf's sharding, when set, is where the array is
        placed, so a checkpoint written on one mesh restores onto whatever
        mesh this run is using. Restoring untyped would silently discard
        opt_state and reset the step counter (and with it the lr schedule) on
        every resume. `None` restores every leaf as a host array.
        """
        manager = self._open()
        if step is None:
            step = manager.latest_step()
            if step is None:
                raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        metadata = manager.item_metadata(step)
        stored = metadata.keys()
        if template is None:
            # Typed as host arrays, so orbax reads no sharding file and warns
            # about none.
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), dict(metadata))
            restored = manager.restore(step, args=ocp.args.PyTreeRestore(
                restore_args=restore_args))
        else:
            item = {name: getattr(template, name) for name in STATE_LEAVES} \
                if not isinstance(template, Mapping) else dict(template)
            restore_args = jax.tree.map(
                lambda leaf: ocp.ArrayRestoreArgs(sharding=getattr(leaf, 'sharding', None)),
                item)
            if 'position' in stored:
                # The table's shape depends on the process count and the
                # iterator's position, so orbax takes it from the checkpoint
                # rather than from this placeholder.
                item['position'] = {'rows': np.zeros((1, 1), np.uint8),
                                    'lengths': np.zeros((1,), np.int64)}
                restore_args['position'] = jax.tree.map(
                    lambda _: ocp.RestoreArgs(), item['position'])
            try:
                # partial_restore: a key the checkpoint holds and the template
                # does not is skipped instead of refused.
                restored = manager.restore(step, args=ocp.args.PyTreeRestore(
                    item=item, restore_args=restore_args, partial_restore=True))
            except (TypeError, ValueError) as mismatch:
                # A structural mismatch surfaces from inside orbax's tree walk
                # as a key path and a pair of container types, which says
                # nothing about what to do. opt_state is shaped by the
                # optimizer and by the MultiSteps wrapper gradient
                # accumulation puts around it, so changing either between
                # runs is what usually lands here.
                raise ValueError(
                    f"The checkpoint at {self.path(step)} does not fit this run's "
                    f"train state ({mismatch}). A checkpoint carries the optimizer "
                    f"state (opt_state), so a resume needs the model, the optimizer "
                    f"and the gradient accumulation it was written with. Resume it "
                    f"with those, or start a fresh run in a directory of its own."
                ) from mismatch
        restored = dict(restored)
        table = restored.pop('position', None)
        if table is None:
            position = None
        else:
            written = len(table['lengths'])
            if written != jax.process_count():
                raise ValueError(
                    f"The checkpoint at {self.path(step)} holds a data iterator "
                    f"position for each of {_processes(written)} and this run has "
                    f"{_processes(jax.process_count())}. A position is where one "
                    f"process's shard of the data stopped and cannot be translated "
                    f"to another shard count, so resume it on {_processes(written)}; "
                    f"only a checkpoint that holds no data position resumes on any "
                    f"count.")
            position = own_position(table)
        if template is not None and not isinstance(template, Mapping):
            restored = template.replace(**restored)
        return restored, position

    def wait(self) -> None:
        """Block until pending async writes have landed on disk.

        Saving is async so it stays off the training loop's critical path;
        anything that reads a checkpoint back has to call this first.
        """
        if self._manager is not None:
            self._manager.wait_until_finished()

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None
