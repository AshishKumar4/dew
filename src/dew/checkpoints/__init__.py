"""A run's checkpoints: the train state and the data position, through orbax.

A checkpoint holds `step`, `params`, `opt_state`, `ema`, `key` and, when the
data iterator can report one, `position`. Metrics, the loss scale and epoch
counters are the loop's business and are rebuilt on resume. A position is
either global, and readable by any process count, or one process's own shard
offset, and readable only by the count that wrote it; `dew.position` is the
difference and `read_position` acts on it.

Beside the persistent directory a run may keep a local checkpoint on every
host, written more often, so a preempted pod resumes from its own disks
instead of from storage. That is orbax's emergency checkpointing
(orbax.checkpoint.experimental.emergency), whose local checkpoints are built
here from the same pieces it uses, an `ArrayHandler` with no primary host
and no replica dedup so every process writes every shard it holds. Its
manager itself is not used: it takes over the persistent directory with a
single fixed-shape state item (no metrics, so no best step, and no room for
the per-process data position), and it restores a local checkpoint only
where each data-parallel replica is a whole set of hosts, which a mesh whose
fsdp axis spans hosts is not. The semantics are its: two directories, and
the newest checkpoint every process can read wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from etils import epath
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils
from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation

from dew import position

STATE_LEAVES = ("step", "microstep", "updates", "params", "opt_state", "ema", "key",
                "scale", "window_size", "accumulation")
"""The train-state fields a checkpoint persists; anything outside this tuple
is rebuilt on resume."""

RUN_FILE = "run.json"
"""The run record `RunConfig.save` writes into the run directory, beside the
step directories, and `dew.io.publish` ships with a step."""


def is_uri(path: str) -> bool:
    """Whether a path names a `<scheme>://` location, such as a gs:// bucket."""
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


def gather_positions(saved: bytes) -> dict:
    """Every process's iterator position as one table, the checkpoint's `position`.

    'rows' is a uint8 [process_count, longest] array with one row per
    process, 'lengths' the unpadded length of each. A process reports the
    position it holds, and orbax writes a host array from process 0 alone,
    so the rows are gathered onto every process before a save. The rows
    differ in length, so the lengths ride along.

    One row per process whichever kind the position is: a global one is the
    same bytes on every process, and gathering it is what lets `read_position`
    check that they really do agree before another process count reads it.
    """
    lengths = multihost_utils.process_allgather(np.asarray(len(saved), np.int64))
    row = np.zeros(int(lengths.max()), np.uint8)
    row[:len(saved)] = np.frombuffer(saved, np.uint8)
    return {'rows': multihost_utils.process_allgather(row), 'lengths': lengths}


def _row(table: dict, index: int) -> bytes:
    """One row of a `gather_positions` table, without its padding."""
    row = np.asarray(table['rows'][index], np.uint8)
    return row[:int(table['lengths'][index])].tobytes()


def read_position(table: dict, where: str) -> bytes:
    """The position this run resumes from, out of a saved table.

    A row is this process's own when the counts match. When they differ the
    saved position has to be a global one, a record count over an order that
    is the same order at any process count, and then every row is this run's
    position: `dew.position` is where that promise is written down and
    `read_position` is where it is taken up. A shard offset has no row a
    different count can take over, so it is refused with both counts named.
    """
    written = len(table['lengths'])
    if written == jax.process_count():
        return _row(table, jax.process_index())
    # Row order, not set order: bytes hash differently per interpreter, and
    # which of the two refusals below a broken checkpoint gets is a diagnostic.
    rows = [_row(table, index) for index in range(written)]
    saved = rows[0]
    if not position.translates(saved):
        raise ValueError(
            f"The checkpoint at {where} holds a data iterator "
            f"position for each of {_processes(written)} and this run has "
            f"{_processes(jax.process_count())}. A position is where one "
            f"process's shard of the data stopped and cannot be translated "
            f"to another shard count, so resume it on {_processes(written)}; "
            f"a stream that reads its records globally, as every `train_stream` "
            f"dataset does, saves a position that resumes on any count.")
    if any(row != saved for row in rows[1:]):
        raise ValueError(
            f"The checkpoint at {where} holds a global data position that "
            f"differs between the {_processes(written)} that wrote it, and this "
            f"run has {_processes(jax.process_count())} to hand it to. A global "
            f"position is one place in one order, so those processes read "
            f"different orders and no single one of their positions is this "
            f"run's.")
    return saved


def placement(tree: Any) -> dict[str, str]:
    """Where each array leaf of `tree` sits, by path, as the string of its
    sharding; a local checkpoint restores onto this placement and no other."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): str(leaf.sharding)
            for path, leaf in leaves
            if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) and leaf.sharding is not None}


class Checkpoints:
    """The checkpoints of one run, in one directory.

    Constructing one opens nothing; the orbax managers are created on first
    use. The directory keeps the latest `keep` steps, so a resume has
    something recent, plus the step with the lowest `loss` metric a save
    reported. A save without metrics can never become the best step.

    `local_directory` names a path on every host's own disk where the run
    keeps one more checkpoint, the latest, written every `local_every` steps
    by `fit`; each process writes the shards its devices hold under a
    directory of its own, so the same path serves a pod and a single host
    running several processes. `latest` is the newest step every process can
    read, local or persistent, and `restore` reads it from wherever it is. A
    local checkpoint restores onto the placement it was written with, since
    no process holds another process's shards; the persistent checkpoint
    restores onto any mesh.
    """

    def __init__(self, directory: str, *, keep: int = 2,
                 local_directory: str | None = None, local_every: int | None = None):
        if (local_directory is None) != (local_every is None):
            raise ValueError(
                "local checkpoints take both local_directory and local_every: where "
                "every host writes its copy, and every how many steps")
        if local_every is not None and local_every < 1:
            raise ValueError(f"local_every must be at least 1, got {local_every}")
        self.directory = str(location(directory))
        self.keep = keep
        self.local_directory = None if local_directory is None else str(location(local_directory))
        self.local_every = local_every
        self._manager = None
        self._local_manager = None

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
    def local_path(self) -> str:
        """This process's own local directory."""
        if self.local_directory is None:
            raise ValueError("this run keeps no local checkpoints")
        return str(epath.Path(self.local_directory) / f"process{jax.process_index()}")

    def _open_local(self) -> ocp.CheckpointManager:
        if self._local_manager is None:
            # No primary host: every process writes its own metadata, and
            # every shard it holds, one copy per process, so each process's
            # directory is complete for its devices.
            # Only device arrays are registered, because orbax writes a host
            # array from process 0 alone whatever the options say; the
            # position table rides as a replicated device array instead. The
            # prefix keeps this manager's barriers apart from the persistent
            # one's, whose keys are otherwise the same at a step both write.
            multiprocessing = MultiprocessingOptions(primary_host=None,
                                                     barrier_sync_key_prefix='local')
            registry = ocp.type_handlers.create_type_handler_registry(
                (jax.Array, ocp.type_handlers.ArrayHandler(
                    primary_host=None, replica_id=None, use_replica_parallel=False)))
            options = ocp.CheckpointManagerOptions(
                max_to_keep=1, create=True, cleanup_tmp_directories=True,
                enable_async_checkpointing=True, multiprocessing_options=multiprocessing)
            self._local_manager = ocp.CheckpointManager(
                self.local_path, options=options,
                item_handlers=ocp.PyTreeCheckpointHandler(
                    use_ocdbt=True, use_zarr3=True, multiprocessing_options=multiprocessing,
                    type_handler_registry=registry))
        return self._local_manager

    def _local_latest(self) -> int | None:
        """The local step every process holds, or None.

        Each process keeps one local step, its newest; a resume can read a
        local step only if every process has it, so the processes agree
        here, and a host that lost its copy or was killed before its write
        landed sends the whole pool to the persistent checkpoint.
        """
        if self.local_directory is None:
            return None
        mine = self._open_local().latest_step()
        steps = multihost_utils.process_allgather(
            np.asarray(-1 if mine is None else mine, np.int64))
        held = int(steps[0])
        return held if held >= 0 and bool(np.all(steps == held)) else None

    @property
    def latest(self) -> int | None:
        """The newest step a resume can read, local or persistent."""
        persistent = self._open().latest_step()
        local = self._local_latest()
        if persistent is None or local is None:
            return local if persistent is None else persistent
        return max(persistent, local)

    @property
    def best(self) -> int | None:
        """The step with the lowest reported loss, or None when no save carried one."""
        return self._open().best_step()

    def path(self, step: int) -> str:
        return str(epath.Path(self.directory) / str(step))

    def source(self, step: int) -> str:
        """The directory `restore` reads `step` from: this process's local one
        when the step is the local one every process holds, else the
        persistent one."""
        return self.local_path if step == self._local_latest() else self.directory

    def save(self, step: int, state: Any, saved: bytes | None,
             metrics: Mapping[str, float] | None = None) -> None:
        """Write `state` under `step`, asynchronously.

        Sharded arrays go straight to orbax: gathering them onto the host
        first would serialise the whole state through one process and undo
        the point of an async checkpointer. A stream reports its position as
        JSON bytes, which tensorstore has no dtype for; the raw bytes ride
        along as uint8 rows instead, one per process, so a global position
        and a shard offset are stored the same way and told apart on restore.
        A write that fails surfaces from `wait`, which is deliberately
        unguarded: a checkpoint that did not land is data loss.
        """
        self._open().save(step, args=ocp.args.PyTreeSave(self._item(state, saved)),
                          metrics=metrics, force=True)

    def save_local(self, step: int, state: Any, saved: bytes | None) -> None:
        """Write `state` under `step` to this process's local directory,
        asynchronously, in place of the local step before it. The placement
        rides along; a resume onto another one raises before reading shards
        from directories that do not hold them."""
        item = self._item(state, saved)
        written = placement(item)
        if saved is not None:
            item['position'] = jax.tree.map(
                lambda leaf: jax.device_put(leaf, state.step.sharding), item['position'])
        self._open_local().save(
            step, args=ocp.args.PyTreeSave(item), force=True,
            custom_metadata={'processes': jax.process_count(), 'placement': written})

    @staticmethod
    def _item(state: Any, saved: bytes | None) -> dict[str, Any]:
        item = {name: getattr(state, name) for name in STATE_LEAVES}
        if saved is not None:
            item['position'] = gather_positions(saved)
        return item
    def stored(self, step: int | None = None) -> dict[str, Any]:
        """What the checkpoint at `step` (the latest by default) holds, as
        shape/dtype trees per state field; an unset field is None."""
        if step is None:
            step = self.latest
            if step is None:
                raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        manager = self._open_local() if step == self._local_latest() else self._open()
        metadata = manager.item_metadata(step)
        return {name: None if value is None else
                jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), value)
                for name, value in dict(metadata).items()}

    def accumulation_template(self, step: int):
        """The persisted pending-array shapes, without reading their values."""
        from dew.training.state import Accumulation
        manager = self._open_local() if step == self._local_latest() else self._open()
        metadata = manager.item_metadata(step)
        missing = set(STATE_LEAVES).difference(metadata.keys())
        if missing:
            raise ValueError(f"training checkpoint lacks required state fields {sorted(missing)}")
        pending = metadata["accumulation"]
        if pending is None:
            return None
        arrays = jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), dict(pending))
        arrays["statistics"] = tuple(arrays["statistics"])
        arrays["effects"] = tuple(arrays["effects"])
        return Accumulation(**arrays)

    def restore(self, template=None, step: int | None = None) -> tuple[Any, bytes | None]:
        """The state at `step` (the latest by default) and this process's data position.

        `template` is a pytree of `jax.ShapeDtypeStruct` naming the state
        leaves to restore; a leaf's sharding, when set, is where the array is
        placed, so a checkpoint written on one mesh restores onto whatever
        mesh this run is using. `None` restores every leaf as a host array.

        A step that is the local one every process holds is read from the
        local directory, onto the placement it was written with; any other
        step from the persistent one. The data position comes back as the
        bytes this run's stream resumes from: its own row when the process
        count is the one that wrote it, and any row of a global position when
        it is not.
        """
        local = self._local_latest()
        if step is None:
            step = self.latest
            if step is None:
                raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        from_local = step == local
        manager = self._open_local() if from_local else self._open()
        where = self.local_path if from_local else self.path(step)
        if from_local and template is None and jax.process_count() > 1:
            raise ValueError(
                f"The local checkpoint at {where} holds each process's own shards, "
                f"so a pool cannot read step {step} as host arrays; restore it with "
                f"the template of a run placed as it was written, or read the "
                f"persistent checkpoint at {self.path(step)}")
        metadata = manager.item_metadata(step)
        stored = metadata.keys()
        if template is not None and not isinstance(template, Mapping):
            missing = set(STATE_LEAVES).difference(stored)
            if missing:
                raise ValueError(f"training checkpoint lacks required state fields {sorted(missing)}")
            if (metadata["scale"] is None) != (template.scale is None):
                raise ValueError("checkpoint dynamic-scaler configuration differs from this run")
            if (metadata["ema"] is None) != (template.ema is None):
                raise ValueError("checkpoint EMA configuration differs from this run")
        if template is None:
            # Typed as host arrays, so orbax reads no sharding file and warns
            # about none. A local checkpoint knows device arrays only, so its
            # leaves land on one device and come home from there.
            untyped = (
                ocp.ArrayRestoreArgs(sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
                if from_local else ocp.ArrayRestoreArgs(restore_type=np.ndarray))
            restored = manager.restore(step, args=ocp.args.PyTreeRestore(
                restore_args=jax.tree.map(lambda _: untyped, dict(metadata))))
            if from_local:
                restored = jax.tree.map(np.asarray, restored)
        else:
            item = {name: getattr(template, name) for name in STATE_LEAVES} \
                if not isinstance(template, Mapping) else dict(template)
            if from_local:
                self._check_placement(step, item)
            restore_args = jax.tree.map(
                lambda leaf: ocp.ArrayRestoreArgs(
                    sharding=leaf.sharding if isinstance(leaf, jax.ShapeDtypeStruct) else None),
                item)
            if 'position' in stored and not isinstance(template, Mapping):
                # A mapping names the leaves it wants and nothing else; a
                # resume takes the whole state, the data position included.
                # The table's shape depends on the process count and the
                # iterator's position, so it comes from the checkpoint's own
                # metadata, not from the template. A local checkpoint
                # holds it as a device array, replicated like the step.
                target = next(iter(jax.tree.leaves(item)), None)
                target_sharding = getattr(target, "sharding", None)
                position_sharding = (
                    jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec())
                    if isinstance(target_sharding, jax.sharding.NamedSharding) else
                    jax.sharding.SingleDeviceSharding(jax.local_devices()[0]))
                item["position"] = jax.tree.map(
                    lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype),
                    dict(metadata['position']))
                restore_args['position'] = jax.tree.map(
                    lambda leaf: (ocp.ArrayRestoreArgs(sharding=position_sharding,
                                                       global_shape=leaf.shape)
                                  if from_local else ocp.RestoreArgs()),
                    item['position'])
            try:
                # partial_restore: a key the checkpoint holds and the template
                # does not is skipped instead of refused.
                restored = manager.restore(step, args=ocp.args.PyTreeRestore(
                    item=item, restore_args=restore_args, partial_restore=True))
            except (TypeError, ValueError) as mismatch:
                # Model, optimizer and retained record shapes are a resume contract.
                raise ValueError(
                    f"The checkpoint at {where} does not fit this run's "
                    f"train state ({mismatch}). A checkpoint carries the optimizer "
                    f"state (opt_state), so a resume needs the model, the optimizer "
                    f"and the gradient accumulation it was written with. Resume it "
                    f"with those, or start a fresh run in a directory of its own."
                ) from mismatch
        restored = dict(restored)
        table = restored.pop('position', None)
        saved = None if table is None else read_position(table, where)
        if template is not None and not isinstance(template, Mapping):
            if int(restored["step"]) != step:
                raise ValueError("checkpoint directory and serialized attempted step disagree")
            if not isinstance(template.window_size, jax.ShapeDtypeStruct):
                if int(restored["window_size"]) != int(template.window_size):
                    raise ValueError("checkpoint accumulation window_size differs from this run")
            restored = template.replace(**restored)
        return restored, saved

    def _check_placement(self, step: int, item: Mapping[str, Any]) -> None:
        """Refuse a local step written for another placement of the state."""
        written = self._open_local().metadata(step).custom_metadata or {}
        wanted = placement(item)
        moved = [path for path in wanted if written.get('placement', {}).get(path) != wanted[path]]
        processes = written.get('processes')
        if processes == jax.process_count() and not moved:
            return
        if processes != jax.process_count():
            difference = (f"written by {_processes(processes or 0)}, and this run has "
                          f"{_processes(jax.process_count())}")
        else:
            difference = (f"written with {moved[0]} placed as "
                          f"{written.get('placement', {}).get(moved[0])}, and this run "
                          f"places it as {wanted[moved[0]]}")
        raise ValueError(
            f"The local checkpoint at {self.local_path} holds step {step} {difference}; "
            f"it holds each process's own shards, so it restores onto the mesh, "
            f"layout and process count it was written with. Resume with those, or "
            f"delete {self.local_directory} to resume from the persistent checkpoint "
            f"at step {self._open().latest_step()} in {self.directory}.")

    def wait(self) -> None:
        """Block until pending async writes have landed on disk.

        Saving is async so it stays off the training loop's critical path;
        anything that reads a checkpoint back has to call this first.
        """
        error = None
        for manager in (self._manager, self._local_manager):
            if manager is not None:
                try:
                    manager.wait_until_finished()
                except BaseException as failure:
                    if error is None:
                        error = failure
                    else:
                        error.add_note(f"Checkpoint wait also failed: {failure!r}")
        if error is not None:
            raise error
