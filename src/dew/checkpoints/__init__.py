"""Save and restore a run's train state and data position through orbax.

A checkpoint holds `step`, `params`, `opt_state`, `ema`, `key` and, when the
data iterator can report one, `position`. Metrics, the loss scale and epoch
counters are the loop's business and are rebuilt on resume. A position is
either global, and readable by any partition of the data, or one share's
own offset, and readable only by a reader of that share; `dew.position` is
the difference and `read_position` acts on it.

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
from typing import TYPE_CHECKING, overload

import jax
import numpy as np
import orbax.checkpoint as ocp
from etils import epath
from jax.experimental import multihost_utils
from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation

from dew import position
from dew.objectives.base import Variables
from dew.telemetry.profile import region

if TYPE_CHECKING:
    import optax
    from flax.training.dynamic_scale import DynamicScale

    from dew.data.dataset import DataPartition
    from dew.training.state import Accumulation, TrainState

# One field of the train state as a checkpoint holds it: a scalar array, a
# tree of arrays, or one of the records the state carries beside them. The
# gathered data position rides along under its own name as uint8 rows.
type StateLeaf = (jax.Array | np.ndarray | Variables | optax.OptState
                  | DynamicScale | Accumulation | None)

STATE_LEAVES = ("step", "microstep", "updates", "params", "opt_state", "ema", "key",
                "scale", "window_size", "accumulation")
"""The train-state fields a checkpoint persists; anything outside this tuple
is rebuilt on resume."""

RUN_FILE = "run.json"
"""The run record `RunConfig.save` writes into the run directory, beside the
step directories, and `dew.io.publish` ships with a step."""


def is_uri(path: str) -> bool:
    """Return whether a path names a `<scheme>://` location, such as a gs:// bucket."""
    return '://' in path


def location(directory: str) -> epath.Path:
    """Return where a run's files go: a bucket URI as given, a local path absolute.

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


def gather_positions(saved: bytes, share: DataPartition) -> dict:
    """Gather every process's iterator position into the checkpoint's `position` table.

    'rows' is a uint8 [process_count, longest] array with one row per
    process, 'lengths' the unpadded length of each, and 'shares' the
    `[index, count]` of the data share each process read. A process reports
    the position it holds, and orbax writes a host array from process 0
    alone, so the rows are gathered onto every process before a save. The
    rows differ in length, so the lengths ride along.

    One row per process whichever kind the position is: a global one is the
    same bytes on every process, and gathering it is what lets `read_position`
    check that they really do agree before another partition reads it.
    """
    lengths = multihost_utils.process_allgather(np.asarray(len(saved), np.int64))
    row = np.zeros(int(lengths.max()), np.uint8)
    row[:len(saved)] = np.frombuffer(saved, np.uint8)
    return {'rows': multihost_utils.process_allgather(row), 'lengths': lengths,
            'shares': multihost_utils.process_allgather(np.asarray([share.index, share.count], np.int64))}


def _row(table: dict, index: int) -> bytes:
    """Return one row of a `gather_positions` table, without its padding."""
    row = np.asarray(table['rows'][index], np.uint8)
    return row[:int(table['lengths'][index])].tobytes()


def read_position(table: dict, where: str, share: DataPartition) -> bytes:
    """Read the position the reader of `share` resumes from, out of a saved table.

    A global position is a record count over an order that is the same order
    at any partition, so every row is every reader's position:
    `dew.position` is where that promise is written down and `read_position`
    is where it is taken up. A share's own offset resumes only the readers of
    that same share, whichever processes they are, and anything else is
    refused with the shares named. A table written before shares were
    recorded held one per process, each process reading its own.
    """
    written = len(table['lengths'])
    # Row order, not set order: bytes hash differently per interpreter, and
    # which of the refusals below a broken checkpoint gets is a diagnostic.
    rows = [_row(table, index) for index in range(written)]
    held = ([(int(index), int(count)) for index, count in np.asarray(table['shares'])]
            if 'shares' in table else [(process, written) for process in range(written)])
    if position.translates(rows[0]):
        if any(row != rows[0] for row in rows[1:]):
            raise ValueError(
                f"The checkpoint at {where} holds a global data position that "
                f"differs between the {_processes(written)} that wrote it. A global "
                f"position is one place in one order, so those processes read "
                f"different orders and no single one of their positions is this "
                f"run's.")
        return rows[0]
    mine = [row for row, written_share in zip(rows, held, strict=True)
            if written_share == (share.index, share.count)]
    if not mine:
        raise ValueError(
            f"The checkpoint at {where} holds data positions for the shares "
            f"{sorted(set(held))} (index, count), and this reader reads share "
            f"{share.index} of {share.count}. A position is where one share of "
            f"the data stopped and cannot be translated to another share, so "
            f"resume it on a mesh whose processes read those shares; a stream "
            f"that reads its records globally, as every `train_stream` dataset "
            f"does, saves a position that resumes on any partition.")
    if any(row != mine[0] for row in mine[1:]):
        raise ValueError(
            f"The checkpoint at {where} holds positions for share {share.index} "
            f"of {share.count} that differ between the processes that read it; "
            f"a share's readers read the same records, so no one of them is "
            f"where the share stopped.")
    return mine[0]


def placement(tree: Mapping[str, StateLeaf]) -> dict[str, str]:
    """Return where each array leaf of `tree` sits, by path, as the string of its
    sharding; a local checkpoint restores onto this placement and no other."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): str(leaf.sharding)
            for path, leaf in leaves
            if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) and leaf.sharding is not None}


def _check_template(template, metadata, stored) -> None:
    """Refuse a train-state template the checkpoint at hand cannot fill.

    A mapping template names the leaves it wants and is checked by the
    restore itself. A whole train state has to find every required field, and
    its scaler and EMA have to be present or absent together with the
    checkpoint's.
    """
    if template is None or isinstance(template, Mapping):
        return
    missing = set(STATE_LEAVES).difference(stored)
    if missing:
        raise ValueError(f"training checkpoint lacks required state fields {sorted(missing)}")
    if (metadata["scale"] is None) != (template.scale is None):
        raise ValueError("checkpoint dynamic-scaler configuration differs from this run")
    if (metadata["ema"] is None) != (template.ema is None):
        raise ValueError("checkpoint EMA configuration differs from this run")


def _position_leaves(state_tree: dict, restore_args: dict, metadata, *,
                     from_local: bool) -> None:
    """Add the data position to the leaves and the args a typed restore reads.

    The table's shape follows the process count and the iterator's position,
    so it comes from the checkpoint's own metadata rather than the template. A
    local checkpoint holds it as a device array, replicated like the step.
    """
    target = next(iter(jax.tree.leaves(state_tree)), None)
    target_sharding = getattr(target, "sharding", None)
    position_sharding = (
        jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec())
        if isinstance(target_sharding, jax.sharding.NamedSharding) else
        jax.sharding.SingleDeviceSharding(jax.local_devices()[0]))
    state_tree["position"] = jax.tree.map(
        lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype),
        dict(metadata['position']))
    restore_args['position'] = jax.tree.map(
        lambda leaf: (ocp.ArrayRestoreArgs(sharding=position_sharding,
                                           global_shape=leaf.shape)
                      if from_local else ocp.RestoreArgs()),
        state_tree['position'])


def _filled(template, restored: dict, step: int):
    """Return `restored` as the train state `template` describes, or as it stands.

    A whole train state is refused where the serialized step or accumulation
    window disagrees with the directory it came from, since both are part of
    the resume contract.
    """
    if template is None or isinstance(template, Mapping):
        return restored
    if int(restored["step"]) != step:
        raise ValueError("checkpoint directory and serialized attempted step disagree")
    if (not isinstance(template.window_size, jax.ShapeDtypeStruct)
            and int(restored["window_size"]) != int(template.window_size)):
        raise ValueError("checkpoint accumulation window_size differs from this run")
    return template.replace(**restored)


class Checkpoints:
    """Holds the checkpoints of one run, in one directory.

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
        """Return this process's own local directory."""
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
        """Return the local step every process holds, or None.

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
        """Return the newest step a resume can read, local or persistent."""
        persistent = self._open().latest_step()
        local = self._local_latest()
        if persistent is None or local is None:
            return local if persistent is None else persistent
        return max(persistent, local)

    @property
    def best(self) -> int | None:
        """Return the step with the lowest reported loss, or None when no save carried one."""
        return self._open().best_step()

    def path(self, step: int) -> str:
        return str(epath.Path(self.directory) / str(step))

    def source(self, step: int) -> str:
        """Return the directory `restore` reads `step` from: this process's local one
        when the step is the local one every process holds, else the
        persistent one."""
        return self.local_path if step == self._local_latest() else self.directory

    def save(self, step: int, state: TrainState, saved: bytes | None,
             metrics: Mapping[str, float] | None = None, *,
             share: DataPartition | None = None) -> None:
        """Write `state` under `step`, asynchronously.

        Sharded arrays go straight to orbax: gathering them onto the host
        first would serialise the whole state through one process and undo
        the point of an async checkpointer. A stream reports its position as
        JSON bytes, which tensorstore has no dtype for; the raw bytes ride
        along as uint8 rows instead, one per process beside the data `share`
        it read, so a global position and a share's offset are stored the
        same way and told apart on restore. A position without its share is
        refused, since no reader could be matched to it.
        A write that fails surfaces from `wait`, which is deliberately
        unguarded: a checkpoint that did not land is data loss.
        """
        with region("checkpoint.submit"):
            self._open().save(step, args=ocp.args.PyTreeSave(self._item(state, saved, share)),
                              metrics=metrics, force=True)

    def save_local(self, step: int, state: TrainState, saved: bytes | None, *,
                   share: DataPartition | None = None) -> None:
        """Write `state` under `step` to this process's local directory,
        asynchronously, in place of the local step before it. The placement
        rides along; a resume onto another one raises before reading shards
        from directories that do not hold them."""
        state_tree = self._item(state, saved, share)
        written = placement(state_tree)
        if saved is not None:
            state_tree['position'] = jax.tree.map(
                lambda leaf: jax.device_put(leaf, state.step.sharding), state_tree['position'])
        with region("checkpoint.submit_local"):
            self._open_local().save(
                step, args=ocp.args.PyTreeSave(state_tree), force=True,
                custom_metadata={'processes': jax.process_count(), 'placement': written})

    @staticmethod
    def _item(state: TrainState, saved: bytes | None,
              share: DataPartition | None) -> dict[str, StateLeaf]:
        state_tree = {name: getattr(state, name) for name in STATE_LEAVES}
        if saved is not None:
            if share is None:
                raise ValueError(
                    "a data position is where one share of the data stopped; save it "
                    "with the share its stream read (share=DataPartition(...))")
            state_tree['position'] = gather_positions(saved, share)
        return state_tree

    def stored(self, step: int | None = None) -> Variables:
        """Return what the checkpoint at `step` holds, without reading its values.

        `step` defaults to the latest. Each state field comes back as a
        shape/dtype tree, and an unset field as None.
        """
        if step is None:
            step = self.latest
            if step is None:
                raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        checkpointer = self._open_local() if step == self._local_latest() else self._open()
        metadata = checkpointer.item_metadata(step)
        return {name: None if value is None else
                jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), value)
                for name, value in dict(metadata).items()}

    def accumulation_template(self, step: int):
        """Return the persisted pending-array shapes, without reading their values."""
        from dew.training.state import Accumulation
        checkpointer = self._open_local() if step == self._local_latest() else self._open()
        metadata = checkpointer.item_metadata(step)
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

    @overload
    def restore[StateT](self, template: StateT, step: int | None = None, *,
                        share: DataPartition | None = None) -> tuple[StateT, bytes | None]: ...

    @overload
    def restore(self, template: None = None, step: int | None = None, *,
                share: DataPartition | None = None) -> tuple[Variables, bytes | None]: ...

    def restore(self, template=None, step: int | None = None, *,
                share: DataPartition | None = None):
        """Restore the state at `step` and the data position of `share`.

        `template` is a pytree of `jax.ShapeDtypeStruct` naming the state
        leaves to restore; a leaf's sharding, when set, is where the array is
        placed, so a checkpoint written on one mesh restores onto whatever
        mesh this run is using. `None` restores every leaf as a host array.

        A step that is the local one every process holds is read from the
        local directory, onto the placement it was written with; any other
        step from the persistent one. The data position comes back as the
        bytes the reader of `share` resumes from (`read_position`); without a
        share, as for a caller that reads weights and no data, it is None.
        """
        local = self._local_latest()
        if step is None:
            step = self.latest
            if step is None:
                raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        from_local = step == local
        checkpointer = self._open_local() if from_local else self._open()
        where = self.local_path if from_local else self.path(step)
        if from_local and template is None and jax.process_count() > 1:
            raise ValueError(
                f"The local checkpoint at {where} holds each process's own shards, "
                f"so a pool cannot read step {step} as host arrays; restore it with "
                f"the template of a run placed as it was written, or read the "
                f"persistent checkpoint at {self.path(step)}")
        metadata = checkpointer.item_metadata(step)
        stored = metadata.keys()
        _check_template(template, metadata, stored)
        if template is None:
            # Typed as host arrays, so orbax reads no sharding file and warns
            # about none. A local checkpoint knows device arrays only, so its
            # leaves land on one device and come home from there.
            untyped = (
                ocp.ArrayRestoreArgs(sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
                if from_local else ocp.ArrayRestoreArgs(restore_type=np.ndarray))
            restored = checkpointer.restore(step, args=ocp.args.PyTreeRestore(
                restore_args=jax.tree.map(lambda _: untyped, dict(metadata))))
            if from_local:
                restored = jax.tree.map(np.asarray, restored)
        else:
            state_tree = {name: getattr(template, name) for name in STATE_LEAVES} \
                if not isinstance(template, Mapping) else dict(template)
            if from_local:
                self._check_placement(step, state_tree)
            restore_args = jax.tree.map(
                lambda leaf: ocp.ArrayRestoreArgs(
                    sharding=leaf.sharding if isinstance(leaf, jax.ShapeDtypeStruct) else None),
                state_tree)
            if 'position' in stored and not isinstance(template, Mapping):
                _position_leaves(state_tree, restore_args, metadata, from_local=from_local)
            try:
                # partial_restore: a key the checkpoint holds and the template
                # does not is skipped instead of refused.
                restored = checkpointer.restore(step, args=ocp.args.PyTreeRestore(
                    item=state_tree, restore_args=restore_args, partial_restore=True))
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
        saved = None if table is None or share is None else read_position(table, where, share)
        restored = _filled(template, restored, step)
        return restored, saved

    def _check_placement(self, step: int, state_tree: Mapping[str, StateLeaf]) -> None:
        """Refuse a local step written for another placement of the state."""
        with region("checkpoint.validate"):
            written = self._open_local().metadata(step).custom_metadata or {}
            wanted = placement(state_tree)
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
        with region("checkpoint.wait"):
            error = None
            for checkpointer in (self._manager, self._local_manager):
                if checkpointer is not None:
                    try:
                        checkpointer.wait_until_finished()
                    except BaseException as failure:
                        if error is None:
                            error = failure
                        else:
                            error.add_note(f"Checkpoint wait also failed: {failure!r}")
            if error is not None:
                raise error
