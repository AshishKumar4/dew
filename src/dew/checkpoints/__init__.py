"""Save and restore a run's train state and data position through orbax.

A checkpoint holds `step`, `params`, `opt_state`, `ema`, `key` and, when the
data iterator can report one, `position`. Metrics, the loss scale and epoch
counters are the loop's business and are rebuilt on resume. A position is
either global, and readable by any partition of the data, or one share's
own offset, and readable only by a reader of that share; `dew.position` is
the difference and `read_position` acts on it.

The EMA copy is stored as its difference from the weights: each EMA leaf
with its weight's floating dtype and shape as the XOR of the two, split
into byte planes (a leading uint8 axis, least significant byte first). An
average agrees with the weights it follows in its leading bits, and zstd,
which orbax applies to every array, stores the runs of zeros that leaves
for little: the EMA of a 176M-parameter DiT at step 1.35M takes 25% fewer
bytes. The step's custom metadata records which leaves are stored so;
`restore` and `stored` hand them back as they were trained, bit for bit,
and a checkpoint that records none reads as it was written.

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

import copy
import dataclasses
import datetime
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, overload, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from etils import epath
from jax.experimental import multihost_utils
from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation

from dew import position
from dew.objectives.base import Variables
from dew.records import JSON, duration, json_value
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


def _is_profiles(node) -> bool:
    """Whether `node` is a `PowerProfilesState`, or the mapping of its fields
    a restore without a template reads it as."""
    from dew.training.optim import PowerProfilesState
    return isinstance(node, PowerProfilesState) or (
        isinstance(node, Mapping) and set(node) == set(PowerProfilesState._fields))


def _power_profiles(opt_state):
    """The `PowerProfilesState` inside `opt_state`, or its mapping, or None
    where the solver keeps none."""
    held = [node for node in jax.tree.leaves(opt_state, is_leaf=_is_profiles) if _is_profiles(node)]
    if len(held) > 1:
        raise ValueError(f"the optimizer state holds {len(held)} power_profiles; wrap the solver once")
    return held[0] if held else None


def _averages_of(profiles):
    """The averages a `PowerProfilesState`, or its mapping, holds."""
    return profiles['averages'] if isinstance(profiles, Mapping) else profiles.averages


def _with_averages(opt_state, averages):
    """`opt_state` with the averages of its power profiles replaced by `averages`."""
    def put(node):
        if not _is_profiles(node):
            return node
        return (
            {**node, "averages": averages} if isinstance(node, Mapping) else node._replace(averages=averages)
        )

    return jax.tree.map(put, opt_state, is_leaf=_is_profiles)


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


@dataclasses.dataclass(frozen=True)
class Keep:
    """Union of latest steps, periodic steps, wall-time-spaced checkpoints and a predicate.

    `interval` keeps checkpoints at least this wall time apart, not all old
    checkpoints. `where` receives a retained-step record with its metrics.
    """
    latest: int = 2
    every: int | None = None
    interval: datetime.timedelta | str | None = None
    where: Callable[[Kept], bool] | None = dataclasses.field(default=None, metadata={'record': False})

    def __post_init__(self):
        if self.latest < 0 or (self.every is not None and self.every < 1):
            raise ValueError("Keep needs latest >= 0 and every >= 1")
        interval = duration(self.interval) if isinstance(self.interval, str) else self.interval
        if interval is not None and interval.total_seconds() <= 0:
            raise ValueError("Keep.interval must be positive")
        object.__setattr__(self, 'interval', interval)


@dataclasses.dataclass(frozen=True)
class Ranking:
    """Evaluation's ranking of these weights; storage only retains it."""
    metric: str
    value: float
    mode: Literal['min', 'max'] = 'min'
    top: int = 1
    weights_only: bool = False


def _recorded_rank(metrics):
    # Orbax records the metrics file only when best_fn is configured. The
    # retention policy and named readers use every independently stored rank;
    # this hands orbax the first, or None for a save that recorded none.
    return metrics.get(next((key for key in metrics if key.startswith('checkpoint/rank/')), None))


class Metrics(Mapping):
    """Recorded scalars indexed by their names or by the objects that produced them."""
    def __init__(self, values: Mapping[str, float]):
        self._values = values

    def __getitem__(self, key):
        from dew.objectives.base import Metric, TrainingScalar
        if isinstance(key, str):
            return self._values[key]
        if isinstance(key, TrainingScalar):
            return self._values[f'train/{key.name}']
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], Metric):
            return self._values[f'{key[0]}/{key[1].name}']
        if isinstance(key, Metric):
            names = [name for name in self._values if not name.startswith('checkpoint/')
                     and (name == key.name or name.endswith('/' + key.name))]
            if len(names) != 1:
                raise KeyError(
                    f"metric {key.name!r} is absent or belongs to several splits; use (split, metric)"
                )
            return self._values[names[0]]
        raise KeyError(key)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


@dataclasses.dataclass(frozen=True)
class Kept:
    step: int
    metrics: Metrics
    ranked_by: str | None
    mode: str | None
    kind: str = 'state'
    rankings: Mapping[str, dict] = dataclasses.field(default_factory=dict)


def _check_shared(directory: str) -> None:
    """Refuse a directory the processes of a pool do not all see.

    Orbax writes one checkpoint across a pool: every process writes the
    shards it holds, process 0 writes the metadata and commits the step once
    they all have, and a restore reads shards other processes wrote. On
    disks of their own, process 0 commits a step without the others' shards
    while they see no step at all, so a resume trains different states on
    different processes. Process 0 leaves a file in the directory and every
    process looks for it; a bucket is one store for every process already.
    """
    if jax.process_count() == 1 or is_uri(directory):
        return
    token = multihost_utils.broadcast_one_to_all(np.frombuffer(os.urandom(8), np.uint8))
    marker = epath.Path(directory) / f".shared-{np.asarray(token).tobytes().hex()}"
    if jax.process_index() == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    multihost_utils.sync_global_devices("dew checkpoints: directory marked")
    seen = np.asarray(multihost_utils.process_allgather(np.asarray(marker.exists())))
    if jax.process_index() == 0:
        marker.unlink()
    blind = [index for index, saw in enumerate(seen.reshape(-1)) if not saw]
    if blind:
        raise ValueError(
            f"The checkpoint directory {directory} is not shared: process(es) {blind} of "
            f"{jax.process_count()} do not see the file process 0 wrote there. A checkpoint "
            f"is written by every process of the pool and read back by any, so the "
            f"directory has to be one every process reads and writes, on a shared "
            f"filesystem or in a bucket. Each host's own disk can hold local checkpoints "
            f"beside it (local_directory).")


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
    refused with the shares named.
    """
    written = len(table['lengths'])
    # Row order, not set order: bytes hash differently per interpreter, and
    # which of the refusals below a broken checkpoint gets is a diagnostic.
    rows = [_row(table, index) for index in range(written)]
    if position.translates(rows[0]):
        if any(row != rows[0] for row in rows[1:]):
            raise ValueError(
                f"The checkpoint at {where} holds a global data position that "
                f"differs between the {_processes(written)} that wrote it. A global "
                f"position is one place in one order, so those processes read "
                f"different orders and no single one of their positions is this "
                f"run's.")
        return rows[0]
    held = [(int(index), int(count)) for index, count in np.asarray(table['shares'])]
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


def _written_in_place(tree: Mapping[str, StateLeaf]) -> bool:
    """Whether orbax writes an array of `tree` from the array's own buffer.

    Orbax copies each array to host memory before its async write and holds
    the copy until the write lands. An array already in pinned host memory,
    as a host layout keeps the optimizer state and the EMA copy, stays
    where it is (`np.asarray` of it is a view), so the caller's buffer is
    what the write reads, and the step after a save donates that buffer: a
    GPU refuses ("Donation requested for buffer with external reference")."""
    return any(isinstance(leaf, jax.Array) and leaf.sharding.memory_kind == "pinned_host"
               for leaf in jax.tree.leaves(tree))


def placement(tree: Mapping[str, StateLeaf]) -> dict[str, str]:
    """Return where each array leaf of `tree` sits, by path, as the string of its
    sharding; a local checkpoint restores onto this placement and no other."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): str(leaf.sharding)
            for path, leaf in leaves
            if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) and leaf.sharding is not None}


# The EMA copy is stored as byte planes of its difference from the weights.
_UNSIGNED = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _delta_planes(average: jax.Array, live: jax.Array) -> jax.Array:
    """Return `average` XOR `live` split into byte planes, least significant first.

    An average agrees with the weights it follows in its sign, exponent and
    leading mantissa bits, so their XOR is zero there, and a plane of each
    byte position hands zstd those zeros as long runs. Shifts, not a bitcast
    to bytes, fix the plane order whatever the platform's byte order.
    """
    width = average.dtype.itemsize
    delta = average.view(_UNSIGNED[width]) ^ live.view(_UNSIGNED[width])
    return jnp.stack([(delta >> (8 * plane)).astype(jnp.uint8) for plane in range(width)])


def _from_delta_planes(planes, live):
    """Invert `_delta_planes`: the average whose difference from `live` is
    `planes`. Works on numpy arrays and on jax arrays alike."""
    kind = _UNSIGNED[live.dtype.itemsize]
    delta = planes[0].astype(kind)
    for plane in range(1, len(planes)):
        delta = delta | (planes[plane].astype(kind) << (8 * plane))
    return (delta ^ live.view(kind)).view(live.dtype)


@jax.jit(static_argnums=2)
def _decode_delta(planes: jax.Array, live: jax.Array, dtype) -> jax.Array:
    """`_from_delta_planes` as one program per leaf signature: a model's
    repeated blocks share a handful, where one program over every leaf
    would compile anew for every model."""
    return _from_delta_planes(planes, live).astype(dtype)


def _planes_sharding(sharding):
    """Where a leaf's byte planes go: the leaf's placement, with the plane axis
    whole on every device, in device memory."""
    if isinstance(sharding, jax.sharding.NamedSharding):
        return jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(None, *sharding.spec),
                                          memory_kind="device")
    return sharding if sharding is None else sharding.with_memory_kind("device")


def _stored_as_delta(average, live) -> bool:
    """Whether the EMA leaf `average` is written as its difference from `live`.

    Both have to be device arrays of one floating dtype and shape, placed
    by a sharding the planes' can be derived from. An average in pinned
    host memory is written as itself: orbax writes it from its own buffer
    (`_written_in_place`), and differencing it would hold a second copy of
    it in the memory it was put there to spare.
    """
    return (isinstance(average, jax.Array) and isinstance(live, jax.Array)
            and average.dtype == live.dtype and average.shape == live.shape
            and jnp.issubdtype(average.dtype, jnp.floating)
            and isinstance(average.sharding, (jax.sharding.NamedSharding,
                                              jax.sharding.SingleDeviceSharding))
            and "pinned_host" not in (average.sharding.memory_kind, live.sharding.memory_kind))


def _by_path(tree):
    """`tree`'s leaves by their key paths: an EMA leaf's path in the EMA
    tree is its weight's in the params tree, whatever containers hold them."""
    return dict(jax.tree_util.tree_flatten_with_path(tree)[0])


def _with_ema_deltas(state_tree: dict[str, StateLeaf]) -> tuple[dict[str, StateLeaf], list[str]]:
    """Return `state_tree` with its EMA leaves, where `_stored_as_delta` allows,
    as byte planes of their difference from the weights, and the paths of
    those leaves in the EMA tree, which the checkpoint's metadata records.

    The planes are computed on the devices the EMA sits on, sharded as it
    is, so a save holds one more EMA-sized buffer there until orbax has
    copied it to the host. A save runs between steps, when the buffers a
    step holds beside the state, among them a gradient as large as the
    weights, are free.
    """
    ema = state_tree['ema']
    if ema is None:
        return state_tree, []
    paths_leaves, structure = jax.tree_util.tree_flatten_with_path(ema)
    leaves = [leaf for _, leaf in paths_leaves]
    held = _by_path(state_tree['params'])
    lives = [held[path] for path, _ in paths_leaves]
    chosen = [index for index, (average, live) in enumerate(zip(leaves, lives, strict=True))
              if _stored_as_delta(average, live)]
    if not chosen:
        return state_tree, []
    for index in chosen:
        encode = jax.jit(_delta_planes, out_shardings=_planes_sharding(leaves[index].sharding))
        leaves[index] = encode(leaves[index], lives[index])
    return ({**state_tree, 'ema': jax.tree.unflatten(structure, leaves)},
            [jax.tree_util.keystr(paths_leaves[index][0]) for index in chosen])


def _ema_deltas(metadata: ocp.metadata.StepMetadata) -> dict[jax.tree_util.KeyPath, jax.ShapeDtypeStruct]:
    """The EMA leaves the checkpoint at `step` stores as byte planes, by path,
    each with the shape and dtype of the weight it is the difference from,
    which are its own. The step's custom metadata records those paths; a
    checkpoint written before the EMA was stored so records none."""
    stored = dict(_item_metadata(metadata))
    if stored.get('ema') is None:
        return {}
    custom = metadata.custom_metadata or {}
    recorded = set(custom.get('ema_deltas', ()))
    if not recorded:
        return {}
    deltas, lives = {}, _by_path(stored['params'])
    for path, _ in jax.tree_util.tree_flatten_with_path(dict(stored['ema']))[0]:
        if jax.tree_util.keystr(path) in recorded:
            live = lives[path]
            deltas[path] = jax.ShapeDtypeStruct(tuple(live.shape), live.dtype)
    return deltas


@runtime_checkable
class _MetadataTree(Protocol):
    """The state tree exposed by Orbax's PyTree metadata wrapper."""

    @property
    def tree(self) -> Variables: ...


def _item_metadata(metadata: ocp.metadata.StepMetadata) -> Variables:
    """The named state tree, a dict or Orbax's newer metadata wrapper's `tree`."""
    tree = metadata.item_metadata
    if isinstance(tree, _MetadataTree):
        tree = tree.tree
    if not isinstance(tree, Mapping) or any(not isinstance(key, str) for key in tree):
        raise ValueError("the checkpoint holds no named state metadata")
    return dict(tree)


def _plane_template(ema, deltas):
    """Return the EMA template with each leaf `deltas` names as the plane stack
    the checkpoint holds, and those leaves as the template had them, by path."""
    targets = {}

    def planes(path, leaf):
        if path not in deltas:
            return leaf
        targets[path] = leaf
        return jax.ShapeDtypeStruct((np.dtype(deltas[path].dtype).itemsize, *leaf.shape), np.uint8,
                                    sharding=_planes_sharding(getattr(leaf, "sharding", None)))
    return jax.tree_util.tree_map_with_path(planes, ema), targets


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


class _RankedSteps(preservation.PreservationPolicy):
    def __init__(self, checkpoints):
        self.checkpoints = checkpoints

    def should_preserve(self, checkpoints: Sequence[preservation.PolicyCheckpointInfo], *,
                        context: preservation.PreservationContext) -> Sequence[bool]:
        held: set[int] = set()
        full = [
            checkpoint
            for checkpoint in checkpoints
            if not (checkpoint.metrics or {}).get("checkpoint/weights_only", 0)
        ]
        held.update(
            checkpoint.step
            for checkpoint in sorted(full, key=lambda checkpoint: checkpoint.step)[
                -self.checkpoints.keep.latest :
            ]
            if self.checkpoints.keep.latest
        )
        groups: dict[str, list[tuple[float, int]]] = {}
        limits = self.checkpoints._rank_limits
        for checkpoint in checkpoints:
            scores = checkpoint.metrics or {}
            for key, value in scores.items():
                if key.startswith('checkpoint/rank/'):
                    groups.setdefault(key.removeprefix("checkpoint/rank/"), []).append(
                        (value, checkpoint.step)
                    )
        for name, scores in groups.items():
            held.update(step for _, step in sorted(scores)[:limits.get(name, 1)])
        keep = self.checkpoints.keep
        previous = None
        interval = duration(keep.interval) if isinstance(keep.interval, str) else keep.interval
        for checkpoint in sorted(checkpoints, key=lambda checkpoint: checkpoint.step):
            if keep.every and checkpoint.step % keep.every == 0:
                held.add(checkpoint.step)
            if interval and (previous is None or checkpoint.time - previous >= interval):
                held.add(checkpoint.step)
                previous = checkpoint.time
            if keep.where:
                try:
                    selected = keep.where(
                        Kept(checkpoint.step, Metrics(checkpoint.metrics or {}), None, None)
                    )
                except KeyError:
                    selected = False
                if selected:
                    held.add(checkpoint.step)
        return [checkpoint.step in held for checkpoint in checkpoints]


class _ProfileSteps(preservation.PreservationPolicy):
    """Keep ordinary checkpoints that also serve as post-hoc EMA snapshots."""
    def __init__(self, steps: set[int]):
        self.steps = steps

    def should_preserve(self, checkpoints: Sequence[preservation.PolicyCheckpointInfo], *,
                        context: preservation.PreservationContext) -> Sequence[bool]:
        return [checkpoint.step in self.steps for checkpoint in checkpoints]


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

    def __init__(self, directory: str, *, keep: int | Keep = 2,
                 local_directory: str | None = None, local_every: int | None = None):
        if (local_directory is None) != (local_every is None):
            raise ValueError(
                "local checkpoints take both local_directory and local_every: where "
                "every host writes its copy, and every how many steps")
        if local_every is not None and local_every < 1:
            raise ValueError(f"local_every must be at least 1, got {local_every}")
        self.directory = str(location(directory))
        self.keep = Keep(latest=keep) if isinstance(keep, int) else keep
        self._rank_limits: dict[str, int] = {}
        self._rank_modes: dict[str, str] = {}
        self._pending: tuple[int, dict[str, float]] | None = None
        self._step_cache: dict[int, Kept] = {}
        self._custom_cache: dict[int, dict] = {}
        self.local_directory = None if local_directory is None else str(location(local_directory))
        self.local_every = local_every
        self._manager = None
        self._local_manager = None
        self._profile_snapshots: set[int] = set()
        self._metadata: tuple[bool, int, ocp.metadata.StepMetadata] | None = None

    def _candidates(self, rankings: Sequence[Ranking]) -> tuple[Ranking, ...]:
        eligible = [rank for rank in rankings if math.isfinite(rank.value)]
        if not eligible:
            return ()
        self._open().check_for_errors()
        retained = {checkpoint.step: checkpoint.metrics for checkpoint in self.kept()}
        if self._pending is not None and self._open().is_saving_in_progress():
            step, scores = self._pending
            retained[step] = Metrics(scores)
        else:
            self._pending = None
        candidates = []
        for rank in eligible:
            score = rank.value if rank.mode == 'min' else -rank.value
            key = f'checkpoint/rank/{rank.metric}'
            held = sorted(scores[key] for scores in retained.values() if key in scores)
            if len(held) < rank.top or score < held[rank.top - 1]:
                candidates.append(rank)
        return tuple(candidates)

    def would_keep(self, ranking: Ranking | Sequence[Ranking]) -> bool:
        """Whether an evaluation would enter any of its trackers' best-K sets."""
        return bool(self._candidates((ranking,) if isinstance(ranking, Ranking) else ranking))

    def _cache_step(self, step: int, metadata) -> Kept:
        custom = metadata.custom_metadata or {}
        rules = custom.get('rankings') or {}
        selection = next(iter(rules.values()), {})
        checkpoint = Kept(
            step=step,
            metrics=Metrics(metadata.metrics or {}),
            ranked_by=custom.get('primary') or next(iter(rules), None),
            mode=selection.get('mode'),
            kind='weights' if custom.get('weights_only') else 'state',
            rankings=copy.deepcopy(rules))
        self._step_cache[step] = checkpoint
        self._custom_cache[step] = copy.deepcopy(custom)
        return checkpoint

    def kept(self) -> list[Kept]:
        """Committed retained steps, oldest first; immutable metadata is cached."""
        persistent = self._open()
        active = set(persistent.all_steps())
        for step in set(self._step_cache) - active:
            del self._step_cache[step]
            self._custom_cache.pop(step, None)
        retained = []
        for step in sorted(active):
            checkpoint = self._step_cache.get(step)
            if checkpoint is None:
                if not self._complete(step):
                    continue
                checkpoint = self._cache_step(step, persistent.metadata(step))
            # Callers own the returned nested rules; changing them must not
            # mutate the committed metadata cached for later queries.
            retained.append(dataclasses.replace(checkpoint, rankings=copy.deepcopy(checkpoint.rankings)))
        return retained

    def artifact(self, step: int | str | None = None) -> JSON:
        """The inference declaration saved with the selected step, or None for old/custom steps."""
        step = self.resolve(step)
        step = self.latest if step is None else step
        if step is None:
            raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        local = step == self._local_latest()
        custom = self._step_metadata(step, local=local).custom_metadata or {}
        return json_value(custom.get('artifact'), 'artifact')

    def control(self, step: int) -> dict:
        if step == self._local_latest():
            custom = self._open_local().metadata(step).custom_metadata or {}
        else:
            self.kept()
            custom = self._custom_cache.get(step) or self._open().metadata(step).custom_metadata or {}
        return copy.deepcopy(custom.get('control', {}))

    def _best_step(self, name: str | None) -> int | None:
        retained = self.kept()
        if name is None:
            for checkpoint in reversed(retained):
                custom = self._custom_cache[checkpoint.step]
                if custom.get('primary'):
                    name = custom['primary']
                    break
        if name is None:
            return None
        key = f'checkpoint/rank/{name}'
        candidates = [(checkpoint.metrics[key], checkpoint.step)
                      for checkpoint in retained if key in checkpoint.metrics]
        return min(candidates)[1] if candidates else None

    def resolve(self, step: int | str | None) -> int | None:
        if not isinstance(step, str):
            return step
        if step != 'best' and not step.startswith('best:'):
            raise ValueError(f"unknown checkpoint selector {step!r}")
        best = self._best_step(None if step == 'best' else step.removeprefix('best:'))
        if best is None:
            raise FileNotFoundError(f"{self.directory} holds no ranked checkpoint for {step}")
        return best

    def _open(self) -> ocp.CheckpointManager:
        if self._manager is None:
            _check_shared(self.directory)
            options = ocp.CheckpointManagerOptions(
                preservation_policy=preservation.AnyPreservationPolicy([
                    _ProfileSteps(self._profile_snapshots),
                    _RankedSteps(self),
                ]),
                best_fn=_recorded_rank, best_mode='min',
                create=True, enable_async_checkpointing=True)
            self._manager = ocp.CheckpointManager(
                self.directory, options=options,
                item_handlers=ocp.PyTreeCheckpointHandler())
            for step in self._manager.all_steps():
                if not self._complete(step):
                    continue
                metadata = self._step_metadata(step)
                self._cache_step(step, metadata)
                custom = metadata.custom_metadata or {}
                if custom.get('profiles') is not None:
                    self._profile_snapshots.add(step)
                for name, rule in (custom.get('rankings') or {}).items():
                    self._rank_limits[name] = rule['top']
                    self._rank_modes[name] = rule['mode']
        return self._manager

    def _step_metadata(self, step: int, *, local: bool = False) -> ocp.metadata.StepMetadata:
        """The most recently inspected committed step, shared by shape and value reads.

        Orbax's metadata opens every array's TensorStore. A restore following
        `stored` needs that same immutable metadata, not a second opening of
        all those arrays. A save invalidates it so a later inspection observes
        the newly committed step. Only one step's metadata is retained.
        """
        held = self._metadata
        if held is not None and held[:2] == (local, step):
            return held[2]
        checkpointer = self._open_local() if local else self._open()
        metadata = checkpointer.metadata(step)
        self._metadata = local, step, metadata
        return metadata

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

    def _complete(self, step: int) -> bool:
        if step in self._step_cache:
            return step in self._open().all_steps()
        path = epath.Path(self.path(step))
        return path.exists() and ocp.utils.is_checkpoint_finalized(path)

    @property
    def latest(self) -> int | None:
        """Return the newest committed step a resume can read, local or persistent."""
        persistent = self._open().latest_step()
        if persistent is not None and (not self._complete(persistent) or
                (self._step_metadata(persistent).custom_metadata or {}).get('weights_only', False)):
            persistent = max(
                (
                    step
                    for step in self._open().all_steps()
                    if self._complete(step)
                    and not (self._step_metadata(step).custom_metadata or {}).get("weights_only", False)
                ),
                default=None,
            )
        local = self._local_latest()
        if persistent is None or local is None:
            return local if persistent is None else persistent
        return max(persistent, local)

    @property
    def best(self) -> int | None:
        return self._best_step(None)

    def path(self, step: int) -> str:
        return str(epath.Path(self.directory) / str(step))

    def source(self, step: int) -> str:
        """Return the directory `restore` reads `step` from: this process's local one
        when the step is the local one every process holds, else the
        persistent one."""
        return self.local_path if step == self._local_latest() else self.directory

    def save(
        self,
        step: int,
        state: TrainState,
        saved: bytes | None,
        metrics: Mapping[str, float] | None = None,
        *,
        share: DataPartition | None = None,
        ranking: Ranking | Sequence[Ranking] | None = None,
        control: dict | None = None,
        weights_only: bool = False,
        primary: str | None = None,
        rung: JSON = None,
        artifact: JSON = None,
    ) -> None:
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

        A state with arrays in pinned host memory is written before this
        returns (`_written_in_place`): the write reads those arrays' own
        buffers, which the next step donates. Handing orbax a copy instead
        would keep the next step waiting only for the copy, but hold a
        second copy of that state in pinned host memory until the write
        lands: 12 bytes a parameter for fp32 Adam moments and EMA, 84 GB at
        7B parameters, on hosts that keep the state there for want of room.
        """
        profiles = None if weights_only else _power_profiles(state.opt_state)
        profile_metadata = None if profiles is None else {
            'updates': int(profiles.updates), 'stds': [float(std) for std in np.asarray(profiles.stds)]}
        state_tree, deltas = _with_ema_deltas(self._item(state, saved, share))
        if weights_only:
            state_tree = {name: state_tree[name] for name in ('params', 'ema')}
        persistent = self._open()
        self._metadata = None
        if profiles is not None:
            self._profile_snapshots.add(step)
        scores = dict(metrics or {})
        if weights_only:
            scores['checkpoint/weights_only'] = 1.0
        rankings = () if ranking is None else (ranking,) if isinstance(ranking, Ranking) else ranking
        rules = {}
        for rank in rankings:
            previous = self._rank_modes.get(rank.metric)
            if previous is not None and previous != rank.mode:
                raise ValueError(f"ranking direction for {rank.metric!r} differs from this run's checkpoints")
            self._rank_modes[rank.metric] = rank.mode
            self._rank_limits[rank.metric] = rank.top
            if math.isfinite(rank.value):
                scores[f'checkpoint/rank/{rank.metric}'] = rank.value if rank.mode == 'min' else -rank.value
                rules[rank.metric] = dataclasses.asdict(rank)
        with region("checkpoint.submit"):
            persistent.save(
                step,
                args=ocp.args.PyTreeSave(state_tree),
                metrics=scores,
                force=True,
                custom_metadata={
                    "ema_deltas": deltas,
                    "profiles": profile_metadata,
                    "rankings": rules,
                    "primary": primary or (rankings[0].metric if rankings else None),
                    "control": copy.deepcopy(control or {}),
                    "weights_only": weights_only,
                    "rung": rung,
                    "artifact": artifact,
                },
            )
        self._pending = (step, scores)
        if _written_in_place(state_tree):
            with region("checkpoint.write_in_place"):
                persistent.wait_until_finished()

    def profile_steps(self) -> list[int]:
        """The complete checkpoints holding post-hoc EMA snapshots, oldest first."""
        persistent = self._open()
        return sorted(step for step in self._profile_snapshots if step in persistent.all_steps()
                      and epath.Path(self.path(step)).exists()
                      and ocp.utils.is_checkpoint_finalized(self.path(step)))

    def profile_metadata(self, step: int) -> tuple[int, tuple[float, ...]]:
        """The updates and relative standard deviations of a snapshot, without reading its averages."""
        self.kept()
        custom = self._custom_cache.get(step) or self._open().metadata(step).custom_metadata or {}
        profiles = custom.get('profiles')
        if profiles is None:
            raise ValueError(f"the checkpoint at step {step} holds no post-hoc EMA snapshot")
        return int(profiles['updates']), tuple(profiles['stds'])

    def restore_profiles(self, step: int) -> tuple[Variables, ...]:
        """Read only a snapshot's averages, as host arrays, from its retained checkpoint."""
        persistent = self._open()
        self.profile_metadata(step)
        opt_state = dict(persistent.item_metadata(step))['opt_state']
        profiles = _power_profiles(opt_state)
        averages = tuple(_averages_of(profiles))
        wanted = _with_averages(jax.tree.map(lambda _: ocp.PLACEHOLDER, opt_state), averages)
        wanted = {'opt_state': wanted}
        host = ocp.ArrayRestoreArgs(restore_type=np.ndarray)
        restored = persistent.restore(step, args=ocp.args.PyTreeRestore(
            item=wanted, partial_restore=True, restore_args=jax.tree.map(lambda _: host, wanted)))
        return tuple(_averages_of(_power_profiles(restored['opt_state'])))

    def posthoc_ema(self, std: float, step: int | None = None) -> Variables:
        """The post-hoc EMA of relative standard deviation `std` at checkpoint
        `step` (default: the latest snapshot), as host arrays in the params'
        structure and dtypes.

        Sums every snapshot up to `step`, of every tracked profile, with the
        weights `coefficients` solves for, one snapshot read at a time and
        accumulated in fp32 or wider. The result goes where the run's params
        go: `merge(params, {"params": checkpoints.posthoc_ema(...)})`.
        """
        from dew.training.posthoc import coefficients

        steps = self.profile_steps()
        if not steps:
            raise FileNotFoundError(f"{self.directory} holds no EMA profile snapshots; train with "
                                    f"OptimConfig.ema_profiles to keep them")
        step = steps[-1] if step is None else step
        if step not in steps:
            raise ValueError(f"{self.directory} holds EMA profile snapshots at steps {steps}, not {step}")
        held = [(each, *self.profile_metadata(each)) for each in steps if each <= step]
        # A snapshot before the first update holds the initial weights, with no
        # profile to fit.
        held = [(each, updates, stds) for each, updates, stds in held if updates > 0]
        if not held or held[-1][0] != step:
            raise ValueError(f"the snapshot at step {step} was taken before the first update")
        weights = iter(coefficients([(updates, deviation) for _, updates, stds in held for deviation in stds],
                                    held[-1][1], std))
        total, dtypes = None, None
        for each, _, _ in held:
            for average in self.restore_profiles(each):
                weight = next(weights)
                if total is None:
                    dtypes = jax.tree.map(lambda leaf: leaf.dtype, average)
                    total = jax.tree.map(
                        lambda leaf, weight=weight: weight
                        * leaf.astype(np.promote_types(leaf.dtype, np.float32)),
                        average,
                    )
                else:
                    total = jax.tree.map(lambda sum_, leaf, weight=weight: sum_ + weight * leaf,
                                         total, average)
        return jax.tree.map(lambda leaf, dtype: leaf.astype(dtype), total, dtypes)

    def save_local(
        self,
        step: int,
        state: TrainState,
        saved: bytes | None,
        *,
        share: DataPartition | None = None,
        control: dict | None = None,
        rung: JSON = None,
        artifact: JSON = None,
    ) -> None:
        """Write `state` under `step` to this process's local directory,
        asynchronously, in place of the local step before it, and as `save`
        does, a state with arrays in pinned host memory before it returns.
        The placement rides along; a resume onto another one raises before
        reading shards from directories that do not hold them."""
        state_tree = self._item(state, saved, share)
        written = placement(state_tree)
        state_tree, deltas = _with_ema_deltas(state_tree)
        if saved is not None:
            state_tree['position'] = jax.tree.map(
                lambda leaf: jax.device_put(leaf, state.step.sharding), state_tree['position'])
        local = self._open_local()
        self._metadata = None
        with region("checkpoint.submit_local"):
            local.save(
                step,
                args=ocp.args.PyTreeSave(state_tree),
                force=True,
                custom_metadata={
                    "processes": jax.process_count(),
                    "placement": written,
                    "ema_deltas": deltas,
                    "control": copy.deepcopy(control or {}),
                    "rung": rung,
                    "artifact": artifact,
                },
            )
        if _written_in_place(state_tree):
            with region("checkpoint.write_in_place"):
                local.wait_until_finished()

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

    def rung(self, step: int) -> JSON:
        """The fit ladder's rung the state at `step` trained on, as `save`
        recorded it, from the directory `restore` reads the step from; None
        for a state saved outside `fit`, which trained on no ladder."""
        checkpointer = self._open_local() if step == self._local_latest() else self._open()
        return json_value((checkpointer.metadata(step).custom_metadata or {}).get('rung'), 'rung')

    def variables(self, *, step: int | str | None = None, ema: bool | None = None,
                  mesh=None, layout=None, param_dtype: str | None = None,
                  parameter_roots: tuple[tuple[str, ...], ...] = (("params",), ("frozen",))) -> Variables:
        """Read the selected step's live or averaged variables onto the requested layout.

        Parameter storage conversion applies only to the owner's parameter
        roots; other collections retain their recorded dtypes and placement.
        """
        from dew.objectives.base import merge
        from dew.registry import resolve_dtype
        from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh, build_mesh

        target = resolve_dtype(param_dtype)
        stored = self.stored(step)
        template = {"params": stored["params"]}
        if ema and stored.get("ema") is None:
            raise ValueError("the run keeps no EMA; request the live policy with ema=False")
        averaged = stored.get("ema") is not None if ema is None else ema
        if averaged:
            template["ema"] = stored["ema"]
        device_mesh = build_mesh(DefaultMesh() if mesh is None else mesh)
        chosen_layout = DefaultLayout() if layout is None else layout
        placement = chosen_layout.shardings(device_mesh, template)
        chosen_layout.check(template["params"], placement["params"], device_mesh)
        selected = set()
        if target is not None:
            roots = tuple(tuple(jax.tree_util.DictKey(name) for name in root) for root in parameter_roots)
            selected = {path for path, leaf in jax.tree_util.tree_flatten_with_path(stored["params"])[0]
                        if jnp.issubdtype(leaf.dtype, jnp.floating) and
                        any(path[:len(root)] == root for root in roots)}
        template = jax.tree_util.tree_map_with_path(
            lambda path, leaf, sharding: jax.ShapeDtypeStruct(
                leaf.shape, target if path[1:] in selected else leaf.dtype, sharding=sharding),
            template, placement)
        values, _ = self.restore(template, step=step)
        params = values["params"]
        if averaged:
            params = merge(params, values["ema"])

        return params

    def stored(self, step: int | str | None = None) -> Variables:
        """Return what the checkpoint at `step` holds, without reading its values.

        `step` defaults to the latest. Each state field comes back as a
        shape/dtype tree, and an unset field as None.
        """
        step = self.resolve(step)
        if step is None:
            step = self.latest
            if step is None:
                raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        snapshot = self._step_metadata(step, local=step == self._local_latest())
        metadata = _item_metadata(snapshot)
        stored = {name: None if value is None else
                  jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), value)
                  for name, value in dict(metadata).items()}
        deltas = _ema_deltas(snapshot)
        if deltas:
            stored['ema'] = jax.tree_util.tree_map_with_path(
                lambda path, leaf: deltas.get(path, leaf), stored['ema'])
        return stored

    def accumulation_template(self, step: int):
        """Return the persisted pending-array shapes, without reading their values."""
        from dew.training.state import Accumulation
        metadata = _item_metadata(self._step_metadata(step, local=step == self._local_latest()))
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
    def restore[StateT](self, template: StateT, step: int | str | None = None, *,
                        share: DataPartition | None = None) -> tuple[StateT, bytes | None]: ...

    @overload
    def restore(self, template: None = None, step: int | str | None = None, *,
                share: DataPartition | None = None) -> tuple[Variables, bytes | None]: ...

    def restore(self, template=None, step: int | str | None = None, *,
                share: DataPartition | None = None):
        """Restore the state at `step` and the data position of `share`.

        `template` is a pytree of `jax.ShapeDtypeStruct` naming the state
        leaves to restore; a leaf's sharding, when set, is where the array is
        placed, so a checkpoint written on one mesh restores onto whatever
        mesh this run is using. `None` restores every leaf as a host array.
        A template leaf the checkpoint lacks is refused by name, unless the
        template holds it as a concrete array, which is then restored as it
        stands.

        A step that is the local one every process holds is read from the
        local directory, onto the placement it was written with; any other
        step from the persistent one. The data position comes back as the
        bytes the reader of `share` resumes from (`read_position`); without a
        share, as for a caller that reads weights and no data, it is None.
        """
        step = self.resolve(step)
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
        snapshot = self._step_metadata(step, local=from_local)
        if template is not None and not isinstance(template, Mapping) and (
                snapshot.custom_metadata or {}).get('weights_only', False):
            raise ValueError("inference-only weights snapshot; resume a full checkpoint")
        metadata = _item_metadata(snapshot)
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
            deltas = _ema_deltas(snapshot)
            if deltas:
                restored = dict(restored)
                weights = _by_path(restored['params'])
                restored['ema'] = jax.tree_util.tree_map_with_path(
                    lambda path, leaf: _from_delta_planes(leaf, weights[path])
                    if path in deltas else leaf, restored['ema'])
        else:
            state_tree = {name: getattr(template, name) for name in STATE_LEAVES} \
                if not isinstance(template, Mapping) else dict(template)
            if from_local:
                self._check_placement(step, state_tree)
            targets, deltas = {}, {}
            if state_tree.get('ema') is not None:
                deltas = _ema_deltas(snapshot)
                state_tree['ema'], targets = _plane_template(state_tree['ema'], deltas)
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
            lacking = [jax.tree_util.keystr(path) for path, leaf
                       in jax.tree_util.tree_flatten_with_path(dict(restored))[0]
                       if isinstance(leaf, jax.ShapeDtypeStruct)]
            if lacking:
                raise ValueError(
                    f"The checkpoint at {where} holds no {', '.join(lacking[:6])}"
                    f"{f' or {len(lacking) - 6} more leaves' if len(lacking) > 6 else ''}, "
                    f"which this run's state has: the checkpoint was written by a model "
                    f"or objective without them. Restore it with the model it was "
                    f"written with.")
            if targets:
                restored = {**restored, 'ema': self._averages(checkpointer, step, metadata, restored,
                                                              targets, deltas)}
        restored = dict(restored)
        table = restored.pop('position', None)
        saved = None if table is None or share is None else read_position(table, where, share)
        restored = _filled(template, restored, step)
        return restored, saved

    @staticmethod
    def _averages(checkpointer: ocp.CheckpointManager, step: int, metadata, restored,
                  targets: dict, deltas: dict) -> Variables:
        """Return the restored EMA tree with each plane stack of `targets` turned
        back into the leaf it holds, typed and placed as the template has it.

        Undoing a difference takes the weight it was taken from, in its
        stored dtype. The weights the restore has just read serve where the
        template asked for them in that dtype, off pinned host memory; the
        rest are read here, placed as the leaves they undo.
        """
        lives, unread = {}, {}
        held = _by_path(restored.get('params'))
        for path, target in targets.items():
            weight = held.get(path)
            if (isinstance(weight, jax.Array) and weight.dtype == deltas[path].dtype
                    and weight.sharding.memory_kind != "pinned_host"):
                lives[path] = weight
            else:
                sharding = getattr(target, "sharding", None)
                unread[path] = jax.ShapeDtypeStruct(
                    deltas[path].shape, deltas[path].dtype,
                    sharding=None if sharding is None else sharding.with_memory_kind("device"))
        if unread:
            # The stored params tree, its containers kept, with every leaf
            # but the ones to read held back by orbax's placeholder.
            weights = {'params': jax.tree_util.tree_map_with_path(
                lambda path, _: unread.get(path, ocp.PLACEHOLDER), dict(metadata)['params'])}
            read = checkpointer.restore(step, args=ocp.args.PyTreeRestore(
                item=weights, partial_restore=True, restore_args=jax.tree.map(
                    lambda leaf: ocp.ArrayRestoreArgs(sharding=getattr(leaf, "sharding", None)),
                    weights)))
            lives.update({path: leaf for path, leaf in _by_path(read['params']).items() if path in unread})

        def average(path, planes):
            if path not in targets:
                return planes
            target = targets[path]
            value = _decode_delta(planes, lives[path], np.dtype(target.dtype))
            sharding = getattr(target, "sharding", None)
            return value if sharding is None else jax.device_put(value, sharding)
        return jax.tree_util.tree_map_with_path(average, restored['ema'])

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
            self._pending = None
