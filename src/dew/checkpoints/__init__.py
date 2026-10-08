"""Save and restore a run's training state and data position with Orbax.

A checkpoint holds `step`, `variables`, `opt_state`, `ema`, `key` and, when the
data iterator can report one, `position`. Metrics, the loss scale and epoch
counters belong to the training loop, which rebuilds them on resume. A position
is either global, which any partition of the data can read, or one share's own
offset, which only a reader of that same share can read; `dew.position` marks
the difference and `read_position` acts on it. A state that is not a training
run's, such as a simulation's, is saved as a mapping of arrays and restored
through a mapping template, with the same asynchronous writes, retention and
step index.

The EMA copy is stored as its difference from the weights. Each EMA leaf with
its weight's floating dtype and shape is stored as the XOR of the two, split
into byte planes (a leading uint8 axis, least significant byte first). An
average agrees with the weights it follows in its leading bits, and the zstd
compression Orbax applies to every array stores the resulting runs of zeros
cheaply: the EMA of a 176M-parameter DiT at step 1.35M takes 25% fewer bytes.
The step's custom metadata records which leaves are stored this way, so
`restore` and `stored` return them bit for bit as they were trained, and a
checkpoint that records none reads as it was written.

Each kept step stores its weights whole; a best step is not a delta of the
latest. Measured on a 1.8M-parameter byte-level decoder trained 3,000 steps
on Tiny Shakespeare (AdamW, warmup and cosine decay), with each step written
through these same Orbax writes: an earlier step's parameters stored as XOR
planes against the latest take 8% fewer bytes 2,700 steps back, 15% fewer
1,000 back, 23% fewer 100 back and 30% fewer 10 back, and its Adam moments
7% to 17% fewer. Over a best and a latest step that is 4% to 15% of their
parameters' bytes. Storing one step against another would make retention keep
a base alive while any delta reads it, or re-encode the best step at every
save, which that saving does not pay for.

Besides the persistent directory, a run can keep a local checkpoint on every
host, written more often, so a preempted pod resumes from its own disks rather
than from remote storage. This is Orbax's emergency checkpointing
(orbax.checkpoint.experimental.emergency). The local checkpoints are built here
from the same pieces it uses, an `ArrayHandler` with no primary host and no
replica dedup, so every process writes every shard it holds. Orbax's manager
itself is not used, for two reasons: it takes over the persistent directory
with a single fixed-shape state item (no metrics, so no best step, and no room
for the per-process data position), and it restores a local checkpoint only
where each data-parallel replica is a whole set of hosts, which is not true
when the fsdp axis spans hosts. The semantics match Orbax's: two directories,
and the newest checkpoint every process can read wins.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
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
from orbax.checkpoint.checkpoint_manager import AsyncOptions, MultiprocessingOptions
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation

from dew import position
from dew.coordination import agreed_same
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

STATE_LEAVES = ("step", "microstep", "updates", "variables", "opt_state", "ema", "key",
                "scale", "window_size", "accumulation")
"""The train-state fields a checkpoint persists; anything outside this tuple
is rebuilt on resume."""

FROZEN_STORE = "frozen"
"""The directory beside the steps that holds every variables collection but
`params` once, under the digest of its content (`_digest`). A step records
each collection's digest and holds `params` and the optimizer state itself,
so weights no step moves (a LoRA run's frozen base, a pipeline's towers) are
written once a run, and a collection a step does move (batch statistics) is
written again only when it changes. A digest no kept step records is
deleted. The step names it relative to the run directory, so a copied run
still restores."""

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


ORDERED_DIRECTORIES = AsyncOptions(create_directories_asynchronously=False)
"""A local save creates its step's tmp directory, then its items', before it returns.

orbax 0.12.4 creates them by default in background threads that wait on
signals keyed by the save's operation id alone (google/orbax#3621), so in a
pool another process's "directories created" can release this process's
item writer early. Its `parents=True` mkdir then creates the step's
directory, and the step's own `exist_ok=False` mkdir raises FileExistsError
(a CI shard's pool hung on it). Two-process local runs on two cores raised
it in 13 of 53 with the threads unordered and in 0 of 70 with them in
order. In order, a save call on local disk took 14.0 ms against 6.9.

Only the local manager, whose processes each write a directory of their
own, takes it. The persistent manager's primary creates one shared
directory, and its signal is meant to release every process's writer, so
a key shared across processes is the design there; its saves keep
creating directories in the background, which matters on GCS, where a
synchronous mkdir chain costs far more than on local disk."""


def location(directory: str) -> epath.Path:
    """Return the directory a run writes to: a bucket URI unchanged, a local path made absolute.

    `epath` reads and writes both, but `Path.resolve` turns `gs://bucket/run`
    into a local `gs:/bucket/run`, so only a path with no scheme is resolved.
    """
    path = epath.Path(directory)
    return path if is_uri(directory) else path.resolve()


def _processes(count: int) -> str:
    return f"{count} process" + ("es" if count != 1 else "")


@dataclasses.dataclass(frozen=True)
class Keep:
    """Keeps the union of the latest steps, periodic steps, wall-time-spaced steps and a predicate's steps.

    `interval` keeps checkpoints at least this much wall time apart; it does
    not keep every old checkpoint. `where` receives a retained-step record
    with its metrics.
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
    """How an evaluation ranked these weights; checkpoint storage records it without computing it."""
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
    """'state' for a train state, 'weights' for a weights-only snapshot, 'tree'
    for a mapping saved in place of a train state."""
    rankings: Mapping[str, dict] = dataclasses.field(default_factory=dict)


def frozen_entries(step_directory: str) -> list[str]:
    """The `FROZEN_STORE` entries the step at `step_directory` records,
    relative to its run directory, which `dew.io.publish` carries beside it."""
    metadata = epath.Path(step_directory) / "_CHECKPOINT_METADATA"
    custom = (json.loads(metadata.read_text()).get("custom_metadata") or {}) if metadata.exists() else {}
    digests = {digest for names in (custom.get("frozen") or {}).values() for digest in names.values()}
    return [f"{FROZEN_STORE}/{digest}" for digest in sorted(digests)]


_SEEDS = (0x9E3779B9, 0x7F4A7C15)
"""The two multipliers `_marks` mixes a word's flat index by."""


@jax.jit
def _marks(leaves: list[jax.Array]) -> jax.Array:
    """Two 32-bit marks of each leaf's stored bits at their positions, `[leaves, 2]`.

    Each 32-bit word of a leaf's storage, in memory order (an 8-byte element
    is two, a narrower one is widened), is mixed with its flat index
    (murmur3's finalizer) and the words summed, so the marks are a function
    of the stored bytes alone, the same on every sharding and computed where
    the shards are. One call marks a whole collection, so a collection
    compiles once however many shapes it holds. `_host_marks` computes the
    same marks of a host array.
    """
    return jnp.stack([_leaf_marks(leaf) for leaf in leaves]) if leaves else jnp.zeros((0, 2), jnp.uint32)


def _leaf_marks(leaf: jax.Array) -> jax.Array:
    bits = leaf.astype(jnp.uint8) if leaf.dtype == jnp.bool_ else leaf
    width = np.dtype(bits.dtype).itemsize
    # An 8-byte element becomes a trailing pair of 32-bit words.
    unsigned = {1: jnp.uint8, 2: jnp.uint16}.get(width, jnp.uint32)
    bits = jax.lax.bitcast_convert_type(bits, unsigned).astype(jnp.uint32)
    index = jnp.zeros(bits.shape, jnp.uint32)
    stride = 1
    for axis in reversed(range(bits.ndim)):
        index = index + jax.lax.broadcasted_iota(jnp.uint32, bits.shape, axis) * jnp.uint32(stride)
        stride = stride * bits.shape[axis] % 2 ** 32
    marks = []
    for seed in _SEEDS:
        mixed = bits ^ (index * jnp.uint32(seed))
        mixed = (mixed ^ (mixed >> 16)) * jnp.uint32(0x85EBCA6B)
        mixed = (mixed ^ (mixed >> 13)) * jnp.uint32(0xC2B2AE35)
        marks.append(jnp.sum(mixed ^ (mixed >> 16), dtype=jnp.uint32))
    return jnp.stack(marks)


def _host_marks(leaf: np.ndarray, chunk: int = 1 << 24) -> np.ndarray:
    """`_marks` of a host array, from its stored bytes, `chunk` words at a time."""
    array = np.ascontiguousarray(leaf)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    width = array.dtype.itemsize
    words = array.reshape(-1).view({1: np.uint8, 2: np.uint16}.get(width, np.uint32))
    marks = np.zeros(2, np.uint32)
    with np.errstate(over="ignore"):
        for start in range(0, words.size, chunk):
            bits = words[start:start + chunk].astype(np.uint32)
            index = (np.arange(start, start + bits.size, dtype=np.uint64) & 0xFFFFFFFF).astype(np.uint32)
            for slot, seed in enumerate(_SEEDS):
                mixed = bits ^ (index * np.uint32(seed))
                mixed = (mixed ^ (mixed >> np.uint32(16))) * np.uint32(0x85EBCA6B)
                mixed = (mixed ^ (mixed >> np.uint32(13))) * np.uint32(0xC2B2AE35)
                marks[slot] += np.sum(mixed ^ (mixed >> np.uint32(16)), dtype=np.uint32)
    return marks


def _digest(tree) -> str:
    """The SHA-256 of `tree`'s paths, shapes, dtypes and each leaf's marks
    (`_marks`), taken of the bytes as stored, whatever dtypes JAX allows now.

    The device leaves are marked in one call. A host array is marked on the
    host, so a float64 leaf restored with x64 off has the digest it was saved
    with. A device leaf in host memory is marked on the devices one leaf at a
    time, so no more than one is copied off the host at once.
    """
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    on_device = [index for index, (_, leaf) in enumerate(flat) if isinstance(leaf, jax.Array)
                 and leaf.sharding.memory_kind in (None, "device")]
    marks = dict(zip(on_device, np.asarray(jax.device_get(_marks([flat[index][1] for index in on_device]))),
                     strict=True))
    digest = hashlib.sha256()
    for index, (path, leaf) in enumerate(flat):
        if index not in marks:
            if isinstance(leaf, jax.Array):
                array = jax.device_put(leaf, leaf.sharding.with_memory_kind("device"))
                marks[index] = np.asarray(jax.device_get(_marks([array])))[0]
            else:
                marks[index] = _host_marks(np.asarray(leaf))
        digest.update(f"{jax.tree_util.keystr(path)}|{np.shape(leaf)}|{np.dtype(leaf.dtype)}|".encode())
        digest.update(marks[index].astype("<u4").tobytes())
    return digest.hexdigest()


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
    the position it holds, and Orbax writes a host array from process 0
    alone, so the rows are gathered onto every process before a save. The
    rows differ in length, so their lengths are stored with them.

    There is one row per process whatever kind the position is. A global
    position is the same bytes on every process, and gathering it lets
    `read_position` check that the rows agree before another partition reads it.
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

    A global position is a record count over an order that is the same at
    any partition, so every row is a valid position for every reader;
    `dew.position` declares that property and `read_position` relies on it.
    A share's own offset resumes only readers of that same share, whichever
    processes they are, and any other reader is refused with the shares named.
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


def _written_in_place(tree: Mapping[str, object]) -> bool:
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
    """Return the sharding of each array leaf of `tree` as a string, keyed by path.

    A local checkpoint restores onto this placement and no other."""
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
    held = _by_path(state_tree['variables'])
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
    deltas, lives = {}, _by_path(stored['variables'])
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


def _check_template(template, metadata, stored, *, mapping: bool) -> None:
    """Refuse a train-state template the checkpoint at hand cannot fill.

    A mapping template names the leaves it wants and is checked by the
    restore itself. A whole train state has to find every required field, and
    its scaler and EMA have to be present or absent together with the
    checkpoint's. A `mapping` step, a mapping saved in place of a train
    state, has none of them.
    """
    if template is None or isinstance(template, Mapping):
        return
    if mapping:
        raise ValueError("the checkpoint holds a mapping saved in place of a train state; restore it "
                         "through a mapping template, or with none")
    missing = set(STATE_LEAVES).difference(stored)
    if missing:
        raise ValueError(f"training checkpoint lacks required state fields {sorted(missing)}")
    if (metadata["scale"] is None) != (template.scale is None):
        raise ValueError("checkpoint dynamic-scaler configuration differs from this run")
    if (metadata["ema"] is None) != (template.ema is None):
        raise ValueError("checkpoint EMA configuration differs from this run")


def _held_apart(state_tree: dict, frozen: Mapping[str, Mapping[str, str]]) -> dict[str, dict]:
    """Take the collections a step holds in the store (`frozen`, by tree) out
    of the restore template `state_tree`, in place, and return them by tree:
    the restore reads them from the store onto these templates."""
    held = {}
    for tree, names in frozen.items():
        collections = state_tree.get(tree)
        if isinstance(collections, Mapping):
            collections = dict(collections)
            held[tree] = {name: collections.pop(name) for name in names if name in collections}
            state_tree[tree] = collections
    return held


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
    # A template's narrow copies are of its own parameters, not the restored ones.
    return template.replace(**restored, compute=None)


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
    """Manages the checkpoints of one run in one directory.

    Constructing one opens nothing; the orbax managers are created on first
    use. The directory keeps the latest `keep` steps, so a resume has
    something recent, plus the step with the lowest `loss` metric a save
    reported. A save without metrics can never become the best step.

    `local_directory` names a path on every host's own disk where the run
    keeps one more checkpoint, the latest, which `fit` writes every
    `local_every` steps. Each process writes the shards its devices hold under
    its own subdirectory, so the same path works for a pod and for one host
    running several processes. `latest` is the newest step every process can
    read, local or persistent, and `restore` reads it from wherever it is. A
    local checkpoint restores onto the placement it was written with, because
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
        # The digests the save still in flight records, which no kept step lists yet.
        self._frozen_pending: set[str] = set()

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
        """Return whether an evaluation would enter the best-K set of any of its trackers."""
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
            kind='weights' if custom.get('weights_only') else 'tree' if custom.get('tree') else 'state',
            rankings=copy.deepcopy(rules))
        self._step_cache[step] = checkpoint
        self._custom_cache[step] = copy.deepcopy(custom)
        return checkpoint

    def kept(self) -> list[Kept]:
        """Return the committed retained steps, oldest first; their immutable metadata is cached."""
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
        """Return the selected step's inference declaration.

        It is None when the step's objective declares no inference record."""
        step = self.pinned(step)
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

    def pinned(self, step: int | str | None = None) -> int:
        """The exact step `step` selects: itself, the best by a ranking, or, for
        None, the latest committed now, which a later save does not move."""
        step = self.resolve(step)
        step = self.latest if step is None else step
        if step is None:
            raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        return step

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
            # Each process is a pool of one for its local manager: it writes
            # its own metadata, and every shard it holds, one copy per
            # process, so each process's directory is complete for its
            # devices, and orbax's barriers wait for this process alone. Over
            # the whole pool, a save that failed in one process's background
            # thread (FileExistsError creating its tmp directory, a CI shard
            # on 2026-10-05) left that process's finalize waiting at a
            # barrier the others had passed, and the pool hung until orbax's
            # 600 s timeout; alone, the failure reaches the process's next
            # save, which raises and ends the pool.
            # Only device arrays are registered, because orbax writes a host
            # array from process 0 alone whatever the options say; the
            # position table rides as a replicated device array instead. The
            # prefix keeps this manager's barriers apart from the persistent
            # one's and from the other processes' own.
            me = jax.process_index()
            multiprocessing = MultiprocessingOptions(primary_host=me, active_processes={me},
                                                     barrier_sync_key_prefix=f'local{me}')
            registry = ocp.type_handlers.create_type_handler_registry(
                (jax.Array, ocp.type_handlers.ArrayHandler(
                    primary_host=None, replica_id=None, use_replica_parallel=False)))
            # orbax creates no directory for a manager over a subset of the pool.
            epath.Path(self.local_path).mkdir(parents=True, exist_ok=True)
            options = ocp.CheckpointManagerOptions(
                max_to_keep=1, create=False, cleanup_tmp_directories=True,
                enable_async_checkpointing=True, multiprocessing_options=multiprocessing,
                async_options=ORDERED_DIRECTORIES)
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
        """Return the newest committed step a resume can read, local or persistent.

        Each process lists the persistent directory itself, and on a shared
        filesystem one listing can lag another's, so the processes agree on
        the step before any restores it."""
        persistent = agreed_same("newest persistent checkpoint", self._persistent_latest)
        local = self._local_latest()
        if persistent is None or local is None:
            return local if persistent is None else persistent
        return max(persistent, local)

    def _persistent_latest(self) -> int | None:
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
        return persistent

    @property
    def best(self) -> int | None:
        return self._best_step(None)

    def path(self, step: int) -> str:
        return str(epath.Path(self.directory) / str(step))

    def source(self, step: int) -> str:
        """Return the directory `restore` reads `step` from.

        That is this process's local directory when the step is the local one
        every process holds, and the persistent directory otherwise.
        """
        return self.local_path if step == self._local_latest() else self.directory

    def save(
        self,
        step: int,
        state: TrainState | Mapping[str, object],
        saved: bytes | None = None,
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

        `state` is a run's `TrainState`, or a mapping of arrays in its place
        for a state that is not a training run's, such as a simulation's,
        which has no optimizer, average or loss scale. A mapping is written as
        it is, with `metrics`, `ranking` and `control` as a train state's are,
        and `restore` reads it back through a mapping template; it takes no
        data position, share, weights-only split, rung or artifact. It is
        written whole at every step, without the store a train state's
        static collections go to (`FROZEN_STORE`), so large arrays no step
        changes are better kept out of it.

        Sharded arrays go straight to Orbax. Gathering them onto the host
        first would serialise the whole state through one process and defeat
        the purpose of an async checkpointer. A stream reports its position as
        JSON bytes, which tensorstore has no dtype for, so the bytes are stored
        as uint8 rows, one per process, next to the data `share` that process
        read. A global position and a share's offset are stored the same way
        and told apart on restore. A position without its share is refused,
        since no reader could be matched to it. A failed write surfaces from
        `wait`, which deliberately does not catch it: a checkpoint that did
        not land is lost data.

        A state with arrays in pinned host memory is written before this
        returns, because the write reads those arrays' own buffers and the
        next step donates them. Giving Orbax a copy instead would make the
        next step wait only for the copy, but it would hold a second copy of
        that state in pinned host memory until the write lands: 12 bytes a
        parameter for fp32 Adam moments and EMA, 84 GB at 7B parameters, on
        hosts that keep the state there because device memory is short.
        """
        persistent = self._open()
        mapping = isinstance(state, Mapping)
        if mapping:
            if (saved is not None or share is not None or weights_only or rung is not None
                    or artifact is not None):
                raise ValueError("a mapping is saved as it is: it takes no data position, share, "
                                 "weights-only split, rung or artifact, which a train state's save records")
            profiles, state_tree, frozen, deltas = None, dict(state), {}, []
        else:
            profiles = None if weights_only else _power_profiles(state.opt_state)
            state_tree, frozen = self._frozen(self._item(state, saved, share))
            state_tree, deltas = _with_ema_deltas(state_tree)
            if weights_only:
                state_tree = {name: state_tree[name] for name in ('variables', 'ema')}
        profile_metadata = None if profiles is None else {
            'updates': int(profiles.updates), 'stds': [float(std) for std in np.asarray(profiles.stds)]}
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
                    "frozen": frozen,
                    "tree": mapping,
                },
            )
        self._pending = (step, scores)
        self._frozen_pending = {digest for held in frozen.values() for digest in held.values()}
        # Pinned-host state is written before returning; the docstring says why.
        if _written_in_place(state_tree):
            with region("checkpoint.write_in_place"):
                persistent.wait_until_finished()

    def _frozen(self, state_tree: dict[str, StateLeaf]
                ) -> tuple[dict[str, StateLeaf], dict[str, dict[str, str]]]:
        """Return `state_tree` with its variables' and its EMA's collections
        but `params` held in the store (`FROZEN_STORE`), and each one's digest
        under its tree, which the step records.

        An EMA collection no average moves has its weights' content, so it is
        the same entry. A collection the store lacks is written now, before
        the step, by every process, as process 0 finds the store. A stored
        collection that no kept step, this step nor the one still in flight
        records is deleted first, by process 0.
        """
        held: dict[str, dict[str, object]] = {}
        for tree in ('variables', 'ema'):
            collections = state_tree.get(tree)
            if isinstance(collections, Mapping):
                held[tree] = {str(name): value for name, value in collections.items()
                              if name != 'params' and jax.tree.leaves(value)}
        digests = {tree: {name: _digest(value) for name, value in values.items()}
                   for tree, values in held.items()}
        root = epath.Path(self.directory) / FROZEN_STORE
        recorded = {digest for values in digests.values() for digest in values.values()}
        if jax.process_index() == 0 and root.exists():
            referenced = {*recorded, *self._frozen_pending}
            for kept in self.kept():
                for values in ((self._custom_cache.get(kept.step) or {}).get('frozen') or {}).values():
                    referenced.update(values.values())
            for path in root.iterdir():
                if path.name not in referenced and ocp.utils.is_checkpoint_finalized(path):
                    path.rmtree()
        stored = ({path.name for path in root.iterdir()}
                  if jax.process_index() == 0 and root.exists() else set())
        written: set[str] = set()
        for tree, values in held.items():
            for name, value in values.items():
                digest = digests[tree][name]
                there = digest in written or bool(multihost_utils.broadcast_one_to_all(
                    np.asarray(digest in stored)))
                if not there:
                    with region("checkpoint.frozen"):
                        ocp.PyTreeCheckpointer().save(root / digest, args=ocp.args.PyTreeSave(value))
                    written.add(digest)
        kept_tree = dict(state_tree)
        for tree, values in held.items():
            collections = state_tree[tree]
            assert isinstance(collections, Mapping)
            kept_tree[tree] = {name: value for name, value in collections.items() if name not in values}
        return kept_tree, digests

    def _with_frozen(self, restored: dict, frozen: Mapping[str, Mapping[str, str]],
                     held: Mapping[str, Mapping] | None) -> dict:
        """`restored` with each collection the step records in the store read
        back into its tree, onto `held`'s templates (`_held_apart`) or, with
        none, as host arrays."""
        for tree, names in frozen.items():
            collections = restored.get(tree)
            if isinstance(collections, Mapping):
                templates = None if held is None else held.get(tree, {})
                wanted = names if templates is None else {name: names[name] for name in templates}
                restored[tree] = {**collections, **self._frozen_values(wanted, templates)}
        return restored

    def _frozen_values(self, digests: Mapping[str, str], templates: Mapping | None) -> dict[str, Variables]:
        """Read each collection a step records from the store, onto its
        template's placement and dtype or as host arrays, refusing one whose
        content no longer has the digest the step recorded."""
        values = {}
        wide = jax.config.jax_enable_x64

        def read(meta, want=None):
            """A leaf in its stored dtype: on the host where no template places
            it or JAX cannot hold that dtype now (a float64 saved with x64 on,
            read with it off), else where the template places it."""
            if want is None or (np.dtype(meta.dtype).itemsize == 8 and not wide):
                return ocp.ArrayRestoreArgs(restore_type=np.ndarray)
            return ocp.ArrayRestoreArgs(sharding=getattr(want, 'sharding', None), dtype=meta.dtype)

        def placed(leaf, want):
            value = jnp.asarray(leaf).astype(want.dtype)
            sharding = getattr(want, 'sharding', None)
            return value if sharding is None else jax.device_put(value, sharding)

        for name, digest in digests.items():
            path = epath.Path(self.directory) / FROZEN_STORE / digest
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is gone: the step records its {name} collection there, so the run "
                    f"directory was copied or published without its {FROZEN_STORE}/ directory")
            checkpointer = ocp.PyTreeCheckpointer()
            stored = dict(_item_metadata(checkpointer.metadata(path)))
            template = None if templates is None else templates.get(name)
            restore_args = (jax.tree.map(read, stored) if template is None
                            else jax.tree.map(read, stored, template))
            tree = checkpointer.restore(path, args=ocp.args.PyTreeRestore(restore_args=restore_args))
            found = _digest(tree)
            if found != digest:
                raise ValueError(f"{path} holds content with digest {found}, not the {digest} the step "
                                 f"recorded for its {name} collection")
            if template is not None:
                tree = jax.tree.map(placed, tree, template)
            values[name] = tree
        return values

    def profile_steps(self) -> list[int]:
        """Return the complete checkpoints that hold post-hoc EMA snapshots, oldest first."""
        persistent = self._open()
        return sorted(step for step in self._profile_snapshots if step in persistent.all_steps()
                      and epath.Path(self.path(step)).exists()
                      and ocp.utils.is_checkpoint_finalized(self.path(step)))

    def profile_metadata(self, step: int) -> tuple[int, tuple[float, ...]]:
        """Return a snapshot's recorded updates and relative standard deviations, without its averages."""
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
        """Return the post-hoc EMA with relative standard deviation `std` at checkpoint `step`.

        `step` defaults to the latest snapshot. The result is host arrays in
        the params' structure and dtypes. It sums every snapshot up to `step`,
        from every tracked profile, with the weights `coefficients` solves for,
        reading one snapshot at a time and accumulating in fp32 or wider. Use
        the result where the run's params go:
        `merge(variables, {"params": checkpoints.posthoc_ema(...)})`.
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
        """Write `state` under `step` to this process's local directory, asynchronously.

        The new step replaces the previous local step. As with `save`, a state
        with arrays in pinned host memory is written before this returns. The
        placement is saved with the state, so a resume onto a different
        placement raises before it reads shards from directories that do not
        hold them.
        """
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
        """Return the rung of `fit`'s ladder that the state at `step` trained on.

        The value is what `save` recorded, read from the directory `restore`
        reads the step from. It is None for a state saved outside `fit`,
        which trained on no ladder.
        """
        checkpointer = self._open_local() if step == self._local_latest() else self._open()
        return json_value((checkpointer.metadata(step).custom_metadata or {}).get('rung'), 'rung')

    def variables(self, *, step: int | str | None = None, ema: bool | None = None,
                  mesh=None, layout=None, param_dtype: jax.typing.DTypeLike | None = None,
                  parameter_roots: tuple[tuple[str, ...], ...] = (("params",), ("frozen",))) -> Variables:
        """Read the selected step's live or averaged variables onto the requested layout.

        Parameter storage conversion applies only to the owner's parameter
        roots; other collections retain their recorded dtypes and placement.
        """
        from dew.objectives.base import merge
        from dew.registry import resolve_dtype
        from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh

        target = resolve_dtype(param_dtype)
        stored = self.stored(step)
        template = {"variables": stored["variables"]}
        if ema and stored.get("ema") is None:
            raise ValueError("the run keeps no EMA; request the live policy with ema=False")
        averaged = stored.get("ema") is not None if ema is None else ema
        if averaged:
            template["ema"] = stored["ema"]
        device_mesh = (DefaultMesh() if mesh is None else mesh).build()
        chosen_layout = DefaultLayout() if layout is None else layout
        placement = chosen_layout.shardings(device_mesh, template)
        chosen_layout.check(template["variables"], placement["variables"], device_mesh)
        selected = set()
        if target is not None:
            roots = tuple(tuple(jax.tree_util.DictKey(name) for name in root) for root in parameter_roots)
            selected = {path for path, leaf in jax.tree_util.tree_flatten_with_path(stored["variables"])[0]
                        if jnp.issubdtype(leaf.dtype, jnp.floating) and
                        any(path[:len(root)] == root for root in roots)}
        template = jax.tree_util.tree_map_with_path(
            lambda path, leaf, sharding: jax.ShapeDtypeStruct(
                leaf.shape, target if path[1:] in selected else leaf.dtype, sharding=sharding),
            template, placement)
        values, _ = self.restore(template, step=step)
        params = values["variables"]
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
        for tree, names in ((snapshot.custom_metadata or {}).get('frozen') or {}).items():
            collections = stored.get(tree)
            if isinstance(collections, Mapping):
                stored[tree] = {**collections, **{
                    name: jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), dict(
                        _item_metadata(ocp.PyTreeCheckpointer().metadata(
                            epath.Path(self.directory) / FROZEN_STORE / digest))))
                    for name, digest in names.items()}}
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
        leaves to restore. When a leaf has a sharding, the array is placed
        there, so a checkpoint written on one mesh restores onto whatever mesh
        this run uses. `None` restores every leaf as a host array. A template
        leaf that the checkpoint lacks raises an error naming it, unless the
        template holds it as a concrete array, which is then kept as it is.

        A step that is the local one every process holds is read from the
        local directory, onto the placement it was written with; any other
        step is read from the persistent directory. The data position is
        returned as the bytes the reader of `share` resumes from (see
        `read_position`). Without a share, as for a caller that reads weights
        and no data, it is None.
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
        mapping = bool((snapshot.custom_metadata or {}).get('tree'))
        if template is not None and not isinstance(template, Mapping) and (
                snapshot.custom_metadata or {}).get('weights_only', False):
            raise ValueError("inference-only weights snapshot; resume a full checkpoint")
        metadata = _item_metadata(snapshot)
        stored = metadata.keys()
        _check_template(template, metadata, stored, mapping=mapping)
        frozen = (snapshot.custom_metadata or {}).get('frozen') or {}
        held: dict[str, dict] | None = None
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
                weights = _by_path(restored['variables'])
                restored['ema'] = jax.tree_util.tree_map_with_path(
                    lambda path, leaf: _from_delta_planes(leaf, weights[path])
                    if path in deltas else leaf, restored['ema'])
        else:
            state_tree = {name: getattr(template, name) for name in STATE_LEAVES} \
                if not isinstance(template, Mapping) else dict(template)
            held = _held_apart(state_tree, frozen)
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
                if mapping:
                    raise ValueError(
                        f"The checkpoint at {where} holds a mapping that does not fit this "
                        f"template ({mismatch}); restore it with the structure, shapes and "
                        f"dtypes it was saved with.") from mismatch
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
        restored = self._with_frozen(dict(restored), frozen, held)
        table = None if mapping else restored.pop('position', None)
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
        held = _by_path(restored.get('variables'))
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
            weights = {'variables': jax.tree_util.tree_map_with_path(
                lambda path, _: unread.get(path, ocp.PLACEHOLDER), dict(metadata)['variables'])}
            read = checkpointer.restore(step, args=ocp.args.PyTreeRestore(
                item=weights, partial_restore=True, restore_args=jax.tree.map(
                    lambda leaf: ocp.ArrayRestoreArgs(sharding=getattr(leaf, "sharding", None)),
                    weights)))
            lives.update({path: leaf for path, leaf in _by_path(read['variables']).items() if path in unread})

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
