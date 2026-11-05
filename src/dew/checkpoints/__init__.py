"""Save and restore a run's training state and data position with Orbax.

A checkpoint holds every train-state field `TrainState` marks checkpointed
(`dew.state.CHECKPOINTED`): the clocks, the variables, the optimizer
state, the EMA, the key, the loss scaler and a partial accumulation window,
and, when the data iterator can report one, `position`. The narrow compute
copies derive from the variables and are rebuilt on resume. A position
is either global, which any partition of the data can read, or one share's own
offset, which only a reader of that same share can read; `dew._position` marks
the difference and `read_position` acts on it. A state that is not a training
run's, such as a simulation's, is saved as a mapping of arrays and restored
through a mapping template, with the same asynchronous writes, retention and
step index.

The EMA copy is stored as its difference from the weights, the run's and each
DiLoCo group's (`_averages_held`). Each EMA leaf with
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

A host that lost its local copy (replaced, its disk wiped, or killed before its
write landed) recovers it from the replicas that hold it, as Orbax's emergency
manager restores a slice from a surviving one: where every shard such a process
held has a copy on a process that still holds the newest local step (the
data-parallel replicas of a layout whose fsdp axis stays within hosts), the
newest local step is still the pool's (`_local_latest`), the others restore it
from their disks, and each shard reaches the process that lost it from one
replica, over the mesh's own collectives (`_from_replicas`). A layout whose
shards have no replica on another host sends the pool to the persistent
checkpoint, as before.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import functools
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, overload, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from etils import epath
from orbax.checkpoint.checkpoint_manager import AsyncOptions, MultiprocessingOptions
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation

from dew import _records
from dew._cleanup import attempted
from dew._coordination import broadcast_from_process_zero, from_every_process, seen_by_every_process
from dew._files import write_atomically
from dew._hashing import fmix32
from dew._records import JSON, duration, json_value, record, share, text
from dew.batch import PARAMETER_ROOTS, Variables
from dew.checkpoints._step_records import counted_processes, gather_positions, read_position, step_record
from dew.checkpoints.served import unpacked
from dew.mesh import LogicalAxes, mesh_axes
from dew.nn import sharding
from dew.nn.kernels.options import owned_program, program
from dew.nn.precision import at_least_fp32
from dew.telemetry.profile import region

if TYPE_CHECKING:
    import optax
    from flax.training.dynamic_scale import DynamicScale

    from dew.data.dataset import DataPartition
    from dew.inference.tasks import SavedRun
    from dew.nn.protocols import ProjectionGroup
    from dew.registry import ModelConfig
    from dew.state import Accumulation, TrainState

# One field of the train state as a checkpoint holds it: a scalar array, a
# tree of arrays, or one of the records the state carries beside them. The
# gathered data position rides along under its own name as uint8 rows.
type StateLeaf = (jax.Array | np.ndarray | Variables | optax.OptState
                  | DynamicScale | Accumulation | None)

Resume = Literal["auto", "require", "never"]
"""Whether a run continues its directory's latest checkpoint (`Checkpoints.resumed`)."""

FROZEN_STORE = "frozen"
"""The directory beside the steps that holds every variables collection but
`params` once, the run's and each DiLoCo group's (`_variable_trees`), under the
digest of its content (`_digest`). A step records
each collection's digest and holds `params` and the optimizer state itself,
so weights no step moves (a LoRA run's frozen base, a pipeline's towers) are
written once a run, and a collection a step does move (batch statistics) is
written again only when it changes. A digest no kept step records is
deleted. The step names it relative to the run directory, so a copied run
still restores."""

RUN_FILE = "run.json"
COOLDOWN_FILE = "cooldown.json"
"""The cooldown branch a run began at a step and has not finished (`Checkpoints.cooldown`), kept beside
the step's checkpoint, which it outlives: a resume from that step runs it first."""
"""The one file at the root of every run directory, beside its steps, that
makes it a run: what `dew.pipeline` looks for, locally and on the Hub, and
what `dew.io.publish` ships with a step. A run built from a config holds the
class record `RunConfig.save` writes, which rebuilds it; a run defined in
Python holds `PYTHON_RUN`, which its first checkpoint writes."""

PYTHON_RUN: dict[str, JSON] = {
    "rebuildable": False,
    "defined": "in Python, without a recorded RunConfig: each checkpoint records the inference "
               "declaration it loads by, and no record rebuilds the run. A run built as a "
               "dew.config.RunConfig in Python, and trained by its run(), writes a run.json that rebuilds it "
               "(docs/recipes.md, Python experiments)",
}
"""The root record of a run defined in Python (`RUN_FILE`), beside which the trainer's checkpoints say
why its record cannot rebuild it (`Checkpoints.python_run`)."""


def _is_profiles(node) -> bool:
    """Whether `node` is a `PowerProfilesState`, or the mapping of its fields
    a restore without a template reads it as."""
    from dew.optim import PowerProfilesState
    return isinstance(node, PowerProfilesState) or (
        isinstance(node, Mapping) and set(node) == set(PowerProfilesState._fields))


def _power_profiles(opt_state):
    """The `PowerProfilesState` inside `opt_state`, or its mapping, or None
    where the solver keeps none."""
    held = [node for node in jax.tree.leaves(opt_state, is_leaf=_is_profiles) if _is_profiles(node)]
    if len(held) > 1:
        raise ValueError(f"the optimizer state holds {len(held)} power_profiles; wrap the solver once")
    return held[0] if held else None


def _profiles_at(tree) -> tuple[TreePath, object] | None:
    """Where a step's state tree (or Orbax's metadata of one) keeps its post-hoc EMA profiles
    (`dew.optim.power_profiles`), and the profiles: in the run's optimizer state, or under DiLoCo in the
    groups', whose configured optimizer it is (`DiLoCo.started`), each field stacked over the groups;
    None where it keeps none."""
    held = [((*holder, 'opt_state'), profiles) for holder, fields in _holders(tree) if 'opt_state' in fields
            for profiles in [_power_profiles(_at(tree, (*holder, 'opt_state')))] if profiles is not None]
    if len(held) > 1:
        raise ValueError("the state keeps power_profiles in both the run's optimizer and its groups'")
    return held[0] if held else None


def _profile_record(profiles) -> dict[str, JSON]:
    """What a step records of the profiles it keeps (`Checkpoints.profile_metadata`): the updates each
    holder's averages have taken, one count, or under DiLoCo each group's, whose rejected updates are its
    own; and their relative standard deviations, every group's alike."""
    from jax.experimental import multihost_utils

    def host(value: jax.Array) -> np.ndarray:
        # A group's clocks are on its own devices, which a pool's processes split.
        return np.asarray(value if value.is_fully_addressable
                          else multihost_utils.process_allgather(value, tiled=True))
    updates, stds = host(profiles.updates), host(profiles.stds)
    return {'updates': int(updates) if updates.ndim == 0 else [int(count) for count in updates],
            'stds': [float(std) for std in (stds if stds.ndim == 1 else stds[0])]}


def _snapshot_weights(held: Sequence[tuple[int, int, tuple[float, ...]]], step: int,
                      std: float) -> dict[tuple[int, int], float]:
    """The weight `coefficients` solves for each snapshot's profile, by the snapshot's step and the
    profile's place, in the post-hoc EMA of relative std `std` at `step`, from one holder's snapshots
    `held` (step, updates, stds). A snapshot before the first update holds the initial weights, with no
    profile to fit; a step that made no update (accumulating a window, or rejected by a loss scaler)
    holds the averages of the step before it, whose identical rows would leave the solve singular: one
    snapshot an update count, the latest."""
    from dew.averaging import coefficients

    counted = [(each, updates, stds) for each, updates, stds in held if updates > 0]
    counted = list({updates: (each, updates, stds) for each, updates, stds in counted}.values())
    if not counted or counted[-1][0] != step:
        raise ValueError(f"the snapshot at step {step} was taken before the first update")
    solved = iter(coefficients([(updates, deviation) for _, updates, stds in counted for deviation in stds],
                               counted[-1][1], std))
    return {(each, index): next(solved) for each, _, stds in counted for index in range(len(stds))}


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


def _handler(**options) -> ocp.PyTreeCheckpointHandler:
    """Orbax's pytree handler as every save of Dew's takes it: moving its arrays to pinned host memory
    with an explicit copy before it reads them, on every backend whose devices have that memory (a
    CPU's do, as a GPU's), which Orbax leaves to GPUs by default. Its implicit read instead would be
    refused under a steady state's transfer guard (tests/steady_state.py), which Dew's JAX build holds
    a CPU array's reads to as a GPU's."""
    return ocp.PyTreeCheckpointHandler(enable_pinned_host_transfer=True, **options)


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


class Envelope(TypedDict):
    """What a checkpoint of a run's state holds on every tier, as `Checkpoints.save` and `save_local` each
    take it, their own metadata beside it: the state as saved, its data position (`saved`), the partition
    it was read by, the stop control, the rung, the artifact, the boxed axes and the rollout's tree
    (`dew.training.Trainer`'s one envelope, row 4126)."""

    state: TrainState
    saved: bytes | None
    share: DataPartition | None
    control: dict | None
    rung: JSON
    artifact: JSON
    boxed: Mapping[tuple[str, ...], LogicalAxes] | None
    rollout: Mapping[str, object] | None


@dataclasses.dataclass(frozen=True)
class Keep:
    """Keeps the union of the latest steps, periodic steps, wall-time-spaced steps and a predicate's steps.

    `interval` keeps checkpoints at least this much wall time apart; it does
    not keep every old checkpoint. `where` receives a retained-step record
    with its metrics. `weights_only` keeps the steps `every`, `interval` and
    `where` keep, beyond the latest, as their model alone (`ROLES`): the newest
    steps stay whole for a resume, and an archive of a long run holds no
    optimizer state it never resumes.
    """
    latest: int = 2
    every: int | None = None
    interval: datetime.timedelta | str | None = None
    where: Callable[[Kept], bool] | None = dataclasses.field(default=None, metadata={'record': False})
    weights_only: bool = False

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

    @property
    def score(self) -> float | None:
        """The value as a lower-is-better score, None where it ranks nothing (a nonfinite value)."""
        if not math.isfinite(self.value):
            return None
        return self.value if self.mode == 'min' else -self.value


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
        from dew.training.evaluation import recorded_name
        return self._values[key if isinstance(key, str) else recorded_name(key, self._values)]

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
    """'state' for a train state, 'weights' for a step that holds its model
    alone (a weights-only snapshot, or a step kept for its model past the
    latest), 'profiles' for a step kept for its post-hoc EMA profiles alone,
    'tree' for a mapping saved in place of a train state."""
    rankings: Mapping[str, dict] = dataclasses.field(default_factory=dict)


def _kept(step: int, metrics: Mapping[str, float], custom: Mapping, items: Sequence[str]) -> Kept:
    """The record of the step saved with `metrics` and `custom` metadata, which holds `items`."""
    rules = custom.get('rankings') or {}
    kind = ('tree' if custom.get('tree') else
            'weights' if custom.get('weights_only') or (MODEL in items and RESUME not in items) else
            'profiles' if tuple(items) == (PROFILES,) else 'state')
    return Kept(step=step, metrics=Metrics(metrics),
                ranked_by=custom.get('primary') or next(iter(rules), None),
                mode=next(iter(rules.values()), {}).get('mode'), kind=kind, rankings=copy.deepcopy(rules))


MODEL, PROFILES, RESUME = "model", "profiles", "resume"
ROLES = (MODEL, PROFILES, RESUME)
"""The Orbax items of a run's persistent step, each the payload of one kind of reader: the model, its
counters, variables and EMA (`dew.state.MODEL_STATE`), which an evaluation, an export and a weights-only
snapshot read; its post-hoc EMA profiles' averages (`dew.optim.power_profiles`), which
`Checkpoints.posthoc_ema` reads; and the rest of its state, which only a resume reads: the optimizer's
state, the key, the loss scale, the window, the position, the rollout and a DiLoCo run's groups'. A step its
retention keeps for some of them alone holds those (`Checkpoints._project`, row 4311)."""

ONE_ITEM = "default"
"""The one Orbax item of a step that holds its whole named state: a step an earlier Dew wrote, and a mapping
saved in a train state's place."""

PROJECTED = "projected.json"
"""The record beside a step's items of those it keeps, written before the rest are deleted, so a step
whose deletion was cut short reads as cut (`_held_items`)."""


def _held_items(step_directory: str) -> tuple[str, ...]:
    """The items the step at `step_directory` holds for its readers (`ROLES`, or `ONE_ITEM`): those its
    projection record keeps (`PROJECTED`), or else the items it has."""
    path = epath.Path(step_directory)
    if (path / PROJECTED).exists():
        return tuple(_records.strings(json.loads((path / PROJECTED).read_text()), PROJECTED))
    return tuple(name for name in (*ROLES, ONE_ITEM) if (path / name).is_dir())


def _item_of(tree: str) -> str:
    """The item that holds a step's variables tree, by its name (`_variable_trees`): the model's and its
    EMA in `model`, a DiLoCo group's in `resume`."""
    from dew.state import MODEL_STATE
    return MODEL if tree.split('/')[0] in MODEL_STATE else RESUME


def _frozen_held(frozen: Mapping[str, Mapping[str, str]],
                 items: Sequence[str]) -> dict[str, Mapping[str, str]]:
    """The store entries of a step's `frozen` record (`FROZEN_STORE`) whose trees the items it holds keep:
    a step cut to its model references its groups' collections no longer."""
    return {tree: names for tree, names in frozen.items() if ONE_ITEM in items or _item_of(tree) in items}


def _step_custom(step_directory: str) -> Mapping[str, object] | None:
    """The custom metadata the step at `step_directory` records, Orbax reading the step's metadata alone,
    opening no manager; None for a directory that is no Orbax step. One whose metadata does not read
    raises, naming it."""
    if not (epath.Path(step_directory) / "_CHECKPOINT_METADATA").exists():
        return None
    try:
        return ocp.metadata.get_step_metadata(step_directory).custom_metadata or {}
    except ValueError as error:
        raise ValueError(f"{step_directory}'s step metadata does not read") from error


def step_artifact(step_directory: str) -> JSON:
    """The inference declaration the step at `step_directory` records (`Checkpoints.artifact`,
    `_step_custom`); None where it records none, holds no model (kept for its profiles alone) or is no
    Orbax step."""
    custom = _step_custom(step_directory)
    if custom is None or not {MODEL, ONE_ITEM} & set(_held_items(step_directory)):
        return None
    return json_value(custom.get("artifact"), "artifact")


def frozen_entries(step_directory: str) -> list[str]:
    """The `FROZEN_STORE` entries the step at `step_directory` records (`_step_custom`) of the items it holds
    (`_frozen_held`), relative to its run directory, which `dew.io.publish` carries beside it; none for a
    directory that is no Orbax step."""
    custom = _step_custom(step_directory)
    frozen = {} if custom is None else _frozen_held(record(custom.get("frozen") or {}, "frozen"),
                                                    _held_items(step_directory))
    digests = {text(digest, "frozen digest") for names in frozen.values()
               for digest in record(names, "frozen").values()}
    return [f"{FROZEN_STORE}/{digest}" for digest in sorted(digests)]


_SEEDS = (0x9E3779B9, 0x7F4A7C15)
"""The two multipliers `_marks` mixes a word's flat index by."""


def _word_type(width: int) -> type:
    """The unsigned type `_marks` reads `width`-byte elements as: their own width up to 2 bytes,
    32-bit words past it (an 8-byte element is two)."""
    return {1: np.uint8, 2: np.uint16}.get(width, np.uint32)


def _mixed(module, words, index):
    """The two marks of 32-bit `words` at their flat positions `index`, in `module` (NumPy on the
    host, jax.numpy on the devices): each word mixed with its index times a seed (`fmix32`) and the
    words summed, wrapping."""
    return module.stack([module.sum(fmix32(words ^ (index * module.uint32(seed))), dtype=module.uint32)
                         for seed in _SEEDS])


@program
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
    # An 8-byte element becomes a trailing pair of 32-bit words.
    bits = jax.lax.bitcast_convert_type(bits, _word_type(np.dtype(bits.dtype).itemsize)).astype(jnp.uint32)
    index = jnp.zeros(bits.shape, jnp.uint32)
    stride = 1
    for axis in reversed(range(bits.ndim)):
        index = index + jax.lax.broadcasted_iota(jnp.uint32, bits.shape, axis) * jnp.uint32(stride)
        stride = stride * bits.shape[axis] % 2 ** 32
    return _mixed(jnp, bits, index)


def _host_marks(leaf: np.ndarray, chunk: int = 1 << 24) -> np.ndarray:
    """`_marks` of a host array, from its stored bytes, `chunk` words at a time: the same mixing
    (`_mixed`) on the host, so a leaf JAX does not hold as stored (float64 with x64 off) is marked
    from its bytes, and no more than a chunk's words are widened at once."""
    array = np.ascontiguousarray(leaf)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    words = array.reshape(-1).view(_word_type(array.dtype.itemsize))
    marks = np.zeros(2, np.uint32)
    with np.errstate(over="ignore"):
        for start in range(0, words.size, chunk):
            bits = words[start:start + chunk].astype(np.uint32)
            index = (np.arange(start, start + bits.size, dtype=np.uint64) & 0xFFFFFFFF).astype(np.uint32)
            marks += _mixed(np, bits, index)
    return marks


def _digest(tree) -> str:
    """The SHA-256 of `tree`'s paths, shapes, dtypes and each leaf's marks (`_marks`), taken of the bytes as
    stored, whatever dtypes JAX allows now. The device leaves are marked in one call a set of devices they
    sit on (a stage-local run's rows sit on their own stage's, `dew.training.stages`). A host array is
    marked on the host, so a float64 leaf restored with x64 off has the digest it was saved with. A device
    leaf in host memory is marked on the devices one leaf at a time, so no more than one is copied off the
    host at once."""
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    on_device: dict[frozenset, list[int]] = {}
    for index, (_, leaf) in enumerate(flat):
        if isinstance(leaf, jax.Array) and leaf.sharding.memory_kind in (None, "device"):
            on_device.setdefault(frozenset(leaf.sharding.device_set), []).append(index)
    marks = {index: mark for indices in on_device.values() for index, mark in zip(
        indices, np.asarray(jax.device_get(_marks([flat[index][1] for index in indices]))), strict=True)}
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
    blind = [index for index, saw in enumerate(seen_by_every_process(directory)) if not saw]
    if blind:
        raise ValueError(
            f"The checkpoint directory {directory} is not shared: process(es) {blind} of "
            f"{jax.process_count()} do not see the file process 0 wrote there. A checkpoint "
            f"is written by every process of the pool and read back by any, so the "
            f"directory has to be one every process reads and writes, on a shared "
            f"filesystem or in a bucket. Each host's own disk can hold local checkpoints "
            f"beside it (local_directory).")


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
    """Return the placement of each array leaf of `tree` as a string, keyed by path:
    its sharding and which slice of it each of this process's devices holds.

    A local checkpoint restores onto this placement and no other. The sharding's
    text names its mesh's axes and sizes but not the order of its devices, and two
    meshes over the same devices in another order give each process other slices."""
    def owned(leaf) -> str:
        slices = leaf.sharding.addressable_devices_indices_map(leaf.shape)
        held = sorted((device.id, tuple((index.start, index.stop) for index in indices))
                      for device, indices in slices.items())
        return f"{leaf.sharding} holding {held}"

    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): owned(leaf)
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


@functools.partial(program, static_argnums=2)
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


type TreePath = tuple[str, ...]
"""Where a tree sits in a step's state tree, field by field: ('variables',), or under DiLoCo a group's,
('inner', 'variables'); a step's metadata names it by its fields joined with '/' (`_named`)."""


def _at(tree, path: TreePath):
    """The subtree of a step's state tree at `path`, whether its nested state is a dataclass (`Replicas`,
    a template's) or the mapping a host restore and Orbax's metadata give; None where a field on the way
    holds none."""
    for name in path:
        if tree is None:
            return None
        tree = tree.get(name) if isinstance(tree, Mapping) else getattr(tree, name)
    return tree


def _with(tree, path: TreePath, value):
    """`tree` with its subtree at `path` (`_at`) replaced by `value`, every container on the way copied."""
    if not path:
        return value
    name, rest = path[0], path[1:]
    if tree is None or isinstance(tree, Mapping):
        return {**(tree or {}), name: _with(None if tree is None else tree.get(name), rest, value)}
    return dataclasses.replace(tree, **{name: _with(getattr(tree, name), rest, value)})


def _named(path: TreePath) -> str:
    return "/".join(path)


def _holders(tree) -> list[tuple[TreePath, tuple[str, ...]]]:
    """Where a step's state tree holds a run's own state, and the fields there: the tree itself, and under
    DiLoCo the groups' (`TrainState.inner`, `dew.state.Replicas`), each field stacked over the groups."""
    from dew.state import CHECKPOINTED, REPLICA_FIELDS
    nested = () if _at(tree, ("inner",)) is None else ((("inner",), REPLICA_FIELDS),)
    return [((), CHECKPOINTED), *nested]


def _variable_trees(tree) -> list[TreePath]:
    """The variables trees a step's state tree holds, each keyed by collection: the variables and the EMA
    (`dew.state.VARIABLE_FIELDS`), the run's and each DiLoCo group's."""
    from dew.state import VARIABLE_FIELDS
    return [(*holder, name) for holder, fields in _holders(tree) for name in VARIABLE_FIELDS
            if name in fields and isinstance(_at(tree, (*holder, name)), Mapping)]


def _averaged(path: TreePath) -> TreePath:
    """The tree whose leaves the average at `path` follows (`dew.state.AVERAGES`): an EMA's weights."""
    from dew.state import AVERAGES
    return (*path[:-1], AVERAGES[path[-1]])


def _averages_held(tree) -> list[TreePath]:
    """The averages a step's state tree holds (`dew.state.AVERAGES`): the run's EMA and each group's."""
    from dew.state import AVERAGES
    return [(*holder, name) for holder, fields in _holders(tree) for name in AVERAGES
            if name in fields and _at(tree, (*holder, name)) is not None]


def _with_ema_deltas(state_tree: dict[str, StateLeaf]) -> tuple[dict[str, StateLeaf], dict[str, list[str]]]:
    """Return `state_tree` with each average's leaves (`_averages_held`), where `_stored_as_delta`
    allows, as byte planes of their difference from the weights they follow, and the paths of those leaves
    in each average's tree, by its name, which the checkpoint's metadata records (`deltas`).

    The planes are computed on the devices the average sits on, sharded as
    it is, so a save holds one more EMA-sized buffer there until orbax has
    copied it to the host. A save runs between steps, when the buffers a
    step holds beside the state, among them a gradient as large as the
    weights, are free.
    """
    recorded = {}
    for path in _averages_held(state_tree):
        paths_leaves, structure = jax.tree_util.tree_flatten_with_path(_at(state_tree, path))
        leaves = [leaf for _, leaf in paths_leaves]
        held = _by_path(_at(state_tree, _averaged(path)))
        lives = [held[keys] for keys, _ in paths_leaves]
        chosen = [index for index, (average, live) in enumerate(zip(leaves, lives, strict=True))
                  if _stored_as_delta(average, live)]
        if not chosen:
            continue
        for index in chosen:
            encode = program(_delta_planes, out_shardings=_planes_sharding(leaves[index].sharding))
            leaves[index] = encode(leaves[index], lives[index])
        state_tree = _with(state_tree, path, jax.tree.unflatten(structure, leaves))
        recorded[_named(path)] = [jax.tree_util.keystr(paths_leaves[index][0]) for index in chosen]
    return state_tree, recorded


def _ema_deltas(metadata: _StepFacts) -> dict[TreePath, dict[jax.tree_util.KeyPath, jax.ShapeDtypeStruct]]:
    """The leaves of each average the checkpoint at `step` stores as byte planes, by the average's path
    and their own, each with the shape and dtype of the weight it is the difference from, which are its
    own. The step's custom metadata records those paths by average (`deltas`); a step written before
    the groups' averages were coded so records its EMA's alone (`ema_deltas`, carried to 1.0), and one
    written before the EMA was, none."""
    stored = dict(_item_metadata(metadata))
    custom = metadata.custom_metadata or {}
    recorded = custom.get('deltas')
    if recorded is None:
        recorded = {'ema': custom.get('ema_deltas') or []}
    deltas = {}
    for name, paths in record(recorded, 'deltas').items():
        path = tuple(name.split('/'))
        average = _at(stored, path)
        wanted = set(_records.strings(paths, 'delta paths'))
        if average is None or not wanted:
            continue
        lives = _by_path(_at(stored, _averaged(path)))
        deltas[path] = {keys: jax.ShapeDtypeStruct(tuple(lives[keys].shape), lives[keys].dtype)
                        for keys, _ in jax.tree_util.tree_flatten_with_path(dict(average))[0]
                        if jax.tree_util.keystr(keys) in wanted}
    return deltas


@runtime_checkable
class _MetadataTree(Protocol):
    """The state tree exposed by Orbax's PyTree metadata wrapper."""

    @property
    def tree(self) -> Variables: ...


def _item_metadata(metadata: _StepFacts) -> Variables:
    """The named state tree a step's metadata holds (`_StepFacts.item_metadata`)."""
    tree = metadata.item_metadata
    if not isinstance(tree, Mapping) or any(not isinstance(key, str) for key in tree):
        raise ValueError("the checkpoint holds no named state metadata")
    return dict(tree)


def _tree_of(metadata) -> Mapping[str, object]:
    """One item's state tree in Orbax's metadata: a dict, or its newer metadata wrapper's `tree`."""
    return metadata.tree if isinstance(metadata, _MetadataTree) else metadata


def _in_items(tree: Mapping, profiles: TreePath | None) -> dict[str, dict]:
    """A step's named state tree, or a restore's template or arguments over one, as its items hold it
    (`ROLES`): the model's fields in `model`; the averages of the profiles its optimizer state keeps at
    `profiles` in `profiles`, their place there left empty; every other field in `resume`; no item of no
    field."""
    from dew.state import MODEL_STATE
    rest = dict(tree)
    items: dict[str, dict] = {MODEL: {}, PROFILES: {}, RESUME: {}}
    held = None if profiles is None else _at(rest, profiles)
    found = None if held is None else _power_profiles(held)
    if found is not None:
        items[PROFILES] = {'averages': tuple(_averages_of(found))}
        rest = _with(rest, profiles, _with_averages(held, ()))
    for name, value in rest.items():
        items[MODEL if name in MODEL_STATE else RESUME][name] = value
    return {item: fields for item, fields in items.items() if fields}


def _from_items(items: Mapping[str, Mapping], profiles: TreePath | None) -> dict:
    """A step's named state tree from what was read of its items (`_in_items`'s inverse): the profiles'
    averages back in their optimizer state where both were read."""
    tree = {name: value for item in (MODEL, RESUME, ONE_ITEM)
            for name, value in (items.get(item) or {}).items()}
    averages = items.get(PROFILES)
    held = None if profiles is None else _at(tree, profiles)
    if averages is not None and held is not None:
        tree = _with(tree, profiles, _with_averages(held, tuple(averages['averages'])))
    return tree


def _profile_path(custom: Mapping[str, object]) -> TreePath | None:
    """Where the step whose custom metadata is `custom` kept its profiles in its state (`_profiles_at`)."""
    profiles = custom.get('profiles')
    at = profiles.get('at') if isinstance(profiles, Mapping) else None
    return None if at is None else tuple(text(at, 'profiles at').split('/'))


def _read(want, dtype) -> ocp.ArrayRestoreArgs:
    """How a stored leaf is read for its template leaf `want` (None where
    there is none): in `dtype`, which Orbax casts it to on the host, where
    `want` places it (a leaf `want` does not place, where it was saved), or
    on the host where there is no template or JAX cannot hold that dtype now
    (a float64 saved with x64 on, read with it off). A step's leaves read in
    the template's dtype; a stored collection's in its own, which its digest
    checks, and `_placed` casts and places them. A key reads as its data."""
    if jnp.issubdtype(dtype, jax.dtypes.prng_key):
        return ocp.ArrayRestoreArgs(sharding=getattr(want, 'sharding', None))
    if want is None or jax.dtypes.canonicalize_dtype(dtype) != np.dtype(dtype):
        return ocp.ArrayRestoreArgs(restore_type=np.ndarray)
    return ocp.ArrayRestoreArgs(sharding=getattr(want, 'sharding', None), dtype=dtype)


def _placed(tree, template):
    """`tree`'s read leaves as `template` holds them, in its dtypes and
    placements. A concrete template leaf the checkpoint lacks comes back
    itself, and a key as its key data, which `request_key` types."""
    def placed(leaf, want):
        if leaf is want or jnp.issubdtype(want.dtype, jax.dtypes.prng_key) or (
                isinstance(leaf, jax.Array) and leaf.dtype == want.dtype):
            return leaf
        value = jnp.asarray(leaf).astype(want.dtype)
        sharding = getattr(want, 'sharding', None)
        return value if sharding is None else jax.device_put(value, sharding)
    return jax.tree.map(placed, tree, template)


def _refuse_reshaped(template, stored, where: str) -> None:
    """Refuse the leaves of `template` that `stored` (Orbax's metadata) holds
    in another shape, by their names; a key is stored as its key data."""
    held = ocp.tree.to_flat_dict(stored)
    wrong = [f"{'/'.join(name)} {tuple(held[name].shape)} for {tuple(want.shape)}"
             for name, want in ocp.tree.to_flat_dict(template).items()
             if name in held and not jnp.issubdtype(want.dtype, jax.dtypes.prng_key)
             and tuple(held[name].shape) != tuple(want.shape)]
    if wrong:
        raise ValueError(f"The checkpoint at {where} holds {', '.join(wrong[:6])}"
                         f"{f' and {len(wrong) - 6} more' if len(wrong) > 6 else ''}: restore it into "
                         f"the shapes it was written with.")


def _paths(tree) -> set[str]:
    return {jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _refuse_lacking(lacking: set[str], where: str) -> None:
    """Refuse a template whose leaves `lacking` the checkpoint at `where` does not hold."""
    if lacking:
        names = sorted(lacking)
        raise ValueError(
            f"The checkpoint at {where} holds no {', '.join(names[:6])}"
            f"{f' or {len(names) - 6} more leaves' if len(names) > 6 else ''}, "
            f"which this run's state has: the checkpoint was written by a model "
            f"or objective without them. Restore it with the model it was "
            f"written with.")


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


RETIRED_RATE = ("learning_rate_scale", "learning_rate_shape")
"""The hyperparameters Dew's optimizer injected before it held a schedule's own rate fields
(`dew.optim.RateState`): a rate's scale and its schedule's shape, two scalars side by side in an
injected state's hyperparameters, which a resume refuses by name rather than as a mismatched tree."""


INJECTED_STATE = ("count", "hyperparams", "inner_state")
"""The fields of Optax's injected state (`optax.inject_hyperparams`), which a checkpoint's metadata
holds as a mapping's keys, as Orbax names a NamedTuple's."""


def _retired_rate(tree) -> bool:
    """Whether a checkpoint's optimizer-state metadata `tree` holds the retired rate: an injected
    state (`INJECTED_STATE`) whose hyperparameters hold both of `RETIRED_RATE`, and not a param group
    or a parameter of either name, which sits in a group's states or a parameter's moments."""
    if isinstance(tree, Mapping):
        held = tree.get("hyperparams") if set(INJECTED_STATE) <= tree.keys() else None
        if isinstance(held, Mapping) and set(RETIRED_RATE) <= held.keys():
            return True
        return any(_retired_rate(value) for value in tree.values())
    return isinstance(tree, list | tuple) and any(_retired_rate(value) for value in tree)


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
    if _retired_rate(metadata.get("opt_state")):
        raise ValueError(
            "the checkpoint's optimizer state holds its rate as a scale times a schedule's shape "
            f"({' and '.join(RETIRED_RATE)}), which Dew's optimizer no longer forms: it holds the "
            "schedule's own rate fields (dew.optim.RateState), so a run cannot resume it. Resume it "
            "with the Dew that wrote it, or start a fresh run from its weights in a directory of its own.")
    from dew.state import CHECKPOINTED, OPTIONAL
    missing = set(CHECKPOINTED).difference(stored, OPTIONAL)
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
        path = tuple(tree.split('/'))
        collections = _at(state_tree, path)
        if isinstance(collections, Mapping):
            collections = dict(collections)
            held[tree] = {name: collections.pop(name) for name in names if name in collections}
            # A template asking for a run's stored collections alone leaves no tree for the step to read.
            if collections or len(path) > 1:
                state_tree[path[0]] = _at(_with(state_tree, path, collections), path[:1])
            else:
                del state_tree[tree]
    return held


def _host_read(*, from_local: bool) -> ocp.ArrayRestoreArgs:
    """How a leaf that no template places is read. It is typed as a host
    array, so Orbax reads no sharding file and warns about none. A local
    checkpoint knows device arrays only, so its leaves land on one of this
    process's devices and come home from there."""
    if from_local:
        return ocp.ArrayRestoreArgs(sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]))
    return ocp.ArrayRestoreArgs(restore_type=np.ndarray)


def _position_reads(metadata, *, from_local: bool) -> tuple[dict, dict]:
    """The data position's restore template and the args that read it.

    The table's shape follows the process count and the iterator's position,
    so it comes from the checkpoint's own metadata rather than the run's
    template. `read_position` reads the table on the host, so it is read
    there (`_host_read`) wherever the state goes.
    """
    template = jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype),
                            dict(metadata['position']))
    return template, jax.tree.map(lambda _: _host_read(from_local=from_local), template)


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
    from dew.state import CHECKPOINTED, OPTIONAL
    # A field no checkpoint holds (the narrow copies) starts over: the
    # template's are of its own parameters, not the restored ones. An optional one the checkpoint
    # holds none of is None (`OPTIONAL`).
    return template.replace(**restored, **{field.name: field.default for field in dataclasses.fields(template)
                                           if field.name not in CHECKPOINTED},
                            **{name: None for name in OPTIONAL if name not in restored})


def _split_axes(spec: jax.sharding.PartitionSpec) -> list[str]:
    """The mesh axes `spec` splits an array over, sorted."""
    return sorted(axis for entry in spec for axis in mesh_axes(entry))


def _grid(tree: Mapping[str, StateLeaf]) -> JSON:
    """What a local save records of where `tree`'s arrays lie, for a process that lost its copy to
    find each shard it held on a replica (`_covered`): the mesh's axes and its devices' ids in grid
    order, and, by path, the axes each array splits over. None where an array lies elsewhere than on
    one mesh by name (a host-owned state's), which no replica recovers."""
    meshes, split = set(), {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)):
            continue
        if not isinstance(leaf.sharding, jax.sharding.NamedSharding):
            return None
        mesh = leaf.sharding.mesh
        if not isinstance(mesh, jax.sharding.Mesh):
            return None
        grid = mesh.devices
        meshes.add((tuple(mesh.axis_names), tuple(device.id for device in grid.flat), grid.shape))
        split[jax.tree_util.keystr(path)] = _split_axes(leaf.sharding.spec)
    if len(meshes) != 1:
        return None
    ((axes, devices, shape),) = meshes
    return {"axes": list(axes), "shape": list(shape), "devices": list(devices), "split": split}


def _sources(grid: Mapping[str, object], split: Sequence[str], holders: frozenset[int],
             process: Mapping[int, int] | None = None) -> np.ndarray | None:
    """Which device, in `grid`'s order, gives its shard of an array split over the axes `split` to the
    devices that share it (the same coordinates on those axes): the first, by grid order, on a process
    of `holders`, each device's process by its id (`process`, the pool's own by default). None where
    some shard has no copy on any holder."""
    shape = tuple(_records.integers(grid["shape"], "grid shape"))
    axes = _records.strings(grid["axes"], "grid axes")
    if process is None:
        process = {device.id: device.process_index for device in jax.devices()}
    devices = _records.integers(grid["devices"], "grid devices")
    held = np.asarray([process[device] in holders for device in devices], bool).reshape(shape)
    # The axes a shard's copies differ along come first, then each group's devices along the rest.
    shared = [axis for axis, name in enumerate(axes) if name not in split]
    order = [*[axis for axis in range(len(axes)) if axis not in shared], *shared]
    groups = np.transpose(held, order).reshape(-1, math.prod(shape[axis] for axis in shared) if shared else 1)
    if not groups.any(axis=1).all():
        return None
    first = np.zeros_like(groups)
    first[np.arange(len(groups)), groups.argmax(axis=1)] = True
    return np.transpose(first.reshape([shape[axis] for axis in order]), np.argsort(order))


def _covered(grid: Mapping[str, object] | None, holders: frozenset[int],
             process: Mapping[int, int] | None = None) -> bool:
    """Whether the processes of `holders` hold a copy of every shard of every array a local save
    placed over `grid` (`_grid`), so a process that lost its copy recovers it from them; each
    device's process by its id (`process`, the pool's own by default)."""
    if grid is None:
        return False
    split = record(grid["split"], "grid split")
    return all(_sources(grid, _records.strings(axes, "split axes"), holders, process) is not None
               for axes in {tuple(_records.strings(axes, "split axes")) for axes in split.values()})


def _from_replicas(tree, grid: Mapping[str, object], holders: frozenset[int]):
    """`tree`'s arrays, each shard on every device that holds it the bytes its first copy on a
    process of `holders` holds (`_sources`): a process that lost its local copy holds zeros there,
    and receives each shard as the sum, over the axes its copies lie along, of the source's bits and
    the others' zeros, which is the source's bits exactly. Every process calls it alike."""
    masks: dict[tuple[str, ...], jax.Array] = {}

    def recovered(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        placed = leaf.sharding
        assert isinstance(placed, jax.sharding.NamedSharding), "a covered save places by name (`_grid`)"
        mesh, split = placed.mesh, tuple(_split_axes(placed.spec))
        # Every axis the copies lie along, those of one device too, so the sum is replicated as its
        # placement says.
        along = tuple(name for name in mesh.axis_names if name not in split)
        if all(mesh.shape[name] == 1 for name in along):
            return leaf
        if split not in masks:
            sources = _sources(grid, split, holders)
            assert sources is not None, "a recovered step's shards are covered (`_covered`)"
            every = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*mesh.axis_names))
            masks[split] = jax.make_array_from_callback(sources.shape, every, lambda index: sources[index])
        typed = jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        values = jax.random.key_data(leaf) if typed else leaf
        held = values.sharding
        assert isinstance(held, jax.sharding.NamedSharding), "a key's data lies where its key does"
        words = _UNSIGNED[max(values.dtype.itemsize, 1)]

        def sum_of_sources(value, source):
            bits = (value.astype(np.uint8) if value.dtype == np.bool_
                    else jax.lax.bitcast_convert_type(value, words))
            return jax.lax.psum(jnp.where(source.reshape(()), bits, jnp.zeros_like(bits)), along)

        summing = jax.shard_map(sum_of_sources, mesh=mesh, out_specs=held.spec,
                                in_specs=(held.spec, jax.sharding.PartitionSpec(*mesh.axis_names)))
        summed = owned_program(summing)(jax.device_put(values, held.with_memory_kind("device")), masks[split])
        value = (summed != 0 if values.dtype == np.bool_
                 else jax.lax.bitcast_convert_type(summed, values.dtype))
        value = jax.device_put(value, held)
        return jax.random.wrap_key_data(value, impl=jax.random.key_impl(leaf)) if typed else value

    return jax.tree.map(recovered, tree)


def _zeros(leaf):
    """Zeros where a placed template leaf lies, each process's shards its own: the shards a process
    that lost its local copy holds until its replicas give it theirs (`_from_replicas`). A key is
    read as its key data, as a restore reads it (`_placed`); a leaf nothing places is kept."""
    sharding = getattr(leaf, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return leaf
    shape, dtype = tuple(leaf.shape), leaf.dtype
    if jnp.issubdtype(dtype, jax.dtypes.prng_key):
        key_words = jax.eval_shape(lambda: jax.random.key_data(jax.random.key(0)))
        shape, dtype = (*shape, *key_words.shape), key_words.dtype
    # Every shard of a placement by name is one shape.
    shard = sharding.shard_shape(shape)
    return jax.make_array_from_callback(shape, sharding, lambda _: np.zeros(shard, dtype))


def _tree_record(tree) -> JSON:
    """A step's item metadata as JSON: its nesting, and each array's shape and dtype, which a process
    that lost its local copy reads its restore's checks from (`_StepFacts`)."""
    if isinstance(tree, Mapping):
        return {"tree": {str(key): _tree_record(value) for key, value in tree.items()}}
    if isinstance(tree, (list, tuple)):
        return {"list": [_tree_record(value) for value in tree]}
    if tree is None:
        return None
    # A key's own dtype names its implementation, which no check of a restore reads.
    key = jnp.issubdtype(tree.dtype, jax.dtypes.prng_key)
    return {"shape": list(tree.shape), "dtype": "key" if key else str(np.dtype(tree.dtype))}


def _from_tree_record(recorded: JSON):
    """The tree `_tree_record` wrote, its arrays as `jax.ShapeDtypeStruct`s."""
    if recorded is None:
        return None
    entry = record(recorded, "tree record")
    if "tree" in entry:
        nested = record(entry["tree"], "tree")
        return {key: _from_tree_record(json_value(value, key)) for key, value in nested.items()}
    if "list" in entry:
        listed = entry["list"]
        assert isinstance(listed, list), "a list record holds a list"
        return [_from_tree_record(value) for value in listed]
    dtype = _records.text(entry["dtype"], "dtype")
    return jax.ShapeDtypeStruct(tuple(_records.integers(entry["shape"], "shape")),
                                jax.random.key(0).dtype if dtype == "key" else np.dtype(dtype))


@dataclasses.dataclass(frozen=True)
class _StepFacts:
    """A step's metadata as its readers read it (`Checkpoints._step_metadata`): its custom metadata, its
    named state tree's metadata over the items it holds, and those items (`ROLES` or `ONE_ITEM`; none for a
    local step, its own manager's one item, or a local step this process lost, as a process holding it
    read it)."""

    custom_metadata: dict
    item_metadata: Mapping[str, object]
    items: tuple[str, ...] = ()
    trees: Mapping[str, Mapping[str, object]] = dataclasses.field(default_factory=dict)
    """Each item's own metadata, which `item_metadata` merges."""


def _facts(metadata: ocp.metadata.StepMetadata, items: tuple[str, ...] = ()) -> _StepFacts:
    """Orbax's metadata of a step, read over the `items` it holds (`_held_items`), none for a local step."""
    custom = dict(metadata.custom_metadata or {})
    if not items:
        return _StepFacts(custom, _tree_of(metadata.item_metadata))
    held = dict(metadata.item_metadata or {})
    trees = {item: dict(_tree_of(held[item])) for item in items if held.get(item) is not None}
    return _StepFacts(custom, _from_items(trees, _profile_path(custom)), items, trees)


_PAYLOADS = {MODEL: "its model (counters, variables and EMA)", PROFILES: "its post-hoc EMA profiles",
             RESUME: "the rest of its state a resume reads (its optimizer state, key and data position)",
             ONE_ITEM: "its whole state"}
"""How a refusal names each item a step holds or lacks (`ROLES`)."""


def _payload(items) -> str:
    return " and ".join(_PAYLOADS[item] for item in (*ROLES, ONE_ITEM) if item in items)


def _table_record(table: Mapping[str, np.ndarray]) -> JSON:
    return {name: {"dtype": str(np.asarray(value).dtype), "shape": list(np.shape(value)),
                   "hex": np.ascontiguousarray(value).tobytes().hex()} for name, value in table.items()}


def _from_table_record(recorded: JSON) -> dict[str, np.ndarray]:
    table = {}
    for name, value in record(recorded, "table").items():
        entry = record(value, name)
        flat = np.frombuffer(bytes.fromhex(_records.text(entry["hex"], "hex")),
                             np.dtype(_records.text(entry["dtype"], "dtype")))
        table[name] = flat.reshape(_records.integers(entry["shape"], "shape"))
    return table


_NEEDS = {MODEL: frozenset({MODEL}), PROFILES: frozenset({PROFILES}), RESUME: frozenset(ROLES)}
"""The items each kind of reader needs of a step (`ROLES`): a resume needs all three, its model and profiles
among its state."""


class _RankedSteps(preservation.PreservationPolicy):
    """Orbax's retention of a run's steps: each step its rules give a reason to keep, the readers of its
    items (`roles`), which `Checkpoints._project` cuts it to once the save that changed them has landed."""

    def __init__(self, checkpoints):
        self.checkpoints = checkpoints

    def should_preserve(self, checkpoints: Sequence[preservation.PolicyCheckpointInfo], *,
                        context: preservation.PreservationContext) -> Sequence[bool]:
        held = self.roles(checkpoints, context)
        newest = max((checkpoint.step for checkpoint in checkpoints), default=None)
        self.checkpoints._plan = None if newest is None else (newest, held)
        return [bool(held[checkpoint.step]) for checkpoint in checkpoints]

    def roles(self, checkpoints: Sequence[preservation.PolicyCheckpointInfo],
              context: preservation.PreservationContext) -> dict[int, frozenset[str]]:
        """The readers each step is kept for (`ROLES`), as the union of the run's rules: the latest steps a
        resume reads, whole; every `every`-th, wall-time-spaced and predicate step, whole or, under
        `Keep.weights_only`, its model; each ranking's best, whole or, ranked `weights_only`, its model;
        and each post-hoc EMA snapshot, its profiles (row 4311)."""
        owner, keep = self.checkpoints, self.checkpoints.keep
        ordered = sorted(checkpoints, key=lambda checkpoint: checkpoint.step)
        roles: dict[int, set[str]] = {checkpoint.step: set() for checkpoint in ordered}

        def chosen(steps, policy, role):
            for checkpoint, kept in zip(steps, policy.should_preserve(steps, context=context), strict=True):
                if kept:
                    roles[checkpoint.step].add(role)

        kept_whole = MODEL if keep.weights_only else RESUME
        # Orbax's selectors over the steps in order: the latest among those a resume reads and every
        # `every`-th step.
        resumable = [checkpoint for checkpoint in ordered if RESUME in owner._roles_held(checkpoint.step)]
        if keep.latest:
            chosen(resumable, preservation.LatestN(n=keep.latest), RESUME)
        if keep.every:
            chosen(ordered, preservation.EveryNSteps(interval_steps=keep.every), kept_whole)
        ranks = {key for checkpoint in checkpoints for key in checkpoint.metrics or {}
                 if key.startswith('checkpoint/rank/')}
        for rank in ranks:
            name = rank.removeprefix('checkpoint/rank/')
            # Orbax's BestN over the steps a ranking scored, its scores lower-is-better (`reverse`) and the
            # newest first, so that of two tied steps its stable sort keeps the older.
            scored = sorted((checkpoint for checkpoint in checkpoints if rank in (checkpoint.metrics or {})),
                            key=lambda checkpoint: -checkpoint.step)
            best = preservation.BestN(get_metric_fn=lambda metrics, rank=rank: metrics[rank], reverse=True,
                                      n=owner._rank_limits.get(name, 1))
            for checkpoint, kept in zip(scored, best.should_preserve(scored, context=context), strict=True):
                if kept:
                    rule = (owner._custom(checkpoint.step).get('rankings') or {}).get(name) or {}
                    roles[checkpoint.step].add(MODEL if rule.get('weights_only') else RESUME)
        previous = None
        interval = duration(keep.interval) if isinstance(keep.interval, str) else keep.interval
        # Orbax's EveryNSeconds takes whole seconds; Keep.interval is a timedelta, kept exact here.
        for checkpoint in ordered:
            if interval and (previous is None or checkpoint.time - previous >= interval):
                roles[checkpoint.step].add(kept_whole)
                previous = checkpoint.time
            if keep.where:
                try:
                    selected = keep.where(_kept(checkpoint.step, checkpoint.metrics or {},
                                                owner._custom(checkpoint.step),
                                                owner._items(checkpoint.step)))
                except KeyError:
                    selected = False
                if selected:
                    roles[checkpoint.step].add(kept_whole)
            if checkpoint.step in owner._profile_snapshots:
                roles[checkpoint.step].add(PROFILES)
        return {step: frozenset().union(*(_NEEDS[role] for role in held)) & owner._roles_held(step)
                for step, held in roles.items()}


class Checkpoints:
    """Manages the checkpoints of one run in one directory.

    Constructing one opens nothing; the orbax managers are created on first
    use. The directory keeps the latest `keep` steps, so a resume has
    something recent, plus the step with the lowest `loss` metric a save
    reported. A save without metrics can never become the best step.

    `local_directory` names a path on every host's own disk where the run
    keeps one more checkpoint, the latest, which `fit` writes every
    `local_every` steps, or at each share of the run it names, such as `5%`,
    as `fit` resolves its other cadences (`dew.training.budget.every_step`).
    Each process writes the shards its devices hold under its own
    subdirectory, so the same path works for a pod and for one host running
    several processes. `latest` is the newest step every process can
    read, local or persistent, and `restore` reads it from wherever it is. A
    local checkpoint restores onto the placement it was written with, because
    no process holds another process's shards; the persistent checkpoint
    restores onto any mesh.
    """

    def __init__(self, directory: str, *, keep: int | Keep = Keep.latest,
                 local_directory: str | None = None, local_every: int | str | None = None):
        if (local_directory is None) != (local_every is None):
            raise ValueError(
                "local checkpoints take both local_directory and local_every: where "
                "every host writes its copy, and every how many steps")
        if isinstance(local_every, str):
            share(local_every)
        elif local_every is not None and local_every < 1:
            raise ValueError(f"local_every must be at least 1, got {local_every}")
        self.directory = str(location(directory))
        self.keep = Keep(latest=keep) if isinstance(keep, int) else keep
        self._rank_limits: dict[str, int] = {}
        self._rank_modes: dict[str, str] = {}
        self._pending: tuple[int, dict[str, float]] | None = None
        # Each committed step's record, read once: its metrics and custom metadata, which `Kept`
        # views are derived from. A save in flight stages its metadata apart, until it commits.
        self._committed: dict[int, tuple[Mapping[str, float], dict, tuple[str, ...]]] = {}
        self._staged: dict[int, tuple[dict, tuple[str, ...]]] = {}
        # The readers each step was kept for at the last retention, and the newest step then, applied once
        # that step's save has landed (`_project`).
        self._plan: tuple[int, dict[int, frozenset[str]]] | None = None
        self._swept = False
        self.local_directory = None if local_directory is None else str(location(local_directory))
        self.local_every = local_every
        # Why the run these checkpoints keep has no record that rebuilds it (`python_run`).
        self._reasons: tuple[str, ...] = ()
        self._manager = None
        self._local_manager = None
        self._profile_snapshots: set[int] = set()
        self._metadata: tuple[bool, int, _StepFacts] | None = None
        # The newest local step, the processes holding it and where its arrays lie, where some process
        # lost its copy and recovers it from them (`_local_latest`); None where every process holds it.
        self._recovery: tuple[int, frozenset[int], Mapping[str, object]] | None = None
        # The digests the save still in flight records, which no kept step lists yet.
        self._frozen_pending: set[str] = set()

    def _candidates(self, rankings: Sequence[Ranking]) -> tuple[Ranking, ...]:
        eligible = [rank for rank in rankings if rank.score is not None]
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
            score = rank.score
            assert score is not None
            key = f'checkpoint/rank/{rank.metric}'
            held = sorted(scores[key] for scores in retained.values() if key in scores)
            if len(held) < rank.top or score < held[rank.top - 1]:
                candidates.append(rank)
        return tuple(candidates)

    def would_keep(self, ranking: Ranking | Sequence[Ranking]) -> bool:
        """Return whether an evaluation would enter the best-K set of any of its trackers."""
        return bool(self._candidates((ranking,) if isinstance(ranking, Ranking) else ranking))

    def _cache_step(self, step: int, store: ocp.CheckpointManager) -> None:
        """Record `step`'s committed metrics and custom metadata, which its staged save, if any, became: its
        metrics item as `store` reads it (`metrics`) and its own metadata file
        (`ocp.metadata.get_step_metadata`), as `_custom` reads it, not the manager's metadata, which opens
        every array's TensorStore besides (row 3763)."""
        custom = ocp.metadata.get_step_metadata(self.path(step)).custom_metadata
        self._committed[step] = (dict(store.metrics(step) or {}), copy.deepcopy(custom or {}),
                                 _held_items(self.path(step)))
        self._staged.pop(step, None)

    def _custom(self, step: int) -> dict:
        """The custom metadata `step` was saved with: a save in flight's as staged, a committed step's
        as this process read it, or else from its committed metadata, which Orbax reads opening no
        manager."""
        if step in self._staged:
            return self._staged[step][0]
        if step in self._committed:
            return self._committed[step][1]
        return ocp.metadata.get_step_metadata(self.path(step)).custom_metadata or {}

    def _items(self, step: int) -> tuple[str, ...]:
        """The items `step` holds (`_held_items`): a save in flight's as it writes them, a committed step's
        as this process read them."""
        if step in self._staged:
            return self._staged[step][1]
        if step in self._committed:
            return self._committed[step][2]
        return _held_items(self.path(step))

    def _roles_held(self, step: int) -> frozenset[str]:
        """The readers `step`'s items serve (`ROLES`): its items, or for a step of one item (`ONE_ITEM`),
        every reader of its whole state but where it was saved as its model alone, and its profiles where
        it records them."""
        items = self._items(step)
        if ONE_ITEM not in items:
            return frozenset(items)
        custom = self._custom(step)
        if custom.get('weights_only'):
            return frozenset({MODEL})
        return frozenset({MODEL, RESUME, *((PROFILES,) if custom.get('profiles') is not None else ())})

    def committed(self) -> list[int]:
        """The steps whose checkpoints have committed, oldest first, read from the store at each call
        without opening it or writing to it, as `latest` reads it: what a process beside the one
        that trains (an evaluation worker) reads as the run saves, a save in flight listed once it
        commits. `kept` is this process's own view, its retained steps and their metadata."""
        # The name format lists finalized steps alone (a GCS step once its commit marker is written), so
        # nothing here opens the store or reads what this process cached of it.
        return sorted(found.step for found in ocp.step.standard_name_format().find_all(self.directory))

    def kept(self) -> list[Kept]:
        """Return the committed retained steps, oldest first; their immutable metadata is cached.
        The steps are those this process's store saw open or saved itself; a process that reads a
        run another saves into lists its new steps with `committed`."""
        persistent = self._open()
        active = set(persistent.all_steps())
        for step in set(self._committed) - active:
            del self._committed[step]
        retained = []
        for step in sorted(active):
            if step not in self._committed:
                if not self._complete(step):
                    continue
                self._cache_step(step, persistent)
            # A view of the committed record; `_kept` copies the nested rules callers may change.
            retained.append(_kept(step, *self._committed[step]))
        return retained

    def python_run(self, reasons: Sequence[str]) -> None:
        """Why no record rebuilds the run these checkpoints keep, each part of it a run's record cannot
        state, which `PYTHON_RUN` carries where a run defined in Python first saves (`Trainer.fit`
        names them from its own fields) and `RunConfig.read` shows."""
        self._reasons = tuple(reasons)

    def _identify_run(self) -> None:
        """Write `PYTHON_RUN`, with why the run is not rebuildable (`python_run`), at the run's root
        unless a run record is already there (`RUN_FILE`), as a config's run writes its own before it
        trains."""
        root = epath.Path(self.directory)
        if jax.process_index() == 0 and not (root / RUN_FILE).exists():
            root.mkdir(parents=True, exist_ok=True)
            record = {**PYTHON_RUN, "reasons": list(self._reasons)}
            write_atomically(root / RUN_FILE, json.dumps(record, indent=2, sort_keys=True))

    def cooldown(self, step: int) -> Mapping[str, object] | None:
        """The cooldown branch the run began at `step` and has not finished, as it was resolved there
        (`dew.training.cooldown`); None where none is (`COOLDOWN_FILE`)."""
        path = epath.Path(self.directory) / COOLDOWN_FILE
        pending = record(json.loads(path.read_text()), COOLDOWN_FILE) if path.exists() else None
        return pending if pending is not None and pending.get("step") == step else None

    def cooling(self, pending: Mapping[str, object] | None) -> None:
        """Keep `pending` as the cooldown branch the run has begun, or with None that it has none
        unfinished; process zero writes it, beside the steps (`cooldown`)."""
        if jax.process_index() != 0:
            return
        path = epath.Path(self.directory) / COOLDOWN_FILE
        if pending is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(path, json.dumps(dict(pending), sort_keys=True))

    def artifact(self, step: int | str | None = None) -> JSON:
        """Return the selected step's inference declaration, from its custom metadata alone (`_custom`), as
        `control` reads a committed step's: the manager's metadata opens every array's TensorStore besides
        (1.7 s of a U-Net run's reload, which reads it before anything, row 3763); a local step's as its
        process holds it.

        It is None when the step's objective declares no inference record."""
        step = self.pinned(step)
        custom = (self._step_metadata(step, local=True).custom_metadata or {} if step == self._local_latest()
                  else self._custom(step))
        return json_value(custom.get('artifact'), 'artifact')

    def control(self, step: int) -> dict:
        if step == self._local_latest():
            custom = self._step_metadata(step, local=True).custom_metadata or {}
        else:
            self.kept()
            custom = self._custom(step)
        return copy.deepcopy(custom.get('control', {}))

    def _best_step(self, name: str | None) -> int | None:
        retained = self.kept()
        if name is None:
            for checkpoint in reversed(retained):
                custom = self._custom(checkpoint.step)
                if custom.get('primary'):
                    name = custom['primary']
                    break
        if name is None:
            return None
        key = f'checkpoint/rank/{name}'
        candidates = [(checkpoint.metrics[key], checkpoint.step)
                      for checkpoint in retained if key in checkpoint.metrics]
        return min(candidates)[1] if candidates else None

    def pinned(self, step: int | str | None = None, *, reads: str = MODEL) -> int:
        """The exact step `step` selects: itself, the best by a ranking, or, for None or `latest`, the
        newest committed now that holds what a reader of `reads` reads (`ROLES`): a model's reader the
        newest model (a step kept for its model alone among them), a resume the latest (`latest`). A
        later save does not move it."""
        step = self.resolve(step, reads=reads)
        if step is None:
            raise FileNotFoundError(f"{self.directory} holds no checkpoint")
        return step

    def resolve(self, step: int | str | None, *, reads: str = MODEL) -> int | None:
        """The step `step` selects (`pinned`), None where it is None and no step holds what `reads` needs."""
        if step is None or step == 'latest':
            return self._newest(reads)
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
                preservation_policy=_RankedSteps(self), best_fn=_recorded_rank, best_mode='min',
                create=True, enable_async_checkpointing=True)
            # A step's items (`ROLES`), or the one item of a step an earlier Dew wrote or a mapping's.
            self._manager = ocp.CheckpointManager(
                self.directory, options=options,
                item_handlers={name: _handler() for name in (*ROLES, ONE_ITEM)})
            for step in self._manager.all_steps():
                if not self._complete(step):
                    continue
                self._cache_step(step, self._manager)
                custom = self._committed[step][1]
                if custom.get('profiles') is not None:
                    self._profile_snapshots.add(step)
                for name, rule in (custom.get('rankings') or {}).items():
                    self._rank_limits[name] = rule['top']
                    self._rank_modes[name] = rule['mode']
        return self._manager

    def _step_metadata(self, step: int, *, local: bool = False) -> _StepFacts:
        """The most recently inspected committed step, shared by shape and value reads.

        Orbax's metadata opens every array's TensorStore. A restore following
        `stored` needs that same immutable metadata, not a second opening of
        all those arrays. A save invalidates it so a later inspection observes
        the newly committed step. Only one step's metadata is retained. A
        local step this process lost is read as the first process holding it
        read it, which every process hands on alike.
        """
        held = self._metadata
        if held is not None and held[:2] == (local, step):
            return held[2]
        metadata: _StepFacts
        recovery = self._recovery
        if local and recovery is not None and recovery[0] == step:
            def read() -> JSON:
                own = _facts(self._open_local().metadata(step))
                return {"custom": own.custom_metadata, "tree": _tree_record(_item_metadata(own))}

            recorded = record(self._from_holder(recovery[1], read), "recovered step")
            metadata = (_StepFacts(dict(record(recorded["custom"], "custom metadata")),
                                   record(_from_tree_record(json_value(recorded["tree"], "tree")),
                                          "item metadata"))
                        if self._lost(step) else _facts(self._open_local().metadata(step)))
        elif local:
            metadata = _facts(self._open_local().metadata(step))
        else:
            metadata = _facts(self._open().metadata(step), self._items(step))
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
                item_handlers=_handler(
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
        steps = from_every_process(self._open_local().latest_step())
        self._recovery = None
        if None not in steps and len(set(steps)) == 1:
            return steps[0]
        newest = max((step for step in steps if step is not None), default=None)
        if newest is None:
            return None
        # A process that lost its copy recovers it where the processes holding it hold a copy of
        # every shard it held (`_covered`).
        holders = frozenset(index for index, step in enumerate(steps) if step == newest)
        recorded = self._from_holder(holders, lambda: (self._open_local().metadata(newest).custom_metadata
                                                       or {}).get("grid"))
        grid = None if recorded is None else record(recorded, "grid")
        if grid is None or not _covered(grid, holders):
            return None
        self._recovery = newest, holders, grid
        return newest

    def _from_holder(self, holders: frozenset[int], read: Callable[[], JSON]) -> JSON:
        """What `read` gives on the first of `holders`, the processes holding the local step a process
        lost, handed to every process (`from_every_process`); every process calls it alike."""
        return from_every_process(read() if jax.process_index() == min(holders) else None)[min(holders)]

    def _lost(self, step: int) -> bool:
        """Whether this process lost its copy of local `step`, which it recovers from the processes
        that hold it (`_local_latest`)."""
        return self._recovery is not None and self._recovery[0] == step and (
            jax.process_index() not in self._recovery[1])

    def _complete(self, step: int) -> bool:
        if step in self._committed:
            return step in self._open().all_steps()
        path = epath.Path(self.path(step))
        return path.exists() and ocp.utils.is_checkpoint_finalized(path)

    @property
    def latest(self) -> int | None:
        """Return the newest committed step a resume can read, local or persistent (`_newest`)."""
        return self._newest(RESUME)

    def _newest(self, reads: str) -> int | None:
        """The newest committed step that holds what a reader of `reads` reads (`ROLES`), local or
        persistent: a local step holds a whole state.

        Asking opens and writes nothing: until this process writes, Orbax's
        step format lists the committed steps read-only, and once it has, its
        manager's view, which holds this process's own saves, does."""
        steps = (self._manager.all_steps() if self._manager is not None
                 else [found.step for found in ocp.step.standard_name_format().find_all(self.directory)])
        found = (step for step in sorted(steps, reverse=True)
                 if self._complete(step) and reads in self._roles_held(step))
        persistent = next(found, None)
        local = self._local_latest()
        if persistent is None or local is None:
            return local if persistent is None else persistent
        return max(persistent, local)

    def resumed(self, resume: Resume = "auto") -> int | None:
        """The step a run continues under `resume`, the `latest`, or None where it starts afresh:
        "auto" continues one where the directory holds it; "require" refuses a directory with none,
        so a job that must continue never trains from scratch; "never" refuses one that holds a run,
        which a run started afresh would write beside."""
        latest = self.latest
        if resume == "require" and latest is None:
            raise ValueError(f"resume='require' continues a run, and {self.directory} holds no checkpoint "
                             "to continue")
        if resume == "never" and latest is not None:
            raise ValueError(f"resume='never' starts a run afresh, and {self.directory} holds one at step "
                             f"{latest}; give the new run a directory of its own")
        return latest

    @property
    def best(self) -> int | None:
        return self._best_step(None)

    def path(self, step: int) -> str:
        return str(epath.Path(self.directory) / str(step))

    def source(self, step: int) -> str:
        """Return the directory `restore` reads `step` from.

        That is this process's local directory when the step is the local one
        every process holds, and the persistent directory otherwise; a local step
        this process lost is read from the first process's copy that holds it.
        """
        if step != self._local_latest():
            return self.directory
        if self._lost(step) and self._recovery is not None:
            holder = epath.Path(self.local_directory or '') / f'process{min(self._recovery[1])}'
            return f"{holder} (a replica)"
        return self.local_path

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
        boxed: Mapping[tuple[str, ...], LogicalAxes] | None = None,
        rollout: Mapping[str, object] | None = None,
    ) -> None:
        """Write `state` under `step`, asynchronously, a train state with its rollout's tree of arrays where
        it holds one (`rollout`), which `rollout_state` reads back.

        `state` is a run's `TrainState`, or a mapping of arrays in its place
        for a state that is not a training run's, such as a simulation's,
        which has no optimizer, average or loss scale. `boxed` is the axes the
        model's modules box on their own parameters (`dew.nn.sharding.boxed_axes`),
        which the state holds unboxed; `variables` places with them. A mapping is
        written as it is, with `metrics`, `ranking` and `control` as a train state's are,
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
        self._project()
        mapping = isinstance(state, Mapping)
        if not mapping:
            self._identify_run()
        if mapping:
            if (saved is not None or share is not None or weights_only or rung is not None
                    or artifact is not None or rollout is not None):
                raise ValueError("a mapping is saved as it is: it takes no data position, share, "
                                 "weights-only split, rung, artifact or rollout, which a train state's save "
                                 "records")
            profiles, state_tree, frozen, deltas = None, dict(state), {}, {}
            items = {ONE_ITEM: state_tree}
        else:
            if weights_only and rollout is not None:
                raise ValueError("a weights-only snapshot resumes nothing, so it holds no rollout's state")
            state_tree = self._item(state, saved, share, rollout)
            profiles = None if weights_only else _profiles_at(state_tree)
            state_tree, frozen = self._frozen(state_tree)
            state_tree, deltas = _with_ema_deltas(state_tree)
            items = _in_items(state_tree, None if profiles is None else profiles[0])
            if weights_only:
                items = {MODEL: items[MODEL]}
                frozen = _frozen_held(frozen, (MODEL,))
        profile_metadata = None if profiles is None else {**_profile_record(profiles[1]),
                                                          'at': _named(profiles[0])}
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
            if rank.score is not None:
                scores[f'checkpoint/rank/{rank.metric}'] = rank.score
                rules[rank.metric] = dataclasses.asdict(rank)
        custom = {
            **step_record(deltas, control, rung, artifact, boxed),
            "profiles": profile_metadata,
            "rankings": rules,
            "primary": primary or (rankings[0].metric if rankings else None),
            "weights_only": weights_only,
            "frozen": frozen,
            "tree": mapping,
        }
        # The retention policy reads this step's record without waiting for its metadata to commit;
        # a save Orbax refuses leaves the step's committed record as it was, and one it takes replaces it.
        self._staged[step] = (copy.deepcopy(custom),
                              tuple(name for name in (*ROLES, ONE_ITEM) if name in items))
        try:
            with region("checkpoint.submit"):
                persistent.save(step, args=ocp.args.Composite(**{
                    name: ocp.args.PyTreeSave(tree) for name, tree in items.items()}),
                                metrics=scores, force=True, custom_metadata=custom)
        except BaseException:
            self._staged.pop(step)
            raise
        self._committed.pop(step, None)
        self._pending = (step, scores)
        self._frozen_pending = {digest for held in frozen.values() for digest in held.values()}
        # Pinned-host state is written before returning; the docstring says why.
        if _written_in_place(state_tree):
            with region("checkpoint.write_in_place"):
                persistent.wait_until_finished()

    def _project(self, manager: ocp.CheckpointManager | None = None) -> None:
        """Cut each committed step to the items its readers read, as the retention that ran with the newest
        save decided (`_RankedSteps.roles`, row 4311), once that save has landed: as Orbax deletes a step
        no reader keeps only once the save that dropped it commits (its `_finalize`), so a save that fails
        leaves every step as it was. Process 0 records a step's kept items beside them (`PROJECTED`),
        then deletes the rest, and hands every process the steps it cut; a step whose deletion was cut
        short reads as cut, and its leftovers go at the first projection of a later process. It runs where
        every process of the pool does, in `save`, `wait` and `close`, before the store's collection
        reads the steps' references (`_frozen`)."""
        manager = self._manager if manager is None else manager
        plan, self._plan = self._plan, None
        if manager is None or plan is None:
            return
        manager.wait_until_finished()
        newest, roles = plan
        cuts: dict[str, list[str]] = {}
        if jax.process_index() == 0 and self._complete(newest):
            steps = set(manager.all_steps())
            for step, needed in roles.items():
                items = self._items(step)
                if not needed or step not in steps or ONE_ITEM in items or not self._complete(step):
                    continue
                root = epath.Path(self.path(step))
                kept = [name for name in ROLES if name in needed and name in items]
                leftover = not self._swept and any((root / name).exists() for name in ROLES
                                                   if name not in kept)
                if len(kept) == len(items) and not leftover:
                    continue
                write_atomically(root / PROJECTED, json.dumps(kept))
                for name in ROLES:
                    if name not in kept and (root / name).exists():
                        (root / name).rmtree()
                cuts[str(step)] = kept
        cuts = broadcast_from_process_zero(cuts)
        self._swept = True
        for step, kept in cuts.items():
            if int(step) in self._committed:
                metrics, custom, _ = self._committed[int(step)]
                self._committed[int(step)] = metrics, custom, tuple(kept)
        if cuts:
            self._metadata = None

    def _frozen(self, state_tree: dict[str, StateLeaf]
                ) -> tuple[dict[str, StateLeaf], dict[str, dict[str, str]]]:
        """Return `state_tree` with the collections but `params` of each of its variables trees
        (`_variable_trees`: the variables and the EMA, the run's and each DiLoCo group's, stacked) held
        in the store (`FROZEN_STORE`), and each one's digest under its tree's name, which the step records.

        An EMA collection no average moves has its weights' content, so it is
        the same entry. A collection the store lacks is written now, before
        the step, by every process, as process 0 finds the store. A stored
        collection that no kept step, this step nor the one still in flight
        records is deleted first, by process 0.
        """
        held: dict[str, dict[str, object]] = {}
        for path in _variable_trees(state_tree):
            collections = _at(state_tree, path)
            held[_named(path)] = {str(name): value for name, value in collections.items()
                                  if name != 'params' and jax.tree.leaves(value)}
        digests = {tree: {name: _digest(value) for name, value in values.items()}
                   for tree, values in held.items()}
        root = epath.Path(self.directory) / FROZEN_STORE
        recorded = {digest for values in digests.values() for digest in values.values()}
        if jax.process_index() == 0 and root.exists():
            referenced = {*recorded, *self._frozen_pending}
            for kept in self.kept():
                frozen = _frozen_held(self._custom(kept.step).get('frozen') or {}, self._items(kept.step))
                for values in frozen.values():
                    referenced.update(values.values())
            for path in root.iterdir():
                if path.name not in referenced and ocp.utils.is_checkpoint_finalized(path):
                    path.rmtree()
        listed = root.exists() and jax.process_index() == 0
        stored = set(broadcast_from_process_zero([path.name for path in root.iterdir()] if listed else []))
        written: set[str] = set()
        for tree, values in held.items():
            for name, value in values.items():
                digest = digests[tree][name]
                if digest not in written and digest not in stored:
                    with region("checkpoint.frozen"), ocp.Checkpointer(_handler()) as checkpointer:
                        checkpointer.save(root / digest, args=ocp.args.PyTreeSave(value))
                    written.add(digest)
        kept_tree = dict(state_tree)
        for tree, values in held.items():
            path = tuple(tree.split('/'))
            kept_tree = _with(kept_tree, path, {name: value for name, value in _at(state_tree, path).items()
                                                if name not in values})
        return kept_tree, digests

    def _with_frozen(self, restored: dict, frozen: Mapping[str, Mapping[str, str]],
                     held: Mapping[str, Mapping] | None) -> dict:
        """`restored` with each collection the step records in the store read
        back into its tree, onto `held`'s templates (`_held_apart`) or, with
        none, as host arrays."""
        for tree, names in frozen.items():
            path = tuple(tree.split('/'))
            templates = None if held is None else held.get(tree, {})
            collections = _at(restored, path)
            if collections is None and templates:
                collections = {}
            if isinstance(collections, Mapping):
                wanted = names if templates is None else {name: names[name] for name in templates}
                restored = _with(restored, path, {**collections, **self._frozen_values(wanted, templates)})
        return restored

    def _frozen_values(self, digests: Mapping[str, str], templates: Mapping | None) -> dict[str, Variables]:
        """Read each collection a step records from the store, onto its
        template's placement and dtype or as host arrays, refusing one whose
        content no longer has the digest the step recorded."""
        values = {}
        with ocp.PyTreeCheckpointer() as checkpointer:
            for name, digest in digests.items():
                path = epath.Path(self.directory) / FROZEN_STORE / digest
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path} is gone: the step records its {name} collection there, so the run "
                        f"directory was copied or published without its {FROZEN_STORE}/ directory")
                stored = dict(_item_metadata(checkpointer.metadata(path)))
                template = None if templates is None else templates.get(name)
                # A template may ask for part of the collection, as a bank reader does
                # one layer of a frozen base; the part is read through the same partial
                # restore a step's own leaves take, and only a whole read can check the
                # content against the digest the step recorded.
                whole = template is None or (jax.tree.structure(template) == jax.tree.structure(stored))
                if template is not None and not whole:
                    from dew.inference.banks import narrowed
                    kept = narrowed(template, stored)
                    _refuse_lacking(_paths(template) - _paths(kept), f"{path} ({name})")
                    stored, template = narrowed(stored, template), kept
                if template is not None:
                    _refuse_reshaped(template, stored, f"{path} ({name})")
                restore_args = (
                    jax.tree.map(lambda meta: _read(None, meta.dtype), stored) if template is None
                    else jax.tree.map(lambda meta, want: _read(want, meta.dtype), stored, template))
                tree = checkpointer.restore(path, args=ocp.args.PyTreeRestore(
                    item=stored, restore_args=restore_args, partial_restore=not whole))
                if whole:
                    found = _digest(tree)
                    if found != digest:
                        raise ValueError(f"{path} holds content with digest {found}, not the {digest} the "
                                         f"step recorded for its {name} collection")
                if template is not None:
                    tree = _placed(tree, template)
                values[name] = tree
        return values

    def profile_steps(self) -> list[int]:
        """Return the complete checkpoints that hold post-hoc EMA snapshots, oldest first."""
        persistent = self._open()
        return sorted(step for step in self._profile_snapshots if step in persistent.all_steps()
                      and epath.Path(self.path(step)).exists()
                      and ocp.utils.is_checkpoint_finalized(self.path(step))
                      and PROFILES in self._roles_held(step))

    def profile_metadata(self, step: int) -> tuple[int, tuple[float, ...]]:
        """Return a snapshot's recorded updates and relative standard deviations, without its averages: the
        run's updates, under DiLoCo the first group's, whose clocks are the run's (`_profile_updates`)."""
        return self._profile_updates(step)[0][0], self._profile_updates(step)[1]

    def _profile_updates(self, step: int) -> tuple[tuple[int, ...], tuple[float, ...], bool]:
        """A snapshot's recorded updates of each holder of its profiles, one, or under DiLoCo each group's
        (`_profile_record`), their stds, and whether they are the groups', stacked over them."""
        self.kept()
        profiles = self._custom(step).get('profiles')
        if profiles is None:
            raise ValueError(f"the checkpoint at step {step} holds no post-hoc EMA snapshot")
        updates = profiles['updates']
        grouped = isinstance(updates, list)
        return (tuple(int(count) for count in updates) if grouped else (int(updates),),
                tuple(profiles['stds']), grouped)

    def restore_profiles(self, step: int) -> tuple[Variables, ...]:
        """Read only a snapshot's averages, as host arrays, from its retained checkpoint: under DiLoCo each
        group's, stacked over the groups on a first axis (`_profiles_at`)."""
        self.profile_metadata(step)
        facts = self._step_metadata(step)
        host = _host_read(from_local=False)
        if PROFILES in facts.trees:
            held = facts.trees[PROFILES]
            restored = self._restored(step, None, {PROFILES: jax.tree.map(lambda _: host, held)}, local=False,
                                      partial=False, items=True)
            return tuple(restored[PROFILES]['averages'])
        # A step of one item, as an earlier Dew wrote it: its averages where its optimizer state keeps them.
        metadata = _item_metadata(facts)
        found = _profiles_at(metadata)
        if found is None:
            raise ValueError(f"the checkpoint at step {step} holds no post-hoc EMA profiles")
        path, profiles = found
        wanted = _with({}, path, _with_averages(jax.tree.map(lambda _: ocp.PLACEHOLDER, _at(metadata, path)),
                                                tuple(_averages_of(profiles))))
        restored = self._restored(step, wanted, jax.tree.map(lambda _: host, wanted), local=False)
        return tuple(_averages_of(_power_profiles(_at(restored, path))))

    def posthoc_ema(self, std: float, step: int | None = None) -> Variables:
        """Return the post-hoc EMA with relative standard deviation `std` at checkpoint `step`.

        `step` defaults to the latest snapshot. The result is host arrays in
        the params' structure and dtypes. It sums every snapshot up to `step`,
        from every tracked profile, with the weights `coefficients` solves for,
        reading one snapshot at a time and accumulating in fp32 or wider. Use
        the result where the run's params go:
        `merge(variables, {"params": checkpoints.posthoc_ema(...)})`.

        Under DiLoCo the profiles are each group's, of the parameters its inner
        solver makes (`DiLoCo.started` moves the configured optimizer into the
        groups), and a group's rejected updates are its own, so each group's
        weights are solved from its own update counts: the result is the groups'
        mean of their own reconstructions, as the groups' EMAs average each its
        own group's parameters, not an average of the outer model's trajectory,
        which no profile follows.
        """
        steps = self.profile_steps()
        if not steps:
            raise FileNotFoundError(f"{self.directory} holds no EMA profile snapshots; train with "
                                    f"OptimConfig.ema_profiles to keep them")
        step = steps[-1] if step is None else step
        if step not in steps:
            raise ValueError(f"{self.directory} holds EMA profile snapshots at steps {steps}, not {step}")
        held = [(each, *self._profile_updates(each)) for each in steps if each <= step]
        grouped = held[-1][3]
        weights = [_snapshot_weights([(each, updates[group], stds) for each, updates, stds, _ in held],
                                     step, std) for group in range(len(held[-1][1]))]
        total, dtypes = None, None
        for each, *_ in held:
            if not any((each, 0) in solved for solved in weights):
                continue
            for index, average in enumerate(self.restore_profiles(each)):
                factors = [solved.get((each, index), 0.0) for solved in weights]

                def weighted(leaf, factors=factors):
                    # Each group's weight on its own row of the stacked average.
                    return (np.reshape(np.asarray(factors), (-1,) + (1,) * (leaf.ndim - 1)) if grouped
                            else factors[0])
                if total is None:
                    dtypes = jax.tree.map(lambda leaf: leaf.dtype, average)
                    total = jax.tree.map(lambda leaf: weighted(leaf) * leaf.astype(at_least_fp32(leaf.dtype)),
                                         average)
                else:
                    total = jax.tree.map(lambda sum_, leaf: sum_ + weighted(leaf) * leaf, total, average)
        if grouped:
            total = jax.tree.map(lambda leaf: np.mean(leaf, axis=0), total)
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
        boxed: Mapping[tuple[str, ...], LogicalAxes] | None = None,
        rollout: Mapping[str, object] | None = None,
    ) -> None:
        """Write `state` under `step` to this process's local directory, asynchronously, with its
        rollout's tree where it holds one (`rollout`, replicated: every process writes it whole).

        The new step replaces the previous local step. As with `save`, a state
        with arrays in pinned host memory is written before this returns. The
        placement is saved with the state, so a resume onto a different
        placement raises before it reads shards from directories that do not
        hold them.
        """
        state_tree = self._item(state, saved, share, rollout)
        written = placement(state_tree)
        grid = _grid({name: leaf for name, leaf in state_tree.items() if name != 'position'})
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
                    **step_record(deltas, control, rung, artifact, boxed),
                    "processes": jax.process_count(),
                    "placement": written,
                    "grid": grid,
                },
            )
        if _written_in_place(state_tree):
            with region("checkpoint.write_in_place"):
                local.wait_until_finished()

    @staticmethod
    def _item(state: TrainState, saved: bytes | None, share: DataPartition | None,
              rollout: Mapping[str, object] | None = None) -> dict[str, StateLeaf]:
        from dew.state import CHECKPOINTED, without_empty_window
        state = without_empty_window(state)
        state_tree = {name: getattr(state, name) for name in CHECKPOINTED}
        if rollout is not None:
            state_tree['rollout'] = dict(rollout)
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
        snapshot = self._step_metadata(step, local=step == self._local_latest())
        return json_value((snapshot.custom_metadata or {}).get('rung'), 'rung')

    def boxed(self, step: int | str | None = None) -> dict[tuple[str, ...], LogicalAxes]:
        """The axes the model at `step` boxed on its own parameters, as `save`
        recorded them; a checkpoint saved without them has none."""
        resolved = self.pinned(step)
        snapshot = self._step_metadata(resolved, local=resolved == self._local_latest())
        recorded = (snapshot.custom_metadata or {}).get('boxed') or []
        if not isinstance(recorded, list):
            raise ValueError(f"step {resolved}'s boxed axes are a list of path and names records, "
                             f"got {recorded!r}")
        table = {}
        for entry in recorded:
            entry = _records.record(entry, 'boxed')
            names = entry['names']
            if not isinstance(names, list):
                raise ValueError(f"step {resolved}'s boxed names are a list, got {names!r}")
            table[_records.strings(entry['path'], 'boxed path')] = tuple(
                None if name is None else _records.text(name, 'boxed name') for name in names)
        return table

    def variables(self, *, step: int | str | None = None, ema: bool | None = None,
                  mesh=None, layout=None, param_dtype: jax.typing.DTypeLike | None = None,
                  parameter_roots: tuple[tuple[str, ...], ...] = PARAMETER_ROOTS,
                  unread: tuple[tuple[str, ...], ...] = ()) -> Variables:
        """Read the selected step's live or averaged variables onto the requested layout.

        Parameter storage conversion applies only to the owner's parameter
        roots; other collections retain their recorded dtypes and placement.
        The subtrees at `unread`'s paths are neither placed nor read
        (`dew._records.UNREAD`).
        """
        from dew.batch import FROZEN, without
        from dew.distributed.placement import (
            Layout as DefaultLayout,
            MeshSpec as DefaultMesh,
            device_mesh as resolve_mesh,
        )
        from dew.inference.banks import banked_placement, placed_rows, shape_rows, spread

        step = self.pinned(step)
        if MODEL not in self._roles_held(step):
            raise ValueError(f"step {step} of {self.directory} holds {_payload(self._items(step))} alone, "
                             "which its retention keeps it for, and no model; read a step that holds one")
        stored = self.stored(step)
        template = {"variables": without(stored["variables"], unread)}
        if ema and stored.get("ema") is None:
            raise ValueError("the run keeps no EMA; request the live policy with ema=False")
        averaged = stored.get("ema") is not None if ema is None else ema
        if averaged:
            template["ema"] = without(stored["ema"], unread)
        device_mesh = resolve_mesh(DefaultMesh() if mesh is None else mesh)
        chosen_layout = DefaultLayout() if layout is None else layout
        # A CPU-owned run stores its frozen collection banked; the layout places
        # its layers, and each bank reads onto its first layer's placement.
        frozen = template["variables"].get(FROZEN)
        rows = template if frozen is None else {
            **template, "variables": {**template["variables"], FROZEN: spread(frozen, shape_rows)}}
        with sharding.boxed(self.boxed(step)):
            placement = {name: chosen_layout.shardings(device_mesh, tree) for name, tree in rows.items()}
            chosen_layout.check(rows["variables"], placement["variables"], device_mesh)
        if frozen is not None:
            placement["variables"] = {**placement["variables"], FROZEN: banked_placement(
                frozen, placement["variables"][FROZEN])}
        params = self.materialized(template, placement, step=step, param_dtype=param_dtype,
                                   parameter_roots=parameter_roots)
        if frozen is not None:
            params = {**params, FROZEN: spread(params[FROZEN], placed_rows)}
        return params

    def materialized(self, template: Mapping[str, Variables], placement: Mapping[str, Variables], *,
                     step: int, param_dtype: jax.typing.DTypeLike | None = None,
                     parameter_roots: tuple[tuple[str, ...], ...] = PARAMETER_ROOTS) -> Variables:
        """The variables at `step` whose leaves `template` names (`stored`'s, under `"variables"`, and
        under `"ema"` the averaged ones, merged over them), restored onto `placement`'s shardings, no
        other leaf read; the floating leaves under `parameter_roots` read as `param_dtype` where one is
        given, the rest in their stored dtypes. Every load of a run's variables restores through it: a
        whole run's onto its layout (`variables`) and a bank's rows onto theirs
        (`dew.inference.banks.CheckpointBanks`)."""
        from dew.batch import merge
        from dew.registry import resolve_dtype

        target = resolve_dtype(param_dtype)
        selected = set()
        if target is not None:
            roots = tuple(tuple(jax.tree_util.DictKey(name) for name in root) for root in parameter_roots)
            selected = {path for path, leaf in jax.tree_util.tree_flatten_with_path(template["variables"])[0]
                        if jnp.issubdtype(leaf.dtype, jnp.floating) and
                        any(path[:len(root)] == root for root in roots)}
        typed = jax.tree_util.tree_map_with_path(
            lambda path, leaf, sharding: jax.ShapeDtypeStruct(
                leaf.shape, target if path[1:] in selected else leaf.dtype, sharding=sharding),
            dict(template), dict(placement))
        values, _ = self.restore(typed, step=step)
        params = values.pop("variables")
        return merge(params, values["ema"]) if "ema" in values else params

    def stored(self, step: int | str | None = None) -> Variables:
        """Return what the checkpoint at `step` holds, without reading its values.

        `step` defaults to the latest. Each state field comes back as a
        shape/dtype tree, and an unset field as None.
        """
        step = self.pinned(step)
        snapshot = self._step_metadata(step, local=step == self._local_latest())
        metadata = _item_metadata(snapshot)
        stored = {name: None if value is None else
                  jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), value)
                  for name, value in dict(metadata).items()}
        for path, deltas in _ema_deltas(snapshot).items():
            stored = _with(stored, path, jax.tree_util.tree_map_with_path(
                lambda keys, leaf, deltas=deltas: deltas.get(keys, leaf), _at(stored, path)))
        frozen = epath.Path(self.directory) / FROZEN_STORE
        with ocp.PyTreeCheckpointer() as checkpointer:
            for tree, names in ((snapshot.custom_metadata or {}).get('frozen') or {}).items():
                path = tuple(tree.split('/'))
                collections = _at(stored, path)
                if isinstance(collections, Mapping):
                    stored = _with(stored, path, {**collections, **{
                        name: jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), dict(
                            _item_metadata(checkpointer.metadata(frozen / digest))))
                        for name, digest in names.items()}})
        return stored

    def accumulation_template(self, step: int):
        """Return the persisted pending-array shapes, a DiLoCo run's groups' stacked, without reading them."""
        from dew.state import CHECKPOINTED, OPTIONAL, Accumulation
        metadata = _item_metadata(self._step_metadata(step, local=step == self._local_latest()))
        missing = set(CHECKPOINTED).difference(metadata.keys(), OPTIONAL)
        if missing:
            raise ValueError(f"training checkpoint lacks required state fields {sorted(missing)}")
        pending = metadata["accumulation"] or (metadata.get("inner") or {}).get("accumulation")
        if pending is None:
            return None
        arrays = jax.tree.map(lambda meta: jax.ShapeDtypeStruct(meta.shape, meta.dtype), dict(pending))
        arrays["statistics"] = tuple(arrays["statistics"])
        arrays["effects"] = tuple(arrays["effects"])
        return Accumulation(**arrays)

    def position(self, step: int, share: DataPartition) -> bytes | None:
        """The data position the reader of `share` resumes from at `step` (`read_position`), read
        alone, before any state is restored: a run opens its data there and reads its first batch
        while its state is still unplaced (`Trainer.fit`). None where the step saved no position."""
        from_local = step == self._local_latest()
        snapshot = self._step_metadata(step, local=from_local)
        metadata = _item_metadata(snapshot)
        if 'position' not in metadata or (snapshot.custom_metadata or {}).get('tree'):
            return None
        def read() -> dict[str, np.ndarray]:
            wanted, reads = _position_reads(metadata, from_local=from_local)
            restored = self._restored(step, {'position': wanted}, {'position': reads}, local=from_local)
            return jax.tree.map(np.asarray, dict(restored['position']))

        recovery = self._recovery
        # A lost step's table is the first holder's, every row of the pool's, handed on.
        table = (_from_table_record(self._from_holder(recovery[1], lambda: _table_record(read())))
                 if from_local and recovery is not None and recovery[0] == step else read())
        return read_position(table, self.source(step), share)

    def rollout_state(self, step: int, template: Mapping[str, object]) -> dict | None:
        """The rollout's tree the step at `step` holds (`save`'s `rollout`), read alone and placed as
        `template`'s leaves (`jax.ShapeDtypeStruct`s) are; None for a step saved without one."""
        from_local = step == self._local_latest()
        metadata = _item_metadata(self._step_metadata(step, local=from_local))
        if 'rollout' not in metadata:
            return None
        # A step saved before the rollout held a key holds the rest, which the rollout reads without it.
        held = metadata['rollout']
        wanted = {'rollout': {key: value for key, value in template.items() if key in held}}
        _refuse_reshaped(wanted, metadata, self.source(step))
        restored = self._restored(step, wanted, jax.tree.map(lambda want: _read(want, want.dtype), wanted),
                                  local=from_local)
        return _placed(dict(restored['rollout']), wanted['rollout'])

    @overload
    def restore[StateT](self, template: StateT, step: int | str | None = None, *,
                        share: DataPartition | None = None,
                        served: Sequence[ProjectionGroup] = ()) -> tuple[StateT, bytes | None]: ...

    @overload
    def restore(self, template: None = None, step: int | str | None = None, *,
                share: DataPartition | None = None,
                served: Sequence[ProjectionGroup] = ()) -> tuple[Variables, bytes | None]: ...

    def restore(self, template=None, step: int | str | None = None, *,
                share: DataPartition | None = None, served: Sequence[ProjectionGroup] = ()):
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

        `served` names the projection groups a step may hold packed where the
        template holds their members (`Objective.served_groups`): each such
        leaf, in any of the state's trees, is read packed and split back into
        its members (`dew.checkpoints.served.unpacked`). Carried to 1.0, when it is deleted (ledger
        1552).
        """
        resumes = template is not None and not isinstance(template, Mapping)
        step = self.pinned(step, reads=RESUME if resumes else MODEL)
        from_local = step == self._local_latest()
        where = self.source(step) if from_local else self.path(step)
        if from_local and template is None and jax.process_count() > 1:
            raise ValueError(
                f"The local checkpoint at {where} holds each process's own shards, "
                f"so a pool cannot read step {step} as host arrays; restore it with "
                f"the template of a run placed as it was written, or read the "
                f"persistent checkpoint at {self.path(step)}")
        snapshot = self._step_metadata(step, local=from_local)
        mapping = bool((snapshot.custom_metadata or {}).get('tree'))
        if resumes and not from_local and RESUME not in self._roles_held(step):
            raise ValueError(f"step {step} of {self.directory} holds {_payload(self._items(step))} alone: a "
                             "weights-only snapshot, or a step kept for its weights or profiles, resumes "
                             "nothing; resume a whole checkpoint")
        metadata = _item_metadata(snapshot)
        stored = metadata.keys()
        _check_template(template, metadata, stored, mapping=mapping)
        frozen = (snapshot.custom_metadata or {}).get('frozen') or {}
        held: dict[str, dict] | None = None
        if template is None:
            untyped = _host_read(from_local=from_local)
            restored = self._restored(step, None, jax.tree.map(lambda _: untyped, dict(metadata)),
                                      local=from_local, partial=False)
            if from_local:
                restored = jax.tree.map(np.asarray, restored)
            restored = dict(restored)
            for path, deltas in _ema_deltas(snapshot).items():
                weights = _by_path(_at(restored, _averaged(path)))
                restored = _with(restored, path, jax.tree_util.tree_map_with_path(
                    lambda keys, leaf, deltas=deltas, weights=weights: _from_delta_planes(leaf, weights[keys])
                    if keys in deltas else leaf, _at(restored, path)))
        elif from_local and self._lost(step):
            restored, held = self._lost_shards(step, template, frozen, snapshot, metadata, where)
        else:
            from dew.state import CHECKPOINTED, OPTIONAL
            state_tree = {name: getattr(template, name) for name in CHECKPOINTED
                          if name in stored or name not in OPTIONAL} \
                if not isinstance(template, Mapping) else dict(template)
            held = _held_apart(state_tree, frozen)
            if from_local:
                self._check_placement(step, state_tree)
            targets, deltas = {}, _ema_deltas(snapshot)
            for path, planes in deltas.items():
                if _at(state_tree, path) is not None:
                    templated, targets[path] = _plane_template(_at(state_tree, path), planes)
                    state_tree = _with(state_tree, path, templated)
            _refuse_reshaped(state_tree, metadata, where)
            restore_args = jax.tree.map(lambda want: _read(want, want.dtype), state_tree)
            if 'position' in stored and not mapping and share is not None:
                state_tree['position'], restore_args['position'] = _position_reads(
                    metadata, from_local=from_local)
            try:
                # partial_restore: a key the checkpoint holds and the template
                # does not is skipped instead of refused. A template asking for
                # stored collections alone asks the step for nothing.
                restored = {} if not state_tree else self._restored(step, state_tree, restore_args,
                                                                     local=from_local)
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
            restored = unpacked(functools.partial(self._restored, step, local=from_local), metadata,
                                dict(restored), served)
            _refuse_lacking({jax.tree_util.keystr(path) for path, leaf
                             in jax.tree_util.tree_flatten_with_path(dict(restored))[0]
                             if isinstance(leaf, jax.ShapeDtypeStruct)}, where)
            restored = {name: value if name == 'position' else _placed(value, state_tree[name])
                        for name, value in restored.items()}
            for path, wanted in targets.items():
                restored = _with(restored, path, self._averages(step, metadata, restored, path, wanted,
                                                                deltas[path], local=from_local))
        if from_local and template is not None:
            restored = self._recovered(step, dict(restored),
                                       positioned='position' in stored and not mapping and share is not None)
        restored = self._with_frozen(dict(restored), frozen, held)
        table = None if mapping else restored.pop('position', None)
        saved = None if table is None or share is None else read_position(table, where, share)
        restored = _filled(template, restored, step)
        return restored, saved

    def _averages(self, step: int, metadata, restored, path: TreePath, targets: dict, deltas: dict, *,
                  local: bool) -> Variables:
        """Return the restored average at `path` with each plane stack of `targets` turned
        back into the leaf it holds, typed and placed as the template has it.

        Undoing a difference takes the weight it was taken from, in its
        stored dtype. The weights the restore has just read serve where the
        template asked for them in that dtype, off pinned host memory; the
        rest are read here, placed as the leaves they undo.
        """
        lives, unread = {}, {}
        weights_path = _averaged(path)
        held = _by_path(_at(restored, weights_path))
        for keys, target in targets.items():
            weight = held.get(keys)
            if (isinstance(weight, jax.Array) and weight.dtype == deltas[keys].dtype
                    and weight.sharding.memory_kind != "pinned_host"):
                lives[keys] = weight
            else:
                sharding = getattr(target, "sharding", None)
                unread[keys] = jax.ShapeDtypeStruct(
                    deltas[keys].shape, deltas[keys].dtype,
                    sharding=None if sharding is None else sharding.with_memory_kind("device"))
        if unread:
            # The stored params tree, its containers kept, with every leaf
            # but the ones to read held back by orbax's placeholder.
            weights = _with({}, weights_path, jax.tree_util.tree_map_with_path(
                lambda keys, _: unread.get(keys, ocp.PLACEHOLDER), _at(dict(metadata), weights_path)))
            read = self._restored(step, weights, jax.tree.map(
                lambda leaf: ocp.ArrayRestoreArgs(sharding=getattr(leaf, "sharding", None)), weights),
                local=local)
            lives.update({keys: leaf for keys, leaf in _by_path(_at(read, weights_path)).items()
                          if keys in unread})

        def average(keys, planes):
            if keys not in targets:
                return planes
            target = targets[keys]
            value = _decode_delta(planes, lives[keys], np.dtype(target.dtype))
            sharding = getattr(target, "sharding", None)
            return value if sharding is None else jax.device_put(value, sharding)
        return jax.tree_util.tree_map_with_path(average, _at(restored, path))

    def _restored(self, step: int, item, restore_args, *, local: bool, partial: bool = True,
                  items: bool = False) -> dict:
        """What `step` holds of its named state as `item` asks (a template of it; None for every leaf
        `restore_args` names), read as `restore_args` say: a local step's one item, or from the persistent
        step's items that hold those leaves (`_in_items`), one of a step an earlier Dew wrote among them
        (`ONE_ITEM`). Every read of a step's arrays goes through it. `items` reads `restore_args`'s own items,
        by name, as they are. A read that needs an item the step does not hold, which its retention cut
        (`_project`), is refused naming what the step holds."""
        if local:
            return dict(self._open_local().restore(step, args=ocp.args.PyTreeRestore(
                item=item, restore_args=restore_args, partial_restore=partial)))
        facts = self._step_metadata(step)
        if ONE_ITEM in facts.items:
            one = ocp.args.PyTreeRestore(item=item, restore_args=restore_args, partial_restore=partial)
            return dict(self._open().restore(step, args=ocp.args.Composite(**{ONE_ITEM: one}))[ONE_ITEM])
        at = _profile_path(facts.custom_metadata)
        args = restore_args if items else _in_items(restore_args, at)
        trees = {} if item is None else _in_items(item, at)
        # An item whose leaves the read holds back (`ocp.PLACEHOLDER`) is not read.
        wanted = [name for name, tree in args.items()
                  if any(leaf is not ocp.PLACEHOLDER for leaf in jax.tree.leaves(trees.get(name, tree)))]
        missing = set(wanted) - set(facts.items)
        if missing:
            raise ValueError(f"step {step} of {self.directory} holds {_payload(facts.items)} alone, which "
                             "its retention keeps it for (Keep, a weights-only ranking, a post-hoc EMA "
                             f"snapshot); it holds no {_payload(missing)}: read a step that holds them")
        if not wanted:
            return {}
        restored = self._open().restore(step, args=ocp.args.Composite(**{name: ocp.args.PyTreeRestore(
            item=trees.get(name), restore_args=args[name], partial_restore=partial) for name in wanted}))
        read = {name: dict(restored[name]) for name in wanted}
        return read if items else _from_items(read, at)

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
                difference = (f"written by {counted_processes(processes or 0)}, and this run has "
                              f"{counted_processes(jax.process_count())}")
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

    def _lost_shards(self, step: int, template, frozen, snapshot: _StepFacts, metadata,
                     where: str) -> tuple[dict, dict | None]:
        """What a process that lost its copy of local `step` restores before its replicas give it
        their shards (`_recovered`): zeros where its template places each leaf, the template checked
        against what the processes holding the step wrote; and the frozen trees held apart."""
        from dew.state import CHECKPOINTED, OPTIONAL
        state_tree = {name: getattr(template, name) for name in CHECKPOINTED
                      if name in metadata or name not in OPTIONAL} \
            if not isinstance(template, Mapping) else dict(template)
        held = _held_apart(state_tree, frozen)
        self._check_recovered(step, state_tree)
        # Each average is stored as byte planes of its difference from its weights, checked as such.
        checked = dict(state_tree)
        for path, planes in _ema_deltas(snapshot).items():
            if _at(checked, path) is not None:
                checked = _with(checked, path, _plane_template(_at(checked, path), planes)[0])
        _refuse_reshaped(checked, metadata, where)
        return jax.tree.map(_zeros, state_tree), held

    def _recovered(self, step: int, restored: dict, *, positioned: bool) -> dict:
        """`restored`, local `step` as each process read it, each shard the first holder's where some
        process lost its copy (`_from_replicas`), and its position table the first holder's where
        `positioned`, which holds every process's row (`gather_positions`); as it is otherwise."""
        recovery = self._recovery
        if recovery is None or recovery[0] != step:
            return restored
        _, holders, grid = recovery
        table = restored.pop('position', None)
        restored = _from_replicas(restored, grid, holders)
        if positioned:
            restored['position'] = _from_table_record(self._from_holder(
                holders, lambda: _table_record(jax.tree.map(np.asarray, dict(table or {})))))
        return restored

    def _check_recovered(self, step: int, state_tree: Mapping[str, StateLeaf]) -> None:
        """Refuse a local step this process lost where this run places the state otherwise than the
        processes holding it wrote it: its replicas hold the shards of that placement alone."""
        recovery = self._recovery
        written = (self._step_metadata(step, local=True).custom_metadata or {})
        if recovery is not None and written.get('processes') == jax.process_count() and (
                _grid(state_tree) == recovery[2]):
            return
        raise ValueError(
            f"This process lost its copy of local step {step}, which the processes "
            f"{sorted(recovery[1]) if recovery else []} hold placed on the mesh, layout and process count "
            f"they wrote it with; this run places the state otherwise. Resume with those, or delete "
            f"{self.local_directory} to resume from the persistent checkpoint in {self.directory}.")

    def close(self) -> None:
        """Wait for pending writes, then close the Orbax managers and the threads and metadata stores
        they hold, each even where another, or its own wait, failed: Orbax reports a failed
        asynchronous write once, from the wait, and its close still releases the manager. The first
        failure is raised with the rest noted on it. A closed `Checkpoints` reopens its managers on
        its next use."""
        attempted(self._closing())

    def __enter__(self) -> Checkpoints:
        return self

    def __exit__(self, kind: type[BaseException] | None, raised: BaseException | None,
                 traceback: TracebackType | None) -> None:
        """`close`, with a failure noted on the error the body raised, which stays the one raised."""
        attempted(self._closing(), primary=raised)

    def _closing(self) -> list[tuple[str, Callable[[], None]]]:
        """The steps that close the open managers, which this forgets: each one's wait, then each
        one's close."""
        opened = [writer for writer in (self._manager, self._local_manager) if writer is not None]
        persistent = self._manager
        self._manager, self._local_manager, self._pending = None, None, None
        cut = ([] if persistent is None
               else [("Checkpoint projection", functools.partial(self._project, persistent))])
        return [*(("Checkpoint wait", writer.wait_until_finished) for writer in opened), *cut,
                *(("Checkpoint close", writer.close) for writer in opened)]

    def wait(self) -> None:
        """Block until pending async writes have landed on disk.

        Saving is async so it stays off the training loop's critical path;
        anything that reads a checkpoint back has to call this first.
        """
        with region("checkpoint.wait"):
            attempted([("Checkpoint wait", writer.wait_until_finished)
                        for writer in (self._manager, self._local_manager) if writer is not None])
            self._pending = None
            self._project()


@dataclasses.dataclass(frozen=True)
class Averaged:
    """The fp32 mean of runs' weights (`averaged`).

    `variables` is the tree their model applies with, its frozen leaves back in `params`; `model` the
    runs' model record with any LoRA adapter merged away, which builds the model the mean applies to;
    and `runs` each run's directory, step and whether its average (EMA) was read."""

    variables: Variables
    model: ModelConfig
    runs: tuple[tuple[str, int, bool], ...]


def averaged(runs: Sequence[str | os.PathLike[str]], *, step: int | str | None = None,
             ema: bool | None = None, mesh=None, layout=None, trust: Sequence[str] = ()) -> Averaged:
    """The fp32 mean of `runs`' effective weights, each run's at `step`, its latest where None
    (`SavedRun.at`, with `trust`), and `ema` as its loaders read them (`averaged_pinned`)."""
    from dew.inference.tasks import SavedRun

    return averaged_pinned([SavedRun.at(str(epath.Path(run)), step, trust) for run in runs], ema=ema,
                           mesh=mesh, layout=layout)


def averaged_pinned(runs: Sequence[SavedRun], *, ema: bool | None = None, mesh=None, layout=None) -> Averaged:
    """The fp32 mean of the pinned `runs`' effective weights (`SavedRun.pinned`), each read with `ema`
    as its loaders read it (its average where None and it keeps one): a model soup of runs trained
    apart from one start, any objective's.

    A run's weights are its model's tree (`SavedRun.variables`, read onto `mesh` as `layout`
    places it, as a run's loaders take them) in fp32 with its frozen leaves back in `params`, a LoRA
    run's factors merged into their kernels first (`Adapter.merge`), as the mean of two runs' factors
    is not the mean of the weights they make: a LoRA run, a partly frozen one and a whole fine-tune of
    one base average. The runs are of one model: their model records, adapters aside, the same, and
    their trees of the same leaves and shapes; any other run is refused (ValueError), naming it. Where
    each run started is the caller's to know: runs drawn from other initial weights average to no
    model of either. The mean is elementwise, so each process takes it over the shards it holds, on
    the host in fp32 (`_taken`), one run's shards beside it at a time, a subnormal kept as float64's
    mean rounds it where a device may flush it, and assembles its leaves as the first run's are
    placed: a pool averages runs no one process holds whole. An integer or boolean leaf must agree
    across the runs."""
    if len(runs) < 2:
        raise ValueError(f"an average is of two or more runs, got {len(runs)}")
    from dew._coordination import agreed
    from dew.distributed.host import place_leaf

    means: list = []
    structure, firsts, model, read = None, [], None, []
    for count, run in enumerate(runs, start=1):
        tree, base, entry = _effective_weights(run, ema, mesh, layout)
        if model is not None and base != model:
            raise ValueError(f"the run in {entry[0]} trains another model than {read[0][0]}'s, so their "
                             "weights are not one model's to average")
        model = base
        leaves, shape = jax.tree.flatten(tree)
        del tree
        if structure is None:
            structure, firsts = shape, [getattr(leaf, "sharding", None) for leaf in leaves]
            means = [_held(leaf, sharding) for leaf, sharding in zip(leaves, firsts, strict=True)]
            del leaves
            read.append(entry)
            continue
        if shape != structure or any(np.shape(leaf) != tuple(held.shape) for held, leaf in zip(
                means, leaves, strict=True)):
            raise ValueError(f"the run in {entry[0]} holds other variables than {read[0][0]}")

        # Every process places the run's leaves as the first run's before any compares them: a placement
        # is a collective, which no process may be left in because a peer refused. Then a process whose
        # shards refuse the run reports it where every process agrees on the run, so none of them reads
        # the next one alone.
        for index, placement in enumerate(firsts):
            leaf = leaves[index]
            if placement is not None and not (isinstance(leaf, jax.Array) and leaf.sharding == placement):
                leaves[index] = jax.device_put(leaf, placement)
        agreed(f"averaging {entry[0]}", functools.partial(_taken_in, means, leaves, firsts, count, entry[0],
                                                          read[0][0]))
        del leaves
        read.append(entry)
    assert structure is not None and model is not None
    placed = [held if sharding is None else place_leaf(held, sharding)
              for held, sharding in zip(means, firsts, strict=True)]
    return Averaged(jax.tree.unflatten(structure, placed), model, tuple(read))


def _taken_in(means: list, leaves: list, firsts: list, count: int, directory: str, first: str) -> None:
    """`means` with the `count`th run's `leaves` taken in (`_mean_of`), each let go in turn, so the run's
    tree leaves as the mean takes it."""
    for index, leaf in enumerate(leaves):
        means[index] = _mean_of(means[index], _held(leaf, firsts[index]), count, directory, first)
        leaves[index] = None


def _held(leaf, sharding):
    """`leaf`, placed as the first run's (`averaged` places it), on the host as the shards this process
    holds of it (`dew.distributed.host.Held`, which holds a replicated block once), a floating one in fp32;
    a host leaf whole. It reads this process's shards alone, no collective."""
    from dew.distributed.host import Held

    if sharding is None:
        return _on_host(leaf)
    held = Held.of(leaf)
    return Held(held.shape, {index: _on_host(block) for index, block in held.shards.items()})


def _mean_of(mean, taken, count: int, directory: str, first: str):
    """The running mean `mean` with the `count`th run's `taken` (`_held`'s), block by block (`_taken`)."""
    if isinstance(mean, np.ndarray):
        return _taken(mean, taken, count, directory, first)
    return type(mean)(mean.shape, {index: _taken(block, taken.shards[index], count, directory, first)
                                   for index, block in mean.shards.items()})


def _effective_weights(run: SavedRun, ema: bool | None, mesh, layout
                       ) -> tuple[Variables, ModelConfig, tuple[str, int, bool]]:
    """The pinned run's model tree as `averaged_pinned` takes it, its model record with any adapter
    merged away, and its directory, step and average read: the adapter's factors merged where the
    recorded model's tree is (`Objective._recorded_part`), beside any part its record leaves out, and
    the frozen leaves back in `params` (`thaw`). What the merge holds goes as this returns."""
    from dew.batch import part, thaw
    from dew.inference.tasks import saved_model
    from dew.lora import Adapter, AdapterRecord, recorded_adapter
    from dew.objectives.base import joined

    directory, pinned, objective = run.directory, run.step, run.objective
    config = saved_model(run.record, None)
    tree = run.variables(ema=ema, mesh=mesh, layout=layout, param_dtype=jnp.float32)
    adapter = recorded_adapter(config.wrappers)
    if adapter is not None:
        name = objective._recorded_part
        own = tree if name is None else part(tree, name)
        merged = Adapter.recorded(config.build(), own, adapter).merge(own)
        if name is None:
            tree = merged
        else:
            others = {key for collection in tree.values() for key in collection} - {name}
            tree = joined({name: merged, **{other: part(tree, other) for other in sorted(others)}})
    with Checkpoints(directory) as checkpoints:
        kept = checkpoints.stored(pinned).get("ema") is not None
    base = dataclasses.replace(config, wrappers=tuple(
        wrapper for wrapper in config.wrappers if not isinstance(wrapper, AdapterRecord)))
    return thaw(tree), base, (directory, pinned, not objective.ema_is_reference and (
        kept if ema is None else bool(ema)))


def _on_host(leaf) -> np.ndarray:
    """`leaf` on the host, a floating one in fp32, bf16 and the float8 types among them, which
    NumPy's own type hierarchy does not count as floating, and a complex one in complex64, both of its
    components fp32."""
    value = np.asarray(leaf)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        return value.astype(np.complex64)
    return value.astype(np.float32) if jnp.issubdtype(value.dtype, jnp.inexact) else value


def _taken(held: np.ndarray, leaf: np.ndarray, count: int, directory: str, first: str) -> np.ndarray:
    """`held`, the fp32 mean of a leaf over `count - 1` runs, with the `count`th run's `leaf` taken in,
    in NumPy's fp32, which keeps subnormals: for a second run `(held + leaf) / 2` below 2**126, a sum
    that cannot overflow (and is exact wherever its half could round), and `held / 2 + leaf / 2`
    above, exact halves, one rounding either way, so float64's mean rounded to fp32; for a later one
    `held + (leaf - held) / count` where both share a sign, whose difference cannot overflow, and
    `held * (1 - 1/count) + leaf / count` where they differ, whose terms cannot. A complex leaf's real
    and imaginary parts are each so averaged. An integer or boolean leaf must equal `held`."""
    if jnp.issubdtype(held.dtype, jnp.complexfloating):
        # Each component as a real leaf of the same shape, a scalar's included.
        real, imaginary = (_taken(np.array(part(held)), np.array(part(leaf)), count, directory, first)
                           for part in (np.real, np.imag))
        averaged = real.astype(np.complex64)
        averaged.imag = imaginary
        return averaged
    if not jnp.issubdtype(held.dtype, jnp.inexact):
        if not np.array_equal(held, leaf):
            raise ValueError(f"an integer or boolean variable of the run in {directory} differs from "
                             f"{first}'s, which no mean of their weights reconciles")
        return held
    two = np.float32(2)
    # Each branch is computed whole and the other's lanes discarded: an overflow there is never read.
    with np.errstate(over="ignore"):
        if count == 2:
            small = np.maximum(np.abs(held), np.abs(leaf)) < np.float32(2.0**126)
            midpoint = np.where(small, (held + leaf) / two, held / two + leaf / two)
            return midpoint.astype(np.float32, copy=False)
        same = np.sign(held) == np.sign(leaf)
        moved = held + (leaf - np.where(same, held, np.float32(0))) / np.float32(count)
        weighted = held * np.float32((count - 1) / count) + leaf / np.float32(count)
        return np.where(same, moved, weighted).astype(np.float32, copy=False)
