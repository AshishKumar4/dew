"""The trees a checkpoint step holds, walked as Dew's state declares them (`dew.state`): where a run's
state sits, the run's and each DiLoCo group's (`_holders`, `_at`, `_with`); its variables trees and
averages; the averages stored as byte planes of their differences from their weights (`_with_ema_deltas`);
and where its optimizer keeps post-hoc EMA profiles (`_profiles_at`). `dew.checkpoints` reads every step
through them."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from dew import _records
from dew._records import JSON, record
from dew.batch import Variables
from dew.nn.kernels.options import program

if TYPE_CHECKING:
    from dew.checkpoints._payloads import _StepFacts


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
    from dew._coordination import collective_host

    # A group's clocks are on its own devices, which a pool's processes split.
    updates, stds = (np.asarray(value) for value in collective_host((profiles.updates, profiles.stds),
                                                                    phase="checkpoint profiles"))
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


def _variable_trees(tree) -> dict[TreePath, Mapping[str, object]]:
    """The variables trees a step's state tree holds, by where they sit, each keyed by collection: the
    variables and the EMA (`dew.state.VARIABLE_FIELDS`), the run's and each DiLoCo group's."""
    from dew.state import VARIABLE_FIELDS
    return {(*holder, name): collections for holder, fields in _holders(tree) for name in VARIABLE_FIELDS
            if name in fields and isinstance(collections := _at(tree, (*holder, name)), Mapping)}


def _averaged(path: TreePath) -> TreePath:
    """The tree whose leaves the average at `path` follows (`dew.state.AVERAGES`): an EMA's weights."""
    from dew.state import AVERAGES
    return (*path[:-1], AVERAGES[path[-1]])


def _averages_held(tree) -> list[TreePath]:
    """The averages a step's state tree holds (`dew.state.AVERAGES`): the run's EMA and each group's."""
    from dew.state import AVERAGES
    return [(*holder, name) for holder, fields in _holders(tree) for name in AVERAGES
            if name in fields and _at(tree, (*holder, name)) is not None]


def _with_ema_deltas(state_tree: Mapping[str, object]) -> tuple[Mapping[str, object], dict[str, list[str]]]:
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


def _item_metadata(metadata: _StepFacts | ocp.metadata.StepMetadata) -> Variables:
    """The named state tree a step's metadata holds (`_StepFacts.item_metadata`), or one item's in Orbax's
    metadata of it (a stored collection's, `FROZEN_STORE`)."""
    tree = _tree_of(metadata.item_metadata)
    if not isinstance(tree, Mapping) or any(not isinstance(key, str) for key in tree):
        raise ValueError("the checkpoint holds no named state metadata")
    return dict(tree)


def _tree_of(metadata) -> Mapping[str, object]:
    """One item's state tree in Orbax's metadata: a dict, or its newer metadata wrapper's `tree`."""
    return metadata.tree if isinstance(metadata, _MetadataTree) else metadata


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
