"""Companion CPU topology and concrete local-shard transport.

The companion replaces devices, not logical axes: every coordinate and every
global shard index is the accelerator mesh's. Cross-backend transfer happens
outside jit/AD using addressable shards, because JAX refuses general global
cross-host device_put between different device sets (dispatch.py:488-516).
"""
from __future__ import annotations

import dataclasses
import mmap
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.layout import Format
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, Sharding, SingleDeviceSharding
from jax.tree_util import KeyPath

from dew.objectives.base import Variables
from dew.training.distributed import Placement


def companion_mesh(accelerator: Mesh, devices=None) -> Mesh:
    """Replace each accelerator coordinate by its process-local CPU partner.

    CPU cardinality is launch-time JAX configuration, never changed here.
    Pairing by device-id rank within each process survives arbitrary mesh
    ordering; a process's CPUs beyond its accelerator devices, and processes
    the mesh leaves out, stay out of the companion. A process with fewer
    CPUs than accelerator devices is refused before any model state is
    allocated.
    """
    try:
        cpus = jax.devices("cpu") if devices is None else devices
    except RuntimeError as error:
        raise ValueError(
            "Layout.host containing params requires a CPU transaction backend; "
            "launch with JAX_PLATFORMS=cuda,cpu (or your accelerator,cpu) before "
            "initializing JAX, then restart this process") from error
    sources, targets = defaultdict(list), defaultdict(list)
    for device in accelerator.devices.flat:
        sources[device.process_index].append(device)
    for device in cpus:
        if device.platform != "cpu":
            raise ValueError("the companion mesh must contain CPU devices")
        targets[device.process_index].append(device)
    partners = {}
    for process, held in sources.items():
        local = targets[process]
        if len(local) < len(held):
            raise ValueError(
                f"process {process} has {len(held)} accelerator devices and {len(local)} CPU devices; "
                f"set JAX_NUM_CPU_DEVICES={len(held)} or "
                f"--xla_force_host_platform_device_count={len(held)} in the existing xla_flags "
                "before any JAX initialization, and restart; no backend configuration was changed")
        chosen = sorted(local, key=lambda d: d.id)[:len(held)]
        partners.update(zip(sorted(held, key=lambda d: d.id), chosen, strict=True))
    devices = np.asarray([partners[d] for d in accelerator.devices.flat], dtype=object)
    mesh = Mesh(devices.reshape(accelerator.devices.shape), accelerator.axis_names,
                axis_types=accelerator.axis_types)
    # Opening a CPU client alone does not prove its distributed collectives
    # are available. Exercise one scalar per device before allocating state.
    placement = NamedSharding(mesh, P(tuple(mesh.axis_names)))
    probe = jax.make_array_from_callback(
        (mesh.size,), placement, lambda _: np.ones((1,), np.int32))
    try:
        total = jax.jit(jnp.sum)(probe)
        if int(total) != mesh.size:
            raise ValueError("the companion CPU collective did not include every device")
    except RuntimeError as error:
        raise ValueError(
            "CPU transaction execution requires distributed CPU collectives; configure "
            "JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo (or MPI) before backend initialization") from error
    return mesh


type Leaf = jax.Array | np.ndarray | np.generic | HostSource
"""A leaf of a variables tree as it is placed: an array on the devices or on the host, or a host
source a placement reads one device shard at a time (`HostSource`)."""


def placed_leaves(tree: Variables, place: Callable[[KeyPath, Leaf], Leaf], *, in_place: bool,
                  rebuilt: Callable[[Leaf, Leaf], None] | None = None) -> Variables:
    """`tree` with each leaf replaced by what `place(path, leaf)` returns, `path` the leaf's key path
    as `jax.tree_util.tree_flatten_with_path` names it, one leaf at a time: a leaf placed lands
    (`block_until_ready`) before the next is read. A node JAX flattens that is not a mapping (a
    list, a quantized array's buffers) is rebuilt over its placed leaves once they have all landed,
    and `rebuilt(node, placed)` hears of it then.

    With `in_place` the tree is the caller's to give up: each plain dict it reaches through mappings
    alone is written in place, so every holder of one sees the placed leaves, any other mapping (a
    FrozenDict) is rebuilt as a dict over the same children, and a memory-mapped host leaf's pages are
    let go once its copy has landed (`evict`), so a tree of dicts is held twice one leaf at a time at
    most; the dicts below another node are rebuilt with it. Without it every node is rebuilt and no
    page is let go: the caller's tree, its host arrays included, stays as it was beside the placed one."""
    def landed(path: KeyPath, leaf: Leaf) -> Leaf:
        placed = place(path, leaf)
        if placed is not leaf:
            jax.block_until_ready(placed)
            if in_place and isinstance(leaf, np.ndarray):
                evict(leaf)
        return placed

    def mapping(node: Mapping, path: KeyPath) -> dict:
        held = node if in_place and type(node) is dict else dict(node)
        for name in list(held):
            held[name] = walk(held[name], (*path, jax.tree_util.DictKey(name)))
        return held

    def walk(node: Mapping | Leaf, path: KeyPath) -> dict | Leaf:
        if isinstance(node, Mapping):
            return mapping(node, path)
        leaves, structure = jax.tree_util.tree_flatten_with_path(node)
        if jax.tree_util.treedef_is_leaf(structure):
            return landed(path, node)
        placed = structure.unflatten([landed((*path, *keys), leaf) for keys, leaf in leaves])
        if rebuilt is not None:
            rebuilt(node, placed)
        return placed

    return mapping(tree, ())


def stream(tree: Variables, placement, held: Variables | None = None) -> Variables:
    """Move a variables tree into `placement`, one leaf at a time (`placed_leaves`, in place).

    Each leaf is placed and written back into its parent dict before the
    next is read, so a tree of dicts is held twice one leaf at a time at most;
    a node of another kind (a quantized array) is rebuilt once its leaves are
    all placed.

    `held` is the tree the sources came from, the one an objective keeps and
    hands the trainer as data. A dict node of it that holds a placed source
    is updated to the placed array as it lands, and one that holds a node of
    another kind to the node rebuilt over its placed leaves, so the objective
    and the state share one copy instead of two.

    The dicts of `tree` reached through mappings alone are updated in place, so
    the caller's other references to them see the placed arrays too; any other
    mapping (a FrozenDict) is rebuilt as a dict over the same children, and the
    dicts below a node of another kind are rebuilt with it. Each leaf goes to
    the target at its key path in `placement` (`target`): a quantized array's
    buffers to the shardings `placement` holds in their places, or all to one
    target that `placement` gives the node whole.
    """
    holders: dict[int, list[tuple[dict, object]]] = {}

    def index(node):
        if not isinstance(node, dict):
            return
        for name, child in node.items():
            if isinstance(child, Mapping):
                index(child)
            else:
                holders.setdefault(id(child), []).append((node, name))

    if held is not None:
        index(held)

    def handed(source: Leaf, placed: Leaf) -> None:
        for holder, key in holders.get(id(source), ()):
            holder[key] = placed

    def landed(path: KeyPath, leaf):
        placed = place_leaf(leaf, target(placement, path))
        handed(leaf, placed)
        return placed

    return placed_leaves(tree, landed, in_place=True, rebuilt=handed)


def target(placement, path: KeyPath) -> Sharding | Format:
    """The target `placement` gives the leaf at `path` (`placed_leaves`' key paths): its mappings read
    by key, any other node it holds by the key entries JAX flattens it to, and a sharding or a layout
    reached before the path ends the target of everything below it."""
    def child(node, key):
        if isinstance(key, jax.tree_util.DictKey) and isinstance(node, Mapping):
            return node[key.key]
        children, _ = jax.tree_util.tree_flatten_with_path(node, is_leaf=lambda held: held is not node)
        return next(held for keys, held in children if keys == (key,))

    for key in path:
        if isinstance(placement, (Sharding, Format)):
            break
        placement = child(placement, key)
    return placement

@runtime_checkable
class HostSource(Protocol):
    """A host leaf a placement reads one device shard at a time.

    `read` returns the values at a global index. `place_leaf` reads a shard,
    puts it on its device and lets it go before it reads the next, so the
    host holds one device shard of the leaf at a time. `release` gives back
    what the reads left resident.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    def read(self, index: tuple[slice, ...] | None) -> np.ndarray: ...

    def release(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class Held:
    """An array's addressable shards copied to host memory: a `HostSource`
    that `place_leaf` puts back where they were, as often as it is asked."""
    shape: tuple[int, ...]
    shards: Mapping[tuple[slice, ...], np.ndarray]

    @classmethod
    def of(cls, array: jax.Array) -> Held:
        return cls(array.shape, {shard.index: np.asarray(shard.data) for shard in array.addressable_shards})

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def read(self, index: tuple[slice, ...] | None) -> np.ndarray:
        return self.shards[index or ()]

    def release(self) -> None:
        """Keep the copy for the next placement."""


def snapshot[T](tree: T) -> Callable[[], T]:
    """Copy `tree`'s arrays, this process's shards of each, to host memory
    (`Held`), a random key as its key data, and return what puts a fresh copy
    back where each was placed (`place_leaf`), its memory kind too: a timing
    that steps a state leaves the run the state it started from, and a run
    restores the best state it held."""
    leaves, structure = jax.tree.flatten(tree)
    keys = [jax.random.key_impl(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else None
            for leaf in leaves]
    bits = [leaf if impl is None else jax.random.key_data(leaf)
            for leaf, impl in zip(leaves, keys, strict=True)]
    held = [(Held.of(leaf), leaf.sharding) for leaf in bits]

    def placed(source: Held, sharding, impl):
        leaf = place_leaf(source, sharding)
        return leaf if impl is None else jax.random.wrap_key_data(leaf, impl=impl)

    return lambda: structure.unflatten([placed(*copy, impl) for copy, impl in zip(held, keys, strict=True)])


def place_leaf(value, target: Sharding | Format) -> jax.Array:
    """Move one array into `target`, releasing the source once it lands.

    A host array or a single device's array is cut into shards by each
    process, so a sharded placement lands sharded. An array already on the
    mesh moves as it is. A `HostSource` is read and placed one device shard
    at a time. Memory-mapped
    source pages are given back to the kernel, which is what keeps a large
    checkpoint from being resident twice. A `Format` target, a physical
    layout beside its sharding (a bank's, `dew.inference.banks`), takes the
    whole value at once: a `HostSource` is read whole and released whether
    or not it lands, and any other value is put as it is and kept."""
    if isinstance(target, Format):
        try:
            whole = value.read(None) if isinstance(value, HostSource) else value
            return jax.block_until_ready(jax.device_put(whole, target))
        finally:
            if isinstance(value, HostSource):
                value.release()
    if isinstance(value, jax.Array) and isinstance(target, NamedSharding) and (
            isinstance(value.sharding, NamedSharding) or jnp.issubdtype(value.dtype, jax.dtypes.prng_key)):
        return transfer(value, target)
    if isinstance(value, HostSource):
        # make_array_from_callback would hold every addressable shard on
        # the host before any transfer.
        shards = []
        for device, index in target.addressable_devices_indices_map(value.shape).items():
            shard = jax.device_put(
                value.read(index), SingleDeviceSharding(device, memory_kind=target.memory_kind)
            )
            shards.append(jax.block_until_ready(shard))
        value.release()
        return jax.make_array_from_single_device_arrays(value.shape, target, shards)
    source = np.asarray(value)
    landed = jax.make_array_from_callback(source.shape, target, lambda index: source[index])
    jax.block_until_ready(landed)
    evict(source)
    return landed


def evict(array: np.ndarray) -> None:
    """Drop a memory-mapped checkpoint tensor's pages from this process.

    A loader maps a checkpoint file once and hands out views into it, so
    letting a view go frees nothing while another view is alive. The pages a
    placed tensor was read from stay resident, and for a base that fills the
    host that is one copy too many.

    The pages the view covers whole are given back to the kernel here. A
    boundary page shared with a neighbour stays until the neighbour is
    placed. A later read of an evicted page faults it back from the file. An
    array that is not such a view is left alone."""
    root: np.ndarray = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    mapping = root.base
    if not isinstance(root, np.memmap) or not isinstance(mapping, mmap.mmap) or array.nbytes == 0:
        return
    # The map starts at the allocation granule below the memmap's file
    # offset; the memmap's data sits that remainder into it.
    start = root.__array_interface__["data"][0] - root.offset % mmap.ALLOCATIONGRANULARITY
    offset = array.__array_interface__["data"][0] - start
    first = -(-offset // mmap.PAGESIZE) * mmap.PAGESIZE
    last = (offset + array.nbytes) // mmap.PAGESIZE * mmap.PAGESIZE
    if last > first:
        mapping.madvise(mmap.MADV_DONTNEED, first, last - first)


def transfer[TreeT](tree: TreeT, placement: Placement[TreeT] | NamedSharding) -> TreeT:
    """Move only concrete local shards, retaining the global shape and indices.

    `placement` is the tree's own placement, or the single sharding that is
    one array's whole placement.

    Host snapshots and CPU gradients are real copies unless the runtime
    proves otherwise. No donation or aliasing assumption can invalidate a
    retained state or an asynchronous checkpoint. The caller bounds snapshot
    assembly by bank.
    """
    def leaf(path, value, target):
        # Integer statistic cotangents are symbolic float0 arrays. They have
        # no device buffer and must remain zero tangents, not numeric zeros.
        if getattr(value, "dtype", None) == jax.dtypes.float0:
            return value
        if not isinstance(value, jax.Array):
            return jax.make_array_from_process_local_data(target, np.asarray(value))
        if value.sharding == target:
            return value
        source = value.sharding
        if not isinstance(source, NamedSharding):
            if not value.is_fully_addressable:
                raise ValueError(f"{jax.tree_util.keystr(path)} requires named global shard coordinates")
            if not target.is_fully_addressable:
                return jax.make_array_from_process_local_data(target, value, global_shape=value.shape)
            return jax.device_put(value, target)
        if source.mesh.shape != target.mesh.shape:
            if source.device_set == target.device_set:
                return jax.device_put(value, target)
            if value.is_fully_addressable:
                return jax.device_put(value, target)
            raise ValueError(
                f"{jax.tree_util.keystr(path)} requires identical logical shard indices for transport")
        source_indices = source.devices_indices_map(value.shape)
        target_indices = target.devices_indices_map(value.shape)
        if not isinstance(source.mesh, Mesh):
            raise ValueError("transport requires a concrete source device mesh")
        partners = dict(zip(source.mesh.devices.flat, target.mesh.devices.flat, strict=True))
        arrays = {}
        for shard in value.addressable_shards:
            device = partners[shard.device]
            if source_indices[shard.device] != target_indices[device]:
                raise ValueError(
                    f"{jax.tree_util.keystr(path)} shard index mismatch at {shard.device}: "
                    f"{source_indices[shard.device]} "
                    f"versus companion {device}: {target_indices[device]}")
            arrays[device] = jax.device_put(
                shard.data, SingleDeviceSharding(device, memory_kind=target.memory_kind))
        return jax.make_array_from_single_device_arrays(
            value.shape, target,
            [arrays[d] for d in target.addressable_devices_indices_map(value.shape)], dtype=value.dtype)
    return jax.tree_util.tree_map_with_path(leaf, tree, placement)
