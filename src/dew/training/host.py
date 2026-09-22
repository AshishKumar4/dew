"""Companion CPU topology and concrete local-shard transport.

The companion replaces devices, not logical axes: every coordinate and every
global shard index is the accelerator mesh's. Cross-backend transfer happens
outside jit/AD using addressable shards, because JAX refuses general global
cross-host device_put between different device sets (dispatch.py:488-516).
"""
from __future__ import annotations

import mmap
from collections import defaultdict
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding

from dew.objectives.base import Variables
from dew.training.distributed import Placement


def companion_mesh(accelerator: Mesh, devices=None) -> Mesh:
    """Replace each accelerator coordinate by its process-local CPU partner.

    CPU cardinality is launch-time JAX configuration, never changed here.
    Pairing by device-id rank within each process survives arbitrary mesh
    ordering. A mismatch is refused before any model state is allocated.
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
    if set(sources) != set(targets):
        raise ValueError("the CPU companion and accelerator must participate on the same processes")
    partners = {}
    for process, held in sources.items():
        local = targets[process]
        if len(held) != len(local):
            raise ValueError(
                f"process {process} has {len(held)} accelerator devices and {len(local)} CPU devices; "
                f"set JAX_NUM_CPU_DEVICES={len(held)} or "
                f"--xla_force_host_platform_device_count={len(held)} in the existing xla_flags "
                "before any JAX initialization, and restart; no backend configuration was changed")
        partners.update(zip(sorted(held, key=lambda d: d.id), sorted(local, key=lambda d: d.id), strict=True))
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


def stream(tree: Variables, placement, held: Variables | None = None) -> Variables:
    """Move a variables tree into `placement`, one leaf at a time.

    Each leaf is placed and written back into its parent node before the
    next is read, so at most one leaf is held twice and the whole tree never
    exists in two places.

    `held` is the tree the sources came from, the one an objective keeps and
    hands the trainer as data. A dict node of it that holds a placed source
    is updated to the placed array, so the objective and the state share one
    copy instead of two. A node that is not a dict is left as it is.

    Every dict node of `tree` is updated in place, so the caller's other
    references to it see the placed arrays too.
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

    def place_dict(node, target):
        for name in list(node):
            child, wanted = node[name], target[name]
            if isinstance(child, dict):
                place_dict(child, wanted)
                continue
            landed = place_leaf(child, wanted)
            for holder, key in holders.get(id(child), ()):
                holder[key] = landed
            node[name] = landed

    place_dict(tree, placement)
    return tree


def place_leaf(value, target: NamedSharding) -> jax.Array:
    """Move one array into `target`, releasing the source once it lands.

    A host array or a single device's array is cut into shards by each
    process, so a sharded placement lands sharded. An array already on the
    mesh moves as it is. Memory-mapped source pages are given back to the
    kernel, which is what keeps a large checkpoint from being resident
    twice."""
    if isinstance(value, jax.Array) and (isinstance(value.sharding, NamedSharding)
                                         or jnp.issubdtype(value.dtype, jax.dtypes.prng_key)):
        return transfer(value, target)
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
