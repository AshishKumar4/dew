"""Companion CPU topology and concrete local-shard transport.

The companion replaces devices, not logical axes: every coordinate and every
global shard index is the accelerator mesh's. Cross-backend transfer happens
outside jit/AD using addressable shards, because JAX refuses general global
cross-host device_put between different device sets (dispatch.py:488-516).
"""
from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding

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
        partners.update(zip(sorted(held, key=lambda d: d.id), sorted(local, key=lambda d: d.id)))
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


def transfer(tree, placement: Placement):
    """Move only concrete local shards, retaining the global shape and indices.

    Host snapshots and CPU gradients are real copies unless the runtime proves
    otherwise; no donation or aliasing assumption can invalidate retained states
    or an asynchronous checkpoint. The caller bounds snapshot assembly by bank.
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
        partners = dict(zip(source.mesh.devices.flat, target.mesh.devices.flat))
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
