"""Orbax's array handler for a pool whose arrays live on part of it: a pipeline stage's rows, each on its own
stage's devices (`dew.training.stages.StageLocal.saved`), which a stage of one process's devices holds
whole.

Orbax 0.12.7 refuses such an array as a `pmap`'s host-local one (`ArrayHandler.serialize`), takes its
first addressable shard where a process holds none (`replica_slices.get_replica_slices`), and makes it on
restore without the dtype a process that holds none of it must state
(`serialization._read_and_create_array`). Here each array on a strict part of the pool (`apart`) is
written by the processes that hold its shards, its metadata by the first of them, the others writing
nothing of it, and read back by them, the others making it with no shard; every other array is Orbax's
own. The files are Orbax's: each such row its own array, read by any topology as the stages wrote it
(docs/design/training.md, checkpoints). Only an array a pipeline's checkpoint made (`dew.mesh.held_apart`,
which `dew.training.stages.StageLocal.saved` names of each row it makes, by the array itself) is written
so; any other array on part of the pool, a `pmap`'s host-local output on each process's own devices under
one name above all, is still refused as Orbax refuses it, with no round between the processes (a local
tier's saves are each process's own)."""

from __future__ import annotations

import asyncio
import dataclasses

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.serialization import jax_array_handlers as handlers, replica_slices

from dew.mesh import is_held_apart


def apart(value) -> bool:
    """Whether `value` is an array whose devices are some processes' of a pool and not every one's, on any
    backend (a CPU-owned state's companion mesh spans every process, as its accelerator's does)."""
    return (isinstance(value, jax.Array) and jax.process_count() > 1
            and {device.process_index for device in value.sharding.device_set} != set(range(
                jax.process_count())))


def _owner(sharding: jax.sharding.Sharding) -> int:
    """The process that writes the metadata of an array placed by `sharding`: the first that holds it."""
    return min(device.process_index for device in sharding.device_set)


class SubmeshArrays(ocp.type_handlers.ArrayHandler):
    """Orbax's `ArrayHandler`, which also writes and reads an array on a strict part of the pool (module
    docstring): the processes that hold it write its shards and the first of them its metadata, and read it
    back, and a process that holds none of it makes it with no shard."""

    async def serialize(self, values, infos, args=None):
        args = list(args or [ocp.SaveArgs()] * len(values))
        parted: list[int] = []
        whole: list[int] = []
        for index, value in enumerate(values):
            # A stage's array alone: any other's on part of the pool, Orbax's own refusal (module docstring).
            (parted if apart(value) and isinstance(value.sharding, jax.sharding.NamedSharding)
             and is_held_apart(value) else whole).append(index)
        futures = [] if not whole else list(await super().serialize(
            [values[index] for index in whole], [infos[index] for index in whole],
            [args[index] for index in whole]))
        if not parted:
            return futures
        if any(jax.dtypes.issubdtype(values[index].dtype, jax.dtypes.prng_key) for index in parted):
            raise ValueError("a PRNG key on part of the pool is not written: place it on every process")
        if self._enable_write_sharding_file:
            futures.append(future.CommitFutureAwaitingContractedSignals(
                handlers._async_serialize_shardings(
                    shardings=[values[index].sharding for index in parted],
                    infos=[infos[index] for index in parted], primary_host=self._primary_host),
                name="serialize_submesh_shardings"))
        absent = [index for index in parted if not values[index].addressable_shards]
        if absent and self._primary_host is None:
            # A local tier's process writes the metadata of each array it holds none of, and no shard, so its
            # step's tree reads whole from its own directory, as Orbax reads a step's.
            empty = [replica_slices.ReplicaSlices(
                global_shape=values[index].shape, local_shape=values[index].sharding.shard_shape(
                    values[index].shape), sharding=values[index].sharding, dtype=values[index].dtype,
                is_on_host=True, replica_slices=[]) for index in absent]
            futures.append(future.CommitFutureAwaitingContractedSignals(handlers._async_serialize_replica_slices(
                empty, [infos[index] for index in absent], [args[index] for index in absent],
                primary_host=None, metadata_key=self._metadata_key, array_metadata_store=None,
                enable_replica_parallel_separate_folder=False, use_replica_parallel=False, ext_metadata={}),
                name="serialize_submesh_metadata"))
        by_owner: dict[int, list[int]] = {}
        for index in parted:
            if values[index].addressable_shards:
                by_owner.setdefault(_owner(values[index].sharding), []).append(index)
        for owner, members in sorted(by_owner.items()):
            futures.append(handlers._serialize_arrays(
                arrays=[values[index] for index in members], infos=[infos[index] for index in members],
                args=[args[index] for index in members], dispatcher=None,
                # A local tier's every process writes its own metadata, into its own directory.
                primary_host=None if self._primary_host is None else owner,
                replica_id=self._replica_id, use_replica_parallel=self._use_replica_parallel,
                min_slice_bytes_for_replica_parallel=self._min_slice_bytes_for_replica_parallel,
                max_replicas_for_replica_parallel=self._max_replicas_for_replica_parallel,
                enable_replica_parallel_separate_folder=self._enable_replica_parallel_separate_folder,
                metadata_key=self._metadata_key, ext_metadata={}, array_metadata_store=None,
                callback=self._callback))
        return futures

    async def deserialize(self, infos, args=None):
        # Orbax's own read where every array is placed where this process holds part of it.
        if args is None or not infos or all(
                not isinstance(arg, ocp.ArrayRestoreArgs) or (isinstance(arg.sharding, jax.sharding.Sharding)
                                                              and arg.sharding.addressable_devices)
                for arg in args):
            return await super().deserialize(infos, args)
        # Each array's placement as Orbax resolves it, the step's own sharding file read where a restore
        # states none, so a row another stage holds is known before Orbax reads it.
        await asyncio.gather(*[param.await_path_creation() for param in infos])
        recorded = await async_path.exists(infos[0].parent_dir / handlers._SHARDING_FILE_NAME)
        placed = await handlers._deserialize_shardings(infos, args, recorded)
        args = [dataclasses.replace(arg, sharding=sharding, mesh=None, mesh_axes=None)
                if isinstance(arg, ocp.ArrayRestoreArgs) and isinstance(sharding, jax.sharding.Sharding)
                else arg for arg, sharding in zip(args, placed, strict=True)]
        none: list[int] = []
        held: list[int] = []
        unheld: dict[int, ocp.ArrayRestoreArgs] = {}
        for index, arg in enumerate(args):
            if (isinstance(arg, ocp.ArrayRestoreArgs) and isinstance(arg.sharding, jax.sharding.Sharding)
                    and not arg.sharding.addressable_devices):
                none.append(index)
                unheld[index] = arg
            else:
                held.append(index)
        read = [] if not held else list(await super().deserialize([infos[index] for index in held],
                                                                   [args[index] for index in held]))
        # Its shape and dtype as the restore states them, else as the step's files do (a local tier's
        # process wrote none of an array it held none of).
        unstated = [index for index, arg in unheld.items() if arg.global_shape is None or arg.dtype is None]
        stated = dict(zip(unstated, await self.metadata([infos[index] for index in unstated]) if unstated
                          else [], strict=True))

        def known(index: int, field: str):
            given = getattr(unheld[index], "global_shape" if field == "shape" else field)
            return getattr(stated[index], field) if given is None else given

        placements = {index: arg.sharding for index, arg in unheld.items()
                      if isinstance(arg.sharding, jax.sharding.Sharding)}
        made = [jax.make_array_from_single_device_arrays(
            tuple(known(index, "shape")), placements[index], [], dtype=known(index, "dtype"))
            for index in none]
        restored: list = [None] * len(infos)
        for index, value in [*zip(held, read, strict=True), *zip(none, made, strict=True)]:
            restored[index] = value
        return restored
