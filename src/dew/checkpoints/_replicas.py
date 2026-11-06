"""A local step another process lost, read from the processes that hold its replicas: each array's grid of
shards by process (`_grid`), which holder sends which shard (`_sources`), whether the holders cover every
shard (`_covered`), and the arrays rebuilt from them (`_from_replicas`). `dew.checkpoints` recovers a lost
local step through them."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from dew import _records
from dew._records import JSON, record
from dew.checkpoints._state_trees import _UNSIGNED
from dew.mesh import mesh_axes
from dew.nn.kernels.options import owned_program

if TYPE_CHECKING:
    from dew.checkpoints import StateLeaf


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
