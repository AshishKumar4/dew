"""A local step another process lost, read from the processes that hold its replicas: each array's grid of
shards by process (`Grid`), which holder sends which shard (`_sources`), whether the holders cover every
shard (`_covered`), and the arrays rebuilt from them (`_from_replicas`). `dew.checkpoints` recovers a lost
local step through them."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from dew.checkpoints._state_trees import _UNSIGNED
from dew.mesh import mesh_axes
from dew.nn.kernels.options import owned_program

if TYPE_CHECKING:
    from dew.checkpoints import StateLeaf


def _split_axes(spec: jax.sharding.PartitionSpec) -> list[str]:
    """The mesh axes `spec` splits an array over, sorted."""
    return sorted(axis for entry in spec for axis in mesh_axes(entry))


@dataclasses.dataclass(frozen=True)
class Grid:
    """Where a local save's arrays lie, for a process that lost its copy to find each shard it held on a
    replica (`_covered`): the mesh's axes, its shape and its devices' ids in grid order, and, by path,
    the axes each array splits over. A save records it and a recovery reads it back by its fields, as
    every record is (`dew.registry.to_record`, `from_record`)."""

    axes: tuple[str, ...]
    shape: tuple[int, ...]
    devices: tuple[int, ...]
    split: Mapping[str, tuple[str, ...]]

    @classmethod
    def each(cls, tree: Mapping[str, StateLeaf]) -> tuple[Grid, ...] | None:
        """The grids `tree`'s arrays lie on, one a mesh, each with the arrays on it (a pipeline's rows each
        on its own stage's, `dew.training.stages.StageLocal.saved`); None where an array lies elsewhere
        than on a mesh by name (a host-owned state's), which no replica recovers (row 10814: one grid alone
        was recorded, and a step whose rows spanned several meshes was recovered from none)."""
        meshes: dict[tuple, dict[str, tuple[str, ...]]] = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            if not isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)):
                continue
            if not isinstance(leaf.sharding, jax.sharding.NamedSharding):
                return None
            mesh = leaf.sharding.mesh
            # A mesh of another backend's devices than the pool's default (a CPU-owned run's companion beside
            # its accelerator's banks) is no grid a recovery resolves by device id.
            if (not isinstance(mesh, jax.sharding.Mesh)
                    or mesh.devices.flat[0].platform != jax.default_backend()):
                return None
            meshes.setdefault(_placed_on(mesh), {})[jax.tree_util.keystr(path)] = tuple(
                _split_axes(leaf.sharding.spec))
        return tuple(cls(axes, shape, devices, split) for (axes, shape, devices), split in meshes.items())

    def holds(self, mesh: jax.sharding.Mesh) -> bool:
        """Whether this grid is `mesh`'s: its axes, shape and devices."""
        return (self.axes, self.shape, self.devices) == _placed_on(mesh)


def _placed_on(mesh: jax.sharding.Mesh) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """`mesh`'s axes, shape and devices' ids in grid order, as a `Grid` records them."""
    grid = mesh.devices
    return tuple(mesh.axis_names), tuple(grid.shape), tuple(device.id for device in grid.flat)



def _sources(grid: Grid, split: Sequence[str], holders: frozenset[int],
             process: Mapping[int, int] | None = None) -> np.ndarray | None:
    """Which device, in `grid`'s order, gives its shard of an array split over the axes `split` to the
    devices that share it (the same coordinates on those axes): the first, by grid order, on a process
    of `holders`, each device's process by its id (`process`, the pool's own by default). None where
    some shard has no copy on any holder."""
    shape, axes = grid.shape, grid.axes
    if process is None:
        process = {device.id: device.process_index for device in jax.devices()}
    held = np.asarray([process[device] in holders for device in grid.devices], bool).reshape(shape)
    # The axes a shard's copies differ along come first, then each group's devices along the rest.
    shared = [axis for axis, name in enumerate(axes) if name not in split]
    order = [*[axis for axis in range(len(axes)) if axis not in shared], *shared]
    groups = np.transpose(held, order).reshape(-1, math.prod(shape[axis] for axis in shared) if shared else 1)
    if not groups.any(axis=1).all():
        return None
    first = np.zeros_like(groups)
    first[np.arange(len(groups)), groups.argmax(axis=1)] = True
    return np.transpose(first.reshape([shape[axis] for axis in order]), np.argsort(order))


def _covered(grids: Sequence[Grid] | None, holders: frozenset[int],
             process: Mapping[int, int] | None = None) -> bool:
    """Whether the processes of `holders` hold a copy of every shard of every array a local save
    placed over `grids` (`Grid.each`), so a process that lost its copy recovers it from them; each
    device's process by its id (`process`, the pool's own by default)."""
    if grids is None:
        return False
    return all(_sources(grid, axes, holders, process) is not None
               for grid in grids for axes in set(grid.split.values()))


def _from_replicas(tree, grids: Sequence[Grid], holders: frozenset[int]):
    """`tree`'s arrays, each shard on every device that holds it the bytes its first copy on a
    process of `holders` holds (`_sources`): a process that lost its local copy holds zeros there,
    and receives each shard as the sum, over the axes its copies lie along, of the source's bits and
    the others' zeros, which is the source's bits exactly. Every process calls it alike."""
    masks: dict[tuple, jax.Array] = {}

    def recovered(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        placed = leaf.sharding
        assert isinstance(placed, jax.sharding.NamedSharding), "a covered save places by name (`Grid.each`)"
        mesh, split = placed.mesh, tuple(_split_axes(placed.spec))
        assert isinstance(mesh, jax.sharding.Mesh), "a covered save places on concrete meshes (`Grid.each`)"
        # Every axis the copies lie along, those of one device too, so the sum is replicated as its
        # placement says.
        along = tuple(name for name in mesh.axis_names if name not in split)
        if all(mesh.shape[name] == 1 for name in along):
            return leaf
        grid = next(grid for grid in grids if grid.holds(mesh))
        if (_placed_on(mesh), split) not in masks:
            sources = _sources(grid, split, holders)
            assert sources is not None, "a recovered step's shards are covered (`_covered`)"
            every = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*mesh.axis_names))
            masks[_placed_on(mesh), split] = jax.make_array_from_callback(sources.shape, every,
                                                                          lambda index: sources[index])
        typed = jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        values = jax.random.key_data(leaf) if typed else leaf
        held = values.sharding
        assert isinstance(held, jax.sharding.NamedSharding), "a key's data lies where its key does"
        mask = masks[_placed_on(mesh), split]
        value = jax.device_put(_summed(values, mask, mesh, along), held)
        return jax.random.wrap_key_data(value, impl=jax.random.key_impl(leaf)) if typed else value

    return jax.tree.map(recovered, tree)


def _summed(values: jax.Array, mask: jax.Array, mesh: jax.sharding.Mesh, along: tuple[str, ...]) -> jax.Array:
    """`values`' every shard as its source holds it (`mask` true on the source), summed over `along` as its
    words: the source's bits and the others' zeros, the source's bits exactly (`_from_replicas`). A complex
    value's real and imaginary parts are its two lanes, side by side on a trailing axis, each summed as its
    words, as no word holds a complex number's bits whole (row 12177), in the one program."""
    held = values.sharding
    assert isinstance(held, jax.sharding.NamedSharding), "a covered save places by name (`Grid.each`)"
    lanes = jnp.issubdtype(values.dtype, jnp.complexfloating)
    lane = jnp.finfo(values.dtype).dtype if lanes else values.dtype
    words = _UNSIGNED[max(lane.itemsize, 1)]

    def sum_of_sources(value, source):
        if lanes:
            value = jnp.stack([jnp.real(value), jnp.imag(value)], axis=-1)
        bits = (value.astype(np.uint8) if value.dtype == np.bool_
                else jax.lax.bitcast_convert_type(value, words))
        summed = jax.lax.psum(jnp.where(source.reshape(()), bits, jnp.zeros_like(bits)), along)
        if not lanes:
            return summed
        parts = jax.lax.bitcast_convert_type(summed, lane)
        return jax.lax.complex(parts[..., 0], parts[..., 1])

    summing = jax.shard_map(sum_of_sources, mesh=mesh, out_specs=held.spec,
                            in_specs=(held.spec, jax.sharding.PartitionSpec(*mesh.axis_names)))
    summed = owned_program(summing)(jax.device_put(values, held.with_memory_kind("device")), mask)
    if lanes:
        return summed
    return summed != 0 if values.dtype == np.bool_ else jax.lax.bitcast_convert_type(summed, values.dtype)


def _zeros(leaf):
    """Zeros where a placed template leaf lies, each process's shards its own: the shards a process
    that lost its local copy holds until its replicas give it theirs (`_from_replicas`). A key is
    read as its key data, as a restore reads it (`_placed`), in the words its own implementation holds
    (an rbg key's four, threefry's two), a concrete key's or a shape's; a leaf nothing places is kept."""
    sharding = getattr(leaf, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return leaf
    shape, dtype = tuple(leaf.shape), leaf.dtype
    if jnp.issubdtype(dtype, jax.dtypes.prng_key):
        key_words = jax.eval_shape(jax.random.key_data, jax.ShapeDtypeStruct(shape, dtype))
        shape, dtype = tuple(key_words.shape), key_words.dtype
    # Every shard of a placement by name is one shape.
    shard = sharding.shard_shape(shape)
    return jax.make_array_from_callback(shape, sharding, lambda _: np.zeros(shard, dtype))
