"""Arrays a program keeps in host memory (`jax.memory.Space.Host`) and brings to the device as it reads them:
a scanned stack's banks of read-only layers a layout places on the host (`dew.nn.backbones.decoder_stack`)
and an offloaded cache's keys and values (`dew.nn.kv_cache.KVCache.offloaded`), a layer or a page at a time,
written back where they lie.

The one owner of that movement. Each copy keeps the bank's axis on the host and reshapes only in device
memory: TPU host tiles cannot in general be bitcast to another rank, and at a constant index XLA:GPU folds a
slice of a host array to its row and the squeeze into the host parameter's own bitcast, which its layout
assignment refuses ("Bitcast cannot have different memory spaces"). The index reaches the slice through an
optimization barrier, so the slice stays the dynamic one a scanned layer takes.
"""

import jax
import jax.numpy as jnp
from numpy.lib.array_utils import normalize_axis_tuple


def on_host(tree) -> bool:
    """Whether any leaf of `tree` sits in host memory."""
    return any(jax.typeof(leaf).memory_space is jax.memory.Space.Host for leaf in jax.tree.leaves(tree))


def fetched(tree):
    """`tree` in device memory, the copy issued and not waited for.

    `jax.device_put` to a memory space alone keeps each leaf's sharding and dtype, so a fetched shard is
    the shard the layout placed on the host and the collectives its layer issues are the ones a resident
    run issues. A leaf already in device memory is not moved, so a bank the layout left resident reads
    the same way and gives the same values.
    """
    return jax.tree.map(lambda leaf: jax.device_put(leaf, jax.memory.Space.Device), tree)


type Index = jax.Array | int
"""An entry's place along one axis, traced or not."""


def _entry(index: Index | tuple[Index, ...], axis: int | tuple[int, ...]
           ) -> tuple[tuple[int, ...], tuple[jax.Array, ...]]:
    """The axes an entry is taken along and its index on each, int32 behind one optimization barrier (a
    slice's starts share one dtype), so the slice stays the dynamic one a scanned layer takes."""
    axes = (axis,) if isinstance(axis, int) else axis
    indices = (index,) if isinstance(axis, int) else index
    if not isinstance(indices, tuple) or len(indices) != len(axes):
        raise ValueError(f"an entry along the axes {axes} takes an index on each, not {index!r}")
    return axes, jax.lax.optimization_barrier(tuple(jnp.asarray(each, jnp.int32) for each in indices))


def _placed(ndim: int, axes: tuple[int, ...], indices: tuple[jax.Array, ...]
            ) -> tuple[tuple[int, ...], list[jax.Array]]:
    """`axes` of an array of rank `ndim`, each counted from its first (a negative axis from its last; one past
    its rank refused, as NumPy refuses it), and a slice's starts over its axes: each of `axes` at its index,
    every other axis at zero."""
    held = normalize_axis_tuple(axes, ndim)
    starts = [jnp.zeros((), jnp.int32)] * ndim
    for axis, index in zip(held, indices, strict=True):
        starts[axis] = index
    return held, starts


def fetched_at(tree, index: Index | tuple[Index, ...], axis: int | tuple[int, ...] = 0):
    """Each leaf's entry `index` along `axis` (a bank's layer, a pool's page), or along each of several axes
    at an index of each (a bank's layer and page), in device memory: copied with the axes intact and
    squeezed there."""
    axes, indices = _entry(index, axis)

    def entry(leaf: jax.Array) -> jax.Array:
        held, starts = _placed(leaf.ndim, axes, indices)
        sizes = [1 if dimension in held else size for dimension, size in enumerate(leaf.shape)]
        return jax.lax.squeeze(fetched(jax.lax.dynamic_slice(leaf, starts, sizes)), held)

    return jax.tree.map(entry, tree)


def placed_like(leaf: jax.Array, value: jax.Array) -> jax.Array:
    """`value`, computed in device memory, where `leaf` lies: in host memory for a host leaf."""
    return jax.device_put(value, jax.memory.Space.Host) if on_host(leaf) else value


def stored_at(bank: jax.Array, value: jax.Array, index: Index | tuple[Index, ...],
              axis: int | tuple[int, ...] = 0, *, kept: jax.Array | None = None) -> jax.Array:
    """`bank` with `value` its entry `index` along `axis` (or along each of several axes), where `bank` lies
    (`fetched_at`'s inverse): the axes restored in device memory, the entry copied to the bank's memory and
    written there; where `kept` is given (a scalar), written only where it holds, the entry copied first
    whatever it says. That is the one form XLA:GPU's host offloading compiles for a bank in host memory
    that a loop writes: a copy to the host made inside a conditional's branch, or a write taken
    unconditionally in a loop's body or unrolled, or a loop whose count the writes kept set, each refuses
    the bank as a host parameter ("not in host memory space"; armada probes on an A100's compile at
    c27860888)."""
    axes, indices = _entry(index, axis)
    held, starts = _placed(bank.ndim, axes, indices)
    entry = placed_like(bank, jnp.expand_dims(value.astype(bank.dtype), held))
    if kept is None:
        return jax.lax.dynamic_update_slice(bank, entry, starts)
    return jax.lax.cond(kept, lambda bank: jax.lax.dynamic_update_slice(bank, entry, starts),
                        lambda bank: bank, bank)
