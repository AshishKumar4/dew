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


def fetched_at(tree, index: jax.Array | int, axis: int = 0):
    """Each leaf's entry `index` along `axis` (a bank's layer, a pool's page) in device memory: copied with
    the axis intact and squeezed there."""
    index = jax.lax.optimization_barrier(jnp.asarray(index, jnp.int32))
    return jax.tree.map(
        lambda leaf: jax.lax.squeeze(fetched(jax.lax.dynamic_slice_in_dim(leaf, index, 1, axis)), (axis,)),
        tree)


def placed_like(leaf: jax.Array, value: jax.Array) -> jax.Array:
    """`value`, computed in device memory, where `leaf` lies: in host memory for a host leaf."""
    return jax.device_put(value, jax.memory.Space.Host) if on_host(leaf) else value


def stored_at(bank: jax.Array, value: jax.Array, index: jax.Array | int, axis: int = 0) -> jax.Array:
    """`bank` with `value` its entry `index` along `axis`, where `bank` lies (`fetched_at`'s inverse): the
    axis restored in device memory, the entry copied to the bank's memory and written there."""
    index = jax.lax.optimization_barrier(jnp.asarray(index, jnp.int32))
    entry = placed_like(bank, jnp.expand_dims(value.astype(bank.dtype), axis))
    return jax.lax.dynamic_update_slice_in_dim(bank, entry, index, axis)
