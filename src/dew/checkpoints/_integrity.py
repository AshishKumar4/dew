"""A checkpoint's arrays' digest, read on device without moving them to the host (`_digest`), which a
restore checks what it read against."""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from dew._hashing import fmix32
from dew.nn.kernels.options import program

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
