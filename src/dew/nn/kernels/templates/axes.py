"""Named axes (design §6, the view vocabulary's first part): a template operand's axes by launch name ('rows',
'depth', 'columns'), and what a launch takes of it, axis by axis, from facts it states by name: a block's
extents, a view's slices, a step's block indices. Every backend's launch reads them so (Mosaic GPU's and TPU's
mainloops, Pallas-Triton's), instead of each spelling the per-axis comprehension its own way."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.sharding import explicit_spec, whole_along

WHOLE = slice(None)
"""An axis taken whole (a view's slice along an axis it does not cut)."""

GROUPS = 'groups'
"""A side operand's axis of one value a group (`dew.nn.kernels.gemm`'s, row 3963): a launch reads the operand
at its tile's group (`at_group`), the axis indexed away (`held`), and hands its hook the value at each of the
tile's rows (`by_rows`), as the portable program reads it repeated at each row."""


def named[T](axes: Sequence[str], by_axis: Mapping[str, T]) -> tuple[T, ...]:
    """`by_axis[axis]` for each of `axes`, in their order: an operand's block extents or origins, every axis
    stated."""
    return tuple(by_axis[axis] for axis in axes)


def whole(count: int, block: int) -> int:
    """`count` rounded up to whole `block`s: the extent a launch's tiles of `block` cover, or an extent
    aligned to a unit of `block` (an allocation's, a step's depth)."""
    return -(-count // block) * block


def laid(array: jax.Array, subscript: Sequence[str], letters: Sequence[str]) -> jax.Array:
    """`array`, whose axes `subscript` names letter by letter (an einsum operand's), laid over `letters`: its
    axes in their order, an axis of extent one at a letter it lacks, so it broadcasts against any other
    operand laid over them (`scaled_contraction`'s factors, the algebra's subnormal contraction's)."""
    ordered = [letter for letter in letters if letter in subscript]
    moved = jnp.transpose(array, [subscript.index(letter) for letter in ordered])
    return moved.reshape([array.shape[subscript.index(letter)] if letter in subscript else 1
                          for letter in letters])


def blocked(ref, block: Sequence[int], indices: Sequence):
    """`ref`'s block of extents `block` at block indices `indices`, axis by axis (`pl.ds(index * extent,
    extent)`): a step's tile of a stream's view, as a `BlockSpec`'s index map reads it (a GPU ring's
    `rings.viewed`, a TPU core's copy, `mosaic_tpu.Streams.copy`)."""
    return ref.at[tuple(pl.ds(index * extent, extent) for index, extent in zip(indices, block, strict=True))]


def placed[T](axes: Sequence[str], by_axis: Mapping[str, T],
              rest: T | slice = WHOLE) -> tuple[T | slice, ...]:
    """`by_axis[axis]` for each of `axes` it states, `rest` for the others: a view whole along every axis but
    those it cuts (`rest` a whole slice), or a step's block indices along its walked axes (`rest` the block
    index of each other axis)."""
    return tuple(by_axis.get(axis, rest) for axis in axes)


def held(axes: Sequence[str]) -> tuple[str, ...]:
    """The axes of a side operand's tile as a launch holds it: all but `GROUPS`, along which it reads one
    group's."""
    return tuple(axis for axis in axes if axis != GROUPS)


def at_group(axes: Sequence[str], shape: Sequence[int], group) -> dict:
    """The origin along `GROUPS` a launch reads a side operand of `shape` at, none where `axes` name no
    groups: its tile's `group` modulo the operand's groups, as a stacked layer's call numbers its groups
    past the layers before it (`dew.nn.kernels.gemm._grouped`)."""
    return {GROUPS: group % shape[axes.index(GROUPS)]} if GROUPS in axes else {}


def at_rows(array: jax.Array, axes: Sequence[str], group_sizes: jax.Array, rows: int) -> jax.Array:
    """A side operand of one value a group (its `GROUPS` axis, `[G, ...]`) at each of `rows` rows: each
    group's value at each of its rows, a row past the groups' the last's (its output unrouted). Its
    derivative, the repetition's transpose, sums each group's rows' cotangents: the group-aware reduction
    its gradient is. The one group-to-row mapping: XLA's program's (`templates.gemm.sides.rows`) and a
    launch's of no depth (`mosaic_gpu.gmm`). Where an Explicit axis splits the groups (an expert-sharded
    bank's column scales), the rows are placed as the operand is but whole along them, which JAX asks to be
    told (train 1409's review); the group sizes are read whole wherever they lie, an Explicit split of them
    gathered first (train 1437's review: a split `[1, 1, 1, 1]` repeated a replicated side's groups by each
    device's half)."""
    axis = list(axes).index(GROUPS)
    sizes = whole_along(group_sizes.astype(jnp.int32), 0)
    placed = explicit_spec(array)
    whole = None
    if placed is not None and placed.spec[axis] is not None:
        spec = jax.sharding.PartitionSpec(*(None if at == axis else entry
                                            for at, entry in enumerate(placed.spec)))
        whole = jax.sharding.NamedSharding(placed.mesh, spec)
    return jnp.repeat(array, sizes, axis=axis, total_repeat_length=rows, out_sharding=whole)


def by_rows(tile: jax.Array, axes: Sequence[str], rows: int) -> jax.Array:
    """A side tile read at one group (`held`) as the hook takes it: along `GROUPS`, each of `rows` rows the
    group's value, a broadcast of the held tile (a vector's, as every launch's lowering takes one)."""
    if GROUPS not in axes:
        return tile
    kept = tuple(i for i, axis in enumerate(axes) if axis != GROUPS)
    shape = tuple(rows if axis == GROUPS else tile.shape[kept.index(i)] for i, axis in enumerate(axes))
    return jax.lax.broadcast_in_dim(tile, shape, kept)
