"""Named axes (design §6, the view vocabulary's first part): a template operand's axes by launch name ('rows',
'depth', 'columns'), and what a launch takes of it, axis by axis, from facts it states by name: a block's
extents, a view's slices, a step's block indices. Every backend's launch reads them so (Mosaic GPU's and TPU's
mainloops, Pallas-Triton's), instead of each spelling the per-axis comprehension its own way."""

from collections.abc import Mapping, Sequence

import jax

WHOLE = slice(None)
"""An axis taken whole (a view's slice along an axis it does not cut)."""

GROUPS = 'groups'
"""A side operand's axis of one value a group (`dew.nn.kernels.gemm`'s, row 3963): a launch reads the operand at its
tile's group (`at_group`), the axis indexed away (`held`), and hands its hook the value at each of the tile's
rows (`by_rows`), as the portable program reads it repeated at each row."""


def named[T](axes: Sequence[str], by_axis: Mapping[str, T]) -> tuple[T, ...]:
    """`by_axis[axis]` for each of `axes`, in their order: an operand's block extents or origins, every axis
    stated."""
    return tuple(by_axis[axis] for axis in axes)


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


def by_rows(tile: jax.Array, axes: Sequence[str], rows: int) -> jax.Array:
    """A side tile read at one group (`held`) as the hook takes it: along `GROUPS`, each of `rows` rows the
    group's value, a broadcast of the held tile (a vector's, as every launch's lowering takes one)."""
    if GROUPS not in axes:
        return tile
    kept = tuple(i for i, axis in enumerate(axes) if axis != GROUPS)
    shape = tuple(rows if axis == GROUPS else tile.shape[kept.index(i)] for i, axis in enumerate(axes))
    return jax.lax.broadcast_in_dim(tile, shape, kept)
