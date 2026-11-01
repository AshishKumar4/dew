"""Named axes (design §6, the view vocabulary's first part): a template operand's axes by launch name ('rows',
'depth', 'columns'), and what a launch takes of it, axis by axis, from facts it states by name: a block's
extents, a view's slices, a step's block indices. Every backend's launch reads them so (Mosaic GPU's and TPU's
mainloops, Pallas-Triton's), instead of each spelling the per-axis comprehension its own way."""

from collections.abc import Mapping, Sequence

WHOLE = slice(None)
"""An axis taken whole (a view's slice along an axis it does not cut)."""


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
