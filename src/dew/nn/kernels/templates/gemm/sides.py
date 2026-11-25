"""Dew: a GEMM's hooks' side operands (`templates.hooks.Side`) as its launches take them: each checked against
the launch axes its hook's operands follow (`split`), read at each row by XLA's program (`rows`), squeezed of
its dimensions of one for a launch (`launchable`), or taken as an output's tiles (`tiles`)."""

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import axes as side_axes_of
from dew.nn.kernels.templates.axes import GROUPS
from dew.nn.kernels.templates.hooks import Hook, Side

LAUNCH_AXES = {'prologue': ('rows', 'depth'), 'epilogue': ('rows', 'columns')}
"""The launch axes each hook's side operands may follow; `groups` too beside them (`split`), which a launch
reads at its tile's group (`axes.GROUPS`) and XLA's program at each row (`axes.at_rows`)."""


def split(sides: tuple[Side, ...], hook: str, group_sizes: jax.Array | None) -> tuple[tuple, tuple]:
    """A hook's side operands' arrays and their axes (`hooks.Side`), each checked: one axis per dimension,
    each a launch axis the hook's operands follow, `groups` beside no rows (one value a group of the call's,
    a dense product's one group: `axes.GROUPS`) or None for a dimension of one, at least one a launch
    axis."""
    groups = 1 if group_sizes is None else group_sizes.shape[0]
    arrays, axes_of = [], []
    for side in sides:
        array, axes = side.value, tuple(side.axes)
        named = [axis for axis in axes if axis not in (None, GROUPS)]
        if (len(axes) != array.ndim or not named or any(axis not in LAUNCH_AXES[hook] for axis in named)
                or any(extent != 1 for axis, extent in zip(axes, array.shape, strict=True) if axis is None)):
            raise ValueError(f"a {hook} side operand of shape {array.shape} does not take axes {axes}: one "
                             f"per dimension, each one of {LAUNCH_AXES[hook]}, {GROUPS!r} beside one, or "
                             f"None for a dimension of one")
        if GROUPS in axes and ('rows' in axes or axes.count(GROUPS) != 1
                               or array.shape[axes.index(GROUPS)] != groups):
            raise ValueError(f"a {hook} side operand of shape {array.shape} on axes {axes} names groups "
                             f"once, beside no rows, at the call's {groups} groups")
        arrays.append(array)
        axes_of.append(axes)
    return tuple(arrays), tuple(axes_of)


def rows(arrays: tuple, axes_of: tuple, group_sizes: jax.Array, rows: int) -> tuple:
    """The side operands as XLA's program reads them: a `groups` axis as `rows` rows, each its group's
    value (`axes.at_rows`), every other operand as it is."""
    return tuple(side_axes_of.at_rows(array, axes, group_sizes, rows) if GROUPS in axes else array
                 for array, axes in zip(arrays, axes_of, strict=True))


def launchable(arrays: tuple, axes_of: tuple) -> tuple[tuple, tuple, tuple]:
    """The side operands and axes a launch takes, each dimension of one squeezed out, and each one's
    positions of them, which the launch gives the hook back (`Hooks`)."""
    positions = tuple(tuple(i for i, axis in enumerate(axes) if axis is None) for axes in axes_of)
    squeezed = tuple(jnp.squeeze(array, where) if where else array
                     for array, where in zip(arrays, positions, strict=True))
    named = tuple(tuple(axis for axis in axes if axis is not None) for axes in axes_of)
    return squeezed, named, positions if any(positions) else ()


def tiles(epilogue: Hook) -> tuple[jax.Array, ...]:
    """A finished product's epilogue's side operands, each a tile of the output's (`[rows, columns]`,
    `LAUNCH_AXES['epilogue']`), as the launch takes them."""
    if any(tuple(side.axes) != LAUNCH_AXES['epilogue'] for side in epilogue.sides):
        raise ValueError(f"a finished product's epilogue takes side tiles on {LAUNCH_AXES['epilogue']}, not "
                         f"{[side.axes for side in epilogue.sides]}")
    return tuple(side.value for side in epilogue.sides)
