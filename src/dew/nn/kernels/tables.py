"""An int32 table's layout declared once: a `NamedTuple` of its columns, its writer stacking a record in that
order (`packed`) and its readers taking a row as the record (`unpacked`) or a field's column by name
(`column`), so no writer or reader states a column's position. The recurrence's descriptors
(`chunked_launch.Span`, `ChunkSpan`, the template's `mainloop.Walked`) and the GEMM's tile groups read it."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Column = jax.Array | np.ndarray
"""A table field's values: an array's column, or a ref's row's scalar load in a kernel (`unpacked`)."""


def packed(record: NamedTuple, xp=jnp):
    """`record`'s fields, each `[N]`, as an int32 table `[N, F]` in its declaration's order (`Span`,
    `ChunkSpan`, ...): the one statement a table's writer (`packed`) and its readers (`unpacked`,
    `column`) derive its layout from."""
    return xp.stack(record, axis=1).astype(np.int32)


def unpacked(declared: type, at: Callable[[int], "Column"]):
    """A table's row as its declaration `declared` (a `NamedTuple` of its columns, `packed`), each field
    read by `at(column)`: a ref's row's scalar loads in a kernel, or an array's columns."""
    return declared(*(at(i) for i in range(len(declared._fields))))


def column(declared: type, name: str) -> int:
    """Field `name`'s column in a table `declared` lays out (`packed`)."""
    return declared._fields.index(name)
