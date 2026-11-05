"""The grouped GEMM template's kernels: Pallas Mosaic GPU (`mosaic_gpu`) and TPU (`mosaic_tpu`), with the
padding they and the Pallas-Triton dense kernels (`triton_dense`) lay their operands out in."""

from collections.abc import Sequence

import jax
from jax import numpy as jnp


def rows_extended(array: jax.Array, axis: int, extent: int) -> jax.Array:
    """`array` with zeros after its own along `axis` up to `extent`, where it holds fewer."""
    pad = extent - array.shape[axis]
    widths = tuple((0, pad if dim == axis else 0) for dim in range(array.ndim))
    return jnp.pad(array, widths) if pad > 0 else array


def rows_padded(array: jax.Array, axis: int, multiple: int) -> jax.Array:
    """`array` with zeros after its own along `axis` up to a multiple of `multiple`: the row blocks a
    kernel's loads or its group metadata tile by."""
    return rows_extended(array, axis, -(-array.shape[axis] // multiple) * multiple)


def sides_padded(sides: Sequence[jax.Array], axes_of: Sequence[tuple[str, ...]], multiple: int | None = None,
                 *, extent: int | None = None, along: str = 'rows') -> tuple[jax.Array, ...]:
    """Each side operand (a prologue's or an epilogue's, `axes_of` naming each one's dimensions by the
    launch axis it follows) padded along the dimension that follows `along` as the operand it follows is:
    to a multiple of `multiple`, or to `extent`; one that follows no `along` as it is."""
    def padded(side: jax.Array, axes: tuple[str, ...]) -> jax.Array:
        if along not in axes:
            return side
        axis = axes.index(along)
        return rows_extended(side, axis, extent) if multiple is None else rows_padded(side, axis, multiple)
    return tuple(padded(side, axes) for side, axes in zip(sides, axes_of, strict=True))
