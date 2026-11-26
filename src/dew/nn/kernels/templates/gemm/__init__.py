"""The grouped GEMM template's kernels: Pallas Mosaic GPU (`mosaic_gpu`) and TPU (`mosaic_tpu`), with the
padding they and the Pallas-Triton dense kernels (`triton_dense`) lay their operands out in."""

from collections.abc import Sequence

import jax
from jax import numpy as jnp

from dew.nn.kernels.templates.axes import whole
from dew.nn.sharding import explicit_spec


def rows_extended(array: jax.Array, axis: int, extent: int) -> jax.Array:
    """`array` with zeros after its own along `axis` up to `extent`, where it holds fewer."""
    pad = extent - array.shape[axis]
    widths = tuple((0, pad if dim == axis else 0) for dim in range(array.ndim))
    return jnp.pad(array, widths) if pad > 0 else array


def rows_padded(array: jax.Array, axis: int, multiple: int) -> jax.Array:
    """`array` with zeros after its own along `axis` up to a multiple of `multiple`: the row blocks a
    kernel's loads or its group metadata tile by."""
    return rows_extended(array, axis, whole(array.shape[axis], multiple))


def sides_padded(sides: Sequence[jax.Array], axes_of: Sequence[tuple[str, ...]], multiple: int, *,
                 along: str = 'rows') -> tuple[jax.Array, ...]:
    """Each side operand (a prologue's or an epilogue's, `axes_of` naming each one's dimensions by the
    launch axis it follows) padded along the dimension that follows `along` to a multiple of `multiple`, as
    the operand it follows is; one that follows no `along` as it is."""
    return tuple(rows_padded(side, axes.index(along), multiple) if along in axes else side
                 for side, axes in zip(sides, axes_of, strict=True))


def sides_extended(sides: Sequence[jax.Array], axes_of: Sequence[tuple[str, ...]],
                   extent: int) -> tuple[jax.Array, ...]:
    """Each side operand (`sides_padded`'s) extended with zeros along the dimension that follows the rows to
    `extent`, as the rows are; one that follows no rows as it is."""
    return tuple(rows_extended(side, axes.index('rows'), extent) if 'rows' in axes else side
                 for side, axes in zip(sides, axes_of, strict=True))


def rows_permuted(x: jax.Array, permutation: jax.Array, axis: int = 0) -> jax.Array:
    """`x`'s rows along `axis` at `permutation`'s order, placed as `x` is where an Explicit axis splits it
    (a permutation of its rows, whose placement its own is): JAX takes no gather of a split axis without its
    output's placement."""
    placed = explicit_spec(x)
    if placed is None:
        return jnp.take(x, permutation, axis=axis)
    return x.at[(slice(None),) * axis + (permutation,)].get(out_sharding=placed)


def rows_gathered_wide(x: jax.Array, rows: jax.Array) -> jax.Array:
    """`x`'s rows at `rows` (`x[rows]`), gathered through its wide form (exactly: at least fp32), so the
    transpose sums the cotangents of every place a row is read at wide and rounds once: a token's routes
    (`gemm`'s portable gather, the expert exchange's, row 12396)."""
    from dew.nn.precision import at_least_fp32

    return x.astype(at_least_fp32(x.dtype))[rows].astype(x.dtype)


def scattered(outs, scatter: jax.Array | None):
    """`outs` (one output or a tuple), each row `i` written at row `scatter[i]` (a product's `scatter`, a
    permutation, so every row is written once), as XLA's program and a launch's stores give them; as they
    are without one. An output an Explicit axis splits along its rows is read at the inverse permutation
    instead, placed as it was (`rows_permuted`): the same rows, where JAX takes no scatter into a split axis
    without its output's placement."""
    if scatter is None:
        return outs

    def written(out):
        if explicit_spec(out) is None:
            return jnp.zeros_like(out).at[scatter].set(out, unique_indices=True)
        return rows_permuted(out, jnp.argsort(scatter))

    return jax.tree.map(written, outs)
