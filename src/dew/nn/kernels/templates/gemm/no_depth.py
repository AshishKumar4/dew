"""A `gmm` call of no depth or no rows (`mosaic_gpu.gmm`): its outputs with no launch, its epilogue on zero
accumulators, each output in its form."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from dew.nn.kernels.templates import axes
from dew.nn.kernels.templates.gemm import quantized, scattered

if TYPE_CHECKING:
    from dew.nn.kernels.templates.gemm.mosaic_gpu import Beside


def outputs(beside: Beside | None, shape: tuple, group_sizes, paired, scatter, dtypes: tuple, out_dtypes,
            epilogue: tuple, forms: tuple = ()):
    """`gmm`'s outputs of a call of no depth or no rows: each matrix side by side's zeros, or its
    epilogue's value on zero accumulators (`of_no_depth`), a scattered product's rows where its table says,
    as its stores put them."""
    rows, n = shape
    if beside is not None:
        return tuple(jnp.zeros((rows, width), dtypes[0]) for width in beside.widths)
    hook, operands, axes_of = epilogue
    outs = quantized.stored_all(scattered(of_no_depth((rows, n), group_sizes, 1 if paired is None else 2,
                                                      hook, operands, axes_of, dtypes), scatter), forms)
    return outs if out_dtypes else outs[0]



def of_no_depth(shape: tuple, group_sizes, accumulators: int, epilogue, epilogue_operands: tuple,
                epilogue_axes: tuple, dtypes: tuple) -> tuple:
    """`gmm`'s outputs of a product of no depth, which sums nothing, or of no rows, which has no tile to
    through the epilogue on whole arrays (its hooks are a tile's elementwise arithmetic), with no launch (a
    walk of no steps has no block of the matrix to view: JAX 0.12 refuses a zero-extent slice; a block of no
    rows none to tile: the interpreter's padding divided by it); its side operands by group read at each row
    as XLA's program reads them (`axes.at_rows`, row 4489)."""
    rows = shape[0]
    zeros = (jnp.zeros(shape, jnp.float32),) * accumulators
    by_row = tuple(axes.at_rows(side, side_axes, group_sizes, rows) if axes.GROUPS in side_axes else side
                   for side, side_axes in zip(epilogue_operands, epilogue_axes, strict=True))
    values = zeros if epilogue is None else epilogue(*zeros, *by_row)
    values = values if isinstance(values, tuple) else (values,)
    return tuple(value.astype(dtype) for value, dtype in zip(values, dtypes, strict=True))
