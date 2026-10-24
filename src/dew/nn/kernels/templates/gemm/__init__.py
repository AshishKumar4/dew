"""The grouped GEMM template's kernels: Pallas Mosaic GPU (`mosaic_gpu`) and TPU (`mosaic_tpu`), with the
padding both lay their operands' rows out in."""

import jax
from jax import numpy as jnp


def rows_padded(array: jax.Array, axis: int, multiple: int) -> jax.Array:
    """`array` with zeros after its own along `axis` up to a multiple of `multiple`: the row blocks a
    kernel's loads or its group metadata tile by."""
    pad = -array.shape[axis] % multiple
    widths = tuple((0, pad if dim == axis else 0) for dim in range(array.ndim))
    return jnp.pad(array, widths) if pad else array
