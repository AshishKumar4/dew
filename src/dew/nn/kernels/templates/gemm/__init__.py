"""The grouped GEMM template's kernels: Pallas Mosaic GPU (`mosaic_gpu`) and TPU (`mosaic_tpu`), with the
padding both lay their operands' rows out in and the cost both state."""

import math

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl


def rows_padded(array: jax.Array, axis: int, multiple: int) -> jax.Array:
    """`array` with zeros after its own along `axis` up to a multiple of `multiple`: the row blocks a
    kernel's loads or its group metadata tile by."""
    pad = -array.shape[axis] % multiple
    widths = tuple((0, pad if dim == axis else 0) for dim in range(array.ndim))
    return jnp.pad(array, widths) if pad else array


def cost(flops: int, *arrays) -> pl.CostEstimate:
    """A GEMM kernel's cost as its launch states it to XLA and to Dew's FLOP count (the custom call's
    cost estimate, `dew.telemetry.instrumentation.sizes`): its products' multiply-adds, `flops`, and the
    bytes of each of `arrays`, its operands and outputs, once."""
    return pl.CostEstimate(flops=int(flops), transcendentals=0, bytes_accessed=int(sum(
        math.prod(array.shape) * jnp.dtype(array.dtype).itemsize for array in arrays)))
