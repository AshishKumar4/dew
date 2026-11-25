"""Dew: `gemm`'s arithmetic in XLA, the template's portable mainloop (docs/design/kernel-templates.md's
`xla_mainloop`): the program a call runs where no launch takes it, and every derivative's tangent."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike

from dew.nn.kernels import xla_products
from dew.nn.kernels.templates import axes as side_axes_of
from dew.nn.kernels.templates.axes import GROUPS
from dew.nn.precision import at_least_fp32, effective_precision, precision_names
from dew.nn.sharding import explicit_spec


@dataclasses.dataclass(frozen=True)
class XlaConfig:
    """The plain-JAX mainloop's config (`xla_mainloop`): whether the matrices lie `[groups, width, depth]`
    (`transpose_rhs`), the products' `precision`, the dtype the operands round to before them (`compute`,
    None for their own), whether the call is `dense` (one group of every row), and whether a primal-only
    CUDA XLA fallback may emit a direct bf16 dot (`direct_bf16`, never set on a differentiated `portable`
    program so JVPs still sum in fp32 before rounding)."""

    transpose_rhs: bool
    precision: PrecisionLike
    compute: jnp.dtype | None = None
    dense: bool = False
    direct_bf16: bool = False


_PLAIN_PRECISIONS = frozenset({'DEFAULT', 'HIGH', 'HIGHEST'})
"""Precision levels (`precision_names`) naming no explicit dot algorithm (`direct_dtype`)."""


def _local_contraction(lhs, lhs_axis: int, rhs, rhs_axis: int) -> bool:
    """Whether a dense contraction of `lhs` along `lhs_axis` against `rhs` along `rhs_axis` runs whole on
    each device, with no SPMD partial reduction across devices: neither the context mesh nor either operand's
    sharding carries an automatic axis larger than one device (`auto_axes`), no Explicit axis splits either
    operand's contracting dimension (`explicit_spec`)."""
    meshes = [jax.sharding.get_abstract_mesh()]
    for operand, axis in ((lhs, lhs_axis), (rhs, rhs_axis)):
        meshes.append(jax.typeof(operand).sharding.mesh)
        placed = explicit_spec(operand)
        if placed is not None and axis < len(placed.spec) and placed.spec[axis] is not None:
            return False
    return not any(mesh.shape[axis] > 1 for mesh in meshes for axis in mesh.auto_axes)


def direct_dtype(lhs, lhs_axis: int, rhs, rhs_axis: int, out_dtype, precision: PrecisionLike, wide):
    """Output dtype for a primal-only CUDA dense XLA dot (`xla_mainloop`, `_gemm_t`, row 12898): bfloat16
    directly when both operands and output are bfloat16, neither the call nor `jax_default_matmul_precision`
    asks for an explicit dot algorithm (`_PLAIN_PRECISIONS`), and the contraction is local to each device
    (`_local_contraction`, so no SPMD partials are rounded before their reduction). cuBLASLt accumulates bf16
    products in fp32 and rounds once in its epilogue without writing an fp32 temporary or trailing convert
    that breaks downstream reduction fusion; `wide` otherwise."""
    if (jnp.dtype(lhs.dtype) == jnp.bfloat16 and jnp.dtype(rhs.dtype) == jnp.bfloat16
            and jnp.dtype(out_dtype) == jnp.bfloat16
            and precision_names(effective_precision(precision)) <= _PLAIN_PRECISIONS
            and _local_contraction(lhs, lhs_axis, rhs, rhs_axis)):
        return jnp.bfloat16
    return wide


def xla_mainloop(config: XlaConfig, lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                 epilogue_arrays, epilogue_axes, dtypes, several: bool, *, paired: jax.Array | None = None,
                 first: jax.Array | None = None, joined: jax.Array | None = None, beside: tuple = (),
                 chained: tuple = ()):
    """The template's portable mainloop at `config` (docs/design/kernel-templates.md's `xla_mainloop`):
    `gemm`'s arithmetic in XLA, the kernels' products of `lhs`'s dtype summed in at least fp32 (an fp64
    product's in fp64, which no kernel takes), one rounding, a dense call's as XLA's own dot
    (`xla_products.xla_dot`: one group of every row, nothing to mask, no ragged dot to trace);
    with `config.compute`, the products of the operands rounded to it, its tangent the MoE projection's
    (`xla_products.portable_projection`, whose sum, at least fp32, it reads); with `paired`, both matrices'
    products, the epilogue's two accumulators; with `first`, the groups' matrices of a stacked `rhs` from
    that one (`_grouped`), sliced out, as XLA's ragged dot reads a matrix a group, a pair's second stack's
    alike; with `joined` too, the paired matrix multiplies those rows, the two accumulators added once
    (`mosaic_gpu.joined_sum`, the launch's own finish) where no epilogue is given; with `beside`, each of
    those matrices' products too, an output each; with `chained`, each `(rows, matrix)` product too, each
    summed apart and added in order into the one output (the plain program's sum)."""
    from dew.nn.kernels.templates.gemm import mosaic_gpu

    transpose_rhs, precision, compute, dense = (config.transpose_rhs, config.precision, config.compute,
                                                config.dense)
    if first is not None:
        def layer_of(stack: jax.Array) -> jax.Array:
            return jax.lax.dynamic_slice_in_dim(stack, first, group_sizes.shape[0])
        rhs = layer_of(rhs)
        paired = None if paired is None else layer_of(paired)
    matrices = tuple(jnp.swapaxes(each, 1, 2) if transpose_rhs else each
                     for each in (rhs, *(() if paired is None else (paired,)), *beside,
                                  *(matrix for _, matrix in chained)))
    rows = ((lhs, *(each for each, _ in chained)) if chained else (lhs,) * len(matrices) if joined is None
            else (lhs, joined))
    epilogue = mosaic_gpu.joined_sum if joined is not None else epilogue
    prologue_arrays = side_rows(prologue_arrays, prologue_axes, group_sizes, lhs.shape[0])
    epilogue_arrays = side_rows(epilogue_arrays, epilogue_axes, group_sizes, lhs.shape[0])
    if compute is not None:
        # Summed and kept at least fp32 (an fp64 product's in fp64), the accumulator the epilogue reads.
        wide = at_least_fp32(compute)
        products = tuple(row if prologue is None
                         else prologue(row.astype(compute), *prologue_arrays).astype(compute)
                         for row in rows)
        accs = tuple(xla_products.portable_projection(product, matrix, group_sizes, compute, precision, wide)
                     for product, matrix in zip(products, matrices, strict=True))
    else:
        products = tuple(row if prologue is None else prologue(row, *prologue_arrays).astype(lhs.dtype)
                         for row in rows)
        wide = at_least_fp32(lhs.dtype)
        direct = (config.direct_bf16 and dense and epilogue is None and not chained
                  and len(products) == len(dtypes))
        accs = []
        for index, (product, each) in enumerate(zip(products, matrices, strict=True)):
            if dense:
                matrix = each[0].astype(lhs.dtype)
                target = (direct_dtype(product, 1, matrix, 0, dtypes[index], precision, wide)
                          if direct else wide)
                accs.append(xla_products.xla_dot(product, matrix, precision=precision,
                                                 preferred_element_type=target))
            else:
                accs.append(xla_products.xla_ragged_dot(product, each.astype(lhs.dtype), group_sizes,
                                                        precision=precision, preferred_element_type=wide))
        accs = tuple(accs)
    accs = (functools.reduce(jnp.add, accs),) if chained else accs
    outs = accs if epilogue is None else as_tuple(epilogue(*accs, *epilogue_arrays))
    # `astype` passes an array already of its strong dtype through; a weak one an epilogue made takes it.
    outs = tuple(out.astype(dtype) for out, dtype in zip(outs, dtypes, strict=True))
    return outs if several else outs[0]


def side_rows(arrays: tuple, axes_of: tuple, group_sizes: jax.Array, rows: int) -> tuple:
    """The side operands as XLA's program reads them: a `groups` axis as `rows` rows, each its group's
    value (`axes.at_rows`), every other operand as it is."""
    return tuple(side_axes_of.at_rows(array, axes, group_sizes, rows) if GROUPS in axes else array
                 for array, axes in zip(arrays, axes_of, strict=True))


def as_tuple(outs) -> tuple:
    """`outs`, one array or a tuple of them, as a tuple."""
    return outs if isinstance(outs, tuple) else (outs,)
