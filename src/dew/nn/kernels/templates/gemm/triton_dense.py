"""The grouped GEMM template's dense products in Pallas Triton: one group of every row, so a plain tiled
matmul, each program one output tile over the whole contraction, with no group to find and no row to mask
(a ragged kernel's work at one group cost 1.6-1.8x on an A100, where this kernel ran within 1.15x of
cuBLAS, GMMDENSE-a100).

`gmm` is `lhs [rows, depth] x rhs [depth, width]` (`transpose_rhs`: `[width, depth]`), the forward and the
input gradient; `tgmm` is `lhs` transposed times `rhs [rows, width]`, the kernel gradient. Both take the
template's hooks as the ragged kernels do (`ragged_dot.gmm`, `ragged_dot.tgmm`): a prologue on each tile of
`lhs`'s rows and its side tiles, an epilogue on the fp32 accumulator (`gmm`), side operands following the
launch axes each names, brought in by the launch's block specs. Rows, depth or width the tiles do not
divide are padded with zeros and dropped; what each contracts (`gmm`'s depth, `tgmm`'s rows) is zeroed
again after the prologue, which need not keep it zero.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from dew.nn.sharding import per_shard


def gmm(lhs: jax.Array, rhs: jax.Array, *, tile, transpose_rhs: bool, out_dtypes: tuple, interpret: bool,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (), exact_sums: bool = False,
        paired: jax.Array | None = None) -> tuple:
    """`lhs` times `rhs` in each of `out_dtypes` (the epilogue's outputs, or the accumulator rounded), a
    `tile` (`grouped_matmul.Tile`) an output tile and its depth step; `exact_sums` adds each step's product
    into the fp32 sum by an IEEE add (`_summed`). With `paired`, a second matrix of `rhs`'s shape (an up
    projection beside a gate's), each program multiplies its `lhs` tile by both at the same columns into
    two accumulators, which the epilogue takes together (`epilogue(acc, paired_acc, *sides)`)."""
    if paired is not None and epilogue is None:
        raise ValueError("a paired product needs an epilogue to combine its two accumulators")
    (rows, depth), width = lhs.shape, rhs.shape[0] if transpose_rhs else rhs.shape[1]
    block_m, block_n, block_k = tile.block_m, tile.block_n, tile.block_k
    # A tile that does not divide the rows, depth or width takes them padded with zeros, which add nothing
    # and are dropped (the launch prefers tiles that divide); padded depth is zeroed again after the
    # prologue, which need not keep it zero.
    depth_axis, width_axis = (1, 0) if transpose_rhs else (0, 1)
    lhs = _padded_axis(_padded_axis(lhs, 1, block_k), 0, block_m)
    rhs = _padded_axis(_padded_axis(rhs, depth_axis, block_k), width_axis, block_n)
    matrices = (rhs,) if paired is None else (
        rhs, _padded_axis(_padded_axis(paired, depth_axis, block_k), width_axis, block_n))
    (padded, padded_depth), padded_width = lhs.shape, rhs.shape[width_axis]
    prologue_operands = _padded_on(_padded_on(prologue_operands, prologue_axes, 'depth', block_k),
                                   prologue_axes, 'rows', block_m)
    epilogue_operands = _padded_on(_padded_on(epilogue_operands, epilogue_axes, 'columns', block_n),
                                   epilogue_axes, 'rows', block_m)
    blocks = {'rows': (block_m, 0), 'depth': (padded_depth, None), 'columns': (block_n, 1)}

    def spec(axes):
        return pl.BlockSpec(tuple(blocks[axis][0] for axis in axes),
                            lambda i, j: tuple(0 if blocks[axis][1] is None else (i, j)[blocks[axis][1]]
                                               for axis in axes))

    matrix_spec = (pl.BlockSpec((block_n, padded_depth), lambda i, j: (j, 0)) if transpose_rhs
                   else pl.BlockSpec((padded_depth, block_n), lambda i, j: (0, j)))
    in_specs = [_LIVE_SPEC, pl.BlockSpec((block_m, padded_depth), lambda i, j: (i, 0)),
                *(matrix_spec for _ in matrices),
                *(spec(axes) for axes in prologue_axes), *(spec(axes) for axes in epilogue_axes)]
    kernel = functools.partial(
        _gmm_kernel, depth=depth, block_k=block_k, transpose_rhs=transpose_rhs, prologue=prologue,
        prologue_axes=prologue_axes, epilogue=epilogue,
        sides=(len(prologue_operands), len(epilogue_operands)), exact_sums=exact_sums,
        matrices=len(matrices))
    per_shard("the dense GEMM's Triton launch")
    outs = pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((padded, padded_width), dtype) for dtype in out_dtypes),
        grid=(padded // block_m, padded_width // block_n),
        in_specs=in_specs,
        out_specs=tuple(pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)) for _ in out_dtypes),
        compiler_params=plgpu.CompilerParams(num_warps=tile.warps, num_stages=tile.stages),
        interpret=interpret,
        name="pallas_triton_dense_dot",
    )(_live(rows), lhs, *matrices, *prologue_operands, *epilogue_operands)
    return tuple(out[:rows, :width] for out in outs)


def _gmm_kernel(live_ref, lhs_ref, *refs, depth: int, block_k: int, transpose_rhs: bool, prologue,
                prologue_axes: tuple, epilogue, sides: tuple, exact_sums: bool, matrices: int):
    matrix_refs, refs = refs[:matrices], refs[matrices:]
    prologue_refs, epilogue_refs, out_refs = refs[:sides[0]], refs[sides[0]:sum(sides)], refs[sum(sides):]
    live = live_ref[0] > 0

    def step(k, accs):
        span = pl.ds(k * block_k, block_k)
        x = lhs_ref[:, span]
        if prologue is not None:
            x = prologue(x, *_cut(prologue_refs, prologue_axes, 'depth', span)).astype(x.dtype)
            if depth % block_k:
                x = _kept(x, k * block_k, depth, axis=1)
        # A paired product's two matrices multiply the one loaded (and transformed) tile of `lhs`.
        return tuple(
            _summed(acc, jnp.dot(x, ref[:, span].T if transpose_rhs else ref[span, :],
                                 precision=lax.Precision.DEFAULT, preferred_element_type=jnp.float32),
                    live, exact_sums=exact_sums)
            for acc, ref in zip(accs, matrix_refs, strict=True))

    zeros = jnp.zeros(out_refs[0].shape, jnp.float32)
    accs = lax.fori_loop(0, lhs_ref.shape[1] // block_k, step, (zeros,) * matrices)
    outs = accs if epilogue is None else epilogue(*accs, *(ref[...] for ref in epilogue_refs))
    for ref, out in zip(out_refs, outs if isinstance(outs, tuple) else (outs,), strict=True):
        ref[...] = out.astype(ref.dtype)


def tgmm(lhs: jax.Array, rhs: jax.Array, *, tile, out_dtype, interpret: bool, prologue=None,
         prologue_operands: tuple = (), prologue_axes: tuple = (), exact_sums: bool = False) -> jax.Array:
    """`prologue(lhs)` transposed times `rhs`: `[depth, width]` in `out_dtype`, a `tile` a `block_k` x
    `block_n` output tile summed over the rows `block_m` at a time (`grouped_matmul.Tgmm`'s tiles)."""
    (rows, depth), width = lhs.shape, rhs.shape[1]
    block_m, block_n, block_k = tile.block_m, tile.block_n, tile.block_k
    # Padded as `gmm`'s: the depth and width dropped from the gradient, the rows it sums zeroed again
    # after the prologue.
    lhs = _padded_axis(_padded_axis(lhs, 1, block_k), 0, block_m)
    rhs = _padded_axis(_padded_axis(rhs, 1, block_n), 0, block_m)
    (padded, padded_depth), padded_width = lhs.shape, rhs.shape[1]
    prologue_operands = _padded_on(_padded_on(prologue_operands, prologue_axes, 'depth', block_k),
                                   prologue_axes, 'rows', block_m)
    blocks = {'rows': (padded, None), 'depth': (block_k, 0)}

    def spec(axes):
        return pl.BlockSpec(tuple(blocks[axis][0] for axis in axes),
                            lambda i, j: tuple(i if blocks[axis][1] == 0 else 0 for axis in axes))

    kernel = functools.partial(_tgmm_kernel, rows=rows, block_m=block_m, prologue=prologue,
                               prologue_axes=prologue_axes, exact_sums=exact_sums)
    per_shard("the dense kernel gradient's Triton launch")
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_depth, padded_width), out_dtype),
        grid=(padded_depth // block_k, padded_width // block_n),
        in_specs=[_LIVE_SPEC, pl.BlockSpec((padded, block_k), lambda i, j: (0, i)),
                  pl.BlockSpec((padded, block_n), lambda i, j: (0, j)),
                  *(spec(axes) for axes in prologue_axes)],
        out_specs=pl.BlockSpec((block_k, block_n), lambda i, j: (i, j)),
        compiler_params=plgpu.CompilerParams(num_warps=tile.warps, num_stages=tile.stages),
        interpret=interpret,
        name="pallas_triton_dense_tdot",
    )(_live(rows), lhs, rhs, *prologue_operands)[:depth, :width]


def _tgmm_kernel(live_ref, lhs_ref, rhs_ref, *refs, rows: int, block_m: int, prologue, prologue_axes: tuple,
                 exact_sums: bool):
    *prologue_refs, out_ref = refs
    live = live_ref[0] > 0

    def step(r, acc):
        span = pl.ds(r * block_m, block_m)
        x = lhs_ref[span, :]
        if prologue is not None:
            x = prologue(x, *_cut(prologue_refs, prologue_axes, 'rows', span)).astype(x.dtype)
            if rows % block_m:
                x = _kept(x, r * block_m, rows, axis=0)
        product = jnp.dot(x.T, rhs_ref[span, :], precision=lax.Precision.DEFAULT,
                          preferred_element_type=jnp.float32)
        return _summed(acc, product, live, exact_sums=exact_sums)

    acc = lax.fori_loop(0, lhs_ref.shape[0] // block_m, step, jnp.zeros(out_ref.shape, jnp.float32))
    out_ref[...] = acc.astype(out_ref.dtype)


_LIVE_SPEC = pl.BlockSpec((1,), lambda i, j: (0,))
"""The block of `_live`'s count every program reads."""


def _live(rows: int) -> jax.Array:
    """The call's row count, which a kernel reads from memory: a condition on it, always true where a
    program runs, the compiler cannot prove (a program id on a one-program axis it can: Pallas Triton
    lowers it to zero)."""
    return jnp.full((1,), rows, jnp.int32)


def _summed(acc, partial, live, *, exact_sums: bool):
    """The running fp32 sum plus a step's product: with `exact_sums` an IEEE add of the product summed
    apart, behind a selection on `live` (always true, unknown to the compiler), which keeps the add from
    folding into the MMA's accumulator (`ragged_dot._summed`)."""
    return acc + jnp.where(live, partial, 0) if exact_sums else acc + partial


def _kept(x: jax.Array, start, extent: int, *, axis: int) -> jax.Array:
    """`x`, a tile at `start` along `axis`, with its entries at `extent` or past it zero."""
    kept = start + lax.broadcasted_iota(jnp.int32, x.shape, axis) < extent
    return jnp.where(kept, x, jnp.zeros_like(x))


def _padded_axis(array: jax.Array, axis: int, block: int) -> jax.Array:
    """`array` padded with zeros along `axis` to a multiple of `block`."""
    extra = -array.shape[axis] % block
    if not extra:
        return array
    widths = [(0, 0)] * array.ndim
    widths[axis] = (0, extra)
    return jnp.pad(array, widths)


def _padded_on(sides: tuple, axes_of: tuple, axis_name: str, block: int) -> tuple:
    """Each side operand padded along the dimension that follows `axis_name`, where one does."""
    return tuple(_padded_axis(side, axes.index(axis_name), block) if axis_name in axes else side
                 for side, axes in zip(sides, axes_of, strict=True))


def _cut(refs, axes_of: tuple, stepped: str, span) -> list:
    """Each side tile of a step: its block cut to `span` along the `stepped` launch axis, whole along the
    others."""
    return [ref[tuple(span if axis == stepped else slice(None) for axis in axes)]
            for ref, axes in zip(refs, axes_of, strict=True)]
