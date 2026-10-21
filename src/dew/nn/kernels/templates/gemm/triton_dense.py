"""The grouped GEMM template's dense products in Pallas Triton: one group of every row, so a plain tiled
matmul, each program one output tile over the whole contraction, with no group to find and no row to mask
(a ragged kernel's work at one group cost 1.6-1.8x on an A100, where this kernel ran within 1.15x of
cuBLAS, GMMDENSE-a100).

`gmm` is `lhs [rows, depth] x rhs [depth, width]` (`transpose_rhs`: `[width, depth]`), the forward and the
input gradient; `tgmm` is `lhs` transposed times `rhs [rows, width]`, the kernel gradient. Both take the
template's hooks as the ragged kernels do (`ragged_dot.gmm`, `ragged_dot.tgmm`): a prologue on each tile of
`lhs`'s rows and its side tiles, an epilogue on the fp32 accumulator (`gmm`), side operands following the
launch axes each names, brought in by the launch's block specs. Rows the tiles do not divide are padded
with zeros and dropped, or, contracted (`tgmm`), zeroed after the prologue.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def gmm(lhs: jax.Array, rhs: jax.Array, *, tile, transpose_rhs: bool, out_dtypes: tuple, interpret: bool,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (), exact_sums: bool = False) -> tuple:
    """`lhs` times `rhs` in each of `out_dtypes` (the epilogue's outputs, or the accumulator rounded), a
    `tile` (`grouped_matmul.Tile`) an output tile and its depth step; `exact_sums` adds each step's product
    into the fp32 sum by an IEEE add (`_summed`)."""
    rows, depth = lhs.shape
    width = rhs.shape[0] if transpose_rhs else rhs.shape[1]
    block_m, block_n, block_k = tile.block_m, tile.block_n, tile.block_k
    if depth % block_k or width % block_n:
        raise ValueError(f"a [{rows}, {depth}] x [{depth}, {width}] product does not take {tile}")
    lhs, padded = _padded_rows(lhs, block_m)
    prologue_operands = _padded_sides(prologue_operands, prologue_axes, padded)
    epilogue_operands = _padded_sides(epilogue_operands, epilogue_axes, padded)
    blocks = {'rows': (block_m, 0), 'depth': (depth, None), 'columns': (block_n, 1)}

    def spec(axes):
        return pl.BlockSpec(tuple(blocks[axis][0] for axis in axes),
                            lambda i, j: tuple(0 if blocks[axis][1] is None else (i, j)[blocks[axis][1]]
                                               for axis in axes))

    in_specs = [pl.BlockSpec((block_m, depth), lambda i, j: (i, 0)),
                pl.BlockSpec((block_n, depth), lambda i, j: (j, 0)) if transpose_rhs
                else pl.BlockSpec((depth, block_n), lambda i, j: (0, j)),
                *(spec(axes) for axes in prologue_axes), *(spec(axes) for axes in epilogue_axes)]
    kernel = functools.partial(
        _gmm_kernel, steps=depth // block_k, block_k=block_k, transpose_rhs=transpose_rhs,
        prologue=prologue, prologue_axes=prologue_axes, epilogue=epilogue,
        sides=(len(prologue_operands), len(epilogue_operands)), exact_sums=exact_sums)
    outs = pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((padded, width), dtype) for dtype in out_dtypes),
        grid=(padded // block_m, width // block_n),
        in_specs=in_specs,
        out_specs=tuple(pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)) for _ in out_dtypes),
        compiler_params=plgpu.CompilerParams(num_warps=tile.warps, num_stages=tile.stages),
        interpret=interpret,
    )(lhs, rhs, *prologue_operands, *epilogue_operands)
    return tuple(out[:rows] for out in outs)


def _gmm_kernel(lhs_ref, rhs_ref, *refs, steps: int, block_k: int, transpose_rhs: bool, prologue,
                prologue_axes: tuple, epilogue, sides: tuple, exact_sums: bool):
    prologue_refs, epilogue_refs, out_refs = refs[:sides[0]], refs[sides[0]:sum(sides)], refs[sum(sides):]
    live = _unknown_true()

    def step(k, acc):
        depth = pl.ds(k * block_k, block_k)
        x = lhs_ref[:, depth]
        if prologue is not None:
            x = prologue(x, *_cut(prologue_refs, prologue_axes, 'depth', depth)).astype(x.dtype)
        w = rhs_ref[:, depth].T if transpose_rhs else rhs_ref[depth, :]
        return _summed(acc, jnp.dot(x, w, preferred_element_type=jnp.float32), live, exact_sums=exact_sums)

    acc = lax.fori_loop(0, steps, step, jnp.zeros(out_refs[0].shape, jnp.float32))
    outs = (acc,) if epilogue is None else epilogue(acc, *(ref[...] for ref in epilogue_refs))
    for ref, out in zip(out_refs, outs if isinstance(outs, tuple) else (outs,), strict=True):
        ref[...] = out.astype(ref.dtype)


def tgmm(lhs: jax.Array, rhs: jax.Array, *, tile, out_dtype, interpret: bool, prologue=None,
         prologue_operands: tuple = (), prologue_axes: tuple = (), exact_sums: bool = False) -> jax.Array:
    """`prologue(lhs)` transposed times `rhs`: `[depth, width]` in `out_dtype`, a `tile` a `block_k` x
    `block_n` output tile summed over the rows `block_m` at a time (`grouped_matmul.Tgmm`'s tiles)."""
    rows, depth = lhs.shape
    width = rhs.shape[1]
    block_m, block_n, block_k = tile.block_m, tile.block_n, tile.block_k
    if depth % block_k or width % block_n:
        raise ValueError(f"a [{rows}, {depth}] by [{rows}, {width}] kernel gradient does not take {tile}")
    (lhs, padded), (rhs, _) = _padded_rows(lhs, block_m), _padded_rows(rhs, block_m)
    prologue_operands = _padded_sides(prologue_operands, prologue_axes, padded)
    blocks = {'rows': (padded, None), 'depth': (block_k, 0)}

    def spec(axes):
        return pl.BlockSpec(tuple(blocks[axis][0] for axis in axes),
                            lambda i, j: tuple(i if blocks[axis][1] == 0 else 0 for axis in axes))

    kernel = functools.partial(_tgmm_kernel, steps=padded // block_m, block_m=block_m, rows=rows,
                               prologue=prologue, prologue_axes=prologue_axes, exact_sums=exact_sums)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((depth, width), out_dtype),
        grid=(depth // block_k, width // block_n),
        in_specs=[pl.BlockSpec((padded, block_k), lambda i, j: (0, i)),
                  pl.BlockSpec((padded, block_n), lambda i, j: (0, j)),
                  *(spec(axes) for axes in prologue_axes)],
        out_specs=pl.BlockSpec((block_k, block_n), lambda i, j: (i, j)),
        compiler_params=plgpu.CompilerParams(num_warps=tile.warps, num_stages=tile.stages),
        interpret=interpret,
    )(lhs, rhs, *prologue_operands)


def _tgmm_kernel(lhs_ref, rhs_ref, *refs, steps: int, block_m: int, rows: int, prologue,
                 prologue_axes: tuple, exact_sums: bool):
    *prologue_refs, out_ref = refs
    live = _unknown_true()

    def step(r, acc):
        span = pl.ds(r * block_m, block_m)
        x = lhs_ref[span, :]
        if prologue is not None:
            x = prologue(x, *_cut(prologue_refs, prologue_axes, 'rows', span)).astype(x.dtype)
            if rows % block_m:
                # The padding rows, which the prologue need not keep zero, contribute nothing.
                kept = r * block_m + lax.broadcasted_iota(jnp.int32, x.shape, 0) < rows
                x = jnp.where(kept, x, jnp.zeros_like(x))
        partial = jnp.dot(x.T, rhs_ref[span, :], preferred_element_type=jnp.float32)
        return _summed(acc, partial, live, exact_sums=exact_sums)

    acc = lax.fori_loop(0, steps, step, jnp.zeros(out_ref.shape, jnp.float32))
    out_ref[...] = acc.astype(out_ref.dtype)


def _summed(acc, partial, live, *, exact_sums: bool):
    """The running fp32 sum plus a step's product: with `exact_sums` an IEEE add of the product summed
    apart, behind a selection on `live` (always true, unknown to the compiler), which keeps the add from
    folding into the MMA's accumulator (`ragged_dot._summed`)."""
    return acc + jnp.where(live, partial, 0) if exact_sums else acc + partial


def _unknown_true():
    """A condition always true that the compiler cannot prove: a program id below 2^30 (a CUDA block index
    is known only to be below 2^31)."""
    return pl.program_id(0) < (1 << 30)


def _padded_rows(array: jax.Array, block: int) -> tuple[jax.Array, int]:
    """`array`'s rows padded with zeros to a multiple of `block`, and the padded count."""
    padded = -(array.shape[0] // -block) * block
    if padded == array.shape[0]:
        return array, padded
    return jnp.pad(array, ((0, padded - array.shape[0]), (0, 0))), padded


def _cut(refs, axes_of: tuple, stepped: str, span) -> list:
    """Each side tile of a step: its block cut to `span` along the `stepped` launch axis, whole along the
    others."""
    return [ref[tuple(span if axis == stepped else slice(None) for axis in axes)]
            for ref, axes in zip(refs, axes_of, strict=True)]


def _padded_sides(sides: tuple, axes_of: tuple, padded: int) -> tuple:
    """Each side operand's rows padded (`_padded_side`)."""
    return tuple(_padded_side(side, axes, padded) for side, axes in zip(sides, axes_of, strict=True))


def _padded_side(side: jax.Array, axes: tuple, padded: int) -> jax.Array:
    """A side operand's rows, wherever they are among its dimensions, padded with zeros to `padded`."""
    if 'rows' not in axes:
        return side
    axis = axes.index('rows')
    widths = [(0, 0)] * side.ndim
    widths[axis] = (0, padded - side.shape[axis])
    return jnp.pad(side, widths) if widths[axis][1] else side
