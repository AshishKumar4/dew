"""A step-scaled walk's scales (`mosaic_gpu.gmm`'s and `tgmm`'s `step_scales`: DeepSeek-V3's blockwise FP8,
row 4518): their streams, their layout as a walk reads them, whether a tile takes them, and a step's exact
partial scaled by them (`scaled_step`)."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels.templates import swizzling
from dew.nn.kernels.templates.gemm import rows_extended, rows_padded
from dew.nn.kernels.templates.gemm.row_tiles import TMA_ROWS
from dew.nn.kernels.templates.gemm.tiling import Tiles
from dew.nn.kernels.templates.product import SCALE_DEPTH


def blocks(block_m: int, block_n: int) -> dict:
    """A step-scaled call's scale streams (`gmm`'s and `tgmm`'s `step_scales`): a step's row of its
    rows' scales at the tile's rows and of its columns' at the tile's columns (a row of the matrix's coarser
    ones holding the tile's, `column_scales`), unswizzled (`SWIZZLES`)."""
    return {'row_scales': ((1, block_m), jnp.float32), 'column_scales': ((1, block_n), jnp.float32)}


SWIZZLES = {'row_scales': swizzling.UNSWIZZLED, 'column_scales': swizzling.UNSWIZZLED}
"""The scale streams' swizzles (`blocks`): rows of fp32 a step, read a row at a time."""


def walked(step_scales: tuple | None, n: int, block_n: int, extent: int) -> tuple:
    """A blockwise-scaled call's `step_scales` as its walk reads them: its rows' scales padded and
    extended as the rows (`mosaic_gpu._walked_rows`), its columns' a tile's (`column_scales`), and the columns a scale
    of a tile (`per`, 1 without them)."""
    if step_scales is None:
        return None, 1
    columns, per = column_scales(step_scales[1], n, block_n)
    return (rows_extended(rows_padded(step_scales[0], 1, TMA_ROWS), 1, extent), columns), per


def takes(step_scales: tuple, tiles: Tiles, depth: int, width: int, *, exact_sums: bool,
                       plain: bool) -> bool:
    """Whether `gmm` takes `step_scales` at `tiles` (`scaled_step`): an exactly summed product of no
    prologue, pair, packed matrix, transposed rows or split (an epilogue reads its sums), a step 128 deep, a
    row scale each step, and a column scale each column or each multiple of the tile's width."""
    rows, columns = step_scales
    per = width // columns.shape[-1] if columns.shape[-1] and width % columns.shape[-1] == 0 else 0
    return (exact_sums and plain and tiles.splits == 1 and tiles.block_k == SCALE_DEPTH
            and rows.shape[0] * SCALE_DEPTH == depth
            and (per == 1 or (per > 0 and per % tiles.block_n == 0)))


def apart(refs: list, scaled: bool) -> tuple:
    """A step-scaled kernel's two scale references, which lead its side operands' (`tgmm`'s
    `step_scales`), and the rest; None and all of them where it has no step scales."""
    return (tuple(refs[:2]), refs[2:]) if scaled else (None, refs)


def column_scales(column_scales: jax.Array, width: int, block_n: int) -> tuple[jax.Array, int]:
    """A step-scaled `gmm`'s column scales (`[groups, depth / 128, width / per]`) as its stream reads
    them (`blocks`), and the columns `per` a scale covers: as they are where each covers a column;
    else each step's row of them padded to whole `block_n`s, a block of which holds a tile's scale
    (`column_block`)."""
    per = width // column_scales.shape[-1]
    if per == 1:
        return column_scales, per
    return jnp.pad(column_scales, ((0, 0), (0, 0), (0, -column_scales.shape[-1] % block_n))), per


def column_block(column_tile, block_n: int, per: int) -> tuple:
    """The block of a step's column scales `column_tile`'s stream reads (`column_scales`'), and where
    in it the tile's own scale lies: its columns' block, at its first, where a scale covers a column; else
    the block holding the tile's one scale, `per` columns (whole tiles of them) a scale."""
    if per == 1:
        return column_tile, 0
    scale = column_tile * block_n // per
    return scale // block_n, scale % block_n


def scaling(scaled: bool, per: int, block_n: int, column_tile, mine_rows) -> Callable | None:
    """A step-scaled `gmm` tile's `scaled_step` for a consumer's share of the rows (`mine_rows`), its
    matrix's column scales a scale every `per` columns (`column_block`), or a `tgmm` tile's for its depth
    rows, a scale a column; none where the call has no step scales."""
    if not scaled:
        return None
    return functools.partial(scaled_step, per=per, offset=column_block(column_tile, block_n, per)[1],
                             mine_rows=mine_rows)


def scaled_step(index, slots, *, per: int, offset, mine_rows) -> Callable:
    """A step-scaled walk's scaling of a step's exact partial (`gmm`'s and `tgmm`'s `step_scales`,
    `mosaic.Issuer.summed_apart`'s `scale(start, columns, partial)`): times this consumer's rows' scales
    (`row_scales`) and the tile's columns' (`column_scales`: one a column, or the tile's block's at `offset`,
    `per` columns a scale), in `precision.scaled_partial`'s order, as XLA's arm scales (two scales near 2^64
    overflow multiplied first though the scaled partial is finite: rows 13257, 13274). `index` the step's."""
    from dew.nn.precision import scaled_partial

    del index
    rows = slots['row_scales'].at[0, mine_rows][...]
    column_scales = slots['column_scales']
    if per == 1:
        return lambda start, columns, partial: scaled_partial(
            partial, rows[:, None], column_scales.at[0, pl.ds(start, columns)][...][None, :])
    # A weight block's one scale beside the rows' vector, each broadcast over the partial's entries.
    block = column_scales[0, offset]
    return lambda start, columns, partial: scaled_partial(partial, rows[:, None], block)
