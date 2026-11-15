"""A tile's rows on their way to global memory, every Mosaic family's (`mosaic`): the output staged a
swizzle's row of columns at a time through two chunk buffers (`output_staging`), a value written to its shared
buffer and its rows stored by a ladder of static-extent copies (`staged_rows`, `store_rows`), or scattered
row by row where a table says (`Scattered`)."""

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


@dataclasses.dataclass(frozen=True)
class Scattered:
    """An output whose row `i` lands at row `table[i]` of `ref` (a product's rows written where a
    permutation says, `staged_rows`: a mixture of experts' slots in token order, row 1376), `table` an int32
    ref over the stored rows, a negative entry a row stored nowhere (padding); `interpreted`, copied a row at
    a time, where no lane-level code runs."""

    ref: Any
    table: Any
    interpreted: bool = False


STAGED_BYTES = 128
"""The widest row, in bytes, a consumer stages an output in at a time: a swizzle's row. A consumer's 256
bf16 columns written to shared memory as one 512-byte row conflicted on its banks 195 times as often as in
two 256-byte halves (ncu, Qwen3's down projection forward: 18.3M conflicts against 94K, the kernel 16%
slower); 128-byte chunks, two of them buffered (`output_staging`), stage a 256-column bf16 tile in half its
width, which leaves the ring a stage more (unmeasured for conflicts at 128 bytes)."""

OUTPUT_BUFFERS = 2
"""The chunk buffers a consumer stages an output tile through: one written while the last one's store
reads it."""


def staged_columns(columns: int, dtypes) -> int:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in: the most that divide the
    tile within `STAGED_BYTES` of the widest output (a 192-column tile's bf16 output in chunks of 64)."""
    return math.gcd(columns, STAGED_BYTES // max(jnp.dtype(dtype).itemsize for dtype in dtypes))


def output_staging(columns: int, dtypes) -> tuple[int, int]:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in (`staged_columns`) and
    the chunk buffers it holds (`OUTPUT_BUFFERS`, at most its chunks): the tile's output staged in two
    chunks' shared memory rather than its whole width, CUTLASS's sub-tiled epilogue, which leaves the
    mainloop's ring the rest of the block (its StageCountAutoCarveout)."""
    chunk = staged_columns(columns, dtypes)
    return chunk, min(columns // chunk, OUTPUT_BUFFERS)


def written(smem, value: jax.Array) -> None:
    """`value` written to the shared buffer `smem` in its dtype by an optimized transfer from its layout."""
    smem[...] = value.astype(smem.dtype)


def staged_rows(smem, value: jax.Array, o_gmem, row_start, first, stored, column, most: int,
                least: int = 1, *, write: Callable = written) -> None:
    """`value` written to the shared buffer `smem` in its dtype (`write`: its issuer's, `Issuer.written`),
    made visible to TMA, and its rows `first` to `first + stored` stored to `o_gmem` at row
    `row_start + first` and `column` (`store_rows`); the copies stay in flight (the caller waits before
    `smem` is written again). A `Scattered` output's rows each land where its table says, copied by the
    warpgroup's threads (`gathered.scattered_rows`), done when it returns, its value written unoptimized
    (no TMA copy reads the buffer, whose plain layout has no optimized transfer from a warpgroup MMA's)."""
    if isinstance(o_gmem, Scattered):
        plgpu.store(smem, value.astype(smem.dtype), optimized=False)
        plgpu.commit_smem()
        from dew.nn.kernels.templates import gathered  # it builds on `Ring`

        gathered.scattered_rows(smem, o_gmem, row_start, first, stored, column)
        return
    write(smem, value)
    plgpu.commit_smem()
    store_rows(smem, o_gmem, row_start, first, stored, column, most, least)


def store_rows(o_smem, o_gmem, row_start, first, stored, column, most: int, least: int = 1,
               copy: Callable = plgpu.copy_smem_to_gmem, widest: int | None = None) -> None:
    """Rows `first` to `first + stored` of `o_smem` to `o_gmem` at row `row_start + first` and `column`, in a
    power-of-two ladder of static-size `copy`s (a TMA store's descriptor has static extents; `copied`
    moves the rows through registers where a GPU has no TMA), the largest `most` rows', the smallest
    `least` (`stored` a multiple of it: a rung below it is never taken, and none is emitted); with
    `widest`, in copies of at most `widest` rows, whole ones first (a copy through registers holds its
    rows: `resources.copied_rows`)."""
    start, size = first, most
    if widest is not None and widest < most:
        for whole in range(most // widest):
            @pl.when(stored // widest > whole)
            def _(whole=whole):
                copy(o_smem.at[pl.ds(first + whole * widest, widest)],
                     o_gmem.at[pl.ds(row_start + first + whole * widest, widest),
                               pl.ds(column, o_smem.shape[1])])

        start, size = first + stored // widest * widest, widest - 1
    while size >= least:
        rung = 1 << int(math.log2(size))
        size //= 2

        @pl.when(stored & rung != 0)
        def _(start=start, rung=rung):
            copy(o_smem.at[pl.ds(start, rung)],
                 o_gmem.at[pl.ds(row_start + start, rung), pl.ds(column, o_smem.shape[1])])

        start = start + (stored & rung)
