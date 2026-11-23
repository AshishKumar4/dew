"""A tile's rows on their way to global memory, every Mosaic family's (`mosaic`): the output staged a
swizzle's row of columns at a time through two chunk buffers (`output_staging`), a value written to its shared
buffer and its rows stored by a ladder of static-extent copies (`staged_rows`, `store_rows`), or scattered
row by row where a table says (`Scattered`)."""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Any

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import swizzling


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

TMA_STORE_BYTES = 16
"""The bytes a TMA store's row is a whole number of: a tensor map's box takes an inner dimension of a
multiple of 16 bytes (CUDA's `cuTensorMapEncodeTiled`: `boxDim[0]` times the element's size), so a staged
chunk stores rows of whole 16-byte units in each output's dtype (8 bf16 columns, 4 fp32)."""


def staged_columns(columns: int, dtypes) -> int:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in: the most that divide the
    tile within `STAGED_BYTES` of the widest output (a 192-column tile's bf16 output in chunks of 64)."""
    return math.gcd(columns, STAGED_BYTES // max(jnp.dtype(dtype).itemsize for dtype in dtypes))


def output_staging(columns: int, dtypes, staging: tuple[int, int] | None = None, *,
                   piece: int | None = None) -> tuple[int, int]:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in (`staged_columns`) and
    the chunk buffers it holds (`OUTPUT_BUFFERS`, at most its chunks), or `staging`'s (chunk columns,
    buffers) where a launch states them: the tile's output staged in its chunks' shared memory rather than
    its whole width, CUTLASS's sub-tiled epilogue, which leaves the mainloop's ring the rest of the block
    (its StageCountAutoCarveout). Fewer buffers or narrower chunks leave the ring a stage more where the
    block falls a stage short (an sm120 GEMM's 128 x 256 tile, row 5634, M-FS-62). Refused where it holds
    no staging (`staging_refusal`)."""
    if (refusal := staging_refusal(columns, dtypes, staging, piece=piece)) is not None:
        raise ValueError(refusal)
    chunk, buffers = (staged_columns(columns, dtypes), OUTPUT_BUFFERS) if staging is None else staging
    return chunk, min(columns // chunk, buffers)


def staging_refusal(columns: int, dtypes, staging: tuple[int, int] | None = None, *,
                    piece: int | None = None) -> str | None:
    """Why a `columns`-wide output tile of `dtypes` stages in no chunks of `staging`'s (chunk columns,
    buffers), its derived ones where none is stated (`output_staging`), None where it does: what a launch's
    candidate asks of a pin before it takes it (a GEMM's tile's). A chunk divides the tile and the `piece`
    its issuer emits at once (`Issuer.chunk_columns`, the tile's width where none is given): a tcgen05 tile
    of 128 columns emits pieces of 64, which a 128-column chunk does not hold. A stated chunk stores whole
    TMA rows in each of `dtypes` (`TMA_STORE_BYTES`), by each dtype's bits (a packed fp4's half byte): a bf16
    output's chunk of 4 columns, 8 bytes a row, is refused here rather than by the tensor map it lowers
    to."""
    chunk, buffers = (staged_columns(columns, dtypes), OUTPUT_BUFFERS) if staging is None else staging
    if chunk < 1 or columns % chunk or (piece or columns) % chunk or buffers < 1:
        return (f"a {columns}-column output tile emitted {piece or columns} columns at once stages in no "
                f"{buffers} buffers of {chunk} columns")
    row_bits = 8 * TMA_STORE_BYTES
    if staging is not None and (partial := [jnp.dtype(dtype).name for dtype in dtypes
                                            if chunk * jax.dtypes.itemsize_bits(dtype) % row_bits]):
        return (f"a stated chunk of {chunk} columns stores no whole {TMA_STORE_BYTES}-byte TMA rows in "
                f"{', '.join(partial)}")
    return None


def written(smem, value: jax.Array) -> None:
    """`value` written to the shared buffer `smem` in its dtype by an optimized transfer from its layout."""
    smem[...] = value.astype(smem.dtype)


def buffer_waits(chunks: int, buffers: int) -> tuple[int, ...]:
    """For each of a tile's `chunks` chunks staged through `buffers` buffers in turn (chunk c in buffer
    c % buffers, `output_staging`), the TMA store groups that may still be reading shared memory as the
    chunk's buffer is written again (`plgpu.wait_smem_to_gmem`'s count): the fewest groups any run issues
    after that buffer's last store, each chunk at least one group (`staged_rows`: a group a copy its rows
    take; a share storing no rows stages no chunk, `staged_chunks`). Before a buffer is rewritten its last
    group is older than those, so it has been read; another output's groups between
    them only make it older. A chunk past the first `buffers` (c >= buffers) wrote its buffer `buffers`
    chunks before, `buffers - 1` groups since; one of the first rewrites the buffer the tile before last
    stored, at its chunk L (the last of that buffer's, `L % buffers == c`), `chunks - 1 - L` of that tile's
    groups and this tile's `c` since. Every tile of a launch stages the same chunks (its value's static
    width: a ragged last tile's rows and columns clipped by TMA, not fewer chunks), and a tile that stores
    nothing (an inactive one, or a share of no rows) stages no chunk and issues no group, so the tile before
    is the last that stored. Waiting for none
    (each chunk's count 0) drained both buffers before every write: a pair of 32-column chunks of an
    emulated B200's gathered kernel gradient waited on the other buffer's store too (M-EM-71, em71)."""
    waits = []
    for chunk in range(chunks):
        if chunk >= buffers:
            waits.append(buffers - 1)
            continue
        last = chunk + buffers * ((chunks - 1 - chunk) // buffers)
        waits.append(chunks - 1 - last + chunk)
    return tuple(waits)


def column_spans(columns: int, width: int) -> list[tuple[int, int]]:
    """`columns` `width` at a time, each span its first column and its width, the last clipped to what is
    left, one span of all of them where they are no wider: the partition `column_chunks` slices a value by,
    its shape alone (a plan's, before any value), and a dQ reducer's chunks (`_PlacedDq.pieces`, row
    8813)."""
    if width >= columns:
        return [(0, columns)]
    return [(first, min(width, columns - first)) for first in range(0, columns, width)]


def column_chunks(value: jax.Array, width: int) -> list[tuple[int, jax.Array]]:
    """`value` `width` of its columns at a time, each chunk with its first column, the whole where it is no
    wider (`column_spans`): a stage narrower than the tile it moves (CUTLASS's sub-tiled epilogue,
    `staged_columns`)."""
    spans = column_spans(value.shape[-1], width)
    if len(spans) == 1:
        return [(0, value)]
    return [(first, value[..., first:first + span]) for first, span in spans]


def staged_chunks(smem, value: jax.Array, ref, place: tuple, *, write: Callable = written,
                  waits: tuple[int, ...] | None = None) -> None:
    """A consumer's share `value` of an output tile staged through its chunk buffers (`smem`, `[buffers,
    rows, columns]`) a buffer's columns at a time (`column_chunks`) and stored to `ref` at `place`
    (`(row_start, first, stored, column, *ladder)`, `staged_rows`), each chunk's buffer written once the
    store groups `waits` leave in flight no longer read it: the caller's own schedule of them, where its
    chunks' groups run otherwise than one output's tiles in turn, else `buffer_waits`'. A share storing no
    rows (a ragged last tile's second consumer's) stages nothing: no buffer written, no group issued, so
    every chunk staged commits at least one (`buffer_waits`)."""
    pl.when(place[2] > 0)(functools.partial(staged, smem, value, ref, place, write=write, waits=waits))


def staged(smem, value: jax.Array, ref, place: tuple, *, write: Callable = written,
           waits: tuple[int, ...] | None = None) -> None:
    """`staged_chunks`' chunks staged whatever rows `place` stores: each its buffer's wait, its write and
    its stores."""
    buffers, width = smem.shape[0], smem.shape[-1]
    row_start, first, stored, column, *ladder = place
    chunks = column_chunks(value, width)
    waits = buffer_waits(len(chunks), buffers) if waits is None else waits
    for chunk, (start, piece) in enumerate(chunks):
        plgpu.wait_smem_to_gmem(waits[chunk], wait_read_only=True)
        staged_rows(smem.at[chunk % buffers], piece, ref, row_start, first, stored, column + start, *ladder,
                    write=write)


def staged_rows(smem, value: jax.Array, o_gmem, row_start, first, stored, column, most: int,
                least: int = 1, *, write: Callable = written) -> None:
    """`value` written to the shared buffer `smem` in its dtype (`write`: its issuer's, `Issuer.written`),
    made visible to TMA, and its rows `first` to `first + stored` stored to `o_gmem` at row
    `row_start + first` and `column` (`store_rows`), each copy its own TMA store group, as CUTLASS's
    epilogue commits each of its stores (`tma_store_arrive`): a chunk of rows commits at least one
    (`buffer_waits` counts a chunk's stores so). One group over a chunk's
    ladder (its copies uncommitted, one commit after them) left Hopper's warpgroup-MMA consumer a register
    short under CUDA 13's ptxas at 128x256 tiles, the persistent walk's bound spilled (row 8669). The
    copies stay in flight (the caller waits before `smem` is written again). A
    `Scattered` output's rows each land where its table says, copied by the warpgroup's threads
    (`gathered.scattered_rows`), done when it returns, its value written unoptimized (no TMA copy reads the
    buffer, whose plain layout has no optimized transfer from a warpgroup MMA's)."""
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


def swizzled_rows_out(smem, o_gmem, row_start, first, stored, column, *, swizzle: int) -> None:
    """Rows `first` to `first + stored` of `smem` (`[rows, width]`, staged at `swizzle` bytes:
    `swizzling.staged`) to `o_gmem` at row `row_start + r` and `column`, by the warpgroup's threads, each a
    16-byte piece of a row at a time read where the staging's swizzle put it (`swizzling.staged_address`)
    and its global store predicated on its row being stored: the copy reads whole swizzled tiles, where a
    ladder of row copies (`store_rows`) would slice a tile at any row, which a swizzled staging refuses
    below its 8-row tiles (CUTLASS's sm80 epilogue's predicated stores). Done when it returns: the threads
    meet before and after it."""
    scalar = plgpu.Layout.WG_SPLAT(())
    dest = o_gmem.at[:, pl.ds(column, smem.shape[1])]

    def pieces(ctx, smem_ref, dest_ref, start_value, first_value, stored_value):
        del ctx
        from dew.nn.kernels.templates import gathered  # its lane-level helpers build on `Ring`

        gathered.swizzled_pieces(smem_ref, dest_ref, gathered.splat_value(start_value),
                                 gathered.splat_value(first_value), gathered.splat_value(stored_value),
                                 swizzle=swizzle)

    transforms = swizzling.swizzle_transforms(smem.dtype, swizzle)
    plgpu.inline_mgpu(arg_types=(plgpu.RefType(transforms), plgpu.RefType(), scalar, scalar, scalar))(pieces)(
        smem, dest, jnp.asarray(row_start, jnp.int32), jnp.asarray(first, jnp.int32),
        jnp.asarray(stored, jnp.int32))
