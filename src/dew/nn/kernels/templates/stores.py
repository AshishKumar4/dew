"""A tile's rows on their way to global memory, every Mosaic family's (`mosaic`): the output staged a
swizzle's row of columns at a time through two chunk buffers (`output_staging`), a value written to its shared
buffer and its rows stored by a ladder of static-extent copies (`staged_rows`, `store_rows`), or scattered
row by row where a table says (`Scattered`)."""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import copies, swizzling


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



class Output(NamedTuple):
    """An output tile a family stages (`Family.output_staging`, `persist`'s stores): its `shape` (its rows and
    tile columns) and stored `dtype`, `divisor` tile columns a stored column (a packed form's: NVFP4's codes
    two to a byte, its scales one a 16-column block), each staged chunk written into its buffer by
    `write(buffer, piece)` from the chunk's tile columns (its issuer's `written` where None: a packed form's
    packing). `transposed`, stored `[columns // divisor, rows]`, its rows contiguous (a block-scaled
    product's scales as its consumer reads them: a 128-row share's 128 bytes of E8M0 or 512 of fp32 a
    stored row, whatever the tile's width): its chunk's buffer `[chunk // divisor, rows]`, stored whole at
    its share's first row, the tensor's bound clipping the rows past its last (`staged_rows`), so a call
    whose shares store whole rows (a dense one's) alone takes one. `narrowed`, its value already a
    `divisor`-th of the tile's columns as its epilogue computes it (a wide gated product's hidden activation,
    one of each gate and up column pair, `templates.gemm.wide`), stored as computed, a chunk's `chunk //
    divisor` columns, placed in its own columns (its caller's `place`)."""

    shape: tuple[int, int]
    dtype: jax.typing.DTypeLike
    divisor: int = 1
    write: Callable | None = None
    transposed: bool = False
    narrowed: bool = False

    @classmethod
    def of(cls, output: "Output | tuple[tuple[int, int], jax.typing.DTypeLike]") -> "Output":
        """`output` as an `Output`: a plain `(shape, dtype)` pair stored as it is computed."""
        return output if isinstance(output, cls) else cls(output[0], output[1])


def staged_columns(columns: int, dtypes, *, swizzle: int = STAGED_BYTES,
                   packed: tuple[Output, ...] = (), piece: int | None = None) -> int:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in: the most that divide the
    tile within `swizzle` bytes of the widest output, a swizzled row's (`STAGED_BYTES` by default; a
    192-column tile's bf16 output in chunks of 64); where outputs are stored packed (`packed`, each
    `Output`'s divisor and form), the least multiple of it dividing the tile whose stored rows each stage
    (`_packed_refusal`: a 128-column tile's NVFP4 scales, 8 bytes a row, stage at 256) within the `piece`
    its issuer emits at once (`Issuer.chunk_columns`: a tcgen05 tile's 64), the tile's width where none does
    (which `staging_refusal` refuses)."""
    if not packed:
        return math.gcd(columns, swizzle // max(jnp.dtype(dtype).itemsize for dtype in dtypes))
    within = columns if piece is None else math.gcd(columns, piece)
    # A narrowed output stores a swizzle's row of its own columns a chunk where the tile holds one, its
    # tile columns that many times its divisor; staged beside any other output, a chunk is bounded by each
    # one's own row, the least of them.
    narrowing = (min(output.divisor for output in packed)
                 if len(dtypes) == len(packed) and all(output.narrowed for output in packed) else 1)
    chunk = math.gcd(within, narrowing * swizzle // max(jnp.dtype(dtype).itemsize for dtype in dtypes))
    return next((wider for wider in range(chunk, within + 1, chunk)
                 if not within % wider and _packed_refusal(wider, packed) is None), columns)


def _packed_refusal(chunk: int, packed: tuple[Output, ...]) -> str | None:
    """Why a chunk of `chunk` tile columns stores no whole rows of each of `packed`'s outputs, None where
    it does: each one's divisor dividing the chunk; its buffer (`[rows, chunk // divisor]`, a transposed
    one's `[chunk // divisor, rows]`, `Output.transposed`) at most `copies.BOX_ELEMENTS` along each axis, its
    stored row (the inner axis) whole `copies.ROW_BYTES`-byte units by its dtype's bits (a packed fp4's half
    byte), and the buffer whole `copies.SHARED_ALIGNMENT`-byte units, so each buffer of their stack starts
    aligned for its tensor copy."""
    for output in packed:
        name, bits = jnp.dtype(output.dtype).name, jax.dtypes.itemsize_bits(output.dtype)
        rows = output.shape[0]
        if chunk % output.divisor:
            return f"a chunk of {chunk} columns holds no whole {output.divisor}-column blocks"
        stored = chunk // output.divisor
        outer, inner = (stored, rows) if output.transposed else (rows, stored)
        if not copies.whole_rows(inner, output.dtype):
            return (f"a chunk of {chunk} columns stored {output.divisor} a column in {name} is no whole "
                    f"{copies.ROW_BYTES}-byte TMA rows")
        if max(outer, inner) > copies.BOX_ELEMENTS:
            return (f"a chunk of {chunk} columns stages a {outer} x {inner} {name} buffer, past a tensor "
                    f"map's {copies.BOX_ELEMENTS} a dimension")
        if outer * inner * bits % (8 * copies.SHARED_ALIGNMENT):
            return (f"a chunk of {chunk} columns stages a {outer} x {inner} {name} buffer of no whole "
                    f"{copies.SHARED_ALIGNMENT} bytes, so its stack's next starts unaligned")
    return None


def output_staging(columns: int, dtypes, staging: tuple[int, int] | None = None, *,
                   piece: int | None = None, packed: tuple[Output, ...] = ()) -> tuple[int, int]:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in (`staged_columns`) and
    the chunk buffers it holds (`OUTPUT_BUFFERS`, at most its chunks), or `staging`'s (chunk columns,
    buffers) where a launch states them: the tile's output staged in its chunks' shared memory rather than
    its whole width, CUTLASS's sub-tiled epilogue, which leaves the mainloop's ring the rest of the block
    (its StageCountAutoCarveout). Fewer buffers or narrower chunks leave the ring a stage more where the
    block falls a stage short (an sm120 GEMM's 128 x 256 tile, row 5634, M-FS-62). Refused where it holds
    no staging (`staging_refusal`). The chunk is in tile columns; an output stored packed (`packed`,
    `Output.divisor`) holds its chunk's stored columns."""
    if (refusal := staging_refusal(columns, dtypes, staging, piece=piece, packed=packed)) is not None:
        raise ValueError(refusal)
    chunk, buffers = ((staged_columns(columns, dtypes, packed=packed, piece=piece), OUTPUT_BUFFERS)
                      if staging is None else staging)
    return chunk, min(columns // chunk, buffers)


def staging_refusal(columns: int, dtypes, staging: tuple[int, int] | None = None, *,
                    piece: int | None = None, packed: tuple[Output, ...] = ()) -> str | None:
    """Why a `columns`-wide output tile of `dtypes` stages in no chunks of `staging`'s (chunk columns,
    buffers), its derived ones where none is stated (`output_staging`), None where it does: what a launch's
    candidate asks of a pin before it takes it (a GEMM's tile's). A chunk divides the tile and the `piece`
    its issuer emits at once (`Issuer.chunk_columns`, the tile's width where none is given): a tcgen05 tile
    of 128 columns emits pieces of 64, which a 128-column chunk does not hold. A stated chunk stores whole
    TMA rows in each of `dtypes` (`copies.ROW_BYTES`), by each dtype's bits (a packed fp4's half byte): a bf16
    output's chunk of 4 columns, 8 bytes a row, is refused here rather than by the tensor map it lowers
    to. Outputs stored packed (`packed`, whose dtypes `dtypes` are) are refused where a chunk, stated or
    derived, stores no whole rows of them (`_packed_refusal`): a 128-column tile's NVFP4 scales."""
    chunk, buffers = ((staged_columns(columns, dtypes, packed=packed, piece=piece), OUTPUT_BUFFERS)
                      if staging is None else staging)
    if chunk < 1 or columns % chunk or (piece or columns) % chunk or buffers < 1:
        return (f"a {columns}-column output tile emitted {piece or columns} columns at once stages in no "
                f"{buffers} buffers of {chunk} columns")
    partial = [jnp.dtype(dtype).name for dtype in dtypes if not copies.whole_rows(chunk, dtype)]
    if staging is not None and not packed and partial:
        return (f"a stated chunk of {chunk} columns stores no whole {copies.ROW_BYTES}-byte TMA rows in "
                f"{', '.join(partial)}")
    return _packed_refusal(chunk, packed)


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
                  waits: tuple[int, ...] | None = None, divisor: int = 1, transposed: bool = False,
                  narrowed: bool = False) -> None:
    """A consumer's share `value` of an output tile staged through its chunk buffers (`smem`, `[buffers,
    rows, columns]`) a buffer's columns at a time (`column_chunks`) and stored to `ref` at `place`
    (`(row_start, first, stored, column, *ladder)`, `staged_rows`), each chunk's buffer written once the
    store groups `waits` leave in flight no longer read it: the caller's own schedule of them, where its
    chunks' groups run otherwise than one output's tiles in turn, else `buffer_waits`'. A share storing no
    rows (a ragged last tile's second consumer's) stages nothing: no buffer written, no group issued, so
    every chunk staged commits at least one (`buffer_waits`). An output stored packed (`divisor`,
    `Output.divisor`) stages a buffer's columns times it of the tile a chunk, stored at its tile column over
    it; `transposed` (`Output.transposed`), its buffer `[columns, rows]`, stored whole (`transposed_rows`);
    `narrowed` (`Output.narrowed`), its value and `place` already its stored columns, a buffer's at a time."""
    pl.when(place[2] > 0)(functools.partial(staged, smem, value, ref, place, write=write, waits=waits,
                                            divisor=divisor, transposed=transposed, narrowed=narrowed))


def staged(smem, value: jax.Array, ref, place: tuple, *, write: Callable = written,
           waits: tuple[int, ...] | None = None, divisor: int = 1, transposed: bool = False,
           narrowed: bool = False) -> None:
    """`staged_chunks`' chunks staged whatever rows `place` stores: each its buffer's wait, its write and
    its stores."""
    buffers, width = smem.shape[0], smem.shape[-2 if transposed else -1]
    row_start, first, stored, column, *ladder = place
    if isinstance(ref, Scattered):
        if divisor > 1 or transposed:
            raise ValueError("a scattered output's rows land a row at a time as computed, none packed")
    elif transposed:
        _whole_shares(smem.shape[-1], width, ref.shape[-1], first, stored)
    elif divisor > 1 and not narrowed:
        _aligned_ladder(width * jax.dtypes.itemsize_bits(smem.dtype) // 8, first, ladder)
        if (ladder[1] if len(ladder) > 1 else 1) == 1 and ref.shape[-1] % width:
            # A rung of one row is one contiguous run, which Mosaic GPU copies by a linear bulk copy clamped
            # at the whole tensor's end, not its row's (`launch_context.async_copy`): a last chunk past the
            # row's end would write into the next row.
            raise ValueError(f"a packed chunk of {width} columns copied a row at a time over "
                             f"{ref.shape[-1]} columns runs its last chunk into the next row")
    # A narrowed value and its place are its stored columns already (its caller's: a tile's column over the
    # divisor is no multiple of the stores' tiling the lowering can prove); a packed one's its tile's.
    chunks = column_chunks(value, width if narrowed else width * divisor)
    waits = buffer_waits(len(chunks), buffers) if waits is None else waits
    for chunk, (start, piece) in enumerate(chunks):
        plgpu.wait_smem_to_gmem(waits[chunk], wait_read_only=True)
        at = column + start if narrowed else (column + start) // divisor
        if transposed:
            transposed_rows(smem.at[chunk % buffers], piece, ref, row_start, at, write=write)
            continue
        staged_rows(smem.at[chunk % buffers], piece, ref, row_start, first, stored, at, *ladder, write=write)


def _whole_shares(rows: int, columns: int, extent: int, first, stored) -> None:
    """Refuse a transposed output's share of `rows` rows, `columns` packed columns a chunk, into an output
    of `extent` rows, stored from `first` for `stored`, where its whole-share copy writes rows not its own:
    its first row and rows not the static 0 and the share's (a grouped tile's, one ending mid-tensor; a dense
    call states its last share whole, the tensor's bound clipping); or one column a chunk whose last share
    passes the rows' end, one contiguous run, which Mosaic GPU copies by a linear bulk copy clamped at the
    whole tensor's end, not its row's (`launch_context.async_copy`), into the next column."""
    if not (_static(first, 0) and _static(stored, rows)):
        raise ValueError(f"a transposed output stores its share's {rows} rows whole, from its first: a "
                         f"share of rows {first} to {first} + {stored} (a grouped tile's, or one ending "
                         "mid-tensor) takes none; a dense call's last share states them all, the tensor's "
                         "bound clipping")
    if columns == 1 and extent % rows:
        raise ValueError(f"a transposed output's one-column chunk of {rows}-row shares over {extent} rows "
                         "runs its last share into its next column")


def _static(value, expected: int) -> bool:
    """Whether `value` is the Python int `expected`, known as it traces (a place's first row, its rows)."""
    return isinstance(value, int) and value == expected


def _aligned_ladder(row_bytes: int, first, ladder: list) -> None:
    """Refuse a packed chunk's row ladder (`store_rows`: copies of a power of two of rows from `first`, the
    smallest `least`, `ladder`'s second) whose copies may start a shared row at no `copies.SHARED_ALIGNMENT`
    bytes: a row of `row_bytes` that is no whole number of them starts aligned only at a static first row 0
    and rungs of whole aligned rows (a dense call's 8-row least of a 16-byte NVFP4 scale row)."""
    least = ladder[1] if len(ladder) > 1 else 1
    aligned = least * row_bytes % copies.SHARED_ALIGNMENT == 0
    if row_bytes % copies.SHARED_ALIGNMENT and not (_static(first, 0) and aligned):
        raise ValueError(f"a packed chunk's {row_bytes}-byte rows copied from row {first} in rungs of "
                         f"{least} start a tensor copy at no {copies.SHARED_ALIGNMENT}-byte shared address")


def transposed_rows(smem, value: jax.Array, o_gmem, row_start, column, *, write: Callable = written) -> None:
    """`value`, a chunk of a share's tile columns, written to the shared buffer `smem` (`[columns, rows]`,
    the share's rows contiguous, `write`'s packing: `Output.transposed`) and stored whole to `o_gmem`
    (`[columns, rows]`) at `column` and the share's first row `row_start`, one TMA copy its own store group:
    its rows its box's inner extent, which no partial ladder takes in whole 16-byte units, so a share stores
    all its rows, the tensor's bound clipping those past its last. Its caller (`staged`) refuses a share
    whose first row and rows are not the static 0 and the share's (a grouped tile's, a share ending
    mid-tensor), so a dense call states each share whole, its last's rows past the tensor clipped."""
    write(smem, value)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(smem, o_gmem.at[pl.ds(column, smem.shape[0]), pl.ds(row_start, smem.shape[1])])


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
