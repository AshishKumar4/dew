"""How a consumer stages an output tile through shared memory in chunks
(`dew.nn.kernels.templates.stores.output_staging`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS, on_the_host
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import bundles, copies, stores, swizzling

NVFP4 = (stores.Output((64, 256), jnp.uint8, 2), stores.Output((64, 256), jnp.float8_e4m3fn, 16))
"""A 64-row share of a 256-column NVFP4 tile's codes (two to a byte) and scales (one a 16-column block)."""


def test_a_stated_chunk_stores_whole_tma_rows_in_each_output_dtype():
    """A launch's stated staging (`stores.output_staging`'s `staging`) holds chunks whose rows are whole
    16-byte TMA stores in each of its outputs' dtypes, a narrower one refused before it lowers: 4 bf16 columns
    store 8 bytes a row, as 2 fp32 columns do, where 8 bf16 and 4 fp32 store 16; a call of fp32 and bf16
    outputs takes a chunk whole in both; a packed fp4 output's 16 columns store 8 bytes, its 32 store 16
    (theme gemm-output-staging-4's review: a bf16 output stated in 4-column chunks reached its tensor map;
    this head's review: an fp4 output's chunk judged by its dtype's byte, not its half byte)."""
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.bfloat16,), (4, 1))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float32, jnp.bfloat16), (4, 2))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float32,), (2, 1))
    with pytest.raises(ValueError):
        stores.output_staging(128, (jnp.float4_e2m1fn,), (16, 1))
    assert stores.output_staging(128, (jnp.bfloat16,), (8, 1)) == (8, 1)
    assert stores.output_staging(128, (jnp.float32, jnp.bfloat16), (8, 2)) == (8, 2)
    assert stores.output_staging(128, (jnp.float4_e2m1fn,), (32, 1)) == (32, 1)


@pytest.mark.parametrize("columns, dtypes, staging, piece, holds", [
    (128, (jnp.bfloat16,), (4, 1), None, False), (128, (jnp.bfloat16,), (8, 1), None, True),
    (128, (jnp.float4_e2m1fn,), (16, 1), None, False), (128, (jnp.float4_e2m1fn,), (32, 1), None, True),
    (128, (jnp.float32,), (128, 1), 64, False), (128, (jnp.float32,), (64, 0), None, False),
    (96, (jnp.float32,), None, None, True), (128, (jnp.float32,), (0, 1), None, False),
    (128, (jnp.float32,), (-8, 1), None, False),
], ids=["bf16-4", "bf16-8", "fp4-16", "fp4-32", "past-a-piece", "no-buffer", "derived", "no-column",
        "negative"])
def test_a_candidate_asks_whether_a_staging_holds_with_no_exception(columns, dtypes, staging, piece, holds):
    """A launch's candidate asks whether a staging holds before it pins it (`stores.staging_refusal`, None
    where it does, its reason where it does not, raising nothing): chunks of whole 16-byte TMA rows in each
    dtype, by its bits; a chunk within the piece its issuer emits; a buffer at least (the GEMM
    output-staging head's review: GEMM's candidates held a copy of the rule to ask it)."""
    assert (stores.staging_refusal(columns, dtypes, staging, piece=piece) is None) == holds


def test_a_packed_outputs_chunk_stores_whole_rows_of_each_or_is_refused():
    """Outputs stored packed (`stores.Output.divisor`) stage chunks whose stored rows are each whole 16-byte
    TMA rows of at most 256 elements: a 128-column tile's NVFP4 scales, 8 bytes a row, refused, a 256-column
    tile's derived chunk 256 (its codes 128 bytes, its scales 16); a transposed output (`Output.transposed`,
    `[columns, rows]`) judged by its share's rows, 64 fp32 or 128 E8M0 whole at any width, 2 fp32 not."""
    codes, scales = NVFP4
    assert stores.staging_refusal(128, (jnp.float8_e4m3fn,), packed=(scales._replace(shape=(64, 128)),))
    assert stores.output_staging(256, (jnp.uint8,), packed=(codes,)) == (128, 2)
    assert stores.output_staging(256, (jnp.float8_e4m3fn,), packed=(scales,)) == (256, 1)
    with pytest.raises(ValueError):
        stores.output_staging(256, (jnp.float8_e4m3fn,), (128, 2), packed=(scales,))
    with pytest.raises(ValueError):
        stores.output_staging(512, (jnp.uint8,), (512, 1), packed=(codes._replace(divisor=1),))
    fp32 = stores.Output((64, 128), jnp.float32, 128, transposed=True)
    assert stores.output_staging(128, (jnp.float32,), packed=(fp32,)) == (128, 1)
    e8m0 = stores.Output((128, 128), jnp.float8_e8m0fnu, 32, transposed=True)
    assert stores.output_staging(128, (jnp.float8_e8m0fnu,), packed=(e8m0,)) == (128, 1)
    assert stores.staging_refusal(128, (jnp.float32,), packed=(fp32._replace(shape=(2, 128)),))


def test_a_packed_buffer_is_refused_unaligned_past_a_box_or_on_a_whole_tile_family():
    """A packed chunk's buffer is a whole number of 128 bytes, so each of a stack of them starts aligned for
    its tensor copy (a transposed [3, 64] byte buffer, 192 bytes, refused), at most 256 elements an axis
    (a transposed 512-column chunk's [512, 64] refused), and a family staging whole tiles (mma.sync's)
    refuses an output packed or written by its own write."""
    aligned = stores.Output((64, 192), jnp.uint8, 32, transposed=True)
    with pytest.raises(ValueError):
        stores.output_staging(192, (jnp.uint8,), (96, 2), packed=(aligned,))
    wide = stores.Output((64, 512), jnp.uint8, 1, transposed=True)
    with pytest.raises(ValueError):
        stores.output_staging(512, (jnp.uint8,), (512, 1), packed=(wide,))
    whole = bundles.named("mma.sync", jnp.bfloat16, None)
    with pytest.raises(ValueError):
        whole.output_staging((stores.Output((64, 64), jnp.bfloat16, write=stores.written),), 1)
    with pytest.raises(ValueError):
        whole.output_staging((NVFP4[0]._replace(shape=(64, 64)),), 1)


def test_a_ring_family_stages_plain_and_packed_outputs_side_by_side():
    """A wgmma ring family's staging of a plain bf16 output beside NVFP4's codes and a transposed fp32 scale:
    each its own buffers, the plain one its derived chunk of 64 bf16 columns, the codes 128 tile columns a
    chunk in 64 bytes, the scale `[1, 64]` a 128-column chunk."""
    family = bundles.named("wgmma", jnp.bfloat16, None)
    scale = stores.Output((64, 256), jnp.float32, 128, transposed=True)
    staged = family.output_staging((((64, 256), jnp.bfloat16), NVFP4[0], scale), 2)
    assert [(buffer.shape, jnp.dtype(buffer.dtype)) for buffer in staged] == [
        ((2, 2, 64, 64), jnp.dtype(jnp.bfloat16)), ((2, 2, 64, 64), jnp.dtype(jnp.uint8)),
        ((2, 2, 1, 64), jnp.dtype(jnp.float32))]


def _packed_store(rows: int, columns: int, divisor: int, chunk: int, *, transposed: bool, first=0,
                  stored=None, least: int | None = None, out_rows: int | None = None,
                  out_columns: int | None = None):
    """A kernel staging a share of `rows` by `columns` tile columns, each its column's index plus a thousand
    times its row, through two buffers of `chunk` tile columns stored packed (`stores.staged_chunks`), its
    write keeping each `divisor`th column (a stand-in for a block's scale), transposed where asked, its
    share's rows `first` to `first + stored` (the share where None), copied in rungs of at least `least` rows
    (the share's), into an output of `out_rows` rows (the share's) and `out_columns` stored columns (the
    tile's)."""
    packed_columns = chunk // divisor
    block = (packed_columns, rows) if transposed else (rows, packed_columns)

    def write(buffer, piece):
        kept = piece[:, ::divisor]
        buffer[...] = kept.T if transposed else kept

    def body(out_gmem):
        def scoped(smem):
            value = (jax.lax.broadcasted_iota(jnp.float32, (rows, columns), 1)
                     + 1000.0 * jax.lax.broadcasted_iota(jnp.float32, (rows, columns), 0))
            place = (0, first, rows if stored is None else stored, 0, rows, rows if least is None else least)
            stores.staged_chunks(smem, value, out_gmem, place, write=write, divisor=divisor,
                                 transposed=transposed)
            plgpu.wait_smem_to_gmem(0)

        swizzling.run_scoped(scoped, plgpu.SMEM((2, *block), jnp.float32))

    total = rows if out_rows is None else out_rows
    width = columns // divisor if out_columns is None else out_columns
    out = (width, total) if transposed else (total, width)
    return plgpu.kernel(body, out_type=jax.ShapeDtypeStruct(out, jnp.float32),
                        interpret=plgpu.InterpretGPUParams())


@INTERPRETS
@pytest.mark.parametrize("transposed", [False, True], ids=["row-major", "transposed"])
def test_a_packed_output_stores_each_chunks_columns_at_its_stored_place(transposed):
    """Interpreted on the host's CPU: a share stored packed four tile columns a stored column through
    16-column chunks (`stores.staged_chunks`' `divisor`) holds each kept column at its stored place, row-major
    `[rows, columns // 4]` or transposed `[columns // 4, rows]`, every chunk's at its own."""
    rows, columns = 8, 64
    with on_the_host():
        got = np.asarray(_packed_store(rows, columns, 4, 16, transposed=transposed)())
    expected = np.arange(0, columns, 4)[None, :] + 1000.0 * np.arange(rows)[:, None]
    np.testing.assert_array_equal(got, expected.T if transposed else expected)


@INTERPRETS
@pytest.mark.parametrize("first, stored", [(jnp.int32(0), 8), (0, 4), (0, jnp.int32(8))],
                         ids=["traced-first", "interior", "traced-rows"])
def test_a_transposed_store_of_a_share_of_some_of_its_rows_is_refused(first, stored):
    """A transposed output stores its share's rows whole, so a share of rows not statically its own (a grouped
    tile's group starting mid-tile, a share ending mid-tensor, rows known only as it runs) is refused as it
    traces, before any store, where its copy would overwrite rows that are not its own."""
    with on_the_host(), pytest.raises(ValueError):
        _packed_store(8, 64, 4, 16, transposed=True, first=first, stored=stored)()


@INTERPRETS
@pytest.mark.parametrize("first, least", [(jnp.int32(0), 8), (0, 1)], ids=["traced-first", "single-rows"])
def test_a_packed_row_ladder_starting_unaligned_is_refused(first, least):
    """A packed chunk's 16-byte rows (four fp32 a row) copied in rungs that may start at a row no 128 bytes
    from its buffer's (a first row known only as it runs, rungs of one row) are refused as they trace: a
    tensor copy's shared address is 128-byte aligned. A static first row 0 and rungs of 8 rows hold
    (`test_a_packed_output_stores_each_chunks_columns_at_its_stored_place`)."""
    with on_the_host(), pytest.raises(ValueError):
        _packed_store(8, 64, 4, 16, transposed=False, first=first, least=least)()




def test_a_ladder_is_aligned_exactly_where_every_copy_it_takes_starts_on_a_shared_unit():
    """`stores.row_starts_aligned` against the copies `store_rows` takes, each power-of-two rung from its
    largest down to its smallest once, by its bit of the rows stored, for every row count it stores (a
    multiple of its smallest below twice its largest): from a static first row 0, aligned exactly where
    every copy starts at a whole number of 128-byte units; from a first row known only as it runs, exactly
    where a row is whole units (the derivation and the trace read one rule, row 14182)."""
    def starts(rows: int, least: int, most: int) -> list[int]:
        at, found, rung = 0, [], most
        while rung >= least:
            if rows & rung:
                found.append(at)
                at += rung
            rung //= 2
        return found

    unit = copies.SHARED_ALIGNMENT
    for row_bytes in (16, 32, 64, 128, 192, 256):
        for least, most in ((1, 1), (1, 8), (2, 8), (4, 8), (8, 8), (1, 64), (8, 64)):
            every = all(start * row_bytes % unit == 0
                        for rows in range(least, 2 * most, least)
                        for start in starts(rows, least, most))
            aligned = {static: stores.row_starts_aligned(row_bytes, first_static=static, least=least,
                                                         most=most) for static in (True, False)}
            assert aligned[True] == every, (row_bytes, least, most)
            assert aligned[False] == (row_bytes % unit == 0)

@INTERPRETS
@pytest.mark.parametrize("chunk, first, refused", [(16, 0, False), (16, jnp.int32(0), True), (8, 0, True)],
                         ids=["dense-64-byte-rows", "grouped-64-byte-rows", "dense-32-byte-rows"])
def test_an_ordinary_row_ladder_starts_each_copy_at_an_aligned_shared_row(chunk, first, refused):
    """An ordinary chunk's rows (fp32, `chunk` columns a row) copied in rungs down to one row: from a static
    first row 0 each copy starts at a multiple of twice the smallest rung, so 64-byte rows start every copy
    128 bytes aligned and store as any share does; a first row known only as it runs (a grouped tile's
    group) and 32-byte rows from row 0 (a one-row rung at row 2, 64 bytes in) are refused as they trace, a
    tensor copy's shared address being 128-byte aligned (row 14182: only packed chunks were held to it)."""
    if refused:
        with on_the_host(), pytest.raises(ValueError):
            _packed_store(8, 64, 1, chunk, transposed=False, first=first, least=1)()
        return
    with on_the_host():
        stored = np.asarray(_packed_store(8, 64, 1, chunk, transposed=False, first=first, least=1)())
    expected = np.arange(64)[None, :] + 1000.0 * np.arange(8)[:, None]
    np.testing.assert_array_equal(stored, expected)

def test_a_packed_chunk_is_sought_within_the_tile_and_its_issuers_piece_alike():
    """A 96-column tile whose issuer emits 64 columns at once stages NVFP4 codes (two a byte) in chunks
    dividing both, 32 columns (16 bytes a row), not 64, which the tile's last 32 columns do not fill."""
    codes = stores.Output((64, 96), jnp.uint8, 2)
    assert stores.output_staging(96, (jnp.uint8,), piece=64, packed=(codes,)) == (32, 2)


@INTERPRETS
def test_a_one_column_transposed_chunk_whose_last_share_passes_the_rows_end_is_refused():
    """A transposed output a column a chunk (a 16-column block a stored column, chunks of 16) is one
    contiguous run a share, which Mosaic GPU copies by a linear bulk copy clamped at the tensor's end, not its
    row's: an 8-row share over 12 rows would run its last share's rows into the next column's, so it is
    refused as it traces; over 16 rows each share ends at a row's end."""
    with on_the_host(), pytest.raises(ValueError):
        _packed_store(8, 32, 16, 16, transposed=True, out_rows=12)()


@INTERPRETS
def test_a_packed_chunk_copied_a_row_at_a_time_past_its_rows_end_is_refused():
    """A packed row-major chunk of 32 fp32 (128 bytes, aligned) copied in rungs of one row is one contiguous
    run a rung, which Mosaic GPU copies by a linear bulk copy clamped at the tensor's end: over 36 stored
    columns its last chunk would run into the next row, so it is refused as it traces."""
    with on_the_host(), pytest.raises(ValueError):
        _packed_store(8, 128, 4, 128, transposed=False, least=1, out_columns=36)()
