"""The GEMM template's gathered rows (row 1376): a grouped product whose rows are read from the tokens where a
table says, a mixture of experts' sort fused into the rows' loads (`mosaic_gpu.gmm`'s `gather`), and a kernel
gradient's operands read so (`mosaic_gpu.tgmm`'s)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import mx_matrix
import numpy as np
import pytest
from gathered_walks import gathered_kernel
from generation_support import cublas_compiled
from jax.sharding import Mesh
from map_judge import marked_seeds
from reference_error import assert_as_exact, equations

from dew.nn.kernels import formats, gemm, generation, grouped_launch, grouped_matmul, tuning
from dew.nn.kernels.templates import product, rings
from dew.nn.kernels.templates.gemm import mosaic_gpu, row_tiles, tiling

SEEDS = 8
"""The gathered product's check-1 seeds, the template's grouped products' count."""


def test_a_gathered_grouped_product_reads_each_row_where_its_table_says():
    """`a_gathered_product_is_on_par` on an H100, interpreted (each row a copy at its slot's swizzled row,
    `Ring.parts`), plain and paired; a gathered product on mma.sync, which has no producer warpgroup, is
    refused (`Family.gathers_rows`)."""
    with generation.assumed("sm90"):
        a_gathered_product_is_on_par(interpret=True)
        a_gathered_product_is_on_par(interpret=True, paired=True, scattered=True)
        a_deep_gathered_step_reads_its_rows_as_their_copy_would(grouped_launch.INTERPRETED)
    tokens, table = jnp.zeros((200, 128), jnp.bfloat16), jnp.zeros((256,), jnp.int32)
    with generation.assumed("sm80"), pytest.raises(ValueError):
        mosaic_gpu.gmm(tokens, jnp.zeros((4, 128, 256), jnp.bfloat16),
                       jnp.asarray([70, 0, 129, 57], jnp.int32),
                       tiles=tiling.Tiles(128, 128, 32, 3, 1, family="mma.sync"), transpose_rhs=False,
                       out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_launch.INTERPRETED,
                       gather=table)


def a_deep_gathered_step_reads_its_rows_as_their_copy_would(interpret):
    """A gathered product's 256-deep step, four swizzled tiles a row (its rows' copies, `rings.piece_copies`,
    the load barrier's arrivals: counted apart, the packed NVFP4 bank's gathered walk hung): bitwise the same
    tile's product of the rows gathered first, its groups ragged, one empty, rows past them."""
    tiles = tiling.Tiles(128, 128, 256, 2, 1, warpgroups=2, family='wgmma')
    sizes = jnp.asarray([70, 0, 129, 51], jnp.int32)
    rng = np.random.default_rng(7)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(200, 512)), jnp.bfloat16)
    table = rng.permutation(np.concatenate([np.arange(150), rng.integers(0, 200, 106)]))
    table = jnp.asarray(table, jnp.int32)
    matrices = jnp.asarray(rng.integers(-3, 4, size=(4, 512, 256)), jnp.bfloat16)

    def product(rows, gather):
        return mosaic_gpu.gmm(rows, matrices, sizes, tiles=tiles, transpose_rhs=False, out_dtype=jnp.float32,
                              multiprocessors=4, interpret=interpret, gather=gather)

    np.testing.assert_array_equal(np.asarray(product(tokens, table)),
                                  np.asarray(product(tokens[table], None)))


def a_gathered_product_is_on_par(interpret: bool = False, *, paired: bool = False, scattered: bool = False,
                                 pinned: bool = True, cublas: bool = False):
    """A grouped product whose rows are gathered from the tokens by a table (`gmm`'s `gather`: a MoE's sort
    fused into the rows' loads, row 1376), its groups ragged, one empty, rows past them, a token read by
    several slots and some by none, through the template named (`gemm.gemm`, `grouped_matmul` chosen `mosaic`:
    the product the call's precision declares, its sums promoted where it asks for them) at a pinned
    cooperative tile (`tuning.pinned`) or, not `pinned`, the tile it derives, on par with the arm Dew's
    selector runs for the same call where the template is not taken (check 1: `grouped_matmul` chosen `xla`,
    the gathered rows' ragged dot), from float64: on this GPU (test_gpu_gemm's cell, the producer's cp.async
    gather), or interpreted (`interpret`); `paired`, a second matrix's product beside it; `scattered`, each
    row stored at its entry of a permutation (`gmm`'s `scatter`: the MoE's un-permute fused into the stores).
    Called through `mosaic_gpu.gmm` itself, the tile summed its products in the tensor cores' running sum
    whatever the precision asked: 1.658x the reference at the suite's highest precision on a GH200 (M-EM-19),
    as summed through gemm.gemm its error is the reference's. `cublas`, at the default precision on a Hopper
    (a user's process's cells): the template's default form, warpgroup MMA's running sum as cuBLAS's own, is
    held to cuBLAS on the same inputs by the library arm (BIBLE §7's exception, row 5491; the owner's
    ruling of 2026-11-12, "Ofcourse Cublas parity!"), within 1.5 of the larger of the reference's error and
    cuBLAS's: XLA's default product there is a Triton fusion on mma.sync, whose running sum the template's
    read 1.658 times the error of (M-EM-19c), where cuBLAS's equals the template's."""
    tiles = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    # NumPy's, formed on no device: a GPU tier's prepare computes on none (its default a compile-only GPU's).
    sizes, depth = np.asarray([70, 0, 129, 51], np.int32), 128
    # A pair summed apart (into fp32 above the default precision) carries four accumulators, which the
    # cooperative 128 x 128 tile's threads spill (the compile gate's "paired forward exact fp32"; 324 bytes
    # on a GH200, M-EM-19c): a pin of it is refused, and its 128 x 64 holds them.
    apart = paired and product.declared(jnp.bfloat16, None, outputs=(jnp.float32,)).promoted
    refused, tiles = (tiles, dataclasses.replace(tiles, block_n=64)) if apart else (None, tiles)

    def draw(seed):
        rng = np.random.default_rng(seed)
        tokens = rng.normal(size=(200, depth)).astype(jnp.bfloat16)
        table = rng.permutation(np.concatenate([np.arange(150), rng.integers(0, 200, 106)])).astype(np.int32)
        matrices = (rng.normal(size=(2, 4, depth, 256)) / np.sqrt(depth)).astype(jnp.bfloat16)
        slots = rng.permutation(256).astype(np.int32)
        return tokens, table, matrices, slots

    def through(kernel, pin=tiles if pinned else None):
        def run(tokens, table, matrices, slots):
            # The call through Dew's selector (`grouped_matmul` chosen `kernel`): the template's at the pinned
            # tile, or XLA's ragged dot of the gathered rows as they are (bf16 products summed into fp32), its
            # rows stored at their slots; the float64 truth XLA's over the gate's widened draws.
            wide = jnp.promote_types(tokens.dtype, jnp.float32)
            pins = {} if pin is None else {grouped_matmul.MosaicGmm: pin}
            with generation.chosen(grouped_matmul=kernel), tuning.pinned(pins):
                outs = gemm.gemm(tokens, matrices[0], sizes, out_dtype=wide,
                                 out_dtypes=(wide,) * 2 if paired else None,
                                 paired=matrices[1] if paired else None, gather=table,
                                 scatter=slots if scattered else None, interpret_on_cpu=interpret)
            return jnp.stack(outs) if paired else outs
        return run

    def by_cublas(tokens, table, matrices, slots):
        # cuBLAS's GEMM of each group's gathered rows by its matrix into fp32 (XLA's dot, its Triton GEMMs
        # off: `generation_support.cublas_compiled`; on a Hopper XLA calls it through cuBLASLt,
        # `__cublas$lt$matmul`), the groups' rows and the unrouted rows' zeros placed as the product places
        # them, its rows stored at their slots: no arithmetic past the library's.
        rows = tokens[table]
        firsts = np.concatenate([[0], np.cumsum(np.asarray(sizes))])

        def product(matrices):
            parts = [jnp.dot(rows[first:last], matrix, preferred_element_type=jnp.float32)
                     for first, last, matrix in zip(firsts[:-1], firsts[1:], matrices, strict=True)]
            out = jnp.concatenate([*parts, jnp.zeros((rows.shape[0] - firsts[-1], 256), jnp.float32)])
            return jnp.zeros_like(out).at[slots].set(out) if scattered else out

        return jnp.stack([product(matrices[0]), product(matrices[1])]) if paired else product(matrices[0])

    library = ({'library_fn': cublas_compiled(by_cublas, draw(0), "__cublas$lt$matmul"), 'library_row': 5491,
                'library_leaves': ("",)} if cublas else {})
    if refused is not None and pinned:
        with pytest.raises(tuning.Unpinnable):
            jax.eval_shape(through("mosaic", refused), *draw(0))
    # One site a variant: a store of compiled truths holds one site a label.
    variant = " ".join(name for name, on in (("paired", paired), ("scattered", scattered), ("pinned", pinned),
                                             ("cublas", cublas)) if on)
    with tuning.seen() as launches:
        assert_as_exact(jax.jit(through("mosaic")), jax.jit(through("xla")), draw,
                        f"a gathered grouped product: {variant or 'derived'}", seeds=SEEDS,
                        truth_fn=through("xla"), **library)
    taken = [config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
             if type(op) is grouped_matmul.MosaicGmm]
    assert taken and all(config == tiles for config in taken) if pinned else taken, taken


@pytest.mark.parametrize("paired", [False, True])
def test_a_gathered_product_lowers_for_cuda(paired):
    """A gathered grouped product on a compile-only H100 lowers through Mosaic GPU: its producer's
    cp.async gather and its load barriers' lane-level arrivals and parity waits (`gathered.Ring`), plain and
    paired, the paired one's rows stored at a permutation's entries (`gathered.scattered_rows`)."""
    tiles = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    sizes = jnp.asarray([700, 0, 1290, 58], jnp.int32)
    product = functools.partial(mosaic_gpu.gmm, tiles=tiles, transpose_rhs=False, out_dtype=jnp.bfloat16,
                                out_dtypes=(jnp.bfloat16,) * 2 if paired else None, multiprocessors=132,
                                interpret=False)

    def gathered(tokens, matrices, table):
        return product(tokens, matrices[0], sizes, gather=table, paired=matrices[1] if paired else None,
                       scatter=table if paired else None)

    with generation.assumed("sm90"):
        jax.jit(gathered).trace(jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2, 4, 512, 256), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2048,), jnp.int32)).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize(("generation_name", "splits"), [("sm90", 1), ("sm90", 2), ("sm100", 1)])
def test_a_gathered_kernel_gradient_reads_each_operands_rows_where_its_table_says(generation_name, splits):
    """`a_gathered_kernel_gradient_is_its_gathered_rows_gradient` on an H100 (its depth split or not) and a
    B200, interpreted, each family's tile (`KERNEL_GRADIENT_TILES`: Hopper's producer warpgroup's cp.async
    gathers, Blackwell's load warp's TMA gathers); a gathered operand beside a prologue runs fused on wgmma
    (row 5391) and is refused on tcgen05 (no consumer-issued prologue) and mma.sync (no producer gather)."""
    with generation.assumed(generation_name):
        tokens, cotangent, table = a_gathered_kernel_gradient_is_its_gathered_rows_gradient(
            grouped_launch.INTERPRETED, splits, tiles=KERNEL_GRADIENT_TILES[generation_name])
    sizes = jnp.asarray([70, 0, 129, 51], jnp.int32)
    if generation_name == "sm90" and splits == 1:
        with generation.assumed("sm90"):
            fused = mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=KERNEL_GRADIENT_TILES["sm90"],
                                    out_dtype=jnp.float32, multiprocessors=4,
                                    interpret=grouped_launch.INTERPRETED, gather=(table, None),
                                    prologue=lambda rows: rows * jnp.bfloat16(1.5))
            plain = mosaic_gpu.tgmm(tokens[table], cotangent, sizes, tiles=KERNEL_GRADIENT_TILES["sm90"],
                                    out_dtype=jnp.float32, multiprocessors=4,
                                    interpret=grouped_launch.INTERPRETED,
                                    prologue=lambda rows: rows * jnp.bfloat16(1.5))
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(plain))
    with generation.assumed("sm100"), pytest.raises(ValueError):
        mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=KERNEL_GRADIENT_TILES["sm100"],
                        out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_launch.INTERPRETED,
                        gather=(table, None), prologue=lambda rows: rows)
    with generation.assumed("sm80"), pytest.raises(ValueError):
        mosaic_gpu.tgmm(tokens, cotangent, sizes,
                        tiles=tiling.Tiles(128, 128, 32, 3, 1, family="mma.sync"), out_dtype=jnp.float32,
                        multiprocessors=4, interpret=grouped_launch.INTERPRETED, gather=(table, None))


def test_a_gathered_kernel_gradients_last_step_addresses_no_step_past_it():
    """A Blackwell's gathered kernel gradient of one step (64 rows, one group, a 64-entry identity table, the
    128 x 128 x 64 tcgen05 tile), interpreted with every out-of-bounds read raised: its loads address no step
    past the walk's last (`rings.Ahead.view` resolved by the load that reads a step ahead alone), and it is
    on par with the gradient of its rows gathered first (check 1); resolved eagerly, the last step's next
    view read `table[64]` (base review 103007773 F1)."""
    strict = dataclasses.replace(grouped_launch.INTERPRETED, out_of_bounds_reads="raise")
    table, sizes = jnp.arange(64, dtype=jnp.int32), jnp.asarray([64], jnp.int32)

    def gathered(tokens):
        return mosaic_gpu.tgmm(tokens, tokens, sizes, tiles=KERNEL_GRADIENT_TILES["sm100"],
                               out_dtype=jnp.float32, multiprocessors=4, interpret=strict,
                               gather=(table, None))

    def first(tokens):
        rows = tokens[table]
        return jnp.dot(rows.T, rows, preferred_element_type=jnp.float32)[None]

    def draw(seed):
        return (np.random.default_rng([seed, 5]).normal(size=(64, 128)).astype(jnp.bfloat16),)

    with generation.assumed("sm100"):
        assert_as_exact(gathered, jax.jit(first), draw, "a one-step gathered kernel gradient, strictly read",
                        seeds=SEEDS, truth_fn=lambda tokens: (tokens[table].T @ tokens[table])[None])


def _strided_gathers(tokens, table, plain, at: int | None = None):
    """`plain @ tokens[table[:, 0]]` a step of `STRIDED_BLOCK`'s rows at a time through a Blackwell's
    gathered ring (`gathered_walks.gathered_kernel`), its table's column zero the ring's (`rings.Rows` over a
    view of stride two): one tile; or, `at` given, every step's rows those of a flat `table` from entry `at`
    (a Python int, past int32's reach where the table is:
    `test_a_gathered_rings_steps_past_int32s_reach_lower_for_cuda`)."""
    (height, width), rows = STRIDED_BLOCK, plain.shape[0]

    def views(operands, tile):
        del tile
        x_gmem, table_gmem, w_gmem, _ = operands

        def entries(step):
            return (rings.Rows(table_gmem.at[:, 0], step * height, (0,), 1) if at is None
                    else rings.Rows(table_gmem, at, (0,), 1))

        return {"x": (x_gmem, entries), "w": (w_gmem, lambda step: (0, step))}

    blocks = {"x": ((height, width), jnp.bfloat16), "w": ((rows, height), jnp.bfloat16)}
    kernel = gathered_kernel(blocks, ("x",), (plain.shape[1] // height,), views, (rows, width), stages=2,
                             out=jax.ShapeDtypeStruct((rows, width), jnp.float32), stored=True,
                             name="strided_gathers")
    return kernel(tokens, table, plain)


STRIDED_BLOCK = (64, 64)
"""`_strided_gathers`' gathered step: 64 rows, a tcgen05 product's least depth (eight 8-row tiles over the
issuing warpgroup's load warps)."""


def a_gathered_ring_reads_a_strided_tables_entries() -> None:
    """A Blackwell's gathered ring (`gathered.Committed`: its load warps' gathers, each step's table entries
    read a step ahead) whose table is column zero of a row-major int32 `[128, 2]` table, the first step's
    rows `[0, 1]` and the second's `[1, 0]` (a view of stride two, `_strided_gathers`): the first step
    gathers token 0's rows, the second token 1's. Two steps of 64 gathered rows times a plain block, on par
    with the product of its rows gathered first (check 1); entries read at stride one gather the other token's
    for half the rows (base review ae8a242ed F1), and the first step's entries read again for the second
    gather token 0's (base review 51d5c0f0a F2). On this GPU (test_gpu_gemm's Blackwell cell): its load
    warps' code runs on none other."""
    (height, width), steps, rows = STRIDED_BLOCK, 2, 128
    table = np.concatenate([np.tile([0, 1], (height, 1)), np.tile([1, 0], (height, 1))]).astype(np.int32)

    def gathered(tokens, plain):
        return _strided_gathers(tokens, jnp.asarray(table, jnp.int32), plain)

    def first(tokens, plain):
        return jnp.dot(plain, tokens[jnp.asarray(table[:, 0], jnp.int32)], preferred_element_type=jnp.float32)

    def draw(seed):
        rng = np.random.default_rng([seed, 9])
        return (rng.normal(size=(2, width)).astype(jnp.bfloat16),
                rng.normal(size=(rows, steps * height)).astype(jnp.bfloat16))

    with generation.assumed("sm100"):
        assert_as_exact(gathered, jax.jit(first), draw, "a gathered ring over a strided table", seeds=SEEDS,
                        truth_fn=lambda tokens, plain: plain @ tokens[jnp.asarray(table[:, 0], jnp.int32)])

def test_a_gathered_ring_over_a_strided_table_lowers_for_cuda():
    """`a_gathered_ring_reads_a_strided_tables_entries`' kernel lowered for a B200 on this CPU, its table a
    stride-two column, the load warps' entries addressed by it (`gathered.Held.filled`)."""
    (height, width), steps, rows = STRIDED_BLOCK, 2, 128
    shaped = jax.ShapeDtypeStruct
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(_strided_gathers).trace(
            shaped((2, width), jnp.bfloat16), shaped((steps * height, 2), jnp.int32),
            shaped((rows, steps * height), jnp.bfloat16)).lower(lowering_platforms=("cuda",))


def test_a_gathered_rings_steps_past_int32s_reach_lower_for_cuda():
    """`_strided_gathers` over a flat int32 table of 2^31 + 64 entries, both steps' rows from entry 2^31 (a
    Python int), lowered for a B200 on this CPU: the next step's slice formed as the load's own (`pl.ds`,
    the table's index type) where the walk has that step, no narrowing to int32 (base review ac90e3e16 F1:
    `OverflowError` in tracing)."""
    (height, width), rows = STRIDED_BLOCK, 128
    shaped = jax.ShapeDtypeStruct
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_strided_gathers, at=2 ** 31)).trace(
            shaped((2, width), jnp.bfloat16), shaped((2 ** 31 + height,), jnp.int32),
            shaped((rows, 2 * height), jnp.bfloat16)).lower(lowering_platforms=("cuda",))


KERNEL_GRADIENT_TILES = {"sm90": tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma'),
                         "sm100": tiling.Tiles(128, 128, 64, 4, 1, family='tcgen05')}
"""A gathered kernel gradient's tile on each family whose producer gathers its rings' rows: Hopper's
producer warpgroup's cp.async gathers and Blackwell's load warp's TMA four-row gathers (row 1376)."""


def a_gathered_kernel_gradient_is_its_gathered_rows_gradient(interpret=False, splits: int = 1,
                                                            tiles: tiling.Tiles | None = None) -> tuple:
    """A kernel gradient's gathered operands (`mosaic_gpu.tgmm`'s `gather`, row 1376: the tokens a mixture of
    experts' slots read, and a scattered product's cotangent at its slots), bitwise the same tile's gradient
    of the rows gathered first, the tokens' rows alone and both operands', its groups ragged, one empty, rows
    past them, its depth split or not, at 257 rows (a table padded to whole steps, an ungathered operand to
    8 rows), at `tiles` (Hopper's by default; `KERNEL_GRADIENT_TILES`). Integers, every sum exact. Its
    tokens, cotangent and table."""
    from reference_error import ahead_arm

    tiles = dataclasses.replace(tiles or KERNEL_GRADIENT_TILES["sm90"], splits=splits)
    sizes = np.asarray([70, 0, 129, 51], np.int32)
    # NumPy's, as a site's draws are: a GPU tier's prepare computes on no device.
    rng = np.random.default_rng(11)
    tokens = rng.integers(-3, 4, size=(200, 256)).astype(jnp.bfloat16)
    table = rng.permutation(np.concatenate([np.arange(150), rng.integers(0, 200, 107)])).astype(np.int32)
    cotangent = rng.integers(-3, 4, size=(257, 128)).astype(jnp.bfloat16)
    slots = rng.permutation(257).astype(np.int32)
    gradient = functools.partial(mosaic_gpu.tgmm, group_sizes=sizes, tiles=tiles, out_dtype=jnp.float32,
                                 multiprocessors=4 if interpret else 132, interpret=interpret)

    def gathered(rows, columns, *tables):
        return gradient(rows, columns, gather=(*tables, None)[:2])

    # Each program compiled in a tier's prepare (`ahead_arm`), one a label.
    for name, tables, rows, columns in (("tokens", (table,), tokens[table], cotangent),
                                        ("both", (table, slots), tokens[table], cotangent[slots])):
        label = f"a gathered kernel gradient of {name}: {splits} splits"
        found = ahead_arm(jax.jit(gathered), label)(tokens, cotangent, *tables)
        expected = ahead_arm(jax.jit(gradient), f"{label}, gathered first")(rows, columns)
        np.testing.assert_array_equal(np.asarray(found), np.asarray(expected))
    return tokens, cotangent, table


@pytest.mark.parametrize("generation_name", KERNEL_GRADIENT_TILES)
def test_each_rings_gathered_kernel_gradient_lowers_for_cuda(generation_name):
    """A kernel gradient of gathered tokens and a scattered product's cotangent on a compile-only H100 and
    B200 lowers through Mosaic GPU: each operand's ring its producer's gather, by the gradient's depth or
    width."""
    sizes = jnp.asarray([700, 0, 1290, 58], jnp.int32)
    target = generation.gpu_target(generation_name)

    def gathered(tokens, cotangent, table, slots):
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=KERNEL_GRADIENT_TILES[generation_name],
                               out_dtype=jnp.float32, multiprocessors=target.multiprocessors, interpret=False,
                               gather=(table, slots))

    with generation.assumed(generation_name):
        jax.jit(gathered).trace(jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2048, 256), jnp.bfloat16),
                                *(jax.ShapeDtypeStruct((2048,), jnp.int32),) * 2).lower(
                                    lowering_platforms=("cuda",))


def test_a_gathered_kernel_gradients_key_meets_its_table_and_rows_at_one_count():
    """A gathered kernel gradient's key (`_GroupedLaunch.keyed`, the call a tuning run materializes and
    times) buckets its table as it buckets its ungathered operand's rows, 500 to 512 each, so the launch it
    measures takes its own operands; the gathered tokens keep a count of their own."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    with generation.assumed("sm90"):
        launch = grouped_matmul.MosaicTgmm()
        table = jax.ShapeDtypeStruct((500,), jnp.int32)
        ba = launch.bind(bf16((100, 256)), bf16((500, 128)), jax.ShapeDtypeStruct((4,), jnp.int32), (),
                         (table, None), out_dtype="float32", interpret=False)
        keyed = launch.keyed(ba).arguments
    assert keyed["gather"][0].shape == (512,) and keyed["gather"][1] is None
    assert keyed["cotangent"].shape[0] == 512 and keyed["tokens"].shape[0] == 128


def test_a_blackwells_gathered_product_reads_its_rows_by_its_load_warps_gathers():
    """`a_blackwells_gathered_product_is_its_gathered_rows_product` on an sm100, interpreted (each row a copy
    where the GPU's load warp gathers four)."""
    with generation.assumed("sm100"):
        a_blackwells_gathered_product_is_its_gathered_rows_product(interpret_on_cpu=True)


def a_blackwells_gathered_product_is_its_gathered_rows_product(interpret_on_cpu: bool = False):
    """A gathered, scattered grouped product on a Blackwell GPU (row 1376): tcgen05's bundle takes it (its
    load warp's TMA four-row gathers, `gathered.Committed`), its value bitwise the product of the rows
    gathered first and put at their slots after; and its kernel gradient's tables on the template too (row
    4852: mma.sync's consumer-fed rings, their loading consumer gathering each step's rows, `gathered.Ring`),
    bitwise the gradient of the rows gathered first. Integers, every sum exact."""

    rng = np.random.default_rng(13)
    tokens = jnp.asarray(rng.integers(-1, 2, size=(200, 256)), jnp.bfloat16)
    matrices = jnp.asarray(rng.integers(-1, 2, size=(3, 256, 256)), jnp.bfloat16)
    table = jnp.asarray(rng.integers(0, 200, 384), jnp.int32)
    slots = jnp.asarray(rng.permutation(384), jnp.int32)
    cotangent = jnp.asarray(rng.integers(-1, 2, size=(384, 256)), jnp.bfloat16)
    sizes = jnp.asarray([128, 0, 200], jnp.int32)
    options = {"out_dtype": jnp.bfloat16, "precision": "default", "interpret_on_cpu": interpret_on_cpu}
    # The template's launch named, its product at tcgen05's tile (`tuning.pinned`, a pin its legality checks).
    tile = tiling.Tiles(128, 128, 64, 4, 1, family='tcgen05')
    with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({grouped_matmul.MosaicGmm: tile}):
        got = jax.jit(lambda: gemm.gemm(tokens, matrices, sizes, gather=table, scatter=slots, **options))()
        apart = jax.jit(lambda: gemm.gemm(tokens[table], matrices, sizes, **options))()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jnp.zeros_like(apart).at[slots].set(apart)))
    gradient = {**options, "out_dtype": jnp.float32}
    with generation.chosen(grouped_matmul="mosaic"):
        got = jax.jit(lambda: gemm.gemm_t(tokens, cotangent, sizes, gather=(table, slots), **gradient))()
    expected = jax.jit(lambda: gemm.gemm_t(tokens[table], cotangent[slots], sizes, **gradient))()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_a_blackwells_gathered_product_lowers_for_cuda():
    """A gathered grouped product on a compile-only B200 lowers through Mosaic GPU on tcgen05's bundle: its
    load warp's TMA four-row gathers into the swizzled ring (`gathered.Gathers`), each slot released by its
    products' commit; and its kernel gradient's tables at the tile its launch derives for a wide and a
    32-column cotangent, one its launch takes (`legal`), on whichever family derives it."""
    tiles = tiling.Tiles(128, 128, 64, 4, 1, family='tcgen05')
    sizes = jnp.asarray([700, 0, 1290, 58], jnp.int32)

    def gathered(tokens, matrices, table):
        return mosaic_gpu.gmm(tokens, matrices, sizes, tiles=tiles, transpose_rhs=False,
                              out_dtype=jnp.bfloat16, multiprocessors=148, interpret=False, gather=table)

    def gradient(tokens, cotangent, table, slots):
        bound = launch.bind(*(jax.ShapeDtypeStruct(each.shape, each.dtype) for each in (tokens, cotangent)),
                            jax.ShapeDtypeStruct(sizes.shape, sizes.dtype), (), (table, slots),
                            out_dtype="float32", interpret=False)
        tile = launch.derived(bound, generation.gpu_target())
        assert tile is not None and launch.legal(bound, tile, generation.gpu_target()), tile
        # The template's call takes its operands padded to whole tiles, as its launch pads them.
        cotangent = jnp.pad(cotangent, ((0, 0), (0, -cotangent.shape[1] % tile.block_n)))
        return mosaic_gpu.tgmm(tokens, cotangent, sizes, tiles=tile, out_dtype=jnp.float32,
                               multiprocessors=148, interpret=False, gather=(table, slots))

    launch = grouped_matmul.MosaicTgmm()

    with generation.assumed("sm100"):
        jax.jit(gathered).trace(jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                                jax.ShapeDtypeStruct((4, 512, 256), jnp.bfloat16),
                                jax.ShapeDtypeStruct((2048,), jnp.int32)).lower(lowering_platforms=("cuda",))
        # A narrow cotangent's stage too, its tile the launch's own, staged at the swizzle its row holds
        # (`gathered.swizzled`).
        for width in (256, 32):
            operands = (jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                        jax.ShapeDtypeStruct((2048, width), jnp.bfloat16),
                        *(jax.ShapeDtypeStruct((2048,), jnp.int32),) * 2)
            jax.jit(gradient).trace(*operands).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("format_", ["mxfp8", "nvfp4"])
def test_a_blackwells_gathered_block_scaled_product_gathers_its_scales_by_its_table(format_):
    """A gathered block-scaled product on an sm100, interpreted (row 4851: `mxfp8_product`'s and
    `nvfp4_block_product`'s `gather`): its codes read where they lie by tcgen05's gather and its rows' block
    scales by the same table, bitwise the product of the operand gathered first (codes and scales), its
    groups whole 128-row blocks, one empty."""

    choice, form, product = {"mxfp8": ("mxfp8", formats.MxFp8, gemm.mxfp8_product),
                             "nvfp4": ("nvfp4_block", formats.NvFp4Block, gemm.nvfp4_block_product)}[format_]
    rng = np.random.default_rng(17)
    tokens = form.quantized(jnp.asarray(rng.normal(size=(300, 256)), jnp.float32))
    matrices = form.quantized(jnp.asarray(rng.normal(size=(4, 128, 256)), jnp.float32))
    blocks = jnp.asarray([1, 0, 2, 1], jnp.int32)
    table = jnp.asarray(rng.integers(0, 300, 512), jnp.int32)
    gathered = dataclasses.replace(tokens, codes=tokens.codes[table], scales=tokens.scales[table])
    named = generation.chosen(**{choice: "mosaic"})
    with generation.assumed("sm100"), named, generation.routes() as routes:
        got = jax.jit(lambda: product(tokens, matrices, blocks, out_dtype=jnp.float32, interpret_on_cpu=True,
                                      gather=table))()
        expected = jax.jit(lambda: product(gathered, matrices, blocks, out_dtype=jnp.float32,
                                           interpret_on_cpu=True))()
    assert {route.kernel for route in routes if route.choice == choice} == {"mosaic"}, routes
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_a_gathered_products_table_covers_every_tile_its_schedule_walks():
    """A gathered product's table, on a family whose loads past the rows are bounded (wgmma's TMA, compiled),
    still extends to every row its schedule's tiles walk (`row_tiles.row_extent`): the producer's threads
    read a tile's entries by plain loads, which no TMA bounds; read past it, a GH200 faulted
    (CUDA_ERROR_ILLEGAL_ADDRESS, FS-4346-base-sm90)."""
    tiles = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family='wgmma')
    table = jnp.zeros((256,), jnp.int32)
    tokens = jnp.zeros((200, 128), jnp.bfloat16)
    _, gathered, _, m, extent = mosaic_gpu._walked_rows(tokens, table, None, tiles, transpose_lhs=False,
                                                        dense=False, bounded=True)
    assert gathered.shape[0] == extent == row_tiles.row_extent(m, tiles.block_m) > m


def test_a_grouped_products_depth_split_sums_as_its_walk_apart():
    """A grouped product's depth split (row 3622's lever: a decode step's experts' one wave of tiles, each
    run a share of the depth, the fp32 partials summed): over small integers, bitwise the unsplit product,
    its groups ragged with one empty and rows past them zero, interpreted as an H100's."""
    rng = np.random.default_rng(31)
    tokens = jnp.asarray(rng.integers(-3, 4, size=(40, 512)), jnp.bfloat16)
    matrices = jnp.asarray(rng.integers(-3, 4, size=(4, 512, 256)), jnp.bfloat16)
    sizes = jnp.asarray([10, 0, 17, 9], jnp.int32)
    tiles = tiling.Tiles(64, 128, 64, 2, 1, warpgroups=1, family='wgmma')

    def product(splits):
        return mosaic_gpu.gmm(tokens, matrices, sizes, tiles=dataclasses.replace(tiles, splits=splits),
                              transpose_rhs=False, out_dtype=jnp.float32, multiprocessors=4,
                              interpret=grouped_launch.INTERPRETED)

    with generation.assumed("sm90"):
        np.testing.assert_array_equal(np.asarray(product(4)), np.asarray(product(1)))


def test_a_hoppers_moe_decode_splits_its_experts_depth():
    """A MoE decode step on an H100 (Qwen3-30B-A3B's down projection: 768 deep, 2048 wide, row 3622):
    unpinned `derived` returns legal tiles across decode and prefill row counts, and `pricing.split_count`
    splits underfilled expert walks while keeping a full-wave prefill unsplit."""
    from dew.nn.kernels.templates.gemm import pricing

    launch = grouped_matmul.MosaicGmm()

    def derived(rows):
        bound = launch.bind(jax.ShapeDtypeStruct((rows, 768), jnp.bfloat16),
                            jax.ShapeDtypeStruct((128, 768, 2048), jnp.bfloat16),
                            jax.ShapeDtypeStruct((128,), jnp.int32), out_dtype="bfloat16", trans_rhs=False,
                            interpret=False)
        target = generation.gpu_target()
        tile = launch.derived(bound, target)
        assert tile is not None and launch.legal(bound, tile, target), tile
        return bound, tile, target

    with generation.assumed("sm90"):
        b4, t4, sm90 = derived(4)
        _, _, _ = derived(8)
        _, t_pre, _ = derived(8192)
        one_prog = tiling.Tiles(64, 256, 64, 4, 1, warpgroups=1, programs=1, family="wgmma")
        assert pricing.split_count(one_prog, **launch._contraction(b4, one_prog, sm90)) > 1
        assert t_pre.splits == 1 and t4.splits >= 1


def test_hopper_decode_derives_two_resident_programs_per_sm_and_compiles():
    """Hopper 1-consumer decode (`rows < 64`, DeepSeek-V3 / Qwen3-30B-A3B 1-token MoE down & dense M=16,
    rows 3622, 483): unpinned public `derived` returns legal tiles across decode, dense and prefill calls,
    both explicit 1- and 2-resident schedules are legal and price cleanly via `walk_seconds`, and explicit
    1-/2-resident Hopper decode schedules compile spill-free on `sm90`."""
    import functools

    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    from dew.nn.kernels.templates.gemm import pricing

    launch = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    i32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.int32)
    with generation.assumed("sm90"):
        sm90 = generation.gpu_target("sm90")
        moe_ba = launch.bind(bf16((8, 2048)), bf16((256, 2048, 7168)), i32((256,)),
                             out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=False)
        qwen_ba = launch.bind(bf16((8, 768)), bf16((128, 768, 2048)), i32((128,)),
                              out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=False)
        dense_ba = launch.bind(bf16((16, 4096)), bf16((1, 4096, 14336)), i32((1,)),
                               out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)
        prefill_ba = launch.bind(bf16((2048, 4096)), bf16((1, 4096, 14336)), i32((1,)),
                                 out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)
        for ba in (moe_ba, qwen_ba, dense_ba, prefill_ba):
            tile = launch.derived(ba, sm90)
            assert tile is not None and launch.legal(ba, tile, sm90), tile
        # Explicit 1-resident and 2-resident schedules both legal and priced by walk_seconds.
        one_res = tiling.Tiles(64, 256, 64, 4, 1, warpgroups=1, programs=1, family="wgmma")
        two_res = tiling.Tiles(64, 128, 64, 3, 1, warpgroups=1, programs=2, family="wgmma")
        for ba in (moe_ba, qwen_ba, dense_ba):
            for sched in (one_res, two_res):
                assert launch.legal(ba, sched, sm90), sched
                assert np.isfinite(pricing.walk_seconds(sched, launch._contraction(ba, sched, sm90)))
        if generation.compile_only_gpus("sm90"):
            cases = [
                KernelCase("sm90_decode_moe_1prog", lambda: launch.lowering(moe_ba, one_res)),
                KernelCase("sm90_decode_moe_2prog", lambda: launch.lowering(moe_ba, two_res)),
                KernelCase("sm90_decode_qwen_2prog", lambda: launch.lowering(qwen_ba, two_res)),
                KernelCase("sm90_decode_dense_2prog", lambda: launch.lowering(dense_ba, two_res)),
            ]
            records = compile_mosaic_kernels(cases, "sm90")
            for rec in records:
                assert not rec.get("error"), rec
                assert rec["kernels"], rec
                assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


def test_a_kernel_gradient_no_launch_reads_through_its_tables_runs_a_launch_on_its_gathered_rows():
    """fp32 rows and cotangent read through tables (an fp32 mixture of experts' gathered tokens and scattered
    slots), which no launch reads where they lie (Mosaic GPU's gathering producer takes 16-bit rows): on an
    H100 the kernel gradient gathers them first and a grouped launch takes the product, as the forward's
    does, never XLA's ragged dot over every group's matrix, whose footprint an MoE step's derived plan bounds
    from below alone (row 5213); interpreted, bitwise the kernel gradient of the gathered rows."""
    rng = np.random.default_rng(5)
    tokens = jnp.asarray(rng.standard_normal((200, 128)), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal((256, 64)), jnp.float32)
    table = jnp.asarray(rng.integers(0, 200, 256), jnp.int32)
    slots = jnp.asarray(rng.permutation(256), jnp.int32)
    sizes = jnp.asarray([100, 0, 156, 0], jnp.int32)
    gradient = {"out_dtype": jnp.float32, "precision": "default", "interpret_on_cpu": True}
    with generation.assumed("sm90"):
        def tabled():
            return gemm.gemm_t(tokens, cotangent, sizes, gather=(table, slots), **gradient)

        graph = jax.make_jaxpr(tabled)()
        got = jax.jit(tabled)()
        expected = jax.jit(lambda: gemm.gemm_t(tokens[table], cotangent[slots], sizes, **gradient))()
    names = {equation.primitive.name for equation in equations(graph, platform="cuda")}
    assert "ragged_dot_general" not in names, names
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("named", [None, "xla"])
def test_a_block_scaled_product_no_tile_holds_takes_the_decoded_product(named):
    """A block-scaled product whose block scales no tile holds (MXFP8 rows `[128, 32]`: one scale a row,
    short of the tiles' four) is selected before its scales are laid out for the template
    (`mosaic_sm100.scales_tileable`): on a CPU as an sm100, or named XLA's, it runs the decoded operands'
    product, on par with the MXFP8 values' product in fp32 (check 1), from float64, where laying its scales
    out first failed to reshape."""
    blocks = jnp.asarray([1], jnp.int32)
    sizes = 128 * blocks

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(128, 32)), jnp.float32),
                jnp.asarray(rng.normal(size=(1, 64, 32)), jnp.float32))

    def blocked(x, w):
        chosen = generation.chosen(mxfp8=named) if named else generation.chosen()
        with generation.assumed("sm100"), chosen:
            return gemm.mxfp8_product(formats.MxFp8.quantized(x), formats.MxFp8.quantized(w), blocks,
                                      out_dtype=jnp.float32, interpret_on_cpu=True)

    assert_as_exact(blocked, jax.jit(mx_matrix.values_product(mx_matrix.mx_values, sizes)), draw,
                    "an MXFP8 product no tile holds", seeds=SEEDS,
                    truth_fn=mx_matrix.values_product(mx_matrix.mx_values, sizes, jnp.float64))


def test_a_stored_nvfp4_matrix_no_tile_holds_takes_the_decoded_product():
    """`nvfp4_quantized_product` of a stored NVFP4 matrix `[256, 64]` (64 columns, short of a tile's 128)
    beside rows quantized as they are used, on a CPU as an sm100: bitwise the NVFP4 block product of its
    rows, padded, and the matrix as one group in XLA, where laying its scales out for the template first
    failed to reshape."""
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(5, 256)), jnp.float32)
    matrix = formats.NvFp4.quantized(jnp.asarray(rng.normal(size=(256, 64)), jnp.float32))
    with generation.assumed("sm100"):
        out = gemm.nvfp4_quantized_product(x, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)
    expected = gemm.nvfp4_block_product(formats.NvFp4Block.quantized(jnp.pad(x, ((0, 123), (0, 0)))),
                                        formats.NvFp4Block.of(matrix), jnp.asarray([1], jnp.int32),
                                        out_dtype=jnp.float32)[:5]
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


@pytest.mark.parametrize("generation_name", ["sm90", "sm100"])
def test_a_gathered_product_with_a_prologue_holds_check_1_with_its_gradients(generation_name):
    """A grouped product of gathered rows through a prologue (row 5391: an expert projection over a scaled
    token stream, the scale a row of the product's, in the table's order), on a CPU as an H100 and a B200,
    interpreted: its output and its tokens', scales' and matrices' gradients on par with the plain formula's
    in float32 from float64 truth (check 1), where `gemm` refused the combination."""
    sizes = jnp.asarray([70, 0, 129, 57], jnp.int32)
    rows = int(sizes.sum())

    def draw(seed):
        rng = np.random.default_rng(seed)
        table = jnp.asarray(rng.integers(0, 200, rows), jnp.int32)
        return (jnp.asarray(rng.normal(size=(200, 128)), jnp.bfloat16), table,
                jnp.asarray(rng.uniform(0.5, 2.0, size=(rows,)), jnp.float32),
                jnp.asarray(rng.normal(size=(4, 128, 256)) / 16, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(rows, 256)), jnp.float32))

    def scaled(x, scale):
        return x * scale[:, None]

    def templated(tokens, table, scale, matrices):
        return gemm.gemm(tokens, matrices, sizes, prologue=scaled, prologue_operands=((scale, ("rows",)),),
                         gather=table, out_dtype=jnp.float32, interpret_on_cpu=True)

    def plain(tokens, table, scale, matrices, wide=jnp.float32):
        x = scaled(tokens[table].astype(wide), scale.astype(wide)).astype(
            wide if wide == jnp.float64 else tokens.dtype).astype(wide)
        group = jnp.repeat(jnp.arange(4), sizes, total_repeat_length=rows)
        return jnp.einsum("rk,rkn->rn", x, matrices.astype(wide)[group])

    def run(fn):
        def call(tokens, table, scale, matrices, cotangent):
            out, pull = jax.vjp(lambda t, s, m: fn(t, table, s, m), tokens, scale, matrices)
            return out, *pull(cotangent.astype(out.dtype))
        return call

    def emulated(*arguments):
        # The forward and its reverse traced as the generation (`interpret_on_cpu`'s emulated device).
        with generation.assumed(generation_name):
            return run(templated)(*arguments)

    assert_as_exact(jax.jit(emulated), jax.jit(run(plain)), draw,
                    "a gathered product with a prologue and its gradients", seeds=SEEDS,
                    truth_fn=run(functools.partial(plain, wide=jnp.float64)))


_LAYER_SIZES = (70, 0, 129, 57)
"""The gathered stacked layer's groups (`_gathered_layer`): ragged, one empty."""


def _normed(x, inv_rms, weight):
    return x * inv_rms[:, None] * weight[None, :]


def _swiglu(gate, up):
    return jax.nn.silu(gate) * up


def _gathered_layer(tokens, sizes, table, inv_rms, weight, gate_w, up_w, **options):
    """The second stacked layer's gate|up over `tokens`' rows at `table`, normed by their inverse RMS and the
    norm's weight in the prologue and SwiGLU in the epilogue (`gemm.gemm`, row 5391)."""
    return gemm.gemm(tokens, gate_w, sizes, paired=up_w, layer=jnp.int32(1), gather=table, prologue=_normed,
                     prologue_operands=((inv_rms, ("rows",)), (weight, ("depth",))), epilogue=_swiglu,
                     out_dtype=jnp.bfloat16, **options)


@pytest.mark.check1_shards(seeds=8, per_task=4)
@pytest.mark.parametrize("gpu", ["sm90", "sm100"])
def test_gathered_prologued_paired_stacked_layer_runs_fused(gpu: str, request):
    """A gathered, prologued, paired, stacked-layer grouped GEMM (`#5391`: scanned MoE gate|up over a
    normalized/scaled token stream) runs inside the one Mosaic mainloop on Hopper `sm90` and Blackwell
    `sm100`, interpreted, and holds check 1 forward + VJP. Its draws run in map tasks of 4: with the compile
    below in one test they took 126 s on sm100, past the group target (row 12639)."""
    sizes = jnp.asarray(_LAYER_SIZES, jnp.int32)
    rows = sum(_LAYER_SIZES)

    def draw(seed):
        rng = np.random.default_rng(5391 + seed)
        table = jnp.asarray(rng.integers(0, 192, rows), jnp.int32)
        return (jnp.asarray(rng.normal(size=(192, 128)) / 8, jnp.bfloat16), table,
                jnp.asarray(rng.uniform(0.5, 1.5, size=(rows,)), jnp.float32),
                jnp.asarray(rng.uniform(0.7, 1.3, size=(128,)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(2, 4, 128, 256)) / 16, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(2, 4, 128, 256)) / 16, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(rows, 256)) / 8, jnp.bfloat16))

    def run_dew(tokens, table, inv_rms, weight, gate_w, up_w, cotangent):
        def fn(t, r, w, gw, uw):
            return _gathered_layer(t, sizes, table, r, w, gw, uw, interpret_on_cpu=True)

        with generation.assumed(gpu):
            out, pull = jax.vjp(fn, tokens, inv_rms, weight, gate_w, up_w)
            return out, *pull(cotangent)

    def run_plain(tokens, table, inv_rms, weight, gate_w, up_w, cotangent, *, wide=jnp.float32):
        def fn(t, r, w, gw, uw):
            x = _normed(t[table].astype(wide), r.astype(wide), w.astype(wide)).astype(
                wide if wide == jnp.float64 else t.dtype).astype(wide)
            grp = jnp.repeat(jnp.arange(4), sizes, total_repeat_length=rows)
            g = jnp.einsum("rk,rkn->rn", x, gw[1].astype(wide)[grp])
            u = jnp.einsum("rk,rkn->rn", x, uw[1].astype(wide)[grp])
            out = _swiglu(g, u)
            return out if wide == jnp.float64 else out.astype(jnp.bfloat16)
        out, pull = jax.vjp(fn, tokens, inv_rms, weight, gate_w, up_w)
        return out, *pull(cotangent.astype(out.dtype))

    t0, tbl0, r0, w0, gw0, uw0, _ = draw(0)
    with generation.assumed(gpu), generation.chosen(grouped_matmul="mosaic"), generation.routes() as seen:
        _gathered_layer(t0, sizes, tbl0, r0, w0, gw0, uw0, interpret_on_cpu=True)
    assert any(r.kernel == "mosaic" for r in seen), seen
    assert_as_exact(jax.jit(run_dew), jax.jit(run_plain), draw, "gathered+prologue+paired+layer",
                    seeds=marked_seeds(request), truth_fn=functools.partial(run_plain, wide=jnp.float64))


@pytest.mark.parametrize("gpu", ["sm90", "sm100"])
def test_gathered_prologued_paired_stacked_layer_compiles_spill_free_without_a_gather_copy(gpu: str):
    """`test_gathered_prologued_paired_stacked_layer_runs_fused`'s layer for a compile-only `gpu`: its CUDA
    program gathers no tokens before the Mosaic mainloop, and its kernels compile spill-free (row 5391)."""
    from fla_gdn_support import KernelCase
    from jax.core import ShapedArray
    from kernel_compile import compile_mosaic_kernels

    if not generation.compile_only_gpus(gpu):
        pytest.skip(f"no compile-only {gpu} in this environment")
    rows = sum(_LAYER_SIZES)
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    i32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.int32)
    specs = (bf16((192, 128)), i32((rows,)), f32((rows,)), bf16((128,)),
             bf16((2, 4, 128, 256)), bf16((2, 4, 128, 256)), i32((4,)))

    def compiled(t, tbl, r, w, gw, uw, sz):
        with generation.chosen(grouped_matmul="mosaic"):
            return _gathered_layer(t, sz, tbl, r, w, gw, uw)

    with generation.assumed(gpu):
        graph = jax.make_jaxpr(compiled)(*specs)
        assert not any(
            eqn.primitive.name == "gather"
            and any(isinstance(v.aval, ShapedArray) and v.aval.shape == (192, 128) for v in eqn.invars)
            for eqn in equations(graph, platform="cuda")
        ), "tokens pre-gathered in CUDA path before Mosaic mainloop"
        records = compile_mosaic_kernels(
            [KernelCase(f"gathered_prologue_paired_layer_{gpu}", lambda: (compiled, specs))], gpu)
    for rec in records:
        assert not rec.get("error"), rec
        assert rec["kernels"], rec
        assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


def test_gathered_prologued_negative_indices_and_sm80_forced_mosaic():
    """Signed gather indices (repeated `-1` and negative offsets) match `lhs[gather]` forward and VJP
    without negative native row loads, and forced `grouped_matmul='mosaic'` on `sm80` (where `mma.sync`
    cannot consume tables) gathers first instead of raising `KernelRefused` (row 5391)."""
    sizes = jnp.asarray([16, 0, 24, 24], jnp.int32)
    rows = int(sizes.sum())
    rng = np.random.default_rng(53910)
    tokens = jnp.asarray(rng.normal(size=(32, 64)) / 8, jnp.bfloat16)
    scale = jnp.asarray(rng.uniform(0.5, 1.5, size=(rows,)), jnp.float32)
    matrices = jnp.asarray(rng.normal(size=(4, 64, 64)) / 8, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(rows, 64)) / 8, jnp.float32)
    base = rng.integers(0, 32, rows, dtype=np.int32)
    base[:10] = -1
    base[10:14] = -32
    base[14:18] = -5
    table = jnp.asarray(base, jnp.int32)

    def scaled(x, s):
        return x * s[:, None]

    def ref_run(t, s, m):
        x = scaled(t[table].astype(jnp.float32), s).astype(t.dtype).astype(jnp.float32)
        grp = jnp.repeat(jnp.arange(4), sizes, total_repeat_length=rows)
        return jnp.einsum("rk,rkn->rn", x, m.astype(jnp.float32)[grp])

    def mosaic_run(gpu: str, t, s, m):
        with generation.assumed(gpu), generation.chosen(grouped_matmul="mosaic"), generation.routes() as seen:
            out, pull = jax.vjp(
                lambda a, b, c: gemm.gemm(a, c, sizes, prologue=scaled, prologue_operands=((b, ("rows",)),),
                                          gather=table, out_dtype=jnp.float32, interpret_on_cpu=True),
                t, s, m)
            grads = pull(cotangent)
        assert any(r.kernel == "mosaic" for r in seen), (gpu, seen)
        return (out, *grads)

    def draw(seed):
        r = np.random.default_rng(53911 + seed)
        return (jnp.asarray(r.normal(size=(32, 64)) / 8, jnp.bfloat16),
                jnp.asarray(r.uniform(0.5, 1.5, size=(rows,)), jnp.float32),
                jnp.asarray(r.normal(size=(4, 64, 64)) / 8, jnp.bfloat16),
                jnp.asarray(r.normal(size=(rows, 64)) / 8, jnp.float32))

    def truth(t, s, m, dy):
        def fn(a, b, c):
            x = scaled(a[table].astype(jnp.float64), b.astype(jnp.float64))
            grp = jnp.repeat(jnp.arange(4), sizes, total_repeat_length=rows)
            return jnp.einsum("rk,rkn->rn", x, c.astype(jnp.float64)[grp])
        out, pull = jax.vjp(fn, t, s, m)
        return (out, *pull(dy.astype(jnp.float64)))

    def plain_vjp(t, s, m, dy):
        out, pull = jax.vjp(ref_run, t, s, m)
        return (out, *pull(dy))

    for gpu in ("sm90", "sm80"):
        def dew_vjp(t, s, m, dy, gpu=gpu):
            with generation.assumed(gpu), generation.chosen(grouped_matmul="mosaic"):
                out, pull = jax.vjp(
                    lambda a, b, c: gemm.gemm(a, c, sizes, prologue=scaled,
                                              prologue_operands=((b, ("rows",)),), gather=table,
                                              out_dtype=jnp.float32, interpret_on_cpu=True),
                    t, s, m)
                return (out, *pull(dy))

        mosaic_run(gpu, tokens, scale, matrices)
        assert_as_exact(jax.jit(dew_vjp), jax.jit(plain_vjp), draw,
                        f"negative gather+prologue {gpu}", seeds=SEEDS, truth_fn=truth)

    # Paired + scatter under forced Mosaic on sm80 (`mma.sync` has `gathers_alone=False`): `'tables'`
    # hands off before `'launch'` so forward AND reverse (`_keeps_launch`) both run Mosaic without raising
    # `KernelRefused` and hold check 1 against float64 truth (rows 12305, 12307).
    perm = jnp.asarray(rng.permutation(rows), jnp.int32)

    def draw_pair(seed):
        r = np.random.default_rng(12305 + seed)
        return (jnp.asarray(r.normal(size=(rows, 64)) / 8, jnp.bfloat16),
                jnp.asarray(r.normal(size=(4, 64, 64)) / 8, jnp.bfloat16),
                jnp.asarray(r.normal(size=(4, 64, 64)) / 8, jnp.bfloat16),
                (jnp.asarray(r.normal(size=(rows, 64)) / 8, jnp.bfloat16),
                 jnp.asarray(r.normal(size=(rows, 64)) / 8, jnp.bfloat16)))

    def pair_ref(t, gw, uw, *, dtype=jnp.float32):
        def one(w):
            y = jax.lax.ragged_dot(t, w, sizes, preferred_element_type=dtype).astype(
                jnp.bfloat16 if dtype == jnp.float32 else dtype)
            return jnp.empty_like(y).at[perm].set(y)

        return one(gw), one(uw)

    def pair_truth(t, gw, uw, dy):
        outs, pull = jax.vjp(lambda a, b, c: pair_ref(a, b, c, dtype=jnp.float64),
                             t.astype(jnp.float64), gw.astype(jnp.float64), uw.astype(jnp.float64))
        return outs, pull(tuple(d.astype(jnp.float64) for d in dy))

    def pair_plain_vjp(t, gw, uw, dy):
        outs, pull = jax.vjp(pair_ref, t, gw, uw)
        return outs, pull(dy)

    def pair_dew_vjp(t, gw, uw, dy):
        with generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"):
            outs, pull = jax.vjp(
                lambda a, b, c: gemm.gemm(a, b, sizes, paired=c, scatter=perm, out_dtype=jnp.bfloat16,
                                          out_dtypes=(jnp.bfloat16, jnp.bfloat16), interpret_on_cpu=True),
                t, gw, uw)
            return outs, pull(dy)

    t0, gw0, uw0, dy0 = draw_pair(0)
    with generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"), generation.routes() as seen:
        pair_dew_vjp(t0, gw0, uw0, dy0)
    assert any(r.kernel == "mosaic" for r in seen), seen
    assert_as_exact(jax.jit(pair_dew_vjp), jax.jit(pair_plain_vjp), draw_pair,
                    "sm80 forced-mosaic paired+scatter", seeds=SEEDS, truth_fn=pair_truth)


def test_producers_rungs_keeps_top_after_stated_cap_start():
    """When `producers.started` picks a `stated` cap below `top` on a tensor-memory family, `producers.rungs`
    passes that same `stated` cap into `Producers.stepped` so the first spill rung visits `top` instead of
    skipping it (Train 1231 F2)."""
    from dew.nn.kernels.templates import report, resources
    from dew.nn.kernels.templates.gemm import producers

    @dataclasses.dataclass(frozen=True)
    class _HighRegGmm(grouped_matmul.MosaicGmm):
        need: int = 160

        def registers(self, ba, config, target=None) -> int:
            del ba, config, target
            return self.need

    op = _HighRegGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    ba = op.bind(bf16((128, 64)), bf16((2, 64, 128)), jax.ShapeDtypeStruct((2,), jnp.int32),
                 (), (), None, None, jax.ShapeDtypeStruct((128,), jnp.int32),
                 out_dtype="bfloat16", trans_rhs=False, interpret=False)
    tile = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=1, cluster=1, family="tcgen05", programs=2)
    with generation.assumed("sm100"), report.refusals():
        sm100 = generation.gpu_target()
        top = producers._top(op, ba, tile, sm100)
        assert top is not None and op.need > top
        started = producers.started(op, ba, tile, sm100)
        start_cap = started.producers.of(op.KERNEL)
        assert start_cap is not None and start_cap < top
        report.refuse([(report._identity(op, ba, sm100), started, (op.KERNEL,))])
        ladder = [rung.producers.of(op.KERNEL) for rung in producers.rungs(op, ba, started, sm100)]
        assert ladder[0] == resources.facts().raises().below(top), (start_cap, top, ladder)
        assert start_cap not in ladder, (start_cap, ladder)


def test_a_refused_gathered_tile_still_offers_its_ladders_next_rung():
    """A gathered paired product on an H100 (an MoE gate/up, 512 tokens gathered to 1024 rows over two
    experts, its gated epilogue's three outputs) whose tile a compile refused spilling, and the first rungs of
    its producer's ladder too (`producers.rungs`): the launch still offers a config, the ladder's next rung,
    which its launch takes (train 1286, row 12972: every rung's modelled registers past the consumers'
    budget, the rung certification held refused when the call took it, and the call Exhausted)."""
    from dew.nn.activations import gated_hook, rounded_gated
    from dew.nn.kernels.templates import report
    from dew.nn.kernels.templates.gemm import producers

    bf16 = jnp.bfloat16
    sizes = jnp.asarray([512, 512], jnp.int32)
    table = jnp.arange(1024, dtype=jnp.int32) % 512

    def paired(tokens, first, second):
        return gemm.gemm(tokens, first, sizes, compute=bf16, out_dtype=bf16, out_dtypes=(bf16,) * 3,
                         paired=second, gather=table, epilogue=rounded_gated(gated_hook("swiglu"), bf16))

    shapes = (jax.ShapeDtypeStruct((512, 512), bf16), jax.ShapeDtypeStruct((2, 512, 512), bf16),
              jax.ShapeDtypeStruct((2, 512, 512), bf16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"), tuning.seen() as launches:
        jax.make_jaxpr(paired)(*shapes)
    [ba] = [bound for _, bound in launches.taken if bound.arguments.get("paired") is not None
            and not bound.arguments.get("trans_rhs")]
    op = ba.op
    with generation.assumed("sm90"), report.refusals():
        target = generation.gpu_target()
        tile = op.derived(ba, target)
        identity = report._identity(op, ba, target)
        report.refuse([(identity, tile, frozenset({op.KERNEL}))])
        rungs = list(producers.rungs(op, ba, tile, target))[:3]
        assert rungs, "a gathered tile's producer steps"
        report.refuse([(identity, rung, frozenset({op.KERNEL})) for rung in rungs])
        refused = [tile, *rungs]
        offered = op.remaining(ba)
        assert offered is not None and offered not in refused and op.legal(ba, offered, target), offered


@pytest.mark.parametrize("in_dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "fp32"])
def test_gathered_prologued_paired_layered_with_compute_holds_check_1_and_compiles(in_dtype):
    """MoE call shape (`compute=bf16` + `prologue` + `gather` + `paired` + `layer`, rows 5391, 12306):
    holds check 1 forward and full VJP against float64 truth for both bf16 and fp32 input tokens, and
    compiles spill-free on `sm90` Mosaic without pre-gathering tokens."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    rng = np.random.default_rng(12306)
    tokens_n, rows, depth, width, layers, groups = 48, 64, 64, 64, 2, 4
    sizes = jnp.asarray([16, 0, 24, 24], jnp.int32)
    table = jnp.asarray(rng.integers(0, tokens_n, size=(rows,)), jnp.int32)
    layer_idx = jnp.asarray(1, jnp.int32)

    def scaled(x, r_scale, d_weight):
        return (x.astype(jnp.float32) * r_scale[:, None] * d_weight[None, :]).astype(x.dtype)

    def draw(seed):
        r = np.random.default_rng(123060 + seed)
        return (jnp.asarray(r.normal(size=(tokens_n, depth)) / 8, in_dtype),
                jnp.asarray(r.normal(size=(rows,)) / 2 + 1.0, jnp.float32),
                jnp.asarray(r.normal(size=(depth,)) / 2 + 1.0, jnp.float32),
                jnp.asarray(r.normal(size=(layers, groups, depth, width)) / 8, jnp.bfloat16),
                jnp.asarray(r.normal(size=(layers, groups, depth, width)) / 8, jnp.bfloat16),
                (jnp.asarray(r.normal(size=(rows, width)) / 8, jnp.bfloat16),
                 jnp.asarray(r.normal(size=(rows, width)) / 8, jnp.bfloat16)))

    def ref_fwd(t, rs, dw, s1, s2, *, dtype=jnp.float32):
        out_d = jnp.bfloat16 if dtype == jnp.float32 else dtype
        x = (t[table].astype(out_d).astype(dtype) * rs.astype(dtype)[:, None]
             * dw.astype(dtype)[None, :]).astype(out_d)
        w1, w2 = s1[layer_idx].astype(out_d), s2[layer_idx].astype(out_d)
        return (jax.lax.ragged_dot(x, w1, sizes, preferred_element_type=dtype).astype(out_d),
                jax.lax.ragged_dot(x, w2, sizes, preferred_element_type=dtype).astype(out_d))

    def truth(t, rs, dw, s1, s2, dy):
        outs, pull = jax.vjp(lambda a, b, c, d, e: ref_fwd(a, b, c, d, e, dtype=jnp.float64),
                             t.astype(jnp.float64), rs.astype(jnp.float64), dw.astype(jnp.float64),
                             s1.astype(jnp.float64), s2.astype(jnp.float64))
        return outs, pull(tuple(g.astype(jnp.float64) for g in dy))

    def plain_vjp(t, rs, dw, s1, s2, dy):
        outs, pull = jax.vjp(ref_fwd, t, rs, dw, s1, s2)
        return outs, pull(dy)

    def dew_vjp(t, rs, dw, s1, s2, dy):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
            outs, pull = jax.vjp(
                lambda a, b, c, d, e: gemm.gemm(
                    a, d, sizes, prologue=scaled,
                    prologue_operands=((b, ("rows",)), (c, ("depth",))),
                    compute=jnp.bfloat16, out_dtype=jnp.bfloat16,
                    out_dtypes=(jnp.bfloat16, jnp.bfloat16), paired=e, layer=layer_idx,
                    gather=table, interpret_on_cpu=True),
                t, rs, dw, s1, s2)
            return outs, pull(dy)

    assert_as_exact(jax.jit(dew_vjp), jax.jit(plain_vjp), draw,
                    f"moe gather+prologue+paired+layer+compute {in_dtype}", seeds=SEEDS, truth_fn=truth)

    if in_dtype == jnp.bfloat16 and generation.compile_only_gpus("sm90"):
        specs = (jax.ShapeDtypeStruct((tokens_n, depth), jnp.bfloat16),
                 jax.ShapeDtypeStruct((rows,), jnp.float32),
                 jax.ShapeDtypeStruct((depth,), jnp.float32),
                 jax.ShapeDtypeStruct((layers, groups, depth, width), jnp.bfloat16),
                 jax.ShapeDtypeStruct((layers, groups, depth, width), jnp.bfloat16),
                 jax.ShapeDtypeStruct((groups,), jnp.int32),
                 jax.ShapeDtypeStruct((rows,), jnp.int32),
                 jax.ShapeDtypeStruct((), jnp.int32))

        def compiled(t, rs, dw, s1, s2, sz, tbl, lyr):
            with generation.chosen(grouped_matmul="mosaic"):
                return gemm.gemm(t, s1, sz, prologue=scaled,
                                 prologue_operands=((rs, ("rows",)), (dw, ("depth",))),
                                 compute=jnp.bfloat16, out_dtype=jnp.bfloat16,
                                 out_dtypes=(jnp.bfloat16, jnp.bfloat16), paired=s2, layer=lyr,
                                 gather=tbl)

        with generation.assumed("sm90"):
            records = compile_mosaic_kernels(
                [KernelCase("moe_gather_prologue_paired_layer_sm90", lambda: (compiled, specs))], "sm90")
        for rec in records:
            assert not rec.get("error"), rec
            assert rec["kernels"], rec
            assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec
