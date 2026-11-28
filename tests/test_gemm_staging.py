"""A GEMM tile's stated output staging (`tiling.Tiles.staging`, row 5634): the chunks and buffers a launch
stages its output in, admitted where they hold the issuer's atoms and whole TMA stores in the output's
dtypes, narrowed with a chained replay, refused by the launches that stage at their family's own."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS, lowers
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates import report
from dew.nn.kernels.templates.gemm import pricing, tiling


@INTERPRETS
def test_sm120_output_staging_admits_three_stage_128x256_and_compiles():
    """Explicit `Tiles.staging` `(chunk, buffers)` lowers output SMEM from 32 KB to 16 KB on `sm120`
    `mma.sync.fed`, admitting 3-stage `128x256x32` arms (`(64, 1)` and `(32, 2)`) where default `(64, 2)`
    exceeds the 99 KB block, holding check 1, compiling spill-free on `sm120`, and refusing invalid or
    non-chunked staging (row 5634, M-FS-62)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    with pytest.raises(ValueError):
        tiling.Tiles(64, 128, 32, 3, 1, warpgroups=1, family="mma.sync", staging=(64, 1))
    with pytest.raises(ValueError):
        tiling.Tiles(128, 256, 32, 3, 1, warpgroups=2, family="mma.sync.fed", staging=(0, 1))
    with pytest.raises(ValueError):
        tiling.Tiles(128, 128, 64, 2, 1, warpgroups=1, family="tcgen05", staging=(128, 1))

    gmm = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    one = jax.ShapeDtypeStruct((1,), jnp.int32)
    base = tiling.Tiles(128, 256, 32, 3, 1, warpgroups=2, family="mma.sync.fed")
    arm_64x1 = dataclasses.replace(base, staging=(64, 1))
    arm_32x2 = dataclasses.replace(base, staging=(32, 2))
    with generation.assumed("sm120"):
        sm120 = generation.gpu_target()
        ba = gmm.bind(bf16((256, 128)), bf16((1, 128, 256)), one, out_dtype="bfloat16",
                      trans_rhs=False, interpret=False, dense=True)
        assert not gmm.legal(ba, base, sm120)
        assert gmm.legal(ba, arm_64x1, sm120)
        assert gmm.legal(ba, arm_32x2, sm120)
        assert gmm.staged_bytes(ba, base, sm120) - gmm.staged_bytes(ba, arm_64x1, sm120) == 16384
        assert gmm.staged_bytes(ba, arm_64x1, sm120) == gmm.staged_bytes(ba, arm_32x2, sm120)
        # One warpgroup's 128 x 128 tile splits its warps two along its rows: 16-column atoms, which an
        # 8-column chunk slices mid-way, refused by the launch and its kernel.
        single = tiling.Tiles(128, 128, 32, 2, 1, warpgroups=1, family="mma.sync.fed")
        square = gmm.bind(bf16((256, 128)), bf16((1, 128, 128)), one, out_dtype="bfloat16", trans_rhs=False,
                          interpret=False, dense=True)
        assert not gmm.legal(square, dataclasses.replace(single, staging=(8, 1)), sm120)
        # Four bf16 columns are 8 bytes a chunk's row: no whole 16-byte TMA store, which the launch refuses
        # and its kernel too where called at that pin; four fp32 columns are one store but half an
        # accumulator fragment, and sixteen 4-bit floats 64 bits. A 64-row share of 256 columns specializes
        # mma.sync's warps along its columns, 32-column atoms: 32 columns of fp32 or bf16 are whole.
        narrow = dataclasses.replace(base, staging=(4, 1))
        assert not gmm.legal(ba, narrow, sm120)
        family = pricing.family_of(base, jnp.bfloat16, sm120)
        for chunk, dtype in ((4, jnp.bfloat16), (4, jnp.float32), (16, jnp.float4_e2m1fn), (16, jnp.float32)):
            with pytest.raises(ValueError):
                dataclasses.replace(base, staging=(chunk, 1)).stated((((64, 256), dtype),), family)
        for dtype in (jnp.bfloat16, jnp.float32):
            assert dataclasses.replace(base, staging=(32, 1)).stated((((64, 256), dtype),), family) == (32, 1)
        atoms = pricing.family_of(single, jnp.bfloat16, sm120)
        with pytest.raises(ValueError):
            dataclasses.replace(single, staging=(8, 1)).stated((((128, 128), jnp.bfloat16),), atoms)
        whole = dataclasses.replace(single, staging=(16, 1))
        assert whole.stated((((128, 128), jnp.bfloat16),), atoms) == (16, 1)

    def ref_fn(x, w, *, dtype=jnp.float32):
        return jax.lax.dot(x.astype(dtype), w.astype(dtype), preferred_element_type=dtype).astype(
            jnp.float64 if dtype == jnp.float64 else jnp.bfloat16)

    for tag, arm in (("64x1", arm_64x1), ("32x2", arm_32x2)):
        def draw(seed, tag=tag):
            rng = np.random.default_rng(5634 + (100 if tag == "32x2" else 0) + seed)
            x = jnp.asarray(rng.standard_normal((128, 64)) / np.sqrt(64), jnp.bfloat16)
            w = jnp.asarray(rng.standard_normal((64, 256)) / np.sqrt(64), jnp.bfloat16)
            return x, w

        def dew_fn(x, w, arm=arm):
            with (generation.assumed("sm120"), generation.chosen(grouped_matmul="mosaic"),
                  tuning.pinned({grouped_matmul.MosaicGmm: arm})):
                return gemm.gemm(x, w, None, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

        assert_as_exact(jax.jit(dew_fn), jax.jit(ref_fn), draw, f"sm120 staging {tag}", seeds=8,
                        truth_fn=functools.partial(ref_fn, dtype=jnp.float64))

    if generation.compile_only_gpus("sm120"):
        specs = (bf16((256, 128)), bf16((128, 256)))
        cases = []
        for tag, arm in (("sm120_staging_64x1", arm_64x1), ("sm120_staging_32x2", arm_32x2)):
            cases.append(KernelCase(tag, lambda arm=arm: (functools.partial(_pinned_product, arm), specs)))
        with generation.assumed("sm120"):
            records = compile_mosaic_kernels(cases, "sm120")
        for rec in records:
            assert not rec.get("error"), rec
            assert rec["kernels"], rec
            assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


def test_stated_output_staging_narrows_with_a_chained_replay_and_stays_off_syrk_and_linear_ce():
    """A Hopper 128-column tile staging its output in one 128-column chunk replays a chained launch of two
    products at 64 columns, its chunk narrowed alike, rather than at a tile its staging no longer divides; and
    linear CE's launch and kernel, which stage at their family's own chunks, take no stated staging (row
    5634)."""
    from dew.nn.kernels.templates.gemm import chained, linear_ce, pricing

    tile = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family="wgmma", staging=(128, 1))
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        family = pricing.family_of(tile, jnp.bfloat16, target)
        assert chained.replay_refusal(tile, 2, family, jnp.bfloat16) is None
        replay = chained.replay_tile(tile, 2)
        assert (replay.block_n, replay.staging) == (64, (64, 1))

        ce, bf16 = linear_ce.MOSAIC_LINEAR_CE, functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
        ba = ce.bind(bf16((128, 128)), bf16((256, 128)), jax.ShapeDtypeStruct((128,), jnp.int32), None,
                     transpose_rhs=True, interpret=False)
        derived = ce.derived(ba, target)
        assert derived is not None and ce.legal(ba, derived, target)
        staged = dataclasses.replace(derived, staging=(derived.block_n, 1))
        assert not ce.legal(ba, staged, target)
    with pytest.raises(ValueError):
        linear_ce.linear_ce(jnp.zeros((128, 128), jnp.bfloat16), jnp.zeros((256, 128), jnp.bfloat16),
                            jnp.zeros((128,), jnp.int32), tiles=staged, multiprocessors=1, interpret=True)


@lowers("sm90")
def test_a_hopper_product_stating_fp32_staging_at_whole_fragments_lowers():
    """An sm90 warpgroup-MMA product of fp32 output staged in 8-column chunks, whole accumulator fragments
    and whole 16-byte TMA stores, is legal and lowers to a kernel; at 4 columns, whole stores but half a
    fragment, the launch refuses it (row 5634)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    gmm = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    tile = tiling.Tiles(128, 128, 64, 2, 1, warpgroups=2, family="wgmma", staging=(8, 2))
    with generation.assumed("sm90"):
        sm90 = generation.gpu_target()
        ba = gmm.bind(bf16((256, 128)), bf16((1, 128, 256)), jax.ShapeDtypeStruct((1,), jnp.int32),
                      out_dtype="float32", trans_rhs=False, interpret=False, dense=True)
        assert gmm.legal(ba, tile, sm90)
        assert not gmm.legal(ba, dataclasses.replace(tile, staging=(4, 2)), sm90)
        case = KernelCase("fp32_staging_8", lambda: gmm.lowering(ba, tile))
        records = compile_mosaic_kernels([case], "sm90")
    for rec in records:
        assert not rec.get("error"), rec
        assert rec["kernels"], rec


@pytest.mark.parametrize("trans_rhs", [False, True], ids=["forward", "input gradient"])
def test_a_g4_dense_products_fed_ring_takes_the_stage_a_narrower_output_staging_gives(trans_rhs):
    """An RTX PRO 6000's (sm120) dense product at a Llama 3 8B projection, its consumers feeding their own
    ring (`mma.sync.fed`), derives its ring past its least busy depth with its output staged narrower than
    unstated (`derivation._ring_staged`): the unstated staging holds no ring that deep in the block, the
    stated one does (row 5634: two stages slept on their full barriers in 14.4% of M-NG-1's samples). An
    A100's tile, its cp.async pipeline staging its outputs whole, states none."""
    gmm = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    one = jax.ShapeDtypeStruct((1,), jnp.int32)
    ba = gmm.bind(bf16((8192, 4096)), bf16((1, 4096, 4096)), one, out_dtype="bfloat16", trans_rhs=trans_rhs,
                  interpret=False, dense=True)
    with generation.assumed("sm120"):
        sm120 = generation.gpu_target()
        tile = gmm.derived(ba, sm120)
        family = pricing.family_of(tile, jnp.bfloat16, sm120)
        assert family.feeds_itself and tile.staging is not None, tile
        assert tile.stages > 1 + family.loads_ahead + family.in_flight, tile
        assert gmm.legal(ba, tile, sm120)
        assert not gmm.legal(ba, dataclasses.replace(tile, staging=None), sm120)
    with generation.assumed("sm80"):
        assert gmm.derived(ba, generation.gpu_target()).staging is None


def test_a_refusal_at_a_fed_rings_least_depth_leaves_its_deeper_restaging_derived():
    """A compile's refusal of a fed ring's staging at its least busy depth (a pinned or tuned neighbour's)
    refuses that config alone: the derivation still takes the deeper ring at that staging
    (`derivation._ring_staged` asks its refusals of the tile it proposes)."""
    gmm = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    ba = gmm.bind(bf16((8192, 4096)), bf16((1, 4096, 4096)), jax.ShapeDtypeStruct((1,), jnp.int32),
                  out_dtype="bfloat16", trans_rhs=True, interpret=False, dense=True)
    with report.refusals(), generation.assumed("sm120"):
        sm120 = generation.gpu_target()
        tile = gmm.derived(ba, sm120)
        shallow = dataclasses.replace(tile, stages=2)
        report.refuse([(report._identity(gmm, ba, sm120), shallow, frozenset({gmm.KERNEL}))])
        assert report.refused(gmm, ba, shallow, sm120) and not gmm.legal(ba, shallow, sm120)
        assert gmm.derived(ba, sm120) == tile, tile


def test_a_partials_merge_slices_whole_atoms_of_its_family():
    """`across.merge_spans` cuts a partial's widest leaf into spans of whole 8-column atoms and of its
    family's atom (row 14073): 160 columns at mma.sync's 32-column atom in five spans of 32 (not two of 80,
    which its layout casts to no tile); 96 columns at a 64-column atom, which no whole split holds, in one
    span; and at an atom of one, the 8-column spans it took before."""
    from dew.nn.kernels.templates import across

    def spans(columns, held, atom):
        return across.merge_spans(np.zeros((64, columns), np.float32), held, atom)

    over = 2 * across.MERGED_REGISTERS
    assert spans(160, over, 32) == (0, [(start, 32) for start in range(0, 160, 32)])
    assert spans(96, over, 64) == (None, [(0, None)])
    assert spans(160, over, 1) == (0, [(0, 80), (80, 80)])


def test_a_pinned_sm120_tile_whose_balanced_tail_merges_its_partials_lowers():
    """The 128 x 160 x 64 two-stage mma.sync tile pinned at Qwen3.5-4B's down projection (8192 x 9216 ->
    2560) lowers on a compile-only RTX PRO 6000: its balanced tail merges each consumer's 64 x 160 partial
    in spans of whole 32-column atoms of mma.sync's one-row-warp layout, where spans of 80 cast to no
    layout (row 14073, M-NG-34b's down forward: 'Cannot cast to layout ... (64, 80)')."""
    from jax.sharding import Mesh

    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    arm = tiling.Tiles(128, 160, 64, 2, 8, warpgroups=2, family="mma.sync.fed")
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm120")
    with generation.assumed("sm120"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_pinned_product, arm)).trace(bf16((8192, 9216)), bf16((9216, 2560))).lower(
            lowering_platforms=("cuda",))


def _pinned_product(arm: tiling.Tiles, x, w):
    """`x @ w` in bf16 through the template's dense launch pinned at `arm`."""
    with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({grouped_matmul.MosaicGmm: arm}):
        return gemm.gemm(x, w, None, out_dtype=jnp.bfloat16)
