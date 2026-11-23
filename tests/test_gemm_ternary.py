"""Check-1 accuracy, stage decode, BitNet b1.58 activation quantization + STE, sm80/sm90 compilation, and
TF32 grouped/dense Mosaic GEMM tests (`gemm.ternary_weight_product`, `formats.TernaryWeights`, rows 1464,
3428)."""

from __future__ import annotations

import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import KernelCase
from generation_support import lowers
from kernel_compile import compile_mosaic_kernels
from low_precision_support import StageRef, decoded_weight_dot
from reference_error import assert_as_exact

from dew.nn import precision
from dew.nn.kernels import formats, gemm, generation, grouped_matmul
from dew.nn.kernels.templates.gemm import packed


def test_ternary_weights_pack_decode_and_stage_match_bitwise():
    """`formats.TernaryWeights` packs 16 2-bit codes per uint32 word (`0->0`, `1->+1`, `2->-1`, `3->0`),
    matches `packed.stage_decoded_ternary` bitwise, and validates form version & scale shapes (row 1464)."""
    rng = np.random.default_rng(1464)
    w = jnp.asarray(rng.standard_normal((256, 64)), jnp.float32)
    packed_w = formats.TernaryWeights.of(w)
    assert packed_w.checked() is packed_w
    assert packed_w.words.shape == (64, 16)
    vals = np.asarray(packed_w.unscaled(jnp.float32))
    assert set(np.unique(vals)).issubset({-1.0, 0.0, 1.0})

    out = StageRef(np.zeros((64, 256), dtype=jnp.bfloat16))
    packed.stage_decoded_ternary([StageRef(packed_w.words)], out, interpret=True)
    np.testing.assert_array_equal(np.asarray(out.arr.T, np.float32), vals)

    # Code 3 (0b11) decodes cleanly to 0.0.
    all_threes = jnp.full((16, 64), jnp.uint32(0xFFFFFFFF))
    np.testing.assert_array_equal(np.asarray(formats.ternary_codes(all_threes, jnp.float32)),
                                  np.zeros((256, 64), np.float32))
    bad = dataclasses.replace(packed_w, form=np.zeros((8,), np.uint8))
    with pytest.raises(ValueError):
        bad.checked()
    with pytest.raises(ValueError):
        formats.TernaryWeights.of(jnp.ones((128, 64), jnp.float32))

    # P0 #3: large-magnitude absmean must not overflow sum-before-divide to inf/NaN.
    huge = jnp.full((256, 64), jnp.float32(2.0 ** 126))
    huge_w = formats.TernaryWeights.of(huge)
    assert np.isfinite(np.asarray(huge_w.scale)).all()
    np.testing.assert_array_equal(np.asarray(huge_w.decoded(jnp.float32)), np.asarray(huge))

    # P0 #4: zero-depth weights pack, decode, and multiply to zeros without reshape(-1, ..., 0) crash.
    empty_w = formats.TernaryWeights.of(jnp.zeros((0, 64), jnp.float32))
    assert empty_w.unscaled(jnp.float32).shape == (0, 64)
    assert empty_w.decoded(jnp.float32).shape == (0, 64)
    out_empty = gemm.ternary_weight_product(jnp.ones((16, 0), jnp.bfloat16), empty_w,
                                            out_dtype=jnp.float32, interpret_on_cpu=False)
    np.testing.assert_array_equal(np.asarray(out_empty), np.zeros((16, 64), np.float32))


def test_ternary_weight_product_holds_check_1_dense_grouped_bitlinear_and_compiles():
    """`gemm.ternary_weight_product` holds check 1 on both default XLA and interpreted Mosaic routes across
    dense & grouped scales, matches BitNet b1.58 per-token int8 activation quantization + STE (including
    zero rows), and compiles spill-free on sm80 and sm90 (row 1464)."""
    def draw(seed):
        rng = np.random.default_rng(1464 + seed)
        x = jnp.asarray(rng.standard_normal((128, 256)) / np.sqrt(256), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((256, 64)), jnp.float32)
        scale = jnp.asarray(rng.uniform(0.5, 2.5, size=(64,)), jnp.float32)
        return x, formats.TernaryWeights.of(w, scale=scale)

    for interpret in (False, True):
        def dew_fn(x, weights, *, interpret=interpret):
            ctx = generation.chosen(grouped_matmul="mosaic") if interpret else contextlib.nullcontext()
            with ctx:
                return gemm.ternary_weight_product(x, weights, quantize_activations=False,
                                                   out_dtype=jnp.float32, interpret_on_cpu=interpret)

        assert_as_exact(jax.jit(dew_fn), jax.jit(decoded_weight_dot), draw,
                        f"ternary_weight interp={interpret}",
                        seeds=8, truth_fn=functools.partial(decoded_weight_dot, dtype=jnp.float64))
    rng = np.random.default_rng(1465)
    x_g = jnp.asarray(rng.integers(-3, 4, size=(64, 256)), jnp.bfloat16)
    w_g = jnp.asarray(rng.standard_normal((2, 256, 64)), jnp.float32)
    sizes = jnp.asarray([40, 24], jnp.int32)
    for scale in (2.0, rng.choice([0.5, 2.0], size=(64,)), [[[0.5]], [[2.0]]],
                  rng.choice([0.5, 2.0], size=(2, 1, 64))):
        bank = formats.TernaryWeights.of(w_g, scale=jnp.asarray(scale, jnp.float32))
        ref = jax.lax.ragged_dot(x_g.astype(jnp.float32), bank.decoded(jnp.float32), sizes,
                                 preferred_element_type=jnp.float32)
        for interpret in (False, True):
            ctx = generation.chosen(grouped_matmul="mosaic") if interpret else contextlib.nullcontext()
            with ctx:
                got = gemm.ternary_weight_product(x_g, bank, sizes, quantize_activations=False,
                                                  out_dtype=jnp.float32, interpret_on_cpu=interpret)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
        np.testing.assert_array_equal(np.asarray(gemm.gemm(x_g, bank, sizes, out_dtype=jnp.float32)),
                                      np.asarray(ref))

    # BitNet b1.58 per-token int8 activation quantization + zero-row safety + unquantized STE gradient.
    x_bit = jnp.asarray(rng.standard_normal((32, 256)), jnp.bfloat16).at[0].set(0.0)
    w_bit = formats.TernaryWeights.of(jnp.asarray(rng.standard_normal((256, 64)), jnp.float32))
    x_f32 = x_bit.astype(jnp.float32)
    peak = jnp.max(jnp.abs(x_f32), axis=-1)
    safe_peak = jnp.where(peak > 0, peak, jnp.float32(1.0))
    x_q = jnp.clip(jnp.round(precision.divided(x_f32, safe_peak[:, None]) * jnp.float32(127.0)),
                   -128.0, 127.0)
    w_bf16 = w_bit.unscaled(jnp.bfloat16)
    dot_int = jax.lax.dot(x_q.astype(jnp.bfloat16), w_bf16, preferred_element_type=jnp.float32)
    ref_bit = precision.twice_scaled(precision.divided(dot_int, jnp.float32(127.0)),
                                     peak[:, None], w_bit.scale)
    max_col = int(jnp.argmax(jnp.abs(x_bit[1])))
    tan_x = jnp.zeros_like(x_bit).at[1, max_col].set(jnp.bfloat16(1.0))
    dx_exact = (jax.lax.dot(jnp.ones((32, 64), jnp.bfloat16), w_bf16.T,
                            preferred_element_type=jnp.float32) * w_bit.scale).astype(jnp.bfloat16)
    dscale_exact = jnp.sum(precision.twice_scaled(precision.divided(dot_int, jnp.float32(127.0)),
                                                  peak[:, None], jnp.float32(1.0)))
    jvp_exact = jax.lax.dot(tan_x, w_bf16, preferred_element_type=jnp.float32) * w_bit.scale
    assert np.asarray(dx_exact[1, max_col]) != 0.0
    for interpret in (False, True):
        ctx = generation.chosen(grouped_matmul="mosaic") if interpret else contextlib.nullcontext()
        with ctx:
            got_bit = gemm.ternary_weight_product(x_bit, w_bit, quantize_activations=True,
                                                  out_dtype=jnp.float32, interpret_on_cpu=interpret)
            _, (dx, dscale) = jax.value_and_grad(
                lambda a, s, interpret=interpret: jnp.sum(gemm.ternary_weight_product(
                    a, dataclasses.replace(w_bit, scale=s), quantize_activations=True,
                    out_dtype=jnp.float32, interpret_on_cpu=interpret)),
                argnums=(0, 1))(x_bit, w_bit.scale)
            _, y_tan = jax.jvp(
                lambda a, interpret=interpret: gemm.ternary_weight_product(
                    a, w_bit, quantize_activations=True, out_dtype=jnp.float32, interpret_on_cpu=interpret),
                (x_bit,), (tan_x,))
        np.testing.assert_array_equal(np.asarray(got_bit[0]), np.zeros((64,), np.float32))
        np.testing.assert_array_equal(np.asarray(got_bit), np.asarray(ref_bit))
        np.testing.assert_array_equal(np.asarray(dx), np.asarray(dx_exact))
        np.testing.assert_array_equal(np.asarray(dscale), np.asarray(dscale_exact))
        np.testing.assert_array_equal(np.asarray(y_tan[1]), np.asarray(jvp_exact[1]))
    for gpu in ("sm80", "sm90"):
        if not generation.compile_only_gpus(gpu):
            continue
        x_s = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
        w_sample = formats.TernaryWeights.of(np.ones((256, 128), np.float32))
        w_spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype), w_sample)

        def compiled_ternary(a, w):
            with generation.chosen(grouped_matmul="mosaic"):
                return gemm.ternary_weight_product(a, w, out_dtype=jnp.bfloat16)

        with generation.assumed(gpu), generation.chosen(grouped_matmul="mosaic"):
            records = compile_mosaic_kernels(
                [KernelCase(f"ternary_weight_{gpu}",
                            lambda fn=compiled_ternary, x=x_s, w=w_spec: (fn, (x, w)))], gpu)
        for rec in records:
            assert not rec.get("error"), rec
            assert rec["kernels"], rec
            assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90")),
                                 pytest.param("sm100", marks=lowers("sm100"))])
def test_fp32_tf32_grouped_and_dense_gemm_derives_and_compiles_on_mosaic(gpu: str):
    """FP32 operands at `DEFAULT` precision (TF32) derive legal tiles and compile through `MosaicGmm` on
    sm80, sm90, and sm100, and transposed dX / dW derive on families with `transposes_tf32` (row 3428)."""
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
    gmm_launch, tgmm_launch = grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()
    with generation.assumed(gpu):
        target = generation.gpu_target()
        fwd_ba = gmm_launch.bind(f32((256, 128)), f32((2, 128, 128)), sizes, out_dtype="float32",
                                 trans_rhs=False, interpret=False, dense=False, exact_sums=False)
        fwd_tile = gmm_launch.derived(fwd_ba, target)
        assert fwd_tile is not None, (gpu, fwd_tile)
        dx_ba = gmm_launch.bind(f32((256, 128)), f32((2, 128, 128)), sizes, out_dtype="float32",
                                trans_rhs=True, interpret=False, dense=False, exact_sums=False)
        dw_ba = tgmm_launch.bind(f32((256, 128)), f32((256, 128)), sizes, out_dtype="float32",
                                 interpret=False, dense=False, exact_sums=False)
        assert gmm_launch.derived(dx_ba, target) is not None
        assert tgmm_launch.derived(dw_ba, target) is not None
        # Public HIGHEST precision on fp32 must refuse TF32 (`exact_sums=True` -> never TF32).
        high_ba = gmm_launch.bind(f32((256, 128)), f32((2, 128, 128)), sizes, out_dtype="float32",
                                  trans_rhs=(gpu == "sm90"), interpret=False, dense=False, exact_sums=True)
        assert gmm_launch.derived(high_ba, target) is None
        from jax.core import ShapedArray
        from reference_error import equations

        with generation.chosen(grouped_matmul="mosaic"):
            closed_high = jax.make_jaxpr(
                lambda x, w: gemm.gemm(x, w, out_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST)
            )(f32((256, 128)), f32((128, 128)))
        checked = 0
        for eqn in equations(closed_high, platform="cuda"):
            if eqn.primitive.name in ("dot_general", "ragged_dot_general"):
                assert eqn.params.get("precision") == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
                checked += 1
            elif eqn.primitive.name in ("mpmd_map", "pallas_call"):
                matrix_dtypes = [
                    v.aval.dtype for v in eqn.invars
                    if isinstance(v.aval, ShapedArray) and v.aval.ndim >= 2
                    and jnp.issubdtype(v.aval.dtype, jnp.floating)
                ]
                if matrix_dtypes:
                    assert len(matrix_dtypes) >= 2 and all(dt == jnp.bfloat16 for dt in matrix_dtypes[:2]), (
                        gpu, matrix_dtypes)
                    checked += 1
        assert checked > 0, f"expected lowered GEMM equations in CUDA branch for {gpu}"
        if generation.compile_only_gpus(gpu):
            def run_dense(x, w):
                with generation.chosen(grouped_matmul="mosaic"):
                    return gemm.gemm(x, w, transpose_rhs=(gpu == "sm90"), out_dtype=jnp.float32,
                                     precision=jax.lax.Precision.DEFAULT)

            def run_grouped(x, w, s):
                with generation.chosen(grouped_matmul="mosaic"):
                    return gemm.gemm(x, w, s, transpose_rhs=(gpu == "sm90"), out_dtype=jnp.float32,
                                     precision=jax.lax.Precision.DEFAULT)

            def run_dx(dy, w, s):
                with generation.chosen(grouped_matmul="mosaic"):
                    return gemm.gemm(dy, w, s, transpose_rhs=True, out_dtype=jnp.float32,
                                     precision=jax.lax.Precision.DEFAULT)

            def run_dw(x, dy, s):
                with generation.chosen(grouped_matmul="mosaic"):
                    return gemm.gemm_t(x, dy, s, out_dtype=jnp.float32,
                                       precision=jax.lax.Precision.DEFAULT)

            cases = [
                KernelCase(f"tf32_dense_{gpu}", lambda: (run_dense, (f32((256, 128)), f32((128, 128))))),
                KernelCase(f"tf32_grouped_{gpu}",
                           lambda: (run_grouped, (f32((256, 128)), f32((2, 128, 128)), sizes))),
                KernelCase(f"tf32_dx_{gpu}",
                           lambda: (run_dx, (f32((256, 128)), f32((2, 128, 128)), sizes))),
                KernelCase(f"tf32_dw_{gpu}",
                           lambda: (run_dw, (f32((256, 128)), f32((256, 128)), sizes))),
            ]
            records = compile_mosaic_kernels(cases, gpu)
            for rec in records:
                assert not rec.get("error"), rec
                assert rec["kernels"], rec
                assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


def test_ternary_ste_cancellation_overflow_and_subnormal_rows():
    """STE pullback never overflows on canceling extreme rows (`2**126` with `+1/-1` and scale `2**10` ->
    forward `0`, grad `[+1024, -1024]`), and tiny normal rows (`2**-125`, whose `peak / 127` underflows to
    subnormal/zero under FTZ, with scale `2**101`) normalize against `peak` before scaling by `127` to
    produce exact `2**-24` without underflow (row 1464)."""
    # Cancellation overflow witness: two entries of 2**126 against +1 and -1 with weight scale 2**10.
    x_cancel = jnp.zeros((16, 256), jnp.float32).at[0, 0].set(2.0 ** 126).at[0, 1].set(2.0 ** 126)
    w_mat = jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0).at[1, 0].set(-1.0)
    w_cancel = dataclasses.replace(formats.TernaryWeights.of(w_mat, scale=jnp.float32(1.0)),
                                   scale=jnp.float32(2.0 ** 10))
    # Tiny normal row witness: 2**-125 (peak / 127 < tiny = 2**-126 flushes to 0 on XLA CPU FTZ) with
    # weight scale 2**101 -> representable result 2**-24.
    x_tiny = jnp.zeros((16, 256), jnp.float32).at[0, 0].set(2.0 ** -125)
    w_tiny = dataclasses.replace(
        formats.TernaryWeights.of(jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0),
                                  scale=jnp.float32(1.0)),
        scale=jnp.float32(2.0 ** 101))

    for kernel in ("mosaic", "xla"):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel):
            y_c, (dx_c, ds_c) = jax.value_and_grad(
                lambda a, s: gemm.ternary_weight_product(
                    a, dataclasses.replace(w_cancel, scale=s), quantize_activations=True,
                    out_dtype=jnp.float32, interpret_on_cpu=True)[0, 0],
                argnums=(0, 1))(x_cancel, w_cancel.scale)
            assert np.asarray(y_c) == np.float32(0.0), (kernel, y_c)
            assert np.asarray(dx_c[0, 0]) == np.float32(1024.0), (kernel, dx_c[0, 0])
            assert np.asarray(dx_c[0, 1]) == np.float32(-1024.0), (kernel, dx_c[0, 1])
            assert np.all(np.isfinite(np.asarray(dx_c))), kernel
            assert np.asarray(ds_c) == np.float32(0.0), (kernel, ds_c)

            y_t, (dx_t, ds_t) = jax.value_and_grad(
                lambda a, s: gemm.ternary_weight_product(
                    a, dataclasses.replace(w_tiny, scale=s), quantize_activations=True,
                    out_dtype=jnp.float32, interpret_on_cpu=True)[0, 0],
                argnums=(0, 1))(x_tiny, w_tiny.scale)
            assert np.asarray(y_t) == np.float32(2.0 ** -24), (kernel, y_t)
            assert np.asarray(dx_t[0, 0]) == np.float32(2.0 ** 101), (kernel, dx_t[0, 0])
            assert np.asarray(ds_t) == np.float32(2.0 ** -125), (kernel, ds_t)


def test_swapped_register_decode_holds_for_fp8_and_ternary_at_skinny_rows():
    """Skinny decode products (`M=13`) for `Fp8Weights` (`e4m3`/`e5m2`) and `TernaryWeights` decode
    directly into `wgmma` register operand A (`packed_rows=True`, zero SMEM staging), matching the decoded
    fp32 dot bitwise under forced `mosaic`, matching staged decode bit-for-bit on encoded E4M3 NaN and E5M2
    Inf/NaN codes with independent mask checks, rejecting `splits=2`, and compiling spill-free on sm90
    (row 5387)."""
    from dew.algebra import bits
    from dew.nn.kernels import tuning
    from dew.nn.kernels.templates.gemm import ternary, tiling

    rng = np.random.default_rng(5387)
    x = jnp.asarray(rng.integers(-3, 4, size=(13, 256)), jnp.bfloat16)
    w_raw = jnp.asarray(rng.integers(-2, 3, size=(256, 128)), jnp.float32)
    with generation.assumed("sm90"):
        sm90 = generation.gpu_target("sm90")
        swapped_fp8 = tiling.Tiles(128, 64, 64, 3, 1, warpgroups=2, family="wgmma.fed", swapped=True)
        for fmt in ("e4m3", "e5m2"):
            fp8_w = formats.Fp8Weights.of(w_raw, fmt=fmt, scale=0.5)
            with (generation.chosen(grouped_matmul="mosaic"),
                  tuning.pinned({grouped_matmul.MosaicFp8PackedGmm: swapped_fp8})):
                got = gemm.fp8_weight_product(x, fp8_w, out_dtype=jnp.float32, interpret_on_cpu=True)
            want = jax.lax.dot(x.astype(jnp.float32), fp8_w.decoded(jnp.float32),
                               preferred_element_type=jnp.float32)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
            fp8_launch = grouped_matmul.MosaicFp8PackedGmm()
            sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
            ba_skinny = fp8_launch.bind(
                jax.ShapeDtypeStruct((16, 4096), jnp.bfloat16),
                jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype),
                             formats.Fp8Weights.of(np.ones((1, 4096, 128), np.float32), fmt=fmt)),
                sizes, (), (), None, None, None, out_dtype="float32", trans_rhs=False, interpret=False,
                dense=True, exact_sums=False)
            assert not fp8_launch.legal(ba_skinny, dataclasses.replace(swapped_fp8, splits=2), sm90)
            derived = fp8_launch.derived(ba_skinny, sm90)
            assert derived is None or derived.splits == 1, derived
        tern_w = formats.TernaryWeights.of(w_raw, scale=0.5)
        swapped_t = tiling.Tiles(128, 64, 256, 3, 1, warpgroups=2, family="wgmma.fed", swapped=True)
        with (generation.chosen(grouped_matmul="mosaic"),
              tuning.pinned({ternary.MosaicTernaryPackedGmm: swapped_t})):
            got_t = gemm.ternary_weight_product(x, tern_w, quantize_activations=False,
                                                out_dtype=jnp.float32, interpret_on_cpu=True)
        want_t = jax.lax.dot(x.astype(jnp.float32), tern_w.decoded(jnp.float32),
                             preferred_element_type=jnp.float32)
        np.testing.assert_array_equal(np.asarray(got_t), np.asarray(want_t))
        t_launch = ternary.MosaicTernaryPackedGmm()
        ba_t = t_launch.bind(
            jax.ShapeDtypeStruct((16, 4096), jnp.bfloat16),
            jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype),
                         formats.TernaryWeights.of(np.ones((1, 4096, 128), np.float32))),
            jax.ShapeDtypeStruct((1,), jnp.int32), (), (), None, None, None, out_dtype="float32",
            trans_rhs=False, interpret=False, dense=True, exact_sums=False)
        assert not t_launch.legal(ba_t, dataclasses.replace(swapped_t, splits=2), sm90)
        derived_t = t_launch.derived(ba_t, sm90)
        assert derived_t is None or derived_t.splits == 1, derived_t
        e4_codes = np.broadcast_to(np.asarray([0x38, 0x00, 0x7F, 0xFF], np.uint32), (64, 16, 4))
        e5_codes = np.broadcast_to(np.asarray([0x7C, 0xFC, 0x7D, 0xFF], np.uint32), (64, 16, 4))
        special_words = {
            "e4m3": np.asarray(bits.fp8_packed(jnp.asarray(e4_codes.reshape(64, 64)), bits.E4M3)),
            "e5m2": np.asarray(bits.fp8_packed(jnp.asarray(e5_codes.reshape(64, 64)), bits.E5M2)),
        }
        for fmt, raw_words in special_words.items():
            reg_out = packed.rows_decoded([StageRef(raw_words)], None, interpret=True, fmt=fmt)
            stg_out = StageRef(np.zeros((64, 64), dtype=jnp.bfloat16))
            packed.stage_decoded_words([StageRef(raw_words)], stg_out, packed.weight_format(fmt),
                                       interpret=True)
            reg_arr, stg_arr = np.asarray(reg_out), np.asarray(stg_out.arr)
            np.testing.assert_array_equal(reg_arr.view(np.uint16), stg_arr.view(np.uint16))
            f32 = reg_arr.astype(np.float32)
            if fmt == "e4m3":
                assert np.all(f32[:, 0:16] == 1.0) and np.all(f32[:, 16:32] == 0.0)
                assert np.all(np.isnan(f32[:, 32:48])) and np.all(np.isnan(f32[:, 48:64]))
            else:
                assert np.all(np.isposinf(f32[:, 0:16])) and np.all(np.isneginf(f32[:, 16:32]))
                assert np.all(np.isnan(f32[:, 32:48])) and np.all(np.isnan(f32[:, 48:64]))
        if generation.compile_only_gpus("sm90"):
            x_s = jax.ShapeDtypeStruct((13, 256), jnp.bfloat16)
            e4_spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype),
                                   formats.Fp8Weights.of(np.ones((256, 128), np.float32), fmt="e4m3"))
            e5_spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype),
                                   formats.Fp8Weights.of(np.ones((256, 128), np.float32), fmt="e5m2"))
            t_spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype), tern_w)

            def c_fp8(a, w):
                with generation.chosen(grouped_matmul="mosaic"), tuning.pinned(
                        {grouped_matmul.MosaicFp8PackedGmm: swapped_fp8}):
                    return gemm.fp8_weight_product(a, w, out_dtype=jnp.bfloat16)

            def c_tern(a, w):
                with generation.chosen(grouped_matmul="mosaic"), tuning.pinned(
                        {ternary.MosaicTernaryPackedGmm: swapped_t}):
                    return gemm.ternary_weight_product(a, w, quantize_activations=False,
                                                       out_dtype=jnp.bfloat16)

            records = compile_mosaic_kernels(
                [KernelCase("swapped_fp8_e4m3_sm90", lambda: (c_fp8, (x_s, e4_spec))),
                 KernelCase("swapped_fp8_e5m2_sm90", lambda: (c_fp8, (x_s, e5_spec))),
                 KernelCase("swapped_ternary_sm90", lambda: (c_tern, (x_s, t_spec)))], "sm90")
            for rec in records:
                assert not rec.get("error"), rec
                assert rec["kernels"], rec
                assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec
