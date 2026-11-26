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
    # And pull back through the rescue's branch, traced, under a jit: an empty gradient.
    empty_product = functools.partial(gemm.ternary_weight_product, weights=empty_w,
                                      quantize_activations=False, out_dtype=jnp.float32)
    pull = jax.jit(lambda dy: jax.vjp(empty_product, jnp.ones((16, 0), jnp.float32))[1](dy))
    (d_empty,) = pull(jnp.full((16, 64), 2.0 ** 126, jnp.float32))
    assert d_empty.shape == (16, 0)


def test_ternary_weight_product_holds_check_1_on_both_routes():
    """`gemm.ternary_weight_product` holds check 1 on both the default XLA and the interpreted Mosaic route
    at a per-column scale (row 1464)."""
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


def test_a_grouped_ternary_weight_product_is_exact_at_every_scale_shape():
    """A grouped ternary product of small integer rows is its decoded ragged dot exactly at a scalar, a
    per-column, a per-group and a per-group-column scale, on both routes and through `gemm.gemm` (row
    1464)."""
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



def test_a_ternary_weight_product_quantizes_its_activations_as_bitnet_does():
    """BitNet b1.58's per-token int8 activation quantization, a zero row's product zero, and the unquantized
    straight-through gradient in the rows and the scale, on both routes (row 1464)."""
    rng = np.random.default_rng(1466)
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


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
def test_a_ternary_weight_product_compiles_spill_free(gpu: str):
    """The ternary product's Mosaic launch compiles spill-free on a compile-only sm80 and sm90 (row 1464)."""
    if generation.compile_only_gpus(gpu):
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


def test_packed_products_hold_extreme_scales_forward_and_backward():
    """Packed ternary products at scales near fp32's edges (row 1464): a ternary row of `2**126` and
    `+1/-1` codes cancels to 0 with STE gradients of the codes times the scale; a subnormal `2**-144` row
    codes on a process that flushes subnormals, `2**-144 * 2**120 = 2**-24`; and a cotangent of `[4, 4]`
    against columns `[+1, -1]` at a scale of `2**126` (each cotangent times its factor past fp32's
    largest) pulls back to exactly 0 (`[4, 2]` to `2**127`), not `inf - inf`, for a tensor scale, with and
    without activation quantization; a tiny cotangent beside a huge one on another input pulls back
    whole."""
    x_cancel = jnp.zeros((16, 256), jnp.float32).at[0, 0].set(2.0 ** 126).at[0, 1].set(2.0 ** 126)
    w_mat = jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0).at[1, 0].set(-1.0)
    w_cancel = dataclasses.replace(formats.TernaryWeights.of(w_mat, scale=jnp.float32(1.0)),
                                   scale=jnp.float32(2.0 ** 10))
    x_sub = jnp.zeros((16, 256), jnp.float32).at[0, 0].set(jnp.float32(2.0 ** -144))
    w_sub = dataclasses.replace(
        formats.TernaryWeights.of(jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0),
                                  scale=jnp.float32(1.0)),
        scale=jnp.float32(2.0 ** 120))
    x_cot = jnp.zeros((16, 256), jnp.float32).at[0, 0].set(1.0)
    w_cot_mat = jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0).at[0, 1].set(-1.0)
    huge = jnp.float32(2.0 ** 126)
    ternary = formats.TernaryWeights.of(w_cot_mat, scale=jnp.float32(1.0))
    products = [
        *((functools.partial(gemm.ternary_weight_product, weights=dataclasses.replace(ternary, scale=huge),
                             quantize_activations=quantized), (quantized, "ternary"))
          for quantized in (True, False)),
    ]
    eye = jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0).at[1, 1].set(1.0)
    unit = jnp.float32(1.0)
    separate = (
        (functools.partial(gemm.ternary_weight_product, weights=formats.TernaryWeights.of(eye, scale=unit),
                           quantize_activations=False), "ternary"),)
    mixed = jnp.zeros((16, 64), jnp.float32).at[0, 0].set(2.0 ** 126).at[0, 1].set(2.0 ** -120)
    cancelling = jnp.zeros((16, 64), jnp.float32).at[0, 0].set(4.0).at[0, 1].set(4.0)
    apart = jnp.zeros((16, 64), jnp.float32).at[0, 0].set(4.0).at[0, 1].set(2.0)

    for kernel in ("mosaic", "xla"):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel):
            y_c, (dx_c, ds_c) = jax.value_and_grad(
                lambda a, s: gemm.ternary_weight_product(
                    a, dataclasses.replace(w_cancel, scale=s), quantize_activations=True,
                    out_dtype=jnp.float32, interpret_on_cpu=True)[0, 0],
                argnums=(0, 1))(x_cancel, w_cancel.scale)
            assert np.asarray(y_c) == np.float32(0.0), (kernel, y_c)
            want = np.zeros((16, 256), np.float32)
            want[0, 0], want[0, 1] = 1024.0, -1024.0
            np.testing.assert_array_equal(np.asarray(dx_c), want, err_msg=kernel)
            assert np.asarray(ds_c) == np.float32(0.0), (kernel, ds_c)

            y_s = gemm.ternary_weight_product(x_sub, w_sub, quantize_activations=True,
                                              out_dtype=jnp.float32, interpret_on_cpu=True)
            assert np.asarray(y_s[0, 0]) == np.float32(2.0 ** -24), (kernel, y_s[0, 0])

            for product, label in products:
                ran = functools.partial(product, out_dtype=jnp.float32, interpret_on_cpu=True)
                y_out, pull = jax.vjp(ran, x_cot)
                assert np.asarray(y_out[0, 0]) == np.float32(2.0 ** 126), (kernel, label, y_out[0, 0])
                assert np.asarray(y_out[0, 1]) == np.float32(-(2.0 ** 126)), (kernel, label, y_out[0, 1])
                (dx_zero,) = pull(cancelling)
                np.testing.assert_array_equal(np.asarray(dx_zero), np.zeros((16, 256), np.float32),
                                              err_msg=f"{kernel} {label}")
                (dx_apart,) = pull(apart)
                want = np.zeros((16, 256), np.float32)
                want[0, 0] = 2.0 ** 127
                np.testing.assert_array_equal(np.asarray(dx_apart), want, err_msg=f"{kernel} {label}")

            # Cotangents of 2**127 against two +1 codes at scales of a quarter and of zero: 2**126 and 0,
            # the cotangent times its scale first, as the plain transpose has it.
            pair = formats.TernaryWeights.of(jnp.abs(w_cot_mat), scale=jnp.float32(1.0))
            for scale, expected in ((0.25, 2.0 ** 126), (0.0, 0.0)):
                _, pull = jax.vjp(lambda a, scale=scale, pair=pair: gemm.ternary_weight_product(
                    a, dataclasses.replace(pair, scale=jnp.float32(scale)), quantize_activations=False,
                    out_dtype=jnp.float32, interpret_on_cpu=True), x_cot)
                (dx_small,) = pull(jnp.zeros((16, 64), jnp.float32).at[0, :2].set(2.0 ** 127))
                want = np.zeros((16, 256), np.float32)
                want[0, 0] = expected
                np.testing.assert_array_equal(np.asarray(dx_small), want, err_msg=f"{kernel} {scale}")

            # A row's tiny cotangent beside a huge one on another input, no product past fp32's largest:
            # each pulls back whole, as the plain transpose gives it.
            for product, label in separate:
                ran = functools.partial(product, out_dtype=jnp.float32, interpret_on_cpu=True)
                (dx_mixed,) = jax.vjp(ran, x_cot)[1](mixed)
                want = np.zeros((16, 256), np.float32)
                want[0, 0], want[0, 1] = 2.0 ** 126, 2.0 ** -120
                np.testing.assert_array_equal(np.asarray(dx_mixed), want, err_msg=f"{kernel} {label}")


@pytest.mark.mesh(devices=2)
def test_a_packed_products_transpose_holds_in_a_checked_map_linear_transpose_and_an_empty_width():
    """A packed ternary product's input gradient in a `shard_map` over two CPU devices that checks the axes
    its values vary over, the rows and codes the same on every device and each device its own scale, is
    each device's partial summed, as the decoded product written plainly gives it, exactly; its transpose
    by `jax.linear_transpose` is its VJP's; and a product of no columns pulls back zeros (row 1464)."""
    rng = np.random.default_rng(1464)
    codes = jnp.asarray(rng.integers(-1, 2, (256, 32)), jnp.float32)
    weights = formats.TernaryWeights.of(codes, scale=jnp.float32(1.0))
    x = jnp.asarray(rng.integers(-3, 4, (16, 256)), jnp.float32)
    scales = jnp.asarray([2.0 ** 100, 2.0 ** -3], jnp.float32)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def packed(rows, scale):
        return gemm.ternary_weight_product(rows, dataclasses.replace(weights, scale=scale[0]),
                                           quantize_activations=False, out_dtype=jnp.float32)

    def plain(rows, scale):
        return rows.astype(jnp.bfloat16).astype(jnp.float32) @ codes * scale[0]

    def gradient(product):
        def each_device(rows, scale):
            return jax.grad(lambda r: jnp.sum(product(r, scale)))(rows)

        return jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(whole, data), out_specs=whole,
                                     check_vma=True))(x, scales)

    np.testing.assert_array_equal(np.asarray(gradient(packed)), np.asarray(gradient(plain)))

    # Each device's scale its own (256 and 257) and its cotangent its own (+1 and -1), the rows and codes
    # every device's: the cotangents times their scales sum to -1 before the pullback, as the plain
    # product's transpose sums them, not each device's pullback rounded to bf16 first (257 to 256) and
    # summed to 0.
    def pulled_back(product):
        def each_device(rows, scale, dy):
            return jax.vjp(lambda r: product(r, scale), rows)[1](dy)[0]

        signs = jnp.zeros((32, 32), jnp.float32).at[0, 0].set(1.0).at[16, 0].set(-1.0)
        return jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(whole, data, data), out_specs=whole,
                                     check_vma=True))(x[:16], jnp.asarray([256.0, 257.0], jnp.float32), signs)

    np.testing.assert_array_equal(np.asarray(pulled_back(packed)), np.asarray(pulled_back(plain)))

    # The rows the same on every device, each device's codes its own (a matrix split over the map's axis), one
    # replicated scale of 2**126 and each device's cotangent +4 or -4: their partials sum to 0 before the
    # scale joins, not to inf - inf.
    def cancelled(rows, dy):
        own = dataclasses.replace(weights, words=jax.lax.pcast(weights.words, ("data",), to="varying"),
                                  scale=jnp.float32(2.0 ** 126))
        return jax.vjp(lambda r: gemm.ternary_weight_product(r, own, quantize_activations=False,
                                                             out_dtype=jnp.float32), rows)[1](dy)[0]

    signs = jnp.zeros((32, 32), jnp.float32).at[0, 0].set(4.0).at[16, 0].set(-4.0)
    d_rows = jax.jit(jax.shard_map(cancelled, mesh=mesh, in_specs=(whole, data), out_specs=whole,
                                   check_vma=True))(x[:16], signs)
    np.testing.assert_array_equal(np.asarray(d_rows), np.zeros((16, 256), np.float32))

    cotangent = jnp.asarray(rng.integers(-3, 4, (16, 32)), jnp.float32)
    one = functools.partial(packed, scale=scales[:1])
    (transposed,) = jax.linear_transpose(one, x)(cotangent)
    (pulled,) = jax.vjp(one, x)[1](cotangent)
    np.testing.assert_array_equal(np.asarray(transposed), np.asarray(pulled))

    empty = formats.TernaryWeights.of(jnp.zeros((256, 0), jnp.float32), scale=jnp.float32(1.0))
    _, pull = jax.vjp(functools.partial(gemm.ternary_weight_product, weights=empty,
                                        quantize_activations=False, out_dtype=jnp.float32), x)
    (nothing,) = pull(jnp.zeros((16, 0), jnp.float32))
    np.testing.assert_array_equal(np.asarray(nothing), np.zeros((16, 256), np.float32))


def test_a_packed_products_input_gradient_differentiates_in_its_scale():
    """A packed ternary product's input gradient, `(cotangent @ codesᵀ) * scale`, has the plain transpose's
    derivative in its scale, `cotangent @ codesᵀ`, at a scale of 3, at 2**126, where cotangents against
    +1 and -1 codes cancel past fp32's largest, and at an infinite one, the cotangent held still meeting no
    term of its own (row 1464)."""
    rng = np.random.default_rng(14640)
    codes = jnp.asarray(rng.integers(-1, 2, (256, 32)), jnp.float32).at[0, 0].set(1.0).at[0, 1].set(-1.0)
    weights = formats.TernaryWeights.of(codes, scale=jnp.float32(1.0))
    x = jnp.asarray(rng.integers(-3, 4, (16, 256)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-3, 4, (16, 32)), jnp.float32).at[0, :2].set(4.0)

    def input_gradient(scale):
        scaled = dataclasses.replace(weights, scale=scale)
        product = functools.partial(gemm.ternary_weight_product, weights=scaled, quantize_activations=False,
                                    out_dtype=jnp.float32)
        return jax.vjp(product, x)[1](cotangent)[0]

    want = np.asarray(cotangent, np.float64) @ np.asarray(codes, np.float64).T
    for scale in (3.0, 2.0 ** 126, float("inf")):
        _, derivative = jax.jvp(input_gradient, (jnp.float32(scale),), (jnp.float32(1.0),))
        np.testing.assert_array_equal(np.asarray(derivative), want.astype(np.float32), err_msg=str(scale))


def test_swapped_register_decode_holds_for_ternary_at_skinny_rows():
    """Skinny decode products (`M=13`) for `TernaryWeights` decode directly into `wgmma` register operand A
    (`packed_rows=True`, zero SMEM staging), matching the decoded fp32 dot bitwise under forced `mosaic`,
    rejecting `splits=2`, and compiling spill-free on sm90 (row 5387)."""
    from dew.nn.kernels import tuning
    from dew.nn.kernels.templates.gemm import ternary, tiling

    rng = np.random.default_rng(5387)
    x = jnp.asarray(rng.integers(-3, 4, size=(13, 256)), jnp.bfloat16)
    w_raw = jnp.asarray(rng.integers(-2, 3, size=(256, 128)), jnp.float32)
    with generation.assumed("sm90"):
        sm90 = generation.gpu_target("sm90")
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
        if generation.compile_only_gpus("sm90"):
            x_s = jax.ShapeDtypeStruct((13, 256), jnp.bfloat16)
            t_spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype), tern_w)

            def c_tern(a, w):
                with generation.chosen(grouped_matmul="mosaic"), tuning.pinned(
                        {ternary.MosaicTernaryPackedGmm: swapped_t}):
                    return gemm.ternary_weight_product(a, w, quantize_activations=False,
                                                       out_dtype=jnp.bfloat16)

            records = compile_mosaic_kernels(
                [KernelCase("swapped_ternary_sm90", lambda: (c_tern, (x_s, t_spec)))], "sm90")
            for rec in records:
                assert not rec.get("error"), rec
                assert rec["kernels"], rec
                assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


@pytest.mark.mesh(devices=2)
def test_a_packed_products_rescue_keeps_an_explicit_meshs_row_placement():
    """A ternary product under an Explicit mesh, its rows split over two CPU devices and its codes whole:
    cotangents of 4 and 4 against codes +1 and -1 at a scale of 2**126 pull back to 0 through the rescue,
    and 4 and 2 to 2**127, the gradient placed as the rows are (row 1464)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    codes = jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0).at[0, 1].set(-1.0)
    weights = dataclasses.replace(formats.TernaryWeights.of(codes, scale=jnp.float32(1.0)),
                                  scale=jnp.float32(2.0 ** 126))
    with jax.set_mesh(mesh):
        rows = jax.device_put(jnp.zeros((16, 256), jnp.float32).at[0, 0].set(1.0),
                              NamedSharding(mesh, PartitionSpec("data")))
        pull = jax.jit(lambda r, dy: jax.vjp(lambda x: gemm.ternary_weight_product(
            x, weights, quantize_activations=False, out_dtype=jnp.float32), r)[1](dy)[0])
        for second, expected in ((4.0, 0.0), (2.0, 2.0 ** 127)):
            dy = jax.device_put(jnp.zeros((16, 64), jnp.float32).at[0, 0].set(4.0).at[0, 1].set(second),
                                NamedSharding(mesh, PartitionSpec("data")))
            d_rows = pull(rows, dy)
            assert d_rows.sharding.spec[0] == "data", d_rows.sharding
            want = np.zeros((16, 256), np.float32)
            want[0, 0] = expected
            np.testing.assert_array_equal(np.asarray(d_rows), want)


def test_a_packed_products_rescue_keeps_a_subnormal_remainder():
    """Codes +1, -1 and +1 at a scale of 4 under cotangents 2**126, 2**126 and 2**-149: the first two
    cancel past fp32's largest and the remainder, 2**-147, is subnormal; the rescue rounds it in integers,
    so a process whose float32 arithmetic flushes subnormals still returns it (rows 1464, 12403)."""
    codes = jnp.zeros((256, 64), jnp.float32).at[0, 0].set(1.0).at[0, 1].set(-1.0).at[0, 2].set(1.0)
    weights = dataclasses.replace(formats.TernaryWeights.of(codes, scale=jnp.float32(1.0)),
                                  scale=jnp.float32(4.0))
    _, pull = jax.vjp(lambda x: gemm.ternary_weight_product(x, weights, quantize_activations=False,
                                                            out_dtype=jnp.float32),
                      jnp.zeros((16, 256), jnp.float32))
    cotangent = jnp.zeros((16, 64), jnp.float32).at[0, :3].set(
        jnp.asarray([2.0 ** 126, 2.0 ** 126, 2.0 ** -149], jnp.float32))
    (d_rows,) = pull(cotangent)
    want = np.zeros((16, 256), np.float32)
    want[0, 0] = np.float32(2.0 ** -147)
    np.testing.assert_array_equal(np.asarray(d_rows).view(np.uint32), want.view(np.uint32))


@pytest.mark.mesh(devices=2)
def test_a_packed_products_rescue_is_exact_within_a_row_and_across_replicas():
    """A ternary product's rescue sums its products exactly before one rounding (rows 12392, 12403): within
    one row, codes +1, -1, +1 under cotangents 2**39, 2**39, 2**-1 at a scale of 2**126 leave 2**125 of
    2**165 - 2**165 + 2**125; across two replicas of a checked `shard_map` (the rows and codes +1, -1 every
    device's, each its own scale and cotangent), 2**126 under 4, 4 beside 2**100 under 1, 0 leaves 2**100,
    and 2**126 under 4, 2**-26 beside 2**100 under -2**28, 0 leaves -2**100."""
    rows = jnp.zeros((16, 256), jnp.float32).at[0, 0].set(1.0)
    three = jnp.zeros((256, 32), jnp.float32).at[0, :3].set(jnp.asarray([1.0, -1.0, 1.0]))
    within = dataclasses.replace(formats.TernaryWeights.of(three, scale=jnp.float32(1.0)),
                                 scale=jnp.float32(2.0 ** 126))
    product = functools.partial(gemm.ternary_weight_product, weights=within, quantize_activations=False,
                                out_dtype=jnp.float32)
    cotangent = jnp.zeros((16, 32), jnp.float32).at[0, :3].set(jnp.asarray([2.0 ** 39, 2.0 ** 39, 0.5]))
    (d_rows,) = jax.vjp(product, rows)[1](cotangent)
    want = np.zeros((16, 256), np.float32)
    want[0, 0] = 2.0 ** 125
    np.testing.assert_array_equal(np.asarray(d_rows), want)

    codes = jnp.zeros((256, 32), jnp.float32).at[0, 0].set(1.0).at[0, 1].set(-1.0)
    weights = formats.TernaryWeights.of(codes, scale=jnp.float32(1.0))
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def each_device(x, scale, dy):
        scaled = dataclasses.replace(weights, scale=scale[0])
        replica = functools.partial(gemm.ternary_weight_product, weights=scaled, quantize_activations=False,
                                    out_dtype=jnp.float32)
        return jax.vjp(replica, x)[1](dy)[0]

    mapped = jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(whole, data, data), out_specs=whole,
                                   check_vma=True))
    scales = jnp.asarray([2.0 ** 126, 2.0 ** 100], jnp.float32)
    for first, second, expected in (((4.0, 4.0), (1.0, 0.0), 2.0 ** 100),
                                    ((4.0, 2.0 ** -26), (-(2.0 ** 28), 0.0), -(2.0 ** 100))):
        dy = jnp.zeros((32, 32), jnp.float32).at[0, :2].set(jnp.asarray(first)).at[16, :2].set(
            jnp.asarray(second))
        want = np.zeros((16, 256), np.float32)
        want[0, 0] = expected
        np.testing.assert_array_equal(np.asarray(mapped(rows, scales, dy)), want)


@pytest.mark.parametrize("use_bias", [False, True], ids=["no bias", "bias"])
def test_a_dense_layer_serves_a_packed_ternary_kernel_as_its_ternary_product(use_bias):
    """A Dense layer whose `quantization` collection holds a packed BitNet b1.58 kernel
    (`formats.TernaryWeights`, a checkpoint's weight served as its codes, row 1464) holds no kernel parameter
    and computes BitNet's product of it plus its bias, rounded once to its dtype, bitwise: its inputs
    quantized a token at a time to int8, times the unpacked {-1, 0, 1} codes, summed exactly, then the
    scales; its scale frozen (no cotangent), its inputs' straight through."""
    from dew.nn.dense import QUANTIZATION, Dense

    rng = np.random.default_rng(1464)
    codes = rng.integers(-127, 128, size=(16, 256)).astype(np.float32)
    codes[:, 0] = 127
    values = codes * 2.0 ** -7
    # One entry off the int8 grid (2^-9 beside a peak of 127 * 2^-7): its code is zero, which an
    # unquantized product would not read.
    values[0, 1] = 2.0 ** -9
    codes[0, 1] = 0
    x = jnp.asarray(values.reshape(2, 8, 256), jnp.bfloat16)
    weights = formats.TernaryWeights.of(jnp.asarray(rng.standard_normal((256, 64)), jnp.float32), scale=0.5)
    bias = jnp.asarray(rng.standard_normal((64,)) * 4, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32)
    layer = Dense(64, dtype=jnp.bfloat16, use_bias=use_bias)

    def run(x, weights):
        variables = {QUANTIZATION: {"kernel": weights}, **({"params": {"bias": bias}} if use_bias else {})}
        return layer.apply(variables, x)

    out = jax.jit(run)(x, weights)
    # BitNet's arithmetic written out (as `test_a_ternary_weight_product_quantizes_its_activations_as_...`):
    # each row's largest is 127 * 2^-7, so its int8 codes are `codes`; their exact products with the
    # unpacked codes, then the row's largest over 127 and the matrix's scale.
    exact = jax.lax.dot(jnp.asarray(codes, jnp.bfloat16), weights.unscaled(jnp.bfloat16),
                        preferred_element_type=jnp.float32)
    summed = precision.twice_scaled(precision.divided(exact, jnp.float32(127.0)),
                                    jnp.full((16, 1), 127 * 2.0 ** -7, jnp.float32), jnp.float32(0.5))
    expected = (summed + (bias if use_bias else 0)).astype(jnp.bfloat16).reshape(2, 8, 64)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
    _, pull = jax.vjp(run, x, weights)
    d_x, d_weights = pull(jnp.ones((2, 8, 64), jnp.bfloat16))
    assert not np.any(np.asarray(d_weights.scale)) and np.any(np.asarray(d_x, np.float32))


def test_a_dense_layer_of_no_depth_serves_its_packed_ternary_kernel():
    """A Dense layer over inputs of no width and a packed ternary kernel of no depth (`[0, 64]`) is its
    bias: the empty contraction's zeros (row 1464's review: its rows inferred from no width failed)."""
    from dew.nn.dense import QUANTIZATION, Dense

    weights = formats.TernaryWeights.of(jnp.zeros((0, 64), jnp.float32))
    bias = jnp.arange(64, dtype=jnp.float32)
    out = Dense(64, dtype=jnp.float32).apply({QUANTIZATION: {"kernel": weights}, "params": {"bias": bias}},
                                             jnp.zeros((2, 3, 0), jnp.float32))
    np.testing.assert_array_equal(np.asarray(out), np.broadcast_to(np.asarray(bias), (2, 3, 64)))
