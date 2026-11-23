"""Check-1 accuracy, stage decode, BitNet b1.58 activation quantization + STE, sm80/sm90 compilation, and
TF32 grouped/dense Mosaic GEMM tests (`gemm.ternary_weight_product`, `formats.TernaryWeights`, rows 1464,
3428)."""

from __future__ import annotations

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
            got = gemm.ternary_weight_product(x_g, bank, sizes, quantize_activations=False,
                                              out_dtype=jnp.float32, interpret_on_cpu=interpret)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

    # BitNet b1.58 per-token int8 activation quantization + zero-row safety + STE gradient.
    x_bit = jnp.asarray(rng.standard_normal((32, 256)), jnp.bfloat16).at[0].set(0.0)
    w_bit = formats.TernaryWeights.of(jnp.asarray(rng.standard_normal((256, 64)), jnp.float32))
    absmax = jnp.max(jnp.abs(x_bit.astype(jnp.float32)), axis=-1)
    act_scale = jnp.where(absmax > 0, precision.divided(absmax, jnp.float32(127.0)), jnp.float32(1.0))
    x_q = jnp.clip(jnp.round(x_bit.astype(jnp.float32) / act_scale[:, None]), -128.0, 127.0)
    ref_bit = jax.lax.dot(x_q, w_bit.unscaled(jnp.float32), preferred_element_type=jnp.float32) * (
        act_scale[:, None] * w_bit.scale)
    for interpret in (False, True):
        got_bit = gemm.ternary_weight_product(x_bit, w_bit, quantize_activations=True,
                                              out_dtype=jnp.float32, interpret_on_cpu=interpret)
        np.testing.assert_array_equal(np.asarray(got_bit[0]), np.zeros((64,), np.float32))
        _, dx = jax.value_and_grad(lambda a, interpret=interpret: jnp.sum(gemm.ternary_weight_product(
            a, w_bit, quantize_activations=True, out_dtype=jnp.float32, interpret_on_cpu=interpret)))(x_bit)
        assert np.all(np.isfinite(np.asarray(dx)))
        np.testing.assert_array_equal(np.asarray(got_bit), np.asarray(ref_bit))
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
        if generation.compile_only_gpus(gpu):
            def run_tf32(x, w, s):
                with generation.chosen(grouped_matmul="mosaic"):
                    return gemm.gemm(x, w, s, transpose_rhs=(gpu == "sm90"), out_dtype=jnp.float32,
                                     precision=jax.lax.Precision.DEFAULT)

            records = compile_mosaic_kernels(
                [KernelCase(f"tf32_grouped_{gpu}",
                            lambda: (run_tf32, (f32((256, 128)), f32((2, 128, 128)), sizes)))], gpu)
            for rec in records:
                assert not rec.get("error"), rec
                assert rec["kernels"], rec
                assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec
