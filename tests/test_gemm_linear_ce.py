"""Check-1 accuracy, normalizer (`peak`/`mass`/`rest`) parity, gating, sm80/sm90 spill-free compilation,
and TPU derivation tests for the `MOSAIC_LINEAR_CE` kernel arm of `dew.objectives.lm.chunked` (rows 6103,
12136, 12139)."""

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
from linear_ce_matrix import dense_truth
from reference_error import assert_as_exact, seeds_for

from dew.nn.kernels import generation, selection
from dew.nn.kernels.templates.gemm import linear_ce
from dew.nn.protocols import OutputTable
from dew.objectives.lm import chunked


@pytest.mark.parametrize(("m", "k", "v", "with_bias"), [(64, 64, 128, False), (70, 64, 140, True)])
def test_chunked_linear_ce_mosaic_arm_holds_check_1_value_vjp_and_normalizer(
    m: int, k: int, v: int, with_bias: bool
):
    """`MOSAIC_LINEAR_CE` arm of `chunked.chunked_cross_entropy` (and `chunked._forward` `peak`/`mass`/`rest`)
    holds check 1 on loss, log_z, gradients, and normalizer terms against the chunked portable path from
    float64 truth at `seeds_for`'s count (rows 6103, 12136)."""
    def draw(seed):
        rng = np.random.default_rng(6103 + seed)
        x = jnp.asarray(rng.standard_normal((m, k)) / np.sqrt(k), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((v, k)) / np.sqrt(k), jnp.bfloat16)
        targets = jnp.asarray(rng.integers(0, v, size=(m,)), jnp.int32)
        bias = jnp.asarray(rng.standard_normal((v,)) * 0.25, jnp.float32) if with_bias else None
        cot = jnp.asarray(rng.standard_normal((m,)), jnp.float32)
        return x, w, targets, bias, cot

    def _run(kernel: str, x, w, targets, bias, cot):
        with generation.assumed("sm90"), generation.chosen(linear_ce=kernel):
            def fwd(a, b, c):
                loss, _, log_z = chunked.chunked_cross_entropy(
                    a, OutputTable(b, vocab_major=True, bias=c, precision=chunked.BF16), targets,
                    chunks=2, tile=(64, 64), predict=False)
                return loss, log_z
            (loss, log_z), pull = jax.vjp(fwd, x, w, bias)
            grads = pull((cot, jnp.zeros_like(cot)))
            cols = chunked._Columns(bias, None)
            _, (peak, mass, rest) = chunked._forward(
                x, w, targets, None, 2, (64, 64), None,
                chunked._head_product(OutputTable(w, vocab_major=True, bias=bias, precision=chunked.BF16), x),
                predict=False, temperature=1.0, columns=cols)
            return loss, log_z, grads, peak, mass, rest

    def run_truth(x, w, targets, bias, cot):
        def fwd(a, b, c):
            loss, log_z, *_ = dense_truth(a, b, targets, c)
            return loss, log_z

        (loss, log_z), pull = jax.vjp(fwd, x.astype(jnp.float64), w.astype(jnp.float64),
                                      None if bias is None else bias.astype(jnp.float64))
        grads = pull((cot.astype(jnp.float64), jnp.zeros_like(cot, jnp.float64)))
        _, _, peak, mass, rest = dense_truth(x, w, targets, bias)
        return loss, log_z, grads, peak, mass, rest

    dew_fn = jax.jit(functools.partial(_run, "mosaic"))
    ref_fn = jax.jit(functools.partial(_run, "xla"))
    label = f"chunked_linear_ce_mosaic m={m} k={k} v={v} bias={with_bias}"
    count = seeds_for(ref_fn, draw, label, truth_fn=run_truth, dew_fn=dew_fn)
    assert_as_exact(dew_fn, ref_fn, draw, label, seeds=count, truth_fn=run_truth)

    x0, w0, t0, b0, _ = draw(0)
    with (generation.assumed("sm90"), generation.chosen(linear_ce="mosaic"),
          generation.routes() as seen):
        chunked.chunked_cross_entropy(x0, OutputTable(w0, vocab_major=True, bias=b0, precision=chunked.BF16),
                                      t0, chunks=2, tile=(64, 64), predict=False)
    assert any(r.kernel == "mosaic" for r in seen), seen


def test_chunked_linear_ce_non_gated_cases_take_existing_path_untouched():
    """Cases outside `MOSAIC_LINEAR_CE`'s gate (`predict=True`, `excluded`, `temperature!=1`, scaled
    columns, `softcap`, or unselected default) take `chunked._forward`'s existing XLA path untouched and
    stay bit-equal to `linear_ce='xla'`; an fp32 head, a dtype the kernel takes none of, does so unpinned and
    refuses a `linear_ce='mosaic'` pin (rows 12136, 12457)."""
    rng = np.random.default_rng(12136)
    x = jnp.asarray(rng.standard_normal((64, 64)) / 8.0, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((128, 64)) / 8.0, jnp.bfloat16)
    t = jnp.asarray(rng.integers(0, 128, size=(64,)), jnp.int32)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, size=(128,)), jnp.float32)

    cases = [
        ("predict", x, OutputTable(w, vocab_major=True, precision=chunked.BF16), {"predict": True}),
        ("excluded", x, OutputTable(w, vocab_major=True, precision=chunked.BF16),
         {"predict": False, "excluded": 0}),
        ("temperature", x, OutputTable(w, vocab_major=True, precision=chunked.BF16),
         {"predict": False, "temperature": 2.0}),
        ("scaled", x, OutputTable(w, vocab_major=True, scale=scale, precision=chunked.BF16),
         {"predict": False}),
        ("softcap", x, OutputTable(w, vocab_major=True, softcap=20.0, precision=chunked.BF16),
         {"predict": False}),
    ]
    for name, hidden, head, kwargs in cases:
        with generation.assumed("sm90"), generation.chosen(linear_ce="xla"):
            ref_loss, _, ref_log_z = chunked.chunked_cross_entropy(
                hidden, head, t, chunks=2, tile=(64, 64), **kwargs)
        with (generation.assumed("sm90"), generation.chosen(linear_ce="mosaic"),
              generation.routes() as seen):
            got_loss, _, got_log_z = chunked.chunked_cross_entropy(
                hidden, head, t, chunks=2, tile=(64, 64), **kwargs)
        assert not any(r.kernel == "mosaic" for r in seen), (name, seen)
        np.testing.assert_array_equal(np.asarray(got_loss), np.asarray(ref_loss))
        np.testing.assert_array_equal(np.asarray(got_log_z), np.asarray(ref_log_z))
    wide, wide_head = x.astype(jnp.float32), OutputTable(w.astype(jnp.float32), vocab_major=True)
    with generation.assumed("sm90"), generation.chosen(linear_ce="mosaic"), pytest.raises(
            selection.KernelRefused):
        chunked.chunked_cross_entropy(wide, wide_head, t, chunks=2, tile=(64, 64), predict=False)
    with generation.assumed("sm90"), generation.chosen(linear_ce="xla"):
        ref_wide, _, _ = chunked.chunked_cross_entropy(wide, wide_head, t, chunks=2, tile=(64, 64),
                                                       predict=False)
    with generation.assumed("sm90"), generation.routes() as seen:
        got_wide, _, _ = chunked.chunked_cross_entropy(wide, wide_head, t, chunks=2, tile=(64, 64),
                                                       predict=False)
    assert not any(r.kernel == "mosaic" for r in seen), seen
    np.testing.assert_array_equal(np.asarray(got_wide), np.asarray(ref_wide))
    # Default unselected mode also stays on XLA untouched.
    with generation.assumed("sm90"), generation.routes() as seen:
        got_def, _, _ = chunked.chunked_cross_entropy(x, OutputTable(w, vocab_major=True), t,
                                                      chunks=2, tile=(64, 64), predict=False)
    assert not any(r.kernel == "mosaic" for r in seen), seen
    with generation.assumed("sm90"), generation.chosen(linear_ce="xla"):
        ref_def, _, _ = chunked.chunked_cross_entropy(x, OutputTable(w, vocab_major=True), t,
                                                      chunks=2, tile=(64, 64), predict=False)
    np.testing.assert_array_equal(np.asarray(got_def), np.asarray(ref_def))


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_cross_entropy_compiles_spill_free(gpu: str, with_bias: bool):
    """`chunked.chunked_cross_entropy(..., predict=False)` compiles spill-free through `MOSAIC_LINEAR_CE`
    on sm80 and sm90 (rows 6103, 12136)."""
    x = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((512, 256), jnp.bfloat16)
    t = jax.ShapeDtypeStruct((256,), jnp.int32)
    b = jax.ShapeDtypeStruct((512,), jnp.float32) if with_bias else None

    def fn(a, weight, targets, *rest):
        bias = rest[0] if rest else None
        with generation.chosen(linear_ce="mosaic"):
            loss, _, _ = chunked.chunked_cross_entropy(
                a, OutputTable(weight, vocab_major=True, bias=bias, precision=chunked.BF16), targets,
                chunks=2, tile=(256, 256), predict=False)
            return loss

    args = (x, w, t, b) if with_bias else (x, w, t)
    with generation.assumed(gpu), generation.chosen(linear_ce="mosaic"):
        records = compile_mosaic_kernels([KernelCase(f"linear_ce_{gpu}_bias{with_bias}",
                                                     lambda: (fn, args))], gpu)
    for rec in records:
        assert not rec.get("error"), rec
        assert any(k["kernel"] == linear_ce.LINEAR_CE for k in rec["kernels"]), rec
        assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


def test_linear_cross_entropy_mixed_dtype_refuses_native():
    """`MOSAIC_LINEAR_CE` refuses mismatched bf16/fp32 operands across eligibility, direct call, and
    `LINEAR_CE_OP` selection (row 6103)."""
    with generation.assumed("sm90"):
        ba = linear_ce.MOSAIC_LINEAR_CE.bind(
            jax.ShapeDtypeStruct((64, 64), jnp.bfloat16), jax.ShapeDtypeStruct((64, 64), jnp.float32),
            jax.ShapeDtypeStruct((64,), jnp.int32), None, transpose_rhs=True, softcap=None, interpret=True)
        assert not linear_ce.MOSAIC_LINEAR_CE.eligible(ba, generation.gpu_target())
        tile = linear_ce.derive_linear_ce((64, 64, 64), jnp.bfloat16, generation.gpu_target(), interpret=True)
        assert tile is not None
        with pytest.raises(ValueError):
            linear_ce.linear_ce(jnp.zeros((64, 64), jnp.bfloat16), jnp.zeros((64, 64), jnp.float32),
                                jnp.zeros((64,), jnp.int32), tiles=tile, multiprocessors=1, interpret=True)
        with generation.chosen(linear_ce="mosaic"), pytest.raises(selection.KernelRefused):
            linear_ce.mosaic_linear_ce(jnp.zeros((64, 64), jnp.bfloat16), jnp.zeros((64, 64), jnp.float32),
                                       jnp.zeros((64,), jnp.int32))


def test_linear_cross_entropy_derives_on_tpu_and_honors_shared_limit():
    """`linear_ce.derive_linear_ce` derives a legal MXU tile on TPU v5e, and `derivation.derived` honors a
    tightened `GpuTarget.shared_limit` returning None when no tile fits (rows 6103, 1762, 12139)."""
    from dew.nn.kernels.templates import bundles
    from dew.nn.kernels.templates.gemm import derivation

    with generation.assumed("v5e"):
        tile = linear_ce.derive_linear_ce((256, 4096, 256), jnp.bfloat16, generation.gpu_target())
        assert tile is not None and tile.family == "mxu"
    with generation.assumed("sm90"):
        tight = dataclasses.replace(generation.gpu_target(), shared_limit=1024)
        layouts = bundles.groups("sm90", jnp.bfloat16, stated=True)[0]
        clamped = derivation.derived(
            layouts, rows=256, depth=256, width=256, dtype=jnp.bfloat16, multiprocessors=132,
            target=tight, candidate=lambda c: True,
            launch_of=lambda c: linear_ce.linear_ce_launch_of(
                c, (256, 256, 256), jnp.bfloat16, transpose_rhs=True, has_bias=False, target=tight),
            grouped=False, dense=True)
        assert clamped is None


def test_linear_ce_rounds_bf16_logits_matches_head_law_and_ignores_unrelated_choices():
    """Fused BF16 `MOSAIC_LINEAR_CE` rounds each completed logit to bf16 before its softmax, as the chunked
    head's product does forward and backward (`chunked._tile_logits`, `bf16_product`): logits near 128,
    where bf16's places are 0.5 below and 1 above, hold check 1 on loss, log Z and gradients against
    float64 truth of the rounded logits (a softmax of the unrounded ones is 0.25 off where a target's 128.25
    rounds to 128). Naming the grouped products' kernel (`grouped_matmul`: `mosaic`, `cutlass`, `triton`)
    leaves chunked XLA untouched, bit-equal: the fused loss is its own choice (rows 6103, 12136, 12457)."""
    m, k, v = 64, 64, 64

    def draw(seed):
        rng = np.random.default_rng(12136 + seed)
        x = jnp.zeros((m, k), jnp.float32).at[:, 0].set(1.0).at[:, 1].set(rng.integers(-3, 4, m) / 4.0)
        w = jnp.zeros((v, k), jnp.float32).at[:, 0].set(128.0).at[:, 1].set(rng.integers(-1, 2, v))
        targets = jnp.asarray(rng.integers(0, v, size=(m,)), jnp.int32)
        cot = jnp.asarray(rng.standard_normal((m,)), jnp.float32)
        return x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), targets, cot

    def run(kernel: str, x, w, targets, cot):
        with generation.assumed("sm90"), generation.chosen(linear_ce=kernel):
            def fwd(a, b):
                loss, _, log_z = chunked.chunked_cross_entropy(
                    a, OutputTable(b, vocab_major=True, precision=chunked.BF16), targets,
                    chunks=1, tile=(64, 64), predict=False)
                return loss, log_z
            (loss, log_z), pull = jax.vjp(fwd, x, w)
            return loss, log_z, pull((cot, jnp.zeros_like(cot)))

    def truth(x, w, targets, cot):
        def fwd(a, b):
            logits = a @ b.T
            # Each logit rounded to bf16 as the head's product rounds it, its derivative the product's.
            logits = logits + jax.lax.stop_gradient(logits.astype(jnp.bfloat16).astype(jnp.float64) - logits)
            log_z = jax.nn.logsumexp(logits, axis=1)
            return log_z - jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0], log_z

        (loss, log_z), pull = jax.vjp(fwd, x.astype(jnp.float64), w.astype(jnp.float64))
        return loss, log_z, pull((cot.astype(jnp.float64), jnp.zeros_like(cot, jnp.float64)))

    dew_fn, ref_fn = (jax.jit(functools.partial(run, kernel)) for kernel in ("mosaic", "xla"))
    label = "linear_ce bf16 logits near 128"
    assert_as_exact(dew_fn, ref_fn, draw, label, seeds=seeds_for(ref_fn, draw, label, truth_fn=truth,
                                                                  dew_fn=dew_fn), truth_fn=truth)

    x, w, targets, _ = draw(0)
    head = OutputTable(w, vocab_major=True, precision=chunked.BF16)
    with generation.assumed("sm90"), generation.chosen(linear_ce="xla"):
        want_loss, _, want_logz = chunked.chunked_cross_entropy(x, head, targets, chunks=1, tile=(64, 64),
                                                                predict=False)
    for other in ("mosaic", "cutlass", "triton"):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul=other):
            got_loss, _, got_logz = chunked.chunked_cross_entropy(
                x, head, targets, chunks=1, tile=(64, 64), predict=False)
        np.testing.assert_array_equal(np.asarray(got_loss), np.asarray(want_loss))
        np.testing.assert_array_equal(np.asarray(got_logz), np.asarray(want_logz))


def test_a_capped_linear_cross_entropy_launch_is_on_par_with_its_portable_cap():
    """The fused linear cross entropy's launch at a softcap of 20, interpreted as an H100's, on par (check 1)
    with the portable arm's loss of its capped logits (`precision.capped_logits`), each from an independent
    float64 cross entropy of `cap * tanh(z / cap)` (row 13353)."""
    from dew.nn.precision import capped_logits

    rows, depth, vocab = 256, 128, 1024  # a loss a row: enough rows a seed that the gate resolves its ratio

    def draw(seed):
        rng = np.random.default_rng(13353 + seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(vocab, depth)), jnp.bfloat16),
                jnp.asarray(rng.integers(0, vocab, size=(rows,)), jnp.int32))

    def fused(x, w, t):
        with generation.assumed("sm90"):
            tile = linear_ce.derive_linear_ce((rows, vocab, depth), jnp.bfloat16, generation.gpu_target(),
                                              interpret=True)
            loss, *_ = linear_ce.linear_ce(x, w, t, tiles=tile, softcap=20.0, multiprocessors=1,
                                           interpret=True)
            return loss

    def portable(x, w, t):
        capped = capped_logits(jnp.dot(x.astype(jnp.float32), w.astype(jnp.float32).T), 20.0)
        return jax.nn.logsumexp(capped, axis=1) - capped[jnp.arange(rows), t]

    def truth(x, w, t):
        # Its operands widened to float64 by the gate (`assert_as_exact`): the cap's quotient exact there.
        capped = 20.0 * jnp.tanh(jnp.dot(x, w.T) / 20.0)
        return jax.nn.logsumexp(capped, axis=1) - capped[jnp.arange(rows), t]

    label = "capped linear_ce"
    dew_fn, ref_fn = jax.jit(fused), jax.jit(portable)
    assert_as_exact(dew_fn, ref_fn, draw, label, seeds=seeds_for(ref_fn, draw, label, truth_fn=truth,
                                                                  dew_fn=dew_fn), truth_fn=truth)


def test_a_capped_logit_of_zero_at_a_cap_past_its_reciprocals_range_is_zero():
    """`precision.capped_logits` of a zero logit at a runtime fp32 cap of 2^-130 (its reciprocal past fp32's
    largest) is zero, its quotient correctly rounded (row 13353's review), where a reciprocal's product
    reads 0 * inf, NaN."""
    from dew.nn.precision import capped_logits

    out = jax.jit(capped_logits)(jnp.zeros((2,), jnp.float32), jnp.asarray([2.0 ** -130, 1.0], jnp.float32))
    np.testing.assert_array_equal(np.asarray(out), np.zeros((2,), np.float32))
