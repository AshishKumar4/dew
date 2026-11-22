"""Check-1 accuracy, Barron derivative, sm80/sm90 spill-free compilation, and TPU derivation tests for
GEMM-fused linear cross-entropy (`gemm.linear_cross_entropy`, `templates.gemm.linear_ce`, row 6103)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import KernelCase
from generation_support import lowers
from kernel_compile import compile_mosaic_kernels, lowered_for
from reference_error import assert_as_exact

from dew.nn import categorical
from dew.nn.kernels import gemm, generation
from dew.nn.kernels.templates import report
from dew.nn.kernels.templates.gemm import linear_ce

SEEDS = 8


def _dense_ref(x, weight, targets, bias, *, transpose_rhs: bool, softcap: float | None, dtype):
    xw = x.astype(dtype)
    ww = weight.astype(dtype)
    dims = (((1,), (1 if transpose_rhs else 0,)), ((), ()))
    logits = jax.lax.dot_general(xw, ww, dims, preferred_element_type=dtype)
    if bias is not None:
        logits = logits + bias.astype(dtype)[None, :]
    if softcap is not None:
        cap = jnp.asarray(softcap, dtype)
        logits = cap * jnp.tanh(logits / cap)
    return categorical.cross_entropy(logits, targets)


@pytest.mark.parametrize(
    ("m", "k", "v", "transpose_rhs", "with_bias", "softcap", "interpret"),
    [
        (19, 64, 150, True, False, None, False),
        (19, 64, 150, False, True, 30.0, False),
        (64, 64, 128, True, False, None, True),
        (70, 64, 140, True, True, 20.0, True),
    ],
)
def test_linear_cross_entropy_holds_check_1_value_and_vjp(
    m: int, k: int, v: int, transpose_rhs: bool, with_bias: bool, softcap: float | None, interpret: bool
):
    """`gemm.linear_cross_entropy` (chunked portable and interpreted Mosaic) forward and VJP hold check 1
    against dense `categorical.cross_entropy` from float64 truth across aligned and ragged shapes, bias, and
    softcap (row 6103)."""
    w_shape = (v, k) if transpose_rhs else (k, v)

    def draw(seed):
        rng = np.random.default_rng(6103 + seed)
        x = jnp.asarray(rng.standard_normal((m, k)) / np.sqrt(k), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal(w_shape) / np.sqrt(k), jnp.bfloat16)
        targets = jnp.asarray(rng.integers(0, v, size=(m,)), jnp.int32)
        bias = jnp.asarray(rng.standard_normal((v,)) * 0.25, jnp.float32) if with_bias else None
        cot = jnp.asarray(rng.standard_normal((m,)), jnp.float32)
        return x, w, targets, bias, cot

    def run_dew(x, w, targets, bias, cot):
        def fwd(a, b, c):
            return gemm.linear_cross_entropy(
                a, b, targets, bias=c, softcap=softcap, transpose_rhs=transpose_rhs,
                chunk_size=64, interpret_on_cpu=interpret)

        loss, pull = jax.vjp(fwd, x, w, bias)
        return loss, pull(cot)

    def run_ref(x, w, targets, bias, cot):
        def fwd(a, b, c):
            return _dense_ref(a, b, targets, c, transpose_rhs=transpose_rhs, softcap=softcap,
                              dtype=jnp.float32)

        loss, pull = jax.vjp(fwd, x, w, bias)
        return loss, pull(cot)

    def run_truth(x, w, targets, bias, cot):
        def fwd(a, b, c):
            return _dense_ref(a, b, targets, c, transpose_rhs=transpose_rhs, softcap=softcap,
                              dtype=jnp.float64)

        loss, pull = jax.vjp(fwd, x.astype(jnp.float64), w.astype(jnp.float64),
                             None if bias is None else bias.astype(jnp.float64))
        return loss, pull(cot.astype(jnp.float64))

    assert_as_exact(
        jax.jit(run_dew), jax.jit(run_ref), draw,
        f"linear_ce m={m} k={k} v={v} tr={transpose_rhs} bias={with_bias} cap={softcap} interp={interpret}",
        seeds=SEEDS, truth_fn=run_truth,
    )


def test_linear_cross_entropy_confident_logits_and_out_of_range_targets():
    """Confident logits keep non-zero, zero-sum Barron label derivatives (`categorical.label_entry`), and
    out-of-range targets take log-partition loss and softmax gradients matching `categorical` (row 6103)."""
    targets = jnp.asarray([0, -1, 99], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(61030 + seed)
        base = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]], np.float32)
        x = jnp.asarray(base + rng.normal(scale=0.05, size=(3, 2)), jnp.float32)
        w = jnp.asarray([[40.0, 0.0], [0.0, 40.0], [-10.0, -10.0], [0.0, 0.0]], jnp.float32)
        return x, w, jnp.ones((3,), jnp.float32)

    def dew_vjp(a, b, cot):
        loss, pull = jax.vjp(lambda x, w: gemm.linear_cross_entropy(x, w, targets, chunk_size=2), a, b)
        return loss, pull(cot)

    def ref_vjp(a, b, cot, *, dtype=jnp.float32):
        loss, pull = jax.vjp(lambda x, w: categorical.cross_entropy(x @ w.T, targets),
                             a.astype(dtype), b.astype(dtype))
        return loss, pull(cot.astype(dtype))

    assert_as_exact(jax.jit(dew_vjp), jax.jit(ref_vjp), draw,
                    "linear_ce confident + out-of-range targets", seeds=SEEDS,
                    truth_fn=functools.partial(ref_vjp, dtype=jnp.float64))
    # Row 0 has logit margin ~40 (p_0 == 1.0 in fp32), yet dW[0, 0] remains strictly negative and non-zero.
    x0, w0, cot0 = draw(0)
    _, (_, dw0) = dew_vjp(x0, w0, cot0)
    assert float(dw0[0, 0]) < 0.0


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_cross_entropy_compiles_spill_free(gpu: str, with_bias: bool):
    """`gemm.linear_cross_entropy` compiles spill-free through `MOSAIC_LINEAR_CE` on sm80 and sm90
    (row 6103)."""
    x = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((512, 256), jnp.bfloat16)
    t = jax.ShapeDtypeStruct((256,), jnp.int32)
    b = jax.ShapeDtypeStruct((512,), jnp.float32) if with_bias else None

    def fn(a, weight, targets, *rest):
        bias = rest[0] if rest else None
        return gemm.linear_cross_entropy(a, weight, targets, bias=bias, softcap=30.0)

    args = (x, w, t, b) if with_bias else (x, w, t)
    before = {id(c) for c in report.certifications()}
    with (report.refusals(), report.certifying(generation.compile_only_gpus(gpu)[0]),
          generation.chosen(grouped_matmul="mosaic")):
        records = compile_mosaic_kernels(
            [KernelCase(f"linear_ce_{gpu}_bias{with_bias}", lambda: lowered_for(fn, args, gpu))], gpu)
    for rec in records:
        assert not rec.get("error"), rec
        assert rec["kernels"], rec
        assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec
    assert any(id(c) not in before and c.op == type(linear_ce.MOSAIC_LINEAR_CE).__name__
               for c in report.certifications())


def test_linear_cross_entropy_derives_on_tpu():
    """`linear_ce.derive_linear_ce` derives a legal MXU tile on TPU v5e (row 6103)."""
    target = generation.gpu_target("v5e")
    with generation.assumed("v5e"):
        tile = linear_ce.derive_linear_ce((256, 1024, 256), jnp.bfloat16, target)
    assert tile is not None
    assert tile.family == "mxu"
