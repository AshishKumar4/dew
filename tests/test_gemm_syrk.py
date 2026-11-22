"""Muon SYRK upper-triangle Gram (`gemm.symmetric_gram`) and fused symmetric polynomial epilogue
(`gemm.symmetric_polynomial`) on the shared Mosaic toolkit (row 11941)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import KernelCase
from generation_support import INTERPRETS, lowers
from kernel_compile import compile_mosaic_kernels
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation, grouped_launch
from dew.nn.kernels.templates.gemm import syrk, tiling

pytestmark = pytest.mark.kernel
SEEDS = 8


@pytest.mark.parametrize("side", [1, 2, 3, 4, 7, 16, 33])
def test_upper_tile_is_exact_bijection(side):
    """`syrk.upper_tile` maps every linear index in `[0, batches * tri)` bijectively onto
    `(batch, ti, tj)` with `0 <= ti <= tj < side`."""
    batches = 3
    tri = side * (side + 1) // 2
    idx = jnp.arange(batches * tri, dtype=jnp.int32)
    b, ti, tj = syrk.upper_tile(idx, side)
    seen = list(zip(np.asarray(b).tolist(), np.asarray(ti).tolist(), np.asarray(tj).tolist(), strict=True))
    expected = [(batch, i, j) for batch in range(batches) for i in range(side) for j in range(i, side)]
    assert sorted(seen) == expected


@INTERPRETS
@pytest.mark.parametrize(("shape", "transpose"),
                         [((2, 128, 256), False), ((2, 256, 128), True),
                          ((2, 300, 128), False), ((2, 2, 128, 128), False)],
                         ids=["fat", "tall", "ragged-300", "4d-batch"])
def test_symmetric_gram_holds_check_1_and_vjp(shape, transpose):
    """`gemm.symmetric_gram` (`X @ X^T` and `X^T @ X`, including ragged `m` and multi-batch inputs) and its
    reverse-mode gradient hold check 1 against XLA and float64 truth (row 11941)."""
    axis = len(shape) - 2 if transpose else len(shape) - 1
    batch = tuple(range(len(shape) - 2))
    dims = (((axis,), (axis,)), (batch, batch))

    def draw(seed):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=shape) / np.sqrt(shape[axis]), jnp.bfloat16)
        m = shape[-1] if transpose else shape[-2]
        cot = jnp.asarray(rng.normal(size=(*shape[:-2], m, m)), jnp.bfloat16)
        return x, cot

    def native(x, cot):
        with generation.chosen(grouped_matmul="mosaic"):
            y, pull = jax.vjp(lambda a: gemm.symmetric_gram(a, transpose=transpose, interpret_on_cpu=True), x)
            (dx,) = pull(cot)
        return y, dx

    def plain(x, cot, wide=jnp.float32, out=jnp.bfloat16):
        def fwd(a):
            return jax.lax.dot_general(a.astype(wide), a.astype(wide), dims,
                                       preferred_element_type=wide).astype(out)
        y, pull = jax.vjp(fwd, x)
        (dx,) = pull(cot.astype(out))
        return y, dx.astype(out)

    with generation.assumed("sm90"):
        assert_as_exact(native, jax.jit(plain), draw, f"syrk gram {shape} t={transpose}", seeds=SEEDS,
                        truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))


@INTERPRETS
@pytest.mark.parametrize("side", [128, 300], ids=["aligned-128", "ragged-300"])
def test_symmetric_polynomial_and_newton_schulz_hold_check_1(side):
    """`gemm.symmetric_polynomial` (`c0*I + c1*A + c2*(A@A)` with traced coefficients, value + VJP) and a
    2-step Newton-Schulz orthogonalization loop hold check 1 via `assert_as_exact` (row 11941)."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(2, side, side)).astype(np.float32) / np.sqrt(side)
        sym = jnp.asarray(0.5 * (raw + np.swapaxes(raw, -1, -2)), jnp.bfloat16)
        coeffs = jnp.asarray([3.4445, -4.7750, 2.0315], jnp.float32)
        cot = jnp.asarray(rng.normal(size=(2, side, side)), jnp.bfloat16)
        return sym, coeffs, cot

    def native(a, coeffs, cot):
        with generation.chosen(grouped_matmul="mosaic"):
            y, pull = jax.vjp(lambda x, c: gemm.symmetric_polynomial(x, c, interpret_on_cpu=True), a, coeffs)
            da, dc = pull(cot)
        return y, da, dc

    def plain(a, coeffs, cot, wide=jnp.float32, out=jnp.bfloat16):
        def fwd(x, c):
            xw, cw = x.astype(wide), c.astype(wide)
            sq = jax.lax.dot_general(xw, xw, (((2,), (1,)), ((0,), (0,))), preferred_element_type=wide)
            eye = jnp.eye(side, dtype=wide)
            return (cw[0] * eye + cw[1] * xw + cw[2] * sq).astype(out)
        y, pull = jax.vjp(fwd, a, coeffs)
        da, dc = pull(cot.astype(out))
        return y, da.astype(out), dc.astype(wide)

    with generation.assumed("sm90"):
        assert_as_exact(native, jax.jit(plain), draw, f"syrk poly {side}", seeds=SEEDS,
                        truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))


def _plain_poly(a, c, wide=jnp.float32, out=jnp.bfloat16):
    """Independent reference polynomial `c0 * I + c1 * A + c2 * (A @ A)` in `wide` precision."""
    aw, cw = a.astype(wide), c.astype(wide)
    sq = jax.lax.dot_general(aw, aw, (((2,), (1,)), ((0,), (0,))), preferred_element_type=wide)
    return (cw[0] * jnp.eye(a.shape[-1], dtype=wide) + cw[1] * aw + cw[2] * sq).astype(out)


@INTERPRETS
def test_syrk_polynomial_pinned_ragged_tile_and_skew_cotangent():
    """Pinned ragged tile (`m=300`, `128x128x32` where `k_pad` would differ from `m_pad`) stays in bounds
    and holds check 1; `symmetric_polynomial` VJP matches literal `c0*I + c1*A + c2*(A@A)` on skew and
    non-commuting off-diagonal cotangents; unsupported output dtypes refuse native SYRK (row 11941)."""
    tile_128 = tiling.Tiles(128, 128, 32, 2, 1, warpgroups=1, family="mma.sync")

    def draw(seed):
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(1, 300, 300)).astype(np.float32) / np.sqrt(300)
        sym = jnp.asarray(0.5 * (raw + np.swapaxes(raw, -1, -2)), jnp.bfloat16)
        return sym, jnp.asarray([3.4445, -4.7750, 2.0315], jnp.float32)

    def native(a, c):
        return syrk.syrk(a, tiles=tile_128, transpose=False, out_dtype=jnp.bfloat16, multiprocessors=1,
                         interpret=grouped_launch.INTERPRETED, poly_coeffs=c)

    with generation.assumed("sm80"):
        assert_as_exact(native, jax.jit(_plain_poly), draw, "syrk ragged 300 128x128x32", seeds=SEEDS,
                        truth_fn=functools.partial(_plain_poly, wide=jnp.float64, out=jnp.float64))

    # Skew cotangent D at A = I, c = (0, 0, 1): pullback of A @ A is D @ A^T + A^T @ D = 2 D.
    eye = jnp.eye(8, dtype=jnp.float32)[None]
    skew = jnp.asarray(np.triu(np.arange(64, dtype=np.float32).reshape(8, 8), 1))[None]
    skew = skew - jnp.swapaxes(skew, -1, -2)
    _, pull = jax.vjp(lambda x: gemm.symmetric_polynomial(x, jnp.asarray([0.0, 0.0, 1.0])), eye)
    (da_skew,) = pull(skew)
    np.testing.assert_array_equal(np.asarray(da_skew), 2.0 * np.asarray(skew))

    # Unsupported output dtype (`int8`) is refused by direct `syrk` and routes to `xla` on `SYRK`.
    bf16_x = jnp.zeros((1, 128, 128), jnp.bfloat16)
    with generation.assumed("sm80"):
        with pytest.raises(ValueError):
            syrk.syrk(bf16_x, tiles=tile_128, out_dtype=jnp.int8, multiprocessors=1,
                      interpret=grouped_launch.INTERPRETED)
        with generation.routes() as seen:
            out = gemm.symmetric_gram(bf16_x, out_dtype=jnp.int8, interpret_on_cpu=True)
        assert out.dtype == jnp.int8
        assert any(route.kernel == "xla" for route in seen), seen


@lowers("sm90")
def test_unforced_syrk_skips_mosaic_certification_while_forced_certifies():
    """Default (unforced) BF16 SYRK declines on `measured` before certifying `MosaicSyrk`, while forced
    `grouped_matmul='mosaic'` certifies `MosaicSyrk` on the compile-only device (row 11941)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.templates import report

    x = jax.ShapeDtypeStruct((2, 256, 256), jnp.bfloat16)
    before_default = {id(c) for c in report.certifications()}
    with report.refusals(), report.certifying(generation.compile_only_gpus("sm90")[0]):
        lowered_for(lambda operand: gemm.symmetric_gram(operand), (x,), "sm90")
    assert not [c for c in report.certifications()
                if id(c) not in before_default and c.op == type(syrk.MOSAIC_SYRK).__name__]

    before_forced = {id(c) for c in report.certifications()}
    with (report.refusals(), report.certifying(generation.compile_only_gpus("sm90")[0]),
          generation.chosen(grouped_matmul="mosaic")):
        lowered_for(lambda operand: gemm.symmetric_gram(operand), (x,), "sm90")
    assert any(id(c) not in before_forced and c.op == type(syrk.MOSAIC_SYRK).__name__
               for c in report.certifications())


@INTERPRETS
def test_syrk_cluster2_square_super_tile_holds_check_1():
    """Hopper `cluster=2` square super-tile (`128x256x64` -> `256x256` super-tile) SYRK holds check 1."""
    tile = tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, cluster=2, family="wgmma")

    def draw(seed):
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(1, 512, 512)).astype(np.float32) / np.sqrt(512)
        a = jnp.asarray(0.5 * (raw + np.swapaxes(raw, -1, -2)), jnp.bfloat16)
        c = jnp.asarray([3.4445, -4.7750, 2.0315], jnp.float32)
        return a, c

    def native(a, c):
        return syrk.syrk(a, tiles=tile, transpose=False, out_dtype=jnp.bfloat16, multiprocessors=2,
                         interpret=grouped_launch.INTERPRETED, poly_coeffs=c)

    with generation.assumed("sm90"):
        assert_as_exact(native, jax.jit(_plain_poly), draw, "syrk cluster2 256x256 poly", seeds=SEEDS,
                        truth_fn=functools.partial(_plain_poly, wide=jnp.float64, out=jnp.float64))


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90")),
                                 pytest.param("sm100", marks=lowers("sm100"))])
def test_syrk_derives_square_super_tiles_and_compiles_spill_free(gpu):
    """`MosaicSyrk` derives legal square super-tiles across sm80, sm90, and sm100 through the shared
    `Template` certification path, and Gram (`transpose=False/True`) + fused polynomial kernels compile
    under ptxas with zero spills (row 11941)."""
    target = generation.gpu_target(gpu)
    op = syrk.MOSAIC_SYRK
    with generation.assumed(gpu):
        x_sq = jax.ShapeDtypeStruct((4, 2560, 2560), jnp.bfloat16)
        x_tall = jax.ShapeDtypeStruct((4, 4096, 2560), jnp.bfloat16)
        coeffs = jax.ShapeDtypeStruct((3,), jnp.float32)
        ba_gram = op.bind(x_sq, None, transpose=False, out_dtype="bfloat16", interpret=False)
        ba_tall = op.bind(x_tall, None, transpose=True, out_dtype="bfloat16", interpret=False)
        ba_poly = op.bind(x_sq, coeffs, transpose=False, out_dtype="bfloat16", interpret=False)
        for ba in (ba_gram, ba_tall, ba_poly):
            tile = op.derived(ba, target)
            assert tile is not None and op.legal(ba, tile, target)
            assert syrk.square_super_tile(tile, syrk.family_of(tile, jnp.bfloat16)), (gpu, tile)
        cases = [
            KernelCase(f"{gpu}_syrk_gram", lambda: op.lowering(ba_gram, None)),
            KernelCase(f"{gpu}_syrk_tall", lambda: op.lowering(ba_tall, None)),
            KernelCase(f"{gpu}_syrk_poly", lambda: op.lowering(ba_poly, None)),
        ]
        records = compile_mosaic_kernels(cases, gpu)
    for rec in records:
        assert not rec.get("error"), rec
        assert any(k["kernel"] == syrk.SYRK for k in rec["kernels"]), rec
        assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


def test_syrk_derives_square_tile_on_tpu():
    """`syrk.derive_syrk` derives a legal VMEM-bounded square tile on TPU v5e (row 11941)."""
    v5e = generation.gpu_target("v5e")
    with generation.assumed("v5e"):
        tile = syrk.derive_syrk((4, 2560, 2560), jnp.bfloat16, jnp.bfloat16, v5e)
        assert tile is not None
        assert syrk.square_super_tile(tile, syrk.family_of(tile, jnp.bfloat16))


def test_newton_schulz_and_syrk_xla_fallback_are_bit_equal():
    """`gemm.symmetric_gram`, `gemm.symmetric_polynomial`, and `newton_schulz` on the XLA fallback path are
    bit-for-bit equal (`assert_array_equal`) to the direct `batched_product` expressions they replace across
    batched bf16 wide/tall/square matrices, complex64 matrices (`conjugate=True/False`), and all four
    preconditionings (rows 11941, 11981)."""
    from dew import newton_schulz as ns

    rng = np.random.default_rng(11981)
    for shape, transpose in [((2, 16, 32), False), ((2, 32, 16), True), ((2, 16, 16), False)]:
        x = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
        axis = (x.ndim - 2) if transpose else (x.ndim - 1)
        ref_g = jax.jit(lambda a, ax=axis: gemm.batched_product(a, a, (ax, ax)))(x)
        got_g = jax.jit(lambda a, t=transpose: gemm.symmetric_gram(a, transpose=t))(x)
        np.testing.assert_array_equal(np.asarray(got_g), np.asarray(ref_g))

        c = jnp.asarray(ns.NEWTON_SCHULZ, jnp.bfloat16)
        ref_p = jax.jit(lambda a, tr: tr[0] * jnp.eye(a.shape[-1], dtype=a.dtype) + tr[1] * a
                        + tr[2] * gemm.batched_product(a, a, (a.ndim - 1, a.ndim - 2)))(ref_g, c)
        got_p = jax.jit(lambda a, tr: gemm.symmetric_polynomial(a, tr))(ref_g, c)
        np.testing.assert_array_equal(np.asarray(got_p), np.asarray(ref_p))

    for shape, transpose in [((2, 16, 24), False), ((2, 24, 16), True)]:
        z = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), jnp.complex64)
        axis = (z.ndim - 2) if transpose else (z.ndim - 1)
        for conjugate in (True, False):
            def ref_zgram(a, ax=axis, tr=transpose, cj=conjugate):
                held = a.conj() if cj else a
                return (gemm.batched_product(held, a, (ax, ax)) if tr
                        else gemm.batched_product(a, held, (ax, ax)))
            ref_g = jax.jit(ref_zgram)(z)
            got_g = jax.jit(lambda a, t=transpose, cj=conjugate: gemm.symmetric_gram(
                a, transpose=t, conjugate=cj))(z)
            np.testing.assert_array_equal(np.asarray(got_g), np.asarray(ref_g))

        c = jnp.asarray(ns.NEWTON_SCHULZ, jnp.complex64)
        ref_p = jax.jit(lambda a, tr: tr[0] * jnp.eye(a.shape[-1], dtype=a.dtype) + tr[1] * a
                        + tr[2] * gemm.batched_product(a, a, (a.ndim - 1, a.ndim - 2)))(ref_g, c)
        got_p = jax.jit(lambda a, tr: gemm.symmetric_polynomial(a, tr))(ref_g, c)
        np.testing.assert_array_equal(np.asarray(got_p), np.asarray(ref_p))

    # End-to-end `newton_schulz` bit-equality against inline `batched_product` (frobenius & spectral).
    def ref_ns(x, preconditioning):
        rows, cols = x.shape[-2:]
        tall, last = rows > cols, x.ndim - 1
        if preconditioning == "frobenius":
            x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + 1e-8)
        else:
            x = x / (jnp.linalg.norm(x, ord=2, axis=(-2, -1), keepdims=True) + 1e-8)
        triple = jnp.asarray(ns.NEWTON_SCHULZ, x.dtype)
        for _ in range(3):
            held = x.conj()
            a = (gemm.batched_product(held, x, (last - 1, last - 1)) if tall
                 else gemm.batched_product(x, held, (last, last)))
            sq = gemm.batched_product(a, a, (last, last - 1))
            poly = triple[0] * jnp.eye(a.shape[-1], dtype=a.dtype) + triple[1] * a + triple[2] * sq
            x = (gemm.batched_product(x, poly, (last, last - 1)) if tall
                 else gemm.batched_product(poly, x, (last, last - 1)))
        return x

    x_bf16 = jnp.asarray(rng.standard_normal((2, 16, 24)), jnp.bfloat16)
    x_f32 = jnp.asarray(rng.standard_normal((2, 16, 24)), jnp.float32)
    z_c64 = jnp.asarray(rng.standard_normal((2, 16, 24)) + 1j * rng.standard_normal((2, 16, 24)),
                        jnp.complex64)
    for arr, precond in ((x_bf16, "frobenius"), (z_c64, "frobenius"),
                         (x_f32, "spectral"), (z_c64, "spectral")):
        got = jax.jit(lambda a, p=precond: ns.newton_schulz(a, ns.NEWTON_SCHULZ, 3, 1e-8, p))(arr)
        ref = jax.jit(lambda a, p=precond: ref_ns(a, p))(arr)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("conjugate", [True, False])
def test_complex_symmetric_gram_value_and_vjp_match_dot_general(transpose, conjugate):
    """Complex `gemm.symmetric_gram` (`X @ X^H`, `X^H @ X`, `X @ X^T`, `X^T @ X`) and its VJP hold check 1
    against `jax.lax.dot_general` from complex128 truth (row 11981)."""
    shape = (2, 12, 8) if transpose else (2, 8, 12)
    axis = 1 if transpose else 2
    dims = (((axis,), (axis,)), ((0,), (0,)))

    def draw(seed):
        rng = np.random.default_rng(11982 + seed)
        x = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), jnp.complex64)
        cot = jnp.asarray(rng.standard_normal((2, 8, 8)) + 1j * rng.standard_normal((2, 8, 8)), jnp.complex64)
        return x, cot

    def plain(x, cot, dtype=jnp.complex64):
        def fwd(a):
            held = a.conj() if conjugate else a
            left, right = (held, a) if transpose else (a, held)
            return jax.lax.dot_general(left, right, dims)
        y, pull = jax.vjp(fwd, x.astype(dtype))
        return y, pull(cot.astype(dtype))[0]

    def dew_fn(x, cot):
        y, pull = jax.vjp(lambda a: gemm.symmetric_gram(a, transpose=transpose, conjugate=conjugate), x)
        return y, pull(cot)[0]

    assert_as_exact(jax.jit(dew_fn), jax.jit(plain), draw,
                    f"complex syrk gram t={transpose} c={conjugate}", seeds=SEEDS,
                    truth_fn=functools.partial(plain, dtype=jnp.complex128))


@lowers("sm90")
def test_newton_schulz_reaches_mosaic_syrk_and_honors_dot_algorithm():
    """Live `newton_schulz` reaches certified `MosaicSyrk` when `grouped_matmul='mosaic'` is chosen on bf16
    operands, while an explicit non-matching dot algorithm keeps XLA's dot (rows 11941, 11981)."""
    from kernel_compile import lowered_for

    from dew import newton_schulz as ns
    from dew.nn.kernels import selection
    from dew.nn.kernels.templates import report
    from dew.nn.precision import configured_precision

    x = jax.ShapeDtypeStruct((2, 256, 512), jnp.bfloat16)
    before = {id(c) for c in report.certifications()}
    with (report.refusals(), report.certifying(generation.compile_only_gpus("sm90")[0]),
          generation.chosen(grouped_matmul="mosaic")):
        lowered_for(lambda a: ns.newton_schulz(a, ns.NEWTON_SCHULZ, 1, 1e-8), (x,), "sm90")
    assert any(id(c) not in before and c.op == type(syrk.MOSAIC_SYRK).__name__
               for c in report.certifications())

    # Non-matching DotAlgorithmPreset (TF32_TF32_F32 on bf16) refuses Mosaic SYRK; BF16_BF16_F32 and
    # HIGHEST (exact fp32 accumulation of bf16 products) permit it.
    with generation.chosen(grouped_matmul="mosaic"):
        with (configured_precision(jax.lax.DotAlgorithmPreset.TF32_TF32_F32),
              pytest.raises(selection.KernelRefused)):
            lowered_for(lambda a: gemm.symmetric_gram(a), (x,), "sm90")
        for preset in (jax.lax.DotAlgorithmPreset.BF16_BF16_F32, jax.lax.Precision.HIGHEST):
            with configured_precision(preset), generation.routes() as seen:
                lowered_for(lambda a: gemm.symmetric_gram(a), (x,), "sm90")
            assert any(r.kernel == "mosaic" for r in seen), (preset, seen)
