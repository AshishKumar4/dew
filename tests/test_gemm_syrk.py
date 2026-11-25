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
        with generation.chosen(syrk="mosaic"):
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
        with generation.chosen(syrk="mosaic"):
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
    """Default (unforced) ragged BF16 SYRK declines on `measured` before certifying `MosaicSyrk`, while forced
    `syrk='mosaic'` certifies `MosaicSyrk` on the compile-only device (row 11941)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.templates import report

    x = jax.ShapeDtypeStruct((4, 150, 200), jnp.bfloat16)
    before_default = {id(c) for c in report.certifications()}
    with report.refusals(), report.certifying(generation.compile_only_gpus("sm90")[0]):
        lowered_for(lambda operand: gemm.symmetric_gram(operand), (x,), "sm90")
    assert not [c for c in report.certifications()
                if id(c) not in before_default and c.op == type(syrk.MOSAIC_SYRK).__name__]

    before_forced = {id(c) for c in report.certifications()}
    with (report.refusals(), report.certifying(generation.compile_only_gpus("sm90")[0]),
          generation.chosen(syrk="mosaic")):
        lowered_for(lambda operand: gemm.symmetric_gram(operand), (x,), "sm90")
    assert any(id(c) not in before_forced and c.op == type(syrk.MOSAIC_SYRK).__name__
               for c in report.certifications())


@pytest.mark.parametrize("other", ["mosaic", "cutlass", "triton"])
def test_naming_the_grouped_products_kernel_leaves_muons_symmetric_products_their_own(other):
    """Muon's symmetric Gram and Newton-Schulz polynomial (`gemm.SYRK`) select under their own choice
    (`syrk`): naming the grouped products' kernel (`grouped_matmul`) routes no SYRK call to it nor refuses
    one, each symmetric product on an sm90's routes the default's, as `syrk` unnamed takes it (row 12485)."""
    x = jax.ShapeDtypeStruct((4, 128, 256), jnp.bfloat16)

    def routes_of(**named):
        with generation.assumed("sm90"), generation.chosen(**named), generation.routes() as found:
            jax.eval_shape(gemm.symmetric_gram, x)
            jax.eval_shape(lambda a: gemm.symmetric_polynomial(a, (3.4445, -4.7750, 2.0315)),
                           jax.ShapeDtypeStruct((4, 128, 128), jnp.bfloat16))
        return [(route.choice, route.kernel) for route in found if route.choice == "syrk"]

    default = routes_of()
    assert default and routes_of(grouped_matmul=other) == default


def test_syrk_measured_gate_admits_whole_tile_ampere_and_hopper_grams_and_polynomials():
    """`M-NG-45-a100`, `M-NG-45b-gh200` and `M-NG-5` admit whole-tile `mma.sync`/`wgmma` symmetric Grams
    (1.52-1.98x XLA) and polynomials (1.17-1.96x) while keeping ragged shapes and unmeasured `sm100` on
    XLA."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    for gpu in ("sm80", "sm90"):
        with generation.assumed(gpu):
            assert gemm._syrk_select(bf16((28, 2560, 4096)), has_poly=False, transpose=False,
                                     precision=None, out_dtype=jnp.bfloat16, interpret_on_cpu=False)
            assert not gemm._syrk_select(bf16((4, 150, 200)), has_poly=False, transpose=False,
                                         precision=None, out_dtype=jnp.bfloat16, interpret_on_cpu=False)
            assert gemm._syrk_select(bf16((28, 2560, 2560)), has_poly=True, transpose=False,
                                     precision=None, out_dtype=jnp.bfloat16, interpret_on_cpu=False)
            assert not gemm._syrk_select(bf16((4, 150, 150)), has_poly=True, transpose=False,
                                         precision=None, out_dtype=jnp.bfloat16, interpret_on_cpu=False)
    with generation.assumed("sm100"):
        assert not gemm._syrk_select(bf16((28, 2560, 4096)), has_poly=False, transpose=False,
                                     precision=None, out_dtype=jnp.bfloat16, interpret_on_cpu=False)


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


def test_newton_schulz_and_syrk_xla_fallback_numerics():
    """`gemm.symmetric_gram` (bf16/fp32/complex64), `gemm.symmetric_polynomial` (fp32/complex64), and
    `newton_schulz` (fp32/complex64 across all four preconditionings: frobenius, spectral, aol, schatten)
    stay bit-equal (`assert_array_equal`) to direct `batched_product`; bf16 `symmetric_polynomial` and bf16
    `newton_schulz` accumulate in fp32 (holding check 1 against float64 truth and exact `-I/512` witness),
    and AOL scaling preserves bitwise symmetry for native SYRK (rows 11941, 11981, 12024, 12025, 12026)."""
    from dew import newton_schulz as ns

    rng = np.random.default_rng(11981)
    # 1. Gram bit-equality on bf16 wide/tall/square and fp32/complex64 (both conjugate settings).
    for shape, transpose in [((2, 16, 32), False), ((2, 32, 16), True), ((2, 16, 16), False)]:
        for dtype in (jnp.bfloat16, jnp.float32):
            x = jnp.asarray(rng.standard_normal(shape), dtype)
            axis = (x.ndim - 2) if transpose else (x.ndim - 1)
            ref_g = jax.jit(lambda a, ax=axis: gemm.batched_product(a, a, (ax, ax)))(x)
            got_g = jax.jit(lambda a, t=transpose: gemm.symmetric_gram(a, transpose=t))(x)
            np.testing.assert_array_equal(np.asarray(got_g), np.asarray(ref_g))
            if dtype == jnp.float32:
                c = jnp.asarray(ns.NEWTON_SCHULZ, jnp.float32)
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

    # 2. Deterministic #12025 witness: bf16 A=I, c=(1, -1-1/512, 0) yields exact -I/512 in fp32 accumulation.
    eye_bf16 = jnp.eye(8, dtype=jnp.bfloat16)
    c_witness = jnp.asarray((1.0, -1.0 - 1.0 / 512.0, 0.0), jnp.float32)
    for batch_shape in ((), (5,), (16,)):
        batched_eye = jnp.broadcast_to(eye_bf16, (*batch_shape, 8, 8))
        got_w = jax.jit(lambda a, c: gemm.symmetric_polynomial(a, c))(batched_eye, c_witness)
        np.testing.assert_array_equal(np.asarray(got_w), np.asarray((-1.0 / 512.0) * batched_eye))

    # 3. End-to-end `newton_schulz` bit-equality on float32 and complex64 across ALL FOUR preconditionings.
    def ref_ns(x, preconditioning, *, dtype=None):
        if dtype is not None:
            x = x.astype(dtype)
        rows, cols = x.shape[-2:]
        tall, last = rows > cols, x.ndim - 1
        eps = ns._FIRST_STEP_EPS

        def _gram(m, cj=True):
            h = m.conj() if (cj and jnp.iscomplexobj(m)) else m
            return (gemm.batched_product(h, m, (last - 1, last - 1)) if tall
                    else gemm.batched_product(m, h, (last, last)))

        def _step(m, a, tr):
            sq = gemm.batched_product(a, a, (last, last - 1))
            poly = tr[0] * jnp.eye(a.shape[-1], dtype=a.dtype) + tr[1] * a + tr[2] * sq
            return (gemm.batched_product(m, poly, (last, last - 1)) if tall
                    else gemm.batched_product(poly, m, (last, last - 1)))

        if preconditioning == "frobenius":
            x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + 1e-8)
        elif preconditioning == "spectral":
            x = x / (jnp.linalg.norm(x, ord=2, axis=(-2, -1), keepdims=True) + 1e-8)
        triple = jnp.asarray(ns.NEWTON_SCHULZ, x.dtype)
        if preconditioning == "aol":
            a = _gram(x)
            scale = jax.lax.rsqrt(jnp.clip(jnp.abs(a).sum(axis=-1), min=eps))
            side = scale[..., None, :] if tall else scale[..., :, None]
            x = _step(x * side, a * (scale[..., :, None] * scale[..., None, :]), triple)
        elif preconditioning == "schatten":
            a = _gram(x, cj=False)
            scale = jax.lax.rsqrt(jnp.clip(jnp.linalg.norm(a, axis=(-2, -1), keepdims=True), min=eps))
            x = _step(x * scale, a * scale ** 2, triple)
        else:
            x = _step(x, _gram(x), triple)
        for _ in range(2):
            x = _step(x, _gram(x), triple)
        return x

    x_f32 = jnp.asarray(rng.standard_normal((2, 16, 24)), jnp.float32)
    z_c64 = jnp.asarray(rng.standard_normal((2, 16, 24)) + 1j * rng.standard_normal((2, 16, 24)),
                        jnp.complex64)
    for precond in ("frobenius", "spectral", "aol", "schatten"):
        for arr in (x_f32, z_c64):
            got = jax.jit(lambda a, p=precond: ns.newton_schulz(a, ns.NEWTON_SCHULZ, 3, 1e-8, p))(arr)
            ref = jax.jit(lambda a, p=precond: ref_ns(a, p))(arr)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_bf16_newton_schulz_aol_symmetry_and_fp32_poly_check_1():
    """bf16 `newton_schulz` AOL preserves transpose equivariance `f(X^T)^T == f(X)` bit-for-bit (proving
    the AOL-scaled Gram passed to `symmetric_polynomial` is bitwise symmetric, red on sequential scaling),
    and bf16 `symmetric_polynomial` / `newton_schulz` (`frobenius`, `aol`, `schatten`) hold check 1 against
    float64 truth (rows 12024, 12025)."""
    from dew import newton_schulz as ns

    # Red-on-parent #12024 witness through `ns.newton_schulz(..., "aol")`: for wide X vs tall X^T with
    # coeffs=(0, 1, 0), f(X) = a_scaled @ (s[:, None] * X) while f(X^T)^T = a_scaled^T @ (s[:, None] * X).
    # Bitwise equality holds iff `a_scaled == a_scaled^T` bit-for-bit (fails under sequential multiplies).
    rng = np.random.default_rng(12024)
    x_wide = jnp.asarray(rng.standard_normal((2, 32, 64)), jnp.bfloat16)
    x_tall = jnp.swapaxes(x_wide, -1, -2)
    out_wide = jax.jit(lambda a: ns.newton_schulz(a, (0.0, 1.0, 0.0), 1, 1e-8, "aol"))(x_wide)
    out_tall = jax.jit(lambda a: ns.newton_schulz(a, (0.0, 1.0, 0.0), 1, 1e-8, "aol"))(x_tall)
    np.testing.assert_array_equal(np.asarray(out_wide), np.asarray(jnp.swapaxes(out_tall, -1, -2)))

    def bf16_poly_ref(a, c, *, dtype=jnp.float32, out_dtype=jnp.bfloat16):
        cw = c.astype(dtype)
        sq = jax.lax.dot_general(a.astype(out_dtype), a.astype(out_dtype),
                                 (((a.ndim - 1,), (a.ndim - 2,)), ((0,), (0,))),
                                 preferred_element_type=dtype)
        eye = jnp.eye(a.shape[-1], dtype=dtype)
        return (cw[0] * eye + cw[1] * a.astype(dtype) + cw[2] * sq).astype(out_dtype)

    def draw_sym(seed):
        r = np.random.default_rng(12025 + seed)
        x = jnp.asarray(r.standard_normal((1, 64, 128)) / np.sqrt(128), jnp.bfloat16)
        a = gemm.symmetric_gram(x)
        s = jax.lax.rsqrt(jnp.clip(jnp.abs(a).sum(axis=-1), min=ns._FIRST_STEP_EPS))
        return a * (s[..., :, None] * s[..., None, :]), jnp.asarray(ns.NEWTON_SCHULZ, jnp.float32)

    for interp in (False, True):
        assert_as_exact(
            jax.jit(lambda a, c, it=interp: gemm.symmetric_polynomial(a, c, interpret_on_cpu=it)),
            jax.jit(bf16_poly_ref), draw_sym, f"bf16 aol syrk poly interp={interp}", seeds=SEEDS,
            truth_fn=functools.partial(bf16_poly_ref, dtype=jnp.float64, out_dtype=jnp.float64),
        )

    def bf16_ns_ref(x, preconditioning, *, work=jnp.float32, out=jnp.bfloat16):
        x = x.astype(out)
        rows, cols = x.shape[-2:]
        tall, last = rows > cols, x.ndim - 1
        eps = ns._FIRST_STEP_EPS
        triple = jnp.asarray(ns.NEWTON_SCHULZ, out).astype(work)

        def _gram(m):
            ax = last - 1 if tall else last
            return gemm.batched_product(m, m, (ax, ax))

        def _step(m, a):
            poly = bf16_poly_ref(a, triple, dtype=work, out_dtype=out)
            return (gemm.batched_product(m, poly, (last, last - 1)) if tall
                    else gemm.batched_product(poly, m, (last, last - 1)))

        if preconditioning == "frobenius":
            x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + 1e-8)
            x = _step(x, _gram(x))
        elif preconditioning == "aol":
            a = _gram(x)
            scale = jax.lax.rsqrt(jnp.clip(jnp.abs(a).sum(axis=-1), min=eps))
            side = scale[..., None, :] if tall else scale[..., :, None]
            x = _step(x * side, a * (scale[..., :, None] * scale[..., None, :]))
        else:
            a = _gram(x)
            scale = jax.lax.rsqrt(jnp.clip(jnp.linalg.norm(a, axis=(-2, -1), keepdims=True), min=eps))
            x = _step(x * scale, a * scale ** 2)
        for _ in range(2):
            x = _step(x, _gram(x))
        return x

    for precond in ("frobenius", "aol", "schatten"):
        def draw_bf16(seed, p=precond):
            r = np.random.default_rng(12025 + seed + len(p))
            return (jnp.asarray(r.standard_normal((2, 16, 24)) / np.sqrt(24), jnp.bfloat16),)

        assert_as_exact(
            jax.jit(lambda a, p=precond: ns.newton_schulz(a, ns.NEWTON_SCHULZ, 3, 1e-8, p)),
            jax.jit(lambda a, p=precond: bf16_ns_ref(a, p)),
            draw_bf16, f"bf16 newton_schulz {precond}", seeds=SEEDS,
            truth_fn=lambda a, p=precond: bf16_ns_ref(a, p, work=jnp.float64, out=jnp.float64),
        )


def test_syrk_interpreting_predicate():
    """`xla_products.interpreted` uses placement platform for SYRK selection and execution (row 12026)."""
    from dew.nn.kernels import xla_products

    assert xla_products.interpreted(interpret_on_cpu=True) is True
    assert xla_products.interpreted(interpret_on_cpu=False) is False
    with generation.assumed("sm90"):
        assert xla_products.interpreted(interpret_on_cpu=True) is True
    if generation.compile_only_gpus("sm90"):
        mesh = generation.gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
        with jax.sharding.use_abstract_mesh(mesh):
            assert xla_products.interpreted(interpret_on_cpu=True) is False


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
    """Live `newton_schulz` reaches certified `MosaicSyrk` when `syrk='mosaic'` is chosen on bf16
    operands, while an explicit non-matching dot algorithm keeps XLA's dot (rows 11941, 11981)."""
    from kernel_compile import lowered_for

    from dew import newton_schulz as ns
    from dew.nn.kernels import selection
    from dew.nn.kernels.templates import report
    from dew.nn.precision import configured_precision

    x = jax.ShapeDtypeStruct((2, 256, 512), jnp.bfloat16)
    before = {id(c) for c in report.certifications()}
    with (report.refusals(), report.certifying(generation.compile_only_gpus("sm90")[0]),
          generation.chosen(syrk="mosaic")):
        lowered_for(lambda a: ns.newton_schulz(a, ns.NEWTON_SCHULZ, 1, 1e-8), (x,), "sm90")
    assert any(id(c) not in before and c.op == type(syrk.MOSAIC_SYRK).__name__
               for c in report.certifications())

    # Non-matching DotAlgorithmPreset (TF32_TF32_F32 on bf16) refuses Mosaic SYRK; BF16_BF16_F32 and
    # HIGHEST (exact fp32 accumulation of bf16 products) permit it.
    with generation.chosen(syrk="mosaic"):
        with (configured_precision(jax.lax.DotAlgorithmPreset.TF32_TF32_F32),
              pytest.raises(selection.KernelRefused)):
            lowered_for(lambda a: gemm.symmetric_gram(a), (x,), "sm90")
        for preset in (jax.lax.DotAlgorithmPreset.BF16_BF16_F32, jax.lax.Precision.HIGHEST):
            with configured_precision(preset), generation.routes() as seen:
                lowered_for(lambda a: gemm.symmetric_gram(a), (x,), "sm90")
            assert any(r.kernel == "mosaic" for r in seen), (preset, seen)


@lowers("sm90")
def test_syrk_measured_gates_grams_and_polynomials_to_measured_bf16_cells():
    """`syrk.measured` admits whole-tile bf16 Gram and polynomial shapes filling at least one SM wave on
    measured generations (`sm80`, `sm90`) with no hardcoded shape table, refusing non-whole-tile,
    underfilled, fp32 output, fp16 input, and unmeasured generations (rows 11941, 12163, 5972)."""
    from kernel_compile import lowered_for

    sm80, sm90 = generation.gpu_target("sm80"), generation.gpu_target("sm90")
    assert syrk.measured(sm80, (28, 2560, 4096), jnp.bfloat16, transpose=False, out_dtype=jnp.bfloat16)
    assert syrk.measured(sm90, (36, 4096, 14336), jnp.bfloat16, transpose=True, out_dtype=jnp.bfloat16)
    # Derived shape-class rule admits arbitrary model layer counts and single large matrices >= 1 wave.
    assert syrk.measured(sm80, (29, 2560, 4096), jnp.bfloat16, transpose=False, out_dtype=jnp.bfloat16)
    assert syrk.measured(sm90, (1, 4096, 4096), jnp.bfloat16, transpose=False, out_dtype=jnp.bfloat16)
    # Refuses non-whole-tile shapes, underfilled grids, non-bf16 dtypes, and unmeasured generations.
    assert not syrk.measured(sm90, (28, 130, 130), jnp.bfloat16, transpose=False,
                                  out_dtype=jnp.bfloat16)
    assert not syrk.measured(sm90, (1, 512, 512), jnp.bfloat16, transpose=False,
                                  out_dtype=jnp.bfloat16)
    assert not syrk.measured(sm90, (28, 2560, 4096), jnp.bfloat16, transpose=False,
                                  out_dtype=jnp.float32)
    assert not syrk.measured(sm80, (28, 2560, 4096), jnp.float16, transpose=False,
                                  out_dtype=jnp.float16)
    for gen in ("sm86", "sm89", "sm120"):
        assert not syrk.measured(generation.gpu_target(gen), (28, 2560, 4096), jnp.bfloat16,
                                 transpose=False, out_dtype=jnp.bfloat16)

    # End-to-end auto selection admits whole-tile bf16->bf16 across layer counts and keeps fp32 / underfilled
    # on XLA.
    for shape in ((28, 2560, 4096), (29, 2560, 4096)):
        with generation.routes() as measured_seen:
            lowered_for(lambda a: gemm.symmetric_gram(a),
                        (jax.ShapeDtypeStruct(shape, jnp.bfloat16),), "sm90")
        assert any(r.kernel == "mosaic" for r in measured_seen), (shape, measured_seen)
    # A polynomial its caller states symmetric (a Gram's, Newton-Schulz's) takes the launch; a square operand
    # of any values keeps XLA's program, whose `a @ a` the launch's mirrored upper triangle is not.
    for symmetric, kernel in ((True, "mosaic"), (False, "xla")):
        with generation.routes() as polynomial_seen:
            lowered_for(lambda a, symmetric=symmetric: gemm.symmetric_polynomial(
                a, jnp.asarray([3.4445, -4.7750, 2.0315]), symmetric=symmetric),
                (jax.ShapeDtypeStruct((28, 2560, 2560), jnp.bfloat16),), "sm90")
        assert {r.kernel for r in polynomial_seen if r.choice == "syrk"} == {kernel}, (symmetric,
                                                                                    polynomial_seen)
    for shape, out_dtype in (((1, 512, 512), jnp.bfloat16), ((28, 2560, 4096), jnp.float32)):
        with generation.routes() as unmeasured_seen:
            lowered_for(lambda a, dt=out_dtype: gemm.symmetric_gram(a, out_dtype=dt),
                        (jax.ShapeDtypeStruct(shape, jnp.bfloat16),), "sm90")
        assert all(r.kernel != "mosaic" for r in unmeasured_seen), (shape, out_dtype, unmeasured_seen)


def test_syrk_portable_row_strip_chunking_matches_full_poly_across_ranks_and_odd_sides():
    """Narrow `_syrk_portable` chunks fp32 polynomial squares by output row strips for single matrices,
    batches, and odd side lengths bit-identically to unchunked fp32 evaluation (row 12162)."""
    rng = np.random.default_rng(12162)
    coeffs = jnp.asarray([3.4445, -4.7750, 2.0315], jnp.float32)
    for shape in ((131, 131), (1, 128, 128), (3, 67, 67)):
        x = jnp.asarray(rng.integers(-3, 4, size=shape), jnp.bfloat16) / jnp.bfloat16(16.0)
        got = gemm._syrk_portable(x, coeffs, transpose=False, conjugate=False, precision=None,
                                  out_dtype=jnp.bfloat16)
        sq = jax.lax.dot_general(
            x, x, (((x.ndim - 1,), (x.ndim - 2,)), (tuple(range(x.ndim - 2)), tuple(range(x.ndim - 2)))),
            preferred_element_type=jnp.float32)
        ref = (coeffs[0] * jnp.eye(shape[-1], dtype=jnp.float32)
               + coeffs[1] * x.astype(jnp.float32) + coeffs[2] * sq).astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
