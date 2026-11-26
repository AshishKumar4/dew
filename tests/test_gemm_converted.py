"""FP8 codes' product on a GPU of no FP8 products (`gemm.fp8_codes_product`, row 8548): each code converted
exactly into bfloat16 as an A100's stages convert them, the converting launch on par with XLA's widened
product (check 1) interpreted as an A100's, its default XLA's until a receipt, and its launch compiled
spill-free for a compile-only sm80."""

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
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation
from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates.gemm import tiling

FORMATS = {"e4m3": jnp.float8_e4m3fn, "e5m2": jnp.float8_e5m2}
ROWS, DEPTH, WIDTH, SKINNY = 72, 200, 136, 8
"""A product's rows, depth and width, none a whole tile: the depth and width padded with zero codes to whole
steps (`converted.STEP`), the rows a partial tile's."""


@pytest.mark.parametrize("name", list(FORMATS))
def test_every_fp8_code_converts_exactly_where_no_tensor_core_multiplies_fp8(name):
    """Each of the 256 codes of a format, as an A100's stage converts it (`mosaic.Family.converted`'s
    `bits.decoded`, no FP8 conversion instruction there), is its exact value: XLA's widening bitwise, its
    signed zeros, subnormals, E5M2's infinities and each NaN a NaN, so the converting launch multiplies the
    values XLA's widened product does (row 8548)."""
    codes = np.arange(256, dtype=np.uint8).view(FORMATS[name])
    with generation.assumed("sm80"):
        converted = np.asarray(mosaic._decoded_codes(jnp.asarray(codes), codes.dtype), np.float32)
    widened = codes.astype(np.float32)
    nan = np.isnan(widened)
    np.testing.assert_array_equal(np.isnan(converted), nan)
    np.testing.assert_array_equal(converted[~nan].view(np.uint32), widened[~nan].view(np.uint32))
    assert np.array_equal(converted[~nan].astype(jnp.bfloat16).astype(np.float32), widened[~nan])


def _operands(name: str, seed: int, rows: int = ROWS) -> tuple[np.ndarray, np.ndarray]:
    """A step's codes and a frozen site's codes, each a normal draw cast to the format (NumPy's: a GPU
    tier's prepare computes on no device)."""
    rng = np.random.default_rng(seed)
    dtype = FORMATS[name]
    return (rng.normal(size=(rows, DEPTH)).astype(dtype), rng.normal(size=(WIDTH, DEPTH)).astype(dtype))


def _product(kernel: str, lhs, rhs, *, interpret_on_cpu: bool) -> jax.Array:
    with generation.chosen(grouped_matmul=kernel):
        return gemm.fp8_codes_product(lhs, rhs, out_dtype=jnp.float32, interpret_on_cpu=interpret_on_cpu)


ORIENTATIONS = {
    "unswapped": tiling.Tiles(64, 64, 64, 4, 2, warpgroups=1, family="mma.sync", programs=2),
    "swapped": tiling.Tiles(64, 32, 64, 4, 1, warpgroups=1, family="mma.sync", programs=2, swapped=True),
}
"""The converting launch's two forms on an A100 (row 5387), each pinned so each runs by name: its matrix's
codes converted a stage at a time, or a decode's, its codes the products' rows, each consumer converting its
own (`MosaicConvertedGmm._swapped`)."""


def an_fp8_codes_product_through_the_converting_template_is_on_par(name: str, interpret_on_cpu: bool = False,
                                                                   orientation: str = "unswapped") -> None:
    """On an A100, the product of FP8 codes through the template named in `orientation` (`ORIENTATIONS`'
    tile pinned: at a partial tile's rows unswapped, at a decode's swapped), its codes converted exactly into
    bfloat16 before its products read them (`converted.MosaicConvertedGmm`), on par with XLA's widened
    product (check 1) from float64; the routes it records name the template."""
    from dew.nn.kernels import tuning
    from dew.nn.kernels.templates.gemm import converted

    tile = ORIENTATIONS[orientation]
    rows = SKINNY if tile.swapped else ROWS

    def truth(lhs, rhs):
        return jnp.asarray(lhs, jnp.float64) @ jnp.asarray(rhs, jnp.float64).T

    with generation.routes() as routes, tuning.pinned({converted.MosaicConvertedGmm: tile}):
        assert_as_exact(jax.jit(functools.partial(_product, "mosaic", interpret_on_cpu=interpret_on_cpu)),
                        jax.jit(functools.partial(_product, "xla", interpret_on_cpu=interpret_on_cpu)),
                        functools.partial(_operands, name, rows=rows),
                        f"fp8 codes product {name} {orientation}", seeds=8, truth_fn=truth)
    assert ("grouped_matmul", "mosaic") in {(route.choice, route.kernel) for route in routes}


@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
@pytest.mark.parametrize("name", list(FORMATS))
def test_an_fp8_codes_product_through_the_converting_template_is_on_par(name, orientation):
    """`an_fp8_codes_product_through_the_converting_template_is_on_par` interpreted on a CPU as an A100 runs
    it (the GPU's native cells, tests/test_gpu_gemm.py, run it compiled), in each orientation."""
    with generation.assumed("sm80"):
        an_fp8_codes_product_through_the_converting_template_is_on_par(name, interpret_on_cpu=True,
                                                                       orientation=orientation)


def test_an_fp8_codes_product_unnamed_is_xlas_widened_product_until_a_receipt():
    """On an A100, the FP8 codes product as its selector chooses is XLA's widened product, bitwise
    `gemm.gemm`'s, until a receipt finds the converting launch faster (`converted.FP8_CODES`)."""
    lhs, rhs = _operands("e4m3", 0)
    with generation.assumed("sm80"), generation.routes() as routes:
        chosen = gemm.fp8_codes_product(jnp.asarray(lhs), jnp.asarray(rhs), out_dtype=jnp.float32,
                                        interpret_on_cpu=True)
        plain = gemm.gemm(jnp.asarray(lhs), jnp.asarray(rhs), transpose_rhs=True, out_dtype=jnp.float32,
                          interpret_on_cpu=True)
    assert ("grouped_matmul", "mosaic") not in {(route.choice, route.kernel) for route in routes}
    np.testing.assert_array_equal(np.asarray(chosen), np.asarray(plain))


@pytest.mark.parametrize("kernel", ["mosaic", "xla"])
def test_an_fp8_codes_product_differentiates_its_rows_alone(kernel):
    """On an A100, through the converting launch named or XLA's arm, the FP8 codes product of rows and a
    matrix cast to E4M3 (an unaligned depth and width, padded and sliced back) takes the rows' tangent times
    the matrix's values, its pullback that product's transpose, and the matrix no cotangent: a frozen
    site's (row 8548's review). Every value dyadic, each sum exact in fp32 and each cotangent a matrix row
    E4M3 holds, so each is compared exactly."""
    rng = np.random.default_rng(8548)
    x = rng.choice([-2.0, -1.0, 0.0, 0.5, 1.0], size=(ROWS, DEPTH)).astype(np.float32)
    w = rng.choice([-1.0, -0.5, 0.5, 1.0], size=(WIDTH, DEPTH)).astype(np.float32)
    dx = rng.choice([-1.0, 0.0, 0.5], size=(ROWS, DEPTH)).astype(np.float32)
    # One column a row, of sign either way: each row's cotangent one of the matrix's rows, exact in E4M3.
    cotangent = np.zeros((ROWS, WIDTH), np.float32)
    cotangent[np.arange(ROWS), rng.integers(0, WIDTH, ROWS)] = rng.choice([-1.0, 1.0], size=ROWS)
    e4m3 = jnp.float8_e4m3fn

    def product(rows, weights):
        return _product(kernel, rows.astype(e4m3), weights.astype(e4m3), interpret_on_cpu=True)

    with generation.assumed("sm80"):
        _, tangent = jax.jvp(lambda rows: product(rows, jnp.asarray(w)), (jnp.asarray(x),),
                             (jnp.asarray(dx),))
        _, pull = jax.vjp(product, jnp.asarray(x), jnp.asarray(w))
        d_rows, d_weights = pull(jnp.asarray(cotangent))
    np.testing.assert_array_equal(np.asarray(tangent), (dx.astype(np.float64) @ w.T.astype(np.float64)))
    np.testing.assert_array_equal(np.asarray(d_rows), cotangent.astype(np.float64) @ w.astype(np.float64))
    assert np.count_nonzero(np.asarray(d_rows)) > d_rows.size // 2
    np.testing.assert_array_equal(np.asarray(d_weights), np.zeros_like(w))


@pytest.mark.parametrize("name", [pytest.param(name, marks=lowers("sm80")) for name in FORMATS])
def test_an_fp8_codes_product_compiles_spill_free_for_an_a100(name):
    """The converting launch compiles spill-free for a compile-only sm80 at a frozen FP8 site's decode rows
    and a prefill's (Llama-3-8B's up projection, 4096 to 14336), its codes a byte each (row 8548)."""
    if not generation.compile_only_gpus("sm80"):
        pytest.skip("no compile-only sm80 in this process")
    dtype = FORMATS[name]

    def product(lhs, rhs):
        with generation.chosen(grouped_matmul="mosaic"):
            return gemm.fp8_codes_product(lhs, rhs, out_dtype=jnp.float32)

    cases = [KernelCase(f"fp8_codes_{name}_{rows}",
                        lambda rows=rows: (product, (jax.ShapeDtypeStruct((rows, 4096), dtype),
                                                     jax.ShapeDtypeStruct((14336, 4096), dtype))))
             for rows in (16, 4096)]
    with generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"):
        records = compile_mosaic_kernels(cases, "sm80")
    for record in records:
        assert not record.get("error"), record
        assert record["kernels"], record
        assert all(kernel.get("spilled") == 0 for kernel in record["kernels"]), record


def test_a_swapped_converting_tiles_stage_is_converted_once_between_its_consumers():
    """A swapped converting tile of two consumers on an A100 moves one stage of its codes through their
    conversion, each converting its own rows (`packed.ROW_CODES`), not a stage each (row 13505): its
    shared bytes a step those of a one-consumer stage's conversion beside the same tile's reads."""
    from dew.nn.kernels.templates.gemm import converted, pricing

    launch = converted.MosaicConvertedGmm()
    bound = launch.bind(converted.Codes(jax.ShapeDtypeStruct((SKINNY * 8, 256), jnp.float8_e4m3fn)),
                        converted.Codes(jax.ShapeDtypeStruct((1, 192, 256), jnp.float8_e4m3fn)),
                        jax.ShapeDtypeStruct((1,), jnp.int32), (), (), None, None, None, out_dtype="float32",
                        trans_rhs=True, interpret=True, dense=True, exact_sums=False)
    tile = dataclasses.replace(ORIENTATIONS["swapped"], block_m=128, warpgroups=2)
    itemsizes, decoded = launch._operand_bytes(bound, tile)
    assert decoded == ((2, 1), None)
    with generation.assumed("sm80"):
        family = pricing.family_of(tile, jnp.bfloat16, generation.gpu_target())
    per_step = pricing.step_shared_bytes(tile, family, itemsizes, decoded)
    twice = pricing.step_shared_bytes(tile, family, itemsizes, ((2, 2), None))
    assert per_step < twice
