"""The GEMM template's product of an NVFP4 matrix (`gemm.nvfp4_product`, row 3621): its weights' E2M1 codes
and E4M3 block scales decoded exactly (`NvFp4.decoded`, its codes in the staged order a stage decodes), the
deferred factor applied to the fp32 accumulator before the one rounding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS

from dew.algebra import bits
from dew.nn.kernels import gemm, generation

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
"""An E2M1 code's magnitude by its low three bits; the fourth is its sign."""


def nvfp4(seed: int, depth: int, width: int, *, scale: float = 0.25) -> tuple[gemm.NvFp4, np.ndarray]:
    """An NVFP4 matrix of random codes under power-of-two block scales (`gemm.NvFp4.of`, each value exact in
    float64), and its values: code times block scale times the tensor's `scale`."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(depth, width))
    magnitudes = np.where(codes >= 8, -1.0, 1.0) * np.asarray(E2M1)[codes & 7]
    block = np.exp2(rng.integers(-3, 4, size=(depth // 16, width))).astype(np.float32)
    matrix = gemm.NvFp4.of(jnp.asarray(magnitudes, jnp.float32), jnp.asarray(block, jnp.float8_e4m3fn), scale)
    return matrix, magnitudes * np.repeat(block.astype(np.float64), 16, axis=0) * scale


def test_an_nvfp4_matrix_holds_its_codes_and_the_staged_order_reads_back():
    """A matrix quantized from fp32 (`gemm.NvFp4.quantized`) holds `bits.nvfp4_quantized`'s codes and scales:
    its codes read back in the depths' own order (`NvFp4.codes`) are the quantizer's natural words' codes,
    its scales the quantizer's; and its words are K-major, in the staged order (`gemm.staged_rows`, within
    each 128 depths word w holding depths w + 16j), not the natural one's."""
    x = jnp.asarray(np.random.default_rng(5).standard_normal((256, 64)), jnp.float32)
    matrix = gemm.NvFp4.quantized(x)
    words, scales, scale = bits.nvfp4_quantized(x)
    natural = bits.nvfp4_decoded(words, jnp.full(scales.shape, bits.NVFP4_OWED, jnp.float8_e4m3fn))
    np.testing.assert_array_equal(np.asarray(matrix.codes()), np.asarray(natural, np.float32))
    np.testing.assert_array_equal(np.asarray(matrix.scales.T, np.float32), np.asarray(scales, np.float32))
    assert float(matrix.scale) == float(scale)
    assert matrix.words.shape == (64, 256 // 8)
    assert not np.array_equal(np.asarray(matrix.words.T), np.asarray(words))
    order = gemm.staged_rows(256)
    assert sorted(order) == list(range(256)) and order[8 * 3 + 5] == 16 * 5 + 3 and order[128] == 128


def test_an_nvfp4_product_is_its_decoded_products_times_the_deferred_factor():
    """XLA's route (a CPU's): the rows rounded to bfloat16 times the decoded matrix, summed in fp32, times
    `bits.NVFP4_OWED` and the tensor's scale, then rounded once: bitwise the program written out, and, its
    operands small integers and powers of two, exactly the float64 product of the matrix's values."""
    matrix, values = nvfp4(0, 256, 192)
    rows = jnp.asarray(np.random.default_rng(1).integers(-3, 4, size=(64, 256)), jnp.bfloat16)
    got = jax.jit(lambda x, m: gemm.nvfp4_product(x, m, out_dtype=jnp.float32))(rows, matrix)
    written = jnp.dot(rows, matrix.decoded(), preferred_element_type=jnp.float32) * matrix.factor
    np.testing.assert_array_equal(np.asarray(got), np.asarray(written))
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_an_nvfp4_product_through_the_template_is_exact(generation_name):
    """The template's launch interpreted on each family, the template named for the product: the same exact
    product; and the rows' gradient through the decoded matrix, the weights frozen, on the route its selector
    picks (the cotangent fp32 beside a bfloat16 matrix)."""
    matrix, values = nvfp4(2, 256, 256)
    rows = jnp.asarray(np.random.default_rng(3).integers(-3, 4, size=(128, 256)), jnp.bfloat16)
    cotangent = jnp.asarray(np.random.default_rng(4).integers(-2, 3, size=(128, 256)), jnp.float32)

    def product(x):
        return gemm.nvfp4_product(x, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed(generation_name):
        with generation.chosen(grouped_matmul="mosaic_gpu"), generation.routes() as routes:
            got = jax.jit(product)(rows)
        _, pull = jax.vjp(product, rows)
        (gradient,) = pull(cotangent)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic_gpu"}
    np.testing.assert_array_equal(np.asarray(got, np.float64), np.asarray(rows, np.float64) @ values)
    # The rows' gradient is exact in fp32 (small integers times powers of two) and rounds once to bfloat16.
    expected = jnp.asarray(np.asarray(cotangent, np.float64) @ values.T, jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(gradient, np.float32), np.asarray(expected, np.float32))
