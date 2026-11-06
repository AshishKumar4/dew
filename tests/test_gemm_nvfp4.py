"""The GEMM template's product of an NVFP4 matrix (`gemm.nvfp4_product`, row 3621): its weights' E2M1 codes
and E4M3 block scales decoded exactly (`bits.nvfp4_decoded`), the deferred factor applied to the fp32
accumulator before the one rounding."""

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
    """An NVFP4 matrix of random codes under power-of-two block scales (each value exact in float64), and its
    values: code times block scale times the tensor's `scale`."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(depth, width))
    exponents = rng.integers(-3, 4, size=(depth // 16, width))
    block = np.exp2(exponents).astype(np.float32)
    packed = codes.reshape(depth // 8, 8, width).astype(np.uint64) << (
        4 * np.arange(8, dtype=np.uint64))[None, :, None]
    natural = jnp.asarray(packed.sum(axis=1).astype(np.uint32))
    matrix = gemm.NvFp4(bits.NVFP4_BFLOAT16_FORM.prepared(natural), jnp.asarray(block, jnp.float8_e4m3fn),
                        jnp.float32(scale))
    values = (np.where(codes >= 8, -1.0, 1.0) * np.asarray(E2M1)[codes & 7]
              * np.repeat(block.astype(np.float64), 16, axis=0) * scale)
    return matrix, values


def test_an_nvfp4_product_is_its_decoded_products_times_the_deferred_factor():
    """XLA's route (a CPU's): the rows rounded to bfloat16 times the decoded matrix, summed in fp32, times
    `bits.NVFP4_OWED` and the tensor's scale, then rounded once: bitwise the program written out, and, its
    operands small integers and powers of two, exactly the float64 product of the matrix's values."""
    matrix, values = nvfp4(0, 256, 192)
    rows = jnp.asarray(np.random.default_rng(1).integers(-3, 4, size=(64, 256)), jnp.bfloat16)
    got = jax.jit(lambda x, m: gemm.nvfp4_product(x, m, out_dtype=jnp.float32))(rows, matrix)
    decoded = bits.nvfp4_decoded(matrix.words, matrix.scales)
    written = jnp.dot(rows, decoded, preferred_element_type=jnp.float32) * matrix.factor
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
