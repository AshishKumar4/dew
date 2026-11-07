"""The NVFP4 operand as Blackwell's block-scaled tensor cores read it (`gemm.NvFp4Block`: E2M1 codes, an E4M3
scale every 16 depths under a tensor scale), its grouped product (`gemm.nvfp4_block_product`), the
inference projection quantizing its rows (`gemm.nvfp4_quantized_product`) and the NVFP4 recipe
(`gemm.nvfp4_projection`), row 4519's NVFP4 arm, on the CPU."""

import functools

import jax
import jax.numpy as jnp
import mx_matrix
import numpy as np
import pytest
from blockwise_matrix import step_operands
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import gemm, generation

SEEDS = 8


def _draw(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    """A normal draw whose 16-depth blocks span magnitudes from 2^-6 to 2^6."""
    rng = np.random.default_rng(seed)
    spread = np.exp2(rng.integers(-6, 7, size=(*shape[:-1], shape[-1] // gemm.NVFP4_BLOCK, 1)))
    blocks = rng.normal(size=(*shape[:-1], shape[-1] // gemm.NVFP4_BLOCK, gemm.NVFP4_BLOCK)) * spread
    return jnp.asarray(blocks.reshape(shape), jnp.float32)


def test_an_nvfp4_operand_is_its_values_again_and_its_stored_matrix():
    """An NVFP4 operand's values quantize to themselves (`NvFp4Block.quantized`), its codes E2M1 values and
    its blocks' largest code at most 6; and a stored NVFP4 matrix (`gemm.NvFp4`) read as the block-scaled
    products read it (`NvFp4Block.of`) holds its values, its depths in their own order."""
    operand = gemm.NvFp4Block.quantized(_draw(0, (64, 256)))
    values = operand.decoded() * operand.scale
    again = gemm.NvFp4Block.quantized(values)
    np.testing.assert_array_equal(np.asarray(again.decoded() * again.scale), np.asarray(values))
    assert float(jnp.max(jnp.abs(operand.codes.astype(jnp.float32)))) <= 6
    stored = gemm.NvFp4.quantized(_draw(1, (64, 256)).T)
    read = gemm.NvFp4Block.of(stored)
    np.testing.assert_array_equal(np.asarray(read.decoded()[0].T * read.scale),
                                  np.asarray(stored.decoded().astype(jnp.float32) * stored.factor))


def _operands(seed: int, shapes: tuple) -> tuple:
    """`_draw`s of each of `shapes`, from `seed` on."""
    return tuple(_draw(seed + index, shape) for index, shape in enumerate(shapes))


GROUP_BLOCKS = (1, 0, 2, 1)
"""Each group's 128-row blocks: a one-block group, an empty one, two blocks, one."""


@pytest.mark.parametrize("kernel", ["xla", "mosaic_gpu"])
def test_an_nvfp4_grouped_product_is_its_decoded_values_product(kernel):
    """Each group's rows times its matrix, both NVFP4, through `kernel` (the template on Blackwell's
    block-scaled tensor cores interpreted as a B200's, `mosaic_gpu.gmm`'s `mx_scales` at E2M1 operands,
    `kind::mxf4nvf4`, or the decoded operands in XLA), on par with the decoded values' grouped product in
    fp32 (check 1), from float64; an empty group's rows contribute nothing."""
    blocks = jnp.asarray(GROUP_BLOCKS, jnp.int32)
    sizes = 128 * blocks
    rows, depth, width = 128 * sum(GROUP_BLOCKS), 256, 128

    draw = functools.partial(_operands, shapes=((rows, depth), (len(GROUP_BLOCKS), width, depth)))

    def dew(x, w):
        with generation.chosen(nvfp4_block=kernel):
            operands = (gemm.NvFp4Block.quantized(x), gemm.NvFp4Block.quantized(w))
            return gemm.nvfp4_block_product(*operands, blocks, out_dtype=jnp.float32,
                                            interpret_on_cpu=kernel == "mosaic_gpu")

    def decoded(x, w, wide=jnp.float32):
        x, w = mx_matrix.nvfp4_values(x), mx_matrix.nvfp4_values(w)
        return jax.lax.ragged_dot(x.astype(wide), jnp.swapaxes(w, 1, 2).astype(wide), sizes,
                                  precision=jax.lax.Precision.HIGHEST, preferred_element_type=wide)

    with generation.assumed("sm100"), generation.routes() as routes:
        assert_as_exact(jax.jit(dew), jax.jit(decoded), draw, f"an nvfp4 grouped product, {kernel}",
                        seeds=SEEDS, truth_fn=lambda x, w: decoded(x, w, jnp.float64))
    assert ("nvfp4_block", kernel) in {(route.choice, route.kernel) for route in routes}


def test_an_nvfp4_quantized_projection_is_its_quantized_rows_product():
    """An inference projection of a stored NVFP4 matrix with its rows quantized as it runs
    (`gemm.nvfp4_quantized_product`), rows no whole 128s: bitwise the NVFP4 block product of its rows,
    padded, and the matrix as one group (XLA's arm on a CPU)."""
    x, w = _draw(4, (200, 256)), _draw(5, (256, 128))
    matrix = gemm.NvFp4.quantized(w)
    out = gemm.nvfp4_quantized_product(x, matrix, out_dtype=jnp.float32)
    padded = jnp.pad(x, ((0, 56), (0, 0)))
    expected = gemm.nvfp4_block_product(gemm.NvFp4Block.quantized(padded), gemm.NvFp4Block.of(matrix),
                                        jnp.asarray([2], jnp.int32), out_dtype=jnp.float32)[:200]
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


def test_an_nvfp4_projections_training_step_is_on_par():
    """One training step through the NVFP4 recipe (`gemm.nvfp4_projection`: its output and both gradients),
    the decoded arms in XLA, on par with the recipe in float64 (tests/mx_matrix.py's `recipe_truth` of NVFP4
    values) over the draws (check 1), against the same step's float32 sums of the decoded operands."""
    blocks = jnp.asarray([1, 0, 2], jnp.int32)

    def projection(x, w):
        with generation.chosen(nvfp4_block="xla"):
            return gemm.nvfp4_projection(x, w, blocks, out_dtype=jnp.float32)

    plain = mx_matrix.recipe_truth(blocks, jnp.float32, quantized=mx_matrix.nvfp4_values)
    assert_as_exact(jax.jit(with_gradients(projection)), jax.jit(with_gradients(plain)), step_operands,
                    "nvfp4 training step", seeds=SEEDS,
                    truth_fn=with_gradients(mx_matrix.recipe_truth(blocks, quantized=mx_matrix.nvfp4_values)))
