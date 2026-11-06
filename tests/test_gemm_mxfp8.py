"""The MXFP8 operand form (`gemm.MxFp8`: E4M3 codes, an E8M0 power-of-two scale every 32 depths, OCP MX) and
its grouped product's portable arm (`gemm.mxfp8_product`, row 4519), on the CPU."""

import jax
import jax.numpy as jnp
import mx_matrix
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import gemm, generation

SEEDS = 8


def _draw(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    """A normal draw whose 32-depth blocks span magnitudes from 2^-6 to 2^6."""
    rng = np.random.default_rng(seed)
    spread = np.exp2(rng.integers(-6, 7, size=(*shape[:-1], shape[-1] // gemm.MX_BLOCK, 1)))
    return jnp.asarray((rng.normal(size=(*shape[:-1], shape[-1] // gemm.MX_BLOCK, gemm.MX_BLOCK)) * spread)
                       .reshape(shape), jnp.float32)


def test_an_mxfp8_operand_holds_each_blocks_largest_code_in_its_scales_binade():
    """Each 32-depth block's scale puts its largest magnitude in [256, 512) (OCP MX's rule), so its code,
    rounded to E4M3 and saturating at 448, lies in [256, 448]; a block at float32's top binade decodes
    finite; and its values are an MXFP8 operand's already, quantizing them again giving them back."""
    operand = gemm.MxFp8.quantized(_draw(0, (64, 256)))
    largest = jnp.max(jnp.abs(operand.codes.astype(jnp.float32)).reshape(64, -1, gemm.MX_BLOCK), axis=-1)
    assert bool(jnp.all((largest >= 256) & (largest <= 448))), largest
    top = jnp.full((1, gemm.MX_BLOCK), jnp.finfo(jnp.float32).max * 0.99, jnp.float32)
    assert bool(jnp.all(jnp.isfinite(gemm.MxFp8.quantized(top).decoded())))
    again = gemm.MxFp8.quantized(operand.decoded())
    np.testing.assert_array_equal(np.asarray(again.decoded()), np.asarray(operand.decoded()))


GROUP_BLOCKS = (1, 0, 2, 1)
"""Each group's 128-row blocks: a one-block group, an empty one, two blocks, one."""


@pytest.mark.parametrize("kernel", ["xla", "mosaic_gpu"])
def test_an_mxfp8_grouped_product_is_its_decoded_values_product(kernel):
    """Each group's rows times its matrix, both MXFP8, through `kernel` (the template on Blackwell's
    block-scaled tensor cores interpreted as a B200's, `mosaic_gpu.gmm`'s `mx_scales`, or the decoded operands
    in XLA), on par with the decoded values' grouped product in fp32 (check 1), from float64; an empty group's
    rows contribute nothing."""
    blocks = jnp.asarray(GROUP_BLOCKS, jnp.int32)
    sizes = 128 * blocks
    rows, depth, width = 128 * sum(GROUP_BLOCKS), 256, 128

    def draw(seed):
        return _draw(seed, (rows, depth)), _draw(seed + 1, (len(GROUP_BLOCKS), width, depth))

    def dew(x, w):
        with generation.chosen(mxfp8=kernel):
            return gemm.mxfp8_product(gemm.MxFp8.quantized(x), gemm.MxFp8.quantized(w), blocks,
                                      out_dtype=jnp.float32, interpret_on_cpu=kernel == "mosaic_gpu")

    def decoded(x, w, wide=jnp.float32):
        x, w = gemm.MxFp8.quantized(x).decoded(), gemm.MxFp8.quantized(w).decoded()
        return jax.lax.ragged_dot(x.astype(wide), jnp.swapaxes(w, 1, 2).astype(wide), sizes,
                                  precision=jax.lax.Precision.HIGHEST, preferred_element_type=wide)

    with generation.assumed("sm100"), generation.routes() as routes:
        assert_as_exact(jax.jit(dew), jax.jit(decoded), draw, f"an mxfp8 grouped product, {kernel}",
                        seeds=SEEDS, truth_fn=lambda x, w: decoded(x, w, jnp.float64))
    assert ("mxfp8", kernel) in {(route.choice, route.kernel) for route in routes}


def test_an_mxfp8_projections_training_step_is_on_par():
    """One training step through the MXFP8 recipe (`gemm.mxfp8_projection`: its output and both gradients),
    the decoded arms in XLA, on par with the recipe in float64 (tests/mx_matrix.py's `recipe_truth`) over the
    draws (check 1), against the same step's float32 sums of the decoded operands."""
    blocks = jnp.asarray([1, 0, 2], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(384, 256)), jnp.float32),
                jnp.asarray(rng.normal(size=(3, 256, 128)) / 16, jnp.float32),
                jnp.asarray(rng.normal(size=(384, 128)), jnp.float32))

    def projection(x, w):
        with generation.chosen(mxfp8="xla"):
            return gemm.mxfp8_projection(x, w, blocks, out_dtype=jnp.float32)

    plain = mx_matrix.recipe_truth(blocks, jnp.float32)
    assert_as_exact(jax.jit(with_gradients(projection)), jax.jit(with_gradients(plain)), draw,
                    "mxfp8 training step", seeds=SEEDS,
                    truth_fn=with_gradients(mx_matrix.recipe_truth(blocks)))
