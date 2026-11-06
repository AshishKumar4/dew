"""The blockwise-scaled FP8 product (`gemm.blockwise_fp8_product`, DeepSeek-V3's blockwise FP8, row 4518)
on the CPU: the template's step-scaled walk, interpreted as an H100's, on par with the scaled blocks summed in
XLA (check 1), from the float64 sum of the dequantized blocks (tests/blockwise_matrix.py's `truth`)."""

import functools

import blockwise_matrix
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import gemm, generation


@pytest.mark.parametrize("groups", ["1", "4"])
@pytest.mark.parametrize("columns", ["128", "1"])
def test_a_blockwise_fp8_product_through_the_template_is_on_par(columns, groups):
    """On an H100, interpreted, the product of FP8 rows and each group's FP8 matrix, each 128-deep block's
    partial times its rows' and its group's column scales (a weight block's, or each column's) before it is
    added, through the template named (`mosaic_gpu.gmm`'s `step_scales`), on par with XLA's (check 1), from
    float64; the routes it records name the template."""
    case = {"columns": columns, "groups": groups}
    sizes = blockwise_matrix._sizes(int(groups))

    def product(kernel, *operands):
        with generation.chosen(blockwise_fp8=kernel):
            return gemm.blockwise_fp8_product(*operands[:2], sizes, *operands[2:], out_dtype=jnp.float32,
                                              interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.routes() as routes:
        assert_as_exact(jax.jit(functools.partial(product, "mosaic_gpu")),
                        jax.jit(functools.partial(product, "xla")),
                        lambda seed: blockwise_matrix._operands(case, seed), f"blockwise {columns} {groups}",
                        seeds=blockwise_matrix.SEEDS, truth_fn=blockwise_matrix.truth(sizes))
    assert ("blockwise_fp8", "mosaic_gpu") in {(route.choice, route.kernel) for route in routes}


def test_a_blockwise_fp8_kernel_gradient_through_the_template_is_on_par():
    """`a_blockwise_kernel_gradient_is_on_par` on an H100, interpreted."""
    with generation.assumed("sm90"):
        a_blockwise_kernel_gradient_is_on_par(interpret=True)


def a_blockwise_kernel_gradient_is_on_par(interpret: bool = False):
    """Each group's rows of FP8 operands (rows contiguous) contracted over its whole 128-row blocks, each
    block's partial times its depth rows' and its columns' scales before it is added to its group's sum,
    through the template named (`mosaic_gpu.tgmm`'s `k_major` and `step_scales`), on par with XLA's (check 1),
    from float64; an empty group's gradient is zero. On this GPU, or `interpret`ed (test_gpu_gemm's cell)."""
    group_blocks = jnp.asarray([1, 0, 2, 1], jnp.int32)
    depth, width, blocks = 256, 256, 4

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(depth, 128 * blocks)), jnp.float8_e4m3fn),
                jnp.asarray(rng.normal(size=(width, 128 * blocks)), jnp.float8_e4m3fn),
                jnp.asarray(rng.uniform(0.5, 2.0, size=(blocks, depth)), jnp.float32),
                jnp.asarray(rng.uniform(0.5, 2.0, size=(blocks, width)), jnp.float32))

    def gradient(kernel, *operands):
        with generation.chosen(blockwise_fp8=kernel):
            return gemm.blockwise_fp8_kernel_gradient(*operands[:2], group_blocks, *operands[2:],
                                                      out_dtype=jnp.float32, interpret_on_cpu=interpret)

    def truth(lhs_t, rhs_t, row_scales, column_scales):
        wide = jnp.float64
        out = jnp.zeros((len(group_blocks), depth, width), wide)
        owners = np.repeat(np.arange(len(group_blocks)), np.asarray(group_blocks))
        for block, owner in enumerate(owners):
            rows = slice(128 * block, 128 * (block + 1))
            partial = lhs_t[:, rows].astype(wide) @ rhs_t[:, rows].astype(wide).T
            out = out.at[owner].add(partial * row_scales[block, :, None].astype(wide)
                                    * column_scales[block, None, :].astype(wide))
        return out

    with generation.routes() as routes:
        assert_as_exact(jax.jit(functools.partial(gradient, "mosaic_gpu")),
                        jax.jit(functools.partial(gradient, "xla")), draw, "blockwise kernel gradient",
                        seeds=blockwise_matrix.SEEDS, truth_fn=truth)
    assert ("blockwise_fp8", "mosaic_gpu") in {(route.choice, route.kernel) for route in routes}


def test_a_blockwise_fp8_projections_training_step_is_on_par():
    """On an H100, interpreted, one training step through the blockwise FP8 recipe
    (`gemm.blockwise_fp8_projection`: its output and both gradients), the template named, on par with the
    recipe through XLA's arms (check 1), from the recipe in float64 (tests/blockwise_matrix.py's
    `recipe_truth`): the step-level parity a preset's projections hold (row 2513)."""
    group_blocks = jnp.asarray([1, 0, 2], jnp.int32)

    def step(kernel):
        def projection(x, w):
            with generation.chosen(blockwise_fp8=kernel):
                return gemm.blockwise_fp8_projection(x, w, group_blocks, out_dtype=jnp.float32,
                                                     interpret_on_cpu=True)
        return with_gradients(projection)

    with generation.assumed("sm90"):
        assert_as_exact(jax.jit(step("mosaic_gpu")), jax.jit(step("xla")), blockwise_matrix.step_operands,
                        "blockwise fp8 training step",
                        seeds=blockwise_matrix.SEEDS,
                        truth_fn=with_gradients(blockwise_matrix.recipe_truth(group_blocks)))
