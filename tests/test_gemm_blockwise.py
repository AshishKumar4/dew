"""The blockwise-scaled FP8 product (`gemm.blockwise_fp8_product`, DeepSeek-V3's blockwise FP8, row 4518)
on the CPU: the template's step-scaled walk, interpreted as an H100's, on par with the scaled blocks summed in
XLA (check 1), from the float64 sum of the dequantized blocks (tests/blockwise_matrix.py's `truth`)."""

import functools

import blockwise_matrix
import jax
import jax.numpy as jnp
import pytest
from reference_error import assert_as_exact

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
