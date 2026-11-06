"""The blockwise-scaled FP8 product's check-1 matrix (tests/kernel_matrix.py): FP8 rows times each group's FP8
matrix, each 128-deep block's partial scaled by its rows' and columns' scales before it is added
(`gemm.blockwise_fp8_product`, DeepSeek-V3's blockwise FP8, row 4518), through the template's exactly summed
walk (`mosaic_gpu.gmm`'s `step_scales`) on par with the scaled blocks summed in XLA.

Each cell runs the product's forward alone: the primitive has no gradient of its own (a recipe's gradients are
FP8 products of their own requantized operands, row 2513). It is held to the product with the selector's other
kernel named, `xla` (`generation.chosen`, check 1), from the float64 sum of the dequantized blocks.

The kernel under test is the template, named where the cells run it (no receipt yet qualifies it, so its
selector, `gemm.BLOCKWISE_FP8`, takes it by name alone): the cells' `CHOICES` are `blockwise_fp8`.

What the route serves is read from the product's own selector: the product traced for its shapes as the
case's generation with the template named, the route its choice reports the template.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation

ROWS, DEPTH, WIDTH = 256, 512, 256
"""The product's rows, depth (four 128-deep blocks) and width."""
SEEDS = 8
CHOICES = frozenset({"blockwise_fp8"})
"""The `generation.CHOICES` entries the cells run: each cell asserts that the routes it records run
`blockwise_fp8` `mosaic_gpu` and every choice outside `CHOICES` at XLA."""
KERNELS = {"blockwise_fp8": frozenset({"mosaic_gpu"})}
"""The kernels under test of each choice the cells run: the template's step-scaled walk."""


def _sizes(groups: int) -> jnp.ndarray:
    """`groups` groups' row counts over `ROWS`, ragged where there are several."""
    return jnp.asarray([ROWS] if groups == 1 else [70, 0, 129, ROWS - 199], jnp.int32)


def _operands(case: dict[str, str], seed: int) -> tuple:
    """`case`'s rows and matrices as FP8 codes, their row scales `[depth / 128, rows]` and column scales
    `[groups, depth / 128, width / per]` (powers of two times a uniform draw)."""
    groups, per = int(case["groups"]), int(case["columns"])
    rng = np.random.default_rng(seed)
    rows = jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.float8_e4m3fn)
    matrices = jnp.asarray(rng.normal(size=(groups, WIDTH, DEPTH)), jnp.float8_e4m3fn)
    row_scales = jnp.asarray(rng.uniform(0.5, 2.0, size=(DEPTH // 128, ROWS)), jnp.float32)
    column_scales = jnp.asarray(rng.uniform(0.5, 2.0, size=(groups, DEPTH // 128, WIDTH // per)), jnp.float32)
    return rows, matrices, row_scales, column_scales


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its product traced for its shapes as its
    generation within an abstract mesh of one of its devices, the template named, reports the template for
    `blockwise_fp8`."""
    rows, matrices, row_scales, column_scales = (jax.ShapeDtypeStruct(array.shape, array.dtype)
                                                 for array in _operands(case, 0))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(blockwise_fp8="mosaic_gpu"), generation.routes() as routes):
        jax.eval_shape(lambda *operands: gemm.blockwise_fp8_product(
            *operands[:2], _sizes(int(case["groups"])), *operands[2:], out_dtype=case["out_dtype"]),
            rows, matrices, row_scales, column_scales)
    return [route.kernel for route in routes if route.choice == "blockwise_fp8"] == ["mosaic_gpu"]


CASES = (
    ('sm90', '128', '1', 'float32'),
    ('sm90', '128', '4', 'bfloat16'),
    ('sm90', '1', '1', 'bfloat16'),
    ('sm90', '1', '4', 'float32'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("blockwise_fp8")`'s pairs)."""


def _product(groups: int, out_dtype: str, kernel: str):
    """The product as `run(rows, matrices, row_scales, column_scales)`, `blockwise_fp8` named `kernel`."""
    def product(rows, matrices, row_scales, column_scales):
        with generation.chosen(blockwise_fp8=kernel):
            return gemm.blockwise_fp8_product(rows, matrices, _sizes(groups), row_scales, column_scales,
                                              out_dtype=out_dtype)
    return product


def truth(sizes) -> callable:
    """The dequantized blocks' scaled sum in float64, each group's rows and each 128-deep block apart (the
    product's definition, written out): `run(rows, matrices, row_scales, column_scales)`."""
    starts = np.cumsum([0, *np.asarray(sizes)])

    def run(rows, matrices, row_scales, column_scales):
        wide = jnp.float64
        rows, matrices = rows.astype(wide), matrices.astype(wide)
        per = matrices.shape[1] // column_scales.shape[-1]
        out = jnp.zeros((rows.shape[0], matrices.shape[1]), wide)
        for group, (first, last) in enumerate(itertools.pairwise(starts)):
            for block in range(row_scales.shape[0]):
                depth = slice(128 * block, 128 * (block + 1))
                partial = rows[first:last, depth] @ matrices[group, :, depth].T
                columns = jnp.repeat(column_scales[group, block].astype(wide), per)
                out = out.at[first:last].add(partial * row_scales[block, first:last, None].astype(wide)
                                             * columns[None, :])
        return out
    return run


@cell("blockwise_fp8", covers=lambda case: dict(zip(AXES["blockwise_fp8"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_blockwise_fp8_product_is_on_par(case: tuple[str, ...]) -> None:
    """The blockwise-scaled FP8 product of `case` on its GPU through the template on par with the scaled
    blocks summed in XLA (check 1), from the float64 sum of the dequantized blocks. The routes it records run
    `blockwise_fp8` `mosaic_gpu` and every other choice at XLA."""
    combination = dict(zip(AXES["blockwise_fp8"], case, strict=True))
    groups, out_dtype = int(combination["groups"]), combination["out_dtype"]
    with generation.routes() as routes:
        assert_as_exact(jax.jit(_product(groups, out_dtype, "mosaic_gpu")),
                        jax.jit(_product(groups, out_dtype, "xla")),
                        lambda seed: _operands(combination, seed), "-".join(case), seeds=SEEDS,
                        truth_fn=truth(_sizes(groups)))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("blockwise_fp8", "mosaic_gpu") in recorded and not unnamed, (sorted(recorded), unnamed)


def recipe_truth(group_blocks) -> callable:
    """The blockwise FP8 recipe's training step in float64 (`gemm.blockwise_fp8_projection`'s truth):
    `run(x, w)` quantizing as the recipe does (`gemm.rows_quantized`, `gemm.blocks_quantized`), each group's
    product of the dequantized values in float64, and its gradients the recipe's: the cotangent quantized 1 x
    128 along the
    width times the dequantized blocks transposed, and the rows and the cotangent each quantized 1 x 128 along
    the rows, their dequantized product; every product in float64 (a check-1 reference arm's truth)."""
    sizes = np.asarray(group_blocks) * 128
    starts = np.cumsum([0, *sizes])

    def dequantized(codes, scales, axis):
        return codes.astype(jnp.float64) * jnp.repeat(scales.astype(jnp.float64), 128, axis=axis)

    def blocks(w):
        codes, scales = gemm.blocks_quantized(w)
        return codes.astype(jnp.float64) * jnp.repeat(jnp.repeat(scales.astype(jnp.float64), 128, axis=1),
                                                      128, axis=2)

    def grouped(lhs, matrices):
        return jnp.concatenate([lhs[first:last] @ matrices[group]
                                for group, (first, last) in enumerate(itertools.pairwise(starts))])

    @jax.custom_vjp
    def run(x, w):
        return grouped(dequantized(*gemm.rows_quantized(x), axis=-1), blocks(w))

    def forward(x, w):
        return run(x, w), (x, w)

    def backward(residuals, cotangent):
        x, w = residuals
        d_x = grouped(dequantized(*gemm.rows_quantized(cotangent), axis=-1), jnp.swapaxes(blocks(w), 1, 2))
        rows_t = dequantized(*gemm.rows_quantized(x.T), axis=-1)
        dy_t = dequantized(*gemm.rows_quantized(cotangent.T), axis=-1)
        d_w = jnp.stack([rows_t[:, first:last] @ dy_t[:, first:last].T
                         for first, last in itertools.pairwise(starts)])
        return d_x.astype(x.dtype), d_w.astype(w.dtype)

    run.defvjp(forward, backward)
    return run
