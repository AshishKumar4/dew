"""The MX product's check-1 matrix (tests/kernel_matrix.py): MXFP8 rows times each group's MXFP8 matrix
(`gemm.mxfp8_product`, row 4519), each 32 depths of each scaled by its E8M0 scale in Blackwell's block-scaled
tensor cores (`mosaic_gpu.gmm`'s `mx_scales`), on par with the decoded operands' product in XLA.

Each cell runs the product's forward alone: a recipe's gradients are their own MX products of requantized
operands (row 2513). It is held to the product with the selector's other kernel named, `xla`
(`generation.chosen`, check 1), from the decoded operands' product in float64.

The kernel under test is the template, named where the cells run it (no receipt yet qualifies it, so its
selector, `gemm.MXFP8`, takes it by name alone): the cells' `CHOICES` are `mxfp8`.

What the route serves is read from the product's own selector: the product traced for its shapes as the
case's generation with the template named, the route its choice reports the template.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blockwise_matrix import grouped
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation

DEPTH = 256
"""The product's depth: eight 32-deep scale blocks."""
SEEDS = 8
CHOICES = frozenset({"mxfp8"})
"""The `generation.CHOICES` entries the cells run: each cell asserts that the routes it records run `mxfp8`
`mosaic_gpu` and every choice outside `CHOICES` at XLA."""
KERNELS = {"mxfp8": frozenset({"mosaic_gpu"})}
"""The kernels under test of each choice the cells run: the template's block-scaled products."""


def _blocks(groups: int) -> jnp.ndarray:
    """`groups` groups' 128-row blocks, an empty group among several."""
    return jnp.asarray([2] if groups == 1 else [1, 0, 2, 1], jnp.int32)


def _operands(case: dict[str, str], seed: int) -> tuple:
    """`case`'s rows and matrices, their 32-depth blocks' magnitudes spread over 2^-6 to 2^6."""
    groups, width = int(case["groups"]), int(case["width"])
    rows = 128 * int(jnp.sum(_blocks(groups)))
    rng = np.random.default_rng(seed)

    def drawn(shape):
        spread = np.exp2(rng.integers(-6, 7, size=(*shape[:-1], shape[-1] // gemm.MX_BLOCK, 1)))
        values = rng.normal(size=(*shape[:-1], shape[-1] // gemm.MX_BLOCK, gemm.MX_BLOCK)) * spread
        return jnp.asarray(values.reshape(shape), jnp.float32)

    return drawn((rows, DEPTH)), drawn((groups, width, DEPTH))


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its product traced for its shapes as its
    generation within an abstract mesh of one of its devices, the template named, reports the template for
    `mxfp8`."""
    rows, matrices = (jax.ShapeDtypeStruct(array.shape, array.dtype) for array in _operands(case, 0))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(mxfp8="mosaic_gpu"), generation.routes() as routes):
        def product(x, w):
            return gemm.mxfp8_product(gemm.MxFp8.quantized(x), gemm.MxFp8.quantized(w),
                                      _blocks(int(case["groups"])), out_dtype=case["out_dtype"])

        jax.eval_shape(product, rows, matrices)
    return [route.kernel for route in routes if route.choice == "mxfp8"] == ["mosaic_gpu"]


CASES = (
    ('sm100', '128', '1', 'float32'),
    ('sm100', '128', '4', 'bfloat16'),
    ('sm100', '256', '1', 'bfloat16'),
    ('sm100', '256', '4', 'float32'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("mxfp8")`'s pairs)."""


def _product(groups: int, out_dtype: str, kernel: str):
    """The product as `run(rows, matrices)`, `mxfp8` named `kernel`."""
    def product(rows, matrices):
        with generation.chosen(mxfp8=kernel):
            return gemm.mxfp8_product(gemm.MxFp8.quantized(rows), gemm.MxFp8.quantized(matrices),
                                      _blocks(groups), out_dtype=out_dtype)
    return product


def _truth(groups: int):
    """The decoded operands' grouped product in float64."""
    def truth(rows, matrices):
        wide = jnp.float64
        x, w = gemm.MxFp8.quantized(rows).decoded(), gemm.MxFp8.quantized(matrices).decoded()
        return jax.lax.ragged_dot(x.astype(wide), jnp.swapaxes(w, 1, 2).astype(wide), 128 * _blocks(groups),
                                  precision=jax.lax.Precision.HIGHEST, preferred_element_type=wide)
    return truth


@cell("mxfp8", covers=lambda case: dict(zip(AXES["mxfp8"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def an_mxfp8_product_is_on_par(case: tuple[str, ...]) -> None:
    """The MX product of `case` on its GPU through the template on par with the decoded operands' product in
    XLA (check 1), from float64. The routes it records run `mxfp8` `mosaic_gpu` and every other choice at
    XLA."""
    combination = dict(zip(AXES["mxfp8"], case, strict=True))
    groups, out_dtype = int(combination["groups"]), combination["out_dtype"]
    with generation.routes() as routes:
        assert_as_exact(jax.jit(_product(groups, out_dtype, "mosaic_gpu")),
                        jax.jit(_product(groups, out_dtype, "xla")),
                        lambda seed: _operands(combination, seed), "-".join(case), seeds=SEEDS,
                        truth_fn=_truth(groups))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("mxfp8", "mosaic_gpu") in recorded and not unnamed, (sorted(recorded), unnamed)


def recipe_truth(group_blocks, wide=jnp.float64) -> callable:
    """The MXFP8 recipe's training step summed in `wide` (float64: `gemm.mxfp8_projection`'s truth):
    `run(x, w)` quantizing as the recipe does (`gemm.MxFp8.quantized` along each product's contraction), each
    group's product of the decoded values, its gradients the recipe's: the cotangent and the matrices
    quantized along the width for the input's, the rows and the cotangent along the tokens for the
    matrices'."""
    starts = np.cumsum([0, *(128 * np.asarray(group_blocks))])

    def decoded(values):
        return gemm.MxFp8.quantized(values).decoded().astype(wide)

    @jax.custom_vjp
    def run(x, w):
        return grouped(group_blocks, decoded(x), jnp.swapaxes(decoded(jnp.swapaxes(w, 1, 2)), 1, 2))

    def forward(x, w):
        return run(x, w), (x, w)

    def backward(residuals, cotangent):
        x, w = residuals
        d_x = grouped(group_blocks, decoded(cotangent), jnp.swapaxes(decoded(w), 1, 2))
        rows, cotangents = decoded(x.T).T, decoded(cotangent.T).T
        d_w = jnp.stack([rows[first:last].T @ cotangents[first:last]
                         for first, last in itertools.pairwise(starts)])
        return d_x.astype(x.dtype), d_w.astype(w.dtype)

    run.defvjp(forward, backward)
    return run
