"""The NVFP4 product's check-1 matrix (tests/kernel_matrix.py): a product of an NVFP4 matrix (`gemm.NvFp4`, a
quantized checkpoint's frozen weights) with its matrix decoded stage by stage in the template's mainloop
(`gemm.nvfp4_product`'s packed kernel, row 3621) on par with its matrix decoded once a call and multiplied.

Each cell runs the product's forward and its reverse mode in one program: the output and the rows' gradient
(the weights frozen; `reference_error.with_gradients`, a drawn cotangent), held to the product with the
selector's other kernel named, `xla`, the decoded matrix's product (`generation.chosen`, check 1), from the
plain product in float64 (the rows times the decoded values times the deferred factor).

The kernel under test is the packed kernel, named where the cells run it (no receipt yet qualifies it, so its
selector, `gemm.NVFP4`, takes it by name alone): the cells' `CHOICES` are `nvfp4`. The rows' gradient and the
`xla` arm are GEMMs whose route, `grouped_matmul`, runs at XLA, named in the cell (`XLA`), so the template's
launches leave this check to their own family's (tests/gemm_matrix.py).

What the route serves is read from the product's own selector: the product traced for its shapes as the
case's generation with the packed kernel named, the route its choice reports the packed kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import gemm, generation

ROWS = 192
"""The product's rows."""
SEEDS = 8
CHOICES = frozenset({"nvfp4"})
"""The `generation.CHOICES` entries the cells run: each cell asserts that the routes it records run `nvfp4`
`packed` and every choice outside `CHOICES` at XLA."""
KERNELS = {"nvfp4": frozenset({"packed"})}
"""The kernels under test of each choice the cells run: the template's in-stage decode."""
XLA = {"grouped_matmul": "xla"}
"""The product's other choice, named XLA in every cell."""
E2M1 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
"""Each E2M1 code's magnitude (its sign the code's high bit)."""


def _matrix(case: dict[str, str]) -> tuple[gemm.NvFp4, np.ndarray]:
    """`case`'s NVFP4 matrix (`gemm.NvFp4.of`: random codes under power-of-two block scales, tensor scale
    0.75) and its values in float64: code times block scale times tensor scale."""
    depth, width = int(case["depth"]), int(case["width"])
    rng = np.random.default_rng(depth + width)
    codes = rng.integers(0, 16, size=(depth, width))
    magnitudes = np.where(codes >= 8, -1.0, 1.0) * E2M1[codes & 7]
    block = np.exp2(rng.integers(-3, 4, size=(depth // 16, width)))
    matrix = gemm.NvFp4.of(jnp.asarray(magnitudes, jnp.float32), jnp.asarray(block, jnp.float8_e4m3fn), 0.75)
    return matrix, magnitudes * np.repeat(block, 16, axis=0) * 0.75


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its product traced for its shapes as its
    generation within an abstract mesh of one of its devices, the packed kernel named and the product at
    XLA, reports the packed kernel for `nvfp4`."""
    matrix, _ = _matrix(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    x = jax.ShapeDtypeStruct((ROWS, int(case["depth"])), jnp.bfloat16)
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(**XLA, nvfp4="packed"), generation.routes() as routes):
        jax.eval_shape(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=case["out_dtype"]), x)
    return [route.kernel for route in routes if route.choice == "nvfp4"] == ["packed"]


CASES = (
    ('sm90', '128', '256', 'float32'),
    ('sm90', '256', '512', 'bfloat16'),
    ('sm80', '128', '256', 'bfloat16'),
    ('sm80', '256', '512', 'float32'),
    ('sm90', '128', '512', 'float32'),
    ('sm90', '256', '256', 'float32'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("nvfp4")`, regenerated where the axes or the route's support change)."""


def _step(matrix: gemm.NvFp4, out_dtype: str, kernel: str):
    """The product's training step as `run(rows, cotangent)`, `nvfp4` named `kernel`."""
    def product(rows):
        with generation.chosen(**XLA, nvfp4=kernel):
            return gemm.nvfp4_product(rows, matrix, out_dtype=out_dtype)
    return with_gradients(product)


def _truth(values: np.ndarray):
    """The plain product's training step in float64: the rows times the matrix's values."""
    wide = jnp.asarray(values, jnp.float64)
    return with_gradients(lambda rows: rows @ wide)


@cell("nvfp4", covers=lambda case: dict(zip(AXES["nvfp4"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def an_nvfp4_products_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The NVFP4 product's training step of `case` on its GPU with its matrix decoded in the mainloop on par
    with it decoded once a call (check 1): the output and the rows' gradient, from the plain product in
    float64. The routes it records run `nvfp4` `packed` and every other choice at XLA."""
    combination = dict(zip(AXES["nvfp4"], case, strict=True))
    matrix, values = _matrix(combination)
    width, depth, out_dtype = int(combination["width"]), int(combination["depth"]), combination["out_dtype"]

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed))
        rows = jax.random.normal(keys[0], (ROWS, depth)).astype(jnp.bfloat16)
        return rows, jax.random.normal(keys[1], (ROWS, width)).astype(out_dtype)

    with generation.routes() as routes:
        assert_as_exact(jax.jit(_step(matrix, out_dtype, "packed")), jax.jit(_step(matrix, out_dtype, "xla")),
                        draw, "-".join(case), seeds=SEEDS, truth_fn=_truth(values))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("nvfp4", "packed") in recorded and not unnamed, (sorted(recorded), unnamed)
