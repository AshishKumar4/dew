"""The symmetric products' check-1 matrix (tests/kernel_matrix.py): Muon's Newton-Schulz Gram `X @ X^T` and
`X^T @ X` and its polynomial `c0 I + c1 A + c2 A @ A` (`gemm.symmetric_gram`, `gemm.symmetric_polynomial`,
row 11941) through the template's upper-triangle kernel and its mirror (`syrk.MOSAIC_SYRK`), forward and
backward, on par with XLA's dot from float64.

The kernel under test is the template, named where the cells run it (its selector, `gemm.SYRK`, admits it
unnamed only on the measured Gram cells): the cells' `CHOICES` are `syrk`, the symmetric products' own
choice, which naming the grouped products' kernel leaves alone (row 12485). The cells run on the GPU tier
(tests/test_gpu_gemm.py), each on a GPU of its generation; the CPU interprets the kernel in
tests/test_gemm_syrk.py.

What the route serves is read from the products' own selector: the product traced for its shapes as the
case's generation with the template named, the route its choice reports the template.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import pinned
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact

from dew import newton_schulz as ns
from dew.nn.kernels import gemm, generation, selection

SIDE, DEPTH = 256, 512
"""The symmetric output's side and the Gram's contraction depth (the polynomial's is its side)."""
CHOICES = frozenset({"syrk"})
"""The `generation.CHOICES` entries the cells run: each cell asserts that the routes it records run `syrk`
`mosaic` and every choice outside `CHOICES` at XLA."""
KERNELS = {"syrk": frozenset({"mosaic"})}
"""The kernels under test of each choice the cells run: the template's upper-triangle symmetric product."""


def _shape(form: str) -> tuple[int, int, int]:
    """A form's operand: two matrices, `[side, depth]` for the Gram, `[depth, side]` for its transpose, and
    `[side, side]` for the polynomial."""
    shapes = {"gram": (2, SIDE, DEPTH), "gram-transposed": (2, DEPTH, SIDE), "polynomial": (2, SIDE, SIDE)}
    return shapes[form]


def _product(form: str):
    """The form's symmetric product of a bf16 operand, its polynomial's coefficients Newton-Schulz's."""
    def product(a):
        if form == "polynomial":
            return gemm.symmetric_polynomial(a, jnp.asarray(ns.NEWTON_SCHULZ, jnp.float32))
        return gemm.symmetric_gram(a, transpose=form == "gram-transposed")
    return product


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its product traced for its shapes as its
    generation within an abstract mesh of one of its devices, the template named, reports the template for
    `syrk` (a refusal, a call it does not serve)."""
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(syrk="mosaic"), generation.routes() as routes):
        try:
            jax.eval_shape(_product(case["form"]), jax.ShapeDtypeStruct(_shape(case["form"]), jnp.bfloat16))
        except selection.KernelRefused:
            return False
    return [route.kernel for route in routes if route.choice == "syrk"] == ["mosaic"]


CASES = (
    ('sm90', 'gram'),
    ('sm90', 'gram-transposed'),
    ('sm90', 'polynomial'),
    ('sm80', 'gram'),
    ('sm80', 'gram-transposed'),
    ('sm80', 'polynomial'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("syrk")`'s pairs)."""


@cell("syrk", covers=lambda case: dict(zip(AXES["syrk"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pinned(case[0])) for case in CASES], ids="-".join)
def a_symmetric_product_is_on_par(case: tuple[str, ...]) -> None:
    """The symmetric product of `case` on its GPU through the template, forward and backward, on par with
    XLA's dot_general at production widths (check 1) from float64 truth (rows 11941, 5972). The routes it
    records run `syrk` `mosaic` and every other choice at XLA."""
    form = dict(zip(AXES["syrk"], case, strict=True))["form"]
    shape, polynomial = _shape(form), form == "polynomial"
    coeffs = np.asarray(ns.NEWTON_SCHULZ, np.float32)

    def draw(seed):
        rng = np.random.default_rng(11941 + seed)
        arr = rng.normal(size=shape) / np.sqrt(shape[-1])
        if polynomial:
            arr = 0.5 * (arr + np.swapaxes(arr, -1, -2))
        return arr.astype(jnp.bfloat16), rng.normal(size=(2, SIDE, SIDE)).astype(jnp.bfloat16)

    def run(x, cot):
        with generation.chosen(syrk="mosaic"):
            y, pull = jax.vjp(_product(form), x)
            return y, pull(cot)[0]

    def plain(x, cot, *, dtype=jnp.float32, out_dtype=jnp.bfloat16):
        def fn(a):
            if polynomial:
                sq = jax.lax.dot_general(a, a, (((2,), (1,)), ((0,), (0,))), preferred_element_type=dtype)
                c = jnp.asarray(coeffs, dtype)
                return (c[0] * jnp.eye(SIDE, dtype=dtype) + c[1] * a.astype(dtype) + c[2] * sq).astype(
                    out_dtype)
            axis = 1 if form == "gram-transposed" else 2
            return jax.lax.dot_general(a, a, (((axis,), (axis,)), ((0,), (0,))),
                                       preferred_element_type=dtype).astype(out_dtype)

        y, pull = jax.vjp(fn, x.astype(out_dtype))
        return y, pull(cot.astype(out_dtype))[0]

    with generation.routes() as routes:
        assert_as_exact(jax.jit(run), jax.jit(plain), draw, f"native syrk {form}", seeds=8,
                        truth_fn=functools.partial(plain, dtype=jnp.float64, out_dtype=jnp.float64))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("syrk", "mosaic") in recorded and not unnamed, (sorted(recorded), unnamed)
