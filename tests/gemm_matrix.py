"""GEMM's check-1 matrix (tests/kernel_matrix.py): the template route's training step on par with the
reference.

Each cell runs `gemm.gemm` as a user's training step does, the forward and its reverse mode in one program,
so the products' launches and the hooks' VJPs run in the program a step builds around them. The output and
the gradient of every differentiable input (`reference_error.with_gradients`, a drawn cotangent) are held to
the reference path's (check 1), the same call with the route forced to XLA's ragged dot, from the plain
formula in float64: a product masked by each row's group, its hook applied. The cells run the one
`generation.CHOICES` entry the call reports, `grouped_matmul`; the epilogue hooks are the call's own
arguments, not `gemm_epilogue`, which is the fuser route's (`fused_gemm`).

The route is the template's where its Mosaic launches take every product of the step, named
(`generation.chosen(grouped_matmul="mosaic_gpu")`: the selector's own pick of a dense product is XLA's dot
where it measured faster, and of an H100's grouped one CUTLASS's, so the default route would hold those
against XLA); a named kernel that refuses any product raises, so where one does (a hook whose cotangent is
wider than the rows, a row scale's or a SwiGLU's, whose input and kernel gradients multiply it by the matrix
in fp32, which no Mosaic launch takes) the selector's own pick is the route, and the case is served where its
forward is the template's. What the route serves is read from the selector, not listed (`supported`): the
step traced abstractly as the generation a case names, which the selector's requirements of each product
answer.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import gemm, generation, selection

ROWS, DEPTH, WIDTH = 256, 256, 384
SIZES = (100, 0, 156)
"""A grouped call's three groups' rows: an empty one between two, covering every row."""
SEEDS = 8
TEMPLATE = "mosaic_gpu"
"""The grouped selector's name for the template's launches, which the cells name where they run them."""
REFERENCE = "xla"
"""The selector's name for XLA's ragged dot, the reference path's route."""
CHOICES = frozenset({"grouped_matmul"})
"""The `generation.CHOICES` entries the cells run, the kernel under test the template's launches (the module's
docstring): each cell asserts that its routes run them (`TEMPLATE`) and that any choice outside `CHOICES` runs
at XLA, which none does."""


def _matrices(case: dict[str, str]) -> tuple[int, ...]:
    """`case`'s matrix: `[groups, depth, width]` under a grouped call, without the groups axis under a dense
    one, its last two axes swapped where it is K-major (`transpose_rhs`)."""
    shape = (WIDTH, DEPTH) if case["matrix"] == "k-major" else (DEPTH, WIDTH)
    return (len(SIZES), *shape) if case["form"] == "grouped" else shape


def _hook_shapes(case: dict[str, str]) -> tuple[tuple[int, ...], ...]:
    """The shapes of the operands `case`'s hook takes beside the product's: a bias a group, a scale a row, a
    second matrix of the first's shape, or none."""
    return {"plain": (), "bias": ((len(SIZES), WIDTH) if case["form"] == "grouped" else (WIDTH,),),
            "row-scale": ((ROWS,),), "paired": (_matrices(case),)}[case["hook"]]


def _biased(acc, bias):
    """The bias hook: the accumulator plus a column vector (a group's, under a grouped call)."""
    return acc + bias


def _scaled(acc, scale):
    """The row-scale hook: each row of the accumulator times its scale."""
    return acc * scale[:, None]


def _gated(gate, up):
    """The paired hook: SwiGLU of the two accumulators."""
    return jax.nn.silu(gate) * up


def _operands(case: dict[str, str], extras: tuple) -> dict:
    """The keyword arguments `gemm.gemm` takes `case`'s hook through, over its drawn `extras`: one function
    a hook, so every case's call of it is the one the template's analyses of a hook are kept for."""
    grouped = case["form"] == "grouped"
    match case["hook"]:
        case "bias":
            return {"epilogue": _biased,
                    "epilogue_operands": ((extras[0], ("groups", "columns") if grouped else ("columns",)),)}
        case "row-scale":
            return {"epilogue": _scaled, "epilogue_operands": ((extras[0], ("rows",)),)}
        case "paired":
            return {"epilogue": _gated, "paired": extras[0]}
    return {}


def _call(case: dict[str, str], kernel: str | None):
    """`case`'s `gemm.gemm` call as a function of its operands (the rows, the matrix, then the hook's), its
    grouped selector named `kernel` (the template's or the reference's route), or deciding where None."""
    dtype = jnp.dtype(case["dtype"])

    def call(x, matrix, *extras):
        sizes = jnp.asarray(SIZES, jnp.int32) if case["form"] == "grouped" else None
        with generation.chosen(grouped_matmul=kernel) if kernel else contextlib.nullcontext():
            return gemm.gemm(x, matrix, sizes, transpose_rhs=case["matrix"] == "k-major", out_dtype=dtype,
                             **_operands(case, extras))

    return call


def grouped_product(x, matrix):
    """Each row's product with its group's matrix `[groups, depth, width]` (`SIZES`), by a mask over the
    groups: the grouped product as written, for the matrices' float64 truths."""
    return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(_groups(), len(SIZES), dtype=x.dtype), x, matrix)


def _plain(case: dict[str, str]):
    """`case`'s call as the plain formula: each row's product with its group's matrix, by a mask over the
    groups (a dense call's one matrix whole), then the hook. Its operands are the draws' own dtype's values,
    so a float64 run of it needs no rounding first."""
    def product(x, matrix):
        matrix = jnp.swapaxes(matrix, -1, -2) if case["matrix"] == "k-major" else matrix
        return x @ matrix if case["form"] == "dense" else grouped_product(x, matrix)

    def plain(x, matrix, *extras):
        out = product(x, matrix)
        match case["hook"]:
            case "bias":
                return out + (extras[0][_groups()] if case["form"] == "grouped" else extras[0])
            case "row-scale":
                return out * extras[0][:, None]
            case "paired":
                return jax.nn.silu(out) * product(x, extras[0])
        return out

    return plain


def _groups() -> jax.Array:
    """Each row's group, the rows in order."""
    return jnp.asarray(np.repeat(np.arange(len(SIZES)), SIZES))


def _spec(case: dict[str, str]) -> tuple[jax.ShapeDtypeStruct, ...]:
    """The shapes and dtypes of `case`'s operands: the rows, the matrix, the hook's (a paired one's matrix in
    the dtype, a bias and a scale fp32), and the output's cotangent last."""
    dtype = jnp.dtype(case["dtype"])
    hooked = [jax.ShapeDtypeStruct(shape, dtype if case["hook"] == "paired" else jnp.float32)
              for shape in _hook_shapes(case)]
    return (jax.ShapeDtypeStruct((ROWS, DEPTH), dtype), jax.ShapeDtypeStruct(_matrices(case), dtype), *hooked,
            jax.ShapeDtypeStruct((ROWS, WIDTH), dtype))


def _drawn(case: dict[str, str]):
    """A draw of `case`'s operands by seed: unit Gaussians in each one's dtype, a matrix (the paired one too)
    scaled to unit sums, and a scale positive, between a half and two."""
    def draw(seed: int):
        rng = np.random.default_rng(seed)
        x, matrix, *hooked, dout = _spec(case)

        def normal(spec, scale: float = 1.0):
            return jnp.asarray(rng.normal(size=spec.shape) * scale, spec.dtype)

        if case["hook"] == "row-scale":
            hooks = [jnp.asarray(rng.uniform(0.5, 2.0, size=spec.shape), spec.dtype) for spec in hooked]
        else:
            hooks = [normal(spec, 1 / np.sqrt(DEPTH) if case["hook"] == "paired" else 1.0) for spec in hooked]
        return normal(x), normal(matrix, 1 / np.sqrt(DEPTH)), *hooks, normal(dout)

    return draw


def _reported(case: dict[str, str], kernel: str | None, *,
              forward: bool) -> list[str] | selection.KernelRefused:
    """The kernels the grouped selector reports, in order, for `case`'s call (its training step, unless
    `forward`) named `kernel`, traced abstractly as `case`'s generation within an abstract mesh of one of its
    devices (`generation.assumed`); the refusal where the named kernel refuses a product."""
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    call = _call(case, kernel)
    spec = _spec(case)
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.routes() as routes):
        try:
            jax.eval_shape(call if forward else with_gradients(call), *(spec[:-1] if forward else spec))
        except selection.KernelRefused as refusal:
            return refusal
    return [route.kernel for route in routes if route.choice == "grouped_matmul"]


def route_of(case: dict[str, str]) -> str | None:
    """The kernel the cell names for `case`'s route (`TEMPLATE`), or None where the selector decides: the
    template where it takes every product of the training step, the forward, the input gradient and the
    kernel gradient (one that does not, named, raises)."""
    reported = _reported(case, TEMPLATE, forward=False)
    return TEMPLATE if isinstance(reported, list) and set(reported) == {TEMPLATE} else None


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its forward is a launch of the template's
    (`route_of`: the whole step's, or the selector's own pick of the forward, which is no declared cell's old
    route, and not the paired product's two apart)."""
    forward = _reported(case, route_of(case), forward=True)
    return isinstance(forward, list) and forward[:1] == [TEMPLATE]


CASES = (
    ('sm90', 'dense', 'n-major', 'plain', 'bfloat16'),
    ('sm90', 'grouped', 'k-major', 'bias', 'float16'),
    ('sm80', 'grouped', 'n-major', 'paired', 'bfloat16'),
    ('sm80', 'dense', 'k-major', 'plain', 'float16'),
    ('sm90', 'grouped', 'n-major', 'row-scale', 'float16'),
    ('sm90', 'dense', 'n-major', 'bias', 'bfloat16'),
    ('sm90', 'grouped', 'k-major', 'row-scale', 'bfloat16'),
    ('sm90', 'grouped', 'k-major', 'paired', 'float16'),
    ('sm90', 'grouped', 'n-major', 'plain', 'bfloat16'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("gemm")`, regenerated where the axes or the route's support change)."""


@cell("gemm", covers=lambda case: dict(zip(AXES["gemm"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_gemm_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The template route's training step of `case` on its GPU on par with the reference path's (check 1):
    the output and the gradients of the rows, the matrix and the hook's operands, from the plain formula in
    float64; its routes run the template's launches (`CHOICES`)."""
    combination = dict(zip(AXES["gemm"], case, strict=True))
    route = route_of(combination)
    with generation.routes() as routes:
        assert_as_exact(jax.jit(with_gradients(_call(combination, route))),
                        jax.jit(with_gradients(_call(combination, REFERENCE))), _drawn(combination),
                        "-".join(case), seeds=SEEDS, truth_fn=with_gradients(_plain(combination)))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("grouped_matmul", TEMPLATE) in recorded and not unnamed, (sorted(recorded), unnamed)
