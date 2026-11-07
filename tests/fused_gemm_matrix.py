"""The fused GEMM's check-1 matrix (tests/kernel_matrix.py): a region's training step through the fuser's
stages on par with the same region's stages run in XLA.

Each cell runs a region through `fused_gemm.fused`, its dot written as `fused_gemm.product`, as a model's
block does: a producer (an RMS norm of the residual stream) into the product's left operand, the product, its
consumers (a bias and a cast, or a SiLU and a residual). The forward and its reverse mode are one program, the
output and the gradient of every differentiable input (`reference_error.with_gradients`, a drawn cotangent)
held to the reference's (check 1), the same region with both stages and the product's launch forced to XLA
(`generation.chosen(gemm_epilogue="xla", gemm_prologue="xla", grouped_matmul="xla")`), from the plain formula
in float64. The cells run three `generation.CHOICES` entries: `gemm_epilogue` and `gemm_prologue`, the kernels
under test, each stage named `fused` where the case fuses it, so a stage the kernels cannot take raises rather
than run XLA beside XLA, and `grouped_matmul`, the launch that runs the product, hooks and all, as its
selector picks it: it runs the template's launch there by design, the product the fused stages are launched
in, so it is kept in the cells' `CHOICES` as a kernel under test, not named XLA.

What the route serves is read from the fusion's own analysis (`fused_gemm.analysis`, `supported`): the region
traced for its shapes as the case's generation, the stages named as the cell names them, each stage the case
fuses reporting the `fused` kernel, no named kernel refusing the region, and the product launched by a kernel
(a launch the kernels take, not XLA's ragged dot).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gemm_matrix import DEPTH, ROWS, SIZES, WIDTH, grouped_product
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact, with_gradients
from test_gemm_template import rms_norm

from dew.nn.kernels import fused_gemm, generation

SEEDS = 8
REFERENCE = {"gemm_epilogue": "xla", "gemm_prologue": "xla", "grouped_matmul": "xla"}
"""The reference path's routes: both stages and the product's launch in XLA."""
CHOICES = frozenset({"gemm_epilogue", "gemm_prologue", "grouped_matmul"})
"""The `generation.CHOICES` entries the cells run (the module's docstring): each cell asserts that its routes
run the stages fused where the case fuses them (`_named`) and the product in a launch of the kernels', not
XLA's, which together reach them all, and that no choice outside `CHOICES` is recorded."""
KERNELS = {"gemm_epilogue": frozenset({"fused"}), "gemm_prologue": frozenset({"fused"}),
           "grouped_matmul": frozenset({"mosaic_gpu"})}
"""The kernels under test of each choice the cells run: each stage fused, the product the template's
launches."""


def _wide(x):
    """`x` in at least fp32."""
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def _bias_cast(product, bias, dtype):
    """The bias consumer: the product plus a column vector, cast."""
    return (product + bias[None, :]).astype(dtype)


def _silu_residual(product, residual, dtype):
    """The residual consumer: a SiLU of twice the product, plus a residual of the product's shape, cast."""
    return (jax.nn.silu(product * 2) + _wide(residual)).astype(dtype)


CONSUMERS = {"bias-cast": _bias_cast, "silu-residual": _silu_residual}
"""The consumer each value of the `consumer` axis names."""


def _produced(case: dict[str, str]) -> bool:
    """Whether `case`'s region has a producer: the stages that fuse one (the prologue's, or both's)."""
    return case["stage"] != "epilogue"


def _region(case: dict[str, str]):
    """`case`'s region as a function of its operands (the rows, the matrices, the producer's weight where it
    has one, the consumer's operand): the producer's rows into the product, the consumer after it. One
    function a consumer, so every case's call of it is the one the fusion's analyses are kept for."""
    dtype = jnp.dtype(case["dtype"])
    consumer = CONSUMERS[case["consumer"]]

    def region(x, w, *rest):
        operand = rms_norm(x, rest[0]) if _produced(case) else x
        sizes = jnp.asarray(SIZES, jnp.int32)
        return consumer(fused_gemm.product(operand, w, sizes, out_dtype=jnp.float32), rest[-1], dtype)

    return region


def _named(case: dict[str, str]) -> dict[str, str]:
    """The stages' routes `case` names: `fused` for each stage it fuses; the consumers after the product in
    XLA where the case fuses the producer alone."""
    return {**({"gemm_prologue": "fused"} if _produced(case) else {}),
            "gemm_epilogue": "xla" if case["stage"] == "prologue" else "fused"}


def _plain(case: dict[str, str]):
    """`case`'s region as the plain formula: the producer's rows times each group's matrix by a mask over the
    groups, the consumer after. Its operands are the draws' own dtype's values, so a float64 run of it needs
    no rounding first."""
    consumer = CONSUMERS[case["consumer"]]

    def plain(x, w, *rest):
        operand = rms_norm(x, rest[0]) if _produced(case) else x
        return consumer(grouped_product(operand, w), rest[-1], x.dtype)

    return plain


def _spec(case: dict[str, str]) -> tuple[jax.ShapeDtypeStruct, ...]:
    """The shapes and dtypes of `case`'s operands: the rows, the three groups' matrices, the producer's weight
    (fp32) where it has one, the consumer's operand (a bias, fp32, or a residual in the dtype) and the
    output's cotangent last."""
    dtype = jnp.dtype(case["dtype"])
    spec = functools.partial(jax.ShapeDtypeStruct, dtype=dtype)
    weight = [jax.ShapeDtypeStruct((DEPTH,), jnp.float32)] if _produced(case) else []
    consumed = (jax.ShapeDtypeStruct((WIDTH,), jnp.float32) if case["consumer"] == "bias-cast"
                else spec((ROWS, WIDTH)))
    return spec((ROWS, DEPTH)), spec((len(SIZES), DEPTH, WIDTH)), *weight, consumed, spec((ROWS, WIDTH))


def _drawn(case: dict[str, str]):
    """A draw of `case`'s operands by seed: unit Gaussians in each one's dtype, the matrices scaled to unit
    sums, a producer's weight between a half and one and a half."""
    def draw(seed: int):
        rng = np.random.default_rng(seed)
        x, w, *rest, dout = _spec(case)

        def normal(spec, scale: float = 1.0):
            return jnp.asarray(rng.normal(size=spec.shape) * scale, spec.dtype)

        *weight, consumed = rest
        weights = [jnp.asarray(rng.uniform(0.5, 1.5, size=spec.shape), jnp.float32) for spec in weight]
        return normal(x), normal(w, 1 / np.sqrt(DEPTH)), *weights, normal(consumed), normal(dout)

    return draw


def _call(case: dict[str, str], routes: dict[str, str]):
    """`case`'s region through `fused_gemm.fused`, as a function of its operands, its stages and launch
    named `routes` (`generation.chosen`)."""
    region = _region(case)

    def call(*operands):
        with generation.chosen(**routes):
            return fused_gemm.fused(region, *operands)

    return call


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): `fused_gemm.analysis` of its region, traced
    as its generation within an abstract mesh of one of its devices, the stages named as the cell names them
    (`_named`), reports no refusal, `fused` for each stage the case fuses, and a launch other than XLA's."""
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(**_named(case))):
        found = fused_gemm.analysis(_region(case), *_spec(case)[:-1])
    wanted = {"epilogue": found.epilogue, "prologue": found.prologue}
    stages = [wanted[stage] for stage in wanted if case["stage"] in (stage, "both")]
    fused = all(route is not None and route.kernel == "fused" for route in stages)
    return found.refused is None and fused and found.launch is not None and found.launch.kernel != "xla"


CASES = (
    ('sm90', 'epilogue', 'bias-cast', 'bfloat16'),
    ('sm90', 'prologue', 'silu-residual', 'float16'),
    ('sm80', 'both', 'bias-cast', 'float16'),
    ('sm80', 'epilogue', 'silu-residual', 'bfloat16'),
    ('sm90', 'both', 'silu-residual', 'bfloat16'),
    ('sm80', 'prologue', 'bias-cast', 'bfloat16'),
    ('sm90', 'epilogue', 'bias-cast', 'float16'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("fused_gemm")`, regenerated where the axes or the route's support change)."""


@cell("fused_gemm", covers=lambda case: dict(zip(AXES["fused_gemm"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_fused_gemm_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The fused region's training step of `case` on its GPU on par with the same region's stages in XLA
    (check 1): the output and the gradients of the rows, the matrices, the producer's weight and the
    consumer's operand, from the plain formula in float64; its routes run the stages the case fuses and the
    product's launch in the kernels', nothing else (`CHOICES`)."""
    combination = dict(zip(AXES["fused_gemm"], case, strict=True))
    with generation.routes() as routes:
        assert_as_exact(jax.jit(with_gradients(_call(combination, _named(combination)))),
                        jax.jit(with_gradients(_call(combination, REFERENCE))), _drawn(combination),
                        "-".join(case), seeds=SEEDS, truth_fn=with_gradients(_plain(combination)))
    recorded = {(found.choice, found.kernel) for found in routes}
    fused = {(stage, "fused") for stage, kernel in _named(combination).items() if kernel == "fused"}
    launched = any(choice == "grouped_matmul" and kernel != "xla" for choice, kernel in recorded)
    assert fused <= recorded and launched, (sorted(recorded), sorted(fused))
    assert {choice for choice, _ in recorded} <= CHOICES, sorted(recorded)
