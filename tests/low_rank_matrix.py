"""The adapted product's check-1 matrix (tests/kernel_matrix.py): a LoRA-adapted projection's training step
with its base product and its update as one template product (`gemm.low_rank`, row 1304) on par with the
adapter's own program, its base product and update apart.

Each cell runs a plain Dense projection adapted by `dew.lora.LoRA` as a model's block does, the forward and
its reverse mode in one program: the output and the gradients of the input and of both factors (the base
weights frozen, as an adapter trains them; `reference_error.with_gradients`, a drawn cotangent), held to the
projection with the selector's other kernel named, `xla`, the base product and update apart
(`generation.chosen`, check 1), from the projection's program in float64.

The kernel under test is the adapted product's `fused` form, named where the cells run it (no receipt yet
qualifies it, so its selector, `gemm.LOW_RANK`, takes it by name alone): the cells' `CHOICES` are `low_rank`.
The product it runs is one GEMM, whose route, `grouped_matmul`, runs at XLA, named in the cell (`XLA`), so the
template's launches leave this check to their own family's (tests/gemm_matrix.py).

What the route serves is read from the adapter's own selector: the projection traced for its shapes as the
case's generation with `fused` named, the route its choice reports for the site the `fused` kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact, with_gradients

from dew.lora import LoRA
from dew.nn.dense import Dense
from dew.nn.kernels import generation

ROWS, FEATURES, OUT = 256, 256, 384
"""The projection's rows, the features it reads and the features it writes."""
SEEDS = 8
CHOICES = frozenset({"low_rank"})
"""The `generation.CHOICES` entries the cells run: each cell asserts that the routes it records run `low_rank`
`fused` and every choice outside `CHOICES` at XLA."""
KERNELS = {"low_rank": frozenset({"fused"})}
"""The kernels under test of each choice the cells run: the adapted product's fused form."""
XLA = {"grouped_matmul": "xla"}
"""The projection's other choice, named XLA in every cell."""


class Projection(nn.Module):
    """One plain Dense projection, the site an adapter targets (`proj`)."""

    features: int
    use_bias: bool
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        return Dense(self.features, use_bias=self.use_bias, dtype=self.dtype, param_dtype=self.dtype,
                     name="proj")(x)


def _adapted(case: dict[str, str], dtype=None):
    """`case`'s projection adapted at rank `case["rank"]`, its factors drawn and its base weights frozen, at
    `dtype` (the case's where None): the adapted model and its variables, B drawn nonzero (an adapter's own
    draw zeroes it, which would leave A's gradient zero)."""
    dtype = jnp.dtype(case["dtype"] if dtype is None else dtype)
    model = Projection(OUT, case["bias"] == "bias", dtype)
    variables = model.init(jax.random.key(0), jnp.zeros((ROWS, FEATURES), dtype))
    if case["bias"] == "bias":
        variables = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(5), leaf.shape,
                                                                             leaf.dtype), variables)
    adapter = LoRA(rank=int(case["rank"]), modules=("proj",)).apply(model, variables, key=1)
    return adapter.model, adapter.variables


def _factors(variables) -> dict:
    """The adapter's factors in `variables` (`params`' `lora_A` and `lora_B` leaves), the leaves it trains."""
    return {"params": {"proj": {name: variables["params"]["proj"][name] for name in ("lora_A", "lora_B")}}}


def _merged(variables, factors) -> dict:
    """`variables` with its factors replaced by `factors`."""
    params = {**variables["params"], "proj": {**variables["params"]["proj"], **factors["params"]["proj"]}}
    return {**variables, "params": params}


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its adapted projection traced for its shapes
    as its generation within an abstract mesh of one of its devices, `fused` named and the product at XLA,
    reports the `fused` kernel for `low_rank`."""
    model, variables = _adapted(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    x = jax.ShapeDtypeStruct((ROWS, FEATURES), jnp.dtype(case["dtype"]))
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(**XLA, low_rank="fused"), generation.routes() as routes):
        jax.eval_shape(model.apply, variables, x)
    return [route.kernel for route in routes if route.choice == "low_rank"] == ["fused"]


CASES = (
    ('sm90', '8', 'none', 'bfloat16'),
    ('sm90', '16', 'bias', 'float16'),
    ('sm80', '8', 'bias', 'float32'),
    ('sm80', '16', 'none', 'bfloat16'),
    ('sm90', '16', 'none', 'float32'),
    ('sm80', '8', 'none', 'float16'),
    ('sm90', '8', 'bias', 'bfloat16'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("low_rank")`, regenerated where the axes or the route's support change)."""


def _step(model, variables, kernel: str):
    """The adapted projection's training step as `run(factors, x, cotangent)`, `low_rank` named `kernel`."""
    def projected(factors, x):
        with generation.chosen(**XLA, low_rank=kernel):
            return model.apply(_merged(variables, factors), x)
    return with_gradients(projected)


def _drawn(case: dict[str, str], variables):
    """A seed's factors (each drawn small, B nonzero), input and cotangent, in the case's dtype."""
    dtype = jnp.dtype(case["dtype"])

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 4)
        factors = jax.tree.map(lambda leaf, key: (jax.random.normal(key, leaf.shape) / 16).astype(dtype),
                               _factors(variables), {"params": {"proj": dict(zip(("lora_A", "lora_B"),
                                                                                  keys[:2], strict=True))}})
        x = jax.random.normal(keys[2], (ROWS, FEATURES)).astype(dtype)
        cotangent = jax.random.normal(keys[3], (ROWS, OUT)).astype(dtype)
        return factors, x, cotangent
    return draw


@cell("low_rank", covers=lambda case: dict(zip(AXES["low_rank"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def an_adapted_projections_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The adapted projection's training step of `case` on its GPU with its base product and update fused
    on par with them apart (check 1): the output and the gradients of the factors and the input, from the
    projection in float64. The routes it records run `low_rank` `fused` and every other choice at XLA."""
    combination = dict(zip(AXES["low_rank"], case, strict=True))
    model, variables = _adapted(combination)
    wide_model, wide_variables = _adapted(combination, jnp.float64)
    draw = _drawn(combination, variables)
    with generation.routes() as routes:
        assert_as_exact(jax.jit(_step(model, variables, "fused")), jax.jit(_step(model, variables, "xla")),
                        draw, "-".join(case), seeds=SEEDS, truth_fn=_step(wide_model, wide_variables, "xla"))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("low_rank", "fused") in recorded and not unnamed, (sorted(recorded), unnamed)
