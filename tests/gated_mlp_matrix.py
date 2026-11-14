"""The fused gated MLP's check-1 matrix (tests/kernel_matrix.py): a gated MLP's training step with its three
products fused (`dew.nn.gated_mlp`, the choice `gated_mlp`) on par with the same layer's products apart.

Each cell runs a `GatedMLP` as a model's block does, the forward and its reverse mode in one program, its
output and the gradients of its input and every parameter (`reference_error.with_gradients`, a drawn
cotangent) held to the layer with the fused MLP named apart (`generation.chosen`, check 1) from the layer in
float64.

The kernel under test is the fused MLP's, named `fused` where the cells run it (its selector's own pick is the
products apart where no receipt declares the fused MLP faster, `dew.nn.gated_mlp.FUSED_CELLS`), its products
on the GEMM template (`grouped_matmul` named `mosaic`, which the fused MLP needs: `gated_mlp.GatedCase`'s
`launched`) and the gated product's own kernel at XLA (`swiglu`, the products-apart arm's).

What the route serves is read from the fused MLP's own selector: the layer traced for its shapes as the case's
generation with it named, the route its choice reports the `fused` kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import generation
from dew.nn.mlp import GatedMLP

ROWS, FEATURES, HIDDEN = (2, 128), 256, 384
"""The layer's input `[batch, tokens]` rows, the features it reads and writes, its hidden width."""
SEEDS = 8
LIMIT = 1.5
"""A clamped gated product's limit (`GatedMLP.swiglu_limit`), which drawn projections cross."""
CHOICES = frozenset({"gated_mlp"})
"""The `generation.CHOICES` entries the cells run."""
KERNELS = {"gated_mlp": frozenset({"fused"})}
"""The kernels under test of each choice the cells run: the three products fused."""
NAMED = {"grouped_matmul": "mosaic", "swiglu": "xla"}
"""The layer's other choices as every cell names them."""


def _layer(case: dict[str, str]) -> GatedMLP:
    """`case`'s layer: its three projections separate parameters, as its checkpoints store them."""
    limit = LIMIT if case["limit"] == "clamped" else None
    return GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, activation=case["activation"],
                    swiglu_limit=limit, dtype=jnp.dtype(case["dtype"]))


def _input(case: dict[str, str]):
    """`case`'s input, abstract: `ROWS` rows of `FEATURES` in its dtype."""
    return jax.ShapeDtypeStruct((*ROWS, FEATURES), jnp.dtype(case["dtype"]))


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its layer traced for its shapes as its
    generation within an abstract mesh of one of its devices, the fused MLP named, reports the `fused`
    kernel."""
    layer = _layer(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    params = jax.eval_shape(layer.init, jax.random.key(0), _input(case))
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(**NAMED, gated_mlp="fused"), generation.routes() as routes):
        jax.eval_shape(layer.apply, params, _input(case))
    return [route.kernel for route in routes if route.choice == "gated_mlp"] == ["fused"]


CASES = (
    ('sm90', 'swiglu', 'none', 'bfloat16'),
    ('sm90', 'geglu', 'clamped', 'float16'),
    ('sm100', 'swiglu', 'clamped', 'float16'),
    ('sm100', 'geglu', 'none', 'float16'),
    ('sm100', 'geglu', 'clamped', 'bfloat16'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("gated_mlp")`, regenerated where the axes or the route's support change)."""


@cell("gated_mlp", covers=lambda case: dict(zip(AXES["gated_mlp"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_fused_mlps_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The layer's training step of `case` on its GPU with its products fused on par with the products apart
    (check 1): the output and the gradients of the input and every parameter, from the layer apart in
    float64 (XLA's products, which a float64 product runs on). The routes it records run the fused MLP
    `fused`."""
    combination = dict(zip(AXES["gated_mlp"], case, strict=True))
    layer = _layer(combination)
    dtype = jnp.dtype(combination["dtype"])
    params = layer.init(jax.random.key(4), jnp.zeros((*ROWS, FEATURES), dtype))

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed))
        return (params, *(jax.random.normal(drawn, (*ROWS, FEATURES)).astype(dtype) for drawn in keys))

    def under(kernel, run):
        def arm(*operands):
            with generation.chosen(**NAMED, gated_mlp=kernel):
                return run(*operands)
        return arm

    run, truth = with_gradients(layer.apply), with_gradients(layer.clone(dtype=jnp.float64).apply)
    with generation.routes() as routes:
        assert_as_exact(under("fused", run), under("apart", run), draw, "-".join(case), seeds=SEEDS,
                        truth_fn=truth)
    assert ("gated_mlp", "fused") in {(found.choice, found.kernel) for found in routes}
