"""The joined gate and up projections' check-1 matrix (tests/kernel_matrix.py): a gated MLP's training step
with its projections run as one GEMM on par with the same layer's projections apart.

Each cell runs a `GatedMLP` as a model's block does, the forward and its reverse mode in one program: the
output and the gradients of its input and every parameter (`reference_error.with_gradients`, a drawn
cotangent) held to the layer with the join named apart (`generation.chosen(gate_up="apart")`, check 1) from
the layer in float64 with them apart, as tests/test_joined.py holds a join. The cells run two
`generation.CHOICES` entries: `gate_up`, named `joined` where they run the join (the selector's own pick is
the projections apart where the join measured no faster on the generation, `dew.nn.joined.JOINED_FASTER`),
and `swiglu`, the packed product that reads the joined output in place (`packed_gated_product`) where the
activation is SwiGLU, as its selector picks it (a bf16 product's the fused kernel's, any other XLA's).

What the route serves is read from the join's own selector: the layer traced for its shapes as the case's
generation with the join named, the route `gate_up` reports for its projections (`joined.join_case`: plain
Dense layers of one width, bias, dtype and precision, holding their own kernels) the `joined` kernel, not the
projections apart a failed requirement hands them to.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import with_gradients
from test_joined import assert_joined_as_exact, drawing

from dew.nn.kernels import generation
from dew.nn.mlp import GatedMLP

ROWS, FEATURES, HIDDEN = (2, 128), 256, 384
"""The layer's input `[batch, tokens]` rows, the features it reads and writes, and its hidden width."""
SEEDS = 8
GATE_MULTIPLIER = 0.75
"""A scaled gate's multiplier (Falcon-H1's `mlp_multipliers`)."""


def _layer(case: dict[str, str]) -> GatedMLP:
    """`case`'s gated MLP: its gate and up projections separate parameters, as its checkpoints store them."""
    return GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, use_bias=case["bias"] == "bias",
                    activation=case["activation"],
                    gate_multiplier=GATE_MULTIPLIER if case["gate"] == "scaled" else 1.0,
                    dtype=jnp.dtype(case["dtype"]))


def _input(case: dict[str, str]):
    """`case`'s input, abstract: `ROWS` rows of `FEATURES` in its dtype."""
    return jax.ShapeDtypeStruct((*ROWS, FEATURES), jnp.dtype(case["dtype"]))


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its layer traced for its shapes as its
    generation within an abstract mesh of one of its devices, the join named, reports the `joined` kernel for
    its projections."""
    layer = _layer(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    params = jax.eval_shape(layer.init, jax.random.key(0), _input(case))
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(gate_up="joined"), generation.routes() as routes):
        jax.eval_shape(layer.apply, params, _input(case))
    return [route.kernel for route in routes if route.choice == "gate_up"] == ["joined"]


CASES = ()
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("gate_up")`, regenerated where the axes or the route's support change)."""


@cell("gate_up", covers=lambda case: dict(zip(AXES["gate_up"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_gated_mlps_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The gated MLP's training step of `case` on its GPU with its gate and up projections joined on par with
    the projections apart (check 1): the output and the gradients of the input and every parameter, from the
    layer with them apart in float64. Drawn biases too, which init leaves at zero."""
    combination = dict(zip(AXES["gate_up"], case, strict=True))
    layer = _layer(combination)
    dtype = jnp.dtype(combination["dtype"])
    params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(6), leaf.shape),
                          layer.init(jax.random.key(4), jnp.zeros((*ROWS, FEATURES), dtype)))
    assert_joined_as_exact("gate_up", with_gradients(layer.apply), drawing(params, (*ROWS, FEATURES), dtype),
                           with_gradients(layer.clone(dtype=jnp.float64).apply), "-".join(case), seeds=SEEDS)
