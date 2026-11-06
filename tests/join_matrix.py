"""The joined projections' check-1 matrix (tests/kernel_matrix.py): a layer's training step with two of its
projections of one input run as one GEMM on par with the same layer's projections apart.

Each cell runs a layer as a model's block does, the forward and its reverse mode in one program: a gated MLP,
whose gate and up projections the join `gate_up` runs as one, or a gated delta net, whose b and a projections
the join `delta_gates` does (`dew.nn.joined.JOINS`, one mechanism, one selector each). The output and the
gradients of its input and every parameter (`reference_error.with_gradients`, a drawn cotangent) are held to
the layer with the join named apart (`generation.chosen`, check 1) from the layer in float64 with them apart,
as tests/test_joined.py holds a join.

The kernels under test are the joins', named `joined` where the cells run them (the selector's own pick is the
projections apart where the join measured no faster on the generation, `dew.nn.joined.JOINED_FASTER`): the
cells' `CHOICES` are `gate_up` and `delta_gates`, a cell running its own. Every other choice a layer reports
(the products' `grouped_matmul`, the gated product's `swiglu`, the delta net's `rms_norm`, `short_conv` and
`gated_delta_rule`) runs at XLA, named in the cell (`XLA`), so the layer's other kernels leave the join's
check as it is.

What the route serves is read from the join's own selector: the layer traced for its shapes as the case's
generation with the join named, the route its choice reports for the layer's projections
(`joined.join_case`: plain Dense layers of one width, bias, dtype and precision, holding their own kernels)
the `joined` kernel, not the projections apart a failed requirement hands them to. An axis a join's layer has
no use of (a delta net has no gated product, bias or gate multiplier) takes its one value that says so,
`none` or `plain`, and no other.
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
from dew.nn.linear import GatedDeltaNet
from dew.nn.mlp import GatedMLP

ROWS, FEATURES, HIDDEN = (2, 128), 256, 384
"""The layer's input `[batch, tokens]` rows, the features it reads and writes, a gated MLP's hidden width."""
HEADS, HEAD_WIDTH = (2, 4), 64
"""A gated delta net's key and value heads, and the width of each head's key and value."""
SEEDS = 8
GATE_MULTIPLIER = 0.75
"""A scaled gate's multiplier (Falcon-H1's `mlp_multipliers`)."""
CHOICES = frozenset({"gate_up", "delta_gates"})
"""The `generation.CHOICES` entries the cells run, a cell its case's join: each cell asserts that the routes
it records run that join `joined` and that every choice outside `CHOICES` runs at XLA, which together reach
them all."""
XLA = dict.fromkeys(("grouped_matmul", "swiglu", "rms_norm", "short_conv", "gated_delta_rule"), "xla")
"""The layers' other choices, named XLA in every cell."""


def _applies(case: dict[str, str]) -> bool:
    """Whether `case` names a layer its join has: a gated MLP's activation and options, or a delta net's none
    (`none` activation, no bias, a plain gate)."""
    if case["join"] == "gate_up":
        return case["activation"] != "none"
    return (case["activation"], case["bias"], case["gate"]) == ("none", "none", "plain")


def _layer(case: dict[str, str]) -> GatedMLP | GatedDeltaNet:
    """`case`'s layer: its two projections separate parameters, as its checkpoints store them."""
    dtype = jnp.dtype(case["dtype"])
    if case["join"] == "delta_gates":
        return GatedDeltaNet(emb_features=FEATURES, num_k_heads=HEADS[0], num_v_heads=HEADS[1],
                             head_k_dim=HEAD_WIDTH, head_v_dim=HEAD_WIDTH, dtype=dtype)
    return GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, use_bias=case["bias"] == "bias",
                    activation=case["activation"],
                    gate_multiplier=GATE_MULTIPLIER if case["gate"] == "scaled" else 1.0, dtype=dtype)


def _input(case: dict[str, str]):
    """`case`'s input, abstract: `ROWS` rows of `FEATURES` in its dtype."""
    return jax.ShapeDtypeStruct((*ROWS, FEATURES), jnp.dtype(case["dtype"]))


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its layer, which its join has (`_applies`),
    traced for its shapes as its generation within an abstract mesh of one of its devices, the join named and
    the layer's other choices at XLA, reports the `joined` kernel for its join."""
    if not _applies(case):
        return False
    layer = _layer(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    params = jax.eval_shape(layer.init, jax.random.key(0), _input(case))
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(**XLA, **{case["join"]: "joined"}), generation.routes() as routes):
        jax.eval_shape(layer.apply, params, _input(case))
    return [route.kernel for route in routes if route.choice == case["join"]] == ["joined"]


CASES = ()
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("join")`, regenerated where the axes or the route's support change)."""


@cell("join", covers=lambda case: dict(zip(AXES["join"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_joined_layers_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The layer's training step of `case` on its GPU with its two projections joined on par with the
    projections apart (check 1): the output and the gradients of the input and every parameter, from the layer
    with them apart in float64. A gated MLP's biases are drawn too, which init leaves at zero. The routes it
    records run its join `joined` and every other choice at XLA (`CHOICES`)."""
    combination = dict(zip(AXES["join"], case, strict=True))
    layer = _layer(combination)
    dtype = jnp.dtype(combination["dtype"])
    params = layer.init(jax.random.key(4), jnp.zeros((*ROWS, FEATURES), dtype))
    if combination["join"] == "gate_up":
        params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(6), leaf.shape),
                              params)
    with generation.chosen(**XLA), generation.routes() as routes:
        assert_joined_as_exact(combination["join"], with_gradients(layer.apply),
                               drawing(params, (*ROWS, FEATURES), dtype),
                               with_gradients(layer.clone(dtype=jnp.float64).apply), "-".join(case),
                               seeds=SEEDS)
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert (combination["join"], "joined") in recorded and not unnamed, (sorted(recorded), unnamed)
