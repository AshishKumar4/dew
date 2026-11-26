"""The fused gated MLP's check-1 matrix (tests/kernel_matrix.py): a gated MLP's training step with its three
products fused (`dew.nn.gated_mlp`, the choice `gated_mlp`) on par with the same layer's products apart.

Each cell runs a `GatedMLP` as a model's block does, the forward and its reverse mode in one program, its
output and the gradients of its input and every parameter (`reference_error.with_gradients`, a drawn
cotangent) held to the layer with the fused MLP named apart (`generation.chosen`, check 1) from the layer in
float64.

Each cell names its fused form (`fused` or `wide`); the selector's own pick stays apart until a receipt
declares a faster form (`gated_mlp.FUSED_CELLS`). The products take the GEMM template
(`grouped_matmul="mosaic"`); the products-apart arm takes the gated product's XLA kernel.
The matrix reads support from the layer's route at its actual case and named form.
"""


import generation_support
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
KERNELS = {"gated_mlp": frozenset({"fused", "wide"})}
"""The paired and wide-accumulator forms, each exercised against the projections apart."""
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
    """Whether the named form serves `case` after tracing the layer on a mesh of its generation."""
    layer = _layer(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    params = jax.eval_shape(layer.init, jax.random.key(0), _input(case))
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(**NAMED, gated_mlp=case["kernel"]), generation.routes() as routes):
        jax.eval_shape(layer.apply, params, _input(case))
    return [route.kernel for route in routes if route.choice == "gated_mlp"] == [case["kernel"]]


def _drawing(layer: GatedMLP, dtype):
    """A draw of `layer`'s step by seed: its parameters (drawn once), an input and its output's cotangent."""
    params = layer.init(jax.random.key(4), jnp.zeros((*ROWS, FEATURES), dtype))

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed))
        return (params, *(jax.random.normal(drawn, (*ROWS, FEATURES)).astype(dtype) for drawn in keys))

    return draw


_COVERED = (
    ('sm90', 'swiglu', 'none', 'bfloat16'),
    ('sm90', 'geglu', 'clamped', 'float16'),
    ('sm100', 'swiglu', 'clamped', 'float16'),
    ('sm100', 'geglu', 'none', 'float16'),
    ('sm100', 'geglu', 'clamped', 'bfloat16'),
)
CASES = tuple((*case, kernel) for kernel in ("fused", "wide") for case in _COVERED)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("gated_mlp")`, regenerated where the axes or the route's support change)."""


def _case_name(case: tuple[str, ...]) -> str:
    """Keep the existing paired cells' names and distinguish the new wide cells."""
    return "-".join(case[:-1] if case[-1] == "fused" else case)


@cell("gated_mlp", covers=lambda case: dict(zip(AXES["gated_mlp"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
    for case in CASES], ids=_case_name)
def a_fused_mlps_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The layer's training step of `case` on its GPU with its products fused on par with the products apart
    (check 1): the output and the gradients of the input and every parameter, from the layer apart in
    float64 (XLA's products, which a float64 product runs on). The reported route is the named form."""
    combination = dict(zip(AXES["gated_mlp"], case, strict=True))
    layer = _layer(combination)
    dtype = jnp.dtype(combination["dtype"])
    draw = _drawing(layer, dtype)

    def under(kernel, run):
        def arm(*operands):
            with generation.chosen(**NAMED, gated_mlp=kernel):
                return run(*operands)
        return arm

    run, truth = with_gradients(layer.apply), with_gradients(layer.clone(dtype=jnp.float64).apply)
    with generation.routes() as routes:
        assert_as_exact(under(combination["kernel"], run), under("apart", run), draw, _case_name(case),
                        seeds=SEEDS, truth_fn=truth)
    assert ("gated_mlp", combination["kernel"]) in {(found.choice, found.kernel) for found in routes}


def per_shard(choice: str, kernel: str) -> dict:
    """`kernel`'s per-shard cells (`sharded.Shard`, tests/test_per_shard.py): the layer of the first case its
    generation runs interpreted (`CASES`, an sm90's) over rows a two-device data axis splits, its parameters
    whole, its three products fused (`kernel`, interpreted on the CPU's devices), its other choices as every
    cell names them (`NAMED`): the output and every gradient on par with one device's from the layer in
    float64."""
    from jax.sharding import PartitionSpec
    from sharded import Shard, layer_scope


    del choice
    shapes = next(case for case in CASES if case[0] == "sm90" and case[-1] == kernel)
    case = dict(zip(AXES["gated_mlp"], shapes, strict=True))
    layer = _layer(case)
    draw = _drawing(layer, jnp.dtype(case["dtype"]))

    def truth(*operands):
        # The layer apart in float64 through XLA's products, which a float64 product runs on.
        with generation.chosen(grouped_matmul="xla", gated_mlp="apart"):
            return with_gradients(layer.clone(dtype=jnp.float64).apply)(*operands)

    return {"data": Shard(with_gradients(layer.apply), draw,
                          lambda mesh: [PartitionSpec(), PartitionSpec("data"), PartitionSpec("data")],
                          truth, SEEDS, "data", context=lambda: layer_scope(**NAMED))}
