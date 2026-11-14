"""The fused gated MLP (`dew.nn.gated_mlp`): its products on the GEMM template's rules for a formed operand
interpreted as a GPU on a CPU (its training step compiled within resources and under remat:
tests/test_gemm_formed.py), and its selector's call site in `GatedMLP`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import generations
from reference_error import assert_as_exact, with_gradients
from test_gemm_formed import FEATURES, HIDDEN, _draw

from dew.nn import activations
from dew.nn.gated_mlp import fused_products
from dew.nn.kernels import generation
from dew.nn.precision import LowPrecision, Numerics

SEEDS = 8
ACTIVATIONS = [("swiglu", None), ("swiglu", 1.5), ("geglu", None)]
"""Each gated product the fused MLP forms: SwiGLU, SwiGLU clamped (GPT-OSS's and GLM's `swiglu_limit`, at
a limit the drawn projections cross), GeGLU."""


def _apart(activation, limit, *, truth: bool = False):
    """The gated MLP's three products apart, as `GatedMLP` runs them: each projection one dot rounded once,
    the gated product's rounded arithmetic (`activations.gated_product`, clamped by `limit`), the down
    projection; as `truth`, in float64 with nothing rounded."""
    product = activations.gated_product(activation, implementation="xla")

    def run(x, gate, up, down):
        if truth:
            x, gate, up, down = (each.astype(jnp.float64) for each in (x, gate, up, down))
        g, u = activations.swiglu_clamped(x @ gate, x @ up, limit)
        hidden = product(g, u)
        return hidden @ down

    return run


@pytest.mark.parametrize("generation_name", generations("sm90", "sm100"))
@pytest.mark.parametrize(("activation", "limit"), ACTIVATIONS, ids=lambda value: str(value))
def test_the_fused_mlp_holds_check_1_with_its_gradients(generation_name, activation, limit):
    """The fused gated MLP (`fused_products`) on the template's launches, interpreted as `generation_name`:
    its output, and the gradients of its rows and its three kernels through the template's own reverse
    modes (the paired product's, and the down projection's with the gated product's VJP in its input
    gradient's epilogue), each as exact as the three products apart from float64 truth (check 1). The routes
    it records run every product on the template."""
    def fused(x, gate, up, down):
        return fused_products(x, (gate, up, down), activation, limit=limit, interpret_on_cpu=True)

    label = f"fused {activation} MLP, limit {limit}, on {generation_name}"
    apart, truth = (with_gradients(_apart(activation, limit, truth=truth)) for truth in (False, True))
    with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"),
          generation.routes() as routes):
        assert_as_exact(with_gradients(fused), jax.jit(apart), _draw, label, seeds=SEEDS, truth_fn=truth)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}


def test_a_gated_hook_rounds_nowhere_at_fp32_and_wider():
    """The fused MLP's hook (`activations.gated_hook`) computes a float64 or fp32 gated product as the
    portable product does, `activate(gate) * up` in the operands' own dtype with no rounding (only a 16-bit
    product's arithmetic rounds), so a float64 fused MLP is the float64 program's: its value equals the
    portable product's (`gated_product`) bit for bit at float64 and fp32, and at bf16 too."""
    rng = np.random.default_rng(5)
    with jax.enable_x64(new_val=True):
        for dtype in (jnp.float64, jnp.float32, jnp.bfloat16):
            gate, up = (jnp.asarray(rng.normal(size=(64, 32)) * 3, dtype) for _ in range(2))
            np.testing.assert_array_equal(
                np.asarray(activations.gated_hook("swiglu")(gate, up), np.float64),
                np.asarray(activations.gated_product("swiglu", implementation="xla")(gate, up), np.float64),
                err_msg=str(dtype))
            assert activations.gated_hook("swiglu")(gate, up).dtype == dtype


@pytest.mark.parametrize("generation_name", generations("sm90", "sm100"))
def test_a_gated_mlp_runs_fused_where_named_and_its_products_apart_where_no_receipt_declares_it(
        generation_name):
    """A `GatedMLP` traced as a generation no receipt declares the fused MLP faster on
    (`gated_mlp.FUSED_CELLS`) runs its products apart, and fused where the block names it
    (`generation.chosen(gated_mlp='fused')`) and the template launches its products, its output the products
    apart's shape and dtype; its products apart where the projections add a bias, which the fused MLP's hooks
    do not add, or where the run's low-precision recipe takes the products (`projections.recipe_projection`,
    which the fused products do not run), each a handoff, even named. Its numerics are `fused_products`'
    (above) and the matrix's (tests/gated_mlp_matrix.py)."""
    from dew.nn.mlp import GatedMLP

    def kernels(layer, **choices):
        x = jax.ShapeDtypeStruct((2, 64, FEATURES), jnp.bfloat16)
        params = jax.eval_shape(layer.init, jax.random.key(0), x)
        with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic", **choices),
              generation.routes() as routes):
            out = jax.eval_shape(layer.apply, params, x)
        assert (out.shape, out.dtype) == ((2, 64, FEATURES), jnp.bfloat16)
        return [route.kernel for route in routes if route.choice == "gated_mlp"]

    plain = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, dtype=jnp.bfloat16)
    biased = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, use_bias=True, dtype=jnp.bfloat16)
    assert kernels(plain) == ["apart"]
    assert kernels(plain, gated_mlp="fused") == ["fused"]
    assert kernels(biased, gated_mlp="fused") == ["apart"]
    with generation.run_precision(Numerics(low_precision=LowPrecision())):
        assert kernels(plain, gated_mlp="fused") == ["apart"]


def test_a_gated_mlps_case_runs_at_the_dtype_its_products_run_and_asks_nothing_apart():
    """The fused MLP's selector reads the dtype its products run at, the promotion of the input and the
    kernels (`gated_case`, `canonicalize_dtype`): fp32 for bf16 rows over fp32 kernels with no dtype set,
    bf16 where the layer states it, so a receipt's cell names the products it timed. And a `GatedMLP` its
    selector runs apart (no cell, nothing named) asks no launch of its products fused: its routes are those
    of the same products apart where nothing could fuse them (a scaled gate), its gated MLP's own aside."""
    from dew.nn.gated_mlp import gated_case
    from dew.nn.mlp import GatedMLP

    x = jnp.zeros((2, 64, FEATURES), jnp.bfloat16)
    for dtype, runs in ((None, jnp.float32), (jnp.bfloat16, jnp.bfloat16)):
        layer = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, dtype=dtype)
        bound = layer.bind(layer.init(jax.random.key(0), x.astype(jnp.float32)))
        case = gated_case(x, (bound.gate_proj, bound.up_proj, bound.down_proj), "swiglu")
        assert case.dtype == runs and case.cell[1] == runs, dtype

    def routes_of(layer):
        params = jax.eval_shape(layer.init, jax.random.key(0), x)
        with generation.assumed("sm90"), generation.routes() as routes:
            jax.eval_shape(layer.apply, params, x)
        return sorted((route.choice, route.kernel) for route in routes if route.choice != "gated_mlp")

    plain = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, dtype=jnp.bfloat16)
    assert routes_of(plain) == routes_of(plain.clone(gate_multiplier=0.5))


@pytest.mark.parametrize("generation_name", ["sm90", "sm100"])
def test_a_gated_mlp_whose_hook_the_template_cannot_lower_runs_apart(generation_name):
    """A gated MLP named fused (`generation.chosen(gated_mlp='fused')`) whose gated product's hook the
    template's Mosaic launches do not lower (GeGLU's exact GELU, an erf behind an optimization barrier) runs
    its products apart, the fused MLP reporting the hook's refusal, where it was fused and the paired
    product's launch refused the call; GeGLU's tanh form fuses (train 813's F3)."""
    from dew.nn.mlp import GatedMLP

    x = jnp.zeros((2, 64, FEATURES), jnp.bfloat16)

    def route(activation):
        layer = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, activation=activation,
                         dtype=jnp.bfloat16)
        params = jax.eval_shape(layer.init, jax.random.key(0), x)
        named = generation.chosen(grouped_matmul="mosaic", gated_mlp="fused")
        with generation.assumed(generation_name), named, generation.routes() as routes:
            jax.eval_shape(layer.apply, params, x)
        return [each.kernel for each in routes if each.choice == "gated_mlp"]

    assert route("geglu") == ["fused"]
    assert route("geglu_exact") == ["apart"]
