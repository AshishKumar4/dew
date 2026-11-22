"""The fused gated MLP (`dew.nn.gated_mlp`): its products on the GEMM template's rules for a formed operand
interpreted as a GPU on a CPU (its training step compiled within resources and under remat:
tests/test_gemm_formed.py), and its selector's call site in `GatedMLP`."""

import functools

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
from dew.nn.precision import LowPrecision, Numerics, projection_role

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
    do not add, or where the run's low-precision recipe takes the products by their role
    (`projections.recipe_projection`, which the fused products do not run), each a handoff, even named. Its
    numerics are `fused_products`' (above) and the matrix's (tests/gated_mlp_matrix.py)."""
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
        # A role-less MLP's products keep the run's dtypes, which the fused MLP runs; one in the role the
        # recipe takes hands them to their own calls.
        assert kernels(plain, gated_mlp="fused") == ["fused"]
        with projection_role("mlp"):
            assert kernels(plain, gated_mlp="fused") == ["apart"]


@pytest.mark.parametrize("generation_name", generations("sm90", "sm100"))
def test_a_gated_mlp_named_fused_on_a_data_split_mesh_runs_its_products_per_shard(generation_name):
    """A `GatedMLP` named fused on a run's mesh of two compile-only GPUs whose data axis splits its rows,
    lowered as a training step traces it (the trainer's layout, `placing`, under the default rules), runs
    its three products fused on each device's own rows (`gated_mlp._placement`: a map per shard, its kernels
    gathered whole there), as a projection's product maps its rows (`projections.mapped_shards`), where the
    template's launch was refused the split rows (row 6594): its launches read one device's rows and each
    kernel whole, none both devices' rows. Its numerics per shard are the matrix's
    (tests/gated_mlp_matrix.py)."""
    from jax.sharding import PartitionSpec
    from kernel_compile import Expected, lowered_for
    from sharded import layer_scope

    from dew.mesh import DATA_AXIS, MESH_AXES
    from dew.nn.kernels.lowered import kernel_calls
    from dew.nn.mlp import GatedMLP

    layer = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((2, 64, FEATURES), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)
    # A run's mesh (`MeshSpec`'s axes): the data axis two devices, every other one.
    mesh = Expected(tuple((axis, 2 if axis == DATA_AXIS else 1) for axis in MESH_AXES),
                    (None, PartitionSpec(DATA_AXIS)), frozenset())
    with layer_scope(grouped_matmul="mosaic", gated_mlp="fused"), generation.routes() as routes:
        lowered = lowered_for(layer.apply, (params, x), generation_name, mesh)
    assert [route.kernel for route in routes if route.choice == "gated_mlp"] == ["fused"]
    assert lowered.out_info.shape == (2, 64, FEATURES)
    read = {shape for call in kernel_calls(lowered) if call.kind == "mosaic_gpu" for shape in call.operands}
    # A device's 64 rows of the 128 and its hidden rows; each kernel whole, the template's one group of it.
    assert {(64, FEATURES), (64, HIDDEN), (1, FEATURES, HIDDEN), (1, HIDDEN, FEATURES)} <= read, read
    assert not {(128, FEATURES), (128, HIDDEN)} & read, read


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


def _activation_spans(jaxpr) -> list:
    """The activation regions (`derivatives.Activation`) a program's jaxpr holds, its calls' bodies' too."""
    from jax.extend.core import jaxprs_in_params

    from dew.nn import derivatives

    found = []
    for equation in jaxpr.eqns:
        if equation.primitive is derivatives.marked_call_p and isinstance(equation.params["kind"],
                                                                          derivatives.Activation):
            found.append(equation.params["kind"])
        for inner in jaxprs_in_params(equation.params):
            found += _activation_spans(inner)
    return found


@pytest.mark.parametrize("implementation", ["xla", "pallas"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_a_gated_product_is_one_activation_region_and_computes_as_it_did(dtype, implementation):
    """A gated product (`gated_product`, 'swiglu'), its XLA arithmetic or SwiGLU's Pallas kernel where it
    is named (on an H100's routes), is one region of its activation (`derivatives.Activation('swiglu',
    gated=True)`) over its gate and up projection, whichever route it takes, the seam a fusion pass's
    epilogue replaces, its body holding the route's launch; and the region computes what its route
    computes unmarked, its value and its operands' gradients the route's own bit for bit: the public product
    on the XLA route, and the dispatcher's region over the Pallas kernel (`activations._dispatched`),
    interpreted on a CPU. An
    ungated MLP's named activation is one region of it ungated, at each place it runs."""
    from jax.experimental.pallas import pallas_call_p

    from dew.nn import derivatives
    from dew.nn.kernels import norm_swiglu
    from dew.nn.mlp import GatedMLP
    from dew.nn.programs import equations

    rng = np.random.default_rng(0)
    gate, up = (jnp.asarray(rng.standard_normal((4, 64)) * 3, dtype) for _ in range(2))
    with generation.assumed("sm90"):
        traced = jax.make_jaxpr(activations.gated_product("swiglu", implementation=implementation))(gate, up)
    (span,) = traced.jaxpr.eqns
    assert span.primitive is derivatives.marked_call_p
    assert span.params["kind"] == derivatives.Activation("swiglu", gated=True)
    launched = {equation.primitive for equation in equations(span.params["call_jaxpr"].jaxpr)}
    assert (pallas_call_p in launched) == (implementation == "pallas")

    portable = activations._portable(activations.silu, lambda gate, up: (gate, up))

    def kernel(gate, up, portable):
        return norm_swiglu.swiglu(gate, up, portable)

    plain = portable if implementation == "xla" else functools.partial(kernel, portable=portable)
    # The public product itself on the XLA route; the dispatcher's region over the kernel, interpreted, on the
    # Pallas route a CPU's selector takes none of.
    region = (activations.gated_product("swiglu", implementation="xla") if implementation == "xla"
              else activations._dispatched("swiglu", portable, lambda gate, up: implementation, kernel))

    def summed(function):
        return jax.jit(jax.value_and_grad(lambda g, u: jnp.sum(function(g, u).astype(jnp.float32)),
                                          argnums=(0, 1)))

    (value, gradients), (own, own_gradients) = summed(region)(gate, up), summed(plain)(gate, up)
    np.testing.assert_array_equal(np.asarray(value), np.asarray(own))
    for mine, theirs in zip(gradients, own_gradients, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    np.testing.assert_array_equal(np.asarray(jax.jit(region)(gate, up)), np.asarray(jax.jit(plain)(gate, up)))
    # An ungated MLP's named activation, and its activation after the down projection too where it takes one
    # (an in-context memory's projection pairs).
    x = jnp.asarray(rng.standard_normal((2, 64)), dtype)
    for after in (False, True):
        mlp = GatedMLP(hidden_features=128, out_features=64, activation="gelu", activation_after=after)
        params = mlp.init(jax.random.key(0), x)
        spans = _activation_spans(jax.make_jaxpr(lambda v, mlp=mlp: mlp.apply(v, x))(params).jaxpr)
        assert spans == [derivatives.Activation("gelu", gated=False)] * (2 if after else 1), (after, spans)
