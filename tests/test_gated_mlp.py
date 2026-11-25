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


@pytest.mark.parametrize(("form", "generation_name"), [
    pytest.param("fused", g.values[0], marks=g.marks, id=f"fused-{g.values[0]}")
    for g in generations("sm90", "sm100")
] + [
    pytest.param("wide", g.values[0], marks=g.marks, id=f"wide-{g.values[0]}")
    for g in generations("sm80", "sm90", "sm100")
])
@pytest.mark.parametrize(("activation", "limit"), ACTIVATIONS, ids=lambda value: str(value))
def test_the_fused_mlp_holds_check_1_with_its_gradients(form, generation_name, activation, limit):
    """Both fused gated MLP forms (`fused_products` paired and Formulation (C) `wide_products`) on the
    template's launches, interpreted as `generation_name`: output and all four gradients (rows, gate, up,
    down) hold check 1 against the three products apart from float64 truth, recording mosaic routes."""
    from dew.nn.gated_mlp import wide_products

    runner = wide_products if form == "wide" else fused_products

    def run_form(x, gate, up, down):
        return runner(x, (gate, up, down), activation, limit=limit, interpret_on_cpu=True)

    label = f"{form} {activation} MLP, limit {limit}, on {generation_name}"
    apart, truth = (with_gradients(_apart(activation, limit, truth=truth)) for truth in (False, True))
    with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"),
          generation.routes() as routes):
        assert_as_exact(with_gradients(run_form), jax.jit(apart), _draw, label, seeds=SEEDS, truth_fn=truth)
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
    do not add, a handoff even named; where the run's low-precision recipe takes the products by their role,
    the recipe's chain of them (`projections.recipe_chain`, row 12689), which no fused MLP's choice reads,
    even named. Its numerics are `fused_products`' (above) and the matrix's (tests/gated_mlp_matrix.py)."""
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
        # recipe takes runs them as the recipe's chain.
        assert kernels(plain, gated_mlp="fused") == ["fused"]
        with projection_role("mlp"):
            assert kernels(plain, gated_mlp="fused") == []


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


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("precision", [None, "BF16_BF16_F32"])
@pytest.mark.parametrize("layout", ["tensor", "tensor and fsdp"])
def test_a_packed_mlp_whose_tensor_axis_splits_its_pair_runs_split_on_its_hidden_width(precision, layout):
    """A packed SwiGLU MLP on an Explicit tensor axis splitting its packed kernel's columns (the gate-and-up
    pair, `[D, 2H]` read as `[..., 2, H]`) and its down projection's rows, alone or with an FSDP axis
    splitting those rows beside it: the pair's axis moves onto the hidden width (`whole_along`, an
    all-to-all in its compiled program), each device computing its share of the product, and the down
    projection contracts the width both sides split alike (`sharding.contracted_alike`), at the default
    precision and at `BF16_BF16_F32`'s product; its forward and its pullback on par with one device's
    (check 1), where the explicit contraction raised (row 12225, and its confirm review's bilateral split
    and bf16 product)."""
    from flax import linen as nn
    from jax.sharding import AxisType, PartitionSpec
    from sharded import collective_axes, placed_on

    from dew.nn.mlp import GatedMLP

    names = ("tensor",) if layout == "tensor" else ("tensor", "fsdp")
    mesh = jax.make_mesh((2,) * len(names), names, axis_types=(AxisType.Explicit,) * len(names),
                         devices=jax.devices()[:2 ** len(names)])
    algorithm = None if precision is None else jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    variables = nn.unbox(GatedMLP(32, 16, activation="swiglu", packed=True).init(
        jax.random.key(1), jnp.zeros((2, 4, 16))))

    def applied(precision):
        mlp = GatedMLP(32, 16, activation="swiglu", packed=True, precision=precision)

        def apply(gate_up, down, x):
            params = {"gate_up_proj": {"kernel": gate_up}, "down_proj": {"kernel": down}}
            return mlp.apply({"params": params}, x)

        return with_gradients(apply)

    run = applied(algorithm)
    specs = [PartitionSpec(None, "tensor"), PartitionSpec(names if len(names) > 1 else names[0], None),
             PartitionSpec(), PartitionSpec()]

    def draw(seed):
        gate_up = variables["params"]["gate_up_proj"]["kernel"]
        down = variables["params"]["down_proj"]["kernel"]
        x, dy = (jax.random.normal(jax.random.key(seed * 2 + offset), (2, 4, 16)) for offset in (0, 1))
        return gate_up, down, x, dy

    split = jax.jit(run)

    def split_arm(*args):
        # Placed by their shardings alone: a weighed check traces the arm under jit, where no mesh is set.
        return split(*placed_on(args, specs, mesh))

    assert_as_exact(split_arm, jax.jit(run), draw, f"a packed MLP split by {layout} at {precision}",
                    seeds=SEEDS, truth_fn=applied(None))
    # The forward exchanges the pair's tensor axis onto the width (`whole_along`) rather than gathering it:
    # the forward program's collectives, as the profiler's parser reads them, hold that one move and no
    # gather of the pair (`collective_axes`).
    forward = GatedMLP(32, 16, activation="swiglu", packed=True, precision=algorithm)

    def forward_apply(gate_up, down, x):
        params = {"gate_up_proj": {"kernel": gate_up}, "down_proj": {"kernel": down}}
        return forward.apply({"params": params}, x)

    compiled = jax.jit(forward_apply).lower(*placed_on(draw(0)[:3], specs[:3], mesh)).compile()
    moves = [(op, axes) for op, axes in collective_axes(compiled, mesh) if op in ("all-to-all", "all-gather")]
    assert ("all-to-all", frozenset({"tensor"})) in moves, moves
    assert ("all-gather", frozenset({"tensor"})) not in moves, moves


@pytest.mark.mesh(devices=2)
def test_a_product_of_a_split_activation_and_a_weight_holding_a_sum_takes_jaxs_route():
    """A product of rows an Explicit axis splits and a weight whose placement holds a sum still to take over
    that axis (`reduced`) keeps JAX's own route (`sharding.contracted_alike` hands both operands back as they
    are, placing nothing), and JAX's product of them computes what it computes unplaced, where reading the
    weight's spec as entries raised before any product (the confirm review of row 12225)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    from dew.nn import precision
    from dew.nn.sharding import contracted_alike

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    x = jax.device_put(jax.random.normal(jax.random.key(0), (4, 8)),
                       NamedSharding(mesh, PartitionSpec("data")))
    w = jax.device_put(jax.random.normal(jax.random.key(1), (8, 3)),
                       NamedSharding(mesh, PartitionSpec(None, None, reduced={"data"})))
    numbers = (((1,), (0,)), ((), ()))

    def ours(x, w):
        lhs, rhs, placed = contracted_alike(x, w, numbers)
        assert lhs is x and rhs is w and placed is None
        return jax.lax.dot_general(lhs, rhs, numbers, precision=jax.lax.Precision.HIGHEST)

    def theirs(x, w):
        return jax.lax.dot_general(x, w, numbers, precision=jax.lax.Precision.HIGHEST)

    with jax.set_mesh(mesh):
        got, expected = jax.jit(ours)(x, w), jax.jit(theirs)(x, w)
    assert jax.typeof(got).sharding.spec == jax.typeof(expected).sharding.spec
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))

    # The bf16 product (`precision.bf16_dot_general`) over that weight, forward and pullback, places each
    # cotangent as JAX's own pullback of the same product does (the weight's unreduced), and computes what it
    # computes unplaced: small integers, whose bf16 products and float32 sums are exact in any order, bit
    # for bit (the confirm review: its pullback read the weight's spec as entries).
    whole_x = jnp.asarray(np.random.default_rng(0).integers(-4, 5, (4, 8)), jnp.float32)
    whole_w = jnp.asarray(np.random.default_rng(1).integers(-4, 5, (8, 3)), jnp.float32)
    x = jax.device_put(whole_x, NamedSharding(mesh, PartitionSpec("data")))
    w = jax.device_put(whole_w, NamedSharding(mesh, PartitionSpec(None, None, reduced={"data"})))

    def pulled(product):
        def run(x, w):
            y, pull = jax.vjp(product, x, w)
            return y, pull(jnp.ones_like(y))

        return run

    def bf16(x, w):
        return precision.bf16_dot_general(x, w, numbers)

    def native(x, w):
        return jax.lax.dot_general(x.astype(jnp.bfloat16).astype(jnp.float32),
                                   w.astype(jnp.bfloat16).astype(jnp.float32), numbers,
                                   precision=jax.lax.Precision.HIGHEST)

    def summed(run):
        # Each value replicated, an unreduced cotangent's pending sum taken: no device holds it whole.
        return lambda x, w: jax.tree.map(lambda part: jax.sharding.reshard(part, PartitionSpec()), run(x, w))

    with jax.set_mesh(mesh):
        placed, jaxs = jax.jit(pulled(bf16))(x, w), jax.jit(pulled(native))(x, w)
        values = jax.jit(summed(pulled(bf16)))(x, w)
    whole = jax.jit(pulled(bf16))(whole_x, whole_w)
    for got, theirs in zip(jax.tree.leaves(placed), jax.tree.leaves(jaxs), strict=True):
        assert jax.typeof(got).sharding.spec == jax.typeof(theirs).sharding.spec
    for got, alone in zip(jax.tree.leaves(values), jax.tree.leaves(whole), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(alone))


@pytest.mark.parametrize("generation_name", generations("sm90", "sm100"))
def test_grouped_gated_experts_selector_defaults_apart_and_honors_named_and_handoff_gates(generation_name):
    """Grouped expert MLPs (`gated_mlp.fuses_grouped`, row 12847) key their own `(dtype, 'grouped')` cell on
    `GATED_MLP`, default to `'apart'` without a receipt, run `'fused'` when named on supported 3D expert
    kernels (both under the default grouped selector and under `grouped_matmul='mosaic'`), and hand off
    cleanly to `'apart'` when `paired`, `down_prologue`, `hook`, `whole`, or `launched` fails."""
    from dew.nn import gated_mlp

    rows = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((4, 128, 256), jnp.bfloat16)
    down = jax.ShapeDtypeStruct((4, 256, 128), jnp.bfloat16)
    hook = activations.gated_hook("swiglu")
    with generation.assumed(generation_name):
        case = gated_mlp.grouped_case(rows, (gate, down), None, hook)
        assert case.cell == (generation_name, (jnp.dtype(jnp.bfloat16), "grouped"))
        with generation.routes() as default_routes:
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook)
        assert [route.kernel for route in default_routes if route.choice == "gated_mlp"] == ["apart"]
        with generation.chosen(gated_mlp="fused"):
            with generation.routes() as named_routes:
                assert gated_mlp.fuses_grouped(rows, (gate, down), None, hook)
            assert [route.kernel for route in named_routes if route.choice == "gated_mlp"] == ["fused"]
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook, paired=False)
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook, down_prologue=False)
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, None)
            assert not gated_mlp.fuses_grouped(rows, None, None, hook)
            assert not gated_mlp.fuses_grouped(
                rows, (gate, down), None, activations.gated_hook("geglu_exact"))
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook, whole=False)
            with jax.enable_x64():
                wide_rows = jax.ShapeDtypeStruct((256, 128), jnp.float64)
                assert not gated_mlp.fuses_grouped(wide_rows, (gate, down), None, hook)
        # Named CUTLASS refuses paired epilogue / down prologue hooks and must hand off to `apart`.
        with generation.chosen(grouped_matmul="cutlass", gated_mlp="fused"):
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook)
        with generation.chosen(grouped_matmul="mosaic", gated_mlp="fused"):
            assert gated_mlp.fuses_grouped(rows, (gate, down), None, hook)
            # Explicit group_sizes, gather/scatter tables, and stacked 4D layer banks pass through cleanly.
            sizes = jax.ShapeDtypeStruct((4,), jnp.int32)
            table = jax.ShapeDtypeStruct((256,), jnp.int32)
            gate_4d = jax.ShapeDtypeStruct((2, 4, 128, 256), jnp.bfloat16)
            down_4d = jax.ShapeDtypeStruct((2, 4, 256, 128), jnp.bfloat16)
            assert gated_mlp.fuses_grouped(rows, (gate_4d, down_4d), None, hook,
                                           group_sizes=sizes, gather=table, scatter=table, layer=0)
            # Forced grouped Mosaic never escapes `KernelRefused` before GATED_MLP's handoff.
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook, whole=False)
            with jax.enable_x64():
                wide_rows = jax.ShapeDtypeStruct((256, 128), jnp.float64)
                assert not gated_mlp.fuses_grouped(wide_rows, (gate, down), None, hook)


def test_grouped_gated_experts_hands_off_default_a100_cutlass_down():
    """On A100 (`sm80`), the ordinary grouped down cell chooses CUTLASS which forms the prologue in XLA, and
    `mma.sync` takes no side-tile prologue (`takes_prologue_tiles = False`), so `fuses_grouped` hands off to
    `'apart'` under both the default selector and forced Mosaic without escaping `KernelRefused`."""
    from dew.nn import gated_mlp

    rows = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((4, 128, 256), jnp.bfloat16)
    down = jax.ShapeDtypeStruct((4, 256, 128), jnp.bfloat16)
    table = jax.ShapeDtypeStruct((256,), jnp.int32)
    hook = activations.gated_hook("swiglu")
    with generation.assumed("sm80"):
        with generation.chosen(gated_mlp="fused"):
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook)
        with generation.chosen(grouped_matmul="mosaic", gated_mlp="fused"):
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook)
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook, gather=table)
            assert not gated_mlp.fuses_grouped(rows, (gate, down), None, hook, scatter=table)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90", "sm100"))
def test_wide_gated_mlp_selector_handoffs_and_remat_recompute(generation_name):
    """`GATED_MLP` selects `'wide'` when named via `dew.kernels.use(Kernels(gated_mlp='wide'))`, hands off
    biased, unaligned-hidden, and grouped cases to `'apart'`, and drops the saved wide residual when an
    enclosing `recomputing((Residual.GATE_PROJ,))` rung is active."""
    import dew
    from dew.nn import gated_mlp, residuals
    from dew.nn.mlp import GatedMLP

    x = jax.ShapeDtypeStruct((2, 64, FEATURES), jnp.bfloat16)

    def kernels(layer, **choose):
        params = jax.eval_shape(layer.init, jax.random.key(0), x)
        with (generation.assumed(generation_name),
              dew.kernels.use(dew.kernels.Kernels(grouped_matmul="mosaic", **choose)),
              generation.routes() as routes):
            jax.eval_shape(layer.apply, params, x)
        return [route.kernel for route in routes if route.choice == "gated_mlp"]

    plain = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, dtype=jnp.bfloat16)
    biased = GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, use_bias=True, dtype=jnp.bfloat16)
    unaligned = GatedMLP(hidden_features=48, out_features=FEATURES, dtype=jnp.bfloat16)
    assert kernels(plain, gated_mlp="wide") == ["wide"]
    assert kernels(biased, gated_mlp="wide") == ["apart"]
    assert kernels(unaligned, gated_mlp="wide") == ["apart"]

    # Grouped experts hand off `'wide'` cleanly to `'apart'`.
    g_rows = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
    g_gate = jax.ShapeDtypeStruct((4, 128, 256), jnp.bfloat16)
    g_down = jax.ShapeDtypeStruct((4, 256, 128), jnp.bfloat16)
    with generation.assumed(generation_name), generation.chosen(gated_mlp="wide"):
        assert not gated_mlp.fuses_grouped(g_rows, (g_gate, g_down), None, activations.gated_hook("swiglu"))

    # Remat `recomputing((Residual.GATE_PROJ,))` runs `wide_products` with `saved=False` and holds check 1.
    def wide_recomputed(rows, gate, up, down):
        with residuals.recomputing((residuals.Residual.GATE_PROJ,)):
            return gated_mlp.wide_products(rows, (gate, up, down), "swiglu", interpret_on_cpu=True)

    apart, truth = (with_gradients(_apart("swiglu", None, truth=t)) for t in (False, True))
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"):
        assert_as_exact(with_gradients(wide_recomputed), jax.jit(apart), _draw,
                        f"wide swiglu recomputed on {generation_name}", seeds=SEEDS, truth_fn=truth)
