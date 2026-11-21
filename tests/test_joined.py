"""Projections joined per call (`dew.nn.joined`): one GEMM in place of a
layer's own projection calls, where a generation's kernel row measured it
faster."""

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact, equations, widened, with_gradients

from dew.nn.dense import Dense
from dew.nn.joined import joined_dense
from dew.nn.kernels import generation
from dew.nn.linear import GatedDeltaNet
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mlp import GatedMLP


def row_joins(choice: str, *, joins: bool = True):
    """Choose joined projections for `choice` where `joins`, else separate projections."""
    return generation.chosen(**{choice: "joined" if joins else "apart"})


def dots(function, *args) -> int:
    """The dot products an accelerator's run of `function` executes, its sub-jaxprs' included: of each
    platform choice, the model's own products, where a CPU sums them in float64 and spreads a bias
    by a product (`dew.nn.precision`, tests/test_cpu_accumulation.py)."""
    graph = jax.make_jaxpr(lambda *operands: function(*operands))(*args)
    return sum(equation.primitive.name == "dot_general" for equation in equations(graph, platform="cuda"))


JOINED_SEEDS = {"recipe gates": 8, ("gate_up", jnp.float32, False): 8, ("gate_up", jnp.float32, True): 8,
                ("gate_up", jnp.bfloat16, False): 8, ("gate_up", jnp.bfloat16, True): 8,
                ("delta_gates", jnp.float32): 8, ("delta_gates", jnp.bfloat16): 8, "promoted bias": 8,
                "gate multiplier": 8, ("qkv", False): 8, ("qkv", True): 8}
"""Each joined row's declared draws against its projections apart (check 1), sized by `seeds_for` paired
with the joined row on the CPU and an A100."""


def drawing(params, shape, dtype):
    """A joined row's draw: `params` beside an input and its output's cotangent of `shape` in `dtype`, from
    the seed's key."""
    def draw(seed):
        keys = jax.random.split(jax.random.key(seed))
        return (params, *(jax.random.normal(drawn, shape).astype(dtype) for drawn in keys))
    return draw


def assert_joined_as_exact(choice: str, run, draw, wide_run, label: str, *, seeds: int):
    """`run` with `choice` joined, every leaf of the same dtype as the run with it apart and on par with
    it from `wide_run`, the apart run in float64, over `draw`'s inputs (check 1)."""
    def under(joins):
        def arm(*operands):
            with row_joins(choice, joins=joins):
                return run(*operands)
        return arm

    def truth(*operands):
        # The float64 truth through XLA's products, which a float64 product runs on, whatever kernel the
        # arms name (head review of b68fc2382: the template named around them refused it).
        # Its routes its own (`routes`' `alone`): the arms' assertion reads theirs (base review 760be77fe F3).
        with (row_joins(choice, joins=False), generation.chosen(grouped_matmul="xla"),
              generation.routes(alone=True)):
            return wide_run(*operands)

    together, apart = (jax.tree.leaves(jax.jit(under(joins))(*draw(0))) for joins in (True, False))
    assert [leaf.dtype for leaf in together] == [leaf.dtype for leaf in apart]
    assert_as_exact(under(joins=True), under(joins=False), draw, label, seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_gate_and_up_joined_are_as_exact_as_apart_from_float64(dtype, bias):
    """A GatedMLP whose row joins gate and up computes both in one GEMM, two
    dots where it ran three, and its output and the gradients of its input
    and every parameter lie within tests/reference_error.py's rule of the
    layer's with them apart, on par from float64 over drawn inputs (check 1)."""
    layer = GatedMLP(hidden_features=96, out_features=64, use_bias=bias, dtype=dtype)
    # Drawn biases too, which init leaves at zero.
    params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(6), leaf.shape),
                          layer.init(jax.random.key(4), jnp.zeros((2, 33, 64), dtype)))

    draw = drawing(params, (2, 33, 64), dtype)

    x = draw(0)[1]
    with row_joins("gate_up", joins=False):
        assert dots(layer.apply, params, x) == 3
    with row_joins("gate_up"):
        assert dots(layer.apply, params, x) == 2
    assert_joined_as_exact("gate_up", with_gradients(layer.apply), draw,
                           with_gradients(layer.clone(dtype=jnp.float64).apply), "gate and up joined",
                           seeds=JOINED_SEEDS["gate_up", dtype, bias])


def test_joined_projections_keep_the_dtype_their_promotion_gives_a_wider_bias():
    """Joined, two projections whose promotion leaves the bias at fp32 add it to their bf16 products in
    the fp32 the addition promotes to, as each projection apart adds its own: the stacked outputs
    carry each projection's own dtype, within tests/reference_error.py's rule of the projections'
    apart against float64."""
    def bias_kept(inputs, kernel, bias, dtype=None):
        return inputs.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16), bias

    class Pair(nn.Module):
        wide: bool = False

        @nn.compact
        def __call__(self, x):
            promotion = {} if self.wide else {"promote_dtype": bias_kept}
            projections = [Dense(96, dtype=jnp.float64 if self.wide else jnp.bfloat16, name=name, **promotion)
                           for name in ("gate", "up")]
            if self.wide:
                return jnp.stack([projection(x) for projection in projections], axis=-2)
            apart = jnp.stack([projection(x) for projection in projections], axis=-2)
            return apart, joined_dense(x, projections, axis=-2)

    params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(12), leaf.shape),
                          Pair().init(jax.random.key(13), jnp.zeros((2, 33, 64), jnp.bfloat16)))

    def draw(seed):
        return (jax.random.normal(jax.random.key(seed), (2, 33, 64)).astype(jnp.bfloat16),)

    apart, joined = Pair().apply(params, *draw(0))
    assert joined.dtype == apart.dtype == jnp.float32
    assert_as_exact(lambda x: Pair().apply(params, x)[1], lambda x: Pair().apply(params, x)[0], draw,
                    "joined", seeds=JOINED_SEEDS["promoted bias"],
                    truth_fn=lambda x: Pair(wide=True).apply(widened(params), x))


def test_joined_projections_round_a_16bit_product_and_its_fp32_bias_once_as_each_does_apart():
    """Joined, two bf16 projections whose biases are stored at fp32 round each product and its bias once,
    as each projection apart does (`dense._biased`): an input of ones and a kernel column of 1 and 1/256 sum
    to 1 + 1/256, which with a bias of 1/256 is 1 + 1/128 rounded once, but 1 where the product is rounded
    to bf16 first (a tie to the even 1, a bf16 step past 1 being 1/128) and the bias added after."""
    class Pair(nn.Module):
        @nn.compact
        def __call__(self, x):
            projections = [Dense(1, dtype=jnp.bfloat16, name=name) for name in ("gate", "up")]
            return jnp.stack([projection(x) for projection in projections], axis=-2), joined_dense(
                x, projections, axis=-2)

    x = jnp.ones((1, 2), jnp.bfloat16)
    column = {"kernel": jnp.asarray([[1.0], [1 / 256]], jnp.float32),
              "bias": jnp.asarray([1 / 256], jnp.float32)}
    params = {"params": {"gate": column, "up": column}}
    apart, joined = Pair().apply(params, x)
    assert float(apart[0, 0, 0]) == 1 + 1 / 128
    np.testing.assert_array_equal(np.asarray(joined, np.float32), np.asarray(apart, np.float32))


def test_projections_of_two_products_do_not_join():
    """Two projections whose products differ (one lax's own, one `float32_dot_general`) do not join: the
    join takes the first projection's product for all of them (`JoinCase.uniform`)."""
    from dew.nn.joined import join_case
    from dew.nn.precision import float32_dot_general

    class Pair(nn.Module):
        @nn.compact
        def __call__(self, x):
            gate, up = Dense(8, name="gate"), Dense(8, name="up", dot_general=float32_dot_general)
            gate(x), up(x)
            return join_case((gate, up)).uniform, join_case((gate, Dense(8, name="other"))).uniform

    x = jnp.ones((2, 4), jnp.float32)
    params = Pair().init(jax.random.key(0), x)
    assert Pair().apply(params, x) == (False, True)


@pytest.mark.parametrize("wrapped", ["direct", "nested", "dew", "input", "input_lora"])
def test_quantized_projection_calls_keep_their_own_paths_under_a_joined_row(wrapped):
    from flax.errors import ScopeParamShapeError

    from dew.quantization import InputQuantization, Quantization

    layer = GatedMLP(hidden_features=96, out_features=64)
    inputs = jax.random.normal(jax.random.key(7), (2, 17, 64))
    if wrapped in ("direct", "nested"):
        qwix = pytest.importorskip("qwix")
        provider = qwix.QtProvider([qwix.QtRule(module_path=".*gate_proj", weight_qtype=jnp.int8,
                                               act_qtype=jnp.int8)])
        quantized = qwix.quantize_model(layer, provider)

        class Parent(nn.Module):
            child: nn.Module

            @nn.compact
            def __call__(self, x):
                return self.child(x)

        quantized = Parent(quantized)
        if wrapped == "nested":
            # A parent quantized by a provider of its own, whose rules match nothing at the projections:
            # the child's provider owns their products (`qwix.pallas.get_current_rule` reaches it).
            quantized = qwix.quantize_model(quantized, qwix.QtProvider(
                [qwix.QtRule(module_path="no_module", weight_qtype=jnp.int8)]))
    elif wrapped == "dew":
        quantized = Quantization().apply(layer)
    else:
        quantized = InputQuantization(sites=(("gate_proj", "fp8"), ("up_proj", "fp8"))).apply(layer)
    variables = quantized.init(jax.random.key(11), inputs)
    if wrapped.startswith("input"):
        variables["quantization"] = {name: {"fp8_input_scale": jnp.asarray(0.01)}
                                      for name in ("gate_proj", "up_proj")}
    if wrapped == "input_lora":
        from dew.lora import LoRA

        adapter = LoRA(rank=2, modules=("gate_proj", "up_proj")).apply(quantized, variables, key=19)
        quantized = adapter.model
        variables = jax.tree_util.tree_map_with_path(
            lambda path, value: jnp.ones_like(value) * 0.02 if path[-1].key == "lora_B" else value,
            adapter.variables)

    def observed(values, x):
        def loss(values, x):
            output = quantized.apply(values, x)
            return jnp.sum(jnp.square(output)), output

        return jax.value_and_grad(loss, (0, 1), has_aux=True)(values, x)

    with row_joins("gate_up", joins=False):
        apart = jax.jit(observed)(variables, inputs)
    with row_joins("gate_up"):
        together = jax.jit(observed)(variables, inputs)
    for actual, expected in zip(jax.tree.leaves(together), jax.tree.leaves(apart), strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises((TypeError, ScopeParamShapeError)):
        quantized.apply(variables, inputs[..., :-1])
    plain = layer.init(jax.random.key(29), inputs)
    with row_joins("gate_up"):
        assert dots(layer.apply, plain, inputs) == 2


@pytest.mark.parametrize("scan", [False, True])
def test_direct_qwix_preserves_projection_sites_in_plain_and_scanned_decoders(scan):
    qwix = pytest.importorskip("qwix")

    from dew.nn.backbones.causal_transformer import CausalTransformer

    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2,
                              mlp_features=32, max_seq_len=8, scan_layers=scan, attention_impl="reference")
    provider = qwix.QtProvider([qwix.QtRule(module_path=".*gate_proj", weight_qtype=jnp.int8,
                                           act_qtype=jnp.int8)])
    quantized = qwix.quantize_model(model, provider)
    tokens = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    variables = quantized.init(jax.random.key(17), tokens)

    def observed(values):
        def loss(params):
            logits = quantized.apply({**values, "params": params}, tokens)
            return jnp.sum(jnp.square(logits)), logits

        return jax.value_and_grad(loss, has_aux=True)(values["params"])

    with row_joins("gate_up", joins=False):
        apart = jax.jit(observed)(variables)
    with row_joins("gate_up"):
        together = jax.jit(observed)(variables)
    for actual, expected in zip(jax.tree.leaves(together), jax.tree.leaves(apart), strict=True):
        np.testing.assert_array_equal(actual, expected)
    plain = model.init(jax.random.key(31), tokens)
    with row_joins("gate_up", joins=False):
        separate = dots(model.apply, plain, tokens)
    with row_joins("gate_up"):
        assert dots(model.apply, plain, tokens) < separate


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_delta_nets_gates_joined_are_as_exact_as_apart_from_float64(dtype):
    """A gated delta net whose row joins its b and a projections ('delta_gates') runs them as one
    GEMM, one dot fewer, and its output and the gradients of its input and every parameter lie within
    the layer's with them apart, on par from float64 over drawn inputs (check 1)."""
    layer = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                          dtype=dtype)
    params = layer.init(jax.random.key(9), jnp.zeros((2, 40, 64), dtype))

    draw = drawing(params, (2, 40, 64), dtype)

    x = draw(0)[1]
    with row_joins("delta_gates", joins=False):
        apart = dots(layer.apply, params, x)
    with row_joins("delta_gates"):
        assert dots(layer.apply, params, x) == apart - 1
    assert_joined_as_exact("delta_gates", with_gradients(layer.apply), draw,
                           with_gradients(layer.clone(dtype=jnp.float64).apply),
                           "the delta net's gates joined", seeds=JOINED_SEEDS["delta_gates", dtype])


def test_a_gate_multiplier_keeps_gate_and_up_joined():
    """Falcon-H1's gate multiplier (`GatedMLP.gate_multiplier`) scales the gate half of the joined GEMM's
    output as it scales the gate apart (ledger 3632): its row still joins gate and up, two dots where it
    ran three, and its output and gradients lie within tests/reference_error.py's rule of the layer's with
    them apart, on par from float64 over drawn inputs (check 1)."""
    layer = GatedMLP(hidden_features=96, out_features=64, gate_multiplier=0.75)
    params = layer.init(jax.random.key(4), jnp.zeros((2, 33, 64)))
    draw = drawing(params, (2, 33, 64), jnp.float32)
    x = draw(0)[1]
    with row_joins("gate_up", joins=False):
        assert dots(layer.apply, params, x) == 3
    with row_joins("gate_up"):
        assert dots(layer.apply, params, x) == 2
    assert_joined_as_exact("gate_up", with_gradients(layer.apply), draw,
                           with_gradients(layer.clone(dtype=jnp.float64).apply), "gate multiplier joined",
                           seeds=JOINED_SEEDS["gate multiplier"])


@pytest.mark.parametrize("recipe", ["blockwise_fp8", "mxfp8", "nvfp4"])
def test_a_delta_nets_gates_joined_under_a_recipe_are_as_exact_as_apart_from_float64(recipe):
    """Under each preset's low-precision recipe, a gated delta net of Qwen3.5-0.8B's 16 value heads, its b
    and a gates 16 wide each and 32 joined, whose row joins them: as exact as the layer with them apart,
    output and every gradient, from float64 (check 1), every arm quantized as the gates are apart. A join
    whose product the recipe would quantize where neither gate is (MXFP8's 32-wide blocks), or would not
    where both are (NVFP4's 16), runs them apart (train 928 F1: joined, MXFP8 took the gates' product
    on the GPUs whose cells join them); where the recipe takes neither and not their join (the blockwise
    recipe's 128-wide blocks), they join, one dot fewer."""
    from dew.nn.precision import NUMERICS_PRESETS, projection_role

    numerics = next(each for each in NUMERICS_PRESETS.values()
                    if each.low_precision is not None and each.low_precision.recipe == recipe)
    layer = GatedDeltaNet(emb_features=128, num_k_heads=4, num_v_heads=16, head_k_dim=32, head_v_dim=32,
                          dtype=jnp.bfloat16)
    params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(23), leaf.shape),
                          layer.init(jax.random.key(22), jnp.zeros((2, 32, 128), jnp.bfloat16)))
    draw = drawing(params, (2, 32, 128), jnp.bfloat16)

    def under(run):
        def recipe_run(*operands):
            with generation.run_precision(numerics), projection_role("mixer"):
                return run(*operands)
        return recipe_run

    x = draw(0)[1]
    with row_joins("delta_gates", joins=False):
        apart = dots(under(layer.apply), params, x)
    with row_joins("delta_gates"):
        assert dots(under(layer.apply), params, x) == apart - (recipe == "blockwise_fp8")
    assert_joined_as_exact("delta_gates", under(with_gradients(layer.apply)), draw,
                           with_gradients(layer.clone(dtype=jnp.float64).apply),
                           f"the delta net's gates joined under {recipe}", seeds=JOINED_SEEDS["recipe gates"])


def _attention(bias: bool, dtype=jnp.bfloat16) -> CausalSelfAttention:
    """An attention layer of four query heads over two key heads, 64 wide: its query, key and value
    projections of 256 features 256, 128 and 128 wide, their bias drawn where `bias`."""
    return CausalSelfAttention(MixerContext(emb_features=256, num_heads=4, num_kv_heads=2, head_dim=64,
                                            max_seq_len=64, attention_bias=bias, dtype=dtype))


@pytest.mark.parametrize("bias", [False, True])
def test_an_attention_layers_query_key_and_value_side_by_side_are_as_exact_as_apart_from_float64(bias):
    """A bf16 attention layer whose query, key and value projections run side by side (`qkv` joined, the GEMM
    template named and interpreted on an assumed H100, so a launch takes them: one product of unequal widths
    beside, `joined.beside_dense`, its input gradient chained) computes its output and the gradients of its
    input and every parameter within tests/reference_error.py's rule of the layer's with them apart, on par
    from float64 over drawn inputs (check 1), its drawn biases each added to its own projection; the joined
    arm binds the side-by-side launch (head review of b68fc2382: at this geometry, no template named, both
    arms ran apart)."""
    from dew.nn.kernels import tuning

    layer = _attention(bias)
    params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(6), leaf.shape),
                          layer.init(jax.random.key(4), jnp.zeros((2, 64, 256), jnp.bfloat16)))
    with (generation.assumed("sm90", interpreted=True),
          generation.chosen(attention="xla", grouped_matmul="mosaic"), tuning.seen() as launches):
        wide = layer.clone(ctx=dataclasses.replace(layer.ctx, dtype=jnp.float64))
        draw = drawing(params, (2, 64, 256), jnp.bfloat16)
        assert_joined_as_exact("qkv", with_gradients(layer.apply), draw, with_gradients(wide.apply),
                               "query, key and value side by side", seeds=JOINED_SEEDS["qkv", bias])
    assert any(bound.arguments.get("beside") for _, bound in launches.taken), launches.taken


def test_an_attention_layers_training_step_takes_its_input_gradient_chained():
    """A bf16 attention layer's training step traced on an assumed H100 with the GEMM template named, the
    `qkv` selector's own pick: its query, key and value run side by side (the route `joined`) and their input
    gradient is one chained launch (`_chained_step`); with the projections named apart, none is. The chained
    launch's numerics are tests/test_gemm_paired.py's (check 1), the side-by-side layer's the test above."""
    layer, x, params, under = _watched_layer()
    assert _qkv_routes(layer, params, x, *under())[:1] == ["joined"]
    assert _chained_step(layer, params, x, *under())
    assert not _chained_step(layer, params, x, *under(), generation.chosen(qkv="apart"))

def _qkv_routes(layer, params, x, *contexts, apply=None) -> list:
    """The `qkv` routes `layer`'s call over `x` (abstract) reports, traced under `contexts` (its `apply`
    where given: a jitted one, or one asking collections)."""
    with contextlib.ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        routes = stack.enter_context(generation.routes())
        jax.eval_shape(apply or layer.apply, params, x)
    return [route.kernel for route in routes if route.choice == "qkv"]


def _training_step(layer):
    """`layer`'s training step: the gradients of its output's sum by its parameters and its input."""
    def step(params, x):
        return jax.grad(lambda params, x: jnp.sum(layer.apply(params, x).astype(jnp.float32)),
                        argnums=(0, 1))(params, x)

    return step


def _chained_step(layer, params, x, *contexts) -> list:
    """The launches a training step of `layer` over `x` (abstract: traced, run nowhere) binds under
    `contexts` that take their input gradient's products as one chained call (`gemm._one_launch`)."""
    from dew.nn.kernels import tuning

    with contextlib.ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        launches = stack.enter_context(tuning.seen())
        jax.eval_shape(_training_step(layer), params, x)
    return [bound for _, bound in launches.taken if bound.arguments.get("chained")]


def _llama_8b() -> CausalSelfAttention:
    """An attention layer at Llama-3.1-8B's projections: 4,096 features, 32 query heads over 8 key heads of
    128, bf16, no query or key norm (Llama's)."""
    return CausalSelfAttention(MixerContext(emb_features=4096, num_heads=32, num_kv_heads=8, head_dim=128,
                                            max_seq_len=4096, qk_norm=False, dtype=jnp.bfloat16))


def test_an_8b_layers_training_step_takes_its_input_gradient_as_one_chained_launch():
    """An attention layer at Llama-3.1-8B's projections (4,096 features, 32 query heads over 8 key heads of
    128) over 12,288 bf16 rows, its training step traced as an assumed H100 runs it, the selectors' own picks
    (nothing named): its query, key and value run side by side (`qkv` `joined`), and the step binds their
    input gradient as one chained launch, its products' cell its own (`Kind.together`, M-EM-73's receipt),
    where the ordinary products' cells, XLA's on Hopper, kept them apart (head review of 4ddd0d7ea F1)."""
    layer = _llama_8b()
    x = jax.ShapeDtypeStruct((3, 4096, 4096), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)

    def under():
        return generation.assumed("sm90"), generation.chosen(attention="xla")
    assert _qkv_routes(layer, params, x, *under())[:1] == ["joined"]
    assert _chained_step(layer, params, x, *under())


def test_projections_no_launch_takes_side_by_side_run_apart_where_the_template_is_named():
    """On an assumed A100 with the GEMM template named, whose cp.async rings take no matrices side by side:
    the attention layer's query, key and value run apart, each its own call, and the layer traces, where the
    side-by-side call refused the named kernel (head review of 4ddd0d7ea F1)."""
    layer = _attention(bias=False)
    x = jax.ShapeDtypeStruct((2, 64, 256), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)
    assert set(_qkv_routes(layer, params, x, generation.assumed("sm80"),
                           generation.chosen(grouped_matmul="mosaic", attention="xla"))) == {"apart"}


@pytest.mark.parametrize("axis", ["data", "fsdp"])
def test_projections_side_by_side_on_a_mesh_splitting_their_rows_take_each_devices_rows(axis):
    """An attention layer's training step on an abstract mesh of two H100s splitting its 4 x 64 rows
    (`generation.gpu_mesh`), its parameters placed by a layout, the GEMM template named: its query, key and
    value run side by side on each device's 128 rows (`shard_placement`), their input gradient one chained
    launch over those rows, the kernels gathered whole there under fsdp (an all-gather in the step), where
    the call over every device's rows refused the named kernel's placement (head review of 4ddd0d7ea F2,
    b68fc2382: a concrete CUDA topology, and replicated parameters, checked no rows or collectives)."""
    from jax.sharding import Mesh

    from dew.distributed.placement import Layout
    from dew.mesh import DEFAULT_RULES
    from dew.nn.kernels import tuning
    from dew.nn.sharding import placing

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:2]), (axis,)), "sm90")
    layer = _attention(bias=False)
    x = jax.ShapeDtypeStruct((4, 64, 256), jnp.bfloat16)
    params = nn.unbox(jax.eval_shape(layer.init, jax.random.key(0), x))
    contexts = (jax.sharding.use_abstract_mesh(mesh), nn.logical_axis_rules(DEFAULT_RULES),
                placing(Layout(min_shard=1)), generation.assumed("sm90"),
                generation.chosen(grouped_matmul="mosaic", attention="xla"))
    with contextlib.ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        launches = stack.enter_context(tuning.seen())
        routes = stack.enter_context(generation.routes())
        graph = jax.make_jaxpr(_training_step(layer))(params, x)
    assert [route.kernel for route in routes if route.choice == "qkv"][:1] == ["joined"]
    chained = [bound for _, bound in launches.taken if bound.arguments.get("chained")]
    assert chained and {bound.arguments["tokens"].shape[0] for bound in chained} == {2 * 64}, chained
    gathered = any(equation.primitive.name == "all_gather" for equation in equations(graph))
    assert gathered == (axis == "fsdp"), axis


def test_projections_whose_kernels_vary_over_a_checked_shard_map_run_side_by_side():
    """Llama-3.1-8B's attention layer (`_llama_8b`) over 12,288 bf16 rows applied inside a checked shard_map
    over two devices, its rows replicated, on an assumed H100, nothing named but the attention kernel: with
    its parameters whole or varying over the axis (`jax.lax.pcast`), each device's launch reads its own rows
    and kernels, its results typed as they vary (row 9808), so its query, key and value run side by side
    (the receipt's geometry), the admission and its launch reading one condition, no automatic mesh axis
    (head review of b68fc2382)."""
    from jax.sharding import Mesh, PartitionSpec

    mesh = Mesh(np.array(jax.devices()[:2]), ("data",))
    layer = _llama_8b()
    x = jax.ShapeDtypeStruct((3, 4096, 4096), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)
    for varying in (False, True):
        def applied(params, x, varying=varying):
            return layer.apply(jax.lax.pcast(params, ("data",), to="varying") if varying else params, x)

        mapped = jax.shard_map(applied, mesh=mesh, in_specs=(PartitionSpec(), PartitionSpec()),
                               out_specs=PartitionSpec("data") if varying else PartitionSpec())
        under = (generation.assumed("sm90"), generation.chosen(attention="xla"))
        assert _qkv_routes(layer, params, x, *under, apply=mapped)[:1] == ["joined"], varying


def test_projections_that_cast_their_input_run_apart():
    """An fp16 attention layer applied to fp32 rows on an assumed H100, the GEMM template named: each
    projection's call casts the rows to fp16, its input cotangent widened back to fp32 before the three sum,
    which one product's fp16 sum would not keep (base review b68fc2382 F1: 40,000 and 30,000 summed to
    fp16's infinity, 70,000 apart), so they run apart; fp16 rows join."""
    layer = _attention(bias=False, dtype=jnp.float16)
    for rows, kernel in ((jnp.float32, "apart"), (jnp.float16, "joined")):
        x = jax.ShapeDtypeStruct((2, 64, 256), rows)
        params = jax.eval_shape(layer.init, jax.random.key(0), x)
        under = (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", attention="xla"))
        assert _qkv_routes(layer, params, x, *under)[:1] == [kernel], rows


def test_a_flax_interceptor_of_a_projection_keeps_its_own_call():
    """A bf16 attention layer traced as an assumed H100 runs it, the GEMM template named and interpreted,
    under a Flax method interceptor that zeros `v_proj`'s output (`nn.intercept_methods`): its query, key
    and value run apart, each through its own call, so the interceptor sees the value projection and the
    layer's output is the one with them named apart, where the joined layer bypassed the call (base review
    760be77fe F1)."""
    layer = _attention(bias=False)
    x = jax.random.normal(jax.random.key(1), (2, 64, 256)).astype(jnp.bfloat16)
    params = layer.init(jax.random.key(0), x)

    def zero_value(next_fun, args, kwargs, context):
        out = next_fun(*args, **kwargs)
        return jnp.zeros_like(out) if context.module.name == "v_proj" else out

    def applied(*named):
        with (generation.assumed("sm90", interpreted=True),
              generation.chosen(grouped_matmul="mosaic", attention="xla", **dict(named)),
              generation.routes() as routes, nn.intercept_methods(zero_value)):
            out = layer.apply(params, x)
        return out, [route.kernel for route in routes if route.choice == "qkv"]

    intercepted, kernels = applied()
    apart, _ = applied(("qkv", "apart"))
    assert set(kernels) == {"apart"}, kernels
    np.testing.assert_array_equal(np.asarray(intercepted, np.float32), np.asarray(apart, np.float32))


def test_a_named_side_by_side_product_launches_where_legal_and_refuses_by_name_where_not():
    """An attention layer of 256 bf16 rows (no receipt timed its geometry) with `qkv` named `joined`, nothing
    else named: on an assumed H100 its query, key and value run side by side and their training step binds
    the chained input gradient, the template's launches legal there, where the measured route's unmeasured
    default took them apart (base review 760be77fe F2); on an assumed A100, whose rings take no matrices side
    by side, the named choice is refused by name (`KernelRefused`)."""
    from dew.nn.kernels.selection import KernelRefused

    layer = _attention(bias=False)
    x = jax.ShapeDtypeStruct((4, 64, 256), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)

    def under(name):
        return generation.assumed(name), generation.chosen(attention="xla", qkv="joined")

    assert _qkv_routes(layer, params, x, *under("sm90"))[:1] == ["joined"]
    assert _chained_step(layer, params, x, *under("sm90"))
    with pytest.raises(KernelRefused):
        _qkv_routes(layer, params, x, *under("sm80"))


def test_a_named_side_by_side_product_keeps_its_callers_grouped_choice_and_refuses_one_it_cannot_run():
    """`qkv` named `joined` beside `grouped_matmul` named: the template's own (`mosaic`) runs side by side;
    another (`xla`) is refused by name (`KernelRefused`, code `grouped`), where the side-by-side product
    replaced the caller's named grouped choice with the template's in its probe and its execution (head
    review of 0099a4701 F1). Named nothing, the template's is the side-by-side product's own."""
    from dew.nn.kernels.selection import KernelRefused

    layer = _attention(bias=False)
    x = jax.ShapeDtypeStruct((4, 64, 256), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)

    def under(**grouped):
        return generation.assumed("sm90"), generation.chosen(attention="xla", qkv="joined", **grouped)

    assert _qkv_routes(layer, params, x, *under(grouped_matmul="mosaic"))[:1] == ["joined"]
    with pytest.raises(KernelRefused) as refused:
        _qkv_routes(layer, params, x, *under(grouped_matmul="xla"))
    assert refused.value.first.code == "grouped", refused.value.first


def test_a_side_by_side_forward_runs_the_route_its_admission_decided():
    """An attention layer's query, key and value admitted side by side on an assumed H100, the GEMM template
    named: the route `gemm.side_by_side` decides for their rows and kernels (the admission's) is the route
    their forward reports, once (base review b68fc2382 F5: the call selected its launch a second time and
    reported that one)."""
    from dew.nn.kernels import gemm

    layer, x, params, under = _watched_layer()
    unboxed = nn.unbox(params)
    kernels = tuple(jax.ShapeDtypeStruct(unboxed["params"][name]["kernel"].shape, jnp.bfloat16)
                    for name in ("q_proj", "k_proj", "v_proj"))
    rows = jax.ShapeDtypeStruct((x.shape[0] * x.shape[1], x.shape[2]), x.dtype)
    with contextlib.ExitStack() as stack:
        for context in under():
            stack.enter_context(context)
        admitted = gemm.side_by_side(rows, kernels, x.dtype, None, whole=True)
        routes = stack.enter_context(generation.routes())
        jax.eval_shape(layer.apply, params, x)
    assert admitted is not None and admitted.kernel == "mosaic", admitted
    assert [route.kernel for route in routes if route.choice == "qkv"] == ["joined"]
    assert list(routes).count(admitted) == 1, routes


def _watched_layer():
    """`_attention`'s layer, its abstract input and parameters, and the contexts its watch cells trace under,
    made afresh at each call (a context is entered once): an assumed H100, the GEMM template named."""
    layer = _attention(bias=False)
    x = jax.ShapeDtypeStruct((2, 64, 256), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)
    return layer, x, params, lambda: (generation.assumed("sm90"),
                                      generation.chosen(grouped_matmul="mosaic", attention="xla"))


def test_a_watched_projection_keeps_its_own_call_and_its_output_is_sown():
    """An attention layer whose key projection a log watches (`inspection.watching`): its query, key and value
    run apart, and the watch sows the key projection's output (head review of 4ddd0d7ea F5); unwatched they
    run side by side."""
    from dew.nn.inspection import INSPECTED, watching

    layer, x, params, under = _watched_layer()
    assert _qkv_routes(layer, params, x, *under())[:1] == ["joined"]

    def inspected(params, x):
        return layer.apply(params, x, mutable=[INSPECTED])

    assert set(_qkv_routes(layer, params, x, *under(), watching(("k_proj",)), apply=inspected)) == {"apart"}
    with contextlib.ExitStack() as stack:
        for context in (*under(), watching(("k_proj",))):
            stack.enter_context(context)
        _, sown = jax.eval_shape(inspected, params, x)
    assert "k_proj" in sown[INSPECTED], sown[INSPECTED].keys()


@pytest.mark.parametrize(("rows", "kernel"), [((4, 3072), "joined"), ((1, 2048), "apart")],
                         ids=["4x3072", "1x2048"])
def test_a_4b_layers_query_key_and_value_join_where_a_receipt_timed_their_chain(rows, kernel):
    """Qwen3.5-4B's attention layer (2,560 features, 16 query heads over 4 key heads of 256, its query
    doubled by its output gate) traced as an assumed H100 runs it, nothing named but the attention kernel:
    at 4 x 3072 rows its query, key and value run side by side, their chained input gradient M-EM-73's win
    (`grouped_cells.TOGETHER`); at 1 x 2048, where the chain took 1.32 times XLA's time, each its own call
    (head review of 4d6c7a69c F1)."""
    layer = CausalSelfAttention(MixerContext(emb_features=2560, num_heads=16, num_kv_heads=4, head_dim=256,
                                             max_seq_len=4096, output_gate=True, dtype=jnp.bfloat16))
    x = jax.ShapeDtypeStruct((*rows, 2560), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)
    assert _qkv_routes(layer, params, x, generation.assumed("sm90"),
                       generation.chosen(attention="xla"))[:1] == [kernel]


def test_captured_intermediates_keep_each_projections_own_call():
    """An attention layer applied capturing each module's output (Flax's `capture_intermediates`): its query,
    key and value run apart, and the key projection's output is captured (head review of f7c7072aa F8)."""
    layer, x, params, under = _watched_layer()

    def captured(params, x):
        return layer.apply(params, x, capture_intermediates=True, mutable=["intermediates"])

    assert set(_qkv_routes(layer, params, x, *under(), apply=captured)) == {"apart"}
    with contextlib.ExitStack() as stack:
        for context in under():
            stack.enter_context(context)
        _, state = jax.eval_shape(captured, params, x)
    assert "k_proj" in state["intermediates"], state["intermediates"].keys()


def test_a_tree_missing_a_projections_bias_is_refused_as_its_own_call_refuses_it():
    """A biased attention layer's variables without the key projection's bias, traced as an assumed H100
    runs it with the GEMM template named: its key projection's own call reports the missing parameter,
    where reading the leaves side by side left the bias out (head review of 4ddd0d7ea F4)."""
    from flax.errors import ScopeParamNotFoundError

    layer = _attention(bias=True)
    x = jax.ShapeDtypeStruct((2, 64, 256), jnp.bfloat16)
    params = jax.eval_shape(layer.init, jax.random.key(0), x)
    del params["params"]["k_proj"]["bias"]
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", attention="xla"),
          pytest.raises(ScopeParamNotFoundError)):
        jax.eval_shape(layer.apply, params, x)
