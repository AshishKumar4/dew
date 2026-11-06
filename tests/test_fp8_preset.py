"""The 'fp8' run numerics (`dew.nn.precision.FP8`, row 2513): DeepSeek-V3's blockwise FP8 recipe on the
projections whose role it takes, the rest in the run's dtypes."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import float64_twin
from reference_error import as_given, assert_as_exact, widened, with_gradients

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import gemm, generation
from dew.nn.precision import FP8, LowPrecision, Numerics, RecipeRefused, numerics_of

VOCAB, WIDTH = 256, 128
SEEDS = 64
"""The preset step's check-1 seeds, sized by `seeds_for` on armada."""


def mixed(dtype=jnp.bfloat16) -> CausalTransformer:
    """Two layers 128 wide, each width a whole number of the recipe's 128-wide blocks: a dense one, then a
    mixture of four experts with a gated shared expert (Qwen2-MoE's), and an untied head."""
    return CausalTransformer(
        vocab_size=VOCAB, emb_features=WIDTH, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=64,
        mlp_features=2 * WIDTH, max_seq_len=64, dtype=dtype, tie_embeddings=False,
        mixture={"experts": 4, "top_k": 2, "expert_features": WIDTH, "layers": (1,),
                 "shared_features": WIDTH, "shared_gate": True})


def test_the_fp8_preset_names_the_recipe_and_keeps_the_runs_dtypes():
    """`--precision fp8` names DeepSeek-V3's recipe over bf16-mixed: the mixers', feed-forwards' and
    experts' projections in blockwise FP8, and a recipe naming a role no projection plays is refused."""
    assert numerics_of("fp8") == FP8 == Numerics("bf16-mixed", low_precision=LowPrecision())
    assert FP8.low_precision is not None and FP8.low_precision.roles == ("mixer", "mlp", "experts")
    with pytest.raises(ValueError):
        LowPrecision(roles=("mixer", "norm"))
    with pytest.raises(ValueError):
        LowPrecision(recipe="nvfp4")


def test_the_routers_gates_and_head_never_reach_the_fp8_projection():
    """Under the 'fp8' numerics a model's forward routes each projection by its role (the routes its trace
    reports, `generation.routes`): every attention projection, the dense feed-forward's, the shared
    expert's and the experts' three take the recipe, seventeen in all; the shared expert's gate keeps the
    run's dtypes, its route naming its role, and the router, whose scores are an fp32 einsum, the untied
    head, whose product keeps its fp32 result, the norms and the embedding reach no product the recipe
    reads."""
    model = mixed()
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (2, 64)), jnp.int32)
    params = jax.eval_shape(model.init, jax.random.key(0), tokens)
    with generation.run_precision(FP8), generation.routes() as found:
        jax.eval_shape(model.apply, params, tokens)
    recipe = [route for route in found if route.choice == "blockwise_fp8"]
    kept = sorted(str(route.refusal) for route in recipe if route.kernel == "unquantized")
    taken = [route for route in recipe if route.kernel != "unquantized"]
    assert kept == ["a shared_gate projection keeps the run's dtypes under blockwise_fp8"], kept
    # Layer 0: q, k, v, o and the dense gate, up and down; layer 1: q, k, v, o, the shared expert's three
    # and the experts' gate, up and down.
    assert len(taken) == 17, [route.site for route in taken]


def test_a_grouped_output_projection_takes_the_recipe_group_by_group():
    """A batched projection (DeepSeek V4's grouped `o_a_proj`, a `DenseGeneral` whose kernel holds a
    matrix a group) under the 'fp8' numerics takes the recipe as its groups' matrices, each group's rows
    against its own: bit for bit the recipe over each group apart."""
    from dew.nn.dense import DenseGeneral
    from dew.nn.precision import projection_role

    layer = DenseGeneral(WIDTH, batch_dims=(0,), axis=-1, use_bias=False, dtype=jnp.float32)
    # [groups, ..., in], as V4's grouped output moves its groups first (`_output`).
    x = jnp.asarray(np.random.default_rng(1).normal(size=(3, 2, 5, WIDTH)), jnp.float32)
    params = layer.init(jax.random.key(0), x)
    with generation.run_precision(FP8), projection_role("mixer"), generation.routes() as found:
        out = layer.apply(params, x)
    assert [route.kernel != "unquantized" for route in found if route.choice == "blockwise_fp8"] == [True]
    kernel = params["params"]["kernel"]
    for group in range(x.shape[0]):
        with generation.run_precision(FP8), projection_role("mixer"):
            alone = gemm.recipe_projection(x[group].reshape(-1, WIDTH), kernel[group], out_dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(out[group]).reshape(-1, WIDTH), np.asarray(alone))


def test_a_projection_the_recipe_takes_under_a_product_wrapper_is_refused():
    """A projection the 'fp8' numerics take by role whose product a wrapper holds (an adapter's or a
    quantizer's interception) would train unquantized, silently: refused, and a role the recipe keeps is
    not."""
    from dew import _cache
    from dew.nn.dense import Dense
    from dew.nn.precision import projection_role

    layer = Dense(WIDTH, dtype=jnp.float32)
    x = jnp.ones((2, WIDTH), jnp.float32)
    params = layer.init(jax.random.key(0), x)
    with generation.run_precision(FP8), _cache.INTERCEPTED(new_value=True):
        with projection_role("mlp"), pytest.raises(RecipeRefused):
            layer.apply(params, x)
        with projection_role("router"):
            layer.apply(params, x)


def test_an_adapted_projection_the_recipe_takes_is_refused_on_its_fused_product():
    """An adapter's projection the 'fp8' numerics take by role, its update fused into the base product
    (`generation.chosen(low_rank='fused')`, `gemm.low_rank`), would train unquantized, silently, as the
    unfused branch would: refused where the fused product is traced."""
    from dew.lora import LoRA

    model = mixed(jnp.float32).clone(mixture=None)
    tokens = jnp.ones((1, 8), jnp.int32)
    adapter = LoRA(rank=4, modules=("up_proj",)).apply(model, model.init(jax.random.key(0), tokens), key=0)
    with generation.run_precision(FP8), generation.chosen(low_rank="fused"), pytest.raises(RecipeRefused):
        adapter.model.apply(adapter.variables, tokens)


def test_the_experts_kernel_gradient_keeps_the_master_dtype():
    """The experts' kernel gradient under the 'fp8' numerics keeps the master's float32, as
    `expert_projection`'s contract says: 257 rows of ones through a matrix of ones, each kernel entry's
    gradient 257, which a bfloat16 rounding would make 256."""
    from dew.nn.moe import expert_projection

    rows = jnp.ones((257, WIDTH), jnp.bfloat16)
    stack = jnp.ones((1, WIDTH, WIDTH), jnp.float32)
    sizes = jnp.asarray([257], jnp.int32)

    def summed(stack):
        with generation.run_precision(FP8):
            projected = expert_projection(rows, stack, sizes, jnp.bfloat16, "auto", None)
            return jnp.sum(projected.astype(jnp.float32))

    gradient = jax.grad(summed)(stack)
    assert gradient.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(gradient), np.full(stack.shape, 257.0, np.float32))


def emulated(dtype):
    """`gemm.blockwise_fp8_projection` emulated in `dtype`: the recipe's quantizers
    (`gemm.rows_quantized`, `gemm.blocks_quantized`), each group's product of the dequantized values in
    `dtype` at its highest precision, and its gradients the recipe's (`blockwise_matrix.recipe_truth`'s),
    each row's group read from `group_blocks` as the model's routing gives it: float64 the step's truth,
    float32 the fake-quantized reference a recipe is measured against."""
    def dequantized(codes, scales):
        return codes.astype(dtype) * jnp.repeat(scales.astype(dtype), 128, axis=-1)

    def blocks(w):
        codes, scales = gemm.blocks_quantized(w)
        return codes.astype(dtype) * jnp.repeat(jnp.repeat(scales.astype(dtype), 128, axis=1), 128, axis=2)

    def owned(group_blocks, rows):
        """`[groups, rows]`: whether each row is its group's."""
        ends = jnp.cumsum(group_blocks) * 128
        group = jnp.searchsorted(ends, jnp.arange(rows), side="right")
        return group[None, :] == jnp.arange(group_blocks.shape[0])[:, None]

    def grouped(lhs, matrices, owners):
        return sum(jnp.where(owners[g][:, None], jnp.matmul(lhs, matrices[g], precision="highest"), 0)
                   for g in range(matrices.shape[0]))

    @jax.custom_vjp
    def run(x, w, group_blocks):
        return grouped(dequantized(*gemm.rows_quantized(x)), blocks(w), owned(group_blocks, x.shape[0]))

    def forward(x, w, group_blocks):
        return run(x, w, group_blocks), (x, w, group_blocks)

    def backward(residuals, cotangent):
        x, w, group_blocks = residuals
        owners = owned(group_blocks, x.shape[0])
        d_x = grouped(dequantized(*gemm.rows_quantized(cotangent)), jnp.swapaxes(blocks(w), 1, 2), owners)
        rows_t = dequantized(*gemm.rows_quantized(x.T))
        dy_t = dequantized(*gemm.rows_quantized(cotangent.T))
        d_w = jnp.stack([jnp.matmul(rows_t * owners[g], dy_t.T, precision="highest")
                         for g in range(w.shape[0])])
        return d_x.astype(x.dtype), d_w.astype(w.dtype), None

    run.defvjp(forward, backward)

    def projection(x, w, group_blocks, *, out_dtype, interpret_on_cpu=False):
        del interpret_on_cpu
        return run(x, w, group_blocks).astype(out_dtype)

    return projection


def step(model):
    """One training step's loss and gradients of `model`'s parameters under the 'fp8' numerics: next-token
    cross entropy over the drawn tokens."""
    def loss(params, tokens):
        with generation.run_precision(FP8):
            logits = model.apply(params, tokens[:, :-1])
        logits = logits.astype(jnp.promote_types(logits.dtype, jnp.float32))
        picked = jnp.take_along_axis(jax.nn.log_softmax(logits), tokens[:, 1:, None], axis=-1)
        return -jnp.mean(picked)
    return loss


def test_a_training_step_in_the_fp8_preset_is_on_par_with_its_fake_quantized_reference():
    """One training step of a dense and MoE stack under the 'fp8' numerics (Dew's recipe through XLA's arm
    on a CPU), its loss and every parameter's gradient, on par with the same step whose quantized
    projections are the recipe emulated in float32 (the fake-quantized reference a recipe is measured
    against), from the step in float64 with the recipe emulated in float64 (check 1)."""
    model = mixed()
    tokens = jnp.zeros((2, 65), jnp.int32)
    shapes = jax.eval_shape(model.init, jax.random.key(0), tokens[:, :-1])

    def draw(seed):
        rng = np.random.default_rng(seed)
        params = jax.tree.map(lambda leaf: jnp.asarray(rng.normal(size=leaf.shape) * 0.05, leaf.dtype),
                              shapes)
        drawn = jnp.asarray(rng.integers(0, VOCAB, (2, 65)), jnp.int32)
        return params, drawn, jnp.ones((), jnp.float32)

    def arm(model, projection=None):
        def run(params, tokens, cotangent):
            stepped = with_gradients(lambda params: step(model)(params, tokens))
            if projection is None:
                return stepped(params, cotangent)
            with mock.patch.object(gemm, "blockwise_fp8_projection", projection):
                return stepped(params, cotangent)
        return run

    @as_given
    def truth(params, tokens, cotangent):
        with jax.enable_x64(new_val=True):
            return arm(float64_twin(model), emulated(jnp.float64))(widened(params), tokens,
                                                                  cotangent.astype(jnp.float64))

    assert_as_exact(jax.jit(arm(model)), jax.jit(arm(model, emulated(jnp.float32))), draw, "fp8 preset step",
                    seeds=SEEDS, truth_fn=truth)
