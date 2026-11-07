"""The 'fp8' run numerics (`dew.nn.precision.FP8`, row 2513): DeepSeek-V3's blockwise FP8 recipe on the
projections whose role it takes, the rest in the run's dtypes."""

import contextlib
import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import float64_twin
from reference_error import as_given, assert_as_exact, assert_fp32_reduction_bound, widened, with_gradients

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import gemm, generation
from dew.nn.precision import FP8, LowPrecision, Numerics, RecipeRefused, numerics_of

VOCAB, WIDTH = 256, 128
SEEDS = 64
"""The recipe's check-1 seeds, the projection's and the step's (the step's sized by `seeds_for` on armada)."""


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
    against its own: each group's output the recipe over that group apart, the two the same FP8 products
    summed in fp32 by two programs, within two fp32 reductions' bound of each other
    (`assert_fp32_reduction_bound`)."""
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
    # The recipe's operands, dequantized: E4M3 codes times power-of-two scales, whose products are exact.
    block_codes, block_scales = gemm.blocks_quantized(kernel)
    matrices = np.asarray(block_codes, np.float64) * np.repeat(np.repeat(
        np.asarray(block_scales, np.float64), WIDTH, axis=1), WIDTH, axis=2)
    for group in range(x.shape[0]):
        rows = x[group].reshape(-1, WIDTH)
        with generation.run_precision(FP8), projection_role("mixer"):
            alone = gemm.recipe_projection(rows, kernel[group], out_dtype=jnp.float32)
        row_codes, row_scales = gemm.rows_quantized(rows)
        dequantized = np.asarray(row_codes, np.float64) * np.repeat(np.asarray(row_scales, np.float64), WIDTH,
                                                                    axis=-1)
        assert_fp32_reduction_bound(np.asarray(out[group]).reshape(-1, WIDTH), alone,
                                    np.abs(dequantized) @ np.abs(matrices[group]), WIDTH)


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


def test_an_adapters_fused_product_refuses_a_projection_the_recipe_takes():
    """An adapter's update fused into its projection's product (`gemm.low_rank`, LoRA's fused site under
    `generation.chosen(low_rank='fused')`, which returns before the layer's own product) on a projection
    the 'fp8' numerics take by role would train it unquantized, silently: refused, and a role the recipe
    keeps is not."""
    from dew.nn.precision import projection_role

    rows, rank = jnp.ones((4, WIDTH), jnp.float32), 4
    kernel, a, b = jnp.ones((WIDTH, WIDTH)), jnp.ones((WIDTH, rank)), jnp.zeros((rank, WIDTH))
    with generation.run_precision(FP8):
        with projection_role("mlp"), pytest.raises(RecipeRefused):
            gemm.low_rank(rows, kernel, a, b, 1.0, out_dtype=jnp.float32)
        with projection_role("router"):
            gemm.low_rank(rows, kernel, a, b, 1.0, out_dtype=jnp.float32)


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


def laid_projection(dtype):
    """`gemm.recipe_projection` of a projection's rows written apart in `dtype`: `run(rows, matrices, sizes)`,
    each group's rows (`sizes` a group's, in order, any rows past their sum outside every group) laid in
    whole 128-row blocks behind zero rows, a run of zero blocks after them, the recipe emulated over them
    (`emulated`), and each row's output read back, a row outside every group's zero."""
    project = emulated(dtype)

    def run(rows, matrices, sizes):
        count, groups = rows.shape[0], matrices.shape[0]
        blocks = -(-sizes // 128)
        ends = jnp.cumsum(sizes)
        group = jnp.searchsorted(ends, jnp.arange(count), side="right")
        held = jnp.minimum(group, groups - 1)
        laid_rows = -(-(count + groups * 127) // 128) * 128
        place = jnp.where(group < groups, ((jnp.cumsum(blocks) - blocks) * 128)[held] + jnp.arange(count)
                          - (ends - sizes)[held], laid_rows)
        laid = jnp.zeros((laid_rows, rows.shape[1]), rows.dtype).at[place].set(rows, mode="drop")
        group_blocks = blocks.at[-1].add(laid_rows // 128 - blocks.sum())
        out = project(laid, matrices, group_blocks, out_dtype=dtype)
        return out.at[place].get(mode="fill", fill_value=0)

    return run


@pytest.mark.parametrize("role", ["mlp", "experts"])
def test_the_recipes_projection_is_on_par_with_its_fake_quantized_reference(role):
    """A projection the 'fp8' numerics take (`gemm.recipe_projection`, which a layer of the role hands its
    rows): a dense one's, every row its one group's, and experts' of ragged groups, one empty, with rows
    past them; its output and both gradients on par with the recipe written apart in float32 over the same
    groups' whole 128-row blocks (`laid_projection`: the fake-quantized reference a recipe is measured
    against), from it in float64 (check 1)."""
    from dew.nn.precision import projection_role

    sizes = (200,) if role == "mlp" else (70, 0, 129)
    count = sum(sizes) + (0 if role == "mlp" else 5)
    matrix = (256, WIDTH) if role == "mlp" else (len(sizes), 256, WIDTH)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(count, 256)), jnp.float32),
                jnp.asarray(rng.normal(size=matrix) / 16, jnp.float32),
                jnp.asarray(rng.normal(size=(count, WIDTH)), jnp.float32))

    def projected(rows, matrix):
        with generation.run_precision(FP8), projection_role(role):
            sized = None if role == "mlp" else jnp.asarray(sizes, jnp.int32)
            return gemm.recipe_projection(rows, matrix, sized, out_dtype=jnp.float32)

    @as_given
    def truth(rows, matrix, cotangent):
        with jax.enable_x64(new_val=True):
            return reference(jnp.float64)(*widened((rows, matrix, cotangent)))

    def reference(dtype):
        def run(rows, matrix):
            return laid_projection(dtype)(rows, matrix if matrix.ndim == 3 else matrix[None],
                                          jnp.asarray(sizes, jnp.int32))
        return with_gradients(run)

    assert_as_exact(jax.jit(with_gradients(projected)), jax.jit(reference(jnp.float32)), draw,
                    f"recipe projection {role}", seeds=SEEDS, truth_fn=truth)


def fake_quantized(dtype):
    """A flax method interceptor (`flax.linen.intercept_methods`) running each projection the 'fp8' numerics
    take by role as the recipe emulated in `dtype` (`laid_projection`), written here apart from the model: the
    attention's and the feed-forwards' `Dense` layers (q, k, v and o; gate, up and down, the shared expert's
    among them, not its gate), each a group of every row, and the routed experts (`ExpertMLP`): their slots
    in expert order, token by token within an expert, through the gate and up projections, the gated product
    and the down projection, summed back with their routing weights. Every other module runs as the model
    runs it."""
    from dew.nn.activations import gated_product
    from dew.nn.dense import Dense
    from dew.nn.moe import ExpertMLP

    project = laid_projection(dtype)
    taken = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def dense(module, x):
        kernel = nn.unbox(module.variables["params"]["kernel"])
        compute = module.dtype or jnp.result_type(x, kernel)
        rows = jnp.asarray(x, compute).reshape(-1, x.shape[-1])
        out = project(rows, jnp.asarray(kernel, compute)[None], jnp.asarray([rows.shape[0]], jnp.int32))
        return out.reshape(*x.shape[:-1], kernel.shape[-1]).astype(compute)

    def experts(module, x, weights, indices):
        params = nn.unbox(module.variables["params"])
        compute = module.dtype or x.dtype
        gate, up, down = (jnp.asarray(params[name]["kernel"], compute)
                          for name in ("gate_proj", "up_proj", "down_proj"))
        width, top_k = x.shape[-1], indices.shape[-1]
        chosen = indices.reshape(-1)
        order = jnp.argsort(chosen, stable=True)
        sizes = jnp.bincount(chosen, length=module.num_experts).astype(jnp.int32)
        rows = jnp.asarray(x, compute).reshape(-1, width)[order // top_k]
        hidden = gated_product("swiglu")(project(rows, gate, sizes).astype(compute),
                                         project(rows, up, sizes).astype(compute))
        slots = jnp.zeros((chosen.shape[0], width), compute).at[order].set(
            project(hidden, down, sizes).astype(compute))
        wide = jnp.promote_types(compute, jnp.float32)
        return jnp.einsum("...ke,...k->...e", slots.reshape(*x.shape[:-1], top_k, width).astype(wide),
                          weights.astype(wide)).astype(compute)

    def interceptor(next_fun, args, kwargs, context):
        module = context.module
        if context.method_name == "__call__" and isinstance(module, Dense) and module.name in taken:
            return dense(module, *args)
        if context.method_name == "__call__" and isinstance(module, ExpertMLP):
            return experts(module, *args)
        return next_fun(*args, **kwargs)

    return interceptor


def step(model, numerics, interceptor=None):
    """One training step's loss of `model`'s parameters under `numerics`, its projections through
    `interceptor` where given (`fake_quantized`): next-token cross entropy over the drawn tokens."""
    def loss(params, tokens):
        with generation.run_precision(numerics), (contextlib.nullcontext() if interceptor is None
                                                  else nn.intercept_methods(interceptor)):
            logits = model.apply(params, tokens[:, :-1])
        logits = logits.astype(jnp.promote_types(logits.dtype, jnp.float32))
        picked = jnp.take_along_axis(jax.nn.log_softmax(logits), tokens[:, 1:, None], axis=-1)
        return -jnp.mean(picked)
    return loss


def test_a_training_step_in_the_fp8_preset_is_on_par_with_a_fake_quantized_model():
    """One training step of a dense and MoE stack under the 'fp8' numerics (Dew's recipe, XLA's arm on a
    CPU), its loss and every parameter's gradient, on par with the same model under bf16-mixed numerics
    whose projections the recipe takes by role run the recipe emulated in float32 in test code
    (`fake_quantized`: the fake-quantized reference a recipe is measured against), from that reference in
    float64 (check 1): a projection given the wrong role, or a model-level gradient the recipe's wiring
    breaks, fails it."""
    model = mixed()
    plain = dataclasses.replace(FP8, low_precision=None)
    tokens = jnp.zeros((2, 65), jnp.int32)
    shapes = jax.eval_shape(model.init, jax.random.key(0), tokens[:, :-1])

    def draw(seed):
        rng = np.random.default_rng(seed)
        params = jax.tree.map(lambda leaf: jnp.asarray(rng.normal(size=leaf.shape) * 0.05, leaf.dtype),
                              shapes)
        drawn = jnp.asarray(rng.integers(0, VOCAB, (2, 65)), jnp.int32)
        return params, drawn, jnp.ones((), jnp.float32)

    def arm(model, numerics, interceptor=None):
        def run(params, tokens, cotangent):
            return with_gradients(lambda params: step(model, numerics, interceptor)(params, tokens))(params,
                                                                                                   cotangent)
        return run

    @as_given
    def truth(params, tokens, cotangent):
        with jax.enable_x64(new_val=True):
            return arm(float64_twin(model), plain, fake_quantized(jnp.float64))(
                widened(params), tokens, cotangent.astype(jnp.float64))

    assert_as_exact(jax.jit(arm(model, FP8)), jax.jit(arm(model, plain, fake_quantized(jnp.float32))), draw,
                    "fp8 preset step", seeds=SEEDS, truth_fn=truth)
