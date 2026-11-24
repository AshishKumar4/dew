"""The low-precision run numerics (`dew.nn.precision.FP8`, `MXFP8`, `NVFP4`, rows 2513 and 6658): a recipe
(DeepSeek-V3's blockwise FP8, MXFP8, NVFP4) on the projections whose role it takes, the rest in the run's
dtypes."""

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
from dew.nn.kernels import generation, low_rank, projections
from dew.nn.precision import (
    FP8,
    NUMERICS_PRESETS,
    PROJECTION_ROLES,
    LowPrecision,
    Numerics,
    RecipeRefused,
    numerics_of,
)

VOCAB, WIDTH = 256, 128
SEEDS = 64
"""The recipe's check-1 seeds, the projection's and the step's (the step's sized by `seeds_for` on armada)."""
RECIPE_NUMERICS = {numerics.low_precision.recipe: numerics for numerics in NUMERICS_PRESETS.values()
                   if numerics.low_precision is not None}
"""Each low-precision recipe's run numerics, as the presets name them (`precision.NUMERICS_PRESETS`)."""


def mixed(dtype=jnp.bfloat16) -> CausalTransformer:
    """Two layers 128 wide, each width a whole number of the recipe's 128-wide blocks: a dense one, then a
    mixture of four experts with a gated shared expert (Qwen2-MoE's), and an untied head."""
    return CausalTransformer(
        vocab_size=VOCAB, emb_features=WIDTH, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=64,
        mlp_features=2 * WIDTH, max_seq_len=64, dtype=dtype, tie_embeddings=False,
        mixture={"experts": 4, "top_k": 2, "expert_features": WIDTH, "layers": (1,),
                 "shared_features": WIDTH, "shared_gate": True})



def _float64_truth(reference):
    """A check's float64 truth: `reference(jnp.float64)` of the draw widened, as given (`as_given`)."""
    @as_given
    def truth(*drawn):
        with jax.enable_x64(new_val=True):
            return reference(jnp.float64)(*widened(drawn))
    return truth

def test_the_fp8_preset_names_the_recipe_and_keeps_the_runs_dtypes():
    """`--precision fp8` names DeepSeek-V3's recipe over bf16-mixed: the mixers', feed-forwards' and
    experts' projections in blockwise FP8, and a recipe naming a role no projection plays is refused."""
    assert numerics_of("fp8") == FP8 == Numerics("bf16-mixed", low_precision=LowPrecision())
    assert FP8.low_precision is not None and FP8.low_precision.roles == ("mixer", "mlp", "experts")
    with pytest.raises(ValueError):
        LowPrecision(roles=("mixer", "norm"))
    unknown = Numerics(low_precision=LowPrecision(recipe="fp6"))
    with pytest.raises(ValueError), generation.run_precision(unknown):
        pass


def test_a_models_projections_take_the_recipe_or_are_refused_for_their_role_alone():
    """Under the 'fp8' numerics a model's forward routes each projection by its role (the routes its trace
    reports, `generation.routes`): every projection the recipe considers is taken, or refused for its role
    (`kept_role`: the shared expert's gate's), never for its shape, one route each; the router, whose scores
    are an fp32 einsum, the untied head, whose product keeps its fp32 result, the norms and the embedding
    reach no product the recipe considers (Train 1223's review F3)."""
    model = mixed()
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (2, 64)), jnp.int32)
    params = jax.eval_shape(model.init, jax.random.key(0), tokens)
    with generation.run_precision(FP8), generation.routes() as found:
        jax.eval_shape(model.apply, params, tokens)
    recipe = [route for route in found if route.choice == "blockwise_fp8"]
    taken = [route for route in recipe if route.kernel != "unquantized"]
    refused = [route.refusal.code for route in recipe if route.kernel == "unquantized" and route.refusal]
    # Each layer's query, key, value and output; the dense layer's and the shared expert's gate, up and down;
    # the experts' three; the shared expert's gate the one its role keeps. No router's, no head's.
    assert len(taken) == model.num_layers * 4 + 3 + 3 + 3 and refused == ["kept_role"], recipe


@pytest.mark.parametrize("role", [*PROJECTION_ROLES, None])
def test_a_projection_takes_the_recipe_for_its_role_alone(role):
    """A 128-wide projection under the 'fp8' numerics in `role` (`projection_role`; None, a head's or an
    embedding's, plays none): where the recipe names its role it takes it, its output the recipe's blockwise
    FP8 product, which differs from the run's own; any other role's is refused for its role (`kept_role`),
    its output the run's own product exactly, in the run's dtype."""
    from dew.nn.dense import Dense
    from dew.nn.precision import projection_role

    layer = Dense(WIDTH, use_bias=False, dtype=jnp.bfloat16)
    x = jnp.asarray(np.random.default_rng(3).normal(size=(4, WIDTH)), jnp.bfloat16)
    params = layer.init(jax.random.key(0), x)
    played = contextlib.nullcontext() if role is None else projection_role(role)
    with generation.run_precision(FP8), played, generation.routes() as found:
        out = layer.apply(params, x)
    with generation.run_precision(Numerics("bf16-mixed")):
        own = layer.apply(params, x)
    [route] = [route for route in found if route.choice == "blockwise_fp8"]
    out, own = (np.asarray(value) for value in (out, own))
    assert out.dtype == own.dtype == jnp.bfloat16
    if role in (() if FP8.low_precision is None else FP8.low_precision.roles):
        assert route.kernel != "unquantized" and not np.array_equal(out, own)
    else:
        assert route.refusal is not None and route.refusal.code == "kept_role", route
        np.testing.assert_array_equal(out, own)


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
    block_codes, block_scales = projections.blocks_quantized(kernel)
    matrices = np.asarray(block_codes, np.float64) * np.repeat(np.repeat(
        np.asarray(block_scales, np.float64), WIDTH, axis=1), WIDTH, axis=2)
    for group in range(x.shape[0]):
        rows = x[group].reshape(-1, WIDTH)
        with generation.run_precision(FP8), projection_role("mixer"):
            alone = projections.recipe_projection(rows, kernel[group], out_dtype=jnp.float32)
        row_codes, row_scales = projections.rows_quantized(rows)
        dequantized = np.asarray(row_codes, np.float64) * np.repeat(np.asarray(row_scales, np.float64), WIDTH,
                                                                    axis=-1)
        assert_fp32_reduction_bound(np.asarray(out[group]).reshape(-1, WIDTH), alone,
                                    np.abs(dequantized) @ np.abs(matrices[group]), WIDTH)


def test_a_retained_jitted_projection_traces_again_under_another_role():
    """A projection jitted once and called under one role, then another, at the same shapes, traces again
    (`projection_role` is a context JAX keys its trace caches on): under the 'fp8' numerics an MLP's call
    takes the recipe and a router's, after it, keeps the run's dtypes, each the output a fresh trace under
    its role gives."""
    from dew.nn.dense import Dense
    from dew.nn.precision import projection_role

    layer = Dense(WIDTH, use_bias=False, dtype=jnp.float32)
    x = jnp.asarray(np.random.default_rng(2).normal(size=(4, WIDTH)), jnp.float32)
    params = layer.init(jax.random.key(0), x)
    retained = jax.jit(layer.apply)
    outputs = {}
    with generation.run_precision(FP8):
        for role in ("mlp", "router"):
            with projection_role(role):
                fresh = jax.jit(lambda params, x: layer.apply(params, x))
                outputs[role] = (np.asarray(retained(params, x)), np.asarray(fresh(params, x)))
    for role, (kept, own) in outputs.items():
        np.testing.assert_array_equal(kept, own, err_msg=role)
    assert not np.array_equal(outputs["mlp"][1], outputs["router"][1])


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
    """An adapter's update fused into its projection's product (`low_rank.low_rank`, LoRA's fused site under
    `generation.chosen(low_rank='fused')`, which returns before the layer's own product) on a projection
    the 'fp8' numerics take by role would train it unquantized, silently: refused, and a role the recipe
    keeps is not."""
    from dew.nn.precision import projection_role

    rows, rank = jnp.ones((4, WIDTH), jnp.float32), 4
    kernel, a, b = jnp.ones((WIDTH, WIDTH)), jnp.ones((WIDTH, rank)), jnp.zeros((rank, WIDTH))
    with generation.run_precision(FP8):
        with projection_role("mlp"), pytest.raises(RecipeRefused):
            low_rank.low_rank(rows, kernel, a, b, 1.0, out_dtype=jnp.float32)
        with projection_role("router"):
            low_rank.low_rank(rows, kernel, a, b, 1.0, out_dtype=jnp.float32)


def test_a_projection_the_recipe_leaves_runs_under_a_product_wrapper_as_without_the_recipe():
    """A projection in a role the 'fp8' numerics take whose widths hold no whole 128-wide blocks, which the
    recipe leaves (`projections.recipe_refusal`), is no recipe's to refuse: under a wrapper's product and
    as an adapter's fused product it runs as it does in a run of no recipe, bit for bit (row 6914)."""
    from dew import _cache
    from dew.nn.dense import Dense
    from dew.nn.precision import projection_role

    narrow = WIDTH - 32
    layer = Dense(narrow, dtype=jnp.float32)
    rng = np.random.default_rng(6914)
    x = jnp.asarray(rng.normal(size=(4, WIDTH)), jnp.float32)
    params = layer.init(jax.random.key(0), x)
    kernel, a, b = (jnp.asarray(rng.normal(size=shape), jnp.float32)
                    for shape in ((WIDTH, narrow), (WIDTH, 4), (4, narrow)))

    def ran():
        with _cache.INTERCEPTED(new_value=True):
            wrapped = layer.apply(params, x)
        return wrapped, low_rank.low_rank(x, kernel, a, b, 1.0, out_dtype=jnp.float32)

    with projection_role("mlp"):
        plain = ran()
        with generation.run_precision(FP8):
            recipes = ran()
    for name, without, under in zip(("wrapped", "adapted"), plain, recipes, strict=True):
        np.testing.assert_array_equal(np.asarray(under), np.asarray(without), err_msg=name)


@pytest.mark.parametrize(("roles", "width", "pairs"), [
    (("mixer", "mlp", "experts"), WIDTH - 32, True), (("mlp",), WIDTH, True),
    (("mixer", "mlp", "experts"), WIDTH, False)], ids=["no-whole-blocks", "a-role-it-keeps", "taken"])
def test_experts_the_recipe_leaves_take_their_gate_and_up_as_one_paired_product(roles, width, pairs):
    """A routed layer's gate and up projections the 'fp8' recipe leaves (`projections.recipe_refusal`), their
    widths of no whole 128-wide blocks or a recipe keeping the experts' role, take one paired product as in a
    run of no recipe (`moe.paired_projections`): the same routes, each product bit for bit, whatever role the
    caller traces in (a block's feed-forward's, `mlp`); experts the recipe takes run their products apart,
    each in the recipe (row 6914)."""
    from dew.nn.moe import paired_projections
    from dew.nn.precision import projection_role

    rng = np.random.default_rng(6914)
    x = jnp.asarray(rng.normal(size=(6, WIDTH)), jnp.float32)
    gate, up = (jnp.asarray(rng.normal(size=(2, WIDTH, width)), jnp.float32) for _ in range(2))
    sizes = jnp.asarray([4, 2], jnp.int32)
    recipe = dataclasses.replace(FP8, low_precision=LowPrecision(roles=roles))

    def paired():
        with generation.routes() as routes:
            products = paired_projections(x, gate, up, sizes, None, "auto", None)
        return products, [(found.choice, found.kernel) for found in routes]

    with projection_role("mlp"):
        plain, plain_routes = paired()
        with generation.run_precision(recipe):
            recipes, recipe_routes = paired()
    if not pairs:
        # Each product apart, in the recipe: its route a kernel of the recipe's choice, none unquantized.
        recipe_kernels = [kernel for choice, kernel in recipe_routes
                          if choice == projections.RECIPES["blockwise_fp8"].choice]
        assert len(recipe_kernels) == 2 and "unquantized" not in recipe_kernels, recipe_routes
        return
    assert recipe_routes == plain_routes
    for without, under in zip(plain, recipes, strict=True):
        np.testing.assert_array_equal(np.asarray(under), np.asarray(without))


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
    """`projections.blockwise_fp8_projection` emulated in `dtype`: the recipe's quantizers
    (`projections.rows_quantized`, `projections.blocks_quantized`), each group's product of the dequantized
    values in `dtype` at its highest precision, and its gradients the recipe's
    (`blockwise_matrix.recipe_truth`'s), each row's group read from `group_blocks` as the model's routing
    gives it: float64 the step's truth, float32 the fake-quantized reference a recipe is measured against."""
    def dequantized(codes, scales):
        return codes.astype(dtype) * jnp.repeat(scales.astype(dtype), 128, axis=-1)

    def blocks(w):
        codes, scales = projections.blocks_quantized(w)
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
        return grouped(dequantized(*projections.rows_quantized(x)), blocks(w),
                       owned(group_blocks, x.shape[0]))

    def forward(x, w, group_blocks):
        return run(x, w, group_blocks), (x, w, group_blocks)

    def backward(residuals, cotangent):
        x, w, group_blocks = residuals
        owners = owned(group_blocks, x.shape[0])
        d_x = grouped(dequantized(*projections.rows_quantized(cotangent)), jnp.swapaxes(blocks(w), 1, 2),
                      owners)
        rows_t = dequantized(*projections.rows_quantized(x.T))
        dy_t = dequantized(*projections.rows_quantized(cotangent.T))
        d_w = jnp.stack([jnp.matmul(rows_t * owners[g], dy_t.T, precision="highest")
                         for g in range(w.shape[0])])
        return d_x.astype(x.dtype), d_w.astype(w.dtype), None

    run.defvjp(forward, backward)

    def projection(x, w, group_blocks, *, out_dtype, interpret_on_cpu=False):
        del interpret_on_cpu
        return run(x, w, group_blocks).astype(out_dtype)

    return projection


def laid_projection(dtype, recipe: str = "blockwise_fp8"):
    """`projections.recipe_projection` of a projection's rows written apart in `dtype` under `recipe`:
    `run(rows, matrices, sizes)`, each group's rows (`sizes` a group's, in order, any rows past their sum
    outside every group) laid in whole 128-row blocks behind zero rows, a run of zero blocks after them, the
    recipe's step over them (the blockwise recipe emulated, `emulated`; a block-scaled one's,
    `mx_matrix.recipe_truth`, its operands' values quantized along each product's contraction, of concrete
    `sizes`), and each row's output read back, a row outside every group's zero."""
    from mx_matrix import mx_values, nvfp4_values, recipe_truth

    def project(laid, matrices, group_blocks):
        if recipe == "blockwise_fp8":
            return emulated(dtype)(laid, matrices, group_blocks, out_dtype=dtype)
        quantized = mx_values if recipe == "mxfp8" else nvfp4_values
        return recipe_truth(np.asarray(group_blocks), wide=dtype, quantized=quantized)(laid, matrices)

    def run(rows, matrices, sizes):
        count, groups = rows.shape[0], matrices.shape[0]
        sizes = jnp.asarray(sizes, jnp.int32) if recipe == "blockwise_fp8" else np.asarray(sizes)
        blocks = -(-sizes // 128)
        ends = jnp.cumsum(sizes)
        group = jnp.searchsorted(ends, jnp.arange(count), side="right")
        held = jnp.minimum(group, groups - 1)
        laid_rows = -(-(count + groups * 127) // 128) * 128
        place = jnp.where(group < groups, ((jnp.cumsum(blocks) - blocks) * 128)[held] + jnp.arange(count)
                          - (ends - sizes)[held], laid_rows)
        laid = jnp.zeros((laid_rows, rows.shape[1]), rows.dtype).at[place].set(rows, mode="drop")
        slack = laid_rows // 128 - blocks.sum()
        group_blocks = (blocks.at[-1].add(slack) if recipe == "blockwise_fp8"
                        else np.concatenate([blocks[:-1], blocks[-1:] + slack]))
        out = project(laid, matrices, group_blocks)
        return out.at[place].get(mode="fill", fill_value=0)

    return run


@pytest.mark.parametrize("recipe", RECIPE_NUMERICS)
def test_a_biased_layer_takes_the_recipes_product_and_rounds_once_after_its_bias(recipe):
    """A biased bf16 `Dense` whose bias is stored fp32 (one rounding of its product and bias, as torch's
    addmm), under each recipe's numerics in a role the recipe takes: its output on par with the recipe
    written apart (`laid_projection`) over its bf16 kernel, the bias added to its fp32 sums and rounded
    once, from it in float64 (check 1)."""
    from dew.nn.dense import Dense
    from dew.nn.precision import projection_role

    layer = Dense(128, dtype=jnp.bfloat16, use_bias=True)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(64, 128)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(128, 128)) / 8, jnp.float32),
                jnp.asarray(rng.normal(size=(128,)), jnp.float32))

    def biased(rows, kernel, bias):
        with generation.run_precision(RECIPE_NUMERICS[recipe]), projection_role("mlp"):
            return layer.apply({"params": {"kernel": kernel, "bias": bias}}, rows)

    def reference(dtype):
        recipes = laid_projection(dtype, recipe)

        def laid(rows, matrices):
            return recipes(rows, matrices, [64])

        def run(rows, kernel, bias):
            summed = laid(rows, kernel.astype(jnp.bfloat16)[None]) + bias.astype(dtype)
            return summed if dtype == jnp.float64 else summed.astype(jnp.bfloat16)
        return run

    assert_as_exact(jax.jit(biased), jax.jit(reference(jnp.float32)), draw, f"{recipe} biased layer",
                    seeds=SEEDS, truth_fn=_float64_truth(reference))


def test_an_mxfp8_biased_layer_rounds_its_inputs_to_their_blocks():
    """The biased layer's product is MXFP8's under the 'mxfp8' numerics, not XLA's: 128 inputs of 1.0625
    (rounded to 1 by their E4M3 codes under a power-of-two scale) against unit weights give 128, where
    XLA's bf16 product gives 136. Integers, every sum exact."""
    from dew.nn.dense import Dense
    from dew.nn.precision import projection_role

    rows = jnp.full((1, 128), 1.0625, jnp.bfloat16)
    params = {"params": {"kernel": jnp.ones((128, 128), jnp.float32), "bias": jnp.zeros((128,), jnp.float32)}}
    with generation.run_precision(RECIPE_NUMERICS["mxfp8"]), projection_role("mlp"):
        out = jax.jit(Dense(128, dtype=jnp.bfloat16, use_bias=True).apply)(params, rows)
    np.testing.assert_array_equal(np.asarray(out, np.float32), 128.0)


@pytest.mark.parametrize("recipe", RECIPE_NUMERICS)
def test_joined_projections_under_a_recipe_are_each_its_own(recipe):
    """Gate and up projections a recipe takes (a `GatedMLP`'s join, `joined.joins`), under each recipe's
    numerics on a compile-only A100 that joins them where no recipe runs: each quantized as its own, as
    apart: a unit gate kernel and an up kernel of 2**24 against unit inputs give 128 in every gate output
    (joined, NVFP4's one tensor scale for both kernels rounded every gate code to zero)."""
    from dew.nn.dense import Dense
    from dew.nn.joined import joined_dense, joins
    from dew.nn.kernels.generation import Choice
    from dew.nn.precision import projection_role

    class Pair(nn.Module):
        @nn.compact
        def __call__(self, x):
            projections = [Dense(128, dtype=jnp.bfloat16, use_bias=False, name=name)
                           for name in ("gate", "up")]
            apart = jnp.stack([projection(x) for projection in projections], axis=-2)
            taken = (joined_dense(x, projections, axis=-2) if joins(projections, Choice.GATE_UP)
                     else apart)
            return apart, taken

    x = jnp.ones((128, 128), jnp.bfloat16)
    params = {"params": {"gate": {"kernel": jnp.ones((128, 128), jnp.float32)},
                         "up": {"kernel": jnp.full((128, 128), 2.0 ** 24, jnp.float32)}}}
    with (generation.assumed("sm80"), generation.chosen(gate_up="joined"),
          generation.run_precision(RECIPE_NUMERICS[recipe]), projection_role("mlp")):
        apart, taken = jax.jit(Pair().apply)(params, x)
    np.testing.assert_array_equal(np.asarray(taken, np.float32), np.asarray(apart, np.float32))
    np.testing.assert_array_equal(np.asarray(taken[:, 0], np.float32), 128.0)


def test_an_nvfp4_product_of_tiny_operands_is_normal_where_its_value_is():
    """A bf16 `Dense(128)` under the 'nvfp4' numerics of inputs and a kernel of 2**-65: each operand's tensor
    scale near 2**-76, whose product underflows float32, applied each in turn
    (`precision.twice_scaled`), so the output is the quantized operands' product in float64 rounded once
    (near 128 * 2**-130 = 2**-123, a normal float), where it was zero."""
    from mx_matrix import nvfp4_values

    from dew.nn.dense import Dense
    from dew.nn.precision import NVFP4, projection_role

    x = jnp.full((128, 128), 2.0 ** -65, jnp.bfloat16)
    params = {"params": {"kernel": jnp.full((128, 128), 2.0 ** -65, jnp.float32)}}
    with generation.run_precision(NVFP4), projection_role("mlp"):
        out = jax.jit(Dense(128, dtype=jnp.bfloat16, use_bias=False).apply)(params, x)
    with jax.enable_x64(new_val=True):
        rows, kernel = (nvfp4_values(jnp.asarray(operand, jnp.float64)) for operand in
                        (x, params["params"]["kernel"].astype(jnp.bfloat16).T))
        truth = np.asarray((rows @ kernel.T).astype(jnp.bfloat16), np.float32)
    assert truth.min() > 0, truth.min()
    np.testing.assert_array_equal(np.asarray(out, np.float32), truth)


def test_an_mxfp8_product_of_operands_whose_scales_are_subnormal_is_their_value():
    """A bf16 `Dense(32)` under the 'mxfp8' numerics of inputs of 2**-120 against weights of 2**120: the
    inputs' E8M0 scale is 2**-128, below float32's least normal, applied as its exponent
    (`formats.MxFp8.decoded`), so each product is 1 and the output 32, where the scale read as a float
    flushed to zero."""
    from dew.nn.dense import Dense
    from dew.nn.precision import MXFP8, projection_role

    x = jnp.full((128, 32), 2.0 ** -120, jnp.bfloat16)
    params = {"params": {"kernel": jnp.full((32, 32), 2.0 ** 120, jnp.float32)}}
    with generation.run_precision(MXFP8), projection_role("mlp"):
        out = jax.jit(Dense(32, dtype=jnp.bfloat16, use_bias=False).apply)(params, x)
    np.testing.assert_array_equal(np.asarray(out, np.float32), 32.0)


@pytest.mark.parametrize("program", ["jitted", "op by op"])
def test_an_mxfp8_operand_of_zero_blocks_and_least_scales_quantizes_to_its_values(program):
    """MXFP8's quantization (`formats.MxFp8.quantized`) of a block of zeros, a block whose largest magnitude
    takes the least scale (2^-125: the scale float32's subnormal 2^-127), and a block of ordinary values,
    jitted and op by op (XLA's CPU flushing subnormals): each decodes to its values, the zeros zero and the
    least-scaled block 2^-125 exactly, where dividing by the scale read it as zero, its zeros over it NaN
    (row 6931)."""
    from dew.nn.kernels.formats import MxFp8

    x = np.zeros((1, 96), np.float32)
    x[0, 32:64] = 2.0 ** -125
    x[0, 64:] = np.linspace(-448.0, 448.0, 32, dtype=np.float32)

    def decoded(x):
        return MxFp8.quantized(x).decoded()

    if program == "jitted":
        out = jax.jit(decoded)(x)
    else:
        with jax.disable_jit():
            out = decoded(x)
    expected = np.asarray(jax.jit(decoded)(x))
    np.testing.assert_array_equal(np.asarray(out)[0, :64], x[0, :64])
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_an_mxfp8_operand_decodes_in_a_narrow_dtype_at_its_own_range():
    """MXFP8's decode (`formats.MxFp8.decoded`) in float16 of a block whose scale is 2^16 (its largest value
    2^24): float16 values, each the float16 of its value, 2^16 and 2^24 past float16's largest infinite and
    1024 and 1536 exact, where the decode scaled in float32 (`precision.power2`) handed back float32 (the
    review of b676f0a14, F3)."""
    from dew.nn.kernels.formats import MxFp8

    x = np.zeros((1, 32), np.float32)
    x[0, :4] = (2.0 ** 24, 2.0 ** 16, 1024.0, 1536.0)
    out = jax.jit(lambda x: MxFp8.quantized(x).decoded(jnp.float16))(x)
    assert out.dtype == jnp.float16
    expected = np.zeros((1, 32), np.float16)
    expected[0, :4] = (np.inf, np.inf, 1024.0, 1536.0)
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_a_recipe_projection_on_a_data_parallel_mesh_lowers_for_cuda():
    """Row 6719: under the 'fp8' numerics, a feed-forward's projection and its gradients on two compile-only
    H100s whose automatic data axis splits the rows (a data-parallel run's mesh, `MeshSpec().build`), the
    recipe's kernel named, lower through the template: each device's rows its own launch, where a launch
    reading its operands whole refused the unmapped rows."""
    from dew.distributed.placement import MeshSpec
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.precision import projection_role

    numbers = (((2,), (0,)), ((), ()))
    mesh = MeshSpec().build(compile_only_gpus("sm90", 2))

    def stepped(rows, matrix):
        def loss(rows, matrix):
            return jnp.sum(projections.dot_general(jax.lax.dot_general)(rows, matrix, numbers))
        return jax.grad(loss, argnums=(0, 1))(rows, matrix)

    split = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    whole = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with (jax.set_mesh(mesh), generation.assumed("sm90"), generation.run_precision(FP8),
          projection_role("mlp"), generation.chosen(blockwise_fp8="mosaic")):
        jax.jit(stepped).trace(jax.ShapeDtypeStruct((8, 128, 256), jnp.bfloat16, sharding=split),
                               jax.ShapeDtypeStruct((256, 256), jnp.bfloat16, sharding=whole)
                               ).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "automatic"])
def test_a_recipe_projection_of_rows_a_data_axis_splits_is_each_devices_own(explicit):
    """Rows 4854 and 6719: under the 'fp8' numerics, a dense projection of rows a data axis splits over
    eight devices, Explicit or automatic (a data-parallel run's mesh, where a launch reading its operands
    whole refused the unmapped rows; `projections.recipe_projection`, through a layer's product too:
    `projections.dot_general`'s `recipe_general`) runs each device's rows apart
    (`projections._projected_apart`, `jax.shard_map`), each
    laid in whole 128-row blocks of its own: its value and the rows' gradient bitwise each device's rows'
    projection alone, the matrix's gradient their gradients' sum. Integers, so every quantization and sum is
    exact whatever adds them."""
    from dew.distributed.placement import MeshSpec
    from dew.nn.precision import projection_role

    rng = np.random.default_rng(29)
    rows = jnp.asarray(rng.integers(-3, 4, size=(8, 128, 256)), jnp.float32)
    matrix = jnp.asarray(rng.integers(-3, 4, size=(256, 256)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-3, 4, size=(8, 128, 256)), jnp.float32)
    numbers = (((2,), (0,)), ((), ()))

    def pulled(rows, matrix, cotangent):
        with generation.run_precision(FP8), projection_role("mlp"):
            value, pull = jax.vjp(lambda r, m: projections.dot_general(jax.lax.dot_general)(r, m, numbers),
                                  rows, matrix)
            return (value, *pull(cotangent))

    alone = [jax.jit(pulled)(rows[i:i + 1], matrix, cotangent[i:i + 1]) for i in range(8)]
    mesh = (MeshSpec(explicit=("data",)) if explicit else MeshSpec()).build(jax.devices()[:8])
    with jax.set_mesh(mesh):
        split = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None, None))
        value, d_rows, d_matrix = jax.jit(pulled)(jax.device_put(rows, split), matrix,
                                                  jax.device_put(cotangent, split))
        # An Explicit axis's split is the value's type; an automatic one's, the partitioner's placement.
        assert (jax.typeof(value) if explicit else value).sharding.spec[0] == "data"
    each = [[np.asarray(leaf) for leaf in leaves] for leaves in alone]
    np.testing.assert_array_equal(np.asarray(value), np.concatenate([leaves[0] for leaves in each]))
    np.testing.assert_array_equal(np.asarray(d_rows), np.concatenate([leaves[1] for leaves in each]))
    np.testing.assert_array_equal(np.asarray(d_matrix), np.sum([leaves[2] for leaves in each], axis=0))


def test_a_recipes_experts_of_rows_an_explicit_axis_splits_are_each_devices_share_of_each_group():
    """Row 4859: under the 'fp8' numerics, experts' projection (`projections.recipe_projection` with group
    sizes counting every device's rows: ragged, one empty, rows past them) of rows an Explicit data axis
    splits over eight devices runs each device's rows with its own share of each group
    (`projections._local_sizes`): its value and the rows' gradient bitwise each device's rows projected alone
    with those counts, the matrices' gradient their gradients' sum. Integers, every quantization and sum
    exact."""
    from dew.distributed.placement import MeshSpec
    from dew.nn.precision import projection_role

    rng = np.random.default_rng(31)
    sizes = np.asarray([300, 0, 450, 150])
    rows = jnp.asarray(rng.integers(-3, 4, size=(1024, 256)), jnp.float32)
    matrices = jnp.asarray(rng.integers(-3, 4, size=(4, 256, 256)), jnp.float32)
    cotangent = jnp.asarray(rng.integers(-3, 4, size=(1024, 256)), jnp.float32)

    def pulled(rows, matrices, cotangent, sizes):
        with generation.run_precision(FP8), projection_role("experts"):
            def projected(r, m):
                return projections.recipe_projection(r, m, sizes, out_dtype=jnp.float32)

            value, pull = jax.vjp(projected, rows, matrices)
            return (value, *pull(cotangent))

    ends, starts = np.cumsum(sizes), np.cumsum(sizes) - sizes
    alone = []
    for device in range(8):
        lo, hi = device * 128, (device + 1) * 128
        local = np.clip(ends, lo, hi) - np.clip(starts, lo, hi)
        alone.append([np.asarray(leaf) for leaf in jax.jit(pulled)(
            rows[lo:hi], matrices, cotangent[lo:hi], jnp.asarray(local, jnp.int32))])
    mesh = MeshSpec(explicit=("data",)).build(jax.devices()[:8])
    with jax.set_mesh(mesh):
        split = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
        value, d_rows, d_matrices = jax.jit(pulled)(jax.device_put(rows, split), matrices,
                                                    jax.device_put(cotangent, split),
                                                    jnp.asarray(sizes, jnp.int32))
    np.testing.assert_array_equal(np.asarray(value), np.concatenate([leaves[0] for leaves in alone]))
    np.testing.assert_array_equal(np.asarray(d_rows), np.concatenate([leaves[1] for leaves in alone]))
    np.testing.assert_array_equal(np.asarray(d_matrices), np.sum([leaves[2] for leaves in alone], axis=0))


@pytest.mark.parametrize("recipe", RECIPE_NUMERICS)
@pytest.mark.parametrize("role", ["mlp", "experts"])
def test_the_recipes_projection_is_on_par_with_its_fake_quantized_reference(role, recipe):
    """A projection each low-precision recipe takes (`projections.recipe_projection`, which a layer of the
    role hands its rows, under the 'fp8', 'mxfp8' or 'nvfp4' numerics): a dense one's, every row its one
    group's, and experts' of ragged groups, one empty, with rows past them; its output and both gradients on
    par with the recipe written apart in float32 over the same groups' whole 128-row blocks
    (`laid_projection`: the fake-quantized reference a recipe is measured against), from it in float64
    (check 1)."""
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
        with generation.run_precision(RECIPE_NUMERICS[recipe]), projection_role(role):
            sized = None if role == "mlp" else jnp.asarray(sizes, jnp.int32)
            return projections.recipe_projection(rows, matrix, sized, out_dtype=jnp.float32)

    def reference(dtype):
        recipes = laid_projection(dtype, recipe)

        def laid(rows, matrices):
            return recipes(rows, matrices, sizes)

        def run(rows, matrix):
            return laid(rows, matrix if matrix.ndim == 3 else matrix[None])
        return with_gradients(run)

    assert_as_exact(jax.jit(with_gradients(projected)), jax.jit(reference(jnp.float32)), draw,
                    f"{recipe} recipe projection {role}", seeds=SEEDS, truth_fn=_float64_truth(reference))


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

    # Compiled whole as the arms are, once for every draw.
    wide = jax.jit(arm(float64_twin(model), plain, fake_quantized(jnp.float64)))

    @as_given
    def truth(params, tokens, cotangent):
        with jax.enable_x64(new_val=True):
            return wide(widened(params), tokens, cotangent.astype(jnp.float64))

    assert_as_exact(jax.jit(arm(model, FP8)), jax.jit(arm(model, plain, fake_quantized(jnp.float32))), draw,
                    "fp8 preset step", seeds=SEEDS, truth_fn=truth)


def test_the_blockwise_recipes_scales_meet_its_products_in_range():
    """A bf16 layer in the blockwise recipe whose inputs scale near float32's largest (2^120: a row scale of
    2^112) and whose weights scale near its least (2^-10: a block scale of 2^-18) projects to their product,
    2^117 for 128 of each, finite: each 128-deep partial meets its row's and its block's scales in the order
    that keeps the intermediate normal (`precision.twice_scaled`, train 925 F1: left to right it overflowed
    to inf)."""
    from dew.nn.dense import Dense
    from dew.nn.precision import projection_role

    layer = Dense(128, dtype=jnp.bfloat16, use_bias=False)
    params = {"params": {"kernel": jnp.full((128, 128), 2.0 ** -10, jnp.float32)}}
    x = jnp.full((4, 128), 2.0 ** 120, jnp.bfloat16)
    with generation.run_precision(RECIPE_NUMERICS["blockwise_fp8"]), projection_role("mlp"):
        out = np.asarray(jax.jit(layer.apply)(params, x), np.float32)
    np.testing.assert_array_equal(out, np.full((4, 128), 2.0 ** 117, np.float32))


def _delayed_nvfp4():
    from dew.nn.precision import DTYPE_PRESETS, LowPrecision, Numerics

    return Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision(recipe="nvfp4",
                                                                             tensor_scaling="delayed"))


def test_a_delayed_nvfp4_dense_reads_its_history_and_records_its_rows():
    """A Dense under a delayed-scaling NVFP4 recipe that takes it (row 12721) holds its amax history
    (`dense.NVFP4_HISTORY`, zeros of the recipe's length where initialized), quantizes its rows at the tensor
    scale of the history's largest, its output bitwise the recipe projection's at that scale, and records its
    rows' largest magnitude under its history's name (`dense.NVFP4_AMAX`); it refuses to run without a
    history, and a current-scaling recipe holds none."""
    from dew.nn.dense import NVFP4_AMAX, NVFP4_HISTORY, Dense
    from dew.nn.kernels import projections
    from dew.nn.precision import NVFP4, projection_role

    rng = np.random.default_rng(12721)
    x = jnp.asarray(rng.normal(size=(100, 128)), jnp.bfloat16)
    layer = Dense(64, use_bias=False, dtype=jnp.bfloat16)
    with generation.run_precision(_delayed_nvfp4()), projection_role("mlp"):
        variables = layer.init(jax.random.key(0), x)
        assert np.asarray(variables[NVFP4_HISTORY]["amax_history"]).tolist() == [0.0] * 16
        history = jnp.asarray([3.0, 7.5] + [0.0] * 14, jnp.float32)
        # The state a step holds: its parameters and histories, no sown collection (an init sows too).
        held = {"params": variables["params"], NVFP4_HISTORY: {"amax_history": history}}
        out, sown = layer.apply(held, x, mutable=[NVFP4_AMAX])
        # The layer's kernel as it multiplies it, promoted to its dtype.
        kernel = jnp.asarray(variables["params"]["kernel"], jnp.bfloat16)
        want = projections.recipe_projection(x, kernel, out_dtype=jnp.bfloat16,
                                             tensor_scale=projections.delayed_tensor_scale(history))
        with pytest.raises(ValueError):
            layer.apply({"params": variables["params"]}, x)
    np.testing.assert_array_equal(np.asarray(out).view(np.uint16), np.asarray(want).view(np.uint16))
    assert np.asarray(sown[NVFP4_AMAX]["amax_history"]).tolist() == [
        float(projections.nvfp4_amax(x))]
    with generation.run_precision(NVFP4), projection_role("mlp"):
        assert NVFP4_HISTORY not in layer.init(jax.random.key(0), x)


def test_a_rolled_history_takes_its_steps_largest_newest():
    """A training step's update of a delayed-scaling NVFP4 run's histories (`dense.rolled`, row 12721):
    each history a step recorded shifted by one along its steps, the step's largest newest and its oldest
    dropped, a scanned stack's `[layers, steps]` each layer's own; a history no call recorded as it was."""
    from dew.nn.dense import rolled

    histories = {"a": {"amax_history": jnp.asarray([1.0, 2.0, 3.0])},
                 "b": {"amax_history": jnp.asarray([4.0, 5.0, 6.0])},
                 "stack": {"up_proj": jnp.asarray([[1.0, 2.0], [3.0, 4.0]])}}
    out = rolled(histories, {"a": {"amax_history": jnp.float32(9.0)},
                             "stack": {"up_proj": jnp.asarray([7.0, 8.0])}})
    assert out["a"]["amax_history"].tolist() == [9.0, 1.0, 2.0]
    assert out["b"]["amax_history"].tolist() == [4.0, 5.0, 6.0]
    assert out["stack"]["up_proj"].tolist() == [[7.0, 1.0], [8.0, 3.0]]


def test_a_delayed_nvfp4_expert_layer_quantizes_each_projection_at_its_own_scale():
    """An expert layer under a delayed-scaling NVFP4 recipe that takes its experts (row 12721) runs its
    gate, up and down projections apart, each quantized at its own history's tensor scale
    (`moe.gated_experts`'s `scales`), and records its down projection's rows, their gated product: its
    output bitwise each projection by `moe.expert_projection` at that scale."""
    from dew.nn import moe
    from dew.nn.kernels import projections

    rng = np.random.default_rng(12722)
    tokens, depth, width = 136, 128, 256
    rows = jnp.asarray(rng.normal(size=(tokens, depth)), jnp.bfloat16)
    gate, up = (jnp.asarray(rng.normal(size=(2, depth, width)) / 8, jnp.bfloat16) for _ in range(2))
    down = jnp.asarray(rng.normal(size=(2, width, depth)) / 8, jnp.bfloat16)
    sizes = jnp.asarray([70, 66], jnp.int32)
    scales = tuple(projections.delayed_tensor_scale(jnp.asarray([value], jnp.float32))
                   for value in (3.0, 5.0, 7.0))
    noted = []
    with generation.run_precision(_delayed_nvfp4()):
        out = moe.gated_experts(rows, gate, up, down, sizes, jnp.bfloat16, "auto", None, "swiglu", None,
                                scales=scales, noted=noted.append)
        gate_rows, up_rows = (moe.expert_projection(rows, kernel, sizes, jnp.bfloat16, "auto", None,
                                                    tensor_scale=scale)
                              for kernel, scale in zip((gate, up), scales[:2], strict=True))
        hidden = moe.gated_product("swiglu")(*moe.swiglu_clamped(gate_rows, up_rows, None))
        want = moe.expert_projection(hidden, down, sizes, jnp.bfloat16, "auto", None, tensor_scale=scales[2])
    np.testing.assert_array_equal(np.asarray(out).view(np.uint16), np.asarray(want).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(noted[0]).view(np.uint16), np.asarray(hidden).view(np.uint16))
