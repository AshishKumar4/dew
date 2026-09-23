"""The run's precision policy: one dtype knob, one attention knob.

`--model.dtype` and `--model.attention-impl` are the only way in;
`with_precision` writes them into the model config that gets built and logged,
`Registry.build` resolves the names back into dtypes, and the attention kernel
raises a ValueError for a knob a fused kernel cannot honor.
"""

import jax
import jax.numpy as jnp
import pytest

from dew import models
from dew.diffusion.process import DenoisingCondition
from dew.nn.attention import local_attention, scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.dit import TextContext
from dew.nn.dsa_kpool import KPoolSparseAttention
from dew.nn.inputs import AttentionMetadata
from dew.nn.llama4 import Llama4Attention
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.registry import dtype_name, resolve_dtype, with_precision

BF16_QKV = (1, 4, 2, 8)  # [B, S, H, D]


def qkv(dtype=jnp.bfloat16):
    return (jnp.ones(BF16_QKV, dtype),) * 3


@pytest.mark.parametrize("implementation", ['xla', 'cudnn', 'tpu'])
@pytest.mark.parametrize("precision", [jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST,
                                       'high', ('highest', 'highest')])
def test_fused_attention_rejects_precision_it_cannot_honor(implementation, precision,
                                                           without_deterministic_ops):
    """jax.nn.dot_product_attention takes no precision argument at all: it
    accumulates the logits in fp32 whatever it is handed, so asking a fused
    kernel by name for HIGH raises."""
    with pytest.raises(ValueError, match="precision"):
        scaled_dot_product_attention(*qkv(), precision=precision,
                                     implementation=implementation)


@pytest.mark.parametrize("implementation", ['xla', 'cudnn', 'tpu'])
def test_fused_attention_rejects_bf16_softmax(implementation, without_deterministic_ops):
    with pytest.raises(ValueError, match="force_fp32_for_softmax"):
        scaled_dot_product_attention(*qkv(), force_fp32_for_softmax=False,
                                     implementation=implementation)


@pytest.mark.parametrize("arguments", [
    {"precision": jax.lax.Precision.HIGHEST},
    {"force_fp32_for_softmax": False},
    {"dtype": jnp.float32},
])
def test_auto_takes_the_reference_path_for_arithmetic_only_it_performs(arguments):
    """'auto' resolves a call that asks for a matmul precision, a softmax
    dtype or a compute dtype no fused kernel honours to the reference path,
    and computes exactly what naming that path computes."""
    query, key, value = jax.random.normal(jax.random.key(1), (3, *BF16_QKV), jnp.bfloat16)
    auto = scaled_dot_product_attention(query, key, value, implementation="auto", **arguments)
    reference = scaled_dot_product_attention(query, key, value, implementation="reference",
                                             **arguments)
    assert jnp.array_equal(auto, reference)


PACKED = jnp.asarray([[1] * 6 + [2] * 7 + [0] * 3, [1] * 16])
LAYERS = {
    "window": lambda **policy: CausalTransformer(
        vocab_size=64, num_layers=2, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=32, layer_types=("sliding", "full_attention"),
        kinds={"sliding": {"window": 4}}, **policy),
    "packed": lambda **policy: CausalTransformer(
        vocab_size=64, num_layers=1, emb_features=32, num_heads=4, num_kv_heads=2,
        max_seq_len=32, **policy),
    "llama4_chunk": lambda **policy: Llama4Attention(
        emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32,
        attention_chunk_size=4, **policy),
    "llama4_global": lambda **policy: Llama4Attention(
        emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32,
        use_rope=False, **policy),
    "mla": lambda **policy: MultiHeadLatentAttention(
        emb_features=32, num_heads=4, max_seq_len=32, q_lora_rank=16, kv_lora_rank=8,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, **policy),
    "kpool": lambda **policy: KPoolSparseAttention(
        emb_features=32, num_heads=4, max_seq_len=32, q_lora_rank=8, kv_lora_rank=8,
        qk_nope_head_dim=8, v_head_dim=8, index_n_heads=2, index_head_dim=8, index_topk=4,
        index_kpool=2, **policy),
}


@pytest.mark.parametrize("arguments", [
    {"precision": "highest"},
    {"force_fp32_for_softmax": False},
])
@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_auto_under_a_window_or_packing_takes_the_reference_path(layer, arguments):
    """A window, a chunk, packed documents or a sparse selection build a
    mask, which sends 'auto' to xla rather than cuDNN; the call still
    resolves to the reference path first when only that path computes what
    it asks for, and matches naming that path."""
    if layer == "mla" and "precision" not in arguments:
        pytest.skip("latent attention always runs its softmax in fp32")
    if layer in ("window", "packed"):
        x = jax.random.randint(jax.random.key(0), (2, 16), 0, 64)
    else:
        x = jax.random.normal(jax.random.key(0), (2, 16, 32))
    call = {} if layer == "window" else {"segment_ids": PACKED}
    if layer == "kpool":
        call = {"attention_metadata": AttentionMetadata(valid=PACKED != 0)}
    variables = LAYERS[layer](**arguments).init(jax.random.key(1), x, **call)
    auto = LAYERS[layer](attention_impl="auto", **arguments).apply(variables, x, **call)
    reference = LAYERS[layer](attention_impl="reference", **arguments).apply(variables, x, **call)
    assert jnp.array_equal(auto, reference)


@pytest.mark.parametrize("span", [{"window": 4}, {"chunk": 4}])
def test_local_attention_resolves_auto_before_the_mask(span):
    query, key, value = jax.random.normal(jax.random.key(1), (3, 2, 64, 4, 8))
    packed = {"segment_ids": jnp.ones((2, 64), jnp.int32), "precision": "highest"}
    auto = local_attention(query, key, value, **span, **packed, implementation="auto")
    reference = local_attention(query, key, value, **span, **packed, implementation="reference")
    assert jnp.array_equal(auto, reference)


@pytest.mark.parametrize("precision", [None, jax.lax.Precision.DEFAULT, "default"])
def test_fused_attention_default_precision_matches_the_attention_equation(precision):
    query, key, value = jax.random.normal(
        jax.random.key(0), (3, *BF16_QKV), dtype=jnp.float32)
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key) / jnp.sqrt(query.shape[-1])
    expected = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1), value)
    actual = scaled_dot_product_attention(
        query, key, value, precision=precision, implementation="xla")
    assert jnp.max(jnp.abs(actual - expected)) < 1e-5


def test_cudnn_rejects_float32_inputs(without_deterministic_ops):
    """cuDNN's fused kernel has no fp32 path; casting behind the caller's back
    would make --model.dtype float32 a lie."""
    with pytest.raises(ValueError, match="bfloat16"):
        scaled_dot_product_attention(*qkv(jnp.float32), implementation='cudnn')


def test_cudnn_rejects_a_head_dimension_it_cannot_honor(without_deterministic_ops):
    narrow = (jnp.ones((1, 4, 2, 4), jnp.bfloat16),) * 3
    with pytest.raises(ValueError, match="multiple of 8"):
        scaled_dot_product_attention(*narrow, implementation="cudnn")


@pytest.mark.parametrize("key,value,flag", [("dtype", "bfloat16", "--model.dtype"),
                                           ("attention_impl", "xla", "--model.attention-impl")])
def test_policy_rejects_a_second_path_for_the_same_knob(key, value, flag):
    """The refusal names the flag that owns the knob the config carried."""
    with pytest.raises(ValueError, match=flag):
        with_precision('simple_dit', {key: value}, dtype="bfloat16",
                       attention_impl="auto")


def test_logged_policy_values_round_trip_through_the_registry():
    """The policy writes strings so a logged config stays a record; the
    registry maps them back on the way in."""
    assert resolve_dtype("bfloat16") is jnp.bfloat16
    assert dtype_name(jnp.bfloat16) == "bfloat16"
    with pytest.raises(ValueError, match="not one of"):
        resolve_dtype("float8")


TINY = {"emb_features": 32, "precision": "default"}
DIT = {"output_channels": 3, "patch_size": 4, "num_layers": 1, "num_heads": 2}
UNET = {"output_channels": 3, "feature_depths": [8, 16],
        "attention_configs": [None, {"heads": 2}],
        "num_res_blocks": 1, "num_middle_res_blocks": 1, "norm_groups": 4}
LM = {"num_layers": 1, "num_heads": 2, "vocab_size": 32, "max_seq_len": 16}
PER_ARCH = {
    "unet": UNET,
    "unet_2d_condition": {"stages": [{"features": 32, "heads": 2}, {"features": 64, "heads": 4}],
                           "blocks_per_level": 1, "precision": "default"},
    "unet_3d": {**UNET, "temporal_heads": 2},
    "uvit": {**DIT, "num_layers": 2},
    "simple_udit": {**DIT, "num_layers": 2},
    "simple_dit": DIT,
    "simple_mmdit": DIT,
    "hybrid_dit": DIT,
    "video_dit": DIT,
    "hierarchical_mmdit": {"output_channels": 3, "emb_features": (16, 32),
                           "num_layers": (1, 1), "num_heads": (2, 2), "base_patch_size": 2},
    "jepa_encoder": {"patch_size": 4, "num_layers": 1, "num_heads": 2},
    "jepa_video_encoder": {"patch_size": 4, "num_layers": 1, "num_heads": 2},
    "jepa_predictor": {"num_layers": 1, "num_heads": 2, "grid": (4, 4), "predictor_features": 16},
    "causal_transformer": LM,
    # The two published transformer families, at one block each.
    "sd3_transformer": {"patch_size": 2, "in_channels": 4, "out_channels": 4, "num_layers": 1,
                        "heads": 2, "head_dim": 8, "joint_attention_dim": 12,
                        "caption_projection_dim": 16, "pooled_projection_dim": 10,
                        "sample_size": 8, "pos_embed_max_size": 8},
    "flux_transformer": {"patch_size": 1, "in_channels": 16, "out_channels": 16, "num_layers": 1,
                         "num_single_layers": 1, "heads": 2, "head_dim": 12,
                         "joint_attention_dim": 16, "pooled_projection_dim": 10,
                         "guidance_embeds": True, "axes_dims_rope": (4, 4, 4)},
}
COMPOSITES = ("diffusion_gemma", "multimodal_transformer")
RES, FRAMES = 16, 2


def build_model(architecture, dtype="bfloat16"):
    """Every registered architecture at a tiny size, the policy applied where
    it enters: leaf models through `with_precision`; the composites wrap a
    language model built that way, so the compute dtype reaches their trunk."""
    if architecture not in COMPOSITES:
        own = ("unet_2d_condition", "sd3_transformer", "flux_transformer")
        fields = PER_ARCH[architecture] if architecture in own else {**TINY, **PER_ARCH[architecture]}
        return models.build(architecture, **with_precision(
            architecture, fields, dtype=dtype, attention_impl="auto"))
    text = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**TINY, **LM, "mlp_features": 64}, dtype=dtype, attention_impl="auto"))
    if architecture == "diffusion_gemma":
        return DiffusionGemma(text, canvas_length=4)
    return MultimodalTransformer(
        text, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                           image_size=8, patch_size=4),
        GemmaProjector(vision_width=16, text_width=TINY["emb_features"],
                       patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1, dtype=resolve_dtype(dtype))


def tiny_inputs(architecture, rng):
    """What each architecture's __call__ takes, at the smallest useful size."""
    image = jax.random.normal(rng, (1, RES, RES, 3))
    video = jax.random.normal(rng, (1, FRAMES, RES, RES, 3))
    text = TextContext(jnp.ones((1, 7, 768)), jnp.ones((1, 7), bool))
    if architecture in ("unet_3d", "video_dit"):
        return video, jnp.ones((1,)), text
    if architecture == "unet_2d_condition":
        latents = jax.random.normal(rng, (1, RES, RES, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(text.hidden)}
    if architecture == "sd3_transformer":
        latents = jax.random.normal(rng, (1, 8, 8, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :12], jnp.ones((1, 10)))}
    if architecture == "flux_transformer":
        latents = jax.random.normal(rng, (1, 8, 8, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :16], jnp.ones((1, 10)), guidance=jnp.full((1,), 3.5))}
    if architecture == "jepa_encoder":
        return (image,)
    if architecture == "causal_transformer":
        return (jnp.zeros((1, 8), jnp.int32),)
    if architecture == "diffusion_gemma":
        return (jnp.zeros((1, 4), jnp.int32),)
    if architecture == "multimodal_transformer":
        tokens = jnp.array([[2, 1, 3, 4, 5, 6, 7, 2]], jnp.int32)
        return (tokens,), {"image_indices": jnp.where(tokens == 1, 0, -1),
                           "conditioning": {"pixel_values": jax.random.normal(rng, (1, 1, 3, 8, 8))}}
    if architecture == "jepa_video_encoder":
        return (video,)
    if architecture == "jepa_predictor":
        return (jax.random.normal(rng, (1, 8, 32)),
                jnp.arange(8)[None], jnp.arange(8, 12)[None])
    return image, jnp.ones((1,)), text


@pytest.mark.parametrize("architecture", sorted(models))
def test_default_policy_computes_in_bf16_and_keeps_params_fp32(architecture, rng):
    """bf16 is a compute dtype: every param leaf stays float32 so checkpoints
    and the optimizer state are unchanged. DiffusionGemma refines a canvas
    against an encoded prompt, so its forward is the encode-then-refine pair."""
    model = build_model(architecture)

    inputs = tiny_inputs(architecture, rng)
    args, kwargs = inputs if isinstance(inputs[0], tuple) else (inputs, {})
    variables = model.init(rng, *args, **kwargs)
    if architecture == "diffusion_gemma":
        prompt = jnp.zeros((1, 8), jnp.int32)
        cache = model.apply(variables, 1, method=model.init_cache, mutable=["cache"])[1]["cache"]
        cache = model.apply({**variables, "cache": cache}, prompt, method=model.encode,
                            mutable=["cache"])[1]["cache"]
        variables = {**variables, "cache": cache}
    demoted = {jax.tree_util.keystr(path): str(leaf.dtype)
               for path, leaf in jax.tree_util.tree_flatten_with_path(variables["params"])[0]
               if leaf.dtype != jnp.float32}
    assert not demoted

    out = model.apply(variables, *args, **kwargs)
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))
    if architecture == "unet_2d_condition":
        assert out.dtype == jnp.bfloat16


def test_a_rounded_operand_rounds_under_jit_on_this_device():
    """XLA's GPU default may delete a cast round trip under jit; the rounding
    has to survive it, on whatever device runs the suite."""
    from dew.nn.precision import rounded_operand
    x = jnp.asarray([1 + 2**-10, -3 - 2**-9, 2**-130], jnp.float32)
    rounded = jax.jit(lambda x: rounded_operand(x, jnp.bfloat16))(x)
    assert jnp.array_equal(rounded, x.astype(jnp.bfloat16).astype(jnp.float32))
    assert not jnp.array_equal(rounded, x)
