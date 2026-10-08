"""Models' own dtype and attention fields, and the arithmetic their kernels honor."""

import collections
import json
from importlib import import_module
from pathlib import Path

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import source_info_util
from reference_error import assert_fp32_reduction_bound
from test_architectures import CASES as ARCHITECTURE_CASES

from dew.diffusion.process import DenoisingCondition
from dew.interop.hf_decoders import translate_config
from dew.nn.attention import local_attention, scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.dit import TextContext
from dew.nn.dsa_kpool import KPoolSparseAttention
from dew.nn.inputs import AttentionMetadata
from dew.nn.llama4 import Llama4Mixer
from dew.nn.mixer_base import MixerContext
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.registry import dtype_name, models, resolve_dtype

import_module("dew.nn.multimodal")  # registers the fixture kind


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
    "llama4_chunk": lambda **policy: Llama4Mixer().build(MixerContext(
        emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32, attention_chunk=4,
        qk_norm=False))(**policy),
    "llama4_global": lambda **policy: Llama4Mixer(use_rope=False).build(MixerContext(
        emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=32, qk_norm=False))(**policy),
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
    mask, which a fused kernel takes; the call still resolves to the
    reference path first when only that path computes what it asks for,
    and matches naming that path."""
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


def test_logged_dtype_values_round_trip_through_the_registry():
    """Recorded dtype names are resolved at the model's build boundary."""
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
    "edm2_unet": {"output_channels": 3, "model_channels": 8, "channel_mult": [1, 2], "num_blocks": 1,
                  "attn_resolutions": [8], "channels_per_head": 8},
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
    "qwen_image_transformer": {"in_channels": 4, "out_channels": 4, "num_layers": 1, "heads": 2,
                               "head_dim": 12, "context_in_dim": 16, "axes_dims_rope": (4, 4, 4)},
    "flux2_transformer": {"in_channels": 16, "out_channels": 16, "num_layers": 1, "num_single_layers": 1,
                          "heads": 2, "head_dim": 16, "joint_attention_dim": 16,
                          "timestep_guidance_channels": 32, "axes_dims_rope": (4, 4, 4, 4)},
    "z_image_transformer": {"in_channels": 4, "dim": 32, "n_layers": 1, "n_refiner_layers": 1, "n_heads": 2,
                            "cap_feat_dim": 16, "axes_dims": (4, 6, 6), "axes_lens": (64, 16, 16)},
    "wan_transformer": {"num_attention_heads": 2, "attention_head_dim": 12, "in_channels": 4,
                        "out_channels": 4, "text_dim": 16, "freq_dim": 16, "ffn_dim": 32,
                        "num_layers": 1, "rope_max_seq_len": 16},
}
COMPOSITES = ("diffusion_gemma", "multimodal_transformer")
RES, FRAMES = 16, 2
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
FIXTURE_DECODERS = ("mamba2-tiny", "deepseek-v4-tiny", "glm5-next-tiny", "kimi-k3-tiny")
"""Tiny released configs whose mixers no toy case reaches: Mamba-2,
DeepSeek V4's compressed attention and hyper-connections, GLM-5 Next's
k-pool sparse attention and Kimi's delta attention, depth attention and
SiTU. DeepSeek V4.1's KV quantizers round through float32 bits on
purpose (`dew.nn.fake_quant`), so its float64 twin is
tools/deepseek_v41_numerics.py's, not this one."""


def _translated(name):
    """A fixture's decoder config, leaving compute settings to the caller."""
    config = translate_config(json.loads((FIXTURES / name / "config.json").read_text()))
    return {key: value for key, value in config.items() if key not in ("dtype", "attention_impl")}


DECODERS = ({case.name: case.config for case in ARCHITECTURE_CASES
             if case.architecture == "causal_transformer" and case.label}
            | {f"causal_transformer+{name}": _translated(name) for name in FIXTURE_DECODERS})
"""The decoder's mixers and mixtures past its default, at toy width
(tests/test_architectures.py): the MoE routers, Gemma 4's, GPT OSS's, MLA,
Llama 4's, gemma3n's and the gated delta net, by case name; and the
fixtures' (`FIXTURE_DECODERS`)."""


def build_model(architecture, dtype="bfloat16"):
    """Every registered architecture at a tiny size with an explicit compute dtype.

    Float64 references use an actual dtype (not a recorded run dtype name),
    under x64. Composite trunks receive the same dtype as leaf models.
    """
    def resolved(name, fields):
        return {**fields, "dtype": jnp.float64 if dtype == "float64" else dtype}

    if architecture in DECODERS:
        return models.build("causal_transformer", **resolved("causal_transformer", DECODERS[architecture]))
    if architecture not in COMPOSITES:
        own = (
            "unet_2d_condition",
            "sd3_transformer",
            "flux_transformer",
            "qwen_image_transformer",
            "edm2_unet",
            "flux2_transformer",
            "z_image_transformer",
            "wan_transformer",
        )
        fields = PER_ARCH[architecture] if architecture in own else {**TINY, **PER_ARCH[architecture]}
        return models.build(architecture, **resolved(architecture, fields))
    text = models.build("causal_transformer", **resolved(
        "causal_transformer", {**TINY, **LM, "mlp_features": 64}))
    if architecture == "diffusion_gemma":
        return DiffusionGemma(text, canvas_length=4)
    return MultimodalTransformer(
        text, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                           image_size=8, patch_size=4),
        GemmaProjector(text_width=TINY["emb_features"],
                       patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1,
        dtype=jnp.float64 if dtype == "float64" else resolve_dtype(dtype))


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
    if architecture == "flux2_transformer":
        latents = jax.random.normal(rng, (1, 4, 4, 16))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :16], guidance=jnp.full((1,), 3.5))}
    if architecture in ("z_image_transformer", "qwen_image_transformer"):
        latents = jax.random.normal(rng, (1, 4, 4, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(
            text.hidden[:, :, :16], mask=text.mask)}
    if architecture == "wan_transformer":
        # 32 patch tokens: the source's time embedder is fp32 on purpose, and
        # fewer tokens would leave it a toy-sized share of the matmuls.
        latents = jax.random.normal(rng, (1, FRAMES, 8, 8, 4))
        return (latents, jnp.ones((1,))), {"conditioning": DenoisingCondition(text.hidden[:, :, :16])}
    if architecture == "jepa_encoder":
        return (image,)
    if architecture == "causal_transformer" or architecture in DECODERS:
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


def forward(architecture, dtype, rng):
    """`architecture` at `dtype`, its variables and inputs, and its forward
    pass as a function of them: DiffusionGemma's encodes its prompt first.
    At float64 the parameters and the float inputs are float64 too."""
    model = build_model(architecture, dtype)
    inputs = tiny_inputs(architecture, rng)
    args, kwargs = inputs if isinstance(inputs[0], tuple) else (inputs, {})
    variables = model.init(rng, *args, **kwargs)
    if dtype == "float64":
        variables, args, kwargs = jax.tree.map(
            lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
            (variables, args, kwargs))

    def run(variables, args, kwargs):
        if architecture == "diffusion_gemma":
            cache = model.apply(variables, 1, method=model.init_cache, mutable=["cache"])[1]["cache"]
            cache = model.apply({**variables, "cache": cache}, jnp.zeros((1, 8), jnp.int32),
                                method=model.encode, mutable=["cache"])[1]["cache"]
            variables = {**variables, "cache": cache}
        return model.apply(variables, *args, **kwargs)

    return run, (variables, args, kwargs)


def indexing_only(jaxpr) -> set:
    """The variables of `jaxpr` whose every use ends in integers: floats
    that compute an index and nothing else. `jax.image.resize` computes its
    nearest source indices in float32 whatever the image's dtype (on
    purpose, jax b/206898375), exact below 2**24 and rounding none of the
    model's values."""
    uses = collections.defaultdict(list)
    for eqn in jaxpr.eqns:
        for var in eqn.invars:
            if isinstance(var, jax.extend.core.Var):
                uses[var].append(eqn)
    returned = {var for var in jaxpr.outvars if isinstance(var, jax.extend.core.Var)}
    only: set = set()
    for eqn in reversed(jaxpr.eqns):
        for var in eqn.outvars:
            if var not in returned and uses[var] and all(
                    jnp.issubdtype(out.aval.dtype, jnp.integer) or out in only
                    for use in uses[var] for out in use.outvars):
                only.add(var)
    return only


def made_in(jaxpr, dtype) -> list[str]:
    """Where `jaxpr`, and every jaxpr it holds, makes a value of `dtype` the
    model computes with (`indexing_only` values are not counted): Dew's
    source line nearest each, in the order they are made."""
    indexing = indexing_only(jaxpr)
    found = []
    for eqn in jaxpr.eqns:
        if any(getattr(getattr(var, "aval", None), "dtype", None) == dtype and var not in indexing
               for var in eqn.outvars):
            frames = [] if eqn.source_info.traceback is None else list(
                source_info_util.user_frames(eqn.source_info.traceback))
            ours = [frame for frame in frames if "/src/dew/" in frame.file_name] or frames
            found.append(f"{ours[0].file_name.split('/src/')[-1]}:{ours[0].start_line}" if ours else "?")
        for inner in jax.extend.core.jaxprs_in_params(eqn.params):
            found += made_in(inner, dtype)
    return found


@pytest.mark.parametrize("architecture", sorted(models) + sorted(DECODERS))
def test_a_float64_model_computes_nothing_in_float32(architecture, rng):
    """A float64 model, under x64, computes in float64 throughout: its norm
    statistics, softmax, rotary angles and heads, which compute in at least
    float32 (`dew.nn.precision.at_least_fp32`), are float64 there. Rounded
    to float32, they were roundings a float64 twin shared with the float32
    model it bounds, so the bound missed them: test_layer_stack's scanned
    gemma3n read 1.07 of its bound on a TPU and 1.1 on an RTX 4080 at full
    float32, where 1383 of its twin's values were float32. The twin runs on
    the CPU backend, which every lane keeps beside its accelerator, since a
    TPU has no float64."""
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        run, operands = forward(architecture, "float64", rng)
        made = made_in(jax.make_jaxpr(run)(*operands).jaxpr, jnp.float32)
    assert not made, f"{len(made)} float32 values, first at {sorted(set(made))[:8]}"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("architecture", sorted(models) + sorted(DECODERS))
def test_a_model_below_float64_computes_the_same_bits_whether_x64_is_on(architecture, dtype, rng):
    """x64 is one flag for the whole process, which a float64 reference
    turns on beside the model it checks. A float32 or bfloat16 model
    computes the same bits either way: nothing it computes widens past
    float32, whatever a Python float or an index widens to under x64.

    Both run on the CPU backend, which every lane keeps beside its
    accelerator. The claim is about the arithmetic the model asks for, and
    a TPU compiles the same arithmetic apart once x64 widens its indices to
    int64, which it emulates."""
    with jax.default_device(jax.devices("cpu")[0]):
        run, operands = forward(architecture, dtype, rng)
        outputs = []
        for x64 in (False, True):
            with jax.enable_x64(x64):
                outputs.append(jax.tree.map(np.asarray, jax.jit(run)(*operands)))
    for without, under in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]), strict=True):
        assert under.dtype == without.dtype
        np.testing.assert_array_equal(under, without)


@pytest.mark.parametrize("architecture", sorted(models))
def test_explicit_bf16_compute_keeps_params_fp32(architecture, rng):
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


@pytest.mark.parametrize("source,target", [(jnp.float16, jnp.bfloat16), (jnp.bfloat16, jnp.float16),
                                           (jnp.float32, jnp.float16)])
def test_a_rounded_operand_rounds_between_formats_of_one_width(source, target):
    """float16 and bfloat16 are both two bytes and each loses bits in the
    other, so the rounding is the cast round trip whatever the byte counts."""
    from dew.nn.precision import rounded_operand
    x = jnp.asarray([1 + 2**-9, 3.0e-5, 1000.25, 70000.0 if source != jnp.float16 else 1.0],
                    source)
    rounded = jax.jit(lambda x: rounded_operand(x, target))(x)
    assert rounded.dtype == source
    assert jnp.array_equal(rounded, x.astype(target).astype(source), equal_nan=True)
    assert not jnp.array_equal(rounded, x)


@pytest.mark.parametrize("helper", ["rounded_operand", "rounded_to"])
def test_a_rounding_survives_the_reduction_it_feeds(helper):
    """XLA:CPU's YNNPACK reduce fusion sums the unrounded values of a bare
    `astype` round trip, even with excess precision off (openxla/xla#49978);
    the helpers' optimization barrier keeps the rounding a sum and a mean
    of squares read, as a norm's statistics do."""
    from dew.nn import precision
    x = np.random.default_rng(0).normal(size=(8, 4096)).astype(np.float32) * 3
    rounded = x.astype(jnp.bfloat16).astype(np.float64)
    held = jax.jit(lambda v: getattr(precision, helper)(v, jnp.bfloat16).astype(jnp.float32))
    sums = np.asarray(jax.jit(lambda v: jnp.sum(held(v), axis=-1))(x), np.float64)
    squares = np.asarray(jax.jit(lambda v: jnp.mean(jnp.square(held(v)), axis=-1))(x), np.float64)
    assert_fp32_reduction_bound(sums, rounded.sum(-1), np.abs(rounded).sum(-1), x.shape[-1])
    mean_square = np.mean(rounded ** 2, -1)
    assert_fp32_reduction_bound(squares, mean_square, mean_square, x.shape[-1] + 2)
    # The bound is a worst case (4.8 on these sums), wider than what dropping
    # the rounding moves them (at most 0.96), so each sum must also be nearer
    # the rounded values' than the unrounded ones'.
    assert np.all(np.abs(sums - rounded.sum(-1)) < np.abs(sums - x.astype(np.float64).sum(-1)))


def test_a_value_already_in_the_dtype_keeps_its_rounding_under_jit():
    """A bf16 value computed in fp32 and cast down is held as bf16 where it is
    already the operand dtype: without the barrier, XLA's GPU excess precision
    fuses the fp32 producer into its consumer and the cast rounds nothing."""
    from dew.nn.precision import rounded_operand
    x = jnp.asarray([1 + 2**-10, -3 - 2**-9, 7 + 2**-7], jnp.float32)

    def held(x):
        narrow = (x * 1.0).astype(jnp.bfloat16)
        return rounded_operand(narrow, jnp.bfloat16).astype(jnp.float32) * 1.0

    assert jnp.array_equal(jax.jit(held)(x), x.astype(jnp.bfloat16).astype(jnp.float32))


def test_bf16_logits_round_as_rounded_to_and_are_held_as_bf16_under_cuda():
    """The head's logits take `rounded_to`'s values and cotangents, bitwise;
    under CUDA the program holds them as bf16 instead of rounding fp32 in a
    reduce-precision of its own, which a serving draw's readers made a full
    fp32 pass (docs/performance.md)."""
    from dew.nn.precision import bf16_logits, rounded_to
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(16, 512)).astype(np.float32) * 8)
    weights = jnp.asarray(rng.normal(size=(16, 512)).astype(np.float32))

    def rounded(v):
        return rounded_to(v, jnp.bfloat16)

    def loss(head):
        return lambda v: jnp.sum(jax.nn.log_softmax(head(v * 1.5)) * weights)

    held = jax.jit(bf16_logits)(x)
    assert jnp.array_equal(held, jax.jit(rounded)(x)) and not jnp.array_equal(held, x)
    assert jnp.array_equal(jax.jit(jax.grad(loss(bf16_logits)))(x), jax.jit(jax.grad(loss(rounded)))(x))
    if jax.default_backend() == "gpu":
        text = jax.jit(jax.value_and_grad(loss(bf16_logits))).lower(x).compile().as_text()
        assert "reduce-precision" not in text and "bf16[16,512]" in text
