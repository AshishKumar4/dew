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
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.dit import TextContext
from dew.registry import dtype_name, resolve_dtype, with_precision

BF16_QKV = (1, 4, 2, 8)  # [B, S, H, D]


def qkv(dtype=jnp.bfloat16):
    return (jnp.ones(BF16_QKV, dtype),) * 3


@pytest.mark.parametrize("implementation", ['auto', 'xla', 'cudnn', 'tpu'])
@pytest.mark.parametrize("precision", [jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST,
                                       'high', ('highest', 'highest')])
def test_fused_attention_rejects_precision_it_cannot_honor(implementation, precision):
    """jax.nn.dot_product_attention takes no precision argument at all: it
    accumulates the logits in fp32 whatever it is handed, so asking for HIGH
    raises."""
    with pytest.raises(ValueError, match="precision"):
        scaled_dot_product_attention(*qkv(), precision=precision,
                                     implementation=implementation)


@pytest.mark.parametrize("implementation", ['auto', 'xla', 'cudnn', 'tpu'])
def test_fused_attention_rejects_bf16_softmax(implementation):
    with pytest.raises(ValueError, match="force_fp32_for_softmax"):
        scaled_dot_product_attention(*qkv(), force_fp32_for_softmax=False,
                                     implementation=implementation)


@pytest.mark.parametrize("precision", [None, jax.lax.Precision.DEFAULT, "default"])
def test_fused_attention_default_precision_matches_the_attention_equation(precision):
    query, key, value = jax.random.normal(
        jax.random.key(0), (3, *BF16_QKV), dtype=jnp.float32)
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key) / jnp.sqrt(query.shape[-1])
    expected = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1), value)
    actual = scaled_dot_product_attention(
        query, key, value, precision=precision, implementation="xla")
    assert jnp.max(jnp.abs(actual - expected)) < 1e-5


def test_cudnn_rejects_float32_inputs():
    """cuDNN's fused kernel has no fp32 path; casting behind the caller's back
    would make --model.dtype float32 a lie."""
    with pytest.raises(ValueError, match="bfloat16"):
        scaled_dot_product_attention(*qkv(jnp.float32), implementation='cudnn')


def test_cudnn_rejects_a_head_dimension_it_cannot_honor():
    narrow = (jnp.ones((1, 4, 2, 4), jnp.bfloat16),) * 3
    with pytest.raises(ValueError, match="multiple of 8"):
        scaled_dot_product_attention(*narrow, implementation="cudnn")


@pytest.mark.parametrize("key,value", [("dtype", "bfloat16"), ("attention_impl", "xla")])
def test_policy_rejects_a_second_path_for_the_same_knob(key, value):
    with pytest.raises(ValueError, match="--model.dtype"):
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
}
RES, FRAMES = 16, 2


def tiny_inputs(architecture, rng):
    """What each architecture's __call__ takes, at the smallest useful size."""
    image = jax.random.normal(rng, (1, RES, RES, 3))
    video = jax.random.normal(rng, (1, FRAMES, RES, RES, 3))
    text = TextContext(jnp.ones((1, 7, 768)), jnp.ones((1, 7), bool))
    if architecture in ("unet_3d", "video_dit"):
        return video, jnp.ones((1,)), text
    if architecture == "jepa_encoder":
        return (image,)
    if architecture == "causal_transformer":
        return (jnp.zeros((1, 8), jnp.int32),)
    if architecture == "jepa_video_encoder":
        return (video,)
    if architecture == "jepa_predictor":
        return (jax.random.normal(rng, (1, 8, 32)),
                jnp.arange(8)[None], jnp.arange(8, 12)[None])
    return image, jnp.ones((1,)), text


@pytest.mark.parametrize("architecture", sorted(models))
def test_default_policy_computes_in_bf16_and_keeps_params_fp32(architecture, rng):
    """bf16 is a compute dtype: every param leaf stays float32 so checkpoints
    and the optimizer state are unchanged. The unets and the jepa models hand
    back bf16; the DiT family casts its final projection to fp32 on purpose."""
    fields = with_precision(
        architecture, {**TINY, **PER_ARCH[architecture]},
        dtype="bfloat16", attention_impl="auto")
    model = models.build(architecture, **fields)

    args = tiny_inputs(architecture, rng)
    params = model.init(rng, *args)
    demoted = {jax.tree_util.keystr(path): str(leaf.dtype)
               for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
               if leaf.dtype != jnp.float32}
    assert not demoted

    out = model.apply(params, *args)
    assert out.dtype in (jnp.bfloat16, jnp.float32)
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))
