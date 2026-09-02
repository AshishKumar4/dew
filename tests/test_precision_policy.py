"""The run's precision policy: one dtype knob, one attention knob.

`--model.dtype` and `--model.attention-impl` are the only way in;
`apply_precision_policy` writes them into the model config that gets built and
logged, and the attention kernel refuses the knobs a fused kernel cannot
honor instead of dropping them silently.
"""

import jax
import jax.numpy as jnp
import pytest

from dew.nn.attention import scaled_dot_product_attention
from dew.registry import (
    MODEL_REGISTRY, apply_precision_policy, build_model, map_config_strings,
)

BF16_QKV = (1, 4, 2, 8)  # [B, S, H, D]


@pytest.fixture
def implementations(monkeypatch):
    """Record what jax.nn.dot_product_attention is asked to dispatch to."""
    seen = []

    def spy(query, key, value, **kwargs):
        seen.append(kwargs.get("implementation"))
        return query

    monkeypatch.setattr(jax.nn, "dot_product_attention", spy)
    return seen


def qkv(dtype=jnp.bfloat16):
    return (jnp.ones(BF16_QKV, dtype),) * 3


def test_auto_resolves_to_xla_off_gpu(implementations):
    scaled_dot_product_attention(*qkv(), implementation='auto')
    assert implementations == ['xla']


def test_auto_resolves_to_cudnn_on_gpu(implementations, monkeypatch):
    """The resolution happens per trace, not at config time, so a config
    logged as 'auto' on this box still runs on the next one."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    scaled_dot_product_attention(*qkv(), implementation='auto')
    assert implementations == ['cudnn']


def test_auto_runs_the_kernel_it_resolved_to():
    q, k, v = qkv()
    assert jnp.array_equal(
        scaled_dot_product_attention(q, k, v, implementation='auto'),
        scaled_dot_product_attention(q, k, v, implementation='xla'))


@pytest.mark.parametrize("implementation", ['auto', 'xla', 'cudnn', 'tpu'])
@pytest.mark.parametrize("precision", [jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST,
                                       'high', ('highest', 'highest')])
def test_fused_attention_rejects_precision_it_cannot_honor(implementation, precision):
    """jax.nn.dot_product_attention takes no precision argument at all: it
    accumulates the logits in fp32 whatever it is handed. Asking for HIGH and
    getting something else silently is worse than an error."""
    with pytest.raises(ValueError, match="precision"):
        scaled_dot_product_attention(*qkv(), precision=precision,
                                     implementation=implementation)


@pytest.mark.parametrize("implementation", ['auto', 'xla', 'cudnn', 'tpu'])
def test_fused_attention_rejects_bf16_softmax(implementation):
    with pytest.raises(ValueError, match="force_fp32_for_softmax"):
        scaled_dot_product_attention(*qkv(), force_fp32_for_softmax=False,
                                     implementation=implementation)


@pytest.mark.parametrize("precision", [None, jax.lax.Precision.DEFAULT, 'default'])
def test_fused_attention_takes_default_precision(implementations, precision):
    scaled_dot_product_attention(*qkv(), precision=precision, implementation='xla')
    assert implementations == ['xla']


def test_reference_attention_keeps_honoring_both_knobs():
    """The reference path is the one that can honor them, so it must not have
    picked up the rejection."""
    q, k, v = qkv(jnp.float32)
    high = scaled_dot_product_attention(q, k, v, precision=jax.lax.Precision.HIGHEST,
                                        force_fp32_for_softmax=False)
    assert high.shape == q.shape


def test_cudnn_rejects_float32_inputs():
    """cuDNN's fused kernel has no fp32 path; casting behind the caller's back
    would make --model.dtype float32 a lie."""
    with pytest.raises(ValueError, match="bfloat16"):
        scaled_dot_product_attention(*qkv(jnp.float32), implementation='cudnn')


def test_auto_on_gpu_rejects_float32_inputs(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="bfloat16"):
        scaled_dot_product_attention(*qkv(jnp.float32), implementation='auto')


def test_policy_reaches_nested_unet_attention_configs():
    """The unet keeps its attention settings in nested dicts that do not
    inherit the model dtype, so the policy has to write into them too."""
    applied = apply_precision_policy(
        'unet', {"attention_configs": [None, {"heads": 8}], "precision": "default"},
        dtype="bfloat16", attention_impl="auto")

    assert applied["dtype"] == "bfloat16"
    assert applied["attention_impl"] == "auto"
    assert applied["precision"] == "default"
    assert applied["attention_configs"] == [
        None, {"heads": 8, "dtype": "bfloat16", "force_fp32_for_softmax": True}]

    model = build_model('unet', applied)
    assert model.attention_configs[1]["dtype"] is jnp.bfloat16
    assert model.attention_configs[1]["force_fp32_for_softmax"] is True


def test_policy_fills_in_the_stages_the_config_left_at_the_default():
    """A config that never mentions attention_configs still gets bf16
    attention: the unet's own default stages compute in fp32."""
    applied = apply_precision_policy('unet', {}, dtype="bfloat16",
                                     attention_impl="reference")
    assert [stage["dtype"] for stage in applied["attention_configs"]] == \
        ["bfloat16"] * len(MODEL_REGISTRY['unet'].attention_configs)


def test_policy_spells_the_reference_kernel_as_none():
    applied = apply_precision_policy('simple_dit', {}, dtype="float32",
                                     attention_impl="reference")
    assert applied == {"dtype": "float32", "attention_impl": None}
    assert build_model('simple_dit', applied).attention_impl is None


@pytest.mark.parametrize("key,value", [("dtype", "bfloat16"), ("attention_impl", "xla")])
def test_policy_rejects_a_second_path_for_the_same_knob(key, value):
    with pytest.raises(ValueError, match="--model.dtype"):
        apply_precision_policy('simple_dit', {key: value}, dtype="bfloat16",
                               attention_impl="auto")


def test_policy_survives_architecture_suffixes():
    applied = apply_precision_policy('simple_dit+hilbert', {}, dtype="bfloat16",
                                     attention_impl="auto")
    assert applied["dtype"] == "bfloat16"


def test_logged_policy_values_round_trip_through_the_registry():
    """The policy writes strings so the wandb config stays a record; the
    registry maps them back on the way in."""
    mapped = map_config_strings({"dtype": "bfloat16", "attention_impl": "auto"})
    assert mapped["dtype"] is jnp.bfloat16
    assert mapped["attention_impl"] == "auto"


TINY = {"emb_features": 32, "output_channels": 3, "patch_size": 4,
        "num_layers": 1, "num_heads": 2, "precision": "default"}
PER_ARCH = {
    "unet": {"feature_depths": [8, 16], "attention_configs": [None, {"heads": 2}],
             "num_res_blocks": 1, "num_middle_res_blocks": 1, "norm_groups": 4},
    "unet_3d": {"feature_depths": [8, 16], "attention_configs": [None, {"heads": 2}],
                "num_res_blocks": 1, "num_middle_res_blocks": 1, "norm_groups": 4,
                "temporal_heads": 2},
    "uvit": {"num_layers": 2},
    "simple_udit": {"num_layers": 2},
    "hierarchical_mmdit": {"emb_features": (16, 32), "num_layers": (1, 1),
                           "num_heads": (2, 2), "base_patch_size": 2},
    "jepa_predictor": {"grid": (4, 4), "predictor_features": 16},
    "causal_transformer": {"vocab_size": 32, "max_seq_len": 16},
}
RES, FRAMES = 16, 2


def tiny_inputs(architecture, rng):
    """What each architecture's __call__ takes, at the smallest useful size."""
    image = jax.random.normal(rng, (1, RES, RES, 3))
    video = jax.random.normal(rng, (1, FRAMES, RES, RES, 3))
    text = jnp.ones((1, 7, 768))
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


@pytest.mark.parametrize("architecture", sorted(MODEL_REGISTRY))
def test_default_policy_computes_in_bf16_and_keeps_params_fp32(architecture, rng):
    """bf16 is a compute dtype: every param leaf stays float32 so checkpoints
    and the optimizer state are unchanged. The unets and the jepa models hand
    back bf16; the DiT family casts its final projection to fp32 on purpose."""
    config = apply_precision_policy(
        architecture, {**TINY, **PER_ARCH.get(architecture, {})},
        dtype="bfloat16", attention_impl="auto")
    model = build_model(architecture, config)
    assert model.dtype is jnp.bfloat16
    assert model.attention_impl == 'auto'

    args = tiny_inputs(architecture, rng)
    params = model.init(rng, *args)
    demoted = {jax.tree_util.keystr(path): str(leaf.dtype)
               for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
               if leaf.dtype != jnp.float32}
    assert not demoted

    out = model.apply(params, *args)
    assert out.dtype in (jnp.bfloat16, jnp.float32)
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))
