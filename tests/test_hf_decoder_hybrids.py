"""Hugging Face decoders whose layers are not all attention: Gemma 4's per-layer
inputs and shared KV, Qwen3.5's and Qwen3-Next's gated delta nets, and Gemma
3n's AltUp copies, each against transformers' logits (see test_hf_decoders).
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import flat_tree
from test_hf_decoders import DEEPSEEK, fixture_config, fp32_decoder, scaled_difference

from dew.interop import PretrainedDecoder
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

# --------------------------------------------------------------------------
# Gemma 4 gaps: per-layer input embeddings and cross-layer KV sharing
# --------------------------------------------------------------------------

GEMMA4 = ("gemma4-ple", "gemma4-kvshare")


def gemma4_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


def test_gemma4_config_translates_field_by_field():
    """The new features are reachable from the config: a gemma4_text config
    with standard rope, uniform head dim, silu MLP and no logit cap
    translates without a caller setting anything by hand. The logit cap is
    read by no text path, so it maps to nothing."""
    config = translate_config(gemma4_config("gemma4-ple"))

    assert config["kv_shared_layers"] is None
    assert config["per_layer_input_dim"] == 8
    assert config["per_layer_input_vocab"] == 64
    assert config["v_norm"] and config["qk_norm"]
    assert config["attention_scale"] == 1.0
    assert config["sandwich_norms"] and config["embedding_scale"]
    assert config["mlp"] == "swiglu" and config["tie_embeddings"]
    assert config["rope_theta"] == 10000.0
    assert config["partial_rotary_factor"] is None
    assert config["kinds"] == {"sliding_attention": {"window": 32}}
    assert "attention_logit_cap" not in config
    assert config["layer_types"] == ("sliding_attention",) * 3 + ("full_attention",)
    assert config["head_dim"] == 8 and not config["scale_after_cast"]

    config = translate_config(gemma4_config("gemma4-kvshare"))
    assert config["kv_shared_layers"] == tuple(range(config["num_layers"] - 2, config["num_layers"]))
    assert config["per_layer_input_dim"] is None


def test_the_e2b_shaped_config_translates_every_gap():
    """The release shape: partial rotary, mixed head dims, a logit cap, the
    double-wide MLP, sharing and per-layer inputs all translate; the cap
    maps to nothing because the text path never reads it."""
    config = translate_config(gemma4_config("gemma4-e2b"))

    assert config["partial_rotary_factor"] == 0.25
    assert "attention_logit_cap" not in config
    assert config["use_double_wide_mlp"]
    assert config["kv_shared_layers"] == tuple(range(config["num_layers"] - 2, config["num_layers"]))
    assert config["per_layer_input_dim"] == 8
    assert config["v_norm"] and config["rope_theta"] == 1000000.0
    # The full layers' own head dim and the sliding kind's window and base
    assert config["head_dim"] == 16
    assert config["kinds"] == {"sliding_attention": {"window": 32, "rope_theta": 10000.0},
                               "full_attention": {"head_dim": 32}}


@pytest.mark.parametrize("field,value", [
    ("hidden_act", "relu"),
    ("use_bidirectional_attention", "all"),
])
def test_a_gemma4_field_with_no_counterpart_is_refused(field, value):
    """Every gemma4 knob Dew cannot express raises a ValueError naming the
    field, and no model is built."""
    config = gemma4_config("gemma4-ple")
    config[field] = value
    with pytest.raises(ValueError, match=field):
        translate_config(config)


def test_proportional_rotary_without_its_factor_is_refused():
    config = gemma4_config("gemma4-e2b")
    del config["rope_parameters"]["full_attention"]["partial_rotary_factor"]
    with pytest.raises(ValueError, match="partial_rotary_factor"):
        translate_config(config)


def test_partial_rotary_on_a_sliding_layer_is_refused():
    config = gemma4_config("gemma4-e2b")
    config["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 0.5
    with pytest.raises(ValueError, match="sliding_attention"):
        translate_config(config)


@pytest.mark.parametrize("model_type", ["gemma4", "gemma3", "gemma4_unified", "gemma3n"])
def test_a_multimodal_wrapper_config_is_refused_by_name(model_type):
    """What a user pointing at google/gemma-4-E2B hits. The repo's
    config.json is the wrapper, not the decoder: its model_type names the
    whole model, its decoder is under text_config, and its weights sit under
    model.language_model.* beside towers nothing here runs. Refusing names
    all three, where before the wrapper passed the family gate and died on a
    missing hidden_size."""
    wrapper = {"model_type": model_type,
               "text_config": gemma4_config("gemma4-e2b"),
               "vision_config": {"hidden_size": 8}}

    with pytest.raises(ValueError, match="multimodal wrapper") as raised:
        translate_config(wrapper)
    assert "text_config" in str(raised.value)
    assert "model.language_model" in str(raised.value)

    # The decoder underneath it still translates, as the message says.
    config = translate_config(wrapper["text_config"])
    assert config["kv_shared_layers"] == tuple(range(config["num_layers"] - 2, config["num_layers"]))


def test_a_wrapper_shaped_config_of_an_unknown_family_is_refused_as_one():
    """A config with a text_config and a model_type this has never heard of
    is the same shape of thing, so it gets the same answer, with no bare
    family list."""
    with pytest.raises(ValueError, match="multimodal wrapper"):
        translate_config({"model_type": "someone_elses_vlm",
                          "text_config": gemma4_config("gemma4-ple")})


@pytest.mark.parametrize("spelling", ["per_layer_config", "global_absent", "global_ungated"])
def test_the_full_layers_geometry_is_read_the_way_the_reference_reads_it(spelling):
    """The reference lets the full layers carry their own head dim and
    key/value head count (Gemma4TextAttention reads layer_config), which a
    transformers-written config spells per layer and a released one as
    global_head_dim and num_global_key_value_heads. Gemma4TextConfig builds
    the per-layer entries from the global pair only when the per_layer_config
    key is absent, and takes the count only under attention_k_eq_v; the
    expected geometry comes from the reference class, not a copy of the
    rule."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    config = gemma4_config("gemma4-e2b")
    if spelling == "per_layer_config":
        config["per_layer_config"] = {"1": {"head_dim": 32, "num_key_value_heads": 1}}
    else:
        del config["per_layer_config"]
        config.update(global_head_dim=32, num_global_key_value_heads=1,
                      attention_k_eq_v=spelling == "global_ungated")
    full = config["layer_types"].index("full_attention")
    reference = Gemma4TextConfig(**config).per_layer_config[full]

    translated = translate_config(config)
    kind = translated["kinds"]["full_attention"]
    assert kind["head_dim"] == reference.head_dim == 32
    assert kind.get("num_kv_heads", translated["num_kv_heads"]) == reference.num_key_value_heads
    assert ("num_kv_heads" in kind) == (spelling != "global_absent")


def test_a_config_with_the_per_layer_key_ignores_the_global_pair_as_the_reference_does():
    """With per_layer_config present, even empty, the reference reads
    global_head_dim and num_global_key_value_heads nowhere: the full layers
    keep the model's geometry, which the released E2B relies on."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    config = gemma4_config("gemma4-ple")
    config.update(global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True)
    full = config["layer_types"].index("full_attention")
    reference = Gemma4TextConfig(**config).per_layer_config[full]
    assert reference.head_dim == config["head_dim"]

    translated = translate_config(config)
    assert "full_attention" not in translated["kinds"]
    assert translated["head_dim"] == config["head_dim"]


def test_a_per_layer_count_equal_to_the_models_still_translates():
    """A count equal to the model's leaves the kind with its head dim alone:
    the reference fills per_layer_config with the model's own value for
    every layer, so nothing varies."""
    config = gemma4_config("gemma4-e2b")
    config["per_layer_config"] = {
        "1": {"head_dim": 32, "num_key_value_heads": config["num_key_value_heads"]}}

    assert translate_config(config)["kinds"]["full_attention"] == {"head_dim": 32}


@pytest.mark.parametrize("name", (*GEMMA4, "gemma4-e2b"))
def test_gemma4_checkpoints_load_through_the_translator(name):
    """The full load path on a gemma4 checkpoint: translate, weights, build,
    shape check. Sharing layers own no K/V leaves and the per-layer table
    lands; _check_tree enforces both leaf for leaf."""
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    assert model.v_norm and model.attention_scale == 1.0
    leaves = flat_tree(variables["params"])
    sharing = {"gemma4-ple": set(), "gemma4-kvshare": {2, 3}, "gemma4-e2b": {4, 5}}[name]
    assert set(model.kv_sharing) == sharing
    for index in sharing:
        assert f"layers_{index}.self_attn.k_proj.kernel" not in leaves
    assert "layers_0.self_attn.k_proj.kernel" in leaves
    if model.per_layer_input_dim:
        assert "embed_tokens_per_layer.embedding" in leaves


@pytest.mark.parametrize("name", (*GEMMA4, "gemma4-e2b"))
def test_gemma4_logits_match_the_reference_implementation(name):
    """Full-model parity, fully live on both branches. Largest observed max
    |logit difference| on CPU: gemma4-ple 4.9e-07, gemma4-kvshare 8.6e-07,
    gemma4-e2b 1.4e-06."""
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    ids = np.load(directory / "input_ids.npy")
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_sharing_layers_own_no_kv_and_name_their_provider(rng):
    """The tree shape of sharing on a translated model: layers past the
    cutoff keep q_proj, o_proj and q_norm but lose k_proj, v_proj and k_norm;
    the provider map follows the layer type, not the position."""
    config = translate_config(gemma4_config("gemma4-kvshare"))
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="xla"))
    params = model.init(rng, jnp.ones((1, 4), jnp.int32))["params"]

    assert set(model.kv_sharing) == {2, 3}
    assert model.kv_sharing[2] == 0 and model.kv_sharing[3] == 1
    for index in (2, 3):
        attention = params[f"layers_{index}"]["self_attn"]
        assert set(attention) == {"q_proj", "o_proj", "q_norm"}, set(attention)
    for index in (0, 1):
        attention = params[f"layers_{index}"]["self_attn"]
        assert {"k_proj", "v_proj", "k_norm"} <= set(attention)


def test_sharing_without_a_provider_and_sharing_everything_are_refused():
    config = translate_config(gemma4_config("gemma4-kvshare"))
    base = with_precision("causal_transformer", config,
                          dtype="float32", attention_impl="xla")
    with pytest.raises(ValueError, match="no earlier full_attention layer"):
        _ = models.build("causal_transformer", **{**base, "kv_shared_layers": (1, 2, 3)}).kv_sharing
    with pytest.raises(ValueError, match="leave a provider"):
        translate_config({**gemma4_config("gemma4-kvshare"), "num_kv_shared_layers": 4})


def test_the_features_leave_a_plain_tree_unchanged(rng):
    """Off by default: no PLE leaves, no missing K/V, same leaves as before."""
    config = translate_config(gemma4_config("gemma4-ple"))
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**config, "per_layer_input_dim": None,
                               "kv_shared_layers": None, "v_norm": False},
        dtype="float32", attention_impl="xla"))
    assert model.kv_sharing == {}
    flat = flat_tree(model.init(rng, jnp.ones((1, 4), jnp.int32))["params"])
    assert not [name for name in flat if "per_layer" in name]
    assert "layers_0.self_attn.k_proj.kernel" in flat


def test_new_leaves_are_declared():
    """The coverage sweep builds default configs only, so the new leaves are
    asserted here: the packed table and the projections are declared, the
    scalar norms fall under rank one, and the values norm holds no weight."""
    from dew.nn.sharding import declared_axes

    config = translate_config(gemma4_config("gemma4-kvshare"))
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**config, "per_layer_input_dim": 8},
        dtype="float32", attention_impl="xla"))
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    uncovered = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(variables)[0]:
        if leaf.ndim < 2:
            continue
        if declared_axes(path, leaf.ndim) is None:
            uncovered.append(jax.tree_util.keystr(path))
    assert uncovered == []


def test_a_sharing_model_decodes_like_it_prefills(rng):
    """The decode path of sharing: prefill writes the provider's cache, each
    single-token step reads it, and the tokens match a full forward."""
    config = translate_config(gemma4_config("gemma4-e2b"))
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**config, "max_seq_len": 16},
        dtype="float32", attention_impl="xla"))
    params = model.init(rng, jnp.ones((1, 4), jnp.int32))
    prompt = jax.random.randint(rng, (1, 3), 0, 64)

    cache = model.apply(params, 1, method="init_cache", mutable=["cache"])[1]["cache"]
    variables = {**params, "cache": cache}
    logits, mutated = model.apply(variables, prompt, decode=True, mutable=["cache"])
    first = jnp.argmax(logits[:, -1], axis=-1)
    token, variables = first[:, None], {**params, "cache": mutated["cache"]}
    generated = [first]
    for _ in range(3):
        logits, mutated = model.apply(variables, token, decode=True, mutable=["cache"])
        token = jnp.argmax(logits[:, -1], axis=-1)[:, None]
        variables = {**params, "cache": mutated["cache"]}
        generated.append(token[:, 0])
    decoded = jnp.concatenate([prompt, jnp.stack(generated, axis=1)], axis=1)

    assert jnp.array_equal(
        model.apply(params, decoded)[:, -1].argmax(axis=-1),
        jnp.asarray(generated[-1]))


def test_a_gemma4_config_without_layer_types_derives_the_reference_pattern():
    """A gemma4_text config need not carry layer_types: its own config class
    fills the 5:1 pattern at a fixed period of six and forces the last layer
    full. This read such a config as an all-full stack, which is a different
    model with the same weights. The expected pattern comes from the
    reference class, not from a copy of the rule."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    config = gemma4_config("gemma4-e2b")
    del config["layer_types"]
    config["num_hidden_layers"] = 14
    config["num_kv_shared_layers"] = 2

    derived = translate_config(config)["layer_types"]

    reference = Gemma4TextConfig(**{**config, "layer_types": None}).layer_types
    assert reference is not None
    assert derived == tuple(reference)
    assert derived.count("sliding_attention") == 11
    assert derived[-1] == "full_attention"


def test_a_gemma4_pattern_ending_in_a_sliding_layer_is_read_as_full():
    """Gemma4TextConfig rewrites a trailing sliding layer to full and warns,
    so the weights of such a checkpoint were trained with a full last layer;
    reading the config at its word would build a different model."""
    config = gemma4_config("gemma4-e2b")
    config["layer_types"] = ["full_attention", "sliding_attention"] * 3

    assert translate_config(config)["layer_types"][-1] == "full_attention"


def test_a_gemma3_pattern_keeps_its_last_layer():
    """The rule is Gemma 4's: gemma3_text has no such rewrite, and its own
    1B checkpoint ends on a sliding layer."""
    config = fixture_config("gemma3-1b")

    assert translate_config(config)["layer_types"][-1] == "sliding_attention"


def test_the_router_bias_lands_in_the_moe_collection():
    """DeepSeek's `e_score_correction_bias` is router state a training step
    moves, not a weight, and `Router` keeps it in the `moe` collection. The
    checkpoint names it `model.layers.N.mlp.gate.e_score_correction_bias`,
    six dot-separated parts, one fewer than the per-expert tensors the map
    also reads under `mlp`; a map that counts it as seven falls through to
    the params map and refuses the name. Its leaf is the checkpoint's tensor
    and the loaded model selects on it: zeroing the bias moves the logits by
    2.1 on deepseek-v3-tiny.
    """
    from dew.interop.hf_decoders import _dew_path
    from dew.interop.sources import load_shards

    directory = FIXTURES / "deepseek-v3-tiny"
    config = translate_config(fixture_config("deepseek-v3-tiny"))
    tensors = load_shards(directory)
    name = 'model.layers.1.mlp.gate.e_score_correction_bias'
    assert _dew_path(name, config) == (
        'moe', 'layers_1', 'mlp', 'gate', 'e_score_correction_bias')

    variables = translate_weights(tensors, config)
    assert set(flat_tree(variables['moe'])) == {
        'layers_1.mlp.gate.e_score_correction_bias'}
    assert np.array_equal(
        variables['moe']['layers_1']['mlp']['gate']['e_score_correction_bias'],
        tensors[name])
    assert not [path for path in flat_tree(variables['params'])
                if path.endswith('e_score_correction_bias')]

    model, loaded = fp32_decoder(directory)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    zeroed = {**loaded, 'moe': jax.tree.map(jnp.zeros_like, loaded['moe'])}
    moved = float(np.max(np.abs(np.asarray(model.apply(loaded, ids))
                                - np.asarray(model.apply(zeroed, ids)))))
    assert moved > 1.0, moved


def test_a_routed_checkpoint_without_its_bias_is_refused(tmp_path):
    """The tree check holds every collection to account, so a checkpoint
    that drops the balancing bias fails naming the leaf, and no router loads
    at zeros."""
    from dew.interop.safetensors_io import save_hf_layout
    from dew.interop.sources import load_shards

    directory = FIXTURES / "deepseek-v3-tiny"
    tensors = load_shards(directory)
    del tensors['model.layers.1.mlp.gate.e_score_correction_bias']
    save_hf_layout(tensors, fixture_config("deepseek-v3-tiny"), str(tmp_path))

    with pytest.raises(ValueError, match=r"missing \['moe\.layers_1\.mlp\.gate\.e_score_correction_bias'\]"):
        fp32_decoder(tmp_path)


def test_deepseek_configs_translate_field_by_field():
    """The V3 config becomes the mla mixer record with the released YaRN
    spelling and the mixture DeepseekV3MoE builds: a dense first layer, eight
    sigmoid-scored experts in four groups with two per token, top-4 scaled
    by 2.5, the balancing bias, and one shared expert of the routed width.
    V3.2 adds the indexer's three fields and names its sparse layer kind."""
    v3 = translate_config(fixture_config("deepseek-v3-tiny"))
    v32 = translate_config(fixture_config("deepseek-v32-tiny"))

    assert v3['mixer'] == {
        'class': 'mla', 'fields': {'q_lora_rank': 8, 'kv_lora_rank': 8,
        'qk_nope_head_dim': 8, 'qk_rope_head_dim': 8, 'v_head_dim': 8,
        'rope_interleave': True,
        'yarn': {'rope_type': 'yarn', 'rope_theta': 10000.0, 'factor': 40.0,
                 'original_max_position_embeddings': 4096, 'beta_fast': 32.0,
                 'beta_slow': 1.0, 'mscale': 1.0, 'mscale_all_dim': 1.0,
                 'truncate': True, 'attention_factor': None},
        'index_topk': None, 'index_n_heads': None, 'index_head_dim': None,
    }}
    assert v3['mixture'] == {
        'experts': 8, 'top_k': 4, 'layers': (1,), 'score_function': 'sigmoid',
        'scaling': 2.5, 'groups': 4, 'groups_per_token': 2, 'bias': True,
        'shared_features': 16, 'expert_features': 16,
    }
    assert v3['head_dim'] == 16 and v3['scale_after_cast'] and not v3['qk_norm']
    assert v3['layer_types'] == ('full_attention', 'full_attention')

    assert v32['mixer'] == {'class': 'mla', 'fields': {**v3['mixer']['fields'], 'index_topk': 4,
                                                      'index_n_heads': 8, 'index_head_dim': 16}}
    assert v32['mixture'] == v3['mixture']
    assert v32['layer_types'] == ('deepseek_sparse_attention',) * 2


def test_the_v32_fixture_is_the_sparse_model():
    """The dense mixer on deepseek-v32-tiny's weights differs from the
    fixture by 3.8, so the parity above covers the indexer's selection; a
    generator that lost the eager mask fold again would fail here."""
    directory = FIXTURES / "deepseek-v32-tiny"
    _, variables = fp32_decoder(directory)
    built = with_precision('causal_transformer',
                           translate_config(fixture_config("deepseek-v32-tiny")),
                           dtype='float32', attention_impl='reference')
    dense = models.build('causal_transformer', **{
        **built, 'mixer': {'class': 'mla', 'fields': {**built['mixer']['fields'], 'index_topk': None,
                                                     'index_n_heads': None, 'index_head_dim': None}}})
    params = {layer: ({**block, 'self_attn': {name: leaf for name, leaf
                                              in block['self_attn'].items()
                                              if name != 'indexer'}}
                      if layer.startswith('layers_') else block)
              for layer, block in variables['params'].items()}
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)

    logits = np.asarray(dense.apply({**variables, 'params': params}, ids))
    assert float(np.max(np.abs(logits - np.load(directory / "logits.npy")))) > 1.0


@pytest.mark.parametrize("name", DEEPSEEK)
def test_export_refuses_a_mixer_and_a_mixture_by_name(name, tmp_path, rng):
    """The writer covers the three attention families; a model with the mla
    mixer and one with routed experts on standard attention are refused
    naming the field their written config cannot carry. Neither writes a
    checkpoint."""
    model, variables = fp32_decoder(FIXTURES / name)
    with pytest.raises(ValueError, match="lacks 'qk_nope_head_dim'"):
        PretrainedDecoder.from_model(model, variables).save(str(tmp_path))

    config = translate_config(fixture_config(name))
    routed = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**config, "mixer": None, "head_dim": None,
                               "layer_types": None, "kinds": {}},
        dtype="float32", attention_impl="reference"))
    variables = routed.init(rng, jnp.ones((1, 4), jnp.int32))
    with pytest.raises(ValueError, match="lacks 'num_local_experts'"):
        PretrainedDecoder.from_model(routed, variables).save(str(tmp_path))


# --------------------------------------------------------------------------
# Qwen3.5: gated delta net layers, a gated attention, a sliced partial rotary
# --------------------------------------------------------------------------

QWEN35_REAL = FIXTURES / "qwen35-0.8b"


def qwen35_real_config():
    """The released Qwen/Qwen3.5-0.8B config's text decoder; the repo's own
    config.json is the multimodal wrapper around it."""
    return json.loads((QWEN35_REAL / "config.json").read_text())["text_config"]


def test_qwen35_config_translates_field_by_field():
    """The tiny hybrid: three linear_attention layers carrying the delta
    net's geometry as the kind's record, one full_attention layer riding
    the model's gated attention, the (1 + w) norms, a quarter-head rope in
    the 'default' convention, and the reference's rope_theta."""
    config = translate_config(fixture_config("qwen35-tiny"))

    assert config["layer_types"] == ("linear_attention",) * 3 + ("full_attention",)
    assert config["kinds"] == {"linear_attention": {"mixer": {
        "class": "gated_delta_net", "fields": {"linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 12,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4}}}}
    assert config["output_gate"] and config["qk_norm"] and config["scale_offset"]
    assert not config["scale_after_cast"] and "sandwich_norms" not in config
    assert config["partial_rotary_factor"] == 0.25
    assert config["partial_rotary_type"] == "default"
    assert config["rope_theta"] == 1000000.0
    assert config["head_dim"] == 32 and config["num_kv_heads"] == 2
    assert config["tie_embeddings"] and config["mlp"] == "swiglu"


def test_the_real_qwen35_0_8b_config_translates():
    """Qwen/Qwen3.5-0.8B's text_config, field for field: 24 layers in the
    3:1 pattern, 16 key and value heads of 128 in the delta net, 8 query
    and 2 key/value heads of 256 in the attention, a 64-dim rope at theta
    1e7, tied embeddings, and the MTP and mRoPE fields mapping to nothing
    because the reference's text forward reads none of them."""
    config = translate_config(qwen35_real_config())

    assert config["num_layers"] == 24
    assert config["layer_types"].count("full_attention") == 6
    assert config["layer_types"][3::4] == ("full_attention",) * 6
    assert config["kinds"]["linear_attention"]["mixer"] == {
        "class": "gated_delta_net", "fields": {"linear_num_key_heads": 16,
        "linear_num_value_heads": 16, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}}
    assert config["num_heads"] == 8 and config["num_kv_heads"] == 2
    assert config["head_dim"] == 256 and config["emb_features"] == 1024
    assert config["partial_rotary_factor"] == 0.25
    assert config["partial_rotary_type"] == "default"
    assert config["rope_theta"] == 10000000.0
    assert config["output_gate"] and config["tie_embeddings"]
    assert config["max_seq_len"] == 8192


def test_the_real_qwen35_rotary_rotates_the_dims_the_reference_rotates():
    """The partial rotary convention, pinned to the reference's numbers: on
    the real config's head_dim 256 and factor 0.25 the reference builds a
    64-dim rope with exponents over 64 (Qwen3_5TextRotaryEmbedding.
    compute_default_rope_parameters, modeling_qwen3_5.py:117-124), 32
    inverse frequencies with no zero tail. Gemma 4's proportional reading
    of the same factor puts the exponents over 256, whose frequencies
    differ from these by up to 4.7e-01, so a translation that guessed the
    convention would rotate every position by different angles. Largest
    observed cosine difference at 5 positions 6.3e-08."""
    from dew.nn.rope import rotary_freqs

    config = translate_config(qwen35_real_config())
    inv_freq = np.load(QWEN35_REAL / "inv_freq.npy")
    assert inv_freq.shape == (32,)
    rot_dim = int(config["head_dim"] * config["partial_rotary_factor"])
    assert rot_dim == 64

    positions = np.arange(5)
    cos, _ = rotary_freqs(jnp.asarray(positions), config["head_dim"], config["rope_theta"],
                          rot_dim=rot_dim, partial_rotary_type=config["partial_rotary_type"],
                          dtype=np.float32)
    assert cos.shape == (5, 32)
    assert float(np.max(np.abs(np.asarray(cos) - np.cos(positions[:, None] * inv_freq[None])))) < 1e-6

    proportional, _ = rotary_freqs(jnp.asarray(positions), config["head_dim"], config["rope_theta"],
                                   rot_dim=rot_dim, partial_rotary_type="proportional", dtype=np.float32)
    assert proportional.shape == (5, 128)
    assert not np.allclose(np.asarray(proportional)[:, :32],
                           np.cos(positions[:, None] * inv_freq[None]), atol=1e-2)


def test_qwen35_logits_match_the_reference_implementation():
    """Full-model parity on the tiny hybrid: the delta net layers, the gated
    attention with its sliced quarter-head rope (the reference applies its
    interleaved mRoPE to text-only positions, which is this rope exactly),
    the (1 + w) norms and the tied head. Largest observed max |logit
    difference| 9.1e-05 on logits of magnitude 6.8, tolerance 5e-4, every
    argmax equal. Translating the rope as Gemma 4's proportional convention
    instead moves the logits by 7.8e-01."""
    directory = FIXTURES / "qwen35-tiny"
    model, variables = fp32_decoder(directory)
    ids = np.load(directory / "input_ids.npy")
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 5e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_qwen35_weights_are_exactly_the_models_param_tree(rng):
    """The linear_attn tensors land under self_attn with the checkpoint's
    leaf names, the doubled q_proj fits the gated attention, and nothing is
    left over or missing."""
    from dew.interop.sources import load_shards

    config = translate_config(fixture_config("qwen35-tiny"))
    built = with_precision("causal_transformer", dict(config),
                           dtype="float32", attention_impl="reference")
    model = models.build("causal_transformer", **built)
    expected = flat_tree(model.init(rng, jnp.ones((1, 4), jnp.int32))["params"])
    loaded = flat_tree(translate_weights(
        load_shards(FIXTURES / "qwen35-tiny"), config)["params"])

    assert set(loaded) == set(expected)
    assert {name: leaf.shape for name, leaf in loaded.items()} == {
        name: leaf.shape for name, leaf in expected.items()}
    assert loaded["layers_0.self_attn.conv1d.weight"].shape == (112, 1, 4)
    assert loaded["layers_3.self_attn.q_proj.kernel"].shape == (64, 2 * 4 * 32)


def test_a_qwen35_config_without_layer_types_derives_the_reference_pattern():
    """Qwen3_5TextConfig fills the pattern from full_attention_interval
    (configuration_qwen3_5.py:112-117); the expected pattern comes from the
    reference class, not from a copy of the rule."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = {**fixture_config("qwen35-tiny"), "num_hidden_layers": 6,
              "full_attention_interval": 3}
    del config["layer_types"]

    derived = translate_config(config)["layer_types"]

    reference = Qwen3_5TextConfig(**{**config, "layer_types": None}).layer_types
    assert reference is not None
    assert derived == tuple(reference)
    assert derived == ("linear_attention", "linear_attention", "full_attention") * 2


@pytest.mark.parametrize("field, value, message", [
    ("attn_output_gate", False, "attn_output_gate"),
    ("layer_types", ["mamba"] * 4, "linear_attention or full_attention"),
    ("rope_parameters", {"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0},
     "rope_type 'yarn'"),
    ("rope_parameters", {"rope_type": "default", "rope_theta": 1e6, "factor": 4.0},
     "scaling fields"),
])
def test_a_qwen35_field_with_no_counterpart_is_refused(field, value, message):
    """Every knob the 5.16.1 reference cannot honour or dew cannot express
    names itself: the attention gate is never off in the reference, a
    Qwen3.5 layer is linear or full attention, and rope is plain."""
    config = {**fixture_config("qwen35-tiny"), field: value}
    with pytest.raises(ValueError, match=message):
        translate_config(config)


def test_the_qwen35_wrapper_config_is_refused_by_name():
    """Qwen/Qwen3.5-0.8B's config.json is the multimodal wrapper; the text
    decoder is its text_config and the weights sit under
    model.language_model.*, so the wrapper refuses like Gemma's."""
    with pytest.raises(ValueError, match="text_config"):
        translate_config(json.loads((QWEN35_REAL / "config.json").read_text()))


def test_qwen_mtp_weights_require_a_declared_prediction_layer():
    """An undeclared prediction component must never disappear on load."""
    config = translate_config(fixture_config("qwen35-tiny"))
    tensors = {"mtp.layers.0.self_attn.q_proj.weight": np.zeros((8, 64), np.float32)}
    with pytest.raises(ValueError, match="mtp tensors require"):
        translate_weights(tensors, config)
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_weights({"model.layers.0.linear_attn.nope.weight": np.zeros((4,), np.float32)},
                          config)


def test_export_refuses_the_qwen35_features(tmp_path):
    """The gate, the delta net kind and the partial rotary have no place in
    the exported families, and a refused model publishes nothing."""
    model, variables = fp32_decoder(FIXTURES / "qwen35-tiny")
    with pytest.raises(ValueError):
        PretrainedDecoder.from_model(model, variables).save(str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_a_qwen35_checkpoint_decodes_as_it_scores_in_parallel():
    """The loaded weights through the cache: a prefill and single-token
    steps against the parallel forward, every argmax equal. Largest
    observed logit difference 1.3e-05."""
    directory = FIXTURES / "qwen35-tiny"
    model, variables = fp32_decoder(directory, max_seq_len=16)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    full = jnp.asarray(model.apply(variables, ids))

    cache = model.apply(variables, ids.shape[0], method="init_cache",
                        mutable=["cache"])[1]["cache"]
    logits, mutated = model.apply({**variables, "cache": cache}, ids[:, :4],
                                  decode=True, mutable=["cache"])
    steps = [logits[:, -1]]
    for position in range(4, ids.shape[1]):
        logits, mutated = model.apply({**variables, **mutated}, ids[:, position:position + 1],
                                      decode=True, mutable=["cache"])
        steps.append(logits[:, -1])
    incremental = jnp.stack(steps, axis=1)

    difference = float(jnp.abs(full[:, 3:] - incremental).max())
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert jnp.array_equal(full[:, 3:].argmax(-1), incremental.argmax(-1))


# --------------------------------------------------------------------------
# Qwen3-Next: the Qwen3.5 hybrid with fused delta net projections and MoE
# --------------------------------------------------------------------------

QWEN3_NEXT = FIXTURES / "qwen3-next-tiny"
QWEN3_NEXT_REAL = FIXTURES / "qwen3-next-80b-a3b"


def test_qwen3_next_config_translates_field_by_field():
    """The delta net kind carries `fused_in_proj`, every layer routes with
    the shared expert gated, the dense width stays the model's, and the rest
    is the Qwen3.5 hybrid."""
    config = translate_config(fixture_config("qwen3-next-tiny"))

    assert config["layer_types"] == ("linear_attention",) * 3 + ("full_attention",)
    assert config["kinds"] == {"linear_attention": {"mixer": {
        "class": "gated_delta_net", "fields": {"linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 12, "linear_conv_kernel_dim": 4, "fused_in_proj": True}}}}
    assert config["mixture"] == {"experts": 8, "top_k": 2, "layers": (0, 1, 2, 3),
                                 "norm_topk_prob": True, "expert_features": 16,
                                 "shared_features": 16, "shared_gate": True}
    assert config["mlp_features"] == 160
    assert config["output_gate"] and config["scale_offset"] and config["qk_norm"]
    assert config["partial_rotary_factor"] == 0.25 and config["partial_rotary_type"] == "default"
    assert config["rope_theta"] == 1e7 and not config["tie_embeddings"]
    assert config["num_nextn_predict_layers"] == 1


def test_the_real_qwen3_next_80b_config_translates():
    """Qwen/Qwen3-Next-80B-A3B-Instruct's config, field for field: 48 layers
    in the 3:1 pattern derived from full_attention_interval, 16 key and 32
    value heads of 128 in the delta net, 16 query and 2 key/value heads of
    256 with a 64-dim rope at theta 1e7, 512 experts with 10 per token
    beside a 512-wide shared expert on every layer, and no prediction layer
    declared."""
    config = translate_config(json.loads((QWEN3_NEXT_REAL / "config.json").read_text()))

    assert config["num_layers"] == 48 and config["layer_types"][3::4] == ("full_attention",) * 12
    assert config["kinds"]["linear_attention"]["mixer"] == {
        "class": "gated_delta_net", "fields": {"linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "fused_in_proj": True}}
    assert config["num_heads"] == 16 and config["num_kv_heads"] == 2 and config["head_dim"] == 256
    assert int(config["head_dim"] * config["partial_rotary_factor"]) == 64
    assert config["rope_theta"] == 1e7 and config["mlp_features"] == 5120
    assert config["mixture"] == {"experts": 512, "top_k": 10, "layers": tuple(range(48)),
                                 "norm_topk_prob": True, "expert_features": 512,
                                 "shared_features": 512, "shared_gate": True}
    assert config["num_nextn_predict_layers"] == 0 and not config["tie_embeddings"]


def test_qwen3_next_logits_match_the_reference_implementation():
    """Full-model parity on the tiny hybrid: three fused delta net layers,
    the gated attention, and softmax top-2 routing with the gated shared
    expert on every layer. Largest observed max |logit difference| 2.3e-05
    on logits of magnitude 6.6, every argmax equal."""
    model, variables = fp32_decoder(QWEN3_NEXT)
    ids = np.load(QWEN3_NEXT / "input_ids.npy")
    reference = np.load(QWEN3_NEXT / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    assert scaled_difference(logits, reference) < 1e-4
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_qwen3_next_prediction_layer_matches_the_published_composition():
    """The mtp.* tensors compose as vLLM's Qwen3NextMultiTokenPredictor does
    over the trunk's hidden states; the fixture records that composition
    run on the reference's own decoder layer. Largest observed difference
    1.8e-05."""
    model, variables = fp32_decoder(QWEN3_NEXT)
    ids = jnp.asarray(np.load(QWEN3_NEXT / "input_ids.npy"), jnp.int32)
    reference = np.load(QWEN3_NEXT / "mtp_reference.npz")

    hidden = model.apply(variables, ids, method=model.hidden_states)
    predictions = model.apply(variables, hidden, ids, method=model.mtp_logits)[0]

    assert scaled_difference(np.asarray(predictions), reference["logits"]) < 1e-4


def test_qwen3_next_weights_are_exactly_the_models_param_tree(rng):
    """The fused in_proj_qkvz and in_proj_ba land under self_attn at the
    checkpoint's widths, the packed experts unpack into stacked kernels, and
    nothing is left over or missing."""
    from dew.interop.sources import load_shards

    config = translate_config(fixture_config("qwen3-next-tiny"))
    built = with_precision("causal_transformer", dict(config),
                           dtype="float32", attention_impl="reference")
    model = models.build("causal_transformer", **built)
    expected = flat_tree(model.init(rng, jnp.ones((1, 4), jnp.int32))["params"])
    loaded = flat_tree(translate_weights(load_shards(QWEN3_NEXT), config)["params"])

    assert set(loaded) == set(expected)
    assert {name: leaf.shape for name, leaf in loaded.items()} == {
        name: leaf.shape for name, leaf in expected.items()}
    assert loaded["layers_0.self_attn.in_proj_qkvz.kernel"].shape == (64, 2 * 16 + 2 * 48)
    assert loaded["layers_0.self_attn.in_proj_ba.kernel"].shape == (64, 8)
    assert loaded["layers_0.mlp.experts.gate_proj.kernel"].shape == (8, 64, 16)
    assert loaded["layers_3.self_attn.q_proj.kernel"].shape == (64, 2 * 8 * 16)


def test_the_fused_delta_net_projection_is_read_by_key_head_group():
    """`fix_query_key_value_ordering` (modeling_qwen3_next.py:558-586) splits
    each key head's row group into its q, k and the v and z of the value
    heads it serves, so the fused kernel is not q|k|v|z stacked whole. The
    split-projection model on the regrouped rows must compute the same
    layer; on the rows stacked whole it must not."""
    from dew.nn.linear import GatedDeltaNet

    def net(fused: bool) -> GatedDeltaNet:
        return GatedDeltaNet(emb_features=16, num_k_heads=2, num_v_heads=4, head_k_dim=4,
                             head_v_dim=6, fused_in_proj=fused)

    fused, split = net(fused=True), net(fused=False)
    x = jax.random.normal(jax.random.key(0), (1, 5, 16))
    variables = fused.init(jax.random.key(1), x)
    params = dict(variables["params"])
    qkvz_kernel = np.asarray(params.pop("in_proj_qkvz")["kernel"])
    ba_kernel = np.asarray(params.pop("in_proj_ba")["kernel"])
    q, k, v, z = np.split(qkvz_kernel.reshape(16, 2, 2 * 4 + 2 * 2 * 6), [4, 8, 8 + 12], axis=-1)
    ba = ba_kernel.reshape(16, 2, 4)
    regrouped = {
        **params,
        "in_proj_qkv": {
            "kernel": np.concatenate([q.reshape(16, -1), k.reshape(16, -1), v.reshape(16, -1)], -1)
        },
        "in_proj_z": {"kernel": z.reshape(16, -1)},
        "in_proj_b": {"kernel": ba[..., :2].reshape(16, -1)},
        "in_proj_a": {"kernel": ba[..., 2:].reshape(16, -1)},
    }
    whole = {**regrouped,
             "in_proj_qkv": {"kernel": qkvz_kernel[:, :2 * 8 + 24]},
             "in_proj_z": {"kernel": qkvz_kernel[:, 2 * 8 + 24:]}}

    wanted = np.asarray(fused.apply(variables, x))
    np.testing.assert_allclose(np.asarray(split.apply({"params": regrouped}, x)), wanted, atol=1e-6, rtol=0)
    assert float(np.abs(np.asarray(split.apply({"params": whole}, x)) - wanted).max()) > 1e-2


def test_a_qwen3_next_config_derives_the_reference_pattern_and_routed_layers():
    """Without layer_types the 3:1 pattern comes from full_attention_interval
    (configuration_qwen3_next.py:131-136), and mlp_only_layers with
    decoder_sparse_step pick the routed layers the way the reference does
    (modeling_qwen3_next.py:813-818); the expected pattern comes from the
    reference class, not from a copy of the rule."""
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    config = {**fixture_config("qwen3-next-tiny"), "num_hidden_layers": 6,
              "full_attention_interval": 3, "decoder_sparse_step": 2, "mlp_only_layers": [3]}
    del config["layer_types"]

    translated = translate_config(config)

    reference = Qwen3NextConfig(**{**config, "layer_types": None}).layer_types
    assert list(translated["layer_types"]) == reference
    assert translated["layer_types"] == ("linear_attention", "linear_attention", "full_attention") * 2
    assert translated["mixture"]["layers"] == (1, 5)


@pytest.mark.parametrize("field, value, message", [
    ("rope_scaling", {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64},
     "rope_scaling"),
    ("num_nextn_predict_layers", 2, "num_nextn_predict_layers"),
    ("decoder_sparse_step", 0, "decoder_sparse_step"),
    ("layer_types", ["mamba"] * 4, "linear_attention or full_attention"),
])
def test_a_qwen3_next_field_with_no_counterpart_is_refused(field, value, message):
    """The YaRN the model card suggests past 256K, more than the one
    prediction layer vLLM builds, a zero sparse step and a foreign layer kind
    each refuse by name."""
    config = {**fixture_config("qwen3-next-tiny"), field: value}
    with pytest.raises(ValueError, match=message):
        translate_config(config)


def test_a_qwen3_next_checkpoint_decodes_as_it_scores_in_parallel():
    """The fused delta net's conv and recurrent states through the cache: a
    prefill and single-token steps against the parallel forward, every
    argmax equal."""
    model, variables = fp32_decoder(QWEN3_NEXT, max_seq_len=16)
    ids = jnp.asarray(np.load(QWEN3_NEXT / "input_ids.npy"), jnp.int32)
    full = jnp.asarray(model.apply(variables, ids))

    cache = model.apply(variables, ids.shape[0], method="init_cache",
                        mutable=["cache"])[1]["cache"]
    logits, mutated = model.apply({**variables, "cache": cache}, ids[:, :4],
                                  decode=True, mutable=["cache"])
    steps = [logits[:, -1]]
    for position in range(4, ids.shape[1]):
        logits, mutated = model.apply({**variables, **mutated}, ids[:, position:position + 1],
                                      decode=True, mutable=["cache"])
        steps.append(logits[:, -1])
    incremental = jnp.stack(steps, axis=1)

    assert scaled_difference(np.asarray(incremental), np.asarray(full[:, 3:])) < 1e-4
    assert jnp.array_equal(full[:, 3:].argmax(-1), incremental.argmax(-1))




# ---------------------------------------------------------------------------
# Gemma 3n: AltUp's residual copies, LAuReL, activation sparsity, per-layer widths
# ---------------------------------------------------------------------------

GEMMA3N = FIXTURES / "gemma3n-tiny"


def test_gemma3n_config_translates_field_by_field():
    """The tiny config: three residual copies with the clip and the output
    scale, a LAuReL rank of 8, sparsity on the first two of four layers,
    widths of 48 and 64, per-layer inputs of 8, the last layer sharing K/V,
    and the sliding kind on its own rope base."""
    config = translate_config(fixture_config("gemma3n-tiny"))

    assert config["altup"] == {"num_inputs": 3, "active_idx": 0, "coef_clip": 120.0,
                               "correct_scale": True}
    assert config["laurel_rank"] == 8
    assert config["activation_sparsity_pattern"] == (0.95, 0.95, 0.0, 0.0)
    assert config["mlp_features"] == (48, 48, 64, 64) and config["mlp"] == "geglu"
    assert config["per_layer_input_dim"] == 8 and config["per_layer_input_vocab"] == 64
    assert config["kv_shared_layers"] == (3,)
    assert config["layer_types"] == ("sliding_attention", "sliding_attention",
                                     "full_attention", "sliding_attention")
    assert config["kinds"] == {"sliding_attention": {"window": 4, "rope_theta": 10000.0}}
    assert config["rope_theta"] == 1000000.0
    assert config["v_norm"] and config["sandwich_norms"] and config["embedding_scale"]
    assert config["attention_scale"] == 1.0 and config["final_logit_softcap"] == 30.0
    assert config["head_dim"] == 8 and config["num_kv_heads"] == 2


def test_the_real_gemma_3n_e2b_text_config_translates():
    """google/gemma-3n-E2B's text_config, from a mirror: 30 layers with
    every fifth full and the last 10 sharing K/V, 8 query and 2 key/value
    heads of 256 over a hidden size of 2048, one width of 8192 (the list is
    uniform), sparsity 0.95 on the first 10 layers, four residual copies,
    LAuReL rank 64, per-layer inputs of 256 from a table of 262144 rows
    under a vocabulary of 262400, and softcap 30."""
    config = translate_config(fixture_config("gemma-3n-e2b")["text_config"])

    assert config["num_layers"] == 30
    assert config["layer_types"].count("full_attention") == 6
    assert config["layer_types"][4] == "full_attention"
    assert config["emb_features"] == 2048 and config["vocab_size"] == 262400
    assert config["num_heads"] == 8 and config["num_kv_heads"] == 2
    assert config["head_dim"] == 256
    assert config["mlp_features"] == 8192 and config["mlp"] == "geglu"
    assert config["activation_sparsity_pattern"] == (0.95,) * 10 + (0.0,) * 20
    assert config["altup"] == {"num_inputs": 4, "active_idx": 0, "coef_clip": 120.0,
                               "correct_scale": True}
    assert config["laurel_rank"] == 64
    assert config["per_layer_input_dim"] == 256 and config["per_layer_input_vocab"] == 262144
    assert config["kv_shared_layers"] == tuple(range(config["num_layers"] - 10, config["num_layers"]))
    assert config["kinds"] == {"sliding_attention": {"window": 512, "rope_theta": 10000.0}}
    assert config["rope_theta"] == 1000000.0
    assert config["final_logit_softcap"] == 30.0
    assert "mixture" not in config and "rope_scaling" not in config


def test_a_gemma3n_config_without_its_lists_takes_the_reference_defaults():
    """Gemma3nTextConfig fills every fifth layer full, expands one width
    to every layer and puts sparsity 0.95 on the first ten layers of a
    deeper model; the expected values come from the reference class."""
    from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig

    config = fixture_config("gemma3n-tiny")
    for field in ("layer_types", "activation_sparsity_pattern", "rope_parameters"):
        del config[field]
    config.update(num_hidden_layers=12, intermediate_size=48, num_kv_shared_layers=2)
    reference = Gemma3nTextConfig(**config)

    translated = translate_config(config)
    assert translated["layer_types"] == tuple(reference.layer_types or ())
    assert isinstance(reference.activation_sparsity_pattern, list)
    assert translated["activation_sparsity_pattern"] == tuple(reference.activation_sparsity_pattern)
    assert translated["mlp_features"] == 48 and reference.intermediate_size == [48] * 12
    assert translated["kinds"]["sliding_attention"]["rope_theta"] == 10000.0
    assert translated["rope_theta"] == 1000000.0


def test_gemma3n_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 4.2e-06
    with identical argmax. The copies' projections, each layer's AltUp and
    LAuReL leaves and the sharing layer's missing K/V are what the tree
    holds."""
    model, variables = fp32_decoder(GEMMA3N)
    ids = np.load(GEMMA3N / "input_ids.npy")
    reference = np.load(GEMMA3N / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    params = variables["params"]
    assert {name for name in params if name.startswith("altup")} == {
        "altup_projections_0", "altup_projections_1",
        "altup_unembed_projections_0", "altup_unembed_projections_1"}
    assert params["layers_0"]["altup"]["prediction_coefs"]["kernel"].shape == (3, 9)
    assert params["layers_0"]["laurel"]["linear_left"]["kernel"].shape == (32, 8)
    assert params["layers_2"]["mlp"]["gate_proj"]["kernel"].shape == (32, 64)
    assert "k_proj" not in params["layers_3"]["self_attn"]


def test_gemma3n_decodes_through_the_cache_as_it_scores_in_parallel():
    """The stream of copies rides the decode path: a prefill and single-token
    steps through the KV cache agree with the whole sequence."""
    model, variables = fp32_decoder(GEMMA3N, max_seq_len=16)
    ids = jnp.asarray(np.load(GEMMA3N / "input_ids.npy")[:1, :12], jnp.int32)
    full = jnp.asarray(model.apply(variables, ids))
    state = model.init(jax.random.PRNGKey(0), ids[:, :1], decode=True)
    steps = []
    for position in range(ids.shape[1]):
        step, state = model.apply(
            {**variables, "cache": state["cache"]}, ids[:, position:position + 1],
            decode=True, mutable=["cache"])
        steps.append(step)
    incremental = jnp.concatenate(steps, axis=1)
    difference = float(jnp.max(jnp.abs(incremental - full)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


@pytest.mark.parametrize("field,value,message", [
    ("activation_sparsity_pattern", [0.95, 0.0], "one fraction per layer"),
    ("rope_scaling", {"rope_type": "linear", "factor": 2.0}, "rope_type 'linear'"),
    ("altup_num_inputs", 1, "altup_num_inputs"),
    ("altup_active_idx", 3, "altup_active_idx"),
    ("hidden_activation", "relu", "hidden_act"),
])
def test_a_gemma3n_field_with_no_counterpart_is_refused(field, value, message):
    """A rope_scaling beside the nested rope_parameters lands on the full
    layers as Gemma3nTextConfig folds it, and a type the rotary table cannot
    express raises there with the field's name."""
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("gemma3n-tiny"), field: value})


def test_a_llama3_rope_scaling_beside_nested_rope_parameters_lands_on_the_full_layers():
    """The fold Gemma3nTextConfig applies: the ramp reaches the full kind
    alone, the sliding kind keeps its plain rope."""
    config = {**fixture_config("gemma3n-tiny"), "rope_scaling": {
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 32}}
    translated = translate_config(config)
    assert translated["kinds"]["full_attention"]["rope_scaling"]["factor"] == 8.0
    assert "rope_scaling" not in translated["kinds"]["sliding_attention"]
    assert "rope_scaling" not in translated


def test_a_gemma3n_wrapper_config_is_refused_by_name():
    """The E2B repo's config.json wraps the text decoder beside vision and
    audio towers; the text_config alone is what translates."""
    with pytest.raises(ValueError, match="multimodal wrapper"):
        translate_config(fixture_config("gemma-3n-e2b"))


def test_gemma3n_export_is_refused_as_an_unrepresentable_model(tmp_path):
    """No exported family carries Gemma 3n's shape, so its export is refused
    by name and leaves nothing behind, whichever feature is met first."""
    model, variables = fp32_decoder(GEMMA3N)
    with pytest.raises(ValueError):
        PretrainedDecoder.from_model(model, variables).save(str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_the_released_llada_config_translates_field_by_field():
    """GSAI-ML/LLaDA-8B-Base reads as a 32-layer Llama geometry with full
    attention on every layer, plain rope at theta 500000 and the mask id the
    objective corrupts with."""
    config = translate_config(fixture_config("llada-8b"))
    assert (config['vocab_size'], config['emb_features'], config['num_layers']) == (126464, 4096, 32)
    assert (config['num_heads'], config['num_kv_heads'], config['head_dim']) == (32, 32, 128)
    assert (config['mlp_features'], config['mlp'], config['rope_theta']) == (12288, 'swiglu', 500000.0)
    assert config['layer_types'] == ('full_attention',) * 32
    assert config['causal'] is False and config['mask_token_id'] == 126336
    assert config['tie_embeddings'] is False and not config['attention_bias']


def test_the_released_dream_config_translates_field_by_field():
    """Dream-org/Dream-v0-Base-7B reads as a 28-layer Qwen2.5 geometry with
    the q/k/v biases over a bias-free o_proj, full attention on every layer
    and the mask id the objective corrupts with."""
    config = translate_config(fixture_config("dream-7b"))
    assert (config['vocab_size'], config['emb_features'], config['num_layers']) == (152064, 3584, 28)
    assert (config['num_heads'], config['num_kv_heads'], config['head_dim']) == (28, 4, 128)
    assert (config['mlp_features'], config['rope_theta']) == (18944, 1000000.0)
    assert config['layer_types'] == ('full_attention',) * 28
    assert config['causal'] is False and config['mask_token_id'] == 151666
    assert config['attention_bias'] and config['o_proj_bias'] is False


def test_llada_without_its_mask_token_is_refused():
    """The mask id is what the objective corrupts with, so a config without
    one raises naming it. Training against id zero would learn the wrong token."""
    config = fixture_config("llada-8b")
    del config['mask_token_id']
    with pytest.raises(ValueError, match="mask_token_id"):
        translate_config(config)


def test_dream_with_mrope_is_refused():
    """Dream ships use_mrope false. A true value would rotate positions the
    backbone has no grid for, so it names the field."""
    config = {**fixture_config("dream-7b"), 'use_mrope': True}
    with pytest.raises(ValueError, match="use_mrope"):
        translate_config(config)


def test_llada_renames_its_tensors_onto_the_shared_map():
    """The checkpoint spells its tensors OLMo-style, so each name lands on the
    llama-layout path with the same transpose rule and no second table. A
    name outside that spelling names itself."""
    config = translate_config(fixture_config("llada-8b"))
    tensors = {
        'model.transformer.wte.weight': np.ones((4, 4), np.float32),
        'model.transformer.blocks.0.attn_norm.weight': np.ones((4,), np.float32),
        'model.transformer.blocks.0.q_proj.weight': np.ones((4, 4), np.float32),
        'model.transformer.blocks.0.k_proj.weight': np.ones((4, 4), np.float32),
        'model.transformer.blocks.0.v_proj.weight': np.ones((4, 4), np.float32),
        'model.transformer.blocks.0.attn_out.weight': np.ones((4, 4), np.float32),
        'model.transformer.blocks.0.ff_norm.weight': np.ones((4,), np.float32),
        'model.transformer.blocks.0.ff_proj.weight': np.ones((4, 4), np.float32),
        'model.transformer.blocks.0.up_proj.weight': np.ones((4, 4), np.float32),
        'model.transformer.blocks.0.ff_out.weight': np.ones((4, 4), np.float32),
        'model.transformer.ln_f.weight': np.ones((4,), np.float32),
        'model.transformer.ff_out.weight': np.ones((4, 4), np.float32),
    }
    variables = translate_weights(tensors, config)['params']
    assert variables['embed_tokens']['embedding'].shape == (4, 4)
    assert variables['layers_0']['self_attn']['q_proj']['kernel'].shape == (4, 4)
    assert variables['layers_0']['mlp']['gate_proj']['kernel'].shape == (4, 4)
    assert variables['layers_0']['mlp']['up_proj']['kernel'].shape == (4, 4)
    assert variables['layers_0']['mlp']['down_proj']['kernel'].shape == (4, 4)
    assert variables['layers_0']['input_layernorm']['scale'].shape == (4,)
    assert variables['norm']['scale'].shape == (4,)
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_weights({'model.transformer.blocks.0.rotary_emb.inv_freq': np.ones((2,))}, config)


def test_diffusion_gemma_text_reuses_the_gemma4_map_in_decoder_mode():
    """The same weights with the diffusion model_type translate as the Gemma 4
    text decoder does, except the record reads decoder mode, values off keys
    on full layers whatever the config says, and the head's divide-by-30. The
    encoder cache, the canvas and the self-conditioning loop live in
    dew.diffusion.block."""
    gemma4 = translate_config(fixture_config("gemma4-ple"))
    diffusion = translate_config({**fixture_config("gemma4-ple"),
                                  'model_type': 'diffusion_gemma_text'})
    assert diffusion['causal'] is False
    assert diffusion['attention_k_eq_v'] is True
    assert diffusion['final_logit_softcap'] == 30.0
    assert {key: value for key, value in diffusion.items()
            if key not in ('causal', 'attention_k_eq_v',
                           'final_logit_softcap')} == {
        key: value for key, value in gemma4.items()
        if key not in ('attention_k_eq_v', 'final_logit_softcap')}


QWEN35_0_8B = "Qwen/Qwen3.5-0.8B"
QWEN35_0_8B_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"),
                    reason=f"DEW_NETWORK_TESTS=1 downloads {QWEN35_0_8B}")
def test_the_real_qwen35_0_8b_delta_rule_chunks_as_it_recurs():
    """Qwen3.5-0.8B's own keys on a 256-token prompt reach the chunk
    inverse's hard case: in the fifth delta-rule layer the series form of
    `strictly_lower_inverse` came out 1e24 off in fp32 and every later
    layer, the logits and a served request NaN (transformers' forward was
    finite, 3.3e-05 from Dew's once fixed). Each layer's chunked rule now
    gives what the token-by-token rule gives on the same inputs, and the
    logits are finite."""
    import dew
    from dew.nn import linear

    task = dew.pipeline(QWEN35_0_8B, revision=QWEN35_0_8B_REVISION, dtype="float32",
                        param_dtype="float32")
    model = task.model.language_model
    params = {"params": task.variables["params"]["language_model"]}
    captured = []
    chunked = linear.chunk_gated_delta_rule

    def recording(query, key, value, g, beta, state=None, chunk_size=linear.CHUNK_SIZE):
        jax.debug.callback(lambda *args: captured.append([np.asarray(x) for x in args]),
                           query, key, value, g, beta)
        return chunked(query, key, value, g, beta, state, chunk_size)

    prompt = np.random.default_rng(0).integers(100, 150_000, (1, 256)).astype(np.int32)
    with jax.default_matmul_precision("highest"):
        linear.chunk_gated_delta_rule = recording
        try:
            logits = jax.jit(lambda p, t: model.apply(p, t))(params, prompt)
            jax.effects_barrier()
        finally:
            linear.chunk_gated_delta_rule = chunked
        assert bool(jnp.isfinite(logits).all())
        assert len(captured) == model.layer_types.count("linear_attention")
        for inputs in captured:
            got = chunked(*inputs)[0]
            want = linear.recurrent_gated_delta_rule(*inputs)[0]
            assert float(jnp.max(jnp.abs(got - want))) <= 1e-4 * float(jnp.max(jnp.abs(want)))
