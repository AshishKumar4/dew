"""Hugging Face decoder checkpoints, loaded into CausalTransformer and back out.

The claim these tests defend is parity, not plausibility: the same weights and
the same token ids through the reference implementation and through dew have to
produce the same logits. tools/hf_reference.py writes the fixtures under torch
and transformers (the reference), including two tiny random-weight checkpoints
whose logits are committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- qwen3-tiny  : max |logit difference| 8.3e-06, tolerance 1e-4
- gemma3-tiny : max |logit difference| 3.3e-06, tolerance 1e-4
- llama-tiny  : max |logit difference| 6.1e-06, tolerance 1e-4 (untied head,
  biased projections)
- Qwen3-0.6B  : max |top-32 logit difference| 1.4e-04, mean 1.2e-05, tolerance
  5e-3, and the argmax of all 48 positions equal. The larger residue is 28
  layers of a real checkpoint accumulating fp32 rounding, not a different
  computation.
- gemma4-ple  : max |logit difference| 4.9e-07, tolerance 1e-5
- gemma4-kvshare: max |logit difference| 8.6e-07, tolerance 1e-5
- gemma4-e2b  : max |logit difference| 1.4e-06, tolerance 1e-5. An E2B-shaped
  tiny config with every Gemma 4 gap at once: partial rotary, mixed head
  dims, double-wide MLP, KV sharing, per-layer inputs and the values norm.
  The logit cap is absent because the text path never reads it.
"""

import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.hf_decoders import (
    load_pretrained_decoder, save_pretrained_decoder, translate_config,
    translate_weights,
)
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
TINY = ("qwen3-tiny", "gemma3-tiny", "llama-tiny")
TORCH_VENV = Path("/tmp/hfref/bin/python")
REAL = FIXTURES / "qwen3-0.6b"


def fixture_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


def fp32_decoder(directory, **kwargs):
    """The fixture as a model plus variables, in fp32 on the reference kernel."""
    return load_pretrained_decoder(str(directory), dtype='float32',
                                   attention_impl='reference', **kwargs)


def flat_tree(tree):
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {'.'.join(str(entry.key) for entry in path): leaf for path, leaf in leaves}


def test_qwen3_config_translates_field_by_field():
    config = translate_config(fixture_config("qwen3-tiny"))

    assert config == {
        'vocab_size': 256, 'emb_features': 64, 'num_layers': 2, 'num_heads': 4,
        'num_kv_heads': 2, 'head_dim': 16, 'mlp': 'swiglu', 'mlp_features': 128,
        'max_seq_len': 64, 'rope_theta': 1e6,
        'layer_types': ('full_attention', 'full_attention'), 'kinds': {},
        'norm_eps': 1e-6, 'scale_after_cast': True, 'qk_norm': True, 'attention_bias': False,
        'tie_embeddings': True,
    }


def test_gemma3_config_carries_the_gemma_switches():
    config = translate_config(fixture_config("gemma3-tiny"))

    assert config['sandwich_norms'] and config['scale_offset']
    assert config['embedding_scale'] and config['mlp'] == 'geglu'
    assert config['qk_norm'] and config['num_kv_heads'] == 1
    # query_pre_attn_scalar 16, not the head_dim of 32
    assert config['attention_scale'] == pytest.approx(0.25)
    assert config['final_logit_softcap'] == 30.0
    assert config['layer_types'] == ('sliding_attention', 'full_attention')
    # rope_local_base_freq and the window belong to the sliding kind,
    # rope_theta to the model the full layers take it from
    assert config['rope_theta'] == 1e6
    assert config['kinds'] == {'sliding_attention': {'window': 4, 'rope_theta': 1e4}}


def test_a_multimodal_gemma3_config_is_refused():
    """Only a text decoder maps, and no published multimodal Gemma 3 has a
    text_config that would: gemma-3-4b, 12b and 27b all carry rope_scaling
    {'rope_type': 'linear', 'factor': 8}, which the field map refuses. So the
    refusal names the model_type instead of loading the text half of a
    checkpoint whose vision tower nothing here runs."""
    wrapped = {'model_type': 'gemma3', 'text_config': fixture_config("gemma3-tiny"),
               'vision_config': {'hidden_size': 8}, 'mm_tokens_per_image': 256,
               'boi_token_index': 255999, 'eoi_token_index': 256000,
               'image_token_index': 262144}

    with pytest.raises(ValueError, match="model_type 'gemma3'"):
        translate_config(wrapped)


def test_the_real_gemma3_1b_config_translates():
    """google/gemma-3-1b-pt is gated, so the fixture is the identical config
    from a mirror. Nothing in it is beyond the field map."""
    config = translate_config(fixture_config("gemma3-1b"))

    assert config['emb_features'] == 1152 and config['num_layers'] == 26
    assert (config['num_heads'], config['num_kv_heads']) == (4, 1)
    assert config['head_dim'] == 256 and config['mlp_features'] == 6912
    # the config names no tie_word_embeddings, and Gemma3TextConfig ties by
    # default where Qwen and Llama do not
    assert config['vocab_size'] == 262144 and config['tie_embeddings'] is True
    assert config['attention_scale'] == pytest.approx(256 ** -0.5)
    assert config['rope_theta'] == 1e6
    assert config['kinds'] == {'sliding_attention': {'window': 512, 'rope_theta': 1e4}}
    assert config['sandwich_norms'] and config['scale_offset']
    # sliding_window_pattern 6: five sliding layers, then a full one
    assert config['layer_types'][:6] == (
        'sliding_attention',) * 5 + ('full_attention',)
    assert config['layer_types'].count('full_attention') == 4
    # the cache is clamped, the config asks for 32768
    assert config['max_seq_len'] == 8192


@pytest.mark.parametrize("field, value, message", [
    ('model_type', 'mamba', "model_type 'mamba'"),
    ('model_type', 'qwen2', "o_proj does not"),
    ('attn_logit_softcapping', 50.0, "attn_logit_softcapping"),
    ('use_bidirectional_attention', True, "use_bidirectional_attention"),
    ('hidden_activation', 'relu', "hidden_act 'relu'"),
    ('rope_parameters', {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1e6},
     "rope_type 'linear'"),
    ('sliding_window', None, "sliding_window is not set"),
    ('quantization_config', {'bits': 4}, "quantization_config"),
])
def test_a_config_field_with_no_counterpart_is_refused(field, value, message):
    config = {**fixture_config("gemma3-tiny"), field: value}
    with pytest.raises(ValueError, match=message):
        translate_config(config)


def test_a_rope_scaling_spelled_the_old_way_is_refused():
    """transformers reads rope_type from the older 'type' key too
    (modeling_rope_utils.py:785, 839), so a config that spells its Yarn
    scaling that way scales there and must not load here as plain rope. A
    bare factor names no type at all and is still a scaling."""
    yarn = {**fixture_config("llama-tiny"),
            'rope_scaling': {'type': 'yarn', 'factor': 4.0,
                             'original_max_position_embeddings': 32}}
    with pytest.raises(ValueError, match="rope_type 'yarn'"):
        translate_config(yarn)

    factor_only = {**fixture_config("llama-tiny"), 'rope_scaling': {'factor': 8.0}}
    with pytest.raises(ValueError, match=r"rope_scaling scaling fields \['factor'\]"):
        translate_config(factor_only)


@pytest.mark.parametrize("name", TINY)
def test_translated_weights_are_exactly_the_models_param_tree(name, rng):
    """Same paths, same shapes, same dtypes as a freshly initialised model."""
    config = translate_config(fixture_config(name))
    built = with_precision('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = models.build('causal_transformer', **built)
    initialised = flat_tree(model.init(rng, jnp.zeros((1, 4), jnp.int32))['params'])

    from dew.interop.hf_decoders import _load_shards
    loaded = flat_tree(translate_weights(_load_shards(FIXTURES / name), config)['params'])

    assert set(loaded) == set(initialised)
    for path, leaf in loaded.items():
        assert leaf.shape == initialised[path].shape, path
        assert leaf.dtype == jnp.float32, path


@pytest.mark.parametrize("name", TINY)
def test_fp32_logits_match_the_reference_implementation(name):
    """The parity claim: transformers' logits, our logits, same weights."""
    directory = FIXTURES / name
    model, variables, _ = fp32_decoder(directory)
    ids = np.load(directory / "input_ids.npy")
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_bf16_gemma_forward_still_tracks_the_reference():
    """gemma3-tiny is the fixture with an attention scale (query_pre_attn_scalar
    16 on head_dim 32), and bf16 is the recipe's default compute dtype. Against
    the fp32 reference logits the observed difference is 5.7e-02 on the
    reference kernel, tolerance 1e-01; dropping the scale moves them by 1.06."""
    directory = FIXTURES / "gemma3-tiny"
    model, variables, _ = load_pretrained_decoder(
        str(directory), dtype='bfloat16', attention_impl='reference')
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(
        variables, jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)),
        np.float32)

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-1, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_tied_head_is_the_embedding_and_untied_heads_load():
    """A tied checkpoint carries lm_head as a copy; the tree keeps one leaf."""
    config = translate_config(fixture_config("qwen3-tiny"))
    embedding = np.arange(256 * 64, dtype=np.float32).reshape(256, 64)
    tensors = {'model.embed_tokens.weight': embedding,
               'lm_head.weight': embedding.copy()}

    tied = translate_weights(tensors, config)['params']
    assert 'lm_head' not in tied
    assert np.array_equal(tied['embed_tokens']['embedding'], embedding)

    untied = translate_weights(tensors, {**config, 'tie_embeddings': False})['params']
    assert np.array_equal(untied['lm_head']['kernel'], embedding.T)

    # Qwen3-0.6B really does ship both, byte for byte identical; a head that
    # is not that copy means the checkpoint is not the tied model it says
    different = {**tensors, 'lm_head.weight': embedding + 1.0}
    with pytest.raises(ValueError, match="not the embedding it claims to copy"):
        translate_weights(different, config)


def test_an_unfamiliar_tensor_name_is_refused():
    config = translate_config(fixture_config("qwen3-tiny"))
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_weights({'model.layers.0.self_attn.rotary_emb.inv_freq':
                           np.zeros((8,), np.float32)}, config)


@pytest.mark.parametrize("name", TINY)
def test_export_round_trips_the_weights_and_the_config(name, tmp_path):
    model, variables, _ = fp32_decoder(FIXTURES / name)
    export = tmp_path / name

    save_pretrained_decoder(model, variables, export, tokenizer_name="byte")
    again, reloaded, _ = fp32_decoder(export)

    # against the fixture's config, not the exported one read twice: a field
    # the export changes and the model does not read back (the context length,
    # the hidden_act spelling) shows up here
    assert (translate_config(json.loads((export / "config.json").read_text()))
            == translate_config(fixture_config(name)))
    assert again == model, "the exported config rebuilds a different model"
    for path, leaf in flat_tree(reloaded['params']).items():
        assert np.array_equal(np.asarray(leaf),
                              np.asarray(flat_tree(variables['params'])[path])), path
    generation = json.loads((export / "generation_config.json").read_text())
    assert generation['tokenizer_name'] == "byte"


def biased_qwen3(rng):
    """The qwen3-tiny shape with the q/k/v/o biases its own config leaves off.

    nn.Dense starts a bias at zero, and zeros would let a mismapped bias name
    through the round-trip, so every bias gets its own draw.
    """
    config = {**translate_config(fixture_config("qwen3-tiny")), 'attention_bias': True}
    built = with_precision('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = models.build('causal_transformer', **built)
    leaves, structure = jax.tree_util.tree_flatten_with_path(
        model.init(rng, jnp.zeros((1, 4), jnp.int32)))
    return model, jax.tree_util.tree_unflatten(structure, [
        jax.random.normal(jax.random.fold_in(rng, index), leaf.shape, leaf.dtype) * 0.2
        if path[-1].key == 'bias' else leaf
        for index, (path, leaf) in enumerate(leaves)])


def test_a_biased_qwen3_round_trips_through_an_export(tmp_path, rng):
    """attention_bias is one flag for all four projections, which is what
    Qwen3Attention builds from config.attention_bias (modeling_qwen3.py:225-236)
    and Gemma3Attention from the same field (modeling_gemma3.py:322-333), so a
    biased qk_norm model is exportable rather than a refusal."""
    model, variables = biased_qwen3(rng)
    export = tmp_path / "biased"

    save_pretrained_decoder(model, variables, export)
    again, reloaded, _ = fp32_decoder(export)

    assert json.loads((export / "config.json").read_text())['attention_bias'] is True
    assert again == model, "the exported config rebuilds a different model"
    for path, leaf in flat_tree(reloaded['params']).items():
        assert np.array_equal(np.asarray(leaf),
                              np.asarray(flat_tree(variables['params'])[path])), path


def test_the_real_checkpoints_tensor_table_matches_the_built_tree(rng):
    """No weights: the 311 names and shapes of Qwen3-0.6B against our tree.

    This is the check that a config translation and a key map stay honest
    about a checkpoint nobody wants to download in CI.
    """
    from dew.interop.hf_decoders import _dew_path

    table = json.loads((REAL / "tensors.json").read_text())
    config = translate_config(json.loads((REAL / "config.json").read_text()))
    built = with_precision('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = models.build('causal_transformer', **built)

    expected = {}
    for name, info in table['tensors'].items():
        path = _dew_path(name, config)
        if path is None:
            continue  # the tied lm_head copy
        shape = tuple(info['shape'])
        expected['.'.join(path)] = (tuple(reversed(shape))
                                    if path[-1] == 'kernel' else shape)

    tree = jax.eval_shape(
        lambda: model.init(rng, jnp.zeros((1, 4), jnp.int32))['params'])
    initialised = {path: leaf.shape for path, leaf in flat_tree(tree).items()}

    assert initialised == expected
    assert len(expected) == len(table['tensors']) - 1, "the tied head was not skipped"


def qwen3_is_available() -> bool:
    if os.environ.get("DEW_NETWORK_TESTS") == "1":
        return True
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache("Qwen/Qwen3-0.6B", "model.safetensors")
    return isinstance(cached, str)


@pytest.mark.network
@pytest.mark.skipif(not qwen3_is_available(),
                    reason="Qwen3-0.6B is neither cached nor DEW_NETWORK_TESTS=1")
def test_qwen3_0_6b_matches_the_reference_on_the_real_weights():
    """The real thing: 28 layers of Qwen3-0.6B, fp32, against torch's top 32."""
    prompt = json.loads((REAL / "prompt.json").read_text())
    reference = np.load(REAL / "reference.npz")
    ids = np.asarray(prompt['input_ids'], np.int32)[None]

    model, variables, _ = load_pretrained_decoder(
        prompt['repo'], dtype='float32', attention_impl='reference',
        max_seq_len=int(ids.shape[1]))
    logits = np.asarray(model.apply(variables, jnp.asarray(ids)), np.float32)[0]

    assert np.array_equal(np.argmax(logits, axis=-1), reference['argmax'])
    ours = np.take_along_axis(logits, reference['top_ids'], axis=-1)
    difference = float(np.max(np.abs(ours - reference['top_logits'])))
    assert difference < 5e-3, f"max |top-32 logit difference| {difference:.3e}"


def transformers_logits(export, ids, tmp_path):
    """What transformers computes for `ids` on the checkpoint at `export`."""
    script = """
import sys
import numpy as np, torch
from transformers import AutoModelForCausalLM
directory, ids_path, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
model.eval()
model.set_attn_implementation("eager")
with torch.no_grad():
    logits = model(input_ids=torch.from_numpy(np.load(ids_path).astype(np.int64))).logits
np.save(out, logits.to(torch.float32).numpy())
"""
    ids_path, out = tmp_path / "ids.npy", tmp_path / "theirs.npy"
    np.save(ids_path, ids)
    subprocess.run([str(TORCH_VENV), "-c", script, str(export), str(ids_path), str(out)],
                   check=True, capture_output=True)
    return np.load(out)


@pytest.mark.skipif(not TORCH_VENV.exists(),
                    reason="no torch venv at /tmp/hfref to load the export with")
def test_our_export_loads_in_transformers_with_the_same_logits(tmp_path):
    """The export is a real HF checkpoint: transformers reads it and agrees."""
    model, variables, _ = fp32_decoder(FIXTURES / "qwen3-tiny")
    export = tmp_path / "exported"
    save_pretrained_decoder(model, variables, export)

    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(transformers_logits(export, ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


@pytest.mark.skipif(not TORCH_VENV.exists(),
                    reason="no torch venv at /tmp/hfref to load the export with")
def test_a_biased_qwen3_export_carries_its_biases_into_transformers(tmp_path, rng):
    """Qwen3Attention builds q, k, v and o with bias=config.attention_bias
    (modeling_qwen3.py:225-236), so the reference applies the biases where dew
    does and a biased export is a checkpoint it reads."""
    model, variables = biased_qwen3(rng)
    export = tmp_path / "biased"
    save_pretrained_decoder(model, variables, export)

    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(transformers_logits(export, ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"



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

    assert config["num_kv_shared_layers"] == 0
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
    assert config["num_kv_shared_layers"] == 2
    assert config["per_layer_input_dim"] is None


def test_the_e2b_shaped_config_translates_every_gap():
    """The release shape: partial rotary, mixed head dims, a logit cap, the
    double-wide MLP, sharing and per-layer inputs all translate; the cap
    maps to nothing because the text path never reads it."""
    config = translate_config(gemma4_config("gemma4-e2b"))

    assert config["partial_rotary_factor"] == 0.25
    assert "attention_logit_cap" not in config
    assert config["use_double_wide_mlp"]
    assert config["num_kv_shared_layers"] == 2
    assert config["per_layer_input_dim"] == 8
    assert config["v_norm"] and config["rope_theta"] == 1000000.0
    # The full layers' own head dim and the sliding kind's window and base
    assert config["head_dim"] == 16
    assert config["kinds"] == {"sliding_attention": {"window": 32, "rope_theta": 10000.0},
                               "full_attention": {"head_dim": 32}}


def test_the_real_e2b_config_translates():
    """google/gemma-4-E2B's text_config translates field for field: partial
    rotary 0.25, head dims 192 and 512, sharing 20 layers, per-layer inputs
    of 256, the double-wide MLP, scale 1.0 and softcap 30."""
    e2b = Path(
        "/home/mrwhite0racle/.cache/huggingface/hub/models--google--gemma-4-E2B"
        "/snapshots/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f/config.json")
    if not e2b.exists():
        pytest.skip("the E2B config is a local hub cache read, not a download")
    config = translate_config(json.loads(e2b.read_text()).get("text_config"))

    assert config["partial_rotary_factor"] == 0.25
    assert config["head_dim"] == 192
    assert config["kinds"]["full_attention"] == {"head_dim": 512}
    assert config["kinds"]["sliding_attention"] == {"window": 512, "rope_theta": 10000.0}
    assert config["num_kv_shared_layers"] == 20
    assert config["per_layer_input_dim"] == 256
    assert config["use_double_wide_mlp"]
    assert config["attention_scale"] == 1.0
    assert config["final_logit_softcap"] == 30.0
    assert config["rope_theta"] == 1000000.0


@pytest.mark.parametrize("field,value", [
    ("attention_k_eq_v", True),
    ("enable_moe_block", True),
    ("hidden_act", "gelu"),
])
def test_a_gemma4_field_with_no_counterpart_is_refused(field, value):
    """Every gemma4 knob Dew cannot express names itself instead of loading
    a model that computes something else."""
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


@pytest.mark.parametrize("name", GEMMA4 + ("gemma4-e2b",))
def test_gemma4_checkpoints_load_through_the_translator(name):
    """The full load path on a gemma4 checkpoint: translate, weights, build,
    shape check. Sharing layers own no K/V leaves and the per-layer table
    lands, which is what _check_tree enforces leaf for leaf."""
    directory = FIXTURES / name
    model, variables, _ = fp32_decoder(directory)
    assert model.v_norm and model.attention_scale == 1.0
    leaves = flat_tree(variables["params"])
    sharing = {"gemma4-ple": set(), "gemma4-kvshare": {2, 3}, "gemma4-e2b": {4, 5}}[name]
    assert set(model.kv_sharing) == sharing
    for index in sharing:
        assert f"layers_{index}.self_attn.k_proj.kernel" not in leaves
    assert "layers_0.self_attn.k_proj.kernel" in leaves
    if model.per_layer_input_dim:
        assert "embed_tokens_per_layer.embedding" in leaves


@pytest.mark.parametrize("name", GEMMA4 + ("gemma4-e2b",))
def test_gemma4_logits_match_the_reference_implementation(name):
    """Full-model parity, fully live on both branches. Largest observed max
    |logit difference| on CPU: gemma4-ple 4.9e-07, gemma4-kvshare 8.6e-07,
    gemma4-e2b 1.4e-06."""
    directory = FIXTURES / name
    model, variables, _ = fp32_decoder(directory)
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


def test_sharing_without_a_provider_and_sharing_everything_are_refused(rng):
    config = translate_config(gemma4_config("gemma4-kvshare"))
    base = with_precision("causal_transformer", config,
                          dtype="float32", attention_impl="xla")
    with pytest.raises(ValueError, match="no earlier full_attention layer"):
        models.build("causal_transformer", **{**base, "num_kv_shared_layers": 3}).kv_sharing
    with pytest.raises(ValueError, match="leave a provider"):
        models.build("causal_transformer", **{**base, "num_kv_shared_layers": 4}).kv_sharing


def test_the_features_leave_a_plain_tree_unchanged(rng):
    """Off by default: no PLE leaves, no missing K/V, same leaves as before."""
    config = translate_config(gemma4_config("gemma4-ple"))
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**config, "per_layer_input_dim": None,
                               "num_kv_shared_layers": 0, "v_norm": False},
        dtype="float32", attention_impl="xla"))
    assert model.kv_sharing == {}
    flat = flat_tree(model.init(rng, jnp.ones((1, 4), jnp.int32))["params"])
    assert not [name for name in flat if "per_layer" in name]
    assert "layers_0.self_attn.k_proj.kernel" in flat


def test_new_leaves_are_declared_or_heuristic(rng):
    """The coverage sweep builds default configs only, so the new leaves are
    asserted here: the packed table and the projections are declared, the
    scalar norms fall under rank one, and the values norm holds no weight."""
    from dew.nn.sharding import declared_axes, is_heuristic

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
        if declared_axes(path, leaf.ndim) is None and not is_heuristic(path):
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

    cache = model.apply(params, 1, method=type(model).init_cache, mutable=["cache"])[1]["cache"]
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


def test_export_refuses_the_new_features(tmp_path, rng):
    """The three exported families have neither, so a model with any of them
    set is refused instead of silently dropping its leaves."""
    config = translate_config(gemma4_config("gemma4-kvshare"))
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="xla"))
    variables = model.init(rng, jnp.ones((1, 4), jnp.int32))
    with pytest.raises(ValueError, match="num_kv_shared_layers"):
        save_pretrained_decoder(model, variables, str(tmp_path))
