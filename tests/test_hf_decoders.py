"""Hugging Face decoder checkpoints, loaded into CausalTransformer and back out.

The claim these tests defend is parity, not plausibility: the same weights and
the same token ids through the reference implementation and through dew have to
produce the same logits. tools/hf_reference.py writes the fixtures under torch
and transformers (the reference), including two tiny random-weight checkpoints
whose logits are committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- gemma-tiny  : max |logit difference| 4.77e-06, tolerance 1e-4, logits up to
  7.0; Gemma 1 with hidden_act 'gelu', the erf form.
- gemma2-tiny : max |logit difference| 4.05e-06, tolerance 1e-4, logits up to
  10; alternating sliding and full layers, the sandwich norms without q/k
  norms, and the attention softcap at 5, which moves the logits by 1.45.
- mistral-tiny: max |logit difference| 6.44e-06, tolerance 1e-4.
- qwen2-tiny  : max |logit difference| 8.39e-06, tolerance 1e-4, logits up to
  6.7; biased q/k/v over a bias-free o_proj, with a window from layer 1 on.
- mixtral-tiny: max |logit difference| 2.32e-06, tolerance 1e-4, logits up
  to 4.4 in magnitude; the released per-expert w1/w2/w3 tensors stack.
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
- deepseek-v3-tiny : max |logit difference| 4.4e-06, tolerance 1e-4. MLA
  with q and kv LoRA, the released YaRN spelling, a dense layer over a
  routed one with a shared expert, the group limit and the balancing bias.
- deepseek-v32-tiny: max |logit difference| 3.6e-06, tolerance 1e-4. The
  same over the sparse indexer; the dense mixer on the same weights differs
  from the fixture by 3.8, so the fixture is the sparse model.
- qwen35-tiny : max |logit difference| 9.1e-05, tolerance 5e-4, on logits of
  magnitude 6.8. Three gated delta net layers and one gated attention layer
  with a sliced quarter-head rope; the delta net's own numbers are in
  tests/test_linear_attention.py.
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
from dew.nn.gpt_oss import dequantize_mxfp4
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
TINY = ("qwen3-tiny", "gemma3-tiny", "llama-tiny", "mistral-tiny", "qwen2-tiny",
        "gemma-tiny", "gemma2-tiny")
DEEPSEEK = ("deepseek-v3-tiny", "deepseek-v32-tiny")
ROUTED = DEEPSEEK + ("mixtral-tiny",)
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


def test_released_mistral_v03_config_translates_every_computational_field():
    # v0.3 deliberately disables the window; the tiny fixture exercises it.
    assert translate_config(fixture_config("mistral-7b-v0.3")) == {
        'vocab_size': 32768, 'emb_features': 4096, 'num_layers': 32,
        'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'mlp': 'swiglu',
        'mlp_features': 14336, 'max_seq_len': 8192, 'rope_theta': 1e6,
        'layer_types': ('full_attention',) * 32, 'kinds': {},
        'norm_eps': 1e-5, 'scale_after_cast': True, 'qk_norm': False,
        'attention_bias': False, 'tie_embeddings': False,
    }


def test_released_qwen2_0_5b_config_translates_every_computational_field():
    """Qwen2-0.5B ships use_sliding_window false with a sliding_window of
    131072, so every layer attends the whole sequence."""
    assert translate_config(fixture_config("qwen2-0.5b")) == {
        'vocab_size': 151936, 'emb_features': 896, 'num_layers': 24,
        'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'mlp': 'swiglu',
        'mlp_features': 4864, 'max_seq_len': 8192, 'rope_theta': 1e6,
        'layer_types': ('full_attention',) * 24, 'kinds': {},
        'norm_eps': 1e-6, 'scale_after_cast': True, 'qk_norm': False,
        'attention_bias': True, 'o_proj_bias': False, 'tie_embeddings': True,
    }


def test_qwen2_loads_its_projection_biases_and_no_o_proj_bias():
    """The split dial in the loaded tree: zeroing the q/k/v biases moves the
    reference logits, and o_proj has no bias leaf to zero."""
    model, variables, _ = fp32_decoder(FIXTURES / 'qwen2-tiny')
    ids = np.load(FIXTURES / 'qwen2-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'qwen2-tiny' / 'logits.npy')
    attention = variables['params']['layers_0']['self_attn']
    assert 'bias' not in attention['o_proj']
    unbiased = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if path[-1].key == 'bias' else leaf,
        variables)
    assert np.max(np.abs(np.asarray(model.apply(unbiased, ids)) - reference)) > 0.1


def test_released_mixtral_8x7b_config_translates_every_computational_field():
    config = translate_config(fixture_config("mixtral-8x7b"))
    assert config['mixture'] == {'experts': 8, 'top_k': 2}
    assert config['layer_types'] == ('full_attention',) * 32
    assert (config['emb_features'], config['mlp_features'], config['num_kv_heads']) == (4096, 14336, 8)


def test_mixtral_experts_stack_in_checkpoint_order():
    model, variables, _ = fp32_decoder(FIXTURES / 'mixtral-tiny')
    ids = np.load(FIXTURES / 'mixtral-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'mixtral-tiny' / 'logits.npy')
    params = variables['params']
    experts = params['layers_0']['mlp']['experts']
    swapped = {**experts, 'gate_proj': {'kernel': experts['gate_proj']['kernel'][::-1]}}
    shuffled = {**variables, 'params': {**params, 'layers_0': {**params['layers_0'], 'mlp': {
        **params['layers_0']['mlp'], 'experts': swapped}}}}
    assert np.max(np.abs(np.asarray(model.apply(shuffled, ids)) - reference)) > 0.1


def test_mistral_window_changes_the_reference_logits():
    model, variables, _ = fp32_decoder(FIXTURES / 'mistral-tiny')
    ids = np.load(FIXTURES / 'mistral-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'mistral-tiny' / 'logits.npy')
    unwindowed = model.clone(kinds={}, layer_types=('full_attention',) * model.num_layers)
    difference = np.max(np.abs(np.asarray(unwindowed.apply(variables, ids)) - reference))
    assert difference > 0.1


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


def test_an_attention_softcap_maps_where_the_reference_applies_it():
    """Gemma 2 squashes its attention logits (modeling_gemma2.py:282); Gemma 3
    reads the same field into its attention and never passes it on
    (modeling_gemma3.py:334, :370-379), so there it changes nothing and maps
    to nothing. Llama's reference has no such field at all."""
    assert translate_config(fixture_config("gemma2-tiny"))['attn_logit_softcap'] == 5.0
    ignored = translate_config({**fixture_config("gemma3-tiny"), 'attn_logit_softcapping': 50.0})
    assert 'attn_logit_softcap' not in ignored
    with pytest.raises(ValueError, match="attn_logit_softcapping"):
        translate_config({**fixture_config("llama-tiny"), 'attn_logit_softcapping': 50.0})


def test_the_released_gemma_2b_config_translates_field_by_field():
    """unsloth/gemma-2b carries google/gemma-2b's config: hidden_act 'gelu',
    which transformers 5.16.1 runs as the erf gelu (modeling_gemma.py:93),
    the (1 + w) norms, sqrt(d) embeddings and a tied head."""
    assert translate_config(fixture_config("gemma-2b")) == {
        'vocab_size': 256000, 'emb_features': 2048, 'num_layers': 18,
        'num_heads': 8, 'num_kv_heads': 1, 'head_dim': 256, 'mlp': 'geglu_exact',
        'mlp_features': 16384, 'max_seq_len': 8192, 'rope_theta': 1e4,
        'layer_types': ('full_attention',) * 18, 'kinds': {},
        'norm_eps': 1e-6, 'scale_after_cast': False, 'qk_norm': False,
        'attention_bias': False, 'tie_embeddings': True,
        'scale_offset': True, 'embedding_scale': True,
    }


def test_the_released_gemma_2_2b_config_translates_field_by_field():
    """unsloth/gemma-2-2b carries google/gemma-2-2b's config: 26 layers
    alternating sliding and full at one rope base, query_pre_attn_scalar
    256 on head_dim 256, both softcaps, and the sandwich norms."""
    config = translate_config(fixture_config("gemma-2-2b"))
    assert config == {
        'vocab_size': 256000, 'emb_features': 2304, 'num_layers': 26,
        'num_heads': 8, 'num_kv_heads': 4, 'head_dim': 256, 'mlp': 'geglu',
        'mlp_features': 9216, 'max_seq_len': 8192, 'rope_theta': 1e4,
        'layer_types': ('sliding_attention', 'full_attention') * 13,
        'kinds': {'sliding_attention': {'window': 4096}},
        'norm_eps': 1e-6, 'scale_after_cast': False, 'qk_norm': False,
        'attention_bias': False, 'tie_embeddings': True,
        'scale_offset': True, 'embedding_scale': True, 'sandwich_norms': True,
        'attention_scale': 256 ** -0.5, 'final_logit_softcap': 30.0,
        'attn_logit_softcap': 50.0,
    }


def test_a_gemma2_config_without_layer_types_alternates_like_the_reference():
    """Gemma2Config fills the pattern itself (configuration_gemma2.py:95-98);
    the expected pattern comes from the reference class."""
    from transformers import Gemma2Config

    config = {**fixture_config("gemma2-tiny"), "num_hidden_layers": 5}
    del config["layer_types"]
    reference = Gemma2Config(**{**config, "layer_types": None}).layer_types
    assert reference is not None
    assert translate_config(config)["layer_types"] == tuple(reference)
    assert translate_config(config)["layer_types"][-1] == "sliding_attention"


def test_dropping_the_attention_softcap_breaks_gemma2_parity():
    """The fixture caps at 5 so the tanh moves the logits by 1.45; a load
    that read the cap and applied none would pass no tolerance below that."""
    model, variables, _ = fp32_decoder(FIXTURES / 'gemma2-tiny')
    ids = np.load(FIXTURES / 'gemma2-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'gemma2-tiny' / 'logits.npy')
    uncapped = model.clone(attn_logit_softcap=None)
    assert np.max(np.abs(np.asarray(uncapped.apply(variables, ids)) - reference)) > 1.0


def test_the_erf_gelu_is_not_the_tanh_gelu_on_gemma():
    """gemma-tiny names hidden_act 'gelu'; run through the tanh approximation
    instead, its logits drift 1.7e-03 from the reference, above the 1e-4
    parity tolerance, so the two activations are two mlp values."""
    model, variables, _ = fp32_decoder(FIXTURES / 'gemma-tiny')
    ids = np.load(FIXTURES / 'gemma-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'gemma-tiny' / 'logits.npy')
    approximate = model.clone(mlp='geglu')
    difference = np.max(np.abs(np.asarray(approximate.apply(variables, ids)) - reference))
    assert 1e-4 < difference < 1e-2


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


@pytest.mark.parametrize("name", TINY + ROUTED)
def test_translated_weights_are_exactly_the_models_variables(name, rng):
    """Same collections, same paths, same shapes, same dtypes as a freshly
    initialised model: `params` for every family, and the `moe` collection
    DeepSeek's routers keep their bias in."""
    config = translate_config(fixture_config(name))
    built = with_precision('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = models.build('causal_transformer', **built)
    initialised = flat_tree(model.init(rng, jnp.zeros((1, 4), jnp.int32)))

    from dew.interop.hf_decoders import _load_shards
    loaded = flat_tree(translate_weights(_load_shards(FIXTURES / name), config))

    assert set(loaded) == set(initialised)
    for path, leaf in loaded.items():
        assert leaf.shape == initialised[path].shape, path
        assert leaf.dtype == jnp.float32, path


@pytest.mark.parametrize("name", TINY + ROUTED)
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

    tree = jax.eval_shape(lambda: model.init(rng, jnp.zeros((1, 4), jnp.int32)))
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
    ("hidden_act", "relu"),
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

    # The decoder underneath it still translates, which is what the message says.
    assert translate_config(wrapper["text_config"])["num_kv_shared_layers"] == 2


def test_a_wrapper_shaped_config_of_an_unknown_family_is_refused_as_one():
    """A config with a text_config and a model_type this has never heard of
    is the same shape of thing, so it gets the same answer rather than the
    bare family list."""
    with pytest.raises(ValueError, match="multimodal wrapper"):
        translate_config({"model_type": "someone_elses_vlm",
                          "text_config": gemma4_config("gemma4-ple")})


@pytest.mark.parametrize("field", ["num_global_key_value_heads", "per_layer_config"])
def test_a_per_layer_key_value_head_count_is_refused(field):
    """The reference lets a layer kind carry its own key/value head count
    (Gemma4TextAttention reads layer_config.num_key_value_heads). The
    backbone has one count for the model, so a config that varies it is
    refused by name instead of building a model whose K/V width is wrong and
    failing later on a shape."""
    config = gemma4_config("gemma4-e2b")
    model_kv = config["num_key_value_heads"]
    if field == "num_global_key_value_heads":
        config[field] = model_kv + 1
    else:
        config[field] = {"1": {"head_dim": 32, "num_key_value_heads": model_kv + 1}}

    with pytest.raises(ValueError, match=field) as raised:
        translate_config(config)
    assert "key/value head count" in str(raised.value)
    assert str(model_kv) in str(raised.value)


def test_a_per_layer_count_equal_to_the_models_still_translates():
    """Only a count that differs is a refusal: the reference fills
    per_layer_config with the model's own value for every layer."""
    config = gemma4_config("gemma4-e2b")
    config["per_layer_config"] = {
        "1": {"head_dim": 32, "num_key_value_heads": config["num_key_value_heads"]}}

    assert translate_config(config)["kinds"]["full_attention"] == {"head_dim": 32}


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
    from dew.interop.hf_decoders import _dew_path, _load_shards

    directory = FIXTURES / "deepseek-v3-tiny"
    config = translate_config(fixture_config("deepseek-v3-tiny"))
    tensors = _load_shards(directory)
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

    model, loaded, _ = fp32_decoder(directory)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    zeroed = {**loaded, 'moe': jax.tree.map(jnp.zeros_like, loaded['moe'])}
    moved = float(np.max(np.abs(np.asarray(model.apply(loaded, ids))
                                - np.asarray(model.apply(zeroed, ids)))))
    assert moved > 1.0, moved


def test_a_routed_checkpoint_without_its_bias_is_refused(tmp_path):
    """The tree check holds every collection to account, so a checkpoint
    that drops the balancing bias fails naming the leaf instead of loading
    a router at zeros."""
    from dew.interop.hf_decoders import _load_shards
    from dew.interop.safetensors_io import save_hf_layout

    directory = FIXTURES / "deepseek-v3-tiny"
    tensors = _load_shards(directory)
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
        'kind': 'mla', 'q_lora_rank': 8, 'kv_lora_rank': 8,
        'qk_nope_head_dim': 8, 'qk_rope_head_dim': 8, 'v_head_dim': 8,
        'rope_interleave': True,
        'yarn': {'rope_type': 'yarn', 'rope_theta': 10000.0, 'factor': 40.0,
                 'original_max_position_embeddings': 4096, 'beta_fast': 32.0,
                 'beta_slow': 1.0, 'mscale': 1.0, 'mscale_all_dim': 1.0,
                 'truncate': True, 'attention_factor': None},
        'index_topk': None, 'index_n_heads': None, 'index_head_dim': None,
    }
    assert v3['mixture'] == {
        'experts': 8, 'top_k': 4, 'layers': (1,), 'score_function': 'sigmoid',
        'scaling': 2.5, 'groups': 4, 'groups_per_token': 2, 'bias': True,
        'shared_features': 16, 'expert_features': 16,
    }
    assert v3['head_dim'] == 16 and v3['scale_after_cast'] and not v3['qk_norm']
    assert v3['layer_types'] == ('full_attention', 'full_attention')

    assert v32['mixer'] == {**v3['mixer'], 'index_topk': 4, 'index_n_heads': 8,
                            'index_head_dim': 16}
    assert v32['mixture'] == v3['mixture']
    assert v32['layer_types'] == ('deepseek_sparse_attention',) * 2


def test_the_v32_fixture_is_the_sparse_model():
    """The dense mixer on deepseek-v32-tiny's weights differs from the
    fixture by 3.8, so the parity above covers the indexer's selection; a
    generator that lost the eager mask fold again would fail here."""
    directory = FIXTURES / "deepseek-v32-tiny"
    _, variables, built = fp32_decoder(directory)
    dense = models.build('causal_transformer', **{
        **built, 'mixer': {**built['mixer'], 'index_topk': None,
                           'index_n_heads': None, 'index_head_dim': None}})
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
    mixer is refused naming the mixer, and one with routed experts on
    standard attention naming the mixture, instead of writing a checkpoint
    without their tensors."""
    model, variables, _ = fp32_decoder(FIXTURES / name)
    with pytest.raises(ValueError, match="a mixer other than attention"):
        save_pretrained_decoder(model, variables, str(tmp_path))

    config = translate_config(fixture_config(name))
    routed = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**config, "mixer": None, "head_dim": None,
                               "layer_types": None, "kinds": {}},
        dtype="float32", attention_impl="reference"))
    variables = routed.init(rng, jnp.ones((1, 4), jnp.int32))
    with pytest.raises(ValueError, match="a model with a mixture"):
        save_pretrained_decoder(routed, variables, str(tmp_path))


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
        "kind": "gated_delta_net", "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 12,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4}}}
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
        "kind": "gated_delta_net", "linear_num_key_heads": 16,
        "linear_num_value_heads": 16, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}
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
    from dew.nn.attention import rotary_freqs

    config = translate_config(qwen35_real_config())
    inv_freq = np.load(QWEN35_REAL / "inv_freq.npy")
    assert inv_freq.shape == (32,)
    rot_dim = int(config["head_dim"] * config["partial_rotary_factor"])
    assert rot_dim == 64

    positions = np.arange(5)
    cos, _ = rotary_freqs(jnp.asarray(positions), config["head_dim"], config["rope_theta"],
                          rot_dim=rot_dim, partial_rotary_type=config["partial_rotary_type"])
    assert cos.shape == (5, 32)
    assert float(np.max(np.abs(np.asarray(cos) - np.cos(positions[:, None] * inv_freq[None])))) < 1e-6

    proportional, _ = rotary_freqs(jnp.asarray(positions), config["head_dim"], config["rope_theta"],
                                   rot_dim=rot_dim, partial_rotary_type="proportional")
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
    model, variables, _ = fp32_decoder(directory)
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
    from dew.interop.hf_decoders import _load_shards

    config = translate_config(fixture_config("qwen35-tiny"))
    built = with_precision("causal_transformer", dict(config),
                           dtype="float32", attention_impl="reference")
    model = models.build("causal_transformer", **built)
    expected = flat_tree(model.init(rng, jnp.ones((1, 4), jnp.int32))["params"])
    loaded = flat_tree(translate_weights(
        _load_shards(FIXTURES / "qwen35-tiny"), config)["params"])

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


def test_the_mtp_weights_of_a_qwen35_checkpoint_are_dropped_like_the_reference_drops_them():
    """The released checkpoints carry mtp.* tensors that the reference lists
    in _keys_to_ignore_on_load_unexpected (modeling_qwen3_5.py:807), so no
    reference forward reads them and they map to nothing; any other
    unfamiliar name still raises."""
    config = translate_config(fixture_config("qwen35-tiny"))
    tensors = {"mtp.layers.0.self_attn.q_proj.weight": np.zeros((8, 64), np.float32)}
    assert translate_weights(tensors, config) == {"params": {}}
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_weights({"model.layers.0.linear_attn.nope.weight": np.zeros((4,), np.float32)},
                          config)


def test_export_refuses_the_qwen35_features(tmp_path):
    """Neither the gate, the delta net kind nor the partial rotary has a
    place in the three exported families."""
    model, variables, _ = fp32_decoder(FIXTURES / "qwen35-tiny")
    with pytest.raises(ValueError, match="output_gate"):
        save_pretrained_decoder(model, variables, str(tmp_path))


def test_a_qwen35_checkpoint_decodes_as_it_scores_in_parallel():
    """The loaded weights through the cache: a prefill and single-token
    steps against the parallel forward, every argmax equal. Largest
    observed logit difference 1.3e-05."""
    directory = FIXTURES / "qwen35-tiny"
    model, variables, _ = fp32_decoder(directory, max_seq_len=16)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    full = model.apply(variables, ids)

    cache = model.apply(variables, ids.shape[0], method=type(model).init_cache,
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
# GPT OSS: attention sinks, biased interleaved experts, YaRN over GQA
# --------------------------------------------------------------------------

GPT_OSS = FIXTURES / "gpt-oss-tiny"


def test_gpt_oss_config_translates_field_by_field():
    config = translate_config(fixture_config("gpt-oss-tiny"))

    assert config["mlp"] == "swigluoai" and config["attention_sinks"]
    assert config["layer_types"] == ("sliding_attention", "full_attention")
    assert config["kinds"] == {"sliding_attention": {"window": 4}}
    assert config["mixture"] == {"experts": 4, "top_k": 2}
    assert config["yarn"]["factor"] == 4.0 and config["yarn"]["truncate"] is False
    assert config["rope_theta"] == 150000.0 and config["attention_bias"]
    assert not config["qk_norm"] and not config["scale_after_cast"]


def test_the_real_gpt_oss_20b_config_translates():
    """openai/gpt-oss-20b, released with MXFP4 experts: 24 alternating
    layers, 64 query and 8 key/value heads of 64, 32 experts with 4 per
    token, YaRN factor 32 over 4096 positions, and the sliding window 128."""
    config = translate_config(fixture_config("gpt-oss-20b"))

    assert config["num_layers"] == 24 and config["layer_types"][:2] == (
        "sliding_attention", "full_attention")
    assert (config["num_heads"], config["num_kv_heads"], config["head_dim"]) == (64, 8, 64)
    assert config["mixture"] == {"experts": 32, "top_k": 4}
    assert config["kinds"] == {"sliding_attention": {"window": 128}}
    assert config["yarn"]["factor"] == 32.0
    assert config["yarn"]["original_max_position_embeddings"] == 4096
    assert config["vocab_size"] == 201088 and config["max_seq_len"] == 8192


@pytest.mark.parametrize("field, value, message", [
    ("swiglu_limit", 3.0, "swiglu_limit"),
    ("quantization_config", {"quant_method": "awq"}, "quantization_config"),
    ("output_router_logits", True, "output_router_logits"),
])
def test_a_gpt_oss_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("gpt-oss-tiny"), field: value})


def test_gpt_oss_logits_match_the_reference_implementation():
    """fp32 parity through xla sink attention: tolerance 1e-4, observed
    max |logit difference| 2.5e-06 with identical argmax."""
    model, variables, _ = load_pretrained_decoder(str(GPT_OSS), dtype="float32",
                                                  attention_impl="xla")
    ids = np.load(GPT_OSS / "input_ids.npy")
    reference = np.load(GPT_OSS / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    flat = flat_tree(variables["params"])
    assert flat["layers_0.self_attn.sinks"].shape == (4,)
    assert flat["layers_1.mlp.experts.gate_up_proj_bias"].shape == (4, 128)


def test_gpt_oss_export_round_trips_sinks_and_fused_experts(tmp_path):
    model, variables, _ = load_pretrained_decoder(str(GPT_OSS), dtype="float32",
                                                  attention_impl="xla")
    export = tmp_path / "gpt-oss"
    save_pretrained_decoder(model, variables, export)
    again, reloaded, _ = load_pretrained_decoder(str(export), dtype="float32",
                                                 attention_impl="xla")
    assert again == model
    for path, leaf in flat_tree(reloaded["params"]).items():
        assert np.array_equal(np.asarray(leaf), np.asarray(flat_tree(variables["params"])[path])), path


def test_an_mxfp4_gpt_oss_checkpoint_loads_through_the_dequantization(tmp_path):
    """The released layout: uint8 blocks and scales in place of the expert
    matrices. Loading them must give the logits of the fixture whose
    experts are those exact bf16 values, so the packed checkpoint is
    written from the fixture's own weights."""
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(GPT_OSS / "model.safetensors"))
    packed = {}
    for name, tensor in tensors.items():
        if not (name.endswith("gate_up_proj") or name.endswith("down_proj")):
            packed[name] = tensor
            continue
        rounded = jnp.asarray(tensor, jnp.bfloat16).astype(jnp.float32)
        blocks, scales = quantize_mxfp4(np.asarray(rounded))
        packed[name + "_blocks"], packed[name + "_scales"] = blocks, scales
        tensors[name] = np.asarray(dequantize_mxfp4(jnp.asarray(blocks), jnp.asarray(scales))
                                   .astype(jnp.float32))
    directory = tmp_path / "packed"
    directory.mkdir()
    save_file(packed, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(
        {**fixture_config("gpt-oss-tiny"), "quantization_config": {"quant_method": "mxfp4"}}))

    model, variables, _ = load_pretrained_decoder(str(directory), dtype="float32",
                                                  attention_impl="xla")
    expected = translate_weights(tensors, translate_config(fixture_config("gpt-oss-tiny")))
    for path, leaf in flat_tree(variables["params"]).items():
        assert np.array_equal(np.asarray(leaf), flat_tree(expected["params"])[path]), path


def quantize_mxfp4(matrix):
    """[expert, input, output] fp32 into the released blocks and scales.

    Every value is scaled to a representable E2M1 magnitude by a shared
    power-of-two exponent per 32 inputs, which is what the test needs to
    pin: the sign nibble order and the transpose back to the input axis.
    """
    values = np.ascontiguousarray(matrix.transpose(0, 2, 1))
    groups = values.reshape(values.shape[0], values.shape[1], -1, 32)
    magnitude = np.max(np.abs(groups), axis=-1, keepdims=True)
    exponent = np.where(magnitude > 0, np.floor(np.log2(np.maximum(magnitude, 1e-30))) - 2, 0)
    scaled = groups / np.exp2(exponent)
    table = np.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    codes = np.abs(scaled[..., None] - table).argmin(-1) + 8 * (scaled < 0)
    blocks = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    return blocks, (exponent[..., 0] + 127).astype(np.uint8)


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"),
                    reason="DEW_NETWORK_TESTS=1 downloads openai/gpt-oss-20b")
def test_gpt_oss_20b_matches_transformers_on_the_real_weights():
    """Not yet run: the released MXFP4 checkpoint through the loader against
    transformers' own dequantized bf16 forward at the same prompt. Tolerance
    on the top-32 logits 5e-2 for bf16 accumulation over 24 layers; the
    argmax must agree everywhere. Observed difference: not recorded until
    the test runs on a machine with the download."""
    import torch
    from transformers import AutoTokenizer, GptOssForCausalLM

    prompt = "The Cascade Range runs from northern California through Oregon"
    ids = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")(
        prompt, return_tensors="np")["input_ids"].astype(np.int32)
    model, variables, _ = load_pretrained_decoder(
        "openai/gpt-oss-20b", dtype="bfloat16", attention_impl="xla",
        max_seq_len=int(ids.shape[1]))
    logits = np.asarray(model.apply(variables, jnp.asarray(ids)), np.float32)[0]

    reference = GptOssForCausalLM.from_pretrained("openai/gpt-oss-20b", dtype=torch.bfloat16)
    reference.eval()
    reference.set_attn_implementation("eager")
    with torch.no_grad():
        theirs = reference(input_ids=torch.from_numpy(ids.astype(np.int64))).logits[0].float().numpy()

    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(theirs, axis=-1))
    top_ids = np.argsort(-theirs, axis=-1)[:, :32]
    difference = float(np.max(np.abs(np.take_along_axis(logits, top_ids, axis=-1)
                                     - np.take_along_axis(theirs, top_ids, axis=-1))))
    assert difference < 5e-2, f"max |top-32 logit difference| {difference:.3e}"



# --------------------------------------------------------------------------
# DeepSeek V2 and Kimi K2: the V2 router under MLA, and V3 under Kimi's name
# --------------------------------------------------------------------------

DEEPSEEK_V2 = FIXTURES / "deepseek-v2-tiny"


def test_deepseek_v2_config_translates_field_by_field():
    """The tiny V2: MLA without the query LoRA, YaRN with the 0.707 mscales,
    a dense first layer over a softmax router under group_limited_greedy
    that never renormalises, and two shared experts of width 16."""
    config = translate_config(fixture_config("deepseek-v2-tiny"))

    assert config["mixer"]["kind"] == "mla" and config["mixer"]["q_lora_rank"] is None
    assert config["mixer"]["yarn"]["mscale"] == 0.707
    assert config["mixture"] == {
        "experts": 8, "top_k": 4, "layers": (1,), "scaling": 2.5, "shared_features": 32,
        "expert_features": 16, "score_function": "softmax", "norm_topk_prob": False,
        "groups": 4, "groups_per_token": 2, "group_score": "max"}
    assert config["head_dim"] == 16 and config["scale_after_cast"]


def test_the_real_deepseek_v2_lite_config_translates():
    """deepseek-ai/DeepSeek-V2-Lite: 27 layers with one dense, 64 experts
    with 6 per token under greedy selection, two shared experts of 1408,
    kv_lora_rank 512 with no query LoRA, YaRN factor 40 at mscale 0.707."""
    config = translate_config(fixture_config("deepseek-v2-lite"))

    assert config["num_layers"] == 27 and config["mixture"]["layers"] == tuple(range(1, 27))
    assert config["mixture"]["experts"] == 64 and config["mixture"]["top_k"] == 6
    assert config["mixture"]["shared_features"] == 2816
    assert config["mixture"]["groups"] == 1 and config["mixture"]["group_score"] == "max"
    assert config["mixture"]["norm_topk_prob"] is False
    assert config["mixer"]["kv_lora_rank"] == 512 and config["mixer"]["q_lora_rank"] is None
    assert config["mixer"]["yarn"]["factor"] == 40.0
    assert config["mixer"]["yarn"]["mscale_all_dim"] == 0.707
    assert config["vocab_size"] == 102400 and config["head_dim"] == 192


@pytest.mark.parametrize("field, value, message", [
    ("norm_topk_prob", True, "norm_topk_prob=True"),
    ("topk_method", "noaux_tc", "topk_method 'noaux_tc'"),
    ("scoring_func", "sigmoid", "scoring_func 'sigmoid'"),
])
def test_a_deepseek_v2_field_the_reference_does_not_run_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("deepseek-v2-tiny"), field: value})


def test_deepseek_v2_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 2.3e-06
    with identical argmax."""
    model, variables, _ = fp32_decoder(DEEPSEEK_V2)
    ids = np.load(DEEPSEEK_V2 / "input_ids.npy")
    reference = np.load(DEEPSEEK_V2 / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    assert "moe" not in variables, "V2 keeps no balancing bias"


def test_the_real_kimi_k2_config_translates():
    """moonshotai/Kimi-K2-Instruct is DeepSeek V3's computation: 61 layers
    with one dense, 384 sigmoid-scored experts with 8 per token in one
    group, scaled by 2.827, one shared expert, MLA with the 1536 query LoRA,
    YaRN factor 32 at theta 50000, under a 163840-token vocabulary."""
    config = translate_config(fixture_config("kimi-k2"))

    assert config["vocab_size"] == 163840 and config["num_layers"] == 61
    assert config["mixture"]["experts"] == 384 and config["mixture"]["top_k"] == 8
    assert config["mixture"]["scaling"] == 2.827 and config["mixture"]["bias"]
    assert config["mixture"]["groups"] == 1 and config["mixture"]["shared_features"] == 2048
    assert config["mixer"]["q_lora_rank"] == 1536
    assert config["mixer"]["yarn"]["factor"] == 32.0 and config["rope_theta"] == 50000.0


def test_a_kimi_k2_checkpoint_loads_as_the_deepseek_v3_it_is(tmp_path):
    """The V3 tiny fixture under Kimi's model_type and Kimi's extra fields
    reaches the same logits, and exports back under deepseek_v3."""
    directory = tmp_path / "kimi"
    directory.mkdir()
    for path in (FIXTURES / "deepseek-v3-tiny").iterdir():
        if path.name != "config.json":
            (directory / path.name).write_bytes(path.read_bytes())
    (directory / "config.json").write_text(json.dumps({
        **fixture_config("deepseek-v3-tiny"), "model_type": "kimi_k2",
        "aux_loss_alpha": 0.001, "seq_aux": True, "num_nextn_predict_layers": 0}))

    model, variables, _ = fp32_decoder(directory)
    reference = np.load(FIXTURES / "deepseek-v3-tiny" / "logits.npy")
    ids = np.load(FIXTURES / "deepseek-v3-tiny" / "input_ids.npy")
    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))
    assert float(np.max(np.abs(logits - reference))) < 1e-4
    from dew.interop.hf_decoders import _export_config
    assert _export_config(model)["model_type"] == "deepseek_v3"



# --------------------------------------------------------------------------
# GLM 4.5: a half rotary over biased GQA, DeepSeek V3 routing, an MTP depth
# --------------------------------------------------------------------------

GLM4_MOE = FIXTURES / "glm4-moe-tiny"


def test_glm4_moe_config_translates_field_by_field():
    config = translate_config(fixture_config("glm4-moe-tiny"))

    assert config["attention_bias"] and config["o_proj_bias"] is False
    assert config["qk_norm"] and config["scale_after_cast"]
    assert config["partial_rotary_factor"] == 0.5 and config["partial_rotary_type"] == "default"
    assert config["rope_theta"] == 1e6 and config["head_dim"] == 8
    assert config["mixture"] == {
        "experts": 8, "top_k": 2, "layers": (1,), "scaling": 1.5, "shared_features": 16,
        "expert_features": 16, "score_function": "sigmoid", "groups": 1,
        "groups_per_token": 1, "bias": True}
    assert config["num_nextn_predict_layers"] == 1


def test_the_real_glm_4_5_air_config_translates():
    """zai-org/GLM-4.5-Air: 46 layers with one dense, 128 sigmoid-scored
    experts with 8 per token, one shared expert of 1408, 96 query and 8
    key/value heads of 128 with biased q/k/v and a bias-free o_proj, a half
    rotary at theta 1e6, no q/k norms, and one MTP depth."""
    config = translate_config(fixture_config("glm-4.5-air"))

    assert config["num_layers"] == 46 and config["mixture"]["layers"] == tuple(range(1, 46))
    assert config["mixture"]["experts"] == 128 and config["mixture"]["top_k"] == 8
    assert config["mixture"]["shared_features"] == 1408 and config["mixture"]["bias"]
    assert (config["num_heads"], config["num_kv_heads"], config["head_dim"]) == (96, 8, 128)
    assert config["attention_bias"] and config["o_proj_bias"] is False
    assert not config["qk_norm"]
    assert config["partial_rotary_factor"] == 0.5 and config["rope_theta"] == 1e6
    assert config["num_nextn_predict_layers"] == 1 and config["vocab_size"] == 151552


@pytest.mark.parametrize("field, value, message", [
    ("rope_scaling", {"rope_type": "yarn", "factor": 4.0}, "plain rotary"),
    ("norm_topk_prob", False, "norm_topk_prob=False"),
    ("topk_method", "greedy", "topk_method 'greedy'"),
])
def test_a_glm4_moe_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("glm4-moe-tiny"), field: value})


def test_glm4_moe_logits_match_the_reference_implementation():
    """fp32 parity of the trunk: tolerance 1e-4, observed max |logit
    difference| 3.3e-06 with identical argmax."""
    model, variables, _ = fp32_decoder(GLM4_MOE)
    ids = np.load(GLM4_MOE / "input_ids.npy")
    reference = np.load(GLM4_MOE / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_glm4_moe_mtp_depth_matches_the_reference_composition():
    """The checkpoint's model.layers.2.* depth loads as mtp_0 with the
    trunk's routing and computes what the engines running the released
    weights compute (tools/hf_reference_b.py Glm4MoeMTP): tolerance 1e-4,
    observed max |logit difference| 3.2e-06 with identical argmax. Swapping
    the two norms, the order today's from-scratch code had, disagrees."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    model, variables, _ = fp32_decoder(GLM4_MOE)
    ids = jnp.asarray(np.load(GLM4_MOE / "input_ids.npy"), jnp.int32)
    reference = np.load(GLM4_MOE / "mtp_logits.npy")
    depth = variables["params"]["mtp_0"]
    assert set(depth) == {"enorm", "hnorm", "eh_proj", "block", "final_norm"}
    assert depth["block"]["mlp"]["experts"]["gate_proj"]["kernel"].shape == (8, 32, 16)
    assert "mtp_0" in variables["moe"]

    hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
    logits = np.asarray(model.apply(variables, hidden, ids, method=CausalTransformer.mtp_logits)[0])
    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |mtp logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))

    swapped = {**variables, "params": {**variables["params"], "mtp_0": {
        **depth, "enorm": depth["hnorm"], "hnorm": depth["enorm"]}}}
    other = np.asarray(model.apply(swapped, hidden, ids, method=CausalTransformer.mtp_logits)[0])
    assert float(np.max(np.abs(other - reference))) > 1e-2


def test_a_glm4_moe_depth_with_its_own_head_is_refused(tmp_path):
    """The depth shares the trunk's embedding and head; a checkpoint whose
    copies differ would load as a model computing something else."""
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(GLM4_MOE / "model.safetensors"))
    tensors["model.layers.2.shared_head.head.weight"] = tensors["lm_head.weight"] * 1.5
    directory = tmp_path / "glm"
    directory.mkdir()
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(fixture_config("glm4-moe-tiny")))
    with pytest.raises(ValueError, match="shared_head.head.weight differs from lm_head.weight"):
        fp32_decoder(directory)



# --------------------------------------------------------------------------
# Llama 4 (text): iRoPE with chunked local layers, input-scaled experts
# --------------------------------------------------------------------------

LLAMA4 = FIXTURES / "llama4-tiny"


def test_llama4_config_translates_field_by_field():
    """The tiny text config: three chunked rotated layers and one global
    layer, each kind naming the llama4 mixer under its own rule, the dense
    layers at intermediate_size_mlp and the routed layers 1 and 3 with a
    shared expert at intermediate_size, sigmoid weights on the inputs."""
    config = translate_config(fixture_config("llama4-tiny"))

    assert config["layer_types"] == ("chunked_attention",) * 3 + ("full_attention",)
    rule = {"kind": "llama4", "use_qk_norm": True, "attn_temperature_tuning": True,
            "floor_scale": 4.0, "attn_scale": 0.1}
    assert config["kinds"] == {
        "full_attention": {"mixer": {**rule, "use_rope": False}},
        "chunked_attention": {"mixer": {**rule, "use_rope": True, "attention_chunk_size": 4}}}
    assert config["mixture"] == {
        "experts": 4, "top_k": 2, "score_function": "sigmoid", "norm_topk_prob": False,
        "scale_inputs": True, "expert_features": 48, "shared_features": 48, "layers": (1, 3)}
    assert config["mlp_features"] == 64 and not config["qk_norm"]
    assert config["rope_theta"] == 500000.0 and config["scale_after_cast"]


def test_a_llama4_wrapper_config_is_refused_by_name():
    """meta-llama/Llama-4-Scout-17B-16E's config.json wraps the text decoder
    beside a vision tower; the text_config alone is what translates."""
    with pytest.raises(ValueError, match="model_type 'llama4'"):
        translate_config(fixture_config("llama-4-scout"))


@pytest.mark.parametrize("field, value, message", [
    ("layer_types", ["full_attention"] * 4, "disagrees with no_rope_layers"),
    ("router_jitter_noise", 0.1, "router_jitter_noise"),
    ("output_router_logits", True, "output_router_logits"),
    ("no_rope_layers", [1, 1, 2, 0], "no_rope_layers"),
])
def test_a_llama4_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("llama4-tiny"), field: value})


def test_llama4_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 3.5e-06
    with identical argmax. The fused expert kernels arrive split in place."""
    model, variables, _ = fp32_decoder(LLAMA4)
    ids = np.load(LLAMA4 / "input_ids.npy")
    reference = np.load(LLAMA4 / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    experts = variables["params"]["layers_1"]["mlp"]["experts"]
    assert experts["gate_proj"]["kernel"].shape == (4, 32, 48)
    assert experts["down_proj"]["kernel"].shape == (4, 48, 32)
    assert "shared_experts" in variables["params"]["layers_1"]["mlp"]
    assert "experts" not in variables["params"]["layers_0"]["mlp"]


def test_llama4_export_is_refused_by_name(tmp_path):
    model, variables, _ = fp32_decoder(LLAMA4)
    with pytest.raises(ValueError, match="a mixer other than attention"):
        save_pretrained_decoder(model, variables, str(tmp_path))

