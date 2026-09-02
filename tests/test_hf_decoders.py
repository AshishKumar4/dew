"""Hugging Face decoder checkpoints, loaded into CausalTransformer and back out.

The claim these tests defend is parity, not plausibility: the same weights and
the same token ids through the reference implementation and through dew have to
produce the same logits. tools/hf_reference.py writes the fixtures under torch
and transformers (the reference), including two tiny random-weight checkpoints
whose logits are committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- qwen3-tiny  : max |logit difference| 8.4e-06, tolerance 1e-4
- gemma3-tiny : max |logit difference| 3.4e-06, tolerance 1e-4
- Qwen3-0.6B  : max |top-32 logit difference| 1.4e-04, mean 1.2e-05, tolerance
  5e-3, and the argmax of all 48 positions equal. The larger residue is 28
  layers of a real checkpoint accumulating fp32 rounding, not a different
  computation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.hf_decoders import (
    load_pretrained_decoder, save_pretrained_decoder, translate_config,
    translate_weights,
)
from dew.registry import apply_precision_policy, build_model

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
TINY = ("qwen3-tiny", "gemma3-tiny")
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
        'max_seq_len': 64, 'rope_theta': 1e6, 'rope_local_theta': None,
        'layer_types': ('full_attention', 'full_attention'), 'sliding_window': None,
        'norm_eps': 1e-6, 'qk_norm': True, 'attention_bias': False,
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
    assert config['sliding_window'] == 4
    # rope_local_base_freq for the sliding layer, rope_theta for the full one
    assert (config['rope_theta'], config['rope_local_theta']) == (1e6, 1e4)


def test_a_multimodal_gemma3_config_translates_its_text_config():
    text_config = fixture_config("gemma3-tiny")
    wrapped = {'model_type': 'gemma3', 'text_config': text_config,
               'vision_config': {'hidden_size': 8}, 'mm_tokens_per_image': 256,
               'boi_token_index': 255999, 'eoi_token_index': 256000,
               'image_token_index': 262144}

    assert translate_config(wrapped) == translate_config(text_config)


@pytest.mark.parametrize("field, value, message", [
    ('model_type', 'mamba', "model_type 'mamba'"),
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


@pytest.mark.parametrize("name", TINY)
def test_translated_weights_are_exactly_the_models_param_tree(name, rng):
    """Same paths, same shapes, same dtypes as a freshly initialised model."""
    config = translate_config(fixture_config(name))
    built = apply_precision_policy('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = build_model('causal_transformer', built)
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
    again, reloaded, hf_config = fp32_decoder(export)

    assert translate_config(hf_config) == translate_config(
        json.loads((export / "config.json").read_text()))
    assert again == model, "the exported config rebuilds a different model"
    for path, leaf in flat_tree(reloaded['params']).items():
        assert np.array_equal(np.asarray(leaf),
                              np.asarray(flat_tree(variables['params'])[path])), path
    generation = json.loads((export / "generation_config.json").read_text())
    assert generation['tokenizer_name'] == "byte"


def test_the_real_checkpoints_tensor_table_matches_the_built_tree(rng):
    """No weights: the 311 names and shapes of Qwen3-0.6B against our tree.

    This is the check that a config translation and a key map stay honest
    about a checkpoint nobody wants to download in CI.
    """
    from dew.interop.hf_decoders import _dew_path

    table = json.loads((REAL / "tensors.json").read_text())
    config = translate_config(json.loads((REAL / "config.json").read_text()))
    built = apply_precision_policy('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = build_model('causal_transformer', built)

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


@pytest.mark.skipif(not TORCH_VENV.exists(),
                    reason="no torch venv at /tmp/hfref to load the export with")
def test_our_export_loads_in_transformers_with_the_same_logits(tmp_path):
    """The export is a real HF checkpoint: transformers reads it and agrees."""
    model, variables, _ = fp32_decoder(FIXTURES / "qwen3-tiny")
    export = tmp_path / "exported"
    save_pretrained_decoder(model, variables, export)

    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))
    script = """
import json, sys
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

    difference = float(np.max(np.abs(np.load(out) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
