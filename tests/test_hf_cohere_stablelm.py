"""StableLM's biased LayerNorm block with per-head query and key norms, and
Cohere's and Cohere2's one-norm parallel block with adjacent-pair rotary and
a scaled head, against transformers 5.16.1."""

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards
from dew.nn.mixers import AttentionMixer

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
FIXTURES = ['stablelm-tiny', 'stablelm-parallel-tiny', 'cohere-tiny', 'cohere2-tiny']


def load(name):
    return Pretrained.load(ROOT / name, dtype='float32', attention_impl='reference')


def assert_another_model(name, logits, label):
    directory = ROOT / name
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(logits, np.load(directory / 'logits.npy'),
                                         np.load(directory / 'logits_f64.npy'), label)


@pytest.mark.parametrize('name', FIXTURES)
def test_padded_logits_and_cached_generation_match_transformers(name):
    from tools.classic_gpt_reference import greedy

    directory = ROOT / name
    loaded = load(name)
    ids = np.load(directory / 'padded_ids.npy')
    mask = np.load(directory / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(directory / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], f'{name} padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(directory / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(directory / 'generated.npy'))


@pytest.mark.parametrize('name', FIXTURES)
def test_export_reads_in_transformers_and_reloads_bit_for_bit(name, tmp_path):
    """Each family writes its own tensors back unchanged, StableLM's head
    norms one tensor a head, and the export reads in transformers to the
    same logits."""
    import torch
    from transformers import AutoModelForCausalLM

    directory = ROOT / name
    loaded = load(name)
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    source = json.loads((directory / 'config.json').read_text())
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == source['model_type']
    original, exported = load_shards(directory), load_shards(tmp_path)
    assert set(original) == set(exported)
    for tensor in original:
        np.testing.assert_array_equal(original[tensor], exported[tensor])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(directory / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    truth = np.load(directory / 'logits_f64.npy')
    assert_as_exact_as_the_reference(actual, expected, truth, f'{name} export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_the_reference_catches_stablelm_head_norms_in_another_order():
    """Each query and key head has its own norm scale, so the heads' scales
    shifted by one compute another model."""
    loaded = load('stablelm-parallel-tiny')
    params = dict(loaded.variables['params'])
    for index in range(loaded.model.num_layers):
        layer = dict(params[f'layers_{index}'])
        attention = dict(layer['self_attn'])
        attention['q_norm'] = {'scale': jnp.roll(attention['q_norm']['scale'], 1, axis=0)}
        layer['self_attn'] = attention
        params[f'layers_{index}'] = layer
    wrong = loaded.model.apply({'params': params}, np.load(ROOT / 'stablelm-parallel-tiny' / 'input_ids.npy'))
    assert_another_model('stablelm-parallel-tiny', wrong, 'StableLM rolled head norms')


@pytest.mark.parametrize('name', ['cohere-tiny', 'cohere2-tiny'])
def test_the_reference_catches_cohere_turning_halves(name):
    """Cohere's rotary turns adjacent channel pairs; turning each channel
    with its partner half a head away computes another model."""
    loaded = load(name)
    model = loaded.model
    assert isinstance(model.mixer, AttentionMixer)
    kinds = {kind: dataclasses.replace(spec, mixer=None if spec.mixer is None
                                       else dataclasses.replace(spec.mixer, rotary_pairs='half'))
             for kind, spec in (model.kinds or {}).items()}
    halves = model.clone(mixer=dataclasses.replace(model.mixer, rotary_pairs='half'), kinds=kinds)
    wrong = halves.apply(loaded.variables, np.load(ROOT / name / 'input_ids.npy'))
    assert_another_model(name, wrong, f'{name} rotating halves')


def test_the_reference_catches_cohere2_rotating_its_full_layer():
    """Cohere2's full layer leaves its heads unrotated."""
    loaded = load('cohere2-tiny')
    model = loaded.model
    rotated = model.clone(kinds={kind: dataclasses.replace(spec, mixer=None)
                                 for kind, spec in (model.kinds or {}).items()})
    wrong = rotated.apply(loaded.variables, np.load(ROOT / 'cohere2-tiny' / 'input_ids.npy'))
    assert_another_model('cohere2-tiny', wrong, 'Cohere2 with a rotated full layer')


def released(name):
    return translate_config(json.loads((ROOT / name / 'config.json').read_text())).value


def test_the_released_configs_translate_their_computation():
    small = released('stablelm-2-1_6b')
    assert (small.emb_features, small.num_heads, small.kv_heads, small.num_layers) == (2048, 32, 32, 24)
    assert small.partial_rotary_factor == 0.25 and small.attention_bias and small.o_proj_bias is False
    assert not small.qk_norm and not small.parallel_residual and small.norm_bias
    large = released('stablelm-2-12b')
    assert (large.emb_features, large.num_heads, large.kv_heads, large.num_layers) == (5120, 32, 8, 40)
    assert large.qk_norm and large.qk_norm_scope == 'head_layernorm'
    assert large.parallel_residual and large.shared_parallel_norm and not large.attention_bias
    command_r = released('command-r-08-2024')
    assert (command_r.emb_features, command_r.num_heads, command_r.kv_heads) == (8192, 64, 8)
    assert command_r.logits_scaling == 16 and command_r.rope_theta == 4_000_000 and command_r.tie_embeddings
    assert command_r.mixer == AttentionMixer(rotary_pairs='adjacent') and not command_r.qk_norm
    command_a = released('command-a-03-2025')
    assert command_a.per_layer_types == ('sliding_attention',) * 3 + ('full_attention',) + (
        'sliding_attention',) * 3 + ('full_attention',) + command_a.per_layer_types[8:]
    assert command_a.per_layer_types.count('full_attention') == 16
    assert command_a.kind_of('sliding_attention').window == 4096
    assert command_a.kind_of('full_attention').mixer == AttentionMixer(rotary_pairs='adjacent', nope=True)
    assert command_a.logits_scaling == 4 and command_a.features_per_head == 128


@pytest.mark.network
@pytest.mark.parametrize('name', ['stablelm-2-1_6b', 'stablelm-2-12b', 'command-r-08-2024',
                                  'command-a-03-2025'])
def test_the_released_configs_are_pinned(name):
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / name / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    pinned = {key: value for key, value in json.loads(Path(filename).read_text()).items()
              if not key.startswith('unsloth')}
    assert pinned == json.loads((ROOT / name / 'config.json').read_text())
