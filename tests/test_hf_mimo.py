"""MiMo-V2-Flash's sliding layers with doubled key and value heads and sinks,
its narrower scaled values and its grouped sigmoid routing, against
transformers 5.16.1: tools/hf_reference.py writes tests/fixtures/hf/mimo-v2-flash-tiny."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'mimo-v2-flash-tiny'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def assert_another_model(logits, label):
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(logits, np.load(TINY / 'logits.npy'),
                                         np.load(TINY / 'logits_f64.npy'), label)


def released(name):
    return json.loads((ROOT / name / 'config.json').read_text())


def test_padded_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = load()
    ids = np.load(TINY / 'padded_ids.npy')
    mask = np.load(TINY / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(TINY / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'MiMo-V2-Flash padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(TINY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(TINY / 'generated.npy'))


def test_export_reads_in_transformers_and_reloads_bit_for_bit(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = load()
    loaded.save(tmp_path)
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == 'mimo_v2_flash'
    original, exported = load_shards(TINY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for tensor in original:
        np.testing.assert_array_equal(original[tensor], exported[tensor])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    reference.set_experts_implementation('eager')
    ids = np.load(TINY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    truth = np.load(TINY / 'logits_f64.npy')
    assert_as_exact_as_the_reference(actual, expected, truth, 'MiMo-V2-Flash export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def sliding(model, **change):
    """`model` with its sliding kind changed."""
    kinds = dict(model.kinds)
    kinds['sliding_attention'] = dataclasses.replace(kinds['sliding_attention'], **change)
    return model.clone(kinds=kinds)


@pytest.mark.parametrize('label', ['unscaled values', 'sliding layers rotating whole heads',
                                   'sliding layers without sinks'])
def test_the_reference_catches_each_mechanism_left_out(label):
    """The value scale, the sliding layers' partial rotary and their sinks
    each change the fixture's logits past the reference's own error."""
    from flax.traverse_util import flatten_dict, unflatten_dict

    loaded = load()
    model, variables = loaded.model, loaded.variables
    if label == 'unscaled values':
        model = model.clone(value_scale=None)
    elif label == 'sliding layers rotating whole heads':
        model = sliding(model, partial_rotary_factor=None)
    else:
        model = sliding(model, sinks=False)
        variables = unflatten_dict({path: leaf for path, leaf in flatten_dict(variables).items()
                                    if path[-1] != 'sinks'})
    assert_another_model(model.apply(variables, np.load(TINY / 'input_ids.npy')), label)


def test_the_released_configs_translate_their_computation():
    """The release spells the layout in the authors' own fields, which
    transformers derives from its defaults instead; they read as that model."""
    model = translate_config(released('mimo-v2-flash')).value
    assert model.per_layer_types[:6] == ('full_attention',) + ('sliding_attention',) * 4 + ('full_attention',)
    assert model.per_layer_types.count('full_attention') == 9
    full, sliding = model.kind_of('full_attention'), model.kind_of('sliding_attention')
    assert (full.num_kv_heads, sliding.num_kv_heads, sliding.window) == (4, 8, 128)
    assert (full.rope_theta, sliding.rope_theta) == (5_000_000, 10_000)
    assert full.partial_rotary_factor == sliding.partial_rotary_factor == 0.334
    assert sliding.sinks and not full.sinks
    assert (model.features_per_head, model.value_head_dim, model.value_scale) == (192, 128, 0.707)
    assert model.mixture.experts == 256 and model.mixture.layers == tuple(range(1, 48))
    assert model.mixture.score_function == 'sigmoid' and model.mixture.bias
    assert model.mixture.shared_features == 0


@pytest.mark.parametrize('field, value', [('hybrid_layer_pattern', [1] * 48),
                                          ('swa_num_key_value_heads', 4),
                                          ('add_full_attention_sink_bias', True)])
def test_an_authors_field_stating_another_model_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        translate_config({**released('mimo-v2-flash'), field: value})


@pytest.mark.network
@pytest.mark.parametrize('name', ['mimo-v2-flash', 'mimo-v2-flash-base'])
def test_the_released_configs_are_pinned(name):
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / name / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    assert json.loads(Path(filename).read_text()) == released(name)
