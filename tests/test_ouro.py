"""Ouro's looped decoder against its own remote code on transformers 4.56.2:
tools/ouro_reference.py writes tests/fixtures/hf/ouro-tiny and the 1.4B probe."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards
from dew.nn.backbones.decoder_stack import Loop

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'ouro-tiny'
RELEASED = ROOT / 'ouro-1.4b'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def fixture(name):
    return np.load(TINY / f'{name}.npy')


def released():
    return json.loads((RELEASED / 'config.json').read_text())


def test_padded_logits_and_cached_generation_match_the_remote_code():
    from tools.classic_gpt_reference import greedy

    loaded = load()
    mask = fixture('attention_mask')
    actual = np.asarray(loaded.model.apply(loaded.variables, fixture('padded_ids'), attention_mask=mask))
    expected, truth = fixture('padded_logits'), fixture('padded_logits_f64')
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'Ouro padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    np.testing.assert_array_equal(greedy(loaded, fixture('input_ids')[:, :4], 6), fixture('generated'))


def test_every_pass_gates_its_exit_as_the_remote_code_does():
    loaded = load()
    logits, sown = loaded.model.apply(loaded.variables, fixture('input_ids'), mutable=['exits'])
    assert_as_exact_as_the_reference(logits, fixture('logits'), fixture('logits_f64'), 'Ouro logits')
    gates = np.stack(sown['exits']['logits'])
    assert_as_exact_as_the_reference(gates, fixture('exit_logits'), fixture('exit_logits_f64'), 'Ouro exit gates')


def test_export_writes_the_remote_codes_names_and_reloads_bit_for_bit(tmp_path):
    loaded = load()
    loaded.save(tmp_path)
    config = json.loads((tmp_path / 'config.json').read_text())
    assert (config['model_type'], config['total_ut_steps'], config['early_exit_threshold']) == ('ouro', 3, 1.0)
    assert config['auto_map']['AutoModelForCausalLM'] == 'ByteDance/Ouro-1.4B--modeling_ouro.OuroForCausalLM'
    original, exported = load_shards(TINY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for tensor in original:
        np.testing.assert_array_equal(original[tensor], exported[tensor])
    ids = fixture('input_ids')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids),
                                  loaded.model.apply(loaded.variables, ids))


@pytest.mark.parametrize('label, loop', [('a pass fewer', Loop(2, exit_gate=True)),
                                         ('no norm between passes', Loop(3, step_norm=False, exit_gate=True))])
def test_the_reference_catches_each_mechanism_left_out(label, loop):
    loaded = load()
    logits = loaded.model.clone(loop=loop).apply(loaded.variables, fixture('input_ids'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(logits, fixture('logits'), fixture('logits_f64'), label)


def test_a_loop_the_remote_code_does_not_read_is_not_exported(tmp_path):
    loaded = load()
    trained = dataclasses.replace(loaded, model=loaded.model.clone(loop=Loop(3, backprop_steps=1, exit_gate=True)))
    trained.save(tmp_path / 'truncated')
    assert json.loads((tmp_path / 'truncated' / 'config.json').read_text())['total_ut_steps'] == 3
    unnormed = dataclasses.replace(loaded, model=loaded.model.clone(loop=Loop(3, step_norm=False, exit_gate=True)))
    with pytest.raises(ValueError, match='loop'):
        unnormed.save(tmp_path / 'unnormed')


def test_the_released_config_translates_its_computation():
    model = translate_config(released()).value
    assert model.loop == Loop(4, exit_gate=True)
    assert (model.num_layers, model.emb_features, model.num_heads, model.kv_heads) == (24, 2048, 16, 16)
    assert (model.features_per_head, model.hidden_features, model.vocab_size) == (128, 5632, 49152)
    assert model.sandwich_norms and model.scale_after_cast and not model.qk_norm and not model.attention_bias
    assert (model.rope_theta, model.norm_eps, model.tie_embeddings) == (1e6, 1e-6, False)
    assert set(model.per_layer_types) == {'full_attention'}


@pytest.mark.parametrize('field, value', [('early_exit_threshold', 0.9), ('early_exit_step', 1)])
def test_an_exit_before_the_last_pass_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        translate_config({**released(), field: value})


@pytest.mark.network
def test_the_released_config_is_pinned():
    from huggingface_hub import hf_hub_download

    source = json.loads((RELEASED / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    assert json.loads(Path(filename).read_text()) == released()


@pytest.mark.network
def test_ouro_1_4b_computes_the_remote_codes_logits():
    """ByteDance/Ouro-1.4B in fp32 on CPU, against the remote code's argmax
    and 64 of its logit columns on the probe ids (probe.npz)."""
    source = json.loads((RELEASED / 'source.json').read_text())
    probe = np.load(RELEASED / 'probe.npz')
    loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype='float32')
    logits = np.asarray(loaded.model.apply(loaded.variables, probe['input_ids']), np.float32)
    np.testing.assert_array_equal(logits.argmax(-1), probe['argmax'])
    assert float(np.max(np.abs(logits[..., probe['columns']] - probe['logits']))) < 1e-3
