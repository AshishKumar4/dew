"""K2 Horizon's dense decoder against its own remote code:
tools/k2_horizon_reference.py writes tests/fixtures/hf/k2-horizon-tiny and pins
the released configs."""

import json
from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'k2-horizon-tiny'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def fixture(name):
    return np.load(TINY / f'{name}.npy')


def test_grouped_norms_and_yarn_compute_the_remote_codes_logits():
    """Norms over two groups of the width and the 0.9B's YaRN, unpadded and
    left-padded, and the cached greedy continuation."""
    from tools.classic_gpt_reference import greedy

    loaded = load()
    assert loaded.model.norm_groups == 2
    logits = np.asarray(loaded.model.apply(loaded.variables, fixture('input_ids')))
    assert_as_exact_as_the_reference(logits, fixture('logits'), fixture('logits_f64'), 'K2 Horizon logits')
    mask = fixture('attention_mask')
    padded = np.asarray(loaded.model.apply(loaded.variables, fixture('padded_ids'), attention_mask=mask))
    assert_as_exact_as_the_reference(padded[mask], fixture('padded_logits')[mask],
                                     fixture('padded_logits_f64')[mask], 'K2 Horizon padded logits')
    np.testing.assert_array_equal(greedy(loaded, fixture('input_ids')[:, :4], 6), fixture('generated'))


def test_a_grouped_model_saves_in_the_remote_codes_layout(tmp_path):
    """The checkpoint writes back bit for bit under a config naming the remote
    code, and reloads to the same logits."""
    loaded = load()
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['model_type'] == 'k2_horizon' and config['layernorm_num_groups'] == 2
    assert config['num_experts'] == 0 and config['query_key_norm'] is False
    assert config['auto_map']['AutoModelForCausalLM'].endswith('modeling_k2_horizon.K2HorizonForCausalLM')
    original, exported = load_shards(TINY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for tensor in original:
        np.testing.assert_array_equal(original[tensor], exported[tensor])
    ids = fixture('input_ids')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids),
                                  loaded.model.apply(loaded.variables, ids))


@pytest.mark.parametrize('size, groups, layers, width', [('0.9b', 1, 28, 1536), ('3.7b', 2, 36, 2560),
                                                         ('7b', 4, 36, 4096)])
def test_each_release_translates_to_the_dense_block_it_ships(size, groups, layers, width):
    """Every released config reads whole: grouped norms that scale before
    their cast, untied heads, and the 0.9B's YaRN with its own betas."""
    fields = translate_config(json.loads((ROOT / f'k2-horizon-{size}' / 'config.json').read_text()))
    assert (fields['norm_groups'], fields['num_layers'], fields['emb_features']) == (groups, layers, width)
    assert fields['scale_after_cast'] is False and fields['tie_embeddings'] is False
    assert fields['mixture'] is None and fields['qk_norm'] is False
    if size == '0.9b':
        yarn = fields['yarn']
        assert (yarn['factor'], yarn['beta_fast'], yarn['beta_slow'], yarn['truncate']) == (16.0, 128.0, 4.0, True)
        assert yarn['attention_factor'] == pytest.approx(1.2772588722239782)
    else:
        assert fields.get('yarn') is None and fields['rope_theta'] == 1e7
