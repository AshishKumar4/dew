"""K2 Horizon's dense decoder against its own remote code:
tools/k2_horizon_reference.py writes tests/fixtures/hf/k2-horizon-tiny and pins
the released configs."""

import json
from pathlib import Path

import numpy as np
import pytest
from interop_support import assert_reloads, assert_same_stored_tensors
from reference_error import as_given, assert_as_exact

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'k2-horizon-tiny'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def fixture(name):
    return np.load(TINY / f'{name}.npy')


def test_the_loaded_model_norms_over_two_groups_of_the_width():
    """The fixture's norms each split the width into two groups, which the shared reference flow's
    logits then hold to the remote code (tests/test_hf_reference_decoders.py)."""
    assert {name: spec.groups for name, spec in load().model.norms.items()} == dict.fromkeys(
        ('input_layernorm', 'post_attention_layernorm', 'norm'), 2)


def test_a_grouped_model_saves_in_the_remote_codes_layout(tmp_path):
    """The checkpoint writes back bit for bit under a config naming the remote
    code, and reloads to the same logits."""
    loaded = load()
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['model_type'] == 'k2_horizon' and config['layernorm_num_groups'] == 2
    assert config['num_experts'] == 0 and config['query_key_norm'] is False
    assert config['auto_map']['AutoModelForCausalLM'].endswith('modeling_k2_horizon.K2HorizonForCausalLM')
    assert_same_stored_tensors(load_shards(tmp_path), load_shards(TINY))
    assert_reloads(loaded, tmp_path, fixture('input_ids'))


YARN_EXPORT_SEEDS = 8
"""The single-group export's declared draws of ids against transformers' Llama read of it (check 1),
sized by `seeds_for` paired with Dew's logits on the CPU."""


def test_a_single_group_exports_as_llama_with_its_yarn(tmp_path):
    """The 0.9B's shape, one group under YaRN, exports as Llama, whose norm
    is the same product in fp32 and whose rotary builds the same YaRN:
    transformers' LlamaForCausalLM reads it back on par with its fp32 forward
    from its float64 one over drawn ids (check 1)."""
    from tools.decoder_export_reference import Case, reference_forward

    loaded = load()
    single = loaded.model.clone(norms=None)
    PretrainedDecoder.from_model(single, loaded.variables).save(tmp_path)
    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['model_type'] == 'llama'
    shape = np.shape(fixture('input_ids'))
    case = Case('llama', 'k2-horizon-tiny')

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, config['vocab_size'], shape).astype(np.int32),)

    def read(wide):
        forward = reference_forward(case, tmp_path, wide=wide)
        return as_given(lambda ids: forward(np.asarray(ids)))

    assert_as_exact(lambda ids: single.apply(loaded.variables, ids), read(wide=False), draw,
                    'Llama YaRN export', seeds=YARN_EXPORT_SEEDS, truth_fn=read(wide=True))


@pytest.mark.parametrize('size, groups, layers, width', [('0.9b', 1, 28, 1536), ('3.7b', 2, 36, 2560),
                                                         ('7b', 4, 36, 4096)])
def test_each_release_translates_to_the_dense_block_it_ships(size, groups, layers, width):
    """Every released config reads whole: grouped norms that scale before
    their cast, untied heads, and the 0.9B's YaRN with its own betas."""
    fields = translate_config(json.loads((ROOT / f'k2-horizon-{size}' / 'config.json').read_text()))
    stated = {spec['groups'] for spec in (fields.get('norms') or {'': {'groups': 1}}).values()}
    assert (stated, fields['num_layers'], fields['emb_features']) == ({groups}, layers, width)
    assert fields['scale_after_cast'] is False and fields['tie_embeddings'] is False
    assert fields.get('mixture') is None and fields['qk_norm'] is False
    if size == '0.9b':
        yarn = fields['yarn']
        assert (yarn['factor'], yarn['beta_fast'], yarn['beta_slow']) == (16.0, 128.0, 4.0)
        assert yarn['truncate']
        assert yarn['attention_factor'] == pytest.approx(1.2772588722239782)
    else:
        assert fields.get('yarn') is None and fields['rope_theta'] == 1e7
