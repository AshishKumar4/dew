"""MiniMax-Text-01 and MiniMax-M1 (ledger 850): tools/minimax_reference.py writes minimax-tiny from the
release's own remote code, whose logits tests/test_hf_reference_decoders.py holds Dew's to; this file
holds the family's own checks: both spellings of a release read as one model, the checkpoint writes back
as it was read, and a config Dew would compute otherwise is refused. tests/test_mixers.py runs the
lightning layer among the registry's mixers."""

import json
from pathlib import Path

import numpy as np
import pytest
from interop_support import assert_reloads, assert_same_stored_tensors

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'minimax-tiny'


def released(name: str) -> dict:
    return json.loads((ROOT / name / 'config.json').read_text())


def load() -> Pretrained:
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def test_a_releases_own_config_and_its_transformers_config_read_as_one_model():
    """MiniMax-Text-01's own config (`attn_type_list`, `layernorm_*`, remote code) and the config its
    -hf release states for transformers (`layer_types`, `*_factor`) translate to the same model, its
    lightning layers each a kind of its own."""
    own, hf = (translate_config(released(name)) for name in ('minimax-text-01', 'minimax-text-01-hf'))
    assert own == hf
    lightning = [kind for kind in own['layer_types'] if kind != 'full_attention']
    assert len(set(lightning)) == len(lightning) == 70


def test_the_checkpoint_writes_back_under_the_releases_names_and_reloads(tmp_path):
    """minimax-tiny writes back as a `minimax` config with its pattern, factors and rotary, every tensor
    bit for bit under the release's names (`block_sparse_moe`, `qkv_proj`, `output_gate`), and reloads
    to the same logits."""
    loaded = load()
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    config = json.loads((tmp_path / 'config.json').read_text())
    tiny = released('minimax-tiny')
    assert config['model_type'] == 'minimax'
    kinds = ('linear_attention', 'full_attention')
    assert config['layer_types'] == [kinds[kind] for kind in tiny['attn_type_list']]
    assert config['mlp_alpha_factor'] == tiny['layernorm_mlp_alpha']
    assert config['rotary_dim'] == tiny['rotary_dim']
    assert_same_stored_tensors(load_shards(tmp_path), load_shards(TINY))
    assert_reloads(loaded, tmp_path, np.load(TINY / 'input_ids.npy'))


@pytest.mark.parametrize('field, value', [('mlp_alpha_factor', 2.0), ('shared_intermediate_size', 8),
                                          ('postnorm', False)])
def test_a_config_dew_would_compute_otherwise_is_refused_by_name(field, value):
    """A sublayer joining at a factor of its own, a shared expert beside the routed ones, and a
    transformers config without the postnorm block, which transformers' MiniMax would build anyway,
    are each refused by name."""
    with pytest.raises(ValueError, match=field):
        translate_config({**released('minimax-text-01-hf'), field: value})
