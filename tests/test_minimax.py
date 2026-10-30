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

from dew.interop import Pretrained, PretrainedDecoder, release_pins
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


CODE = ('configuration_minimax_text_01.py', 'modeling_minimax_text_01.py')
"""The release's own code, which an export writes beside it."""


def test_the_checkpoint_writes_back_as_the_release_beside_its_own_code(tmp_path, monkeypatch):
    """minimax-tiny writes back as the release's own config (`minimax_text_01`: its pattern as
    `attn_type_list`, its factors as `layernorm_*`, its rotary's width) beside the release's own code,
    which computes the partial rotary transformers' MiniMax drops, every tensor bit for bit under the
    release's names (`block_sparse_moe`, `qkv_proj`, `output_gate`), and reloads to the same logits.
    The code is read from the hub's cache, seeded at the release's commit from its pinned copies
    (tools/minimax_reference.py --vendor)."""
    import huggingface_hub.constants
    from tool_support import load as tool

    repo, revision, auto_map = release_pins.MINIMAX_TEXT_01
    snapshot = tmp_path / 'hub' / f"models--{repo.replace('/', '--')}" / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    code = {file: tool('fetched_fixtures').local(ROOT / 'minimax-text-01' / 'release' / file).read_bytes()
            for file in CODE}
    for file, data in code.items():
        (snapshot / file).write_bytes(data)
    monkeypatch.setattr(huggingface_hub.constants, 'HF_HUB_CACHE', str(tmp_path / 'hub'))
    loaded = load()
    export = tmp_path / 'export'
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(export)
    config, tiny = json.loads((export / 'config.json').read_text()), released('minimax-tiny')
    assert (config['model_type'], config['auto_map']) == ('minimax_text_01', auto_map)
    for field in ('attn_type_list', 'postnorm', 'rotary_dim', 'layernorm_full_attention_alpha',
                  'layernorm_linear_attention_beta', 'layernorm_mlp_alpha', 'num_local_experts'):
        assert config[field] == tiny[field], field
    assert {file: (export / file).read_bytes() for file in CODE} == code
    assert_same_stored_tensors(load_shards(export), load_shards(TINY))
    assert_reloads(loaded, export, np.load(TINY / 'input_ids.npy'))


@pytest.mark.parametrize('field, value', [('mlp_alpha_factor', 2.0), ('shared_intermediate_size', 8),
                                          ('postnorm', False)])
def test_a_config_dew_would_compute_otherwise_is_refused(field, value):
    """A sublayer joining at a factor of its own, a shared expert beside the routed ones, and a
    transformers config without the postnorm block, which transformers' MiniMax would build anyway,
    are each refused."""
    with pytest.raises(ValueError):
        translate_config({**released('minimax-text-01-hf'), field: value})
