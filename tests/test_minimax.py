"""MiniMax-Text-01 and MiniMax-M1 (ledger 850): tools/minimax_reference.py writes minimax-tiny from the
release's own remote code, whose logits tests/test_hf_reference_decoders.py holds Dew's to; this file
holds the family's own checks: both spellings of a release read as one model, its export is refused
where no other runtime would compute it, and a config Dew would compute otherwise is refused.
tests/test_mixers.py runs the lightning layer among the registry's mixers."""

import json
from pathlib import Path

import pytest

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config

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


def test_a_minimax_model_writes_no_checkpoint_another_runtime_would_misread(tmp_path):
    """No runtime outside Dew computes MiniMax as its release does (transformers' MiniMax turns every
    dimension and scales the lightning rates otherwise; the release's code does not import under
    transformers 5), so minimax-tiny's export is refused rather than written as a checkpoint that would
    open as another model."""
    loaded = load()
    with pytest.raises(ValueError):
        PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)


@pytest.mark.parametrize('field, value', [('mlp_alpha_factor', 2.0), ('shared_intermediate_size', 8),
                                          ('postnorm', False)])
def test_a_config_dew_would_compute_otherwise_is_refused(field, value):
    """A sublayer joining at a factor of its own, a shared expert beside the routed ones, and a
    transformers config without the postnorm block, which transformers' MiniMax would build anyway,
    are each refused."""
    with pytest.raises(ValueError):
        translate_config({**released('minimax-text-01-hf'), field: value})
