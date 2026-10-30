"""MiniMax-Text-01 and MiniMax-M1 (ledger 850): tools/minimax_reference.py writes minimax-tiny from the
release's own remote code, whose logits tests/test_hf_reference_decoders.py holds Dew's to; this file
holds the family's own checks: both spellings of a release read as one model, a model writes back as a
`minimax` config where transformers computes it as Dew does and is refused where it would not, and a
config Dew would compute otherwise is refused. tests/test_mixers.py runs the
lightning layer among the registry's mixers."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import from_record

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


def test_the_releases_geometry_writes_no_config_transformers_would_misread(tmp_path):
    """minimax-tiny turns half of each head over six heads, as the releases turn half of each of 64,
    which transformers' MiniMax does not compute (it turns every dimension, its slopes a power of two's):
    its export is refused rather than written as a `minimax` config that would open as another model."""
    loaded = load()
    with pytest.raises(ValueError):
        PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)


EXPORT_SEEDS = 8
"""The export's declared draws of ids against transformers' MiniMax read of it (check 1), sized by
`seeds_for` paired with Dew's logits on the CPU."""


def test_a_model_transformers_computes_writes_back_as_minimax_and_reads_on_par(tmp_path):
    """A MiniMax of four heads turning every dimension, minimax-tiny's config otherwise, at Dew's own
    initialization, writes back as a `minimax` config with every tensor under the releases' names
    (`block_sparse_moe`, `qkv_proj`, `output_gate`), which transformers' MiniMaxForCausalLM reads on par
    with Dew from its float64 forward over drawn ids (check 1)."""
    from tools.decoder_export_reference import Case, reference_logits

    config = {**released('minimax-tiny'), 'num_attention_heads': 4, 'rotary_dim': 8}
    model = from_record(CausalTransformer, {**translate_config(config), 'attention_impl': 'reference'})
    ids = np.load(TINY / 'input_ids.npy')
    variables = model.init(jax.random.key(0), ids)
    PretrainedDecoder.from_model(model, variables).save(tmp_path)
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == 'minimax'
    assert set(load_shards(tmp_path)) == set(load_shards(TINY))
    case = Case('minimax', 'minimax-tiny',
                derived=('slope_rate', 'query_decay', 'key_decay', 'diagonal_decay'))

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, config['vocab_size'], ids.shape).astype(np.int32),)

    def read(wide):
        return as_given(lambda ids: reference_logits(case, tmp_path, np.asarray(ids), wide=wide))

    assert_as_exact(lambda ids: model.apply(variables, ids), read(wide=False), draw, 'MiniMax export',
                    seeds=EXPORT_SEEDS, truth_fn=read(wide=True))


@pytest.mark.parametrize('field, value', [('mlp_alpha_factor', 2.0), ('shared_intermediate_size', 8),
                                          ('postnorm', False)])
def test_a_config_dew_would_compute_otherwise_is_refused(field, value):
    """A sublayer joining at a factor of its own, a shared expert beside the routed ones, and a
    transformers config without the postnorm block, which transformers' MiniMax would build anyway,
    are each refused."""
    with pytest.raises(ValueError):
        translate_config({**released('minimax-text-01-hf'), field: value})
