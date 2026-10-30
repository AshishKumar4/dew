"""MiniMax-Text-01 and MiniMax-M1 (ledger 850): tools/minimax_reference.py writes minimax-tiny from the
release's own remote code, whose logits tests/test_hf_reference_decoders.py holds Dew's to; this file
holds the family's own checks: both spellings of a release read as one model, its export is refused
where no other runtime would compute it, and a config Dew would compute otherwise is refused.
tests/test_mixers.py runs the lightning layer among the registry's mixers."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import token_draw
from model_support import stepped_decode
from reference_error import assert_as_exact, widened

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
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
    -hf release states for transformers (`layer_types`, `*_factor`) translate to the same model, whose
    80 layers a decode step scans as one body of the release's period of eight (ledger 2564): its
    lightning layers one kind, each one's rates data its place in the stack hands it."""
    own, hf = (translate_config(released(name)) for name in ('minimax-text-01', 'minimax-text-01-hf'))
    assert own == hf
    # The release's pattern and blocks at a width a trace holds.
    narrow = {'hidden_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 8,
              'rotary_dim': 4, 'intermediate_size': 16, 'num_local_experts': 2, 'num_experts_per_tok': 1,
              'vocab_size': 32, 'max_position_embeddings': 64}
    model = from_record(CausalTransformer, {**translate_config({**released('minimax-text-01'), **narrow}),
                                            'scan_layers': True, 'attention_impl': 'reference'})
    ids = jnp.zeros((1, 1), jnp.int32)
    variables = jax.eval_shape(lambda: model.init(jax.random.key(0), ids))
    cache = jax.eval_shape(lambda: model.init(jax.random.key(0), ids, decode=True))['cache']
    graph = jax.make_jaxpr(lambda variables, cache: model.apply(
        {**variables, 'cache': cache}, ids, decode=True, mutable=['cache']))(variables, cache)
    assert [equation.params['length'] for equation in graph.jaxpr.eqns
            if equation.primitive.name == 'scan'] == [10]


SCAN_SEEDS = 8
"""The scanned decode's declared draws (check 1)."""


def test_a_scanned_minimax_decodes_as_its_plain_loop():
    """A MiniMax of minimax-tiny's config over eight layers repeating three lightning layers and a softmax
    one, at Dew's initialization, decodes under `scan_layers` (one scan of the period's two repeats,
    each lightning layer's rates by its place) as exactly as its plain loop, both from the plain loop's
    float64 steps (check 1)."""
    config = {**released('minimax-tiny'), 'num_hidden_layers': 8, 'attn_type_list': [0, 0, 0, 1] * 2}
    record = {**translate_config(config), 'attention_impl': 'reference'}
    plain, scanned = (from_record(CausalTransformer, {**record, 'scan_layers': scan})
                      for scan in (False, True))
    variables = plain.init(jax.random.key(0), jnp.zeros((2, 4), jnp.int32))

    def truth(ids):
        with jax.enable_x64(new_val=True):
            return np.asarray(stepped_decode(plain.clone(dtype=jnp.float64), widened(variables), ids)[0])

    def steps(model):
        return lambda ids: stepped_decode(model, variables, ids)[0]

    assert_as_exact(steps(scanned), steps(plain), token_draw(0x2564, config['vocab_size'], (2, 10)),
                    'scanned MiniMax decode', seeds=SCAN_SEEDS, truth_fn=truth)


def test_a_record_saved_with_a_kind_for_each_lightning_layer_computes_as_it_did():
    """A run saved before the stack handed each lightning layer its place recorded a kind for each,
    `linear_attention_{index}` stating its `layer_idx` (ledger 2564); its record still builds and
    computes minimax-tiny's logits bit for bit as the record of one kind does."""
    loaded = load()
    record = translate_config(released('minimax-tiny'))
    lightning = record['kinds']['linear_attention']
    legacy = dict(record)
    legacy['layer_types'] = tuple(kind if kind != 'linear_attention' else f'linear_attention_{index}'
                                  for index, kind in enumerate(record['layer_types']))
    others = {name: kind for name, kind in record['kinds'].items() if name != 'linear_attention'}
    fields = lightning['mixer']['fields']
    legacy['kinds'] = {**others, **{
        name: {**lightning, 'mixer': {**lightning['mixer'], 'fields': {**fields, 'layer_idx': index}}}
        for index, name in enumerate(legacy['layer_types']) if name.startswith('linear_attention_')}}
    model = from_record(CausalTransformer, {**legacy, 'attention_impl': 'reference'})
    ids = np.load(TINY / 'input_ids.npy')
    np.testing.assert_array_equal(model.apply(loaded.variables, ids),
                                  loaded.model.apply(loaded.variables, ids))


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
