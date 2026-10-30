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
from reference_error import assert_as_exact, widened

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import run_period, scan_groups
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
    keys = from_record(CausalTransformer, own).scan_keys
    assert scan_groups(keys, periodic=True) == ((0, 80),) and run_period(keys, 0, 80) == 8


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

    def decoded(model, variables, ids):
        def step(cache, column):
            logits, state = model.apply({**variables, 'cache': cache}, column[:, None], decode=True,
                                        mutable=['cache'])
            return state['cache'], logits[:, 0]

        cache = model.init(jax.random.key(0), ids[:, :1], decode=True)['cache']
        return jnp.moveaxis(jax.lax.scan(step, cache, jnp.moveaxis(ids, 1, 0))[1], 0, 1)

    def truth(ids):
        with jax.enable_x64(new_val=True):
            return np.asarray(decoded(plain.clone(dtype=jnp.float64), widened(variables), jnp.asarray(ids)))

    def draw(seed):
        drawn = np.random.default_rng([0x2564, seed]).integers(0, config['vocab_size'], (2, 10))
        return (drawn.astype(np.int32),)

    assert_as_exact(lambda ids: decoded(scanned, variables, ids), lambda ids: decoded(plain, variables, ids),
                    draw, 'scanned MiniMax decode', seeds=SCAN_SEEDS, truth_fn=truth)


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
