"""OPT's learned-position offset and pre-norm block against transformers."""

from pathlib import Path

import numpy as np
import pytest
from interop_support import fixture_ids, respelled, transformers_arms, transformers_model
from reference_error import assert_as_exact

from dew.interop import Pretrained
from dew.sampling import Sampling, generate

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'opt-tiny'

OPT_SEEDS = 8
"""The logits' declared draws of ids against transformers' (check 1)."""


def test_opt_logits_follow_transformers_and_its_greedy_continuation():
    """OPT's logits, its learned positions read two places on (OPT's offset), are on par with
    transformers' from its float64 twin over drawn ids (check 1), and a cached greedy continuation draws
    the tokens transformers drew."""
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    reference, truth = transformers_arms(transformers_model(DIRECTORY),
                                         lambda model, ids: model(ids, use_cache=False).logits)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids), reference, fixture_ids(DIRECTORY),
                    'OPT logits', seeds=OPT_SEEDS, truth_fn=truth)
    ids = np.load(DIRECTORY / 'input_ids.npy')
    generated = generate(loaded.model, loaded.variables, ids[:1, :4], 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), np.load(DIRECTORY / 'generated.npy'))


@pytest.mark.parametrize('change', [
    {'activation_dropout': 0.1}, {'do_layer_norm_before': False}, {'_remove_final_layer_norm': True},
    {'layer_norm_elementwise_affine': False}, {'word_embed_proj_dim': 16}, {'layerdrop': 0.1}],
    ids=['activation dropout', 'post-norm', 'no final norm', 'unscaled norms', 'projected embeddings',
         'layer drop'])
def test_an_opt_dew_does_not_compute_is_refused(change, tmp_path):
    """An OPT checkpoint whose config asks what Dew's OPT does not compute (dropout inside the
    feed-forward, post-residual or scaleless norms, no final norm, embeddings projected to another
    width, dropped layers) is refused at load rather than computed without it."""
    with pytest.raises(ValueError):
        Pretrained.load(respelled(DIRECTORY, tmp_path / 'opt', lambda config: {**config, **change}),
                        dtype='float32', attention_impl='reference')
