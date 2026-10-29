"""OPT's learned-position offset and pre-norm block against transformers."""

import json
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
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


def respelled(change: dict, directory: Path) -> Path:
    """The tiny OPT under `directory` with `change` over its config."""
    shutil.copytree(DIRECTORY, directory)
    config = json.loads((DIRECTORY / 'config.json').read_text())
    (directory / 'config.json').write_text(json.dumps({**config, **change}))
    return directory


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
        Pretrained.load(respelled(change, tmp_path / 'opt'), dtype='float32', attention_impl='reference')



def test_a_tied_checkpoint_keeping_its_embedding_under_the_head_name_alone_loads_the_same_model(tmp_path):
    """SFconvertbot's safetensors conversion of OPT-125m keeps lm_head.weight
    and drops model.decoder.embed_tokens.weight, the tensor the config ties it
    to. The head is then the embedding's one copy and fills it: the model
    loads leaf for leaf as the checkpoint storing the embedding does."""
    import json

    from dew.interop.safetensors_io import save_hf_layout
    from dew.interop.sources import load_shards

    tensors = load_shards(DIRECTORY)
    tensors['lm_head.weight'] = tensors.pop('model.decoder.embed_tokens.weight')
    save_hf_layout(tensors, json.loads((DIRECTORY / 'config.json').read_text()), tmp_path)
    head_only = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    stored = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    assert jax.tree.structure(head_only.variables) == jax.tree.structure(stored.variables)
    for ours, theirs in zip(jax.tree.leaves(head_only.variables), jax.tree.leaves(stored.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


@pytest.mark.network
def test_released_opt125m_logits_and_cached_greedy_generation(tmp_path):
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('facebook/opt-125m', tmp_path / 'parity.json',
                     revision='27dcfa74d334bc871f3234de431e71c6eeba5dd6')
