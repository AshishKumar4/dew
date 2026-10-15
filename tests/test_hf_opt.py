"""OPT's learned-position offset and pre-norm block against transformers."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained
from dew.sampling import Sampling, generate

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'opt-tiny'


def test_opt_same_weight_logits_and_cached_greedy_generation():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'input_ids.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    expected = np.load(DIRECTORY / 'logits.npy')
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    generated = generate(loaded.model, loaded.variables, ids[:1, :4], 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), np.load(DIRECTORY / 'generated.npy'))
    wrong = loaded.model.clone(position_embedding_offset=0)
    assert float(np.max(np.abs(np.asarray(wrong.apply(loaded.variables, ids)) - expected))) > 1e-2



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
