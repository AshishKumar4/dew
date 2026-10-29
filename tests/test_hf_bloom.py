"""BLOOM against transformers 5.17.0 under check 1 of the accuracy gate."""

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import assert_as_exact

from dew.interop import Pretrained

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-tiny'

CACHED_SEEDS = 8
"""The cached padded rows' declared draws of ids against transformers' uncached forward (check 1), sized by
`seeds_for` paired with Dew's prefill on the CPU."""


def test_bloom_padded_rows_prefill_and_decode_through_the_cache_as_the_reference_reads_them():
    """Beyond the shared flow (tests/test_hf_reference_decoders.py): the left-padded rows prefilled into
    the cache give logits on par with transformers' over the same rows and mask, from its float64 twin,
    over drawn ids (check 1); and one token decoded after them gives the argmax a full forward over each
    row's real tokens gives."""
    import torch

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=32)
    mask = np.load(DIRECTORY / 'attention_mask.npy')
    draw = fixture_ids(DIRECTORY, mask)

    def prefilled(ids):
        _, cache = loaded.model.apply(loaded.variables, ids, attention_mask=mask, decode=True,
                                      mutable=['cache'])
        return loaded.model.apply({**loaded.variables, **cache}, ids, attention_mask=mask,
                                  decode=True, mutable=['cache'])

    # BLOOM's ALiBi reads its positions off the mask: its recorder passed none.
    reference, truth = transformers_arms(transformers_model(DIRECTORY), lambda model, ids: model(
        ids, attention_mask=torch.from_numpy(mask), use_cache=False).logits[torch.from_numpy(mask)])
    assert_as_exact(lambda ids: prefilled(ids)[0][mask], reference, draw, 'BLOOM cached padded logits',
                    seeds=CACHED_SEEDS, truth_fn=truth)

    (ids,) = draw(0)
    _, cache = prefilled(ids)
    token = jnp.asarray([[5], [7]], jnp.int32)
    decoded, _ = loaded.model.apply({**loaded.variables, **cache}, token, decode=True, mutable=['cache'])
    for row in range(2):
        joined = np.concatenate((ids[row, mask[row]], np.asarray(token[row])))[None]
        full = loaded.model.apply(loaded.variables, joined)
        np.testing.assert_array_equal(np.asarray(decoded[row, -1]).argmax(), np.asarray(full[0, -1]).argmax())


def respelled(change: dict, directory: Path) -> Path:
    """The tiny BLOOM under `directory` with `change` over its config."""
    shutil.copytree(DIRECTORY, directory)
    config = json.loads((DIRECTORY / 'config.json').read_text())
    (directory / 'config.json').write_text(json.dumps({**config, **change}))
    return directory


@pytest.mark.parametrize('change', [
    {'apply_residual_connection_post_layernorm': True}, {'slow_but_exact': True, 'pretraining_tp': 2},
    {'n_head': 5}], ids=['post-norm residuals', 'tensor-parallel exact path', 'heads not dividing the width'])
def test_a_bloom_dew_does_not_compute_is_refused(change, tmp_path):
    """A BLOOM checkpoint whose config asks what Dew's BLOOM does not compute (residuals read after the
    norm, the tensor-parallel reference's exact path that drops projection biases, a width no whole
    number of heads divides) is refused at load rather than computed without it."""
    with pytest.raises(ValueError):
        Pretrained.load(respelled(change, tmp_path / 'bloom'), dtype='float32', attention_impl='reference')


def test_the_exact_path_of_one_tensor_parallel_rank_computes_bloom_itself(tmp_path):
    """`slow_but_exact` over a single tensor-parallel rank is BLOOM's own computation: it loads and
    computes the plain checkpoint's logits bit for bit."""
    exact = Pretrained.load(respelled({'slow_but_exact': True, 'pretraining_tp': 1}, tmp_path / 'bloom'),
                            dtype='float32', attention_impl='reference')
    plain = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    np.testing.assert_array_equal(np.asarray(exact.model.apply(exact.variables, ids)),
                                  np.asarray(plain.model.apply(plain.variables, ids)))


def test_a_tied_checkpoint_keeping_its_embedding_under_the_head_name_alone_loads_the_same_model(tmp_path):
    """A tied BLOOM whose safetensors keep lm_head.weight and drop transformer.word_embeddings.weight, the
    tensor the config ties it to: the head is the embedding's one copy and fills it, so the model loads
    leaf for leaf as the checkpoint storing the embedding does."""
    from dew.interop.safetensors_io import save_hf_layout
    from dew.interop.sources import load_shards

    tensors = load_shards(DIRECTORY)
    tensors['lm_head.weight'] = tensors.pop('transformer.word_embeddings.weight')
    save_hf_layout(tensors, json.loads((DIRECTORY / 'config.json').read_text()), tmp_path)
    head_only = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    stored = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    assert jax.tree.structure(head_only.variables) == jax.tree.structure(stored.variables)
    for ours, theirs in zip(jax.tree.leaves(head_only.variables), jax.tree.leaves(stored.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
