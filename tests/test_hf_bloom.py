"""BLOOM against transformers 5.17.0 under check 1 of the accuracy gate."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import assert_as_exact

from dew.interop import Pretrained

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-tiny'
RELEASE = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-560m'

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
