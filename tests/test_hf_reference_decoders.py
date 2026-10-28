"""Every decoder fixture with padded rows against the reference that wrote it: tools/hf_reference.py,
tools/mamba_reference.py, tools/ouro_reference.py and tools/k2_horizon_reference.py write each
directory's ids, its left-padded rows and mask, and its greedy continuation. A family's own checks stay in
its own file."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import assert_as_exact

from dew.interop import Pretrained
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / 'fixtures' / 'hf'

FIXTURES = {
    'phi-tiny': {}, 'gptj-tiny': {}, 'gpt-neo-tiny': {}, 'falcon-tiny': {}, 'falcon-mha-tiny': {},
    'gpt-bigcode-tiny': {}, 'gpt-bigcode-mha-tiny': {}, 'starcoder2-tiny': {}, 'mimo-v2-flash-tiny': {},
    'mamba-tiny': {}, 'ouro-tiny': {}, 'k2-horizon-tiny': {},
    # BLOOM's ALiBi config states no context length.
    'bloom-tiny': {'max_seq_len': 32},
}
"""Each fixture directory, with what loading it needs beyond float32 and the reference kernel."""

SEEDS = {'bloom-tiny': 0, 'falcon-mha-tiny': 0, 'falcon-tiny': 0, 'gpt-bigcode-mha-tiny': 0, 'gpt-bigcode-tiny': 0, 'gpt-neo-tiny': 0, 'gptj-tiny': 0, 'mamba-tiny': 0, 'mimo-v2-flash-tiny': 0, 'phi-tiny': 0, 'starcoder2-tiny': 0}
"""Each transformers family's declared draws of ids against transformers' padded and unpadded logits
(check 1), sized by `seeds_for` paired with Dew's logits on the CPU."""


def load(name: str) -> Pretrained:
    return Pretrained.load(ROOT / name, dtype='float32', attention_impl='reference', **FIXTURES[name])


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_padded_and_unpadded_logits_and_the_greedy_continuation_match_the_reference(name):
    """Left-padded rows under their mask and the unpadded rows, each on par with the reference from
    float64 over drawn ids (check 1), with the same argmax on the first draw, and four ids continued six
    tokens greedily through Dew's own cached generation."""
    import torch

    directory, loaded = ROOT / name, load(name)
    mask = np.load(directory / 'attention_mask.npy')
    draw = fixture_ids(directory)

    def ours(ids):
        padded = loaded.model.apply(loaded.variables, jnp.where(mask, ids, 0), attention_mask=mask)
        return {'padded_logits': padded[mask], 'logits': loaded.model.apply(loaded.variables, ids)}

    # The fields its recorder (tools/hf_reference.py `write_classic_tiny`) gave the padded rows: BLOOM's
    # ALiBi reads its positions off the mask.
    valid = torch.from_numpy(mask)
    fields = {'attention_mask': valid, **({} if name == 'bloom-tiny' else {
        'position_ids': torch.from_numpy(np.maximum(np.cumsum(mask, -1) - 1, 0))})}

    def read(model, ids):
        padded = model(torch.where(valid, ids, 0), **fields, use_cache=False).logits
        return {'padded_logits': padded[valid], 'logits': model(ids, use_cache=False).logits}

    reference, truth = transformers_arms(transformers_model(directory), read)
    assert_as_exact(ours, reference, draw, f'{name} logits', seeds=SEEDS[name], truth_fn=truth)
    (ids,) = draw(0)
    expected = reference(ids)
    for rows, actual in ours(jnp.asarray(ids)).items():
        np.testing.assert_array_equal(np.asarray(actual).argmax(-1), expected[rows].argmax(-1))

    ids = np.load(directory / 'input_ids.npy')
    continued = generate(loaded.model, loaded.variables, ids[:, :4], 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(continued.tokens), np.load(directory / 'generated.npy'))
