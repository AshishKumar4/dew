"""Every decoder fixture with padded rows against the reference that wrote it: tools/hf_reference.py,
tools/mamba_reference.py, tools/ouro_reference.py and tools/k2_horizon_reference.py write each
directory's ids, its left-padded rows and mask, the reference's logits for both in float32 and float64,
and its greedy continuation. A family's own checks stay in its own file."""

from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, float64_forward

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


def load(name: str) -> Pretrained:
    return Pretrained.load(ROOT / name, dtype='float32', attention_impl='reference', **FIXTURES[name])


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_padded_and_unpadded_logits_and_the_greedy_continuation_match_the_reference(name):
    """Left-padded rows under their mask and the unpadded rows, each as exact as the reference from
    float64 with the same argmax, and four ids continued six tokens greedily through Dew's own
    cached generation."""
    directory, loaded = ROOT / name, load(name)

    def fixture(array: str) -> np.ndarray:
        return np.load(directory / f'{array}.npy')

    mask, ids = fixture('attention_mask'), fixture('input_ids')
    for rows, valid, given, options in (
            ('padded_logits', mask, fixture('padded_ids'), {'attention_mask': mask}),
            ('logits', np.ones(ids.shape, bool), ids, {})):
        logits = loaded.model.apply(loaded.variables, given, **options)
        _, terms = float64_forward(loaded.model, loaded.variables, given, **options)
        actual, expected = np.asarray(logits)[valid], fixture(rows)[valid]
        assert_as_exact_as_the_reference(actual, expected, fixture(f'{rows}_f64')[valid], f'{name} {rows}',
                                         terms=np.asarray(terms)[valid])
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    continued = generate(loaded.model, loaded.variables, ids[:, :4], 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(continued.tokens), fixture('generated'))
