"""ModernBERT against transformers 5.16.1 under the float64 rounding rule."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = FIXTURES / 'modernbert-tiny'
def _fixture(name: str) -> np.ndarray:
    return np.load(DIRECTORY / f'{name}.npy')


def test_modernbert_masked_lm_logits_match_the_reference():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    logits = np.asarray(loaded.model.apply(loaded.variables, _fixture('input_ids')))
    assert_as_exact_as_the_reference(logits, _fixture('logits'), _fixture('logits_f64'), 'ModernBERT logits')
    np.testing.assert_array_equal(logits.argmax(-1), _fixture('logits').argmax(-1))


@pytest.mark.parametrize('implementation', ['reference', 'xla'])
def test_modernbert_states_match_the_encoder_through_every_window_path(implementation):
    """The symmetric window reaches the kernel's flag, the validity mask and
    packed documents alike: plain rows, right-padded rows under an
    attention mask, and two documents packed in one row, each against
    ModernBertModel."""
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl=implementation)

    def states(ids, **kwargs):
        return np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids),
                                             method=CausalTransformer.hidden_states, **kwargs))

    assert_as_exact_as_the_reference(states(_fixture('input_ids')), _fixture('hidden'),
                                     _fixture('hidden_f64'), 'ModernBERT states')
    mask = _fixture('attention_mask')
    padded = states(_fixture('padded_ids'), attention_mask=mask)
    assert_as_exact_as_the_reference(padded[mask], _fixture('padded_hidden')[mask],
                                     _fixture('padded_hidden_f64')[mask], 'ModernBERT padded states')
    documents = _fixture('segment_ids')
    positions = np.where(documents == 1, np.arange(documents.shape[1]), np.arange(documents.shape[1]) - 5)
    packed = states(_fixture('input_ids'), segment_ids=documents, positions=positions)
    assert_as_exact_as_the_reference(packed, _fixture('packed_hidden'), _fixture('packed_hidden_f64'),
                                     'ModernBERT packed states')


@pytest.mark.gpu(reason='the bf16 kernels it selects are the GPU ones')
def test_modernbert_states_in_bf16_on_the_gpu_kernels(without_deterministic_ops):
    """'auto' runs the global layers on cuDNN and keeps the local layers'
    two-sided window off it, which cuDNN refuses; the bf16 states are held to
    the reference's own bf16 run."""
    loaded = Pretrained.load(DIRECTORY, dtype='bfloat16', attention_impl='auto')
    states = loaded.model.apply(loaded.variables, _fixture('input_ids'),
                                method=CausalTransformer.hidden_states)
    assert_as_exact_as_the_reference(np.asarray(states, np.float32), _fixture('hidden_bf16'),
                                     _fixture('hidden_f64'), 'ModernBERT bf16 states')
