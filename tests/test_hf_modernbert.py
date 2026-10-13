"""ModernBERT against transformers 5.16.1 under the float64 rounding rule."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.sources import load_shards
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


def test_modernbert_export_is_same_weight_transformers_and_a_bitwise_reload(tmp_path):
    import torch
    from transformers import AutoModelForMaskedLM

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    original, exported = load_shards(DIRECTORY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for name in original:
        np.testing.assert_array_equal(original[name], exported[name])
    reference = AutoModelForMaskedLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = _fixture('input_ids')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long()).logits.numpy()
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    assert_as_exact_as_the_reference(actual, expected, _fixture('logits_f64'), 'ModernBERT exported logits')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_modernbert_trains_through_its_unnormed_first_block_and_head():
    import optax

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(_fixture('input_ids'))

    def loss(params):
        logits = loaded.model.apply({'params': params}, ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids).mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    assert 'input_layernorm' not in loaded.variables['params']['layers_0']
    assert all(np.abs(leaf).sum() > 0 for leaf in jax.tree.leaves(gradient))
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert float(loss(changed)) < float(value)
