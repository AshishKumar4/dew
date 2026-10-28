"""ModernBERT against transformers 5.17.0 under check 1 of the accuracy gate."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import assert_as_exact, assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = FIXTURES / 'modernbert-tiny'
def _fixture(name: str) -> np.ndarray:
    return np.load(DIRECTORY / f'{name}.npy')


LOGITS_SEEDS = 0
"""The masked LM's declared draws of ids against ModernBertForMaskedLM's logits (check 1), sized by
`seeds_for` paired with Dew's logits on the CPU."""


def test_modernbert_masked_lm_logits_match_the_reference():
    """The masked LM's logits are on par with ModernBertForMaskedLM's, from its float64 twin's, over
    drawn ids (check 1), with the same argmax."""
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    draw = fixture_ids(DIRECTORY)
    reference, truth = transformers_arms(transformers_model(DIRECTORY, masked=True),
                                         lambda model, ids: model(ids).logits)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids), reference, draw,
                    'ModernBERT logits', seeds=LOGITS_SEEDS, truth_fn=truth)
    (ids,) = draw(0)
    np.testing.assert_array_equal(np.asarray(loaded.model.apply(loaded.variables, ids)).argmax(-1),
                                  reference(ids).argmax(-1))


STATES_SEEDS = {'reference': 0, 'xla': 0}
"""The encoder states' declared draws of ids against ModernBertModel's (check 1), on each attention path,
sized by `seeds_for` paired with Dew's states on the CPU."""


@pytest.mark.parametrize('implementation', ['reference', 'xla'])
def test_modernbert_states_match_the_encoder_through_every_window_path(implementation):
    """The symmetric window reaches the kernel's flag, the validity mask and
    packed documents alike: plain rows, right-padded rows under an
    attention mask, and two documents packed in one row, each on par with
    ModernBertModel's, the packed ones with each document's run alone, from
    its float64 twin's, over drawn ids (check 1)."""
    import torch

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl=implementation)
    mask = _fixture('attention_mask')
    documents = _fixture('segment_ids')
    positions = np.where(documents == 1, np.arange(documents.shape[1]), np.arange(documents.shape[1]) - 5)

    def ours(ids):
        def states(ids, **kwargs):
            return loaded.model.apply(loaded.variables, ids, method=CausalTransformer.hidden_states, **kwargs)
        return {'states': states(ids),
                'padded': states(jnp.where(mask, ids, 0), attention_mask=mask)[mask],
                'packed': states(ids, segment_ids=documents, positions=positions)}

    def read(model, ids):
        def states(ids, **kwargs):
            return model.model(ids, **kwargs).last_hidden_state
        valid = torch.from_numpy(mask)
        return {'states': states(ids),
                'padded': states(torch.where(valid, ids, 0), attention_mask=valid)[valid],
                'packed': torch.cat([states(ids[:, :5]), states(ids[:, 5:])], dim=1)}

    reference, truth = transformers_arms(transformers_model(DIRECTORY, masked=True), read)
    assert_as_exact(ours, reference, fixture_ids(DIRECTORY), 'ModernBERT states',
                    seeds=STATES_SEEDS[implementation], truth_fn=truth)


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
