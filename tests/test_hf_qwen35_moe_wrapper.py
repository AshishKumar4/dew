"""Text, image conditioning and generation through Qwen 3.5 MoE's wrapper."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.interop import Pretrained
from dew.sampling.text import Sampling

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'qwen35-moe-native-tiny'


def test_qwen35_moe_text_and_images_match_the_conditional_reference():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    assert loaded.processor is not None
    text = loaded.processor.from_hf({
        'input_ids': np.load(DIRECTORY / 'text_input_ids.npy'),
        'attention_mask': np.load(DIRECTORY / 'text_attention_mask.npy')})
    actual = np.asarray(loaded.model.apply(loaded.variables, text.tokens, **text.kwargs()))
    expected = np.load(DIRECTORY / 'text_logits.npy')
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    images = np.load(DIRECTORY / 'raw_images.npy')
    inputs = loaded.processor(json.loads((DIRECTORY / 'prompts.json').read_text()),
                              images=[[images[0]], [images[1], images[2]]])
    valid = np.asarray(inputs.token_fields['attention_mask'])
    actual = np.asarray(jax.jit(lambda variables: loaded.model.apply(
        variables, inputs.tokens, **inputs.kwargs()))(loaded.variables))
    expected = np.load(DIRECTORY / 'logits.npy')
    np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual[valid].argmax(-1), expected[valid].argmax(-1))
    generated = loaded.text_generation()(inputs, 3, key=jax.random.key(0), sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], np.load(DIRECTORY / 'continuation.npy'))


def test_advertised_mtp_without_weights_loads_the_actual_conditional_trunk(tmp_path):
    from shutil import copytree

    checkpoint = tmp_path / 'checkpoint'
    copytree(DIRECTORY, checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    config['text_config']['mtp_num_hidden_layers'] = 1
    (checkpoint / 'config.json').write_text(json.dumps(config))
    advertised = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference')
    original = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'text_input_ids.npy')
    np.testing.assert_array_equal(advertised.model.apply(advertised.variables, ids),
                                  original.model.apply(original.variables, ids))


def test_qwen35_moe_conditioning_has_finite_trainable_gradients():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    images = np.load(DIRECTORY / 'raw_images.npy')
    inputs = loaded.processor(json.loads((DIRECTORY / 'prompts.json').read_text()),
                              images=[[images[0]], [images[1], images[2]]])
    mask = inputs.token_fields['attention_mask'][:, 1:]
    def loss(params):
        logits = loaded.model.apply({'params': params}, inputs.tokens, **inputs.kwargs())[:, :-1]
        values = optax.softmax_cross_entropy_with_integer_labels(logits, inputs.tokens[:, 1:])
        return jnp.sum(jnp.where(mask, values, 0)) / jnp.sum(mask)
    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables['params'])
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
    for name in ('language_model', 'tower', 'projector'):
        assert float(optax.tree.norm(gradient[name])) > 0
    updated = jax.tree.map(lambda weight, grad: weight - .001 * grad, loaded.variables['params'], gradient)
    assert float(loss(updated)) < float(value)


@pytest.mark.network
def test_published_qwen35_moe_tiny_checkpoint_logits_and_generation(tmp_path):
    from tools.wrapper_checkpoint_parity import check_checkpoint

    check_checkpoint('trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6',
                     tmp_path / 'measurement.json', revision='57faabc3b49ec3ae69b8d6158130ce19496bc2d3')
