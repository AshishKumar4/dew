"""Text, image conditioning and generation through Qwen 3.5 MoE's wrapper."""

import json
from pathlib import Path

import jax
import numpy as np
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


@pytest.mark.network
def test_published_qwen35_moe_tiny_checkpoint_logits_and_generation(tmp_path):
    """The revision-pinned published tiny Qwen 3.5 MoE's text and image logits as near their float64
    truth as the reference's fp32 run, and its greedy continuation the reference's where no step is near
    a tie (`assert_published_checkpoint`)."""
    from interop_support import assert_published_checkpoint

    from tools.wrapper_checkpoint_parity import measure_checkpoint

    measured = measure_checkpoint('trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6',
                                  tmp_path / 'measurement.json',
                                  revision='57faabc3b49ec3ae69b8d6158130ce19496bc2d3')
    assert_published_checkpoint(measured, 'Qwen3.5 MoE')


def test_images_read_through_a_config_that_leaves_the_merge_size_to_its_default(tmp_path):
    """A Qwen 3.5 config may omit vision_config.spatial_merge_size, whose reference default is 2;
    the tower and merger are built at that default, so the processor places the images at it too
    and the model computes the conditional reference's logits."""
    from interop_support import without_config_field

    checkpoint = without_config_field(DIRECTORY, tmp_path / 'checkpoint',
                                      'vision_config', 'spatial_merge_size')
    loaded = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference')
    assert loaded.processor is not None
    images = np.load(DIRECTORY / 'raw_images.npy')
    inputs = loaded.processor(json.loads((DIRECTORY / 'prompts.json').read_text()),
                              images=[[images[0]], [images[1], images[2]]])
    valid = np.asarray(inputs.token_fields['attention_mask'])
    actual = np.asarray(loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs()))
    np.testing.assert_allclose(actual[valid], np.load(DIRECTORY / 'logits.npy')[valid], atol=1e-4, rtol=0)
