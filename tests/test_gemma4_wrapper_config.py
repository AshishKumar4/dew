"""Raw Gemma 4 configs retain the text class their wrapper declares."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import respelled, without_config_field

from dew.interop import Pretrained

SOURCE = Path(__file__).parent / 'fixtures' / 'hf' / 'gemma4-native-tiny'


def test_gemma4_wrapper_supplies_the_missing_nested_text_type(tmp_path):
    checkpoint = without_config_field(SOURCE, tmp_path / 'raw-config', 'text_config', 'model_type')
    loaded = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference')
    original = Pretrained.load(SOURCE, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(SOURCE / 'input_ids.npy'))
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids),
                                  original.model.apply(original.variables, ids))


def test_gemma4_unshared_layers_ignore_the_double_width_flag(tmp_path):
    def doubled(config):
        assert config['text_config']['num_kv_shared_layers'] == 0
        return {**config, 'text_config': {**config['text_config'], 'use_double_wide_mlp': True}}

    checkpoint = respelled(SOURCE, tmp_path / 'unshared', doubled)
    loaded = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference')
    original = Pretrained.load(SOURCE, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(SOURCE / 'input_ids.npy'))
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids),
                                  original.model.apply(original.variables, ids))


@pytest.mark.network
def test_published_gemma4_tiny_logits_and_cached_generation():
    """The revision-pinned published tiny Gemma 4's text and image logits on par with the reference's fp32
    run over drawn prompts, and its cached greedy continuation the reference's where no step is near
    a tie (`assert_published_checkpoint`)."""
    from interop_support import assert_published_checkpoint

    assert_published_checkpoint('trl-internal-testing/tiny-Gemma4ForConditionalGeneration',
                                '0dc1746b7f9f623b748e735ed5a4302eb4baf346', 'Gemma4')
