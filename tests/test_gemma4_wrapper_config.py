"""Raw Gemma 4 configs retain the text class their wrapper declares."""

import json
from pathlib import Path
from shutil import copytree

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import without_config_field

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
    checkpoint = tmp_path / 'unshared'
    copytree(SOURCE, checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    assert config['text_config']['num_kv_shared_layers'] == 0
    config['text_config']['use_double_wide_mlp'] = True
    (checkpoint / 'config.json').write_text(json.dumps(config))
    loaded = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference')
    original = Pretrained.load(SOURCE, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(SOURCE / 'input_ids.npy'))
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids),
                                  original.model.apply(original.variables, ids))


@pytest.mark.network
def test_published_gemma4_tiny_logits_and_cached_generation(tmp_path):
    """The revision-pinned published tiny Gemma 4's text and image logits as near their float64 truth as
    the reference's fp32 run, and its cached greedy continuation the reference's where no step is near
    a tie (`assert_published_checkpoint`)."""
    from interop_support import assert_published_checkpoint

    from tools.wrapper_checkpoint_parity import measure_checkpoint

    measured = measure_checkpoint('trl-internal-testing/tiny-Gemma4ForConditionalGeneration',
                                  tmp_path / 'measurement.json',
                                  revision='0dc1746b7f9f623b748e735ed5a4302eb4baf346')
    assert_published_checkpoint(measured, 'Gemma4')
