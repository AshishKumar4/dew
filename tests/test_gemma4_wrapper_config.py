"""Raw Gemma 4 configs retain the text class their wrapper declares."""

import json
from pathlib import Path
from shutil import copytree

import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import load_pretrained

SOURCE = Path(__file__).parent / 'fixtures' / 'hf' / 'gemma4-native-tiny'


def test_gemma4_wrapper_supplies_the_missing_nested_text_type(tmp_path):
    checkpoint = tmp_path / 'raw-config'
    copytree(SOURCE, checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    del config['text_config']['model_type']
    (checkpoint / 'config.json').write_text(json.dumps(config))
    loaded = load_pretrained(checkpoint, dtype='float32', attention_impl='reference')
    original = load_pretrained(SOURCE, dtype='float32', attention_impl='reference')
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
    loaded = load_pretrained(checkpoint, dtype='float32', attention_impl='reference')
    original = load_pretrained(SOURCE, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(SOURCE / 'input_ids.npy'))
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids),
                                  original.model.apply(original.variables, ids))


@pytest.mark.network
def test_published_gemma4_tiny_logits_and_cached_generation(tmp_path):
    from tools.wrapper_checkpoint_parity import check_checkpoint

    check_checkpoint('trl-internal-testing/tiny-Gemma4ForConditionalGeneration',
                     tmp_path / 'measurement.json', revision='0dc1746b7f9f623b748e735ed5a4302eb4baf346')
