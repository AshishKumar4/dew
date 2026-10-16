"""Raw Gemma 4 configs retain the text class their wrapper declares."""

import json
from pathlib import Path
from shutil import copytree

import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained

SOURCE = Path(__file__).parent / 'fixtures' / 'hf' / 'gemma4-native-tiny'


def test_gemma4_wrapper_supplies_the_missing_nested_text_type(tmp_path):
    checkpoint = tmp_path / 'raw-config'
    copytree(SOURCE, checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    del config['text_config']['model_type']
    (checkpoint / 'config.json').write_text(json.dumps(config))
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
    from reference_error import assert_as_exact_as_the_reference

    from tools.gemma4_fp64_reference import CHECKPOINT, REVISION, reference_logits
    from tools.wrapper_checkpoint_parity import measure_checkpoint

    result, logits = measure_checkpoint(CHECKPOINT, tmp_path / 'measurement.json', revision=REVISION)
    assert all(row['argmax_agreement'] and row['generation_agreement']
               for row in result['observations']), result
    # Each modality's fp32 runs, Dew's and the reference's, from the float64 oracle on the same batch:
    # the image logits are held as the text's are, not to a largest difference a layer count bounds,
    # which the reference's own fp32 rounding crossed in 1 of 6 runs (2.77e-6 against 2.59e-6).
    for (actual, reference, ids), (truth, oracle_ids), modality in zip(
            logits, reference_logits(), ('text', 'image'), strict=True):
        np.testing.assert_array_equal(ids, oracle_ids)
        assert truth.dtype == np.float64
        assert_as_exact_as_the_reference(actual, reference, truth, f'published Gemma4 {modality}')
