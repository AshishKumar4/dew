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
    import torch
    from reference_error import assert_as_exact_as_the_reference

    from tools.gemma4_fp64_reference import CHECKPOINT, OUTPUT, REVISION
    from tools.wrapper_checkpoint_parity import measure_checkpoint

    # A CPU-only Torch build still supplies the reference when Dew runs on a GPU.
    reference_device = None if torch.cuda.is_available() else 'cpu'
    result, logits = measure_checkpoint(CHECKPOINT, tmp_path / 'measurement.json', revision=REVISION,
                                        reference_device=reference_device)
    assert all(row['argmax_agreement'] and row['generation_agreement']
               for row in result['observations']), result
    actual, reference, ids = logits[0]
    with np.load(OUTPUT) as oracle:
        assert oracle['checkpoint'].item() == CHECKPOINT
        assert oracle['revision'].item() == REVISION
        np.testing.assert_array_equal(ids, oracle['input_ids'])
        assert oracle['logits'].dtype == np.float64
        # The CPU max delta is 2.50e-6, above the RTX 4080's 1.24e-6
        # calibration. Both fp32 runs are 1.36e-7 RMS from the fp64 oracle.
        assert_as_exact_as_the_reference(actual, reference, oracle['logits'], 'published Gemma4 text')
    image = result['observations'][1]
    assert image['max_abs_error'] < image['bound'], image
