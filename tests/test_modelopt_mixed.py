"""Per-layer ModelOpt FP8 and NVFP4 dispatch through the same checkpoint provider."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, codecs
from dew.interop.safetensors_io import read_weights
from dew.training.quantization import FP8Input, fp8_input_qdq

FIXTURE = Path(__file__).parent / 'fixtures' / 'codecs' / 'modelopt' / 'mixed'


@pytest.fixture(scope='module')
def mixed():
    with np.load(FIXTURE / 'reference.npz') as reference:
        return dict(reference), Pretrained.load(FIXTURE, dtype='float32', attention_impl='reference')


def test_each_mixed_weight_decodes_as_its_own_author_reader(mixed):
    reference, _ = mixed
    stored = read_weights(FIXTURE)
    codec = codecs.source_quantization(json.loads((FIXTURE / 'config.json').read_text()))
    assert codec is not None
    assert len(codec.names(stored)) == 7
    for name in codec.names(stored):
        expected = reference[name.removesuffix('.weight') + '/weight']
        np.testing.assert_array_equal(codec.decode(stored, name).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
def test_static_fp8_inputs_match_modelopts_own_fake_quantizer(mixed, dtype):
    reference, _ = mixed
    for key in reference:
        if not key.endswith('/inputs'):
            continue
        stem = key.removesuffix('/inputs')
        spec = FP8Input(float(reference[stem + '/global'].reshape(())))
        quantize = jax.jit(lambda value, spec=spec: fp8_input_qdq(value, spec))
        actual = quantize(jnp.asarray(reference[key], dtype))
        expected = reference[stem + ('/qdq' if dtype == jnp.float32 else '/qdq_bf16')]
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      expected.view(np.uint32))


def test_mixed_logits_compute_each_layers_published_algorithm(mixed):
    reference, loaded = mixed
    actual = loaded.model.apply(loaded.variables, reference['ids'].astype(np.int32))
    assert_as_exact_as_the_reference(actual, reference['logits'], reference['logits_f64'], 'mixed logits')
    np.testing.assert_array_equal(np.argmax(actual, -1), np.argmax(reference['logits'], -1))


def test_a_mixed_export_keeps_its_source_codes_and_layer_table(mixed, tmp_path):
    reference, loaded = mixed
    loaded.save(tmp_path)
    original, written = read_weights(FIXTURE), read_weights(tmp_path)
    assert set(original) == set(written)
    for name, value in original.items():
        assert value.dtype == written[name].dtype, name
        np.testing.assert_array_equal(value.reshape(-1).view(np.uint8),
                                      written[name].reshape(-1).view(np.uint8), err_msg=name)
    reloaded = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    actual = reloaded.model.apply(reloaded.variables, reference['ids'].astype(np.int32))
    assert_as_exact_as_the_reference(actual, reference['logits'], reference['logits_f64'], 'mixed export')
