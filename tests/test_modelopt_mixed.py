"""Per-layer ModelOpt FP8 and NVFP4 dispatch through the same checkpoint provider."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import codecs
from dew.interop.safetensors_io import read_weights
from dew.training.quantization import InputQuantizer, fp8_input_qdq

FIXTURE = Path(__file__).parent / 'fixtures' / 'codecs' / 'modelopt' / 'mixed'


@pytest.fixture(scope='module')
def reference():
    with np.load(FIXTURE / 'reference.npz') as arrays:
        return dict(arrays)


def test_each_mixed_weight_decodes_as_its_own_author_reader(reference):
    stored = read_weights(FIXTURE)
    codec = codecs.source_quantization(json.loads((FIXTURE / 'config.json').read_text()))
    assert codec is not None
    assert len(codec.names(stored)) == 7
    for name in codec.names(stored):
        expected = reference[name.removesuffix('.weight') + '/weight']
        np.testing.assert_array_equal(codec.decode(stored, name).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
def test_static_fp8_inputs_match_modelopts_own_fake_quantizer(reference, dtype):
    for key in reference:
        if not key.endswith('/inputs'):
            continue
        stem = key.removesuffix('/inputs')
        spec = InputQuantizer.fp8(float(reference[stem + '/global'].reshape(())))
        quantize = jax.jit(lambda value, spec=spec: fp8_input_qdq(value, spec))
        actual = quantize(jnp.asarray(reference[key], dtype))
        expected = reference[stem + ('/qdq' if dtype == jnp.float32 else '/qdq_bf16')]
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      expected.view(np.uint32))
