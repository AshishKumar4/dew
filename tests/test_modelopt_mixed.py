"""Per-layer ModelOpt FP8 and NVFP4 dispatch through the same checkpoint provider."""

import json
import os
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


def test_the_layer_table_overrides_generic_four_bit_activation_metadata(mixed):
    config = json.loads((FIXTURE / 'config.json').read_text())
    group = next(group for group in config['quantization_config']['config_groups'].values()
                 if group['weights']['num_bits'] == 4)
    group['input_activations'] = {'num_bits': 4, 'type': 'float', 'group_size': 16, 'dynamic': False}
    codec = codecs.source_quantization(config)
    assert codec is not None
    assert codec.input_kind('model.layers.0.mlp.down_proj.weight') == 'none'
    assert codec.input_kind('model.layers.0.self_attn.q_proj.weight') == 'fp8'


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


@pytest.mark.network
@pytest.mark.skipif(os.environ.get('DEW_NETWORK_TESTS') != '1',
                    reason='set DEW_NETWORK_TESTS=1 for the pinned Hub config/index')
def test_real_mixed_dispatch_uses_the_layer_table_even_when_unused_input_scales_are_stored():
    """The largest refused mixed repo exports expert input scales but declares W4A16."""
    from huggingface_hub import hf_hub_download

    repo, revision = 'nvidia/Qwen3.6-35B-A3B-NVFP4', '1355db6a052410cfd62085d94b58866fd0f2c3c5'
    config = json.loads(Path(hf_hub_download(repo, 'config.json', revision=revision)).read_text())
    index_path = Path(hf_hub_download(repo, 'model.safetensors.index.json', revision=revision))
    index = json.loads(index_path.read_text())['weight_map']
    codec = codecs.source_quantization(config)
    assert codec is not None
    stem = 'model.language_model.layers.0.mlp.experts.0.gate_proj'
    assert stem + '.input_scale' in index
    assert codec.input_kind(stem + '.weight') == 'none'
    assert stem + '.input_scale' not in codec.partners(stem + '.weight')
