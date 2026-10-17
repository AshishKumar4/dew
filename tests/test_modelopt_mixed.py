"""Per-layer ModelOpt FP8 and NVFP4 dispatch through the same checkpoint provider."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained, codecs
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


def test_a_dense_head_copy_beside_the_mixed_table_is_checked_before_narrowing(tmp_path):
    """The mixed table quantizes the projections alone, so a tied source's
    dense lm_head.weight is read as stored and checked against the
    embedding before the bfloat16 cast: an equal copy loads as the source
    without it does, and an unequal one is refused."""
    from dew.interop.safetensors_io import save_hf_layout

    tensors = read_weights(FIXTURE)
    config = json.loads((FIXTURE / 'config.json').read_text())
    embedding = tensors['model.embed_tokens.weight']
    without = Pretrained.load(FIXTURE, dtype='float32', param_dtype='bfloat16', attention_impl='reference')

    for name, head in (('equal', embedding.copy()), ('unequal', embedding + 1)):
        directory = tmp_path / name
        save_hf_layout({**tensors, 'lm_head.weight': head}, config, directory)
        (directory / 'hf_quant_config.json').write_text((FIXTURE / 'hf_quant_config.json').read_text())
    copied = Pretrained.load(tmp_path / 'equal', dtype='float32', param_dtype='bfloat16',
                             attention_impl='reference')
    assert jax.tree.structure(copied.variables) == jax.tree.structure(without.variables)
    for ours, theirs in zip(jax.tree.leaves(copied.variables), jax.tree.leaves(without.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path / 'unequal', dtype='float32', param_dtype='bfloat16',
                        attention_impl='reference')


@pytest.mark.parametrize('uniform', ['FP8', 'W4A16_NVFP4'])
def test_a_checkpoint_of_one_modelopt_algorithm_loads_as_that_algorithms_layers_in_a_mixed_one(
        tmp_path, uniform):
    """ModelOpt writes a checkpoint of one scheme as a uniform quant_algo with no layer table. Its
    quantized layers decode as the same layers of a mixed table do, the rest read dense, and with
    FP8 layers its activations quantize as the mixed table's do."""
    from dew.interop.safetensors_io import save_hf_layout

    stored = read_weights(FIXTURE)
    config = json.loads((FIXTURE / 'config.json').read_text())
    codec = codecs.source_quantization(config)
    assert codec is not None
    tensors = dict(stored)
    for layer, entry in config['quantization_config']['quantized_layers'].items():
        if entry['quant_algo'] != uniform:
            tensors[layer + '.weight'] = codec.decode(stored, layer + '.weight')
            for part in ('.weight_scale', '.weight_scale_2', '.input_scale'):
                tensors.pop(layer + part, None)
    quantization = {'quant_algo': uniform, 'kv_cache_quant_algo': None,
                    **({'group_size': 16} if uniform == 'W4A16_NVFP4' else {})}
    producer = config['quantization_config']['producer']
    save_hf_layout(tensors, {**config, 'quantization_config': {**quantization, 'producer': producer,
                                                                 'quant_method': 'modelopt'}}, tmp_path)
    (tmp_path / 'hf_quant_config.json').write_text(json.dumps({'producer': producer,
                                                               'quantization': quantization}))
    load = {'dtype': 'float32', 'param_dtype': 'bfloat16', 'attention_impl': 'reference'}
    single, mixed = Pretrained.load(tmp_path, **load), Pretrained.load(FIXTURE, **load)
    params = (jax.tree.leaves(single.variables['params']), jax.tree.leaves(mixed.variables['params']))
    for ours, theirs in zip(*params, strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    if uniform == 'FP8':
        ids = np.arange(8, dtype=np.int32).reshape(1, 8)
        np.testing.assert_array_equal(np.asarray(single.model.apply(single.variables, ids)),
                                      np.asarray(mixed.model.apply(mixed.variables, ids)))


@pytest.mark.parametrize('experts', ['num_experts', 'num_local_experts', 'n_routed_experts'])
def test_a_modelopt_nvfp4_description_of_a_model_with_experts_is_read_by_its_stored_layers(experts):
    """ModelOpt may leave a model's experts dense and quantize its projections; the quantized
    layers are those whose scales are stored, so a uniform NVFP4 description of a model with
    experts reads as one, and only an expert stack's input quantizers are refused, where the
    weights bind."""
    config = {'model_type': 'qwen3_moe', experts: 8,
              'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4', 'group_size': 16}}
    assert codecs.source_quantization(config) is not None

