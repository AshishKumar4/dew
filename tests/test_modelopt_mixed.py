"""Per-layer ModelOpt FP8 and NVFP4 dispatch through the same checkpoint provider."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained, codecs
from dew.interop.safetensors_io import read_weights
from dew.quantization import InputQuantizer, fp8_input_qdq

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


def nvfp4(name: str, weight: np.ndarray) -> dict[str, np.ndarray]:
    """`weight` as ModelOpt's NVFP4 exporter stores it: E2M1 codes under one E4M3 scale per 16
    inputs and one float32 global scale, beside the static input multiplier it calibrated."""
    import ml_dtypes

    stem = name.removesuffix('.weight')
    overall = np.float32(np.abs(weight).max() / (6 * 448))
    blocks = np.abs(weight).reshape(weight.shape[0], -1, 16).max(-1) / (6 * overall)
    grid = {stem + '.weight_scale': blocks.astype(ml_dtypes.float8_e4m3fn),
            stem + '.weight_scale_2': np.asarray(overall, np.float32),
            stem + '.input_scale': np.asarray(0.05, np.float32),
            name: np.zeros((weight.shape[0], weight.shape[1] // 2), np.uint8)}
    quantization = {'quant_method': 'modelopt', 'quant_algo': 'NVFP4', 'group_size': 16}
    codec = codecs.source_quantization({'quantization_config': quantization}, grid=grid)
    assert codec is not None
    return codec.encode(name, weight)


@pytest.mark.parametrize('quantized', ['attention', 'experts'])
def test_a_modelopt_nvfp4_model_with_experts_is_read_by_its_stored_layers(tmp_path, quantized):
    """ModelOpt may quantize a model's attention and leave its experts dense: written as one
    algorithm, it loads and computes, input quantizers and all, what the same layers listed
    in a MIXED_PRECISION table compute. Quantized experts are refused either way, since an
    expert stack's input quantizers are computed by no provider here."""
    from dew.interop.safetensors_io import save_hf_layout

    source = FIXTURE.parents[2] / 'hf' / 'qwen3-moe-tiny'
    stored, config = read_weights(source), json.loads((source / 'config.json').read_text())
    marker = '.self_attn.' if quantized == 'attention' else '.mlp.experts.0.'
    tensors, layers = dict(stored), {}
    for name, weight in stored.items():
        if marker in name and name.endswith('_proj.weight'):
            tensors.update(nvfp4(name, np.asarray(weight, np.float32)))
            layers[name.removesuffix('.weight')] = {'quant_algo': 'NVFP4', 'group_size': 16}
    layers_weights = {stem + '.weight' for stem in layers}
    producer = json.loads((FIXTURE / 'config.json').read_text())['quantization_config']['producer']

    def written(directory, quantization):
        quantization = {**quantization, 'kv_cache_quant_algo': None}
        save_hf_layout(tensors, {**config, 'quantization_config': {
            **quantization, 'producer': producer, 'quant_method': 'modelopt'}}, directory)
        (directory / 'hf_quant_config.json').write_text(json.dumps({'producer': producer,
                                                                   'quantization': quantization}))
        return directory

    uniform = written(tmp_path / 'uniform', {'quant_algo': 'NVFP4', 'group_size': 16})
    mixed = written(tmp_path / 'mixed', {'quant_algo': 'MIXED_PRECISION', 'quantized_layers': layers})
    load = {'dtype': 'float32', 'attention_impl': 'reference'}
    if quantized == 'experts':
        for directory in (uniform, mixed):
            with pytest.raises(ValueError):
                Pretrained.load(directory, **load)
        return
    ids = np.load(source / 'input_ids.npy')
    single, table = Pretrained.load(uniform, **load), Pretrained.load(mixed, **load)
    logits = np.asarray(single.model.apply(single.variables, ids))
    np.testing.assert_array_equal(logits, np.asarray(table.model.apply(table.variables, ids)))
    # The same decoded weights with no input quantizers: what the two compute beyond them is
    # the attention inputs' NVFP4 rounding.
    codec = codecs.source_quantization(json.loads((uniform / 'config.json').read_text()))
    assert codec is not None
    decoded = {name: (codec.decode(tensors, name) if name in layers_weights else value)
               for name, value in stored.items()}
    save_hf_layout(decoded, config, tmp_path / 'decoded')
    weights_only = Pretrained.load(tmp_path / 'decoded', **load)
    assert np.abs(logits - np.asarray(weights_only.model.apply(weights_only.variables, ids))).max() > 0


def fp8(name: str, weight: np.ndarray, input_scale: float) -> dict[str, np.ndarray]:
    """`weight` as ModelOpt's FP8 exporter stores it: E4M3 codes under one float32 scale (amax/448),
    beside the static input multiplier it calibrated."""
    import ml_dtypes

    stem = name.removesuffix('.weight')
    grid = {stem + '.weight_scale': np.asarray(np.abs(weight).max() / 448, np.float32),
            stem + '.input_scale': np.asarray(input_scale, np.float32),
            name: np.zeros(weight.shape, ml_dtypes.float8_e4m3fn)}
    quantization = {'quant_method': 'modelopt', 'quant_algo': 'FP8'}
    codec = codecs.source_quantization({'quantization_config': quantization}, grid=grid)
    assert codec is not None
    return codec.encode(name, weight)


def test_a_modelopt_fp8_model_with_fused_projections_quantizes_every_member_input(tmp_path):
    """Phi-3 stores its query, key and value as one `qkv_proj`, and its gate and up as one
    `gate_up_proj`. Written as ModelOpt FP8, each fused weight's one input scale quantizes the one
    input its members read: the loaded model holds that scale at every member Linear and none
    elsewhere, computes bit for bit what the decoded weights compute under those sites' FP8 input
    QDQ, and exports the scale to load back the same model."""
    from flax.traverse_util import flatten_dict, unflatten_dict

    from dew.interop.safetensors_io import save_hf_layout
    from dew.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization

    source = FIXTURE.parents[2] / 'hf' / 'phi3-tiny'
    stored, config = read_weights(source), json.loads((source / 'config.json').read_text())
    scales = {'qkv_proj': 0.05, 'o_proj': 0.03, 'gate_up_proj': 0.07, 'down_proj': 0.02}
    tensors, decoded = dict(stored), dict(stored)
    for name, weight in stored.items():
        projection = name.split('.')[-2]
        if name.startswith('model.layers.') and projection in scales:
            encoded = fp8(name, np.asarray(weight, np.float32), scales[projection])
            tensors.update(encoded)
    producer = json.loads((FIXTURE / 'config.json').read_text())['quantization_config']['producer']
    quantization = {'quant_algo': 'FP8', 'kv_cache_quant_algo': None, 'producer': producer,
                    'quant_method': 'modelopt'}
    save_hf_layout(tensors, {**config, 'quantization_config': quantization}, tmp_path / 'fp8')
    (tmp_path / 'fp8' / 'hf_quant_config.json').write_text(json.dumps(
        {'producer': producer, 'quantization': {'quant_algo': 'FP8', 'kv_cache_quant_algo': None}}))
    codec = codecs.source_quantization(json.loads((tmp_path / 'fp8' / 'config.json').read_text()))
    assert codec is not None
    for name in codec.names(tensors):
        decoded[name] = codec.decode(tensors, name)
    save_hf_layout(decoded, config, tmp_path / 'decoded')

    load = {'dtype': 'float32', 'attention_impl': 'reference'}
    quantized = Pretrained.load(tmp_path / 'fp8', **load)
    dense = Pretrained.load(tmp_path / 'decoded', **load)
    members = {'q_proj': 'qkv_proj', 'k_proj': 'qkv_proj', 'v_proj': 'qkv_proj', 'o_proj': 'o_proj',
               'gate_proj': 'gate_up_proj', 'up_proj': 'gate_up_proj', 'down_proj': 'down_proj'}
    expected = {path[:-1]: scales[members[path[-2]]] for path in flatten_dict(dense.variables['params'])
                if path[-1] == 'kernel' and len(path) > 1 and path[-2] in members}
    held = flatten_dict(quantized.variables[QUANTIZATION])
    assert {path[:-1]: float(value) for path, value in held.items()} == pytest.approx(expected)
    assert {path[-1] for path in held} == {INPUT_SCALES['fp8']}

    scheme = InputQuantization(e4m3_scale=False, format='compressed-tensors',
                               sites=tuple(sorted(('/'.join(path), 'fp8') for path in expected)))
    under = scheme.apply(dense.model)
    variables = {**dense.variables, QUANTIZATION: unflatten_dict(
        {(*path, INPUT_SCALES['fp8']): np.float32(value) for path, value in expected.items()})}
    ids = np.load(source / 'input_ids.npy')
    logits = np.asarray(quantized.model.apply(quantized.variables, ids))
    np.testing.assert_array_equal(logits, np.asarray(under.apply(variables, ids)))
    assert not np.array_equal(logits, np.asarray(dense.model.apply(dense.variables, ids)))

    quantized.save(tmp_path / 'export')
    exported = read_weights(tmp_path / 'export')
    for name, value in scales.items():
        block = 'mlp' if name in ('gate_up_proj', 'down_proj') else 'self_attn'
        np.testing.assert_array_equal(exported[f'model.layers.0.{block}.{name}.input_scale'],
                                      np.asarray(value, np.float32))
    reloaded = Pretrained.load(tmp_path / 'export', **load)
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)), logits)
