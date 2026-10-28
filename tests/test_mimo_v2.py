"""MiMo-V2.6's text decoder (`mimo_v2`) against its own remote code at
XiaomiMiMo/MiMo-V2.6-Pro-MOPD@6a7f202: tools/remote_code_reference.py writes
tests/fixtures/hf/mimo-v2-tiny in the release class's split layout, and the
fused layout the Pro release stores is built here from the same weights as
vLLM reads it (mimo_v2.py `_shard_fp8_qkv_proj` at 78b32b3): one run a key
head, each holding its query, key and value rows in turn."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import load_file, save_file

from dew.interop import Pretrained, codecs
from dew.interop.sources import load_shards

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'mimo-v2-tiny'


def fixture(name):
    return np.load(TINY / f'{name}.npy')


def logits(directory):
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    return loaded, np.asarray(loaded.model.apply(loaded.variables, fixture('input_ids')))


def fused(tensors, config):
    """The split q, k and v projections fused as the Pro release stores them."""
    runs, heads = config['num_key_value_heads'], config['num_attention_heads']
    widths = (heads // runs * config['head_dim'], config['num_key_value_heads'] // runs * config['head_dim'],
              config['num_key_value_heads'] // runs * config['v_head_dim'])
    out = dict(tensors)
    for layer in range(config['num_hidden_layers']):
        stem = f'model.layers.{layer}.self_attn.'
        parts = [out.pop(f'{stem}{part}_proj.weight').reshape(runs, width, -1)
                 for part, width in zip('qkv', widths, strict=True)]
        out[f'{stem}qkv_proj.weight'] = np.concatenate(parts, axis=1).reshape(-1, parts[0].shape[-1])
    return out


def test_the_text_decoder_computes_the_remote_codes_logits():
    """Full and sliding layers, sinks on the sliding ones alone, 12-wide
    queries and keys over 8-wide scaled values, a third of each head rotated
    under each kind's base and grouped sigmoid routing after a dense layer."""
    _, ours = logits(TINY)
    assert_as_exact_as_the_reference(ours, fixture('logits'), fixture('logits_f64'), 'MiMo-V2 logits')


def test_a_fused_qkv_source_loads_to_the_same_model_and_saves_back_fused(tmp_path):
    """The same weights stored fused, in runs over the key heads, load to the
    split source's model leaf for leaf, compute its logits bit for bit, and
    save back to the fused tensors they were read from."""
    config = json.loads((TINY / 'config.json').read_text())
    source = tmp_path / 'fused'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps({**config, 'attention_projection_layout': 'fused_qkv'}))
    tensors = fused(load_file(TINY / 'model.safetensors'), config)
    save_file(tensors, source / 'model.safetensors')
    _, expected = logits(TINY)
    loaded, actual = logits(source)
    np.testing.assert_array_equal(actual, expected)
    loaded.save(tmp_path / 'saved')
    written = load_file(tmp_path / 'saved' / 'model.safetensors')
    assert sorted(written) == sorted(tensors)
    for name, tensor in tensors.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)


@pytest.mark.parametrize('stated', ['absent', 'null'])
def test_a_window_named_by_sliding_window_size_alone_loads_the_same_model(tmp_path, stated):
    """The author's `sliding_window_size` sizes the sliding layers' window
    where `sliding_window` is absent or null: the same model, its logits bit
    for bit, and the config written back as it was published."""
    config = json.loads((TINY / 'config.json').read_text())
    config.pop('sliding_window')
    if stated == 'null':
        config['sliding_window'] = None
    source = tmp_path / 'size'
    shutil.copytree(TINY, source)
    (source / 'config.json').write_text(json.dumps(config))
    expected, reference = logits(TINY)
    loaded, actual = logits(source)
    assert loaded.model == expected.model
    np.testing.assert_array_equal(actual, reference)
    loaded.save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text()) == config


def test_the_towers_and_prediction_layers_ride_along_unread(tmp_path):
    """A release's tower and prediction-layer tensors, which the text decoder
    never runs, leave its logits as they were and are written back as they
    came."""
    source = tmp_path / 'release'
    shutil.copytree(TINY, source)
    tensors = load_file(TINY / 'model.safetensors')
    extra = {'visual.blocks.0.attn.qkv.weight': np.full((6, 4), 0.25, np.float32),
             'audio_encoder.layers.0.mlp.fc1.weight': np.arange(8, dtype=np.float32).reshape(2, 4),
             'speech_embeddings.0.weight': np.ones((3, 4), np.float32),
             'model.mtp.layers.0.enorm.weight': np.full((4,), 2.0, np.float32)}
    save_file({**tensors, **extra}, source / 'model.safetensors')
    loaded, actual = logits(source)
    np.testing.assert_array_equal(actual, logits(TINY)[1])
    loaded.save(tmp_path / 'saved')
    written = load_file(tmp_path / 'saved' / 'model.safetensors')
    for name, tensor in extra.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)


STORED = {'quant_method': 'fp8', 'fmt': 'e4m3', 'weight_block_size': [128, 128],
          'activation_scheme': 'dynamic', 'store_dtype': 'mxfp4', 'mxfp4_block_size': 32}
"""The Pro release's quantization_config, its ignored_layers set per source."""


def test_a_store_dtype_mxfp4_source_loads_its_decoded_weights_and_saves_its_codes_back(tmp_path):
    """The tiny stored as the Pro release stores itself: the fused qkv in FP8
    blocks a run a key head, the dense MLP in FP8 blocks, the routed experts
    in MXFP4 and o_proj as it is (ignored_layers). It loads to the model its
    decoded weights make, logits bit for bit, and saves back every FP8 and
    E2M1 code and every E8M0 exponent it read; an FP8 block's float32 scale
    comes back as amax / 448 of the decoded block, within a rounding of the
    scale it was read with."""
    config = json.loads((TINY / 'config.json').read_text())
    tensors = fused(load_file(TINY / 'model.safetensors'), config)
    store = codecs.mxfp4_store(128, config['num_key_value_heads'])
    quantized = [name for name in tensors if name.endswith(('qkv_proj.weight', 'mlp.gate_proj.weight',
                                                            'mlp.up_proj.weight', 'mlp.down_proj.weight'))]
    stored = store.requantize(tensors, quantized)
    ignored = sorted(name.removesuffix('.weight') for name in tensors if name.endswith('o_proj.weight'))
    source, dense = tmp_path / 'stored', tmp_path / 'decoded'
    for directory, quantization, weights in ((source, {**STORED, 'ignored_layers': ignored}, stored),
                                             (dense, None, store.dequantize(stored))):
        directory.mkdir()
        fields = {**config, 'attention_projection_layout': 'fused_qkv'}
        (directory / 'config.json').write_text(json.dumps(
            fields if quantization is None else {**fields, 'quantization_config': quantization}))
        save_file(weights, directory / 'model.safetensors')
    loaded, actual = logits(source)
    np.testing.assert_array_equal(actual, logits(dense)[1])
    loaded.save(tmp_path / 'saved')
    written = load_shards(tmp_path / 'saved')
    assert sorted(written) == sorted(stored)
    for name, tensor in stored.items():
        if name.endswith('_scale_inv'):
            np.testing.assert_allclose(written[name], tensor, rtol=2 ** -23, atol=0, err_msg=name)
        else:
            assert written[name].dtype == tensor.dtype, name
            np.testing.assert_array_equal(written[name].view(np.uint8), tensor.view(np.uint8), err_msg=name)


@pytest.mark.network
def test_a_saved_release_opens_offline_through_its_own_code_and_computes_its_logits(tmp_path):
    """A release whose config names its own code (`auto_map`), here the tiny checkpoint beside MiMo's code
    at the pinned commit, loads natively and saves its modules and their relative imports
    (`sources.code_modules`) beside its config; a fresh interpreter opens the save offline through that
    code (`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`) and computes logits as
    exact as the release's own over the fixture's tokens (ledger 2038)."""
    import os
    import subprocess
    import sys

    from huggingface_hub import hf_hub_download

    repo, revision = 'XiaomiMiMo/MiMo-V2.6-Pro-MOPD', '6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8'
    released = json.loads(Path(hf_hub_download(repo, 'config.json', revision=revision)).read_text())
    source = tmp_path / 'release'
    shutil.copytree(TINY, source)
    config = json.loads((source / 'config.json').read_text())
    (source / 'config.json').write_text(json.dumps({**config, 'auto_map': released['auto_map']}))
    for file in ('configuration_mimo_v2.py', 'modeling_mimo_v2.py'):
        shutil.copy(hf_hub_download(repo, file, revision=revision), source / file)
    Pretrained.load(source, dtype='float32', attention_impl='reference').save(tmp_path / 'saved')
    script = ("import sys, numpy, torch\n"
              "from transformers import AutoModelForCausalLM\n"
              "model = AutoModelForCausalLM.from_pretrained(sys.argv[1], trust_remote_code=True, "
              "dtype=torch.float32)\n"
              "with torch.no_grad():\n"
              "    ids = torch.from_numpy(numpy.load(sys.argv[2]).astype('int64'))\n"
              "    logits = model(input_ids=ids).logits\n"
              "numpy.save(sys.argv[3], logits.numpy())\n")
    subprocess.run([sys.executable, '-c', script, str(tmp_path / 'saved'), str(TINY / 'input_ids.npy'),
                    str(tmp_path / 'logits.npy')], check=True,
                   env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'HF_MODULES_CACHE': str(tmp_path / 'modules')})
    assert_as_exact_as_the_reference(np.load(tmp_path / 'logits.npy'), np.load(TINY / 'logits.npy'),
                                     np.load(TINY / 'logits_f64.npy'), 'MiMo-V2 saved through its code')

