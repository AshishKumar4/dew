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
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import load_file, save_file

from dew.interop import Pretrained

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
