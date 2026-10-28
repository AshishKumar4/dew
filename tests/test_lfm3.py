"""Liquid's LFM3 text decoder (`lfm3`) against its own remote code at LiquidAI/LFM3-1.7B@40ebe0a:
tools/remote_code_reference.py writes tests/fixtures/hf/lfm3-tiny, its six layers gated short
convolutions between full and sliding attention layers."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import decode_steps
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import load_file

from dew.interop import Pretrained
from dew.interop.families.lfm3 import LFM3

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'lfm3-tiny'


def fixture(name: str) -> np.ndarray:
    return np.load(TINY / f'{name}.npy')


@pytest.fixture
def release_cache(tmp_path, monkeypatch) -> Path:
    """The hub's cache seeded with stand-ins for LFM3's release modules at the commit Dew is held to
    (`DecoderFamily.release_code`), which a native export reads from it without the network: the
    snapshot directory. The network test fetches the release's own."""
    import huggingface_hub.constants

    code = LFM3.release_code
    snapshot = tmp_path / 'hub' / f"models--{code.repo.replace('/', '--')}" / 'snapshots' / code.revision
    snapshot.mkdir(parents=True)
    for file in code.files:
        (snapshot / file).write_text(f'# {file} at {code.revision}\n')
    monkeypatch.setattr(huggingface_hub.constants, 'HF_HUB_CACHE', str(tmp_path / 'hub'))
    return snapshot


def test_the_text_decoder_computes_the_remote_codes_logits():
    """Gated short convolutions of seven taps, attention with per-head q/k norms and a sigmoid gate a
    head, the full layers turning half of each head at 1e6 and the sliding ones all of it at 1e4 over
    a window of four, sandwich norms and a tied head: the release's logits, on par with its own
    against float64 (check 1)."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    ours = np.asarray(loaded.model.apply(loaded.variables, fixture('input_ids')))
    assert_as_exact_as_the_reference(ours, fixture('logits'), fixture('logits_f64'), 'LFM3 logits')


def test_a_decode_through_the_cache_computes_the_remote_codes_logits():
    """A prefill of four tokens, then one token a step: each convolution's last six inputs and each
    attention layer's keys through the cache, the sliding window over the cached keys; every step's
    logits on par with the release's (check 1)."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(fixture('input_ids'), jnp.int32)
    made, step = decode_steps(loaded.model)
    logits, cache = step(loaded.variables, made(loaded.variables, ids.shape[0]), ids[:, :4])
    steps = [logits[:, -1]]
    for position in range(4, ids.shape[1]):
        logits, cache = step(loaded.variables, cache, ids[:, position:position + 1])
        steps.append(logits[:, -1])
    ours = np.asarray(jnp.stack(steps, axis=1))
    assert_as_exact_as_the_reference(ours, fixture('logits')[:, 3:], fixture('logits_f64')[:, 3:],
                                     'LFM3 decoded logits')


def test_a_saved_decoder_writes_the_releases_tensors_back(tmp_path):
    """A load and a save write every tensor the release stores under its own name, bit for bit."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'saved')
    stored = load_file(TINY / 'model.safetensors')
    written = load_file(tmp_path / 'saved' / 'model.safetensors')
    assert sorted(written) == sorted(stored)
    for name, tensor in stored.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)


@pytest.mark.parametrize('sandwich', [True, False], ids=['sandwich', 'plain'])
def test_a_native_decoder_saves_as_lfm3_and_reloads_the_same_model(tmp_path, sandwich, release_cache):
    """A model built in Dew (`PretrainedDecoder.from_model`) is an LFM3 by its convolution kind, ahead of
    the families that match its gated attention or its sliding layers, and saves a config and tensors
    that load back to it, its logits bit for bit, with sandwich norms or without and with its full and
    sliding layers at their own bases."""
    from dew.interop import PretrainedDecoder

    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    model, variables = loaded.model, loaded.variables
    if not sandwich:
        model = model.clone(sandwich_norms=False)
        variables = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    ids = jnp.asarray(fixture('input_ids'), jnp.int32)
    expected = np.asarray(model.apply(variables, ids))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['model_type'] == 'lfm3'
    restored = Pretrained.load(tmp_path / 'saved', dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, ids)), expected)


def test_a_config_that_omits_its_pattern_and_rotary_reads_lfm3s_own():
    """A config that states no layer_types, rope_parameters or norm_eps reads Lfm3Config's: full
    attention at every fourth layer from the third, sliding from the fifth, convolutions between; the
    full layers turning half of each head at 1e6 and the sliding ones all of theirs at 1e4; an epsilon
    of 1e-5. One naming another activation than silu is refused."""
    from dew.interop.hf_decoders import translate_config

    config = json.loads((TINY / 'config.json').read_text())
    for field in ('layer_types', 'rope_parameters', 'norm_eps'):
        config.pop(field)
    fields = translate_config({**config, 'num_hidden_layers': 9})
    assert fields['layer_types'] == ('conv', 'conv', 'full_attention', 'conv', 'sliding_attention', 'conv',
                                     'full_attention', 'conv', 'sliding_attention')
    assert fields['rope_theta'] == 1e6 and fields['norm_eps'] == 1e-5
    assert fields['kinds']['full_attention']['partial_rotary_factor'] == 0.5
    assert fields['kinds']['sliding_attention']['rope_theta'] == 1e4
    with pytest.raises(ValueError):
        translate_config({**config, 'hidden_act': 'gelu'})
    # A stated rotary names a record for each attention type its layers hold, as Lfm3Config checks.
    for stated in ({}, {'full_attention': None, 'sliding_attention': {'rope_theta': 1e4}},
                   {'full_attention': {'rope_theta': 1e6}}):
        with pytest.raises(ValueError):
            translate_config({**config, 'num_hidden_layers': 9, 'rope_parameters': stated})


def wrapped(directory: Path, **wrapper: object) -> Path:
    """The tiny LFM3 as an LFM3-VL wrapper holds it: its decoder's tensors under
    `model.language_model.`, a vision tower's and a projector's beside them, and `wrapper`'s fields."""
    from safetensors.numpy import save_file

    config = json.loads((TINY / 'config.json').read_text())
    tensors = {f"model.language_model.{name.removeprefix('model.')}": tensor
               for name, tensor in load_file(TINY / 'model.safetensors').items()}
    tensors['model.vision_tower.embedding_norm.weight'] = np.ones(8, np.float32)
    tensors['model.multi_modal_projector.linear_1.weight'] = np.ones((8, 8), np.float32)
    directory.mkdir()
    (directory / 'config.json').write_text(json.dumps({
        'architectures': ['Lfm3VlForConditionalGeneration'], 'model_type': 'lfm3_vl', 'text_config': config,
        'vision_config': {'model_type': 'lfm3_vl_vision', 'hidden_size': 8}, 'image_token_id': 5,
        'downsample_factor': 2, 'projector_hidden_size': 8, 'tie_word_embeddings': True, **wrapper}))
    save_file(tensors, directory / 'model.safetensors')
    return directory


def test_an_lfm3_vl_wrappers_language_model_loads_alone_and_its_unported_tower_refuses(tmp_path):
    """LFM3-VL's wrapper (LiquidAI/LFM3-VL-2B) holds an LFM3 decoder beside a vision tower Dew has no port
    of: `text_only` loads the decoder, computing the decoder's own logits bit for bit, and a multimodal
    load refuses the tower by name. d1's checkpoint of it (LiquidAI/d1-2B), whose full-attention layers
    train under a visibility of their own (`d1_attention`), refuses that by name."""
    from dew.interop.hf_decoders import translate_wrapper_config, translate_wrapper_weights

    ids = jnp.asarray(fixture('input_ids'), jnp.int32)
    alone = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    source = wrapped(tmp_path / 'wrapper')
    text = Pretrained.load(source, dtype='float32', attention_impl='reference', text_only=True)
    np.testing.assert_array_equal(np.asarray(text.model.apply(text.variables, ids)),
                                  np.asarray(alone.model.apply(alone.variables, ids)))
    with pytest.raises(ValueError):
        Pretrained.load(source, dtype='float32', attention_impl='reference')
    config = json.loads((source / 'config.json').read_text())
    with pytest.raises(ValueError):
        translate_wrapper_weights(load_file(source / 'model.safetensors'), translate_wrapper_config(config))
    with pytest.raises(ValueError):
        Pretrained.load(wrapped(tmp_path / 'd1', d1_attention='set', d1_set_size=16), text_only=True)
    # Lfm3VlConfig types its text_config as lfm3 and places its images at 125030 where a config is silent.
    bare = wrapped(tmp_path / 'bare')
    stated = json.loads((bare / 'config.json').read_text())
    stated['text_config'].pop('model_type')
    stated.pop('image_token_id')
    (bare / 'config.json').write_text(json.dumps(stated))
    record = translate_wrapper_config(stated)
    assert (record['text_model_type'], record['image_token_id'], record['video_token_id']) == ('lfm3', 125030,
                                                                                               125029)
    reread = Pretrained.load(bare, dtype='float32', attention_impl='reference', text_only=True)
    np.testing.assert_array_equal(np.asarray(reread.model.apply(reread.variables, ids)),
                                  np.asarray(alone.model.apply(alone.variables, ids)))


@pytest.mark.network
def test_the_released_lfm3_vl_and_d1_headers_load_their_language_model_or_refuse_d1s_attention():
    """LiquidAI/LFM3-VL-2B's language model loads from its safetensors headers alone (`weights=False`,
    `text_only`); LiquidAI/d1-2B's refuses its d1_attention by name."""
    loaded = Pretrained.load('LiquidAI/LFM3-VL-2B', revision='f83d2b0a2296aaa277f394990a9b8b949ddf9691',
                             weights=False, text_only=True)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(loaded.variables))
    with pytest.raises(ValueError):
        Pretrained.load('LiquidAI/d1-2B', revision='cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41', weights=False,
                        text_only=True)


def test_a_saved_remote_code_release_keeps_the_modules_its_auto_map_names(tmp_path):
    """A release whose config names its own code (`auto_map`), which Dew maps natively and never runs,
    saves those modules beside its config, which keeps the auto_map: the directory opens through the
    release's own transformers loader as the release does (ledger 2038). The modules here stand in for
    the release's, which the network test fetches."""
    import shutil

    source = tmp_path / 'release'
    shutil.copytree(TINY, source)
    config = json.loads((source / 'config.json').read_text())
    code = LFM3.release_code
    (source / 'config.json').write_text(json.dumps({**config, 'auto_map': dict(code.auto_map)}))
    for file in code.files:
        (source / file).write_text(f'# {file} of the release\n')
    loaded = Pretrained.load(source, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['auto_map'] == dict(code.auto_map)
    for file in code.files:
        assert (tmp_path / 'saved' / file).read_text() == (source / file).read_text()


def test_a_native_export_writes_its_familys_release_code_beside_itself(tmp_path, release_cache):
    """A model built in Dew of a family whose reference is a release's own code (`release_code`) exports
    with that release's auto_map and its modules at the commit Dew is held to, fetched through the hub's
    cache, so transformers opens the export with that code (ledger 2038)."""
    from dew.interop import PretrainedDecoder

    code = LFM3.release_code
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['auto_map'] == dict(code.auto_map)
    for file in code.files:
        assert (tmp_path / 'saved' / file).read_text() == (release_cache / file).read_text()


@pytest.mark.network
def test_a_native_export_fetches_its_familys_release_code_at_its_commit(tmp_path):
    """The network's half of `test_a_native_export_writes_its_familys_release_code_beside_itself`: the
    release's own modules, fetched at the commit, written beside the export."""
    from huggingface_hub import hf_hub_download

    from dew.interop import PretrainedDecoder

    code = LFM3.release_code
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / 'saved')
    for file in code.files:
        fetched = Path(hf_hub_download(code.repo, file, revision=code.revision))
        assert (tmp_path / 'saved' / file).read_bytes() == fetched.read_bytes()
