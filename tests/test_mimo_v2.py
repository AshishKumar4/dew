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
from interop_support import respelled
from model_support import encoded_form
from safetensors.numpy import load_file, save_file

from dew.interop import Pretrained, codecs
from dew.interop.sources import load_shards

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'mimo-v2-tiny'
WHOLE = Path(__file__).parent / 'fixtures' / 'hf' / 'mimo-v2-vl-tiny'
"""The release with its towers, its audio tokenizer's folder and its prediction layer
(tools/remote_code_reference.py), whose check 1 against the release is
tests/test_remote_code_releases.py's."""


def fixture(name):
    return np.load(TINY / f'{name}.npy')


def logits(directory):
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    return loaded, np.asarray(loaded.model.apply(loaded.variables, fixture('input_ids')))


def fused(tensors, config):
    """The split q, k and v projections fused as the Pro release stores them, in runs over each layer's key
    heads: a sliding layer's of its `swa_*` geometry, a full layer's of the model's."""
    out = dict(tensors)
    for layer, slides in enumerate(config['hybrid_layer_pattern']):
        prefix = 'swa_' if slides else ''
        runs, heads = config[f'{prefix}num_key_value_heads'], config[f'{prefix}num_attention_heads']
        width = config[f'{prefix}head_dim']
        widths = (heads // runs * width, width, config['v_head_dim'])
        stem = f'model.layers.{layer}.self_attn.'
        parts = [out.pop(f'{stem}{part}_proj.weight').reshape(runs, part_width, -1)
                 for part, part_width in zip('qkv', widths, strict=True)]
        out[f'{stem}qkv_proj.weight'] = np.concatenate(parts, axis=1).reshape(-1, parts[0].shape[-1])
    return out


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


def flash_ratio_source(directory: Path) -> Path:
    """A trunk whose sliding layers hold MiMo-V2-Flash's twice the full layers' key heads but query heads
    and a width of their own (`SLIDING`), which Flash's attention fixes at the model's: its sliding layers'
    projections and sinks drawn at their shapes, at `directory`."""
    config = {**json.loads((TINY / 'config.json').read_text()), 'swa_num_attention_heads': 8,
              'swa_num_key_value_heads': 4, 'swa_head_dim': 24}
    rng = np.random.default_rng(4290)
    tensors = dict(load_file(TINY / 'model.safetensors'))
    heads, kv, width, value = 8, 4, 24, config['v_head_dim']
    for layer, slides in enumerate(config['hybrid_layer_pattern']):
        if not slides:
            continue
        stem = f'model.layers.{layer}.self_attn.'
        for name, shape in (('q_proj.weight', (heads * width, 32)), ('k_proj.weight', (kv * width, 32)),
                            ('v_proj.weight', (kv * value, 32)), ('o_proj.weight', (32, heads * value)),
                            ('attention_sink_bias', (heads,))):
            tensors[stem + name] = (0.3 * rng.normal(size=shape)).astype(np.float32)
    directory.mkdir()
    (directory / 'config.json').write_text(json.dumps(config))
    save_file(tensors, directory / 'model.safetensors')
    return directory


@pytest.mark.parametrize('source', ['tiny', 'flash-ratio'])
def test_a_native_export_writes_mimo_v2_and_loads_back_the_same_model(tmp_path, monkeypatch, source):
    """A MiMo-V2 model Dew holds (the fixture's: sinks on its sliding layers alone, over the full layers'
    key heads; or over Flash's twice the key heads, its sliding query heads and width its own,
    `flash_ratio_source`) exports natively (`PretrainedDecoder.from_model`) as MiMo-V2 under its authors'
    config, with its release's code named (`auto_map`) and its q, k and v stored apart, not as
    MiMo-V2-Flash, whose sliding layers hold twice the key heads and the model's query heads and width;
    the export loads back to the same variables, leaf for leaf (ledgers 2192, 4290). The release's code
    comes from the hub's cache, seeded offline at the commit Dew holds it to from the release's files
    vendored into the fetched store."""
    import huggingface_hub.constants
    import jax
    from tool_support import run_from

    from dew.interop import PretrainedDecoder, release_pins

    with run_from("remote_code_reference", "tools") as reference:
        repo, revision, files, _ = reference.RELEASES["mimo-v2-tiny"]
        snapshot = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        for file in files:
            shutil.copy(reference.release_file("mimo-v2-tiny", file), snapshot / file)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    directory = TINY if source == 'tiny' else flash_ratio_source(tmp_path / 'source')
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / 'export')
    written = json.loads((tmp_path / 'export' / 'config.json').read_text())
    assert (written['model_type'], written['architectures']) == ('mimo_v2', ['MiMoV2ForCausalLM'])
    assert written['auto_map'] == release_pins.MIMO_V2[2]
    assert written['attention_projection_layout'] == 'split'
    assert not any('qkv_proj' in name for name in load_file(tmp_path / 'export' / 'model.safetensors'))
    again = Pretrained.load(tmp_path / 'export', dtype='float32', attention_impl='reference')
    assert again.model_config == loaded.model_config
    jax.tree.map(np.testing.assert_array_equal, again.variables, loaded.variables)


@pytest.mark.parametrize('stated', ['absent', 'null'])
def test_a_window_named_by_sliding_window_size_alone_loads_the_same_model(tmp_path, stated):
    """The author's `sliding_window_size` sizes the sliding layers' window
    where `sliding_window` is absent or null: the same model, its logits bit
    for bit, and the config written back as it was published."""
    config = json.loads((TINY / 'config.json').read_text())
    config.pop('sliding_window')
    if stated == 'null':
        config['sliding_window'] = None
    source = respelled(TINY, tmp_path / 'size', lambda _: config)
    expected, reference = logits(TINY)
    loaded, actual = logits(source)
    assert loaded.model == expected.model
    np.testing.assert_array_equal(actual, reference)
    loaded.save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text()) == config


def test_the_towers_of_a_config_that_names_none_ride_along_unread(tmp_path):
    """A checkpoint's tower tensors beside a config that names no tower, which
    its text decoder never runs, leave its logits as they were and are
    written back as they came."""
    source = tmp_path / 'release'
    shutil.copytree(TINY, source)
    tensors = load_file(TINY / 'model.safetensors')
    extra = {'visual.blocks.0.attn.qkv.weight': np.full((6, 4), 0.25, np.float32),
             'audio_encoder.layers.0.mlp.fc1.weight': np.arange(8, dtype=np.float32).reshape(2, 4),
             'speech_embeddings.0.weight': np.ones((3, 4), np.float32)}
    save_file({**tensors, **extra}, source / 'model.safetensors')
    loaded, actual = logits(source)
    np.testing.assert_array_equal(actual, logits(TINY)[1])
    loaded.save(tmp_path / 'saved')
    written = load_file(tmp_path / 'saved' / 'model.safetensors')
    for name, tensor in extra.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)


def test_a_release_with_its_towers_loads_whole_and_saves_back_as_it_came(tmp_path):
    """MiMo-V2.6 with its towers is a bundle its family reads whole (row 357): the decoder at the config's
    root, the ViT and its merger, the audio encoder over the codes of the tokenizer the release keeps in a
    folder of its own, and the prediction layer vLLM drafts with. It saves back, and so does its language
    model alone (`text_only`), every tensor it read under its own name, the folder's in the folder beside
    the folder's config, the config as it was published, and loads again to the same variables; its
    language model alone (`text_only`) computes the bundle's logits over text bit for bit."""
    import jax

    from dew.nn.multimodal import MultimodalTransformer

    loaded = Pretrained.load(WHOLE, dtype='float32', attention_impl='reference')
    model = loaded.model
    assert isinstance(model, MultimodalTransformer)
    assert None not in (model.vision, model.projection, model.audio)
    assert model.num_nextn_predict_layers == 1
    loaded.save(tmp_path / 'saved')
    text = Pretrained.load(WHOLE, dtype='float32', attention_impl='reference', text_only=True)
    # Its language model alone keeps every other tensor, the folder's among them, and writes it back.
    text.save(tmp_path / 'text')
    for folder, root in ((Path(), 'saved'), (Path('audio_tokenizer'), 'saved'), (Path(), 'text'),
                         (Path('audio_tokenizer'), 'text')):
        saved, released = tmp_path / root / folder, WHOLE / folder
        assert (json.loads((saved / 'config.json').read_text())
                == json.loads((released / 'config.json').read_text()))
        written, read = load_file(saved / 'model.safetensors'), load_file(released / 'model.safetensors')
        assert sorted(written) == sorted(read)
        for name, tensor in read.items():
            np.testing.assert_array_equal(written[name], tensor, err_msg=name)
    for root in ('saved', 'text'):
        again = Pretrained.load(tmp_path / root, dtype='float32', attention_impl='reference')
        jax.tree.map(np.testing.assert_array_equal, again.variables, loaded.variables)
    ids = np.load(WHOLE / 'input_ids.npy')
    np.testing.assert_array_equal(np.asarray(model.apply(loaded.variables, ids)),
                                  np.asarray(text.model.apply(text.variables, ids)))


def test_a_release_without_its_audio_tokenizer_is_refused_whole_and_loads_as_text(tmp_path):
    """A release whose config names its audio tower and whose folder lacks the tokenizer the tower reads
    is refused whole, before any weight is read; its language model alone still loads."""
    source = tmp_path / 'release'
    shutil.copytree(WHOLE, source)
    shutil.rmtree(source / 'audio_tokenizer')
    with pytest.raises(ValueError):
        Pretrained.load(source, dtype='float32', attention_impl='reference')
    text = Pretrained.load(source, dtype='float32', attention_impl='reference', text_only=True)
    assert text.model.num_nextn_predict_layers == 1


@pytest.mark.parametrize('emptied', [('audio_config',), ('vision_config',),
                                     ('vision_config', 'audio_config')],
                         ids=['vision alone', 'audio alone', 'text alone'])
def test_a_tower_section_that_states_nothing_builds_no_tower(tmp_path, emptied):
    """MiMoV2Config reads an absent tower section as `{}` and MiMoV2ForCausalLM builds a tower only for a
    section that states something, so a release that saves one empty is the model without that tower: its
    other tower loads, without the audio tokenizer's folder where audio is gone, an audio-only release
    states no image placeholder, and with neither the release is its text decoder. The decoder computes
    the whole release's logits over text, bit for bit, and the absent tower's tensors ride along."""
    from dew.nn.multimodal import MultimodalTransformer

    def edit(config):
        config.update({section: {} for section in emptied})
        if 'vision_config' in emptied:
            config['image_token_id'] = None
        return config

    source = respelled(WHOLE, tmp_path / 'release', edit)
    if 'audio_config' in emptied:
        shutil.rmtree(source / 'audio_tokenizer')
    loaded = Pretrained.load(source, dtype='float32', attention_impl='reference')
    whole = Pretrained.load(WHOLE, dtype='float32', attention_impl='reference')
    model = loaded.model
    if len(emptied) == 2:
        assert not isinstance(model, MultimodalTransformer)
    else:
        assert isinstance(model, MultimodalTransformer)
        assert (model.vision is None, model.audio is None) == ('vision_config' in emptied,
                                                              'audio_config' in emptied)
    ids = np.load(WHOLE / 'input_ids.npy')
    np.testing.assert_array_equal(np.asarray(model.apply(loaded.variables, ids)),
                                  np.asarray(whole.model.apply(whole.variables, ids)))


@pytest.mark.parametrize('stated', [None, 0, 1])
def test_a_trunk_of_full_layers_alone_builds_the_prediction_layer_its_config_states(stated):
    """A prediction layer attends with the sliding geometry its config states (vLLM's `MiMoV2MTPLayer`,
    whatever the trunk's pattern): a trunk of full layers alone builds none where its config states none
    or 0 (Dew's export of such a model), and one it states attends with the `swa_*` query and key heads and
    width, the window, the sliding base and the sliding sinks, in a kind no trunk layer has. Its values are
    the model's width, which no kind sets apart, so another value width is refused by name."""
    from dew.interop.hf_decoders import translate_config

    config = {**json.loads((TINY / 'config.json').read_text()), 'hybrid_layer_pattern': [0, 0, 0, 0],
              **PREDICTOR}
    if stated is not None:
        config['num_nextn_predict_layers'] = stated
    fields = translate_config(config)
    assert fields['num_nextn_predict_layers'] == (stated or 0)
    if stated:
        attending = fields['kinds'][fields['mtp_layer_type']]
        assert fields['mtp_layer_type'] not in fields['layer_types']
        assert ((attending['num_heads'], attending['head_dim'], attending['num_kv_heads'],
                 attending['window'], attending['rope_theta'], attending['sinks'])
                == (2, 24, 1, config['sliding_window_size'], config['swa_rope_theta'], True))
        with pytest.raises(ValueError):
            translate_config({**config, 'swa_v_head_dim': 12})


PREDICTOR = {'swa_num_attention_heads': 2, 'swa_num_key_value_heads': 1, 'swa_head_dim': 24,
             'swa_v_head_dim': 8}
"""A prediction layer's sliding geometry other than the tiny trunk's: half its query heads, one key head,
twice its width."""


def test_a_fused_prediction_layer_of_its_own_geometry_loads_and_saves_back_as_it_came(tmp_path):
    """A trunk of full layers alone, stored fused as the Pro release stores itself, beside a prediction layer
    whose fused q, k and v runs are its own geometry (`PREDICTOR`), as vLLM reads them: the layer's runs
    split by its own kind (`Packed.owner`), its depth built of them, and every tensor written back as it
    came, fused in its own runs."""
    config = {**json.loads((TINY / 'config.json').read_text()), 'hybrid_layer_pattern': [0, 0, 0, 0],
              'attention_projection_layout': 'fused_qkv', 'num_nextn_predict_layers': 1, **PREDICTOR}
    rng = np.random.default_rng(357)
    tensors = {name: tensor for name, tensor in load_file(TINY / 'model.safetensors').items()
               if not name.endswith('attention_sink_bias')}
    stem, width, inner = 'model.mtp.layers.0.', config['hidden_size'], config['intermediate_size']
    run = PREDICTOR['swa_num_attention_heads'] * PREDICTOR['swa_head_dim'] + PREDICTOR['swa_head_dim'] + 8
    drawn = {'enorm.weight': (width,), 'hnorm.weight': (width,), 'eh_proj.weight': (width, 2 * width),
             'input_layernorm.weight': (width,), 'self_attn.qkv_proj.weight': (run, width),
             'self_attn.o_proj.weight': (width, PREDICTOR['swa_num_attention_heads'] * 8),
             'self_attn.attention_sink_bias': (PREDICTOR['swa_num_attention_heads'],),
             'pre_mlp_layernorm.weight': (width,), 'mlp.gate_proj.weight': (inner, width),
             'mlp.up_proj.weight': (inner, width), 'mlp.down_proj.weight': (width, inner),
             'final_layernorm.weight': (width,)}
    tensors = {**fused(tensors, config),
               **{stem + name: rng.normal(size=shape).astype(np.float32) for name, shape in drawn.items()}}
    source = tmp_path / 'fused'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps(config))
    save_file(tensors, source / 'model.safetensors')
    loaded, _ = logits(source)
    assert loaded.model.num_nextn_predict_layers == 1
    loaded.save(tmp_path / 'saved')
    written = load_file(tmp_path / 'saved' / 'model.safetensors')
    assert sorted(written) == sorted(tensors)
    for name, tensor in tensors.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)


SLIDING_TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'mimo-v2-sliding-tiny'
"""The tiny trunk whose sliding layers attend in a geometry of their own, half the full layers' query
heads, one key head and twice their width, their values the model's, its weights the release's class drew
(tools/remote_code_reference.py --draw), its logits held to that class's by tests/test_remote_code_releases.py
(ledger 4290)."""

SLIDING = ('swa_num_attention_heads', 'swa_num_key_value_heads', 'swa_head_dim')
"""The sliding layers' own geometry, as MiMoV2Attention reads it."""


def sliding_source(directory: Path, layout: str) -> dict:
    """`SLIDING_TINY` stored `layout` ('split', as the release draws it, or fused in each layer's own runs,
    'fused_qkv') at `directory`: the tensors it holds."""
    config = {**json.loads((SLIDING_TINY / 'config.json').read_text()), 'attention_projection_layout': layout}
    tensors = dict(load_file(SLIDING_TINY / 'model.safetensors'))
    if layout == 'fused_qkv':
        tensors = fused(tensors, config)
    directory.mkdir()
    (directory / 'config.json').write_text(json.dumps(config))
    save_file(tensors, directory / 'model.safetensors')
    return tensors


def test_sliding_layers_of_their_own_geometry_load_split_or_fused_and_save_back_as_they_came(tmp_path):
    """A trunk whose sliding layers state query heads, key heads and a width of their own (`SLIDING_TINY`,
    MiMoV2Attention's `swa_*`) builds them in a kind of that geometry, the full layers in the model's,
    whose logits tests/test_remote_code_releases.py holds to the release's; stored fused, each layer's runs
    split by its own kind (`Packed.owner`), it computes the split source's logits bit for bit; each saves
    back every tensor as it came and the `swa_*` geometry it was read with. Values of another width than
    the model's, which no kind sets apart, are refused by name (ledger 4290)."""
    from dew.interop.hf_decoders import translate_config

    split = sliding_source(tmp_path / 'split', 'split')
    fused_tensors = sliding_source(tmp_path / 'fused', 'fused_qkv')
    config = json.loads((tmp_path / 'split' / 'config.json').read_text())
    sliding = translate_config(config)['kinds']['sliding_attention']
    assert (sliding['num_heads'], sliding['num_kv_heads'], sliding['head_dim']) == (2, 1, 24)
    with pytest.raises(ValueError):
        translate_config({**config, 'swa_v_head_dim': 12})
    loaded, expected = logits(tmp_path / 'split')
    reread, actual = logits(tmp_path / 'fused')
    np.testing.assert_array_equal(actual, expected)
    for source, tensors in ((loaded, split), (reread, fused_tensors)):
        saved = tmp_path / f'saved-{len(tensors)}'
        source.save(saved)
        written = load_file(saved / 'model.safetensors')
        assert sorted(written) == sorted(tensors)
        for name, tensor in tensors.items():
            np.testing.assert_array_equal(written[name], tensor, err_msg=name)
        stated = json.loads((saved / 'config.json').read_text())
        assert {key: stated[key] for key in SLIDING} == {key: config[key] for key in SLIDING}


PROCESSOR = WHOLE / 'processor.npz'
"""vLLM's MiMo processor over raw images, recorded by tools/mimo_reference.py --record-images."""


def test_a_raw_image_is_processed_as_vllms_mimo_processor_processes_it():
    """MiMo-V2.6 ships no processor, only its config's `processor_config`: Dew's own
    (`dew.interop.mimo_media`) resizes, standardizes and cuts a drawn raw image into the tower's patch
    rows on par with vLLM's MiMo processor from its float64 run's (check 1, the recording's); resizes a
    table of image sizes to the sides vLLM's does; and lays a prompt around an image out as vLLM's
    `process` does, token for token, and as the model's inputs (`Processor.from_hf`), a chat's image part
    alike. Raw videos and audio, which it does not process, are refused (ledger 4289)."""
    import jax
    import jax.numpy as jnp
    from model_support import recorded_site
    from reference_error import as_given, assert_as_exact
    from tool_support import run_from

    from dew.interop.mimo_media import MiMoProcessor, smart_resize

    loaded = Pretrained.load(WHOLE, dtype='float32', attention_impl='reference')
    processor = loaded.processor._media_reference
    assert isinstance(processor, MiMoProcessor)
    with run_from('mimo_reference', 'tools') as module:
        images_draw, sizes = module.images_draw, module.IMAGE_SIZES

    def twin(inputs):
        with jax.enable_x64(new_val=True):
            return {'patches': processor.patch_rows(jnp.asarray(inputs['image'], jnp.float64))}

    for height, width in sizes:
        site = f'mimo image patches {height}x{width}'
        draw, reference, truth, seeds = recorded_site(PROCESSOR, site, images_draw(height, width), twin)
        assert_as_exact(as_given(lambda index, inputs: {'patches': processor.patches(inputs['image'])}),
                        reference, draw, site, seeds=seeds, truth_fn=truth)
    with np.load(PROCESSOR) as recorded:
        sides, resized, prompt = recorded['sides'], recorded['resized'], recorded['prompt_ids']
    factor = processor.patch * processor.merge
    assert [smart_resize(*side, factor, processor.least, processor.most) for side in sides.tolist()] == \
        [tuple(pair) for pair in resized.tolist()]
    image = images_draw(*sizes[1])(0)['image']
    fields = processor(text='w1 w2<|vision_start|><|image|><|vision_end|>w0', images=[image])
    np.testing.assert_array_equal(fields['input_ids'][0], prompt)
    inputs = loaded.processor('w1 w2<|vision_start|><|image|><|vision_end|>w0', images=[image])
    np.testing.assert_array_equal(np.asarray(inputs.tokens)[0], prompt)
    # A chat's image part, its template writing the placeholder: the images the parts hold, processed.
    template = ("{% for message in messages %}{% for part in message['content'] %}"
                "{% if part['type'] == 'image' %}<|vision_start|><|image|><|vision_end|>"
                "{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endfor %}")
    parts = [{'type': 'text', 'text': 'w1 w2'}, {'type': 'image', 'image': image},
             {'type': 'text', 'text': 'w0'}]
    chat = loaded.processor.chat([{'role': 'user', 'content': parts}], add_generation_prompt=False,
                                 chat_template=template)
    np.testing.assert_array_equal(np.asarray(chat.tokens)[0], prompt)
    with pytest.raises(ValueError):
        processor(text='w1', audio=[np.zeros(240, np.float32)])
    # A text chat of the language model alone, which reads no media processor, runs its tokenizer's
    # template.
    text = Pretrained.load(WHOLE, dtype='float32', attention_impl='reference', text_only=True).processor
    held = text.chat([{'role': 'user', 'content': [{'type': 'text', 'text': 'w1 w2'}]}],
                     add_generation_prompt=False, chat_template=template)
    np.testing.assert_array_equal(np.asarray(held.tokens)[0], text.reference.encode('w1 w2'))


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
    stored = encoded_form(store, tensors, quantized)
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


OPENED = """import json, sys, torch
from transformers import AutoModelForCausalLM
models = []
for directory in sys.argv[1:]:
    model, info = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True,
                                                       dtype=torch.float32, output_loading_info=True)
    report = {key: sorted(value) for key, value in info.items()}
    models.append((type(model).__name__, report, model.state_dict()))
(name, info, state), (other, other_info, other_state) = models
same = sorted(state) == sorted(other_state) and all(torch.equal(state[key], other_state[key])
                                                    for key in state)
print(json.dumps({"classes": [name, other], "info": [info, other_info], "same": same}))
"""
"""A fresh interpreter's offline loads of two directories through their own code's transformers loader:
the classes built, the loader's reports of the weights matched, and whether the two state dicts are
equal, tensor for tensor."""


@pytest.mark.network
def test_a_saved_release_opens_offline_through_its_own_code_as_the_release_does(tmp_path):
    """A release whose config names its own code (`auto_map`), here the tiny checkpoint beside MiMo's code
    at the pinned commit, loads natively and saves its modules and their relative imports
    (`sources.code_modules`; modeling_mimo_v2.py imports configuration_mimo_v2.py) beside its config; a
    fresh interpreter opens the save offline through that code
    (`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`) as it opens the release: the
    release's class, every weight matched, and the same state, tensor for tensor (ledger 2038)."""
    import os
    import subprocess
    import sys

    from huggingface_hub import hf_hub_download

    repo, revision = 'XiaomiMiMo/MiMo-V2.6-Pro-MOPD', '6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8'
    released = json.loads(Path(hf_hub_download(repo, 'config.json', revision=revision)).read_text())
    source = respelled(TINY, tmp_path / 'release',
                       lambda config: {**config, 'auto_map': released['auto_map']})
    for file in ('configuration_mimo_v2.py', 'modeling_mimo_v2.py'):
        shutil.copy(hf_hub_download(repo, file, revision=revision), source / file)
    Pretrained.load(source, dtype='float32', attention_impl='reference').save(tmp_path / 'saved')
    run = subprocess.run([sys.executable, '-c', OPENED, str(tmp_path / 'saved'), str(source)], check=True,
                         capture_output=True, text=True,
                         env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                              'HF_MODULES_CACHE': str(tmp_path / 'modules')})
    opened = json.loads(run.stdout.strip().splitlines()[-1])
    assert opened['classes'] == ['MiMoV2ForCausalLM'] * 2
    matched = {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': []}
    assert opened['info'] == [matched, matched] and opened['same']
