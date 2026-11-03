"""Liquid's LFM3 text decoder (`lfm3`) at LiquidAI/LFM3-1.7B@40ebe0a: tools/remote_code_reference.py
writes tests/fixtures/hf/lfm3-tiny, its six layers gated short convolutions between full and sliding
attention layers, whose release outputs tests/test_remote_code_releases.py holds Dew to."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import respelled
from safetensors.numpy import load_file

from dew.interop import Pretrained
from dew.interop.families.lfm3 import LFM3

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'
TINY = FIXTURES / 'lfm3-tiny'
VL = FIXTURES / 'lfm3-vl-tiny'


def fixture(name: str) -> np.ndarray:
    return np.load(TINY / f'{name}.npy')


RELEASED = {'configuration_lfm3.py': 'from transformers import PretrainedConfig\n',
            'modeling_lfm3.py': 'from .lfm3_conv import conv\n', 'lfm3_conv.py': 'conv = None\n'}
"""Stand-ins for LFM3's release modules: its config's two auto_map modules, the modeling one importing a
third relatively, as the release's does (modeling_lfm3.py:48)."""


@pytest.fixture
def release_cache(tmp_path, monkeypatch) -> Path:
    """The hub's cache seeded with stand-ins for LFM3's release modules (`RELEASED`) at the commit Dew is
    held to (`DecoderFamily.release_code`), which a native export reads from it without the network:
    the snapshot directory. The network test fetches the release's own."""
    import huggingface_hub.constants

    code = LFM3.release_code
    repository = f"models--{str(code.source).replace('/', '--')}"
    snapshot = tmp_path / 'hub' / repository / 'snapshots' / code.revision
    snapshot.mkdir(parents=True)
    for file, text in RELEASED.items():
        (snapshot / file).write_text(text)
    monkeypatch.setattr(huggingface_hub.constants, 'HF_HUB_CACHE', str(tmp_path / 'hub'))
    return snapshot


@pytest.mark.parametrize('fixture', [TINY, VL], ids=['decoder', 'wrapper'])
def test_a_saved_source_writes_the_releases_tensors_back(tmp_path, fixture):
    """A load and a save write every tensor the release stores under its own name, bit for bit, and its
    config's model type: LFM3's decoder, and LFM3-VL's wrapper, its trunk's, projector's and decoder's."""
    loaded = Pretrained.load(fixture, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'saved')
    stored = load_file(fixture / 'model.safetensors')
    written = load_file(tmp_path / 'saved' / 'model.safetensors')
    assert sorted(written) == sorted(stored)
    for name, tensor in stored.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)
    model_type = json.loads((fixture / 'config.json').read_text())['model_type']
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['model_type'] == model_type


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
        variables = nn.unbox(model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    ids = jnp.asarray(fixture('input_ids'), jnp.int32)
    expected = np.asarray(model.apply(variables, ids))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['model_type'] == 'lfm3'
    if sandwich:
        # The release's own tensor names, its attention's gate_proj beside its feed-forward's w1.
        written = load_file(tmp_path / 'saved' / 'model.safetensors')
        assert sorted(written) == sorted(load_file(TINY / 'model.safetensors'))
    restored = Pretrained.load(tmp_path / 'saved', dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, ids)), expected)


@pytest.mark.parametrize("arm", ["plain", "looped"])
def test_plain_and_looped_lfm3_train_through_a_decision_run(tmp_path, release_cache, arm):
    """Plain and looped LFM3 (LiquidAI/LFM3-1.7B at da90f4c, whose config, code and weights are
    40ebe0a's, which Dew is held to; ledger 2011): a decision run from an LFM3 release trains at a fixed
    rate, plain and with a block of its layers looped twice (`--loop-layers`, `--loop-steps`), a short
    convolution and full and sliding attention among them, under a text-only decision source (its
    language model, no prediction depth, its full-attention layers reading blocks both ways): every
    weight it trains finite, each kind of layer's weights moved, its short convolutions', its full
    attention's and its sliding attention's, and the run records its loop."""
    import shutil

    from model_support import flat_tree
    from test_decision_recipe import QWEN, TICKETS

    from dew.config import SourceOptions
    from dew.decision.config import DecisionRunConfig
    from dew.interop import PretrainedDecoder

    # The tiny release at the vocabulary of a tokenizer a decision layout reads its specials from.
    model = Pretrained.load(TINY, dtype='float32', attention_impl='reference').model.clone(vocab_size=289)
    variables = nn.unbox(model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / 'release')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        shutil.copy(QWEN / name, tmp_path / 'release' / name)
    looped = ["--loop-layers", "1", "5", "--loop-steps", "2"] if arm == "looped" else []
    config = DecisionRunConfig.cli([
        "--pretrained", str(tmp_path / 'release'), "--trainer.name", arm,
        "--trainer.checkpoint-dir", str(tmp_path / "runs"), "--trainer.steps", "2",
        "--trainer.eval-every", "None", "--trainer.checkpoint-every", "None", "--trainer.log-every", "1",
        "--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None",
        "--model.dtype", "float32", "--optim.learning-rate", "5e-06", *looped],
        default=DecisionRunConfig(data=TICKETS, source=SourceOptions(
            text_only=True, prediction_depths=0, attention={"full_attention": "block_bidirectional"})))
    state = config.run()
    trained = flat_tree(jax.device_get(state.variables["params"]["backbone"]))
    start = flat_tree(variables["params"])
    assert set(trained) == set(start)
    assert all(np.isfinite(trained[name]).all() for name in trained)
    kinds = json.loads((TINY / 'config.json').read_text())['layer_types']
    for kind in set(kinds):
        layers = {f"layers_{index}." for index, layer in enumerate(kinds) if layer == kind}
        moved = [name for name in trained if any(name.startswith(layer) for layer in layers)
                 and not np.array_equal(trained[name], start[name])]
        assert moved, kind
    recorded = DecisionRunConfig.load(str(tmp_path / "runs" / arm))
    assert (recorded.loop_layers, recorded.loop_steps) == (((1, 5), 2) if looped else (None, None))


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
    """The tiny LFM3 as an LFM3-VL wrapper holds it: its decoder's tensors under `model.language_model.`,
    lfm3-vl-tiny's vision tower and projector beside them, under lfm3-vl-tiny's config with `wrapper`'s
    fields."""
    from safetensors.numpy import save_file

    config = json.loads((TINY / 'config.json').read_text())
    tensors = {f"model.language_model.{name.removeprefix('model.')}": tensor
               for name, tensor in load_file(TINY / 'model.safetensors').items()}
    tensors.update({name: tensor for name, tensor in load_file(VL / 'model.safetensors').items()
                    if name.startswith(('model.vision_tower.', 'model.multi_modal_projector.'))})
    directory.mkdir()
    (directory / 'config.json').write_text(json.dumps({
        **json.loads((VL / 'config.json').read_text()), 'text_config': config, **wrapper}))
    save_file(tensors, directory / 'model.safetensors')
    return directory


def test_an_lfm3_vl_wrapper_loads_whole_and_its_language_model_alone(tmp_path):
    """LFM3-VL's wrapper (LiquidAI/LFM3-VL-2B) holds an LFM3 decoder beside its vision trunk and projector:
    a load reads all three, and on ids without placeholders computes its decoder's own logits bit for
    bit, as `text_only`, which loads the decoder alone, does. d1's checkpoint of it (LiquidAI/d1-2B),
    whose full-attention layers train under a visibility of their own (`d1_attention`), refuses that by
    name."""
    from dew.interop.hf_decoders import translate_wrapper_config

    ids = jnp.asarray(fixture('input_ids'), jnp.int32)
    alone = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    expected = np.asarray(alone.model.apply(alone.variables, ids))
    source = wrapped(tmp_path / 'wrapper')
    text = Pretrained.load(source, dtype='float32', attention_impl='reference', text_only=True)
    np.testing.assert_array_equal(np.asarray(text.model.apply(text.variables, ids)), expected)
    whole = Pretrained.load(source, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(np.asarray(whole.model.apply(whole.variables, ids)), expected)
    with pytest.raises(ValueError):
        Pretrained.load(wrapped(tmp_path / 'd1', d1_attention='set', d1_set_size=16), text_only=True)
    # Lfm3VlConfig types its text_config as lfm3 and places its images at 125030 and its videos at 125029
    # where a config is silent.
    bare = wrapped(tmp_path / 'bare')
    stated = json.loads((bare / 'config.json').read_text())
    stated['text_config'].pop('model_type')
    stated.pop('image_token_id')
    stated.pop('video_token_id')
    (bare / 'config.json').write_text(json.dumps(stated))
    record = translate_wrapper_config(stated)
    assert (record['text_model_type'], record['image_token_id'], record['video_token_id']) == ('lfm3', 125030,
                                                                                               125029)
    reread = Pretrained.load(bare, dtype='float32', attention_impl='reference', text_only=True)
    np.testing.assert_array_equal(np.asarray(reread.model.apply(reread.variables, ids)), expected)


@pytest.mark.network
def test_the_released_lfm3_vl_and_d1_headers_load_whole_or_refuse_d1s_attention():
    """LiquidAI/LFM3-VL-2B loads whole, its trunk, projector and language model, from its safetensors
    headers alone (`weights=False`), every released tensor read; LiquidAI/d1-2B's refuses its d1_attention
    by name."""
    loaded = Pretrained.load('LiquidAI/LFM3-VL-2B', revision='f83d2b0a2296aaa277f394990a9b8b949ddf9691',
                             weights=False)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(loaded.variables))
    with pytest.raises(ValueError):
        Pretrained.load('LiquidAI/d1-2B', revision='cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41', weights=False,
                        text_only=True)


def test_a_saved_remote_code_release_keeps_the_modules_its_auto_map_names(tmp_path):
    """A release whose config names its own code (`auto_map`), which Dew maps natively and never runs,
    saves those modules and every module they import relatively, as transformers resolves them, beside
    its config, which keeps the auto_map, and no other file of its: the directory opens through the
    release's own transformers loader as the release does (ledger 2038). The modules here stand in for
    the release's (`RELEASED`), which the network test loads through transformers."""

    code = LFM3.release_code
    source = respelled(TINY, tmp_path / 'release', lambda config: {**config, 'auto_map': dict(code.auto_map)})
    for file, text in {**RELEASED, 'unrelated.py': 'raise ImportError\n'}.items():
        (source / file).write_text(text)
    loaded = Pretrained.load(source, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['auto_map'] == dict(code.auto_map)
    assert {path.name for path in (tmp_path / 'saved').glob('*.py')} == set(RELEASED)
    for file, text in RELEASED.items():
        assert (tmp_path / 'saved' / file).read_text() == text


def test_a_component_keeps_the_code_beside_its_config_and_none_outside_its_source(tmp_path):
    """A release loaded from a component folder (`subfolder`) keeps the code beside that folder's config,
    not the source root's; a module whose relative import names a path outside the source refuses the
    save, which writes none of it (ledger 2038)."""
    code = LFM3.release_code
    component = respelled(TINY, tmp_path / 'release' / 'decoder',
                          lambda config: {**config, 'auto_map': dict(code.auto_map)})
    for file, text in RELEASED.items():
        (component / file).write_text(text)
    (tmp_path / 'release' / 'modeling_lfm3.py').write_text('raise ImportError\n')
    loaded = Pretrained.load(tmp_path / 'release', subfolder='decoder', dtype='float32',
                             attention_impl='reference')
    loaded.save(tmp_path / 'saved')
    assert {path.name: path.read_text() for path in (tmp_path / 'saved').glob('*.py')} == RELEASED
    (component / 'lfm3_conv.py').write_text('from .../victim import conv\n')
    (tmp_path / 'victim.py').write_text('conv = None\n')
    with pytest.raises(ValueError):
        loaded.save(tmp_path / 'escaped' / 'saved')
    assert not (tmp_path / 'escaped' / 'victim.py').exists()


def test_a_native_export_writes_its_familys_release_code_beside_itself(tmp_path, release_cache):
    """A model built in Dew of a family whose reference is a release's own code (`release_code`) exports
    with that release's auto_map and the modules it names and imports, at the commit Dew is held to,
    read through the hub's cache, so transformers opens the export with that code (ledger 2038)."""
    from dew.interop import PretrainedDecoder

    code = LFM3.release_code
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['auto_map'] == dict(code.auto_map)
    assert {path.name for path in (tmp_path / 'saved').glob('*.py')} == set(RELEASED)


@pytest.mark.network
def test_an_export_writes_the_releases_own_modules_at_its_commit(tmp_path):
    """The network's half (ledger 2038): a model built in Dew exports with its release's own modules at the
    commit, byte for byte: the two its auto_map names and lfm3_conv.py, which modeling_lfm3.py imports and
    no entry names. (The release's code opens under transformers 5.18.0 on, which the fixture is recorded
    under: tests/fixtures/hf/lfm3-tiny/source.json.)"""
    from huggingface_hub import hf_hub_download

    from dew.interop import PretrainedDecoder

    code = LFM3.release_code
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / 'saved')
    written = {path.name: path.read_bytes() for path in (tmp_path / 'saved').glob('*.py')}
    released = {file: Path(hf_hub_download(str(code.source), file, revision=code.revision))
                for file in ('configuration_lfm3.py', 'modeling_lfm3.py', 'lfm3_conv.py')}
    assert written == {file: path.read_bytes() for file, path in released.items()}


@pytest.mark.release_env("transformers==5.18.0")
def test_an_lfm3_vl_processor_lays_out_its_releases_own_processor_output(tmp_path):
    """LFM3-VL's processor is its release's own code (Lfm3VlProcessor and its image and video processors
    at f83d2b0), which a load that trusts the source's code runs (`trust_remote_code`, `Processor`): over a
    prompt with an image, and one with a video sampled at its own rate, Dew's processor gives the release
    processor's tokens, and each image's or video frame's patch vectors and grid where its placeholders
    read them, the placeholders in text order reading their media's merge blocks in order. So does the
    processor a run reloads from its recorded source (`load_processor`) in the process that trusts the
    source, the processor of the bundle's save, which writes the processor's own modules beside it, and
    the processor of a task the public pipeline loads where its caller trusts the source
    (`dew.pipeline(trust=...)`); one loaded without that trust refuses to run the source's code."""
    import shutil

    from tool_support import run_from
    from transformers import AutoProcessor

    import dew
    from dew.inference.tasks import recorded_processor
    from dew.interop.pretrained import load_processor

    source = tmp_path / 'source'
    shutil.copytree(VL, source)
    with run_from('remote_code_reference', 'tools') as reference:
        for file in reference.RELEASES['lfm3-vl-tiny'][2]:
            shutil.copy(reference.release_file('lfm3-vl-tiny', file), source / file)
    loaded = Pretrained.load(source, dtype='float32', attention_impl='reference', trust_remote_code=True)
    release = AutoProcessor.from_pretrained(str(source), trust_remote_code=True, local_files_only=True)
    loaded.save(tmp_path / 'saved')
    saved = AutoProcessor.from_pretrained(str(tmp_path / 'saved'), trust_remote_code=True,
                                          local_files_only=True)
    reloaded = load_processor(recorded_processor(loaded.processor), loaded.model)
    # Copies under names of their own, which no load above trusted.
    public, untrusted = tmp_path / 'public', tmp_path / 'untrusted'
    for directory in (public, untrusted):
        shutil.copytree(source, directory)
    task = dew.pipeline(str(public), dtype='float32', trust=(str(public),))
    config = json.loads((VL / 'config.json').read_text())
    rng = np.random.default_rng(0x2067)
    image = rng.integers(0, 256, (24, 40, 3), dtype=np.uint8)
    video = rng.integers(0, 256, (4, 24, 24, 3), dtype=np.uint8)
    metadata = [{'fps': 2.0, 'total_num_frames': 4, 'frames_indices': [0, 1, 2, 3]}]
    for text, media, token in (('w5 w6 <|image|> w7', {'images': [image]}, config['image_token_id']),
                               ('w5 <|video|> w9', {'videos': [video], 'video_metadata': metadata},
                                config['video_token_id'])):
        theirs = release(text=[text], return_tensors='np', **media)
        written = saved(text=[text], return_tensors='np', **media)
        assert sorted(written) == sorted(theirs)
        for name, value in theirs.items():
            np.testing.assert_array_equal(written[name], value, err_msg=name)
        ours = loaded.processor(text, **media)
        np.testing.assert_array_equal(np.asarray(ours.tokens), theirs['input_ids'])
        again = reloaded(text, **media)
        np.testing.assert_array_equal(np.asarray(again.tokens), theirs['input_ids'])
        np.testing.assert_array_equal(np.asarray(task.processor(text, **media).tokens), theirs['input_ids'])
        np.testing.assert_array_equal(np.asarray(again.conditioning['pixel_values']),
                                      np.asarray(ours.conditioning['pixel_values']))
        kind = 'image' if 'images' in media else 'video'
        pixels = theirs['pixel_values' if kind == 'image' else 'pixel_values_videos']
        frames, height, width = (int(side) for side in theirs[f'{kind}_grid_thw'][0])
        # An image is one frame of its own; a video's frames each one, as its placeholders' runs are.
        chunks = 1 if kind == 'image' else frames
        patches = frames * height * width // chunks
        placed = np.asarray(ours.conditioning['pixel_values'])[0]
        grids = np.asarray(ours.conditioning['image_grid_thw'])[0]
        for chunk in range(chunks):
            np.testing.assert_array_equal(placed[chunk, :patches],
                                          pixels[chunk * patches:(chunk + 1) * patches])
            assert tuple(grids[chunk]) == (frames // chunks, height, width)
        indices = np.asarray(ours.token_fields['image_indices'])[0]
        held = theirs['input_ids'][0] == token
        assert held.sum() == frames * height * width // 4 and np.all(indices[~held] == -1)
        assert np.all(np.diff(indices[held]) > 0)
    with pytest.raises(ValueError):
        dew.pipeline(str(untrusted), dtype='float32').processor('w5 <|image|>', images=[image])


def test_a_sources_record_runs_its_code_only_under_its_callers_trust(tmp_path):
    """A source's record (`SourceReference`), which a run keeps for its processor, keeps whether its load
    ran the source's own code (`remote_code`), and a record written before the field reads as not
    trusted. A processor reloaded from a record that ran the code (LFM3-VL's, its release's code beside
    it) refuses unless the process trusts the source (`load_processor`, `dew.registry.trusts_source`): a
    record arrives from the Hub, and its flag alone runs nothing; trusted, it reloads. A directory named
    as a package the process trusts (Flax's) is no source it trusts."""
    import shutil

    from tool_support import run_from

    from dew.interop.pretrained import load_processor
    from dew.interop.sources import SourceReference
    from dew.registry import from_record, to_record, trust_sources, trusts_source

    source = tmp_path / 'source'
    shutil.copytree(VL, source)
    with run_from('remote_code_reference', 'tools') as reference:
        for file in reference.RELEASES['lfm3-vl-tiny'][2]:
            shutil.copy(reference.release_file('lfm3-vl-tiny', file), source / file)
    model = Pretrained.load(source, dtype='float32', attention_impl='reference',
                            trust_remote_code=False).model
    recorded = SourceReference(str(source), remote_code=True)
    with pytest.raises(ValueError):
        load_processor(recorded, model)
    trust_sources((str(source),))
    assert load_processor(recorded, model) is not None
    (tmp_path / 'flax').mkdir()
    assert not trusts_source(str(tmp_path / 'flax'))

    trusted = SourceReference('LiquidAI/LFM3-VL-2B', 'f83d2b0a2296aaa277f394990a9b8b949ddf9691',
                              remote_code=True)
    assert from_record(SourceReference, to_record(trusted, SourceReference)) == trusted
    record = to_record(trusted, SourceReference)
    earlier = {key: value for key, value in record.items() if key != 'remote_code'}
    assert from_record(SourceReference, earlier) == SourceReference(trusted.name, trusted.revision)
