"""Inkling 2's vision half against transformers' own (huggingface/transformers at 3a96649,
models/inkling2/modeling_inkling2.py's trunk and merger under modeling_inkling2_loop.py's wrapper), which the
suite's transformers predates: tools/inkling2_fixture.py writes tests/fixtures/hf/inkling2-vision-tiny, the
tiny decoder beside a tiny trunk and merger in the release's names, and records each seed's merged image
features and the wrapper's logits over the ids around two images there (recorded_vision.npz)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending, float64_twin
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.interop import Pretrained

BUNDLE = Path(__file__).parent / 'fixtures' / 'hf' / 'inkling2-vision-tiny'
PATCH, MERGE = 4, 2
"""The tiny trunk's patch side and merge size (its vision_config's `native_patch_size`,
`adapter_merge_size`)."""
IMAGES = ((2, 3), (3, 2))
"""Each drawn image's grid in merged tokens, as the recorder drew them (tools/inkling2_fixture.py)."""
PROMPT, IMAGE_TOKEN = 6, 120
"""The text tokens before each image and after the last, and the bundle's image placeholder id."""


def draw(seed):
    """Check 1's inputs, as the recorder drew them (`vision_draw`, in its order): each image NCHW in [-1, 1],
    and the ids around them, each image's placeholders one a merged token."""
    rng = np.random.default_rng(seed)
    side = PATCH * MERGE
    drawn = {f'image_{index}': rng.uniform(-1, 1, (3, rows * side, columns * side)).astype(np.float32)
             for index, (rows, columns) in enumerate(IMAGES)}
    spans = [np.concatenate([rng.integers(0, IMAGE_TOKEN, PROMPT), np.full(rows * columns, IMAGE_TOKEN)])
             for rows, columns in IMAGES]
    ids = np.concatenate([*spans, rng.integers(0, IMAGE_TOKEN, PROMPT)])
    return {'input_ids': ids[None].astype(np.int32), **drawn}


def media(inputs, dtype):
    """The images as the trunk takes several of other shapes: each one's patches `[patches, channels *
    PATCH * PATCH]` merge block by merge block, each patch's channels, rows and columns, beside its grid in
    patches."""
    patches = []
    for index, (rows, columns) in enumerate(IMAGES):
        image = jnp.asarray(inputs[f'image_{index}'], dtype).reshape(
            3, rows, MERGE, PATCH, columns, MERGE, PATCH)
        patches.append(image.transpose(1, 4, 2, 5, 0, 3, 6).reshape(rows * columns * MERGE * MERGE, -1))
    grids = [[1, rows * MERGE, columns * MERGE] for rows, columns in IMAGES]
    return {'pixel_values': jnp.stack(patches)[None], 'image_grid_thw': jnp.asarray([grids], jnp.int32)}


def placed(ids):
    """Each image placeholder's merged token, counted across the images, -1 at the text."""
    held = ids == IMAGE_TOKEN
    return jnp.where(held, jnp.cumsum(held, axis=-1) - 1, -1)


@attending("reference")
def test_its_image_features_and_logits_are_on_par_with_transformers():
    """The release-layout bundle's fp32 merged image features (its conditioner's: the trunk's patches
    through the merger, the class token left behind) and its logits over the ids around two rectangular
    images (2 by 3 and 3 by 2 merged tokens, so the rotary's columns and rows and the position table's
    resampling read apart), on par with transformers' fp32 run from its float64 twin (check 1) over the
    recording's seeds: Dew's own float64 run of the bundle is the truth, held at every seed to the
    recording's sketch."""
    loaded = Pretrained.load(BUNDLE, dtype='float32')
    path = BUNDLE / 'recorded_vision.npz'
    seeds = declared_seeds(path)['vision']

    def run(model, dtype):
        def outputs(variables, inputs):
            ids = jnp.asarray(inputs['input_ids'])
            given = media(inputs, dtype)
            features, logits = model.apply(
                variables, ids, placed(ids), given, method=lambda module, ids, at, given: (
                    module.conditioner(given), module(ids, image_indices=at, conditioning=given)))
            return {'image_features': features[0], 'logits': logits}
        return outputs

    wide = jax.jit(run(float64_twin(loaded.model), jnp.float64))
    recorded, reference, truth = indexed_recording(
        recording(path), 'vision', seeds, inputs_of=draw,
        truth_fn=lambda inputs: wide(widened(loaded.variables), inputs))
    ran = jax.jit(run(loaded.model, jnp.float32))
    assert_as_exact(lambda index, inputs: ran(loaded.variables, inputs), reference, recorded,
                    'Inkling 2 vision', seeds=seeds, truth_fn=truth)


class _NoText:
    """A host processor whose text operations no test here reads: the processor's fields arrive processed
    (`Processor.from_hf`)."""

    def __call__(self, **_):
        raise AssertionError("the fields arrive processed")

    def save_pretrained(self, save_directory: str) -> None:
        raise AssertionError("nothing is saved")

    def apply_chat_template(self, conversation, **_):
        raise AssertionError("no chat is templated")

    def batch_decode(self, sequences, *, skip_special_tokens: bool) -> list[str]:
        raise AssertionError("nothing is decoded")


def test_the_processor_reads_its_merged_blocks_as_the_tower_takes_them():
    """transformers' processor's fields for two images of other shapes (each merged token's pixels
    `[8, 8, 3]`, the grids counted in merged tokens: processing_inkling2.py) read by Dew's processor
    (`Processor.from_hf`, its `merged_blocks`) are, bit for bit, the patches, patch grids and placeholder
    slots the parity test hands the bundle (`media`, `placed`; row 12943's review)."""
    import json

    from dew.interop.processors import Processor

    loaded = Pretrained.load(BUNDLE, dtype='float32')
    config = json.loads((BUNDLE / 'config.json').read_text())
    processor = Processor(_NoText(), config, loaded.model_config, config['text_config']['vocab_size'])
    inputs = draw(0)
    side = PATCH * MERGE
    blocks = [inputs[f'image_{index}'].transpose(1, 2, 0).reshape(rows, side, columns, side, 3)
              .transpose(0, 2, 1, 3, 4).reshape(-1, side, side, 3)
              for index, (rows, columns) in enumerate(IMAGES)]
    ids = inputs['input_ids']
    processed = processor.from_hf({'input_ids': ids, 'attention_mask': np.ones_like(ids),
                                   'pixel_values': np.concatenate(blocks),
                                   'image_grid_thw': np.asarray([[1, *grid] for grid in IMAGES])})
    # The fields it reads are the patches, grids and placements the tower takes, bit for bit.
    packed = media(inputs, jnp.float32)
    for name in ('pixel_values', 'image_grid_thw'):
        np.testing.assert_array_equal(np.asarray(processed.conditioning[name]), np.asarray(packed[name]),
                                      err_msg=name)
    np.testing.assert_array_equal(np.asarray(processed.kwargs()['image_indices']),
                                  np.asarray(placed(jnp.asarray(ids))))
    # A grid not [images, 3] of positive integers, or none, is refused before it is scaled: `[[1]]` over one
    # merged block and one placeholder would broadcast to a grid the packed reader accepts.
    one = np.asarray([[IMAGE_TOKEN]], np.int32)
    single = {'input_ids': one, 'attention_mask': np.ones_like(one), 'pixel_values': blocks[0][:1]}
    for given in ({**single, 'image_grid_thw': np.asarray([[1]])}, single):
        with pytest.raises(ValueError):
            processor.from_hf(given)


def _bundle_with_tokenizer(directory):
    """The vision bundle beside a word-level tokenizer whose image placeholder is the bundle's
    (`IMAGE_TOKEN`), as the release's tokenizer holds `<|unused_200054|>`."""
    import shutil

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    for path in BUNDLE.glob('*'):
        shutil.copy(path, directory / path.name)
    vocabulary = {'[PAD]': 0, **{f'w{index}': index + 1 for index in range(10)}, '<|image|>': IMAGE_TOKEN}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[PAD]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[PAD]',
                                        additional_special_tokens=['<|image|>'])
    tokenizer.save_pretrained(directory)
    return directory


def test_raw_images_are_processed_as_transformers_inkling2_processor_processes_them(tmp_path):
    """Inkling 2's release names `Inkling2Processor`, which no transformers Dew supports ships: Dew's own
    (`dew.interop.inkling_media`, chosen by its model type whatever processor files the release ships)
    pads, standardizes and cuts raw images (an RGB one of no whole merged tokens, an RGBA one over its
    checkerboard, one of whole tokens) into the merged blocks and grids transformers' records, bit for bit;
    and a public raw-image call (`Pretrained.processor`) lays a prompt around an image out as the model
    reads the reference's fields (`Processor.from_hf`), each placeholder as many as the image's merged
    tokens (row 13251)."""
    from tool_support import run_from

    from dew.interop.inkling_media import InklingProcessor

    directory = _bundle_with_tokenizer(tmp_path)
    (directory / 'processor_config.json').write_text('{"processor_class": "Inkling2Processor"}')
    loaded = Pretrained.load(directory, dtype='float32')
    processor = loaded.processor._media_reference
    assert isinstance(processor, InklingProcessor)
    with np.load(BUNDLE / 'recorded_processor.npz') as recorded:
        expected = {name: recorded[name] for name in recorded.files if not name.endswith('provenance')}
    with run_from('inkling2_fixture', 'tools') as module:
        images = module.processor_draws()
    for name, image in images.items():
        fields = processor(text='w1 <|image|>', images=[image])
        for field in ('pixel_values', 'image_grid_thw'):
            np.testing.assert_array_equal(fields[field], expected[f'{name}.{field}'],
                                          err_msg=f'{name} {field}')
    image = images['rgb']
    rows, columns = expected['rgb.image_grid_thw'][0, 1:]
    ids = np.asarray([[2, *[IMAGE_TOKEN] * int(rows * columns), 3]], np.int64)
    read = loaded.processor('w1 <|image|> w2', images=[image])
    reference = loaded.processor.from_hf({'input_ids': ids, 'attention_mask': np.ones_like(ids),
                                          'pixel_values': expected['rgb.pixel_values'],
                                          'image_grid_thw': expected['rgb.image_grid_thw']})
    np.testing.assert_array_equal(np.asarray(read.tokens), ids)
    for name, value in reference.conditioning.items():
        np.testing.assert_array_equal(np.asarray(read.conditioning[name]), np.asarray(value), err_msg=name)
    # The same image as a file's path and as its encoded bytes, loaded as transformers loads them.
    from PIL import Image

    stored = tmp_path / 'image.png'
    Image.fromarray(image).save(stored)
    for given in (stored, [stored], stored.read_bytes()):
        held = loaded.processor('w1 <|image|> w2', images=given)
        for name, value in reference.conditioning.items():
            np.testing.assert_array_equal(np.asarray(held.conditioning[name]), np.asarray(value),
                                          err_msg=f'{type(given).__name__} {name}')


def _media_reader(tmp_path):
    """The public processor of a bundle with a tokenizer and Inkling 2's processor, and its read of one
    image (each conditioning field as an array), with a transparent RGBA image `[16, 24]` it reads."""
    directory = _bundle_with_tokenizer(tmp_path)
    (directory / 'processor_config.json').write_text('{"processor_class": "Inkling2Processor"}')
    loaded = Pretrained.load(directory, dtype='float32')
    rgba = np.zeros((16, 24, 4), np.uint8)
    rgba[4:12, 8:16] = (200, 30, 90, 255)

    def read(given):
        return {name: np.asarray(value) for name, value in
                loaded.processor('w1 <|image|> w2', images=[given]).conditioning.items()}
    return loaded, rgba, read


def test_an_image_a_url_names_reads_as_its_file_does(tmp_path):
    """A transparent PNG served over HTTP reads, through the public processor, as its file's path does
    (Inkling 2's alpha rule on the image as served, `bounded_read` its fetch, row 13348); a URL its server
    answers 404 and a redirect off HTTP (to FTP, which a read could not hold to its deadline) are each
    refused by name, and so is a body past the cap a read holds it to (`opened`'s `url_bytes`)."""
    import http.server
    import threading

    from PIL import Image

    from dew.interop.inkling_media import opened

    _, rgba, read = _media_reader(tmp_path)
    served = tmp_path / 'served'
    served.mkdir()
    Image.fromarray(rgba).save(served / 'transparent.png')

    class Served(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(served), **kwargs)

        def do_GET(self):
            if self.path == '/elsewhere.png':
                self.send_response(302)
                self.send_header('Location', 'ftp://127.0.0.1/transparent.png')
                self.end_headers()
                return
            super().do_GET()

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Served)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}'
        fetched, stored = read(f'{url}/transparent.png'), read(served / 'transparent.png')
        for field, value in fetched.items():
            np.testing.assert_array_equal(value, stored[field], err_msg=field)
        for refused, named in (('missing.png', 'HTTP 404'), ('elsewhere.png', 'HTTP 302')):
            with pytest.raises(ValueError) as refusal:
                read(f'{url}/{refused}')
            assert named in str(refusal.value), refusal.value
        size = (served / 'transparent.png').stat().st_size
        assert opened(f'{url}/transparent.png', url_bytes=size).size == (24, 16)
        with pytest.raises(ValueError) as refusal:
            opened(f'{url}/transparent.png', url_bytes=size - 1)
        assert f'past {size - 1} bytes' in str(refusal.value), refusal.value
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_a_transparent_or_turned_image_reads_alike_however_it_is_given(tmp_path):
    """An RGBA PNG's transparent pixels take the checkerboard's shades (Inkling 2's alpha rule), and a JPEG
    turned by its EXIF orientation reads upright, whether the image is given through the public processor
    as a PIL image, its encoded bytes, its file's path, a data URI (one past a path's length too) or
    line-wrapped base64: transformers' `load_image` converts a
    path's image to RGB first, its transparent pixels black (row 13341). A tokenizer padding a side that is
    neither is refused (row 13342)."""
    import base64

    from PIL import Image

    from dew.interop.inkling_media import InklingProcessor

    loaded, rgba, read = _media_reader(tmp_path)
    transparent = tmp_path / 'transparent.png'
    Image.fromarray(rgba).save(transparent)
    turned = tmp_path / 'turned.jpg'
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: the stored image turned a quarter, read upright.
    drawn = np.random.default_rng(13341).integers(0, 256, (16, 40, 3), np.uint8)
    Image.fromarray(drawn).save(turned, exif=exif)

    for stored, kind in ((transparent, 'png'), (turned, 'jpeg')):
        encoded = stored.read_bytes()
        with Image.open(stored) as image:
            image.load()
            forms = {'PIL': read(image), 'bytes': read(encoded), 'path': read(stored),
                     'data URI': read(f'data:image/{kind};base64,{base64.b64encode(encoded).decode()}'),
                     'wrapped base64': read(base64.encodebytes(encoded).decode())}
        for name in ('bytes', 'path', 'data URI', 'wrapped base64'):
            for field, value in forms[name].items():
                np.testing.assert_array_equal(value, forms['PIL'][field],
                                              err_msg=f'{stored.name} {name} {field}')
    # A data URI longer than a path may be (an image of 64x64 noise, its PNG past 4 KiB).
    large = tmp_path / 'large.png'
    Image.fromarray(np.random.default_rng(13342).integers(0, 256, (64, 64, 4), np.uint8)).save(large)
    inline = f'data:image/png;base64,{base64.b64encode(large.read_bytes()).decode()}'
    assert len(inline) > 4096
    path_read = read(large)
    for field, value in read(inline).items():
        np.testing.assert_array_equal(value, path_read[field], err_msg=f'large data URI {field}')
    blacked = read(rgba[..., :3])
    assert any(not np.array_equal(blacked[field], value) for field, value in read(transparent).items())
    upright = read(np.asarray(Image.open(turned).transpose(Image.Transpose.ROTATE_270)))
    for field, value in read(turned).items():
        np.testing.assert_array_equal(value, upright[field], err_msg=field)
    media = loaded.processor._media_reference
    assert isinstance(media, InklingProcessor)
    media.tokenizer.padding_side = 'middle'
    with pytest.raises(ValueError):
        loaded.processor('w1 <|image|>', images=[rgba])


def test_a_saved_bundle_writes_back_each_spelling_its_source_stores(tmp_path):
    """A save of the bundle, its language model written back through Inkling 2's `unprepare` (row 13561),
    writes every tensor the source stores under each name it stores it, an alias of its head (`model.`
    taken off, one tensor to the wrapper's reading) among them, and loads back as the variables
    it was saved from, bit for bit."""
    import shutil

    from flax.traverse_util import flatten_dict
    from safetensors.numpy import load_file, save_file

    directory = tmp_path / 'source'
    shutil.copytree(BUNDLE, directory)
    stored = load_file(directory / 'model.safetensors')
    head = 'model.llm.unembed.weight'
    alias = head.removeprefix('model.')
    # The position table spelled twice beside one class position, which its read joins into it.
    table = 'model.visual.backbone.embeddings.position_embedding'
    aliases = {alias: stored[head], table.removeprefix('model.'): stored[table]}
    save_file({**stored, **aliases}, directory / 'model.safetensors')
    loaded = Pretrained.load(directory, dtype='float32')
    loaded.save(tmp_path / 'saved')
    written = {name: value for file in (tmp_path / 'saved').glob('*.safetensors') for name, value in
               load_file(file).items()}
    source = {**stored, **aliases}
    assert sorted(written) == sorted(source)
    for name, tensor in source.items():
        assert (written[name].dtype, written[name].tobytes()) == (tensor.dtype, tensor.tobytes()), name
    reloaded = flatten_dict(Pretrained.load(tmp_path / 'saved', dtype='float32').variables)
    held = flatten_dict(loaded.variables)
    assert sorted(reloaded) == sorted(held)
    for path, value in held.items():
        np.testing.assert_array_equal(np.asarray(reloaded[path]), np.asarray(value), err_msg=str(path))


def test_a_bundle_whose_tower_computes_otherwise_is_refused_its_save():
    """A save writes the bundle's vision_config as published, so a loaded bundle whose tower runs at another
    norm epsilon than its config states is refused before a tensor is written (row 13561; review)."""
    import dataclasses

    from dew.interop.decoder_parts import ExportRefused

    loaded = Pretrained.load(BUNDLE, dtype='float32')
    vision = dataclasses.replace(loaded.model.vision, layer_norm_eps=2 * loaded.model.vision.layer_norm_eps)
    with pytest.raises(ExportRefused):
        dataclasses.replace(loaded, model=loaded.model.clone(vision=vision)).export()


def test_a_bundle_whose_published_config_was_edited_since_its_load_is_refused_its_save(tmp_path):
    """A save validates the config it publishes: a loaded bundle whose vision config's norm epsilon was
    edited after its load, which the model it would write does not compute, is refused before a tensor is
    written, as is one whose text config's residual normalization was (train 1366's review F2)."""
    import dataclasses

    from dew.interop.decoder_parts import ExportRefused

    loaded = Pretrained.load(BUNDLE, dtype='float32')
    for section, field in (('vision_config', 'layer_norm_eps'), ('text_config', 'attn_res_normalize_values')):
        value = loaded.config[section][field]
        changed = (not value) if isinstance(value, bool) else 2 * value
        edited = dataclasses.replace(loaded, config={**loaded.config,
                                                     section: {**loaded.config[section], field: changed}})
        with pytest.raises(ExportRefused):
            edited.save(tmp_path / field)
        assert not any((tmp_path / field).glob('*.safetensors')), field


def test_a_bundles_language_model_saved_alone_refuses_an_edited_towers_section(tmp_path):
    """A bundle loaded for its language model alone (`text_only`) retains its towers' tensors unread and
    writes back the config that names them, so a save whose config's vision section was edited since its
    load is refused before a tensor is written: no model computes the tower to check it (train 1370's
    review F1); one as loaded saves."""
    import dataclasses

    from dew.interop.decoder_parts import ExportRefused

    loaded = Pretrained.load(BUNDLE, dtype='float32', text_only=True)
    loaded.save(tmp_path / 'as-loaded')
    vision = loaded.config['vision_config']
    edited = dataclasses.replace(loaded, config={
        **loaded.config, 'vision_config': {**vision, 'layer_norm_eps': 2 * vision['layer_norm_eps']}})
    with pytest.raises(ExportRefused):
        edited.save(tmp_path / 'edited')
    # Edited in place, the section the load's snapshot holds stays as loaded.
    vision['layer_norm_eps'] = 2 * vision['layer_norm_eps']
    with pytest.raises(ExportRefused):
        loaded.save(tmp_path / 'edited-in-place')
