"""Inkling 2's vision half against transformers' own (huggingface/transformers at 3a96649,
models/inkling2/modeling_inkling2.py's trunk and merger under modeling_inkling2_loop.py's wrapper), which the
suite's transformers predates: tools/inkling2_fixture.py writes tests/fixtures/hf/inkling2-vision-tiny, the
tiny decoder beside a tiny trunk and merger in the release's names, and records each seed's merged image
features and the wrapper's logits over the ids around two images there (recorded_vision.npz)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
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
    """Check 1's inputs, as the recorder drew them (`vision_draw`): each image NCHW in [-1, 1], and the ids
    around them, each image's placeholders one a merged token."""
    rng = np.random.default_rng(seed)
    side = PATCH * MERGE
    images = [rng.uniform(-1, 1, (3, rows * side, columns * side)).astype(np.float32)
              for rows, columns in IMAGES]
    ids = []
    for rows, columns in IMAGES:
        ids += list(rng.integers(0, IMAGE_TOKEN, PROMPT)) + [IMAGE_TOKEN] * (rows * columns)
    ids += list(rng.integers(0, IMAGE_TOKEN, PROMPT))
    return {'input_ids': np.asarray([ids], np.int32),
            **{f'image_{index}': image for index, image in enumerate(images)}}


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
