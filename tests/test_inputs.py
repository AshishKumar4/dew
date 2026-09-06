"""Pixel batches beside token batches for multimodal decoders.

A multimodal batch carries token ids with image_token_id marks under "text"
and processed images under "pixel_values". The tower and projector turn the
pixels into soft tokens and merge_soft_tokens places them at the marks, which
is what the decoder's input-embeddings hook receives once it lands.
"""

import numpy as np

from dew.inputs import Field, pixel_field
from dew.nn.vision import PIXEL_VALUES_KEY, merge_soft_tokens


def test_pixel_field_names_the_batch_key_and_shape():
    assert pixel_field(28, 28) == Field("pixel_values", (3, 28, 28))
    assert pixel_field(16, 16, 1).shape == (1, 16, 16)


def test_a_multimodal_batch_carries_pixels_beside_tokens():
    """Two rows of six ids with two image marks each, and two images: the
    merged rows equal the projector rows at the marks and the token
    embeddings elsewhere."""
    rng = np.random.RandomState(0)
    ids = np.array([[5, 202, 7, 202, 9, 11], [202, 3, 202, 4, 5, 6]], np.int32)
    pixels = rng.rand(2, 3, 28, 28).astype(np.float32)
    assert pixels.shape[1:] == pixel_field(28, 28).shape
    assert PIXEL_VALUES_KEY == "pixel_values"
    token_embeds = rng.rand(2, 6, 8).astype(np.float32)
    soft_tokens = rng.rand(2, 2, 8).astype(np.float32)
    merged = np.asarray(merge_soft_tokens(
        token_embeds, soft_tokens, ids == 202))
    assert merged.shape == token_embeds.shape
    for row in range(2):
        assert (merged[row][ids[row] == 202] == soft_tokens[row]).all()
        assert (merged[row][ids[row] != 202] == token_embeds[row][ids[row] != 202]).all()
