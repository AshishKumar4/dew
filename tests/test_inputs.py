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
    assert pixel_field(28, 28) == Field(PIXEL_VALUES_KEY, (3, 28, 28))
    assert pixel_field(16, 16, 1).shape == (1, 16, 16)


def test_soft_tokens_land_at_the_image_marks():
    """Two rows of six ids with two image marks each: the merged rows equal
    the projector rows at the marks and the token embeddings elsewhere."""
    rng = np.random.RandomState(0)
    ids = np.array([[5, 202, 7, 202, 9, 11], [202, 3, 202, 4, 5, 6]], np.int32)
    token_embeds = rng.rand(2, 6, 8).astype(np.float32)
    soft_tokens = rng.rand(2, 2, 8).astype(np.float32)
    merged = np.asarray(merge_soft_tokens(
        token_embeds, soft_tokens, ids == 202))
    assert merged.shape == token_embeds.shape
    for row in range(2):
        assert (merged[row][ids[row] == 202] == soft_tokens[row]).all()
        assert (merged[row][ids[row] != 202] == token_embeds[row][ids[row] != 202]).all()



def test_row_plan_preserves_resident_rows_padding_and_random_keys():
    import jax
    import jax.numpy as jnp
    from dew.nn.inputs import RowPlan, local_rows
    from dew.training import MeshSpec
    from dew.training.distributed import build_mesh

    pixels = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3)
    host_pixels = np.asarray(pixels)
    plan = RowPlan(build_mesh(MeshSpec()), 3, 8, 0, 1)
    key = jax.random.key(11)
    with jax.transfer_guard_device_to_host("disallow"):
        local = local_rows(pixels, host=False)
        placed = plan.place(plan.pad({"pixels": local, "labels": np.arange(3)}))
        keys = plan.keys(key)
        jax.block_until_ready((placed, keys))
    assert isinstance(local_rows(pixels), np.ndarray)
    np.testing.assert_array_equal(local_rows(pixels), host_pixels)
    np.testing.assert_array_equal(placed["pixels"], host_pixels[np.arange(8) % 3])
    np.testing.assert_array_equal(placed["labels"], np.arange(8) % 3)
    np.testing.assert_array_equal(plan.host(placed["pixels"]), host_pixels)
    expected_keys = jax.vmap(lambda row: jax.random.fold_in(key, row))(jnp.arange(8))
    np.testing.assert_array_equal(jax.random.key_data(keys), jax.random.key_data(expected_keys))

