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


def test_row_plan_folds_a_mesh_replicated_request_key_on_device():
    """A request key placed replicated over the mesh derives its rows without
    a host round trip: a pool's key arrives this way, and folding the placed
    key itself is what a multi-process request cannot do."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.nn.inputs import BATCH_AXES, RowPlan
    from dew.training import MeshSpec
    from dew.training.distributed import build_mesh

    mesh = build_mesh(MeshSpec())
    plan = RowPlan(mesh, 3, 8, 0, 1)
    key = jax.device_put(jax.random.key(11), NamedSharding(mesh, P()))
    with jax.transfer_guard_device_to_host("disallow"):
        keys = plan.keys(key)
        jax.block_until_ready(keys)
    expected = jax.vmap(lambda row: jax.random.fold_in(jax.random.key(11), row))(jnp.arange(8))
    np.testing.assert_array_equal(jax.random.key_data(keys), jax.random.key_data(expected))
    assert jax.random.key_impl(keys) == jax.random.key_impl(key)
    assert keys.sharding == NamedSharding(mesh, P(BATCH_AXES))


def test_shard_batch_preserves_resident_nested_inputs_without_host_transfer():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.nn.inputs import ModelInputs
    from dew.training.distributed import build_mesh, shard_batch

    mesh = build_mesh()
    rows = 2 * jax.device_count()
    tokens = jnp.arange(rows * 4, dtype=jnp.int32).reshape(rows, 4)
    pixels = jnp.arange(rows * 12, dtype=jnp.float32).reshape(rows, 2, 2, 3) / 16
    labels = np.arange(rows, dtype=np.int32)
    batch = {"text": ModelInputs(tokens, conditioning={"pixel_values": pixels}),
             "labels": labels, "weight": np.float32(0.5)}
    jax.block_until_ready(batch)
    with jax.transfer_guard_device_to_host("disallow"):
        placed = shard_batch(mesh, batch)
        jax.block_until_ready(placed)
    np.testing.assert_array_equal(placed["text"].tokens, np.arange(rows * 4).reshape(rows, 4))
    np.testing.assert_array_equal(placed["text"].conditioning["pixel_values"],
                                  np.arange(rows * 12, dtype=np.float32).reshape(rows, 2, 2, 3) / 16)
    np.testing.assert_array_equal(placed["labels"], labels)
    np.testing.assert_array_equal(placed["weight"], np.float32(0.5))
    assert placed["text"].tokens.dtype == jnp.int32
    assert placed["text"].conditioning["pixel_values"].dtype == jnp.float32
    assert "attention_mask" not in placed["text"].token_fields
    row_axes = ("data", "expert", "fsdp")
    assert placed["text"].tokens.sharding == NamedSharding(mesh, P(row_axes, "sequence"))
    assert placed["text"].conditioning["pixel_values"].sharding == NamedSharding(mesh, P(row_axes))
    assert placed["labels"].sharding == NamedSharding(mesh, P(row_axes))
    assert placed["weight"].sharding == NamedSharding(mesh, P())


def test_char_table_compute_override_preserves_supplied_storage():
    import jax.numpy as jnp

    from dew.inputs import CharTable
    from dew.inputs.encoders import rebuild

    source = CharTable.from_pretrained(dtype="bfloat16", param_dtype="float32", tokens=4, features=3)
    tokens = source.tokenize(["ab"])
    table = source.params["table"]
    assert table.dtype == jnp.float32
    expected = table[tokens["input_ids"]].astype(jnp.bfloat16)
    np.testing.assert_array_equal(source.encode(source.params, tokens).hidden, expected)
    saved = {"table": (table + 0.25).astype(jnp.bfloat16)}
    rebuilt = rebuild("char_table", {**source.to_json(), "dtype": "float32"}, params=saved)
    actual = rebuilt.encode(rebuilt.params, tokens).hidden
    assert actual.dtype == jnp.float32
    assert rebuilt.params["table"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(rebuilt.params["table"], saved["table"])
    np.testing.assert_array_equal(actual, saved["table"][tokens["input_ids"]].astype(jnp.float32))

