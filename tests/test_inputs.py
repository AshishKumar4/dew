"""Batch fields and their placement on the mesh.

A multimodal batch carries token ids under "text" and processed images under
"pixel_values"; `RowPlan` and `shard_batch` pad and place such rows without
copying resident inputs through the host.
"""

import numpy as np
import pytest
from steady_state import guarded


def test_row_plan_preserves_resident_rows_padding_and_random_keys():
    import jax
    import jax.numpy as jnp

    from dew.nn.inputs import RowPlan, local_rows
    from dew.training import MeshSpec

    pixels = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3)
    host_pixels = np.asarray(pixels)
    plan = RowPlan(MeshSpec().build(), 3, 8, 0, 1)
    key = jax.random.key(11)
    with guarded(allow=("host_to_device",)):
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

    from dew.nn.inputs import RowPlan, request_axes
    from dew.training import MeshSpec

    mesh = MeshSpec().build()
    plan = RowPlan(mesh, 3, 8, 0, 1)
    key = jax.device_put(jax.random.key(11), NamedSharding(mesh, P()))
    with guarded(allow=("host_to_device",)):
        keys = plan.keys(key)
        jax.block_until_ready(keys)
    expected = jax.vmap(lambda row: jax.random.fold_in(jax.random.key(11), row))(jnp.arange(8))
    np.testing.assert_array_equal(jax.random.key_data(keys), jax.random.key_data(expected))
    assert jax.random.key_impl(keys) == jax.random.key_impl(key)
    assert keys.sharding == NamedSharding(mesh, P(request_axes(mesh)))


def test_shard_batch_preserves_resident_nested_inputs_without_host_transfer():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.nn.inputs import ModelInputs
    from dew.nn.sharding import batch_axes
    from dew.training.distributed import MeshSpec, shard_batch

    mesh = MeshSpec().build()
    rows = 2 * jax.device_count()
    tokens = jnp.arange(rows * 4, dtype=jnp.int32).reshape(rows, 4)
    pixels = jnp.arange(rows * 12, dtype=jnp.float32).reshape(rows, 2, 2, 3) / 16
    labels = np.arange(rows, dtype=np.int32)
    batch = {"text": ModelInputs(tokens, conditioning={"pixel_values": pixels}),
             "labels": labels, "weight": np.float32(0.5)}
    jax.block_until_ready(batch)
    with guarded(allow=("host_to_device",)):
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
    row_axes = batch_axes(mesh)
    assert placed["text"].tokens.sharding == NamedSharding(mesh, P(row_axes, "sequence"))
    assert placed["text"].conditioning["pixel_values"].sharding == NamedSharding(mesh, P(row_axes))
    assert placed["labels"].sharding == NamedSharding(mesh, P(row_axes))
    assert placed["weight"].sharding == NamedSharding(mesh, P())


@pytest.mark.parametrize("parser", ["preference", "prompt", "request", "rollout", "completion"])
def test_every_token_id_parser_reads_ids_by_one_law(parser):
    """A dataset's and a request's token ids are read alike (`token_id`): a Python or NumPy integer
    from 0 through int32's largest passes as an int, and a bool, a negative id or one past int32 is
    refused, where the parsers once disagreed on all three."""
    from dew.data.preferences import PreferencePair
    from dew.data.prompts import _prompt_ids
    from dew.data.text import ByteTokenizer
    from dew.inference.clients import _token_rows
    from dew.inference.rollouts import _prompt
    from dew.inference.tasks import prepared_inputs

    def read(ids: list) -> list[int]:
        if parser == "preference":
            pair = PreferencePair(seq_len=4).map({"chosen": ids, "rejected": ids})
            return pair["input_ids"][0, : len(ids)].tolist()
        if parser == "prompt":
            return list(_prompt_ids(ByteTokenizer(), ids, None, "prompt", None))
        if parser == "request":
            return np.asarray(prepared_inputs(None, [ids], images=None).tokens)[0].tolist()
        if parser == "rollout":
            return list(_prompt(ids))
        rows = _token_rows([ids])
        assert rows is not None
        return rows[0]

    assert read([np.int64(7), 3]) == [7, 3]
    for refused in (True, -1, 2**31):
        with pytest.raises(ValueError):
            read([5, refused])
