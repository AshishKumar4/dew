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

    from dew.distributed.batches import shard_batch
    from dew.distributed.placement import MeshSpec
    from dew.mesh import batch_axes
    from dew.nn.inputs import ModelInputs

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


@pytest.mark.parametrize("parser", ["preference", "prompt", "request", "array", "rollout", "completion"])
def test_every_token_id_parser_reads_ids_by_one_law(parser):
    """A dataset's and a request's token ids are read alike (`token_id`): a Python or NumPy integer
    from 0 through int32's largest passes as an int, and a bool, a negative id or one past int32 is
    refused, where the parsers once disagreed on all three. A request's rows as a NumPy array are
    read by the same law as the same rows in a list; a NumPy row holds no bool beside integers."""
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
        if parser == "array":
            return np.asarray(prepared_inputs(None, np.asarray([ids]), images=None).tokens)[0].tolist()
        if parser == "rollout":
            return list(_prompt(ids))
        rows = _token_rows([ids])
        assert rows is not None
        return rows[0]

    assert read([np.int64(7), 3]) == [7, 3]
    for refused in (-1, 2**31) if parser == "array" else (True, -1, 2**31):
        with pytest.raises(ValueError):
            read([5, refused])


def test_a_request_refuses_a_bool_row_beside_integer_rows():
    """A request's rows are each read by the one law before they join into an array, so a NumPy row
    of bools beside a list of integers is refused, not promoted into ids 1 and 0."""
    from collections import UserList

    from dew.inference.tasks import prepared_inputs

    for row in (np.array([True, False]), [True, 3], UserList([True, 3])):
        with pytest.raises(ValueError):
            prepared_inputs(None, [row, [2, 3]], images=None)
    assert np.asarray(prepared_inputs(None, [np.array([4, 5]), [2, 3]], images=None).tokens).tolist() == [
        [4, 5], [2, 3]]


def test_a_packed_request_reads_alike_as_coordinates_and_as_its_fields():
    """The same packed, left-padded rows, as `Coordinates` (a data pipeline's
    packing) or as the validity, segment and position fields (a rollout's), are
    one request: each validates, and after taking rows, aligning left, slicing
    and extending into a response the decoder computes the same logits from
    either form."""
    import jax
    import jax.numpy as jnp

    from dew.batch import COORDINATES_FIELD, coordinates
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.inputs import ModelInputs

    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=1, num_heads=2, num_kv_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=16, dtype=jnp.float32,
                              attention_impl="reference")
    tokens = jnp.asarray([[0, 0, 3, 4, 5, 6, 7, 8], [0, 9, 10, 11, 12, 13, 14, 15]], jnp.int32)
    valid = tokens != 0
    segments = jnp.asarray([[0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 1, 2, 2, 2, 3, 3]], jnp.int32)
    positions = jnp.asarray([[0, 0, 0, 1, 2, 0, 1, 2], [0, 0, 1, 0, 1, 2, 0, 1]], jnp.int32)
    params = model.init(jax.random.key(0), tokens)
    forms = (ModelInputs(tokens, {"attention_mask": valid, "segment_ids": segments, "positions": positions}),
             ModelInputs(tokens, {COORDINATES_FIELD: coordinates(segments=segments, valid=valid,
                                                                   positions=positions)}))
    steps = (lambda inputs: inputs.take_rows(jnp.asarray([1, 0])),
             lambda inputs: inputs.align_left(jnp.asarray([2, 1])),
             lambda inputs: inputs.slice_tokens(0, 6),
             lambda inputs: inputs.extended(jnp.full((2, 3), 5, jnp.int32)))
    for step in (lambda inputs: inputs, *steps):
        fields, laid = (step(ModelInputs.from_value(form)) for form in forms)
        expected, actual = (model.apply(params, inputs.tokens, **inputs.kwargs())
                            for inputs in (fields, laid))
        real = np.asarray(jnp.asarray(fields.token_fields["attention_mask"]))
        np.testing.assert_array_equal(np.asarray(actual)[real], np.asarray(expected)[real])


@pytest.mark.parametrize("placed", [False, True])
def test_a_request_generates_alike_from_coordinates_and_from_its_fields(placed):
    """An unpacked, left-padded request with its own positions generates the same
    tokens whether its validity and positions come as `Coordinates` or as the
    fields, through `generate`'s request (`Request.prepare`), its prefill and every
    step; on a mesh its rows are padded to the devices, the repeats marked invalid
    in the form the request holds (`Request.padded`)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.batch import COORDINATES_FIELD, coordinates
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.inputs import ModelInputs
    from dew.sampling import Sampling, generate
    from dew.training import MeshSpec

    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=1, num_heads=2, num_kv_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=16, dtype=jnp.float32,
                              attention_impl="reference")
    tokens = jnp.asarray([[0, 0, 3, 4, 5], [6, 7, 8, 9, 10]], jnp.int32)
    valid = tokens != 0
    positions = jnp.asarray([[0, 0, 4, 5, 6], [0, 1, 2, 3, 4]], jnp.int32)
    params = model.init(jax.random.key(0), tokens)
    if placed:
        params = jax.device_put(params, NamedSharding(MeshSpec().build(), P()))
    forms = (ModelInputs(tokens, {"attention_mask": valid, "positions": positions}),
             ModelInputs(tokens, {COORDINATES_FIELD: coordinates(valid=valid, positions=positions)}))
    fields, laid = (generate(model, params, form, 4, key=jax.random.key(1), sampling=Sampling(temperature=0))
                    for form in forms)
    np.testing.assert_array_equal(np.asarray(laid.tokens), np.asarray(fields.tokens))
    np.testing.assert_array_equal(np.asarray(laid.lengths), np.asarray(fields.lengths))


def test_a_feed_forward_reads_blocks_of_its_own_size_from_the_layers_metadata():
    """The model numbers its block router's blocks once, in its metadata (`AttentionMetadata.blocks`);
    a feed-forward routing blocks of another size numbers its own from the call's coordinates rather
    than read blocks of the wrong width."""
    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata, numbered_blocks, routing_blocks

    class Routed:
        def __init__(self, block):
            self.routing_block = block

    tokens = coordinates((1, 8))
    metadata = AttentionMetadata(coordinates=tokens, blocks=numbered_blocks(tokens, (1, 8), 4),
                                 routing_block=4)
    four = routing_blocks(Routed(4), metadata, (1, 8, 3))["blocks"]
    two = routing_blocks(Routed(2), metadata, (1, 8, 3))["blocks"]
    np.testing.assert_array_equal(four, [[0, 0, 0, 0, 1, 1, 1, 1]])
    np.testing.assert_array_equal(two, [[0, 0, 1, 1, 2, 2, 3, 3]])
    assert routing_blocks(Routed(None), metadata, (1, 8, 3)) == {}
