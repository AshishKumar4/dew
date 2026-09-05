"""Sequence-parallel attention on the simulated 8-device mesh.

Under a mesh whose sequence axis is above one, the attention seam splits its
queries over that axis, gathers keys and values whole once a layer, and
reorders a causal call's queries into the striped layout that gives every
shard the same causal work, with the positions carried into the mask and
the output put back in sequence order. The proof is equality with whole
sequences: the same loss, the same gradients, the same attention output.
"""

import collections
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

import dew.nn.backbones.causal_transformer  # registers the model built below
from dew.nn.attention import (
    NormalAttention, causal_attention_mask, rotary_freqs, scaled_dot_product_attention, stripe,
    unstripe,
)
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, build_mesh
from dew.training.distributed import shard_batch

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256
WHOLE = MeshSpec(fsdp=8)
SPLIT = MeshSpec(fsdp=4, sequence=2)
# Observed 2.4e-7 on the loss and 4.8e-7 on the gradients between the two
# meshes over the dense and the packed batch on CPU at highest matmul
# precision; the collectives sum in another order, so fp32 tolerance is 1e-6.
TOLERANCE = 1e-6


def tiny():
    return models.build(
        "causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=2,
        num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)


def dense_batch():
    rng = np.random.default_rng(0)
    return {"text": rng.integers(1, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}


def packed_batch():
    """Two documents a row, the second starting at a different column on
    every row, and the tail of the last row padding (segment 0)."""
    batch = dense_batch()
    rng = np.random.default_rng(1)
    segments = np.ones((BATCH, SEQ_LEN + 1), np.int32)
    positions = np.zeros((BATCH, SEQ_LEN + 1), np.int32)
    for row in range(BATCH):
        boundary = int(rng.integers(3, SEQ_LEN - 2))
        segments[row, boundary:] = 2
        positions[row, :boundary] = np.arange(boundary)
        positions[row, boundary:] = np.arange(SEQ_LEN + 1 - boundary)
    segments[-1, -3:] = 0
    batch["text_segment_ids"] = segments
    batch["text_positions"] = positions
    return batch


# --------------------------------------------------------------------------
# The striped order
# --------------------------------------------------------------------------


def test_stripe_writes_maxtexts_order_and_unstripe_undoes_it():
    """Two shards turn [0..7] into [0, 1, 6, 7, 2, 3, 4, 5]
    (maxtext_utils.reorder_sequence's own example), and the inverse holds
    for every shard count the eight-device mesh can hold."""
    np.testing.assert_array_equal(
        stripe(jnp.arange(8), 2, axis=0), [0, 1, 6, 7, 2, 3, 4, 5])
    rows = jnp.arange(2 * 32 * 3).reshape(2, 32, 3)
    for shards in (1, 2, 4, 8):
        np.testing.assert_array_equal(unstripe(stripe(rows, shards), shards), rows)


@pytest.mark.parametrize("shards", [2, 4, 8])
def test_every_shard_holds_the_same_causal_work(shards):
    """Row p of a causal sequence attends p + 1 keys; the striped order gives
    each shard's rows the same total."""
    length = 64
    positions = np.asarray(stripe(jnp.arange(length), shards, axis=0)).reshape(shards, -1)
    work = (positions + 1).sum(axis=1)
    assert np.all(work == work[0]), work
    contiguous = (np.arange(length).reshape(shards, -1) + 1).sum(axis=1)
    assert contiguous.max() > contiguous.min()


def test_a_length_the_shards_cannot_pair_is_refused():
    with pytest.raises(ValueError, match="multiple of 4, got 15"):
        stripe(jnp.zeros((2, 15, 4, 8)), 2)


# --------------------------------------------------------------------------
# The attention seam
# --------------------------------------------------------------------------


def heads(key, kv_heads):
    q_key, k_key, v_key = jax.random.split(key, 3)
    query = jax.random.normal(q_key, (BATCH, SEQ_LEN, 4, 8), jnp.float32)
    key_ = jax.random.normal(k_key, (BATCH, SEQ_LEN, kv_heads, 8), jnp.float32)
    value = jax.random.normal(v_key, (BATCH, SEQ_LEN, kv_heads, 8), jnp.float32)
    return query, key_, value


def packed_mask(segments):
    inside = ((segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0))
    return jnp.logical_and(inside[:, None], causal_attention_mask(jnp.arange(SEQ_LEN), SEQ_LEN))


CALLS = {
    "causal": dict(causal=True),
    "window": dict(causal=True, sliding_window=5),
    "packed": dict(mask=packed_mask(jnp.asarray(packed_batch()["text_segment_ids"][:, :-1]))),
    # Query-broadcast rows must survive the reorder without being expanded.
    "broadcast_mask": dict(
        causal=True, mask=(jnp.arange(SEQ_LEN) < SEQ_LEN // 2)[None, None, None, :]),
    "broadcast_bias": dict(
        causal=True, bias=jnp.linspace(-4.0, 4.0, SEQ_LEN)[None, None, None, :]),
    "bias": dict(causal=True,
                 bias=jax.random.normal(jax.random.key(3), (1, 4, SEQ_LEN, SEQ_LEN))),
    "full": dict(),
}


@pytest.mark.parametrize("implementation", [None, "xla"])
@pytest.mark.parametrize("name", sorted(CALLS))
def test_the_seam_agrees_with_whole_sequences(name, implementation):
    """The same output, row for row, whether the sequence axis is one or two:
    the mask follows the positions through the reorder, and the output comes
    back in sequence order. Observed 4.8e-7 at most on CPU."""
    query, key, value = heads(jax.random.key(0), kv_heads=2)
    call = dict(CALLS[name], implementation=implementation)
    whole = scaled_dot_product_attention(query, key, value, **call)
    with jax.set_mesh(build_mesh(SPLIT)):
        split = jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))(
            query, key, value)
    np.testing.assert_allclose(np.asarray(split), np.asarray(whole), atol=TOLERANCE, rtol=0)


def test_rotary_positions_and_the_causal_mask_survive_the_reorder():
    """A full attention module: rotary angles from the row's position, then
    the causal mask. Both were applied in sequence order, and the reordered
    call has to match them exactly (observed 4.8e-7)."""
    module = NormalAttention(query_dim=32, heads=4, dim_head=8, causal=True)
    x = jax.random.normal(jax.random.key(1), (BATCH, SEQ_LEN, 32), jnp.float32)
    freqs = rotary_freqs(jnp.arange(SEQ_LEN), 8, 10000.0)
    variables = module.init(jax.random.key(2), x, freqs_cis=freqs)
    whole = module.apply(variables, x, freqs_cis=freqs)
    with jax.set_mesh(build_mesh(SPLIT)):
        split = jax.jit(lambda v, x: module.apply(v, x, freqs_cis=freqs))(variables, x)
    np.testing.assert_allclose(np.asarray(split), np.asarray(whole), atol=TOLERANCE, rtol=0)


def test_decoding_is_refused_under_a_sequence_axis():
    model = tiny()
    tokens = jnp.ones((1, SEQ_LEN), jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    with jax.set_mesh(build_mesh(SPLIT)):
        with pytest.raises(ValueError, match="sequence axis of 2"):
            jax.jit(lambda v, t: model.apply(v, t, decode=True, mutable=["cache"]))(
                variables, tokens)


# --------------------------------------------------------------------------
# The model, through the trainer
# --------------------------------------------------------------------------


def one_step(spec, batch):
    """The loss of one step and the parameters after it, on `spec`; with
    sgd(1.0) the parameters move by exactly the gradient."""
    trainer = Trainer(LMObjective(tiny(), SEQ_LEN), optax.sgd(1.0), key=jax.random.key(0),
                      mesh=spec, layout=Layout(min_shard=TINY_SHARD))
    state, _, _ = trainer.place()
    placed = shard_batch(trainer.device_mesh, batch)
    state, _, loss, _, _ = trainer.compile(state, placed)(state, None, placed)
    return float(loss), jax.tree.map(np.asarray, state.params["params"])


@pytest.mark.parametrize("make_batch", [dense_batch, packed_batch])
def test_loss_and_gradients_agree_with_whole_sequences(make_batch):
    """Loss and every gradient leaf equal between fsdp=8 and fsdp=4,sequence=2,
    on a dense batch and on a packed one with segment ids and positions."""
    batch = make_batch()
    whole_loss, whole_params = one_step(WHOLE, batch)
    split_loss, split_params = one_step(SPLIT, batch)

    assert abs(whole_loss - split_loss) < TOLERANCE, (whole_loss, split_loss)
    differences = jax.tree.map(
        lambda a, b: float(np.max(np.abs(a - b))), whole_params, split_params)
    assert max(jax.tree.leaves(differences)) < TOLERANCE, differences


def collectives(spec, batch):
    """Every collective of the compiled step, counted by kind and shape."""
    trainer = Trainer(LMObjective(tiny(), SEQ_LEN), optax.sgd(1.0), key=jax.random.key(0),
                      mesh=spec, layout=Layout(min_shard=TINY_SHARD))
    state, _, _ = trainer.place()
    placed = shard_batch(trainer.device_mesh, batch)
    trainer.compile(state, placed)
    with jax.set_mesh(trainer.device_mesh):
        text = jax.jit(trainer._step_body()).lower(state, None, placed).compile().as_text()
    assert text is not None
    counted = collections.Counter()
    for line in text.splitlines():
        found = re.search(
            r"= (\w+\[[^\]]*\])\{[^}]*\} (all-gather|all-reduce|reduce-scatter|"
            r"all-to-all|collective-permute)\(", line)
        if found:
            counted[(found.group(2), found.group(1))] += 1
    return counted


def test_keys_and_values_are_gathered_once_a_layer_and_queries_never():
    """MeshSpec(fsdp=2, sequence=2): every device holds two rows of the batch.
    The forward pass gathers each layer's keys and values once
    ([2, 16, 2, 8], the kv heads over the whole sequence) and no array of the
    query's shape ([2, 16, 4, 8]) is ever gathered; the queries move between
    shards as collective-permutes of the striped chunks."""
    counted = collectives(MeshSpec(fsdp=2, sequence=2), dense_batch())
    gathers = {shape: count for (kind, shape), count in counted.items() if kind == "all-gather"}

    assert gathers.get("f32[2,16,2,8]") == 2 * 2, gathers
    assert "f32[2,16,4,8]" not in gathers, gathers
    assert any(kind == "collective-permute" for kind, _ in counted), counted
