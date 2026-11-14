"""Sequence-parallel attention and Mamba-2 on the simulated 8-device mesh,
and cuDNN's attention inside both exchanges on two GPUs.

Under a mesh whose sequence axis is above one, the attention seam runs one
of two exchanges. The all-to-all one (Ulysses) trades every shard's rows of
the sequence for a slice of the heads, attends whole sequences, and trades
back. The all-gather one splits the queries over that axis, gathers keys and
values whole once a layer, and reorders a causal call's queries into the
striped layout that gives every shard the same causal work, with the
positions carried into the mask and the output put back in sequence order.
The proof is equality with whole sequences: the same loss, the same
gradients, the same attention output, for both.

Mamba-2 splits its conv and its scan over the same axis: each shard reads
the previous shard's conv tail and starts its scan from the state the
earlier shards leave, and the proof is the same, the layer's output and
gradients and a hybrid model's training step against whole sequences.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec
from model_support import two_documents
from reference_error import FEWEST_SEEDS, assert_as_exact, with_gradients
from sharded import MOVES, assert_runs_per_shard
from token_support import token_rows

from dew.nn.attention import attention_kernel, scaled_dot_product_attention
from dew.nn.attention_blocks import NormalAttention
from dew.nn.attention_masks import causal_attention_mask
from dew.nn.attention_parallel import exchanged_heads_attention, gathered_keys_attention
from dew.nn.inputs import MixerCall
from dew.nn.kernels.templates import attention as template
from dew.nn.rope import rotary_freqs
from dew.training import MeshSpec

EXCHANGES = {"all_to_all": exchanged_heads_attention, "all_gather": gathered_keys_attention}

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
SPLIT = MeshSpec(fsdp=4, sequence=2)


def dense_batch():
    return token_rows(VOCAB, BATCH, SEQ_LEN + 1, low=1)


def packed_batch():
    """Two documents a row, the second starting at a different column on
    every row, and the tail of the last row padding (segment 0)."""
    return {**dense_batch(), "text_coordinates": two_documents(np.random.default_rng(1), BATCH, SEQ_LEN + 1)}


# --------------------------------------------------------------------------
# The striped order
# --------------------------------------------------------------------------


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
    "causal": {"causal": True},
    "window": {"causal": True, "sliding_window": 5},
    "packed": {"mask": packed_mask(jnp.asarray(packed_batch()["text_coordinates"].segments[:, :-1]))},
    # Query-broadcast rows must survive the reorder without being expanded.
    "broadcast_mask": {
        "causal": True, "mask": (jnp.arange(SEQ_LEN) < SEQ_LEN // 2)[None, None, None, :]},
    "broadcast_bias": {
        "causal": True, "bias": jnp.linspace(-4.0, 4.0, SEQ_LEN)[None, None, None, :]},
    "bias": {"causal": True,
                 "bias": jax.random.normal(jax.random.key(3), (1, 4, SEQ_LEN, SEQ_LEN))},
    "full": {},
    # One learned logit per query head, split with the heads by the exchange.
    "sinks": {"causal": True, "sinks": jax.random.normal(jax.random.key(4), (4,))},
    # Each row's keys end at its own length; the rows split with the batch.
    "key_lengths": {"key_value_seq_lengths": jnp.asarray([16, 9, 3, 1, 12, 16, 5, 7], jnp.int32)},
    "causal_key_lengths": {"causal": True, "key_value_seq_lengths": jnp.asarray(
        [16, 9, 3, 1, 12, 16, 5, 7], jnp.int32)},
}


def through(exchange, query, key, value, *, implementation="reference", **call):
    """One attention call through the named exchange, over the mesh's
    sequence axis, whichever the per-call choice would have taken; a dropping
    call (`dropout_rate`, `dropout_rng`) draws from its rng's words as
    `scaled_dot_product_attention` does."""
    kernel = functools.partial(attention_kernel, implementation=implementation)
    rng = call.get("dropout_rng")
    return EXCHANGES[exchange](
        kernel, query, key, value, jax.sharding.get_abstract_mesh().shape["sequence"],
        causal=call.get("causal", False), sliding_window=call.get("sliding_window"),
        mask=call.get("mask"), bias=call.get("bias"), sinks=call.get("sinks"),
        key_value_seq_lengths=call.get("key_value_seq_lengths"),
        dropout_rate=call.get("dropout_rate", 0.0),
        dropout_seed=None if rng is None else template.dropout_seed(rng))


@pytest.mark.mesh
@pytest.mark.parametrize("exchange", sorted(EXCHANGES))
@pytest.mark.parametrize("implementation", ["reference", "xla"])
@pytest.mark.parametrize("name", sorted(CALLS))
def test_the_seam_agrees_with_whole_sequences(name, implementation, exchange):
    """The same output, row for row, whether the sequence axis is one or two:
    the mask follows the positions through either exchange, and the output
    comes back in sequence order, on par with whole sequences' (check 1)."""
    call = dict(CALLS[name], implementation=implementation)
    assert_exchanged_on_par(lambda q, k, v: through(exchange, q, k, v, **call),
                            lambda q, k, v: scaled_dot_product_attention(q, k, v, **call), SPLIT,
                            lambda seed: heads(jax.random.key(seed), kv_heads=2),
                            f"{exchange} exchange, {name}, {implementation}")


def assert_exchanged_on_par(split, whole, mesh_spec, draw, label: str) -> None:
    """`split`, a call through an exchange over `mesh_spec`'s sequence axis, held as the gate holds a
    split arm (`sharded.assert_runs_per_shard`): compiled once in the scope the gate runs its arms, and on
    par with `whole`, the same call on one device over whole sequences, over `draw`'s seeds (check 1, its
    truth `whole` on the inputs widened). An exchange moves rows by whichever collectives XLA lowers it to
    (an all-to-all, its collective-permutes on a CPU, the gather of whole operands), so every one of
    `sharded.MOVES` is the call's own. Every site of this file is sized by `seeds_for` on armada's CPUs at
    the fewest (2026-11-14). The operands are placed whole, the exchange's own shard_map splitting them."""
    specs = [PartitionSpec()] * len(draw(0))
    assert_runs_per_shard(split, draw, specs, mesh_spec.build(), label, seeds=FEWEST_SEEDS, truth_fn=whole,
                          moves=MOVES, arm=whole)


# Heads split over tensor then sequence (2x2), and four sequence shards over
# two key heads, which the exchange repeats to four and no further.
HEAD_SPLITS = [MeshSpec(fsdp=2, tensor=2, sequence=2), MeshSpec(fsdp=2, sequence=4)]


HEADS_BEFORE_ROWS = (("activation_heads", ("tensor", "expert")), ("activation_kv", ("tensor", "expert")),
                     ("activation_batch", "expert"))
"""Rules under which the query heads take the expert axis before the rows can: the query keeps every row on
each shard, where a `[B]` or `[B, H, Q, K]` argument alone would split its rows over the expert axis."""


@pytest.mark.mesh
@pytest.mark.parametrize("exchange", sorted(EXCHANGES))
@pytest.mark.parametrize("name", ["causal_key_lengths", "bias", "sinks"])
def test_row_arguments_split_as_the_query_rows_do_where_the_heads_contend_for_the_axis(name, exchange):
    """Eight query heads over a tensor and an expert axis of two each, which the heads take before the rows
    (`HEADS_BEFORE_ROWS`), and a sequence axis of two: a call's key lengths, bias and sinks split as the
    query's own rows and heads do (`query_split`), not as their names alone would, so each shard reads its
    own rows'. The same output as whole sequences."""
    from flax import linen as nn

    from dew.mesh import DEFAULT_RULES

    call = dict(CALLS[name])
    if name == "bias":
        call["bias"] = jax.random.normal(jax.random.key(3), (BATCH, 8, SEQ_LEN, SEQ_LEN))
    if name == "sinks":
        call["sinks"] = jax.random.normal(jax.random.key(4), (8,))

    def draw(seed):
        q_key, k_key, v_key = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(q_key, (BATCH, SEQ_LEN, 8, 8), jnp.float32)
        key, value = (jax.random.normal(k, (BATCH, SEQ_LEN, 2, 8), jnp.float32) for k in (k_key, v_key))
        return query, key, value

    def whole(query, key, value):
        return scaled_dot_product_attention(query, key, value, **call)

    # The exchange compiled under the mesh and the contending rules; the whole sequence outside them.
    with nn.logical_axis_rules((*HEADS_BEFORE_ROWS, *DEFAULT_RULES)):
        assert_exchanged_on_par(lambda q, k, v: through(exchange, q, k, v, **call), whole,
                                MeshSpec(tensor=2, expert=2, sequence=2), draw,
                                f"{exchange} exchange, {name}, under contending heads")


@pytest.mark.mesh
@pytest.mark.parametrize("spec", HEAD_SPLITS, ids=["tensor2_sequence2", "sequence4"])
@pytest.mark.parametrize("name", ["causal", "packed", "bias", "sinks", "causal_key_lengths"])
def test_the_exchange_agrees_forward_and_backward_where_heads_split_further(spec, name):
    """The all-to-all's gradients are its transposes: the query, key and
    value cotangents of a whole-sequence call come back in the split one,
    with the heads cut over tensor and sequence at once and grouped keys
    repeated to the split: output and gradients on par with whole sequences' (check 1)."""
    call = CALLS[name]

    def draw(seed):
        query, key, value = heads(jax.random.key(seed), kv_heads=2)
        return query, key, value, jax.random.normal(jax.random.fold_in(jax.random.key(seed), 1), query.shape)

    assert_exchanged_on_par(with_gradients(lambda q, k, v: through("all_to_all", q, k, v, **call)),
                            with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call)),
                            spec, draw, f"all_to_all exchange, {name}, heads over tensor {spec.tensor} and "
                                        f"sequence {spec.sequence}")


def _dropping_draw(seed):
    """Queries, keys and values of `heads`' shapes, and an output cotangent."""
    q_key, k_key, v_key, c_key = jax.random.split(jax.random.key(seed), 4)
    query = jax.random.normal(q_key, (BATCH, SEQ_LEN, 4, 8), jnp.float32)
    key, value = (jax.random.normal(k, (BATCH, SEQ_LEN, 2, 8), jnp.float32) for k in (k_key, v_key))
    return query, key, value, jax.random.normal(c_key, query.shape, jnp.float32)


@pytest.mark.mesh
@pytest.mark.parametrize("exchange", sorted(EXCHANGES))
@pytest.mark.parametrize("name", ["causal", "full", "packed", "sinks"])
def test_a_dropping_call_drops_what_one_device_drops(name, exchange):
    """Probability dropout over a sequence the mesh splits draws each weight at
    its global row, head, query and key (`template.DropPlaces`), so either
    exchange drops the weights one device drops, the gather's striped queries
    too: its output and its query, key and value gradients are on par with the
    whole sequence's (row 1709: a split sequence refused dropout)."""
    call = dict(CALLS[name], dropout_rate=0.3, dropout_rng=jax.random.key(7))
    assert_exchanged_on_par(
        with_gradients(lambda q, k, v: through(exchange, q, k, v, **call)),
        with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, deterministic=False, **call)),
        SPLIT, _dropping_draw, f"{exchange} exchange, {name}, dropping")


@pytest.mark.mesh
@pytest.mark.parametrize("exchange", sorted(EXCHANGES))
def test_a_dropping_template_call_of_no_keys_runs_inside_either_exchange(exchange):
    """A dropping call of no keys on the template's route inside either exchange, whose padded tiles draw
    at places no table holds (`template.Draw`): every output is zero, as the whole call's (the reviewer's
    witness: an empty table failed the draw's lookup)."""
    query = jax.random.normal(jax.random.key(0), (BATCH, SEQ_LEN, 4, 8), jnp.float32)
    empty = jnp.zeros((BATCH, 0, 2, 8), jnp.float32)
    call = {"dropout_rate": 0.3, "dropout_rng": jax.random.key(7), "implementation": "template"}
    whole = scaled_dot_product_attention(query, empty, empty, deterministic=False, **call)
    with jax.set_mesh(SPLIT.build()):
        split = jax.jit(lambda q, k, v: through(exchange, q, k, v, **call))(query, empty, empty)
    np.testing.assert_array_equal(np.asarray(split), np.asarray(whole))


@pytest.mark.mesh
def test_the_entry_drops_over_a_split_sequence():
    """`scaled_dot_product_attention` under a split sequence takes a dropping
    call (it refused one) and drops what the whole sequence does."""
    call = {"causal": True, "dropout_rate": 0.3, "dropout_rng": jax.random.key(7), "deterministic": False}
    entry = with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))
    assert_exchanged_on_par(entry, entry, SPLIT, _dropping_draw, "the entry, dropping over a split sequence")


@pytest.mark.mesh
def test_splash_runs_inside_the_exchange():
    """The all-to-all hands the kernel whole sequences, so splash takes a
    causal call under a split sequence with its causal descriptor. The
    all-gather exchange can only hand it a striped traced mask, which splash
    has no form for. Run under pallas's interpreter off a TPU."""
    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        return tuple(jax.random.normal(k, (2, 256, 4, 8), jnp.float32) for k in keys)

    assert_exchanged_on_par(lambda q, k, v: through("all_to_all", q, k, v, causal=True, implementation="tpu"),
                            lambda q, k, v: scaled_dot_product_attention(q, k, v, causal=True),
                            MeshSpec(fsdp=2, sequence=2), draw, "splash inside the all-to-all exchange")


def cudnn_runs_inside_either_exchange(exchange, causal):
    """cuDNN's partitioning rule refuses queries split unlike their keys
    (`_check_qkv_bias_mask_spec` in jax/_src/cudnn/fused_attention_stablehlo.py),
    which is what the gather hands a kernel it leaves to GSPMD: every call
    that took the gather failed to compile on a GPU. Both exchanges run the
    kernel on local arrays: the split call's output and every gradient on par
    with the same cuDNN call on one GPU, from float64 (check 1, the truth the
    reference path's float64 run)."""
    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 4)
        query = jax.random.normal(keys[0], (2, 256, 4, 64), jnp.bfloat16)
        key, value = (jax.random.normal(k, (2, 256, 2, 64), jnp.bfloat16) for k in keys[1:3])
        return query, key, value, jax.random.normal(keys[3], query.shape, jnp.bfloat16)

    def whole(implementation):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, causal=causal,
                                                            implementation=implementation)

    def split(q, k, v):
        with jax.set_mesh(MeshSpec(sequence=2).build(jax.devices()[:2])):
            return through(exchange, q, k, v, causal=causal, implementation="cudnn")

    assert_as_exact(jax.jit(with_gradients(split)), jax.jit(with_gradients(whole("cudnn"))), draw,
                    f"cudnn inside the {exchange}", seeds=FEWEST_SEEDS,
                    truth_fn=with_gradients(whole("reference")))


def test_equal_multimodal_coordinates_rotate_the_shared_keys_like_plain_positions():
    """Equal temporal and spatial coordinates reduce M-RoPE to plain RoPE,
    including the keys a later layer shares. Both must read the same table.
    """
    from dew.batch import coordinates as token_coordinates
    from dew.nn.inputs import AttentionMetadata
    from dew.nn.mixer_base import MixerContext
    from dew.nn.mixers.attention import CausalSelfAttention

    plain = CausalSelfAttention(MixerContext(emb_features=16, num_heads=2, num_kv_heads=1, head_dim=128,
                                             max_seq_len=4096, rope_theta=5e5, partial_rotary_type="default",
                                             kv_store_key="shared"))
    multimodal = plain.clone(mrope_section=(16, 24, 24))
    hidden = jax.random.normal(jax.random.key(1), (1, 3, 16))
    positions = jnp.asarray([[0, 1, 4093]], jnp.int32)
    places = token_coordinates(positions=positions)
    variables = plain.init(jax.random.key(2), hidden, MixerCall(AttentionMetadata(coordinates=places)))
    coordinates = jnp.repeat(positions[..., None], 3, axis=-1)

    def shared_keys(layer, metadata):
        store = {}
        layer.apply(variables, hidden, MixerCall(metadata, kv_store=store))
        return store["shared"][0]

    np.testing.assert_array_equal(
        shared_keys(multimodal, AttentionMetadata(coordinates=places, rotary_positions=coordinates)),
        shared_keys(plain, AttentionMetadata(coordinates=places)))


@pytest.mark.mesh
def test_rotary_positions_survive_the_exchange():
    """A full attention module: rotary angles from the row's position,
    applied in sequence order, and the exchanged call's output on par with
    the whole sequence's (check 1)."""
    module = NormalAttention(query_dim=32, heads=4, dim_head=8)

    def draw(seed):
        x = jax.random.normal(jax.random.key(seed), (BATCH, SEQ_LEN, 32), jnp.float32)
        freqs = rotary_freqs(jnp.arange(SEQ_LEN), 8, 10000.0, dtype=np.float32)
        return module.init(jax.random.fold_in(jax.random.key(seed), 1), x, freqs_cis=freqs), x

    def applied(module, dtype):
        # The angles in the arm's own width, made where it runs (a float64 truth's under x64).
        return lambda variables, x: module.apply(
            variables, x, freqs_cis=rotary_freqs(jnp.arange(SEQ_LEN), 8, 10000.0, dtype=dtype))

    # The module's own exchange, its rows moved by whichever collectives XLA lowers it to.
    assert_runs_per_shard(applied(module, np.float32), draw, [PartitionSpec()] * 2, SPLIT.build(),
                          "rotary positions through the exchange", seeds=FEWEST_SEEDS,
                          truth_fn=applied(module.clone(dtype=jnp.float64), np.float64), moves=MOVES)


# --------------------------------------------------------------------------
# The model, through the trainer
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Mamba-2: the conv and the scan split over the sequence axis
# --------------------------------------------------------------------------
