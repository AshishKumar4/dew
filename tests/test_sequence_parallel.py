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
    # Each query's own logit per head (projected sinks), split with its row, place and head.
    "query_sinks": {"causal": True, "sinks": jax.random.normal(jax.random.key(5), (BATCH, SEQ_LEN, 4))},
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


def _draw(seed, heads: int = 4, kv_heads: int = 2):
    """Queries of `heads` heads, keys and values of `kv_heads`, and an output cotangent."""
    q_key, k_key, v_key, c_key = jax.random.split(jax.random.key(seed), 4)
    query = jax.random.normal(q_key, (BATCH, SEQ_LEN, heads, 8), jnp.float32)
    key, value = (jax.random.normal(k, (BATCH, SEQ_LEN, kv_heads, 8), jnp.float32) for k in (k_key, v_key))
    return query, key, value, jax.random.normal(c_key, query.shape, jnp.float32)


_dropping_draw = _draw
_gathered_draw = functools.partial(_draw, heads=3, kv_heads=1)
"""Three query heads over one key head, which no two-way sequence's all-to-all divides: the gather."""


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


def _groups(offsets: tuple[int, ...]) -> jnp.ndarray:
    """`[BATCH, SEQ_LEN]` block numbers: each row two blocks of four keys from its offset, -1 elsewhere."""
    rows = [[(column - offset) // 4 if 0 <= column - offset < 8 else -1 for column in range(SEQ_LEN)]
            for offset in offsets]
    return jnp.asarray(rows, jnp.int32)


VISIBLE = {
    "blocks": {"causal": True, "block_groups": _groups((0, 1, 2, 3, 4, 5, 6, 7))},
    "images": {"causal": True, "sliding_window": 6, "image_groups": _groups((7, 6, 5, 4, 3, 2, 1, 0))},
    "documents and blocks": {
        "causal": True, "block_groups": _groups((2,) * BATCH),
        "segment_ids": jnp.asarray(packed_batch()["text_coordinates"].segments[:, :-1], jnp.int32)},
    "starts": {"key_value_seq_starts": jnp.asarray([0, 3, 5, 1, 0, 7, 2, 4], jnp.int32),
               "key_value_seq_lengths": jnp.asarray([16, 9, 12, 16, 11, 16, 8, 13], jnp.int32)},
    "slots": {"causal": True, "query_slots": jnp.broadcast_to(jnp.arange(SEQ_LEN, dtype=jnp.int32),
                                                              (BATCH, SEQ_LEN)),
              "block_groups": _groups((4,) * BATCH)},
}
"""Calls whose visibility is per-token ids (row 1495): block and image groups, documents beside blocks, key
starts, a cached call's query slots."""

GATHERED_VISIBLE = {
    **VISIBLE,
    # Real keys and places a window counts by: row r's places step by one past each of its first r keys.
    "valid keys and places": {
        "causal": True, "sliding_window": 5,
        "key_valid": jnp.asarray([[column % 5 != 3 for column in range(SEQ_LEN)]] * BATCH),
        "places": jnp.asarray([[column + min(column, row) for column in range(SEQ_LEN)]
                               for row in range(BATCH)], jnp.int32)},
    # Each query's runs a row apiece `[B, Q, R, 2]` and every row's `[Q, R, 2]`: its own key on and the first.
    "runs": {"causal": True, "key_runs": jnp.broadcast_to(jnp.asarray(
        [[[0, 2], [q, SEQ_LEN]] for q in range(SEQ_LEN)], jnp.int32), (BATCH, SEQ_LEN, 2, 2))},
    "shared runs": {"key_runs": jnp.asarray([[[0, 1], [q // 2, q + 1]] for q in range(SEQ_LEN)], jnp.int32)},
    # A cached call's live slots in no index order: query q's own key 15 - q.
    "reversed slots": {"causal": True, "block_groups": _groups((3,) * BATCH),
                       "query_slots": jnp.broadcast_to(SEQ_LEN - 1 - jnp.arange(SEQ_LEN, dtype=jnp.int32),
                                                       (BATCH, SEQ_LEN))},
    # A cached call's slots with padding, its queries reading the block holding their slot.
    "padded slots": {"causal": True, "block_groups": _groups((1,) * BATCH),
                     "query_slots": jnp.asarray([[column if column < 12 + row % 4 else -1
                                                  for column in range(SEQ_LEN)] for row in range(BATCH)],
                                                jnp.int32)},
}
"""`VISIBLE` and the ids it holds none of: key validity, places, runs a row apiece and every row's, live slots
in no index order and padded slots (row 1495's gather path)."""


EXCHANGED_VISIBLE = [("all-to-all", name) for name in sorted(VISIBLE)] + [
    ("gather", name) for name in sorted(GATHERED_VISIBLE)]
"""Each exchange's calls of per-token ids: the all-to-all's (`_dropping_draw`'s heads, which it divides) and
the gather's (`_gathered_draw`'s, which it does not)."""


def _exchanged(exchange: str) -> tuple[dict, object]:
    """`exchange`'s calls and the draw whose heads take it (`EXCHANGED_VISIBLE`)."""
    return (VISIBLE, _dropping_draw) if exchange == "all-to-all" else (GATHERED_VISIBLE, _gathered_draw)


@pytest.mark.mesh
@pytest.mark.parametrize("implementation", ["reference", "xla", "template"])
@pytest.mark.parametrize(("exchange", "name"), EXCHANGED_VISIBLE,
                         ids=[" ".join(pair) for pair in EXCHANGED_VISIBLE])
def test_per_token_visibility_passes_through_either_exchange(exchange, name, implementation):
    """The entry over a split sequence whose call reads per-token ids hands them to the kernel, so no global
    `[B, 1, S, S]` mask is built (row 1495; a kernel that reads masks builds its shard's own, the template
    none): the all-to-all whole beside the whole sequences it gives (`VISIBLE`), the gather its key-side ids
    whole, each striped query reading them at its slot (`GATHERED_VISIBLE`,
    `attention_parallel._striped_visibility`). Its output and its gradients, forward and backward, on par
    with the whole sequence's (check 1) from float64, the reference's where the kernel takes no float64
    (the template)."""
    calls, draw = _exchanged(exchange)
    call = dict(calls[name], implementation=implementation)
    entry = with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))
    whole = entry if implementation != "template" else with_gradients(
        lambda q, k, v: scaled_dot_product_attention(q, k, v, **dict(call, implementation="reference")))
    assert_exchanged_on_par(entry, whole, SPLIT, draw,
                            f"the entry through the {exchange}, {name}, {implementation}")
    with jax.set_mesh(SPLIT.build()):
        program = jax.make_jaxpr(entry)(*draw(0))
    masks = [var.aval.shape for var in _variables(program.jaxpr)
             if var.aval.dtype == jnp.bool_ and var.aval.shape[-2:] == (SEQ_LEN, SEQ_LEN)
             and var.aval.shape[0] == BATCH]
    assert not masks, masks


@pytest.mark.mesh
def test_a_cross_call_through_the_gather_takes_its_visibility_as_a_mask():
    """A cross call (eight queries over sixteen keys) of heads the all-to-all does not divide takes its key
    validity as the mask it means (`attention_parallel.CROSS_GATHERED`: its queries hold no slots among its
    keys), on par with the whole call's."""
    call = {"key_valid": jnp.asarray([[column % 3 != 1 for column in range(SEQ_LEN)]] * BATCH),
            "implementation": "reference"}

    def draw(seed):
        query, key, value, _ = _gathered_draw(seed)
        cotangent = jax.random.normal(jax.random.key(seed + 99), (BATCH, 8, 3, 8), jnp.float32)
        return query[:, :8], key, value, cotangent

    entry = with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))
    assert_exchanged_on_par(entry, entry, SPLIT, draw, "a cross call's key validity through the gather")


@pytest.mark.mesh
@pytest.mark.parametrize(("exchange", "name"), EXCHANGED_VISIBLE,
                         ids=[" ".join(pair) for pair in EXCHANGED_VISIBLE])
def test_the_template_builds_no_mask_of_a_split_calls_visibility(exchange, name):
    """The template over a split sequence reads a call's per-token ids as its spans, on each device beside
    the all-to-all's whole sequences or at each striped query's slot through the gather, so its program
    holds no `[rows, 1, queries, S]` mask at all, its shard's or the batch's (row 1495; a key length's
    `[rows, 1, 1, S]` is no query's)."""
    calls, draw = _exchanged(exchange)
    call = dict(calls[name], implementation="template")
    entry = with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))
    with jax.set_mesh(SPLIT.build()):
        program = jax.make_jaxpr(entry)(*draw(0))
    masks = [var.aval.shape for var in _variables(program.jaxpr)
             if var.aval.dtype == jnp.bool_ and len(var.aval.shape) == 4 and var.aval.shape[1] == 1
             and var.aval.shape[2] > 1 and var.aval.shape[-1] == SEQ_LEN]
    assert not masks, masks


SHARED = MeshSpec(fsdp=2, sequence=2)
"""Batch rows split two ways beside the sequence: a shared vector of the batch's length (`B == K == 4`)."""


def _square_draw(seed):
    """Four rows of four tokens, four query heads over two key heads, and an output cotangent."""
    q_key, k_key, v_key, c_key = jax.random.split(jax.random.key(seed), 4)
    query = jax.random.normal(q_key, (4, 4, 4, 8), jnp.float32)
    key, value = (jax.random.normal(k, (4, 4, 2, 8), jnp.float32) for k in (k_key, v_key))
    return query, key, value, jax.random.normal(c_key, query.shape, jnp.float32)


SHARED_IDS = {
    "key_valid": jnp.asarray([True, True, False, True]),
    "query_slots": jnp.arange(4, dtype=jnp.int32),
    # Every query's runs `[R, 2]` and every row's `[Q, R, 2]`, each of the batch's leading length.
    "key_runs": jnp.asarray([[0, 1], [1, 1], [2, 3], [3, 4]], jnp.int32),
    "key_runs by query": jnp.asarray([[[0, 1], [q, q + 1]] for q in range(4)], jnp.int32),
}
"""Ids every row shares whose leading length is the batch's (four): `[K]` validity, `[Q]` slots, `[R, 2]` and
`[Q, R, 2]` runs (row 12701)."""


@pytest.mark.mesh
@pytest.mark.parametrize("name", sorted(SHARED_IDS))
def test_a_shared_id_of_the_batchs_length_stays_whole(name):
    """A shared `[K]` validity, `[Q]` slots or `[R, 2]` and `[Q, R, 2]` runs of the batch's length (four rows
    of four tokens, the rows split two ways) travels whole, the keys' or the queries' as it is, not split as
    rows (reviews of #4023 and #4026, row 12701): the entry on par with whole sequences, forward and
    backward."""
    call = {"causal": True, name.split(" by ")[0]: SHARED_IDS[name], "implementation": "reference"}
    entry = with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))
    assert_exchanged_on_par(entry, entry, SHARED, _square_draw, f"a shared {name} of the batch's length")


@pytest.mark.mesh
def test_runs_of_another_batch_are_refused_before_the_exchange():
    """Runs a row apiece `[2, Q, R, 2]` beside a batch of four, its rows split two ways, are refused before
    the all-to-all, where replicated whole each shard read its own two rows' runs as its rows' and rows two
    and three took rows zero and one's visibility (review of #4035)."""
    runs = jnp.broadcast_to(jnp.asarray([[0, 2]], jnp.int32), (2, 4, 1, 2))
    query, key, value, _ = _square_draw(0)
    with jax.set_mesh(SHARED.build()), pytest.raises(ValueError):
        jax.jit(lambda q, k, v: scaled_dot_product_attention(
            q, k, v, causal=True, key_runs=runs, implementation="reference")).trace(query, key, value)


def _cross_draw(seed):
    """Four rows of four tokens, three query heads over one key head (the gather on `SHARED`), and an
    output cotangent."""
    query, key, value, cotangent = _draw(seed, heads=3, kv_heads=1)
    return query[:4, :4], key[:4, :4], value[:4, :4], cotangent[:4, :4]


def _documents(rows: int):
    """Cross documents of equal lengths, `rows` rows: queries `[1, 2, 2, 2]` over keys `[1, 1, 2, 2]`."""
    from dew.nn.documents import CrossDocuments

    return CrossDocuments(jnp.asarray([[1, 2, 2, 2]] * rows, jnp.int32),
                          jnp.asarray([[1, 1, 2, 2]] * rows, jnp.int32))


@pytest.mark.mesh
@pytest.mark.parametrize("implementation", ["reference", "cudnn"])
def test_equal_length_cross_documents_through_the_gather_keep_their_alignment(implementation):
    """Causal cross documents of equal lengths through the gather align each query to its document's last
    keys (query 0 sees keys 0 and 1), taken as their mask (`attention_parallel.ALIGNED_GATHERED`), where the
    striped indices as slots overwrote the alignment and query 0 saw key 0 alone (review of #4052); a call
    naming cuDNN runs that mask on the masked kernel (`kernel_for_materialized_mask`; review of #4053): on
    par with the whole call's xla, forward and backward."""
    def attend(name):
        call = {"causal": True, "segment_ids": _documents(4), "implementation": name}
        return with_gradients(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call))

    assert_exchanged_on_par(attend(implementation), attend("reference" if implementation == "reference"
                                                          else "xla"), SHARED, _cross_draw,
                            f"equal-length cross documents through the gather, {implementation}")


@pytest.mark.mesh
def test_documents_of_another_batch_are_refused_before_their_mask():
    """Cross documents of two rows beside a batch of four, its rows split two ways, are refused before the
    gather's masked route folds them into a mask, where that `[2, 1, 4, 4]` mask replicated and rows two and
    three read rows zero and one's documents (review of #4053)."""
    query, key, value, _ = _cross_draw(0)
    call = {"causal": True, "segment_ids": _documents(2), "implementation": "reference"}
    with jax.set_mesh(SHARED.build()), pytest.raises(ValueError):
        jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call)).trace(query, key, value)


@pytest.mark.mesh
def test_key_lengths_of_another_batch_are_refused_before_the_exchange():
    """Key lengths of two rows beside a batch of four, its rows split two ways, are refused before the
    gather, where replicated whole each shard read rows zero and one's lengths as its own (review of
    #4052)."""
    query, key, value, _ = _cross_draw(0)
    call = {"key_value_seq_lengths": jnp.asarray([1, 2], jnp.int32),
            "key_valid": jnp.ones((4, 4), bool), "implementation": "reference"}
    with jax.set_mesh(SHARED.build()), pytest.raises(ValueError):
        jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, **call)).trace(query, key, value)


@pytest.mark.mesh
def test_a_call_naming_cudnn_takes_its_visibility_on_the_masked_kernel():
    """A call naming cuDNN whose visibility is per-token ids, over a split sequence, runs on the kernel a
    built mask takes (`kernel_for_materialized_mask`: cuDNN takes no mask), as it did when the ids were
    that mask (review of #4023): on par with xla's whole sequences."""
    split = with_gradients(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, **dict(VISIBLE["blocks"], implementation="cudnn")))
    whole = with_gradients(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, **dict(VISIBLE["blocks"], implementation="xla")))
    assert_exchanged_on_par(split, whole, SPLIT, _dropping_draw, "blocks naming cudnn over a split sequence")


def _variables(jaxpr) -> list:
    """Every variable `jaxpr` and the programs it holds define."""
    found = []
    for equation in jaxpr.eqns:
        found.extend(equation.outvars)
        for value in equation.params.values():
            for inner in (value if isinstance(value, (tuple, list)) else (value,)):
                held = getattr(inner, "jaxpr", inner)
                if hasattr(held, "eqns"):
                    found.extend(_variables(held))
    return found


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
